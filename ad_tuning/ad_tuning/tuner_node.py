from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.util
import math
import os
from pathlib import Path
import re
import threading
import time

from ament_index_python.packages import (
    get_package_prefix,
    get_package_share_directory,
)
import optuna
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sqlalchemy.exc import SQLAlchemyError

from ad_tuning.objective import ObjectiveConfig, evaluate_trial
from ad_tuning.ros_morai_runner import (
    RosMoraiGlobalPathRunner,
    RouteSnapshot,
)
from ad_tuning.scenario_identity import (
    ScenarioIdentity,
    default_morai_save_root,
    validate_scenario_identity,
)
from ad_tuning.search import (
    DWA_STUDY_VERSION,
    PROFILE_STANLEY_STUDY_VERSION,
    build_experiment_fingerprint,
    create_dwa_study,
    create_profile_stanley_study,
    normalize_storage_url,
    redact_storage_url,
    resolve_worker_id,
    runtime_artifact_manifest,
    runtime_source_digest,
    suggest_dwa_parameters,
    suggest_profile_stanley_parameters,
)
from ad_tuning.storage import TrialStorage


FIXED_PROFILE_STANLEY_PARAMETERS = {
    "profile_stanley.lookahead_min_m": 1.5,
    "profile_stanley.lookahead_max_m": 5.0,
    "profile_stanley.target_speed_mps": 58.5 / 3.6,
    "profile_stanley.maximum_speed_mps": 60.0 / 3.6,
    "profile_stanley.lateral_acceleration_mps2": 6.0,
    "profile_stanley.acceleration_mps2": 5.0,
    "profile_stanley.deceleration_mps2": 2.0,
    "profile_stanley.speed_pid.ki": 0.0,
    "profile_stanley.speed_pid.derivative_limit": 10.0,
    "profile_stanley.speed_pid.derivative_filter_time_constant_s": 0.1,
    "profile_stanley.brake_pid.ki": 0.0,
}
TUNING_STACK_CONDITIONS = {
    "path_tracking.backend": "profile_stanley",
    "perception.enabled": False,
    "tuning.lease_required": True,
}

FIXED_DWA_PARAMETERS = {
    "dwa.minimum_speed_mps": 0.0,
    "dwa.maximum_speed_mps": 16.25,
    "dwa.speed_step_mps": 1.0,
    "dwa.minimum_steering_rad": -0.52,
    "dwa.maximum_steering_rad": 0.52,
    "dwa.steering_step_rad": 0.04,
    "dwa.simulation_dt": 0.2,
    "dwa.horizon_sec": 1.5,
    "dwa.dynamic_window_time_sec": 0.5,
    "dwa.maximum_acceleration_mps2": 5.0,
    "dwa.maximum_deceleration_mps2": 1.8,
    "dwa.emergency_deceleration_mps2": 6.0,
    "dwa.initial_inflation_escape_sec": 0.6,
    "dwa.maximum_steering_rate_radps": 2.0943951023931953,
    "dwa.maximum_lateral_acceleration_mps2": 6.0,
    "dwa.clearance_saturation_m": 8.0,
    "dwa.maximum_path_distance_m": 4.5,
    "dwa.prediction.covariance_sigma": 2.0,
    "dwa.prediction.minimum_margin_m": 0.20,
    "dwa.progress_weight": 1.0,
    "dwa.speed_pid.ki": 0.0,
    "dwa.speed_pid.integral_limit": 10.0,
    "dwa.speed_pid.derivative_limit": 10.0,
    "dwa.speed_pid.derivative_filter_time_constant_s": 0.1,
    "dwa.speed_pid.brake_deadband_mps": 0.1,
    "dwa.brake_pid.ki": 0.0,
}
DWA_STACK_CONDITIONS = {
    "local_motion.backend": "dwa",
    "local_motion.prediction.mode": "required",
    "path_tracking.backend": "profile_stanley",
    "perception.enabled": True,
    "road_gate.enabled": True,
    "perception.route_aligned_activation": True,
    "tuning.lease_required": True,
}


@dataclass(frozen=True)
class TuningSpec:
    algorithm: str
    node_name: str
    study_version: str
    fixed_parameters: dict[str, float]
    stack_conditions: dict[str, object]
    tuning_config_filename: str
    bridge_config_filename: str
    sqlite_filename: str
    create_study: Callable[..., optuna.Study]
    suggest_parameters: Callable[[optuna.Trial], dict[str, float]]


PROFILE_STANLEY_SPEC = TuningSpec(
    algorithm="profile_stanley",
    node_name="ad_global_path_tuner",
    study_version=PROFILE_STANLEY_STUDY_VERSION,
    fixed_parameters=FIXED_PROFILE_STANLEY_PARAMETERS,
    stack_conditions=TUNING_STACK_CONDITIONS,
    tuning_config_filename="tuning.yaml",
    bridge_config_filename="morai_tuning_bridge.yaml",
    sqlite_filename="profile_stanley_optuna.sqlite3",
    create_study=create_profile_stanley_study,
    suggest_parameters=suggest_profile_stanley_parameters,
)
DWA_SPEC = TuningSpec(
    algorithm="dwa",
    node_name="ad_dwa_tuner",
    study_version=DWA_STUDY_VERSION,
    fixed_parameters=FIXED_DWA_PARAMETERS,
    stack_conditions=DWA_STACK_CONDITIONS,
    tuning_config_filename="dwa_tuning.yaml",
    bridge_config_filename="morai_dwa_tuning_bridge.yaml",
    sqlite_filename="dwa_optuna.sqlite3",
    create_study=create_dwa_study,
    suggest_parameters=suggest_dwa_parameters,
)

# Backward-compatible public name used by existing static contracts.
SEARCH_SPACE_VERSION = PROFILE_STANLEY_STUDY_VERSION
DATABASE_EXCEPTIONS = (
    optuna.exceptions.StorageInternalError,
    SQLAlchemyError,
)


def resolve_tuning_output_dir(
    explicit: Path | str,
    *,
    algorithm: str,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve tuning output without creating a second implicit data root."""
    explicit_text = str(explicit).strip()
    if explicit_text:
        return Path(explicit_text).expanduser()

    environment = os.environ if environ is None else environ
    data_root = environment.get("AD_DATA_DIR", "").strip()
    if not data_root:
        raise ValueError("set output_dir or AD_DATA_DIR")
    return Path(data_root).expanduser() / "tuning" / algorithm


def _python_module_source(module_name: str) -> Path:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise FileNotFoundError(
            f"required runtime Python module is missing: {module_name}"
        )
    return Path(spec.origin)


class InfrastructureTrialFailure(RuntimeError):
    """Marks a simulator/ROS failure as a failed, retryable Optuna trial."""


class GlobalPathTunerNode(Node):
    def __init__(self, spec: TuningSpec = PROFILE_STANLEY_SPEC) -> None:
        super().__init__(spec.node_name)
        self.spec = spec
        self.maximum_trials = int(
            self.declare_parameter("maximum_trials", 30).value
        )
        if self.maximum_trials < 0:
            raise ValueError("maximum_trials must be nonnegative")
        self.maximum_total_trials = int(
            self.declare_parameter("maximum_total_trials", 45).value
        )
        self.minimum_feasible_trials = int(
            self.declare_parameter("minimum_feasible_trials", 6).value
        )
        self.maximum_consecutive_infrastructure_failures = int(
            self.declare_parameter(
                "maximum_consecutive_infrastructure_failures", 3
            ).value
        )
        self.maximum_worker_wall_time_s = float(
            self.declare_parameter(
                "maximum_worker_wall_time_sec", 7200.0
            ).value
        )
        if (
            self.maximum_total_trials < 0
            or self.minimum_feasible_trials < 0
            or self.maximum_consecutive_infrastructure_failures <= 0
            or not math.isfinite(self.maximum_worker_wall_time_s)
            or self.maximum_worker_wall_time_s < 0.0
        ):
            raise ValueError("invalid distributed tuning limits")
        self.seed = int(self.declare_parameter("seed", 20260726).value)
        inheritance_default = 40 if self.spec.algorithm == "dwa" else 36
        self.inherit_minimum_complete_trials = int(
            self.declare_parameter(
                "warm_start.inherit_minimum_complete_trials",
                inheritance_default,
            ).value
        )
        self.inherit_top_k = int(
            self.declare_parameter("warm_start.inherit_top_k", 5).value
        )
        if (
            self.inherit_minimum_complete_trials < 0
            or self.inherit_top_k < 0
        ):
            raise ValueError("warm-start inheritance limits must be nonnegative")
        self.worker_id = resolve_worker_id(
            str(self.declare_parameter("worker_id", "").value)
        )

        output_parameter = str(
            self.declare_parameter("output_dir", "").value
        )
        output_dir = resolve_tuning_output_dir(
            output_parameter,
            algorithm=self.spec.algorithm,
        )
        self.output_dir = output_dir
        self.storage = TrialStorage(
            output_dir, self.spec.algorithm, self.worker_id
        )

        storage_url_env = str(
            self.declare_parameter(
                "storage.url_env", "OPTUNA_STORAGE_URL"
            ).value
        ).strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", storage_url_env):
            raise ValueError("storage.url_env must be an environment name")
        self.storage_url = normalize_storage_url(
            output_dir,
            os.environ.get(storage_url_env, ""),
            sqlite_filename=self.spec.sqlite_filename,
        )
        self.heartbeat_interval_s = int(
            self.declare_parameter(
                "storage.heartbeat_interval_sec", 15
            ).value
        )
        self.heartbeat_grace_period_s = int(
            self.declare_parameter(
                "storage.heartbeat_grace_period_sec", 90
            ).value
        )
        self.heartbeat_retry_count = int(
            self.declare_parameter(
                "storage.heartbeat_retry_count", 1
            ).value
        )
        self.database_connect_timeout_s = int(
            self.declare_parameter(
                "storage.connect_timeout_sec", 3
            ).value
        )
        self.database_retry_attempts = int(
            self.declare_parameter(
                "storage.connection_retry_attempts", 12
            ).value
        )
        self.database_retry_interval_s = float(
            self.declare_parameter(
                "storage.connection_retry_interval_sec", 5.0
            ).value
        )
        self.database_health_check_interval_s = float(
            self.declare_parameter(
                "storage.health_check_interval_sec", 2.0
            ).value
        )
        self.database_outage_abort_s = float(
            self.declare_parameter(
                "storage.db_outage_abort_sec", 30.0
            ).value
        )
        if self.heartbeat_interval_s <= 0:
            raise ValueError(
                "storage.heartbeat_interval_sec must be positive"
            )
        if self.heartbeat_grace_period_s <= self.heartbeat_interval_s:
            raise ValueError(
                "storage.heartbeat_grace_period_sec must exceed heartbeat"
            )
        if self.heartbeat_retry_count < 0:
            raise ValueError(
                "storage.heartbeat_retry_count must be nonnegative"
            )
        if (
            self.database_retry_attempts < 0
            or self.database_connect_timeout_s <= 0
            or not math.isfinite(self.database_retry_interval_s)
            or self.database_retry_interval_s <= 0.0
            or not math.isfinite(self.database_health_check_interval_s)
            or self.database_health_check_interval_s <= 0.0
            or not math.isfinite(self.database_outage_abort_s)
            or self.database_outage_abort_s <= 0.0
        ):
            raise ValueError("invalid storage connection retry settings")

        maximum_cte_default = 4.5 if self.spec.algorithm == "dwa" else 0.7
        self.objective = ObjectiveConfig(
            maximum_cte_m=float(
                self.declare_parameter(
                    "objective.maximum_cte_m", maximum_cte_default
                ).value
            ),
            elapsed_time_weight=float(
                self.declare_parameter(
                    "objective.elapsed_time_weight", 1.0
                ).value
            ),
            front_cte_squared_weight=float(
                self.declare_parameter(
                    "objective.front_cte_squared_weight", 30.0
                ).value
            ),
            rear_cte_squared_weight=float(
                self.declare_parameter(
                    "objective.rear_cte_squared_weight", 30.0
                ).value
            ),
            competition_overspeed_penalty_s=float(
                self.declare_parameter(
                    "objective.competition_overspeed_penalty_s", 15.0
                ).value
            ),
            competition_overspeed_interval_s=float(
                self.declare_parameter(
                    "objective.competition_overspeed_interval_s", 3.0
                ).value
            ),
            target_overspeed_squared_weight=float(
                self.declare_parameter(
                    "objective.target_overspeed_squared_weight", 1.0
                ).value
            ),
            unnecessary_brake_squared_weight=float(
                self.declare_parameter(
                    "objective.unnecessary_brake_squared_weight", 5.0
                ).value
            ),
            brake_saturation_time_weight=float(
                self.declare_parameter(
                    "objective.brake_saturation_time_weight", 1.0
                ).value
            ),
            maximum_collision_count=int(
                self.declare_parameter(
                    "objective.maximum_collision_count", 0
                ).value
            ),
            minimum_local_planner_active_s=float(
                self.declare_parameter(
                    "objective.minimum_local_planner_active_sec", 0.0
                ).value
            ),
            collision_penalty=float(
                self.declare_parameter(
                    "objective.collision_penalty", 5000.0
                ).value
            ),
            local_planner_failure_time_weight=float(
                self.declare_parameter(
                    "objective.local_planner_failure_time_weight", 0.0
                ).value
            ),
            incomplete_penalty=float(
                self.declare_parameter(
                    "objective.incomplete_penalty", 2000.0
                ).value
            ),
            incomplete_cte_weight=float(
                self.declare_parameter(
                    "objective.incomplete_cte_weight", 0.0
                ).value
            ),
        )
        self.experiment_context = {
            "scenario": str(
                self.declare_parameter(
                    "experiment.scenario", "unspecified"
                ).value
            ),
            "weather": str(
                self.declare_parameter(
                    "experiment.weather", "unspecified"
                ).value
            ),
            "morai_version": str(
                self.declare_parameter(
                    "experiment.morai_version", "S4.251001"
                ).value
            ),
            "code_revision": str(
                self.declare_parameter(
                    "experiment.code_revision", "unspecified"
                ).value
            ),
            "vehicle_profile_id": str(
                self.declare_parameter(
                    "experiment.vehicle_profile_id",
                    "20260727-ioniq5-accelerator40-brake20-v1",
                ).value
            ),
            "timing_source": "morai_ego_status_device_stamp",
            "data_dir": str(
                self.declare_parameter("experiment.data_dir", "").value
            ),
            "route_corridor_file": str(
                self.declare_parameter(
                    "experiment.route_corridor_file",
                    "map/route_corridor.json",
                ).value
            ),
            "runtime_source_digest": runtime_source_digest(
                Path(__file__).resolve().parent
            ),
        }
        self._scenario_identity: ScenarioIdentity | None = None
        if self.spec.algorithm == "dwa":
            scenario_file = str(
                self.declare_parameter(
                    "experiment.scenario_file", ""
                ).value
            )
            self._scenario_identity = validate_scenario_identity(
                scenario_file=scenario_file,
                experiment_scenario=self.experiment_context["scenario"],
                save_root=default_morai_save_root(),
            )
            self.experiment_context.update(
                {
                    "scenario_file_name": (
                        self._scenario_identity.actual_path.name
                    ),
                    "scenario_file_sha256": (
                        self._scenario_identity.sha256
                    ),
                }
            )
        if self.storage_url.startswith("postgresql") and any(
            self.experiment_context[name] == "unspecified"
            for name in ("scenario", "weather", "code_revision")
        ):
            raise ValueError(
                "distributed study requires explicit scenario, weather, "
                "and code_revision on every worker"
            )
        self.experiment_context["runtime_artifacts"] = (
            self._runtime_artifacts()
        )

        self.runner = RosMoraiGlobalPathRunner(self)
        self._database_guard_lock = threading.Lock()
        self._database_outage_started_s: float | None = None
        self._database_abort_reason = ""
        self._database_guard_stop = threading.Event()
        self.runner.set_external_abort_checker(
            self._external_abort_checker
        )
        self._database_guard = threading.Thread(
            target=self._run_database_guard,
            name="optuna-database-guard",
            daemon=True,
        )
        self._database_guard.start()
        self._experiment_fingerprint = ""
        self._route: RouteSnapshot | None = None
        self._course_length_m = 0.0
        self._consecutive_infrastructure_failures = 0
        self._worker = threading.Thread(
            target=self._run_tuning, name="global-path-tuning", daemon=True
        )
        self._worker_started = False
        # Starting from the constructor can issue the first service request
        # before main() has entered executor.spin().  Dispatch the start from
        # the executor so subscriptions and service responses can be handled
        # before the worker waits on them.
        self._start_timer = self.create_timer(0.05, self._start_worker)

    def _external_abort_checker(self) -> str:
        with self._database_guard_lock:
            return self._database_abort_reason

    def _database_is_available(self) -> bool:
        if not self.storage_url.startswith("postgresql+psycopg://"):
            return True
        import psycopg

        connection_url = self.storage_url.replace(
            "postgresql+psycopg://", "postgresql://", 1
        )
        with psycopg.connect(
            connection_url,
            connect_timeout=self.database_connect_timeout_s,
            application_name=f"ad_tuning_guard_{self.worker_id}"[:63],
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                return cursor.fetchone() == (1,)

    def _run_database_guard(self) -> None:
        while not self._database_guard_stop.is_set():
            available = False
            try:
                available = self._database_is_available()
            except Exception:
                available = False
            now_s = time.monotonic()
            with self._database_guard_lock:
                if available:
                    self._database_outage_started_s = None
                    self._database_abort_reason = ""
                else:
                    if self._database_outage_started_s is None:
                        self._database_outage_started_s = now_s
                    outage_s = now_s - self._database_outage_started_s
                    if outage_s >= self.database_outage_abort_s:
                        self._database_abort_reason = (
                            "PostgreSQL outage exceeded tuning safety limit"
                        )
            self._database_guard_stop.wait(
                self.database_health_check_interval_s
            )

    def _stop_database_guard(self) -> None:
        self._database_guard_stop.set()
        if (
            self._database_guard.is_alive()
            and threading.current_thread() is not self._database_guard
        ):
            self._database_guard.join(
                timeout=self.database_connect_timeout_s + 1.0
            )

    def _runtime_artifacts(self) -> dict[str, dict[str, object]]:
        planner_prefix = Path(get_package_prefix("ad_planner"))
        control_prefix = Path(get_package_prefix("ad_control"))
        planner_share = Path(get_package_share_directory("ad_planner"))
        tuning_share = Path(get_package_share_directory("ad_tuning"))
        bridge_share = Path(
            get_package_share_directory("ad_morai_bridge")
        )
        bridge_dev_share = Path(
            get_package_share_directory("ad_morai_bridge_dev")
        )
        description_share = Path(
            get_package_share_directory("ad_description")
        )
        localization_share = Path(
            get_package_share_directory("ad_localization")
        )
        rclcpp_share = Path(get_package_share_directory("rclcpp"))
        rcl_share = Path(get_package_share_directory("rcl"))
        rmw_share = Path(get_package_share_directory("rmw"))
        tf2_ros_share = Path(get_package_share_directory("tf2_ros"))
        save_root = default_morai_save_root()
        scenario = Path(str(self.experiment_context["scenario"]))
        if scenario.suffix != ".json":
            scenario = scenario.with_suffix(".json")
        network_file = Path(
            os.environ.get(
                "AD_TUNING_MORAI_NETWORK_FILE",
                str(
                    save_root
                    / "Network"
                    / "25.S4.MolitComp03"
                    / "NetworkInfo_2023_Hyundai_Ioniq5.json"
                ),
            )
        )
        sensor_file = Path(
            os.environ.get(
                "AD_TUNING_MORAI_SENSOR_FILE",
                str(
                    save_root
                    / "Sensor"
                    / "25.S4.MolitComp03"
                    / "SensorInfo_2023_Hyundai_Ioniq5.json"
                ),
            )
        )
        tuning_launch_filename = (
            "dwa_morai_optuna.launch.py"
            if self.spec.algorithm == "dwa"
            else "profile_stanley_morai_optuna.launch.py"
        )
        components_filename = (
            "dwa_components.yaml"
            if self.spec.algorithm == "dwa"
            else "profile_stanley_components.yaml"
        )
        artifacts = {
            "ad_planner_executable": (
                planner_prefix / "lib" / "ad_planner" / "ad_planner_node"
            ),
            "ad_control_archive": (
                control_prefix / "lib" / "libad_control_core.a"
            ),
            "ad_planner_core_archive": (
                planner_prefix / "lib" / "libad_planner_core.a"
            ),
            "planner_yaml": planner_share / "config" / "planner.yaml",
            "tuning_yaml": (
                tuning_share / "config" / self.spec.tuning_config_filename
            ),
            "morai_tuning_bridge_yaml": (
                tuning_share / "config" / self.spec.bridge_config_filename
            ),
            "tuning_launch_python": (
                tuning_share / "launch" / tuning_launch_filename
            ),
            "tuning_components_yaml": (
                tuning_share / "config" / components_filename
            ),
            "bringup_stack_python": _python_module_source(
                "ad_bringup.bringup_stack"
            ),
            "bringup_component_config_python": _python_module_source(
                "ad_bringup.component_config"
            ),
            "description_launch_python": (
                description_share / "launch" / "description.launch.py"
            ),
            "localization_launch_python": (
                localization_share / "launch" / "localization.launch.py"
            ),
            "planner_launch_python": (
                planner_share / "launch" / "planner.launch.py"
            ),
            "morai_bridge_launch_python": (
                bridge_share / "launch" / "bridge.launch.py"
            ),
            "morai_reset_bridge_python": _python_module_source(
                "ad_morai_bridge_dev.bridge.node"
            ),
            "morai_reset_yaml": (
                tuning_share / "config" / "morai_reset.yaml"
            ),
            "vehicle_parameters_yaml": (
                description_share / "config" / "vehicle_parameters.yaml"
            ),
            "sensor_mounts_yaml": (
                description_share / "config" / "sensor_mounts.yaml"
            ),
            "morai_network_json": network_file,
            "morai_sensor_json": sensor_file,
            "ros_rclcpp_package_xml": rclcpp_share / "package.xml",
            "ros_rcl_package_xml": rcl_share / "package.xml",
            "ros_rmw_package_xml": rmw_share / "package.xml",
            "ros_tf2_ros_package_xml": tf2_ros_share / "package.xml",
        }
        if self.spec.algorithm == "dwa":
            if self._scenario_identity is None:
                raise RuntimeError("DWA scenario identity was not validated")
            lidar_prefix = Path(get_package_prefix("ad_lidar_perception"))
            lidar_share = Path(
                get_package_share_directory("ad_lidar_perception")
            )
            patchwork_prefix = Path(get_package_prefix("patchworkpp"))
            tracker_prefix = Path(
                get_package_prefix("autoware_multi_object_tracker")
            )
            tracker_share = Path(
                get_package_share_directory("autoware_multi_object_tracker")
            )
            data_dir_text = str(
                self.experiment_context["data_dir"]
            ).strip()
            if not data_dir_text:
                raise ValueError(
                    "DWA tuning requires an explicit experiment.data_dir"
                )
            data_dir = Path(data_dir_text).expanduser()
            route_corridor_text = str(
                self.experiment_context["route_corridor_file"]
            ).strip()
            if not route_corridor_text:
                route_corridor_text = "map/route_corridor.json"
            route_corridor_path = Path(route_corridor_text).expanduser()
            if not route_corridor_path.is_absolute():
                route_corridor_path = data_dir / route_corridor_path
            artifacts.update(
                {
                    "dwa_yaml": (
                        planner_share
                        / "config"
                        / "local_planning"
                        / "dwa.yaml"
                    ),
                    "dwa_components_yaml": (
                        tuning_share / "config" / "dwa_components.yaml"
                    ),
                    "lidar_perception_launch_python": (
                        lidar_share
                        / "launch"
                        / "lidar_perception.launch.py"
                    ),
                    "lidar_ground_segmentation_launch_python": (
                        lidar_share
                        / "launch"
                        / "ground_segmentation.launch.py"
                    ),
                    "lidar_occupancy_grid_launch_python": (
                        lidar_share
                        / "launch"
                        / "occupancy_grid.launch.py"
                    ),
                    "lidar_euclidean_clustering_launch_python": (
                        lidar_share
                        / "launch"
                        / "euclidean_clustering.launch.py"
                    ),
                    "lidar_tracking_launch_python": (
                        lidar_share / "launch" / "tracking.launch.py"
                    ),
                    "lidar_prediction_launch_python": (
                        lidar_share / "launch" / "prediction.launch.py"
                    ),
                    "lidar_dynamic_occupancy_grid_launch_python": (
                        lidar_share
                        / "launch"
                        / "dynamic_occupancy_grid.launch.py"
                    ),
                    "lidar_combined_occupancy_grid_launch_python": (
                        lidar_share
                        / "launch"
                        / "combined_occupancy_grid.launch.py"
                    ),
                    "lidar_selection_python": _python_module_source(
                        "ad_lidar_perception.selection"
                    ),
                    "scenario_reset_node_python": _python_module_source(
                        "ad_morai_bridge_dev.scenarios.reset_node"
                    ),
                    "scenario_reset_python": _python_module_source(
                        "ad_morai_bridge_dev.scenarios.reset"
                    ),
                    "morai_grpc_client_python": _python_module_source(
                        "ad_morai_bridge_dev.simulator_grpc.client"
                    ),
                    "morai_grpc_descriptors_python": _python_module_source(
                        "ad_morai_bridge_dev.simulator_grpc.descriptors"
                    ),
                    "morai_grpc_descriptor_set": (
                        bridge_dev_share / "data" / "morai_api.desc"
                    ),
                    "occupancy_grid_executable": (
                        lidar_prefix
                        / "lib"
                        / "ad_lidar_perception"
                        / "ad_lidar_perception_node"
                    ),
                    "dynamic_occupancy_grid_executable": (
                        lidar_prefix
                        / "lib"
                        / "ad_lidar_perception"
                        / "ad_dynamic_occupancy_grid_node"
                    ),
                    "prediction_executable": (
                        lidar_prefix
                        / "lib"
                        / "ad_lidar_perception"
                        / "ad_autoware_prediction_node"
                    ),
                    "adaptive_euclidean_cluster_executable": (
                        lidar_prefix
                        / "lib"
                        / "ad_lidar_perception"
                        / "ad_adaptive_euclidean_cluster_node"
                    ),
                    "multi_object_tracker_executable": (
                        tracker_prefix
                        / "lib"
                        / "autoware_multi_object_tracker"
                        / "multi_object_tracker_node"
                    ),
                    "road_corridor_mask_executable": (
                        planner_prefix
                        / "lib"
                        / "ad_planner"
                        / "ad_road_corridor_mask_node"
                    ),
                    "road_corridor_mask_yaml": (
                        planner_share
                        / "config"
                        / "road_corridor_mask.yaml"
                    ),
                    "occupancy_grid_combiner_executable": (
                        lidar_prefix
                        / "lib"
                        / "ad_lidar_perception"
                        / "ad_combined_occupancy_grid_node"
                    ),
                    "patchworkpp_executable": (
                        patchwork_prefix
                        / "lib"
                        / "patchworkpp"
                        / "patchworkpp_node"
                    ),
                    "lidar_perception_yaml": (
                        lidar_share / "config" / "lidar_perception.yaml"
                    ),
                    "autoware_perception_lock_yaml": (
                        lidar_share
                        / "config"
                        / "autoware_perception.lock.yaml"
                    ),
                    "adaptive_euclidean_cluster_yaml": (
                        lidar_share
                        / "config"
                        / "clustering"
                        / "adaptive_euclidean_cluster.yaml"
                    ),
                    "finite_point_filter_yaml": (
                        lidar_share
                        / "config"
                        / "preprocessing"
                        / "finite_point_filter.yaml"
                    ),
                    "occupancy_grid_yaml": (
                        lidar_share / "config" / "occupancy_grid" / "static.yaml"
                    ),
                    "combined_occupancy_grid_yaml": (
                        lidar_share
                        / "config"
                        / "occupancy_grid"
                        / "combined.yaml"
                    ),
                    "dynamic_occupancy_grid_yaml": (
                        lidar_share
                        / "config"
                        / "occupancy_grid"
                        / "dynamic.yaml"
                    ),
                    "prediction_yaml": (
                        lidar_share
                        / "config"
                        / "tracking"
                        / "prediction.yaml"
                    ),
                    "tracker_yaml": (
                        lidar_share
                        / "config"
                        / "tracking"
                        / "autoware.yaml"
                    ),
                    "tracker_launch_xml": (
                        tracker_share
                        / "launch"
                        / "multi_object_tracker.launch.xml"
                    ),
                    "ground_segmentation_yaml": (
                        lidar_share
                        / "config"
                        / "preprocessing"
                        / "ground_segmentation.yaml"
                    ),
                    "route_corridor_json": (
                        route_corridor_path
                    ),
                    "morai_scenario_reset_json": (
                        self._scenario_identity.actual_path
                    ),
                }
            )
        if self._scenario_identity is not None:
            artifacts["morai_scenario_json"] = (
                self._scenario_identity.expected_path
            )
        elif str(self.experiment_context["scenario"]) != "unspecified":
            artifacts["morai_scenario_json"] = (
                save_root / "Scenario" / scenario
            )
        return runtime_artifact_manifest(artifacts)

    def _start_worker(self) -> None:
        self._start_timer.cancel()
        if self._worker_started:
            return
        self._worker_started = True
        self._worker.start()

    @staticmethod
    def _completed_count(study: optuna.Study) -> int:
        return sum(
            trial.state is optuna.trial.TrialState.COMPLETE
            for trial in study.trials
        )

    @staticmethod
    def _finished_count(study: optuna.Study) -> int:
        finished = {
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.FAIL,
            optuna.trial.TrialState.PRUNED,
        }
        return sum(trial.state in finished for trial in study.trials)

    def _feasible_count(self, study: optuna.Study) -> int:
        return sum(
            trial.state is optuna.trial.TrialState.COMPLETE
            and trial.user_attrs.get("experiment_fingerprint")
            == self._experiment_fingerprint
            and trial.user_attrs.get("feasible") is True
            for trial in study.trials
        )

    def _create_study(
        self,
        experiment_fingerprint: str,
        experiment_metadata: dict[str, object],
    ) -> optuna.Study:
        attempts = 0
        while rclpy.ok():
            try:
                return self.spec.create_study(
                    self.output_dir,
                    experiment_fingerprint,
                    self.seed,
                    storage_url=self.storage_url,
                    worker_id=self.worker_id,
                    heartbeat_interval_s=self.heartbeat_interval_s,
                    heartbeat_grace_period_s=(
                        self.heartbeat_grace_period_s
                    ),
                    heartbeat_retry_count=self.heartbeat_retry_count,
                    connect_timeout_s=self.database_connect_timeout_s,
                    experiment_metadata=experiment_metadata,
                    inherit_minimum_complete_trials=(
                        self.inherit_minimum_complete_trials
                    ),
                    inherit_top_k=self.inherit_top_k,
                )
            except DATABASE_EXCEPTIONS as error:
                attempts += 1
                if (
                    self.database_retry_attempts > 0
                    and attempts >= self.database_retry_attempts
                ):
                    raise RuntimeError(
                        "Optuna database connection retry limit exceeded"
                    ) from error
                self.runner.hold_control(True)
                self.get_logger().error(
                    "Optuna database unavailable; planner held, retrying "
                    f"in {self.database_retry_interval_s:.1f} s "
                    f"(attempt {attempts}): {error}"
                )
                time.sleep(self.database_retry_interval_s)
        raise RuntimeError("ROS shutdown while connecting to Optuna database")

    def _experiment(
        self, route: RouteSnapshot, course_length_m: float
    ) -> tuple[str, dict[str, object]]:
        metadata: dict[str, object] = {
            "search_space_version": self.spec.study_version,
            "route_digest": route.digest,
            "route_length_m": round(route.length_m, 3),
            "course_length_m": round(course_length_m, 3),
            "fixed_parameters": self.spec.fixed_parameters,
            "stack_conditions": self.spec.stack_conditions,
            "objective": asdict(self.objective),
            "context": self.experiment_context,
        }
        return build_experiment_fingerprint(metadata)

    def _trial_row(
        self,
        trial: optuna.Trial,
        candidate: dict[str, float],
        metrics,
        evaluation,
        study_value: float,
        trajectory: str,
        infrastructure_failure: bool,
    ) -> dict[str, object]:
        if self._route is None:
            raise RuntimeError("route was not initialized")
        return {
            "status": (
                "failed" if infrastructure_failure else "completed"
            ),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "trial": trial.number,
            "worker_id": self.worker_id,
            "experiment_fingerprint": self._experiment_fingerprint,
            "route_digest": self._route.digest,
            "search_space_version": self.spec.study_version,
            "route_length_m": round(self._route.length_m, 3),
            "course_length_m": round(self._course_length_m, 3),
            "parameters": candidate,
            "fixed_parameters": self.spec.fixed_parameters,
            "metrics": asdict(metrics),
            "feasible": evaluation.feasible,
            "cost": round(evaluation.cost, 6),
            "optuna_value": round(study_value, 6),
            "trajectory": trajectory,
        }

    @staticmethod
    def _best_trial(
        study: optuna.Study,
        experiment_fingerprint: str,
    ) -> optuna.trial.FrozenTrial | None:
        feasible = [
            trial
            for trial in study.get_trials(
                deepcopy=False,
                states=(optuna.trial.TrialState.COMPLETE,),
            )
            if trial.user_attrs.get("experiment_fingerprint")
            == experiment_fingerprint
            and trial.user_attrs.get("feasible") is True
            and isinstance(trial.user_attrs.get("cost"), (int, float))
        ]
        if not feasible:
            return None
        return min(
            feasible, key=lambda trial: float(trial.user_attrs["cost"])
        )

    def _best_row(
        self, trial: optuna.trial.FrozenTrial
    ) -> dict[str, object]:
        attrs = trial.user_attrs
        return {
            "status": "completed",
            "timestamp_utc": attrs.get("timestamp_utc", ""),
            "trial": trial.number,
            "worker_id": attrs.get("worker_id", ""),
            "experiment_fingerprint": attrs.get(
                "experiment_fingerprint", ""
            ),
            "route_digest": attrs.get("route_digest", ""),
            "search_space_version": self.spec.study_version,
            "parameters": dict(trial.params),
            "metrics": attrs["metrics"],
            "feasible": True,
            "cost": float(attrs["cost"]),
            "optuna_value": trial.value,
            "trajectory": attrs.get("trajectory", ""),
        }

    def _export_global_best(self, study: optuna.Study) -> str:
        best = self._best_trial(study, self._experiment_fingerprint)
        if best is None:
            return "best=none(feasible trial required)"
        self.storage.write_best(
            self._best_row(best), self.spec.fixed_parameters
        )
        return (
            f"global_best=#{best.number} "
            f"cost={float(best.user_attrs['cost']):.3f}"
        )

    def _objective_for_study(self, trial: optuna.Trial) -> float:
        if not rclpy.ok():
            raise InfrastructureTrialFailure("ROS is shutting down")
        candidate = self.spec.suggest_parameters(trial)
        applied = dict(candidate)
        applied.update(self.spec.fixed_parameters)
        trial.set_user_attr("worker_id", self.worker_id)
        trial.set_user_attr(
            "experiment_fingerprint", self._experiment_fingerprint
        )
        trial.set_user_attr(
            "route_digest", self._route.digest if self._route else ""
        )
        trial.set_user_attr(
            "search_space_version", self.spec.study_version
        )
        trial.set_user_attr(
            "timestamp_utc", datetime.now(timezone.utc).isoformat()
        )
        self.get_logger().info(
            f"worker={self.worker_id} trial {trial.number} starting: "
            + ", ".join(
                f"{name}={value:.4g}"
                for name, value in candidate.items()
            )
        )

        metrics, samples = self.runner.run_trial(applied)
        evaluation = evaluate_trial(
            metrics, self._course_length_m, self.objective
        )
        infrastructure_failure = (
            metrics.reset_failed
            or metrics.disconnected
            or metrics.aborted
        )
        study_value = evaluation.cost
        trajectory = self.storage.write_trajectory(trial.number, samples)
        row = self._trial_row(
            trial,
            candidate,
            metrics,
            evaluation,
            study_value,
            trajectory,
            infrastructure_failure,
        )
        self.storage.stage_pending(row)
        self.storage.append(row)

        trial.set_user_attr("metrics", asdict(metrics))
        trial.set_user_attr("feasible", evaluation.feasible)
        trial.set_user_attr("cost", float(evaluation.cost))
        trial.set_user_attr("optuna_value", float(study_value))
        trial.set_user_attr("reason", metrics.reason)
        trial.set_user_attr("trajectory", trajectory)
        trial.set_user_attr(
            "constraint_values",
            [
                0.0 if metrics.completed else 1.0,
                float(metrics.max_cte_m - self.objective.maximum_cte_m),
                float(
                    metrics.rear_max_cte_m
                    - self.objective.maximum_cte_m
                ),
                float(
                    metrics.collision_count
                    - self.objective.maximum_collision_count
                ),
                float(
                    self.objective.minimum_local_planner_active_s
                    - metrics.local_planner_active_s
                ),
            ],
        )
        self.get_logger().info(
            f"trial {trial.number} measured: "
            f"completed={metrics.completed}, "
            f"progress={metrics.progress_m:.1f} m, "
            f"time={metrics.elapsed_s:.2f} s, "
            f"front_max_cte={metrics.max_cte_m:.3f} m, "
            f"rear_max_cte={metrics.rear_max_cte_m:.3f} m, "
            f"collisions={metrics.collision_count}, "
            f"local_planner={metrics.local_planner_active_s:.2f} s, "
            f"overspeed={metrics.overspeed_s:.2f} s, "
            f"cost={evaluation.cost:.3f}, optuna={study_value:.3f}, "
            f"feasible={evaluation.feasible}, reason={metrics.reason}"
        )
        if infrastructure_failure:
            self._consecutive_infrastructure_failures += 1
            raise InfrastructureTrialFailure(metrics.reason)
        self._consecutive_infrastructure_failures = 0
        return float(study_value)

    def _after_trial(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
    ) -> None:
        # Optuna calls callbacks only after the RDB trial state is committed.
        self.storage.acknowledge_pending(trial.number)
        try:
            best_text = self._export_global_best(study)
            self.get_logger().info(
                f"trial {trial.number} committed as {trial.state.name}; "
                f"{best_text}"
            )
        except Exception as error:
            self.get_logger().error(
                f"trial committed but best export failed: {error}"
            )

    def _run_tuning(self) -> None:
        try:
            route = self.runner.wait_until_ready()
            self.runner.verify_parameters(self.spec.stack_conditions)
            self._route = route
            course_length = (
                route.length_m - self.runner.completion_margin_m
                if self.runner.course_length_m <= 0.0
                else min(self.runner.course_length_m, route.length_m)
            )
            self._course_length_m = max(1.0, course_length)
            (
                self._experiment_fingerprint,
                experiment_metadata,
            ) = self._experiment(route, self._course_length_m)
            study = self._create_study(
                self._experiment_fingerprint, experiment_metadata
            )
            pending = self.storage.pending_results()
            if pending:
                self.get_logger().warning(
                    f"{len(pending)} unacknowledged local trial result(s) "
                    "remain; stale RDB trials will be failed and retried"
                )
            target_trials = (
                self.maximum_trials
                if self.maximum_trials > 0
                else "unlimited"
            )
            self.get_logger().info(
                f"distributed ROS2 {self.spec.algorithm} tuning ready: "
                f"worker={self.worker_id}, "
                f"storage={redact_storage_url(self.storage_url)}, "
                f"experiment={self._experiment_fingerprint}, "
                f"route={route.digest}, points={len(route.xy)}, "
                f"course={self._course_length_m:.1f} m, "
                f"completed={self._completed_count(study)}, "
                f"target={target_trials}"
            )

            tuning_started_s = time.monotonic()
            stop_reason = ""
            while rclpy.ok():
                completed_count = self._completed_count(study)
                feasible_count = self._feasible_count(study)
                finished_count = self._finished_count(study)
                if (
                    self.maximum_trials > 0
                    and completed_count >= self.maximum_trials
                    and feasible_count >= self.minimum_feasible_trials
                ):
                    stop_reason = "complete and feasible trial targets reached"
                    break
                if (
                    self.maximum_total_trials > 0
                    and finished_count >= self.maximum_total_trials
                ):
                    stop_reason = "maximum total trial limit reached"
                    break
                if (
                    self.maximum_worker_wall_time_s > 0.0
                    and time.monotonic() - tuning_started_s
                    >= self.maximum_worker_wall_time_s
                ):
                    stop_reason = "worker wall-time limit reached"
                    break
                if (
                    self._consecutive_infrastructure_failures
                    >= self.maximum_consecutive_infrastructure_failures
                ):
                    stop_reason = (
                        "consecutive infrastructure failure limit reached"
                    )
                    break
                try:
                    study.optimize(
                        self._objective_for_study,
                        n_trials=1,
                        catch=(InfrastructureTrialFailure,),
                        callbacks=[self._after_trial],
                        gc_after_trial=True,
                        show_progress_bar=False,
                    )
                except DATABASE_EXCEPTIONS as error:
                    self.runner.hold_control(True)
                    self.get_logger().error(
                        "database connection lost after a trial; local "
                        f"result retained and reconnecting: {error}"
                    )
                    study = self._create_study(
                        self._experiment_fingerprint,
                        experiment_metadata,
                    )

            self.runner.hold_control(True)
            feasible_count = self._feasible_count(study)
            self.get_logger().info(
                f"tuning loop stopped: {stop_reason or 'ROS shutdown'}; "
                f"complete={self._completed_count(study)}, "
                f"finished={self._finished_count(study)}, "
                f"feasible={feasible_count}"
            )
            best = self._best_trial(study, self._experiment_fingerprint)
            if best is not None:
                self._export_global_best(study)
                applied = dict(best.params)
                applied.update(self.spec.fixed_parameters)
                self.runner.hold_control(True)
                self.runner.apply_parameters(applied)
                self.runner.reset_tracker()
                self.get_logger().info(
                    f"tuning stopped safely; global best trial #{best.number} "
                    f"applied under full-brake hold. "
                    f"result={self.storage.best_path}"
                )
            if feasible_count < self.minimum_feasible_trials:
                self.get_logger().error(
                    "tuning stopped before the minimum feasible-trial "
                    f"requirement ({feasible_count}/"
                    f"{self.minimum_feasible_trials})"
                )
            elif best is None:
                self.get_logger().error(
                    "tuning stopped without a feasible completed trial"
                )
        except Exception as error:
            if rclpy.ok():
                try:
                    self.runner.hold_control(True)
                except Exception as hold_error:
                    self.get_logger().error(
                        "tuning failed and planner hold also failed: "
                        f"{hold_error}"
                    )
                self.get_logger().error(f"tuning stopped: {error}")
        finally:
            self.runner.close()
            self._stop_database_guard()
            if rclpy.ok():
                rclpy.shutdown()

    def stop(self) -> None:
        self.runner.close()
        self._stop_database_guard()
        self._start_timer.cancel()
        if self._worker_started and self._worker.is_alive():
            self._worker.join(timeout=3.0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GlobalPathTunerNode(PROFILE_STANLEY_SPEC)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def dwa_main(args=None) -> None:
    rclpy.init(args=args)
    node = GlobalPathTunerNode(DWA_SPEC)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
