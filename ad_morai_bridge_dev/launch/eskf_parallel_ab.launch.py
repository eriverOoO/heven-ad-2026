from pathlib import Path
import re
from typing import NamedTuple

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import yaml

from ad_morai_bridge_dev.eskf_experiment.artifacts import validate_run_id


ALLOWED_CANDIDATE_PARAMETERS = frozenset(
    {
        "gravity_mps2",
        "initial_imu_acc_bias_covariance",
        "stationary_initialization_estimate_accel_bias",
        "var_imu_acc_bias",
    }
)
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class CandidateConfig(NamedTuple):
    name: str
    enabled: bool
    parameters: dict[str, object]


class ExperimentConfig(NamedTuple):
    drive_enabled: bool
    profile: str
    shared_inputs: dict[str, str]
    candidates: tuple[CandidateConfig, ...]
    safety: dict[str, float]
    allowed_candidate_parameters: frozenset[str]


def _mapping(value: object, field: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return value


def _safe_name(value: object, field: str) -> str:
    name = str(value)
    if _SAFE_NAME.fullmatch(name) is None:
        raise ValueError(f"{field} has an unsafe name: {name!r}")
    return name


def _boolean(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be a boolean")
    return value


def load_experiment_config(path: Path) -> ExperimentConfig:
    document = _mapping(
        yaml.safe_load(Path(path).read_text(encoding="utf-8")), "document"
    )
    if document.get("schema_version") != 1:
        raise ValueError("unsupported ESKF experiment schema_version")

    shared = _mapping(document.get("shared_inputs", {}), "shared_inputs")
    expected_inputs = {
        "imu_topic": "/ad/localization/input/eskf_imu",
        "gnss_pose_topic": "/ad/localization/input/gnss_pose",
        "wheel_speed_topic": "/ad/localization/input/wheel_speed",
    }
    if shared and shared != expected_inputs:
        raise ValueError("shared_inputs must use the canonical corrected inputs")
    if not shared:
        shared = expected_inputs

    raw_candidates = _mapping(document.get("candidates"), "candidates")
    candidates = []
    for raw_name, raw_candidate in raw_candidates.items():
        name = _safe_name(raw_name, "candidate")
        candidate = _mapping(raw_candidate, f"candidate {name}")
        parameters = dict(
            _mapping(candidate.get("parameters", {}), f"candidate {name} parameters")
        )
        unknown = set(parameters) - ALLOWED_CANDIDATE_PARAMETERS
        if unknown:
            raise ValueError(
                f"unknown candidate parameter(s) for {name}: {sorted(unknown)}"
            )
        candidates.append(
            CandidateConfig(
                name=name,
                enabled=_boolean(
                    candidate.get("enabled", False), f"candidate {name} enabled"
                ),
                parameters=parameters,
            )
        )
    if not any(candidate.enabled for candidate in candidates):
        raise ValueError("at least one ESKF candidate must be enabled")

    safety_document = _mapping(document.get("safety", {}), "safety")
    safety = {str(key): float(value) for key, value in safety_document.items()}
    return ExperimentConfig(
        drive_enabled=_boolean(document.get("drive_enabled", False), "drive_enabled"),
        profile=str(document.get("profile", "stationary")),
        shared_inputs={str(key): str(value) for key, value in shared.items()},
        candidates=tuple(candidates),
        safety=safety,
        allowed_candidate_parameters=ALLOWED_CANDIDATE_PARAMETERS,
    )


def load_eskf_parameters(path: Path) -> dict[str, object]:
    document = _mapping(
        yaml.safe_load(Path(path).read_text(encoding="utf-8")), "ESKF document"
    )
    if len(document) != 1:
        raise ValueError("ESKF YAML must contain exactly one node root")
    try:
        node_document = next(iter(document.values()))
        parameters = node_document["ros__parameters"]
    except (StopIteration, KeyError, TypeError) as exc:
        raise ValueError("ESKF YAML must contain one node ros__parameters mapping") from exc
    return dict(_mapping(parameters, "ESKF ros__parameters"))


def make_candidate_node(
    *,
    run_id: str,
    candidate: CandidateConfig,
    base_parameters: dict[str, object],
    shared_inputs: dict[str, str] | None = None,
) -> Node:
    safe_run_id = validate_run_id(run_id)
    candidate_name = _safe_name(candidate.name, "candidate")
    unknown = set(candidate.parameters) - ALLOWED_CANDIDATE_PARAMETERS
    if unknown:
        raise ValueError(f"unknown candidate parameter(s): {sorted(unknown)}")

    inputs = shared_inputs or {
        "imu_topic": "/ad/localization/input/eskf_imu",
        "gnss_pose_topic": "/ad/localization/input/gnss_pose",
        "wheel_speed_topic": "/ad/localization/input/wheel_speed",
    }
    namespace = f"/ad/experiment/eskf/{safe_run_id}"
    candidate_prefix = f"{namespace}/{candidate_name}"
    parameters = dict(base_parameters)
    parameters.update(candidate.parameters)
    parameters.update(inputs)
    # These are appended last and cannot be overridden by experiment YAML.
    parameters.update(
        {
            "initial_pose_topic": f"{candidate_prefix}/initial_pose",
            "output_odometry_topic": f"{candidate_prefix}/odometry",
            "publish_tf": False,
            "publish_debug_topics": True,
        }
    )
    return Node(
        package="kalman_filter_localization",
        executable="ekf_localization_node",
        namespace=namespace,
        name=f"eskf_{candidate_name}",
        output="screen",
        parameters=[parameters],
    )


def _launch_setup(context):
    experiment_path = Path(LaunchConfiguration("experiment_config").perform(context))
    eskf_path = Path(LaunchConfiguration("eskf_config").perform(context))
    run_id = LaunchConfiguration("run_id").perform(context)
    experiment = load_experiment_config(experiment_path)
    base_parameters = load_eskf_parameters(eskf_path)
    candidates = [
        make_candidate_node(
            run_id=run_id,
            candidate=candidate,
            base_parameters=base_parameters,
            shared_inputs=experiment.shared_inputs,
        )
        for candidate in experiment.candidates
        if candidate.enabled
    ]
    namespace = f"/ad/experiment/eskf/{validate_run_id(run_id)}"
    harness = Node(
        package="ad_morai_bridge_dev",
        executable="eskf_ab_harness",
        namespace=namespace,
        name="eskf_ab_harness",
        output="screen",
        condition=IfCondition(LaunchConfiguration("harness_enabled")),
        parameters=[
            {
                "run_id": run_id,
                "experiment_config": str(experiment_path),
                "eskf_config": str(eskf_path),
                "grpc_target": LaunchConfiguration("grpc_target"),
                "grpc_timeout_sec": ParameterValue(
                    LaunchConfiguration("grpc_timeout_sec"), value_type=float
                ),
                "drive_enabled": ParameterValue(
                    LaunchConfiguration("drive_enabled"), value_type=bool
                ),
                "profile": LaunchConfiguration("profile"),
                "stationary_duration_sec": ParameterValue(
                    LaunchConfiguration("stationary_duration_sec"), value_type=float
                ),
                "maximum_pair_age_sec": ParameterValue(
                    LaunchConfiguration("maximum_pair_age_sec"), value_type=float
                ),
                "data_root": LaunchConfiguration("data_root"),
                "repository_root": LaunchConfiguration("repository_root"),
                "active_sensor_file": LaunchConfiguration("active_sensor_file"),
            }
        ],
    )
    shutdown_on_harness_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=harness,
            on_exit=[
                EmitEvent(
                    event=Shutdown(reason="ESKF A/B harness process exited")
                )
            ],
        ),
        condition=IfCondition(LaunchConfiguration("harness_enabled")),
    )
    # Register before starting the short-lived harness so even an immediate
    # startup failure tears down every candidate process.
    return [shutdown_on_harness_exit, *candidates, harness]


def generate_launch_description():
    share = Path(get_package_share_directory("ad_morai_bridge_dev"))
    localization_share = Path(get_package_share_directory("ad_localization"))
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "experiment_config",
                default_value=str(share / "config" / "eskf_parallel_ab.yaml"),
            ),
            DeclareLaunchArgument(
                "eskf_config",
                default_value=str(localization_share / "config" / "eskf.yaml"),
            ),
            DeclareLaunchArgument("run_id", default_value="stationary_ab"),
            DeclareLaunchArgument("grpc_target", default_value="127.0.0.1:7789"),
            DeclareLaunchArgument("grpc_timeout_sec", default_value="2.0"),
            DeclareLaunchArgument("drive_enabled", default_value="false"),
            DeclareLaunchArgument("harness_enabled", default_value="true"),
            # Empty delegates profile selection to the reviewed YAML.  An
            # explicit launch override remains available for diagnostics.
            DeclareLaunchArgument("profile", default_value=""),
            DeclareLaunchArgument("stationary_duration_sec", default_value="15.0"),
            DeclareLaunchArgument("maximum_pair_age_sec", default_value="0.10"),
            DeclareLaunchArgument("data_root", default_value=""),
            DeclareLaunchArgument("repository_root", default_value=""),
            DeclareLaunchArgument("active_sensor_file", default_value=""),
            OpaqueFunction(function=_launch_setup),
        ]
    )
