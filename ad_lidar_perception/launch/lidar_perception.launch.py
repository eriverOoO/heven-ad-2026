from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import SetParameter
from launch_ros.parameter_descriptions import ParameterValue

from ad_lidar_perception.selection import load_selection


RAW_TOPIC = "/ad/sensors/lidar/points"
DESKEWED_TOPIC = "/ad/perception/lidar/deskewed"
CROPPED_TOPIC = "/ad/perception/lidar/cropped"


def _launch_file(name: str) -> PythonLaunchDescriptionSource:
    package_share = Path(get_package_share_directory("ad_lidar_perception"))
    return PythonLaunchDescriptionSource(str(package_share / "launch" / name))


def _tracking_config(name: str) -> str:
    package_share = Path(get_package_share_directory("ad_lidar_perception"))
    return str(package_share / "config" / "tracking" / name)


def _include(name: str, arguments=None):
    return IncludeLaunchDescription(
        _launch_file(name),
        launch_arguments=(arguments or {}).items(),
    )


def _perform(context, name):
    return LaunchConfiguration(name).perform(context)


def _parse_enabled(name, value):
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise RuntimeError(f"{name} must be 'true' or 'false'")


def _parse_platform_profile(value):
    profile = str(value).strip()
    if profile not in {"morai", "real_hardware"}:
        raise RuntimeError(
            "platform_profile must be exactly 'morai' or 'real_hardware'"
        )
    return profile


def _launch_setup(context):
    composition_path = Path(_perform(context, "composition_config"))
    detector_backend = _perform(context, "detector_backend").strip()
    if detector_backend not in {
        "euclidean", "centerpoint", "autoware_centerpoint",
    }:
        raise RuntimeError(
            "detector_backend must be euclidean, centerpoint, or "
            "autoware_centerpoint"
        )
    # ``centerpoint`` is the historical local OpenPCDet wrapper.  Keep it
    # intact for reproducibility.  The competition candidate is explicitly
    # named ``autoware_centerpoint`` so that it cannot silently select the
    # wrong model/runtime contract.
    heven_centerpoint = detector_backend == "centerpoint"
    autoware_centerpoint = detector_backend == "autoware_centerpoint"
    selection_path = (
        Path(_perform(context, "autoware_selection_config"))
        if autoware_centerpoint
        else composition_path
    )
    selection = load_selection(selection_path)
    if autoware_centerpoint and selection.detector.backend != "centerpoint":
        raise RuntimeError(
            "autoware_centerpoint requires a selection whose detector "
            "backend is exactly centerpoint"
        )
    tracker_backend_override = _perform(context, "tracker_backend").strip()
    if tracker_backend_override not in {"", "autoware", "ab3dmot"}:
        raise RuntimeError(
            "tracker_backend must be empty, autoware, or ab3dmot"
        )
    tracker_backend = tracker_backend_override or selection.tracker.backend
    ab3dmot_state_estimator = LaunchConfiguration(
        "ab3dmot_state_estimator", default="linear_kf"
    ).perform(context).strip()
    if ab3dmot_state_estimator not in {"linear_kf", "kalmannet"}:
        raise RuntimeError(
            "ab3dmot_state_estimator must be linear_kf or kalmannet"
        )
    prediction_yaw_rate_source = (
        LaunchConfiguration("prediction_yaw_rate_source", default="tracker")
        .perform(context)
        .strip()
    )
    if prediction_yaw_rate_source not in {"tracker", "motion_history"}:
        raise RuntimeError(
            "prediction_yaw_rate_source must be tracker or motion_history"
        )
    use_predicted_future_sweep = _parse_enabled(
        "use_predicted_future_sweep",
        LaunchConfiguration(
            "use_predicted_future_sweep", default="false"
        ).perform(context),
    )
    future_sweep_horizon_s = (
        LaunchConfiguration("future_sweep_horizon_s", default="3.0")
        .perform(context)
        .strip()
    )
    try:
        parsed_future_sweep_horizon_s = float(future_sweep_horizon_s)
    except ValueError as error:
        raise RuntimeError(
            "future_sweep_horizon_s must be a number"
        ) from error
    if not 0.0 <= parsed_future_sweep_horizon_s <= 10.0:
        raise RuntimeError("future_sweep_horizon_s must be in [0, 10]")
    if selection.detector.build_only:
        raise RuntimeError(
            "build_only selection cannot activate the runtime composition"
        )

    platform_profile = _parse_platform_profile(
        _perform(context, "platform_profile")
    )
    ground_enabled = _parse_enabled(
        "start_ground_segmentation",
        _perform(context, "start_ground_segmentation"),
    )
    deskew_enabled = _parse_enabled(
        "deskew_enabled", _perform(context, "deskew_enabled")
    )
    if platform_profile == "morai" and deskew_enabled:
        raise RuntimeError(
            "MORAI instantaneous-scan profile: motion deskew is prohibited"
        )
    self_crop_enabled = _parse_enabled(
        "self_crop_enabled", _perform(context, "self_crop_enabled")
    )
    _parse_enabled(
        "self_crop_input_reliable",
        _perform(context, "self_crop_input_reliable"),
    )
    point_layout_adapter_available = _parse_enabled(
        "point_layout_adapter_enabled",
        _perform(context, "point_layout_adapter_enabled"),
    )
    _parse_enabled(
        "patchwork_leveling_enabled",
        _perform(context, "patchwork_leveling_enabled"),
    )
    _parse_enabled(
        "finite_filter_enabled", _perform(context, "finite_filter_enabled")
    )
    _parse_enabled(
        "densifier_enabled", _perform(context, "densifier_enabled")
    )
    start_visualization = _parse_enabled(
        "start_visualization", _perform(context, "start_visualization")
    )
    start_rviz = _parse_enabled(
        "start_rviz", _perform(context, "start_rviz")
    )
    deskew_mode = _perform(context, "deskew_mode")
    if deskew_mode not in {"2d", "3d"}:
        raise RuntimeError("deskew_mode must be '2d' or '3d'")
    if (
        not heven_centerpoint
        and selection.detector.backend == "euclidean_cluster"
        and not ground_enabled
    ):
        raise RuntimeError(
            "euclidean_cluster detector requires ground segmentation"
        )
    learned_detector = selection.detector.backend not in {
        "none",
        "euclidean_cluster",
    }
    if learned_detector and not point_layout_adapter_available:
        raise RuntimeError(
            "point layout adapter is required by the selected learned detector"
        )

    selected_topic = _perform(context, "raw_input_topic")
    if deskew_enabled:
        selected_topic = DESKEWED_TOPIC
    if self_crop_enabled:
        selected_topic = CROPPED_TOPIC

    actions = [
        _include(
            "preprocessing.launch.py",
            {
                "platform_profile": LaunchConfiguration(
                    "platform_profile"
                ),
                "deskew_enabled": LaunchConfiguration("deskew_enabled"),
                "deskew_mode": LaunchConfiguration("deskew_mode"),
                "self_crop_enabled": LaunchConfiguration("self_crop_enabled"),
                "self_crop_input_reliable": LaunchConfiguration(
                    "self_crop_input_reliable"
                ),
                "raw_input_topic": LaunchConfiguration("raw_input_topic"),
                "crop_clearance_m": LaunchConfiguration(
                    "crop_clearance_m"
                ),
                "point_layout_adapter_enabled": (
                    "true" if learned_detector else "false"
                ),
            },
        )
    ]
    if ground_enabled:
        actions.append(
            _include(
                "ground_segmentation.launch.py",
                {
                    "patchwork_leveling_enabled": LaunchConfiguration(
                        "patchwork_leveling_enabled"
                    ),
                    "cropped_input_topic": selected_topic,
                    "ground_config": LaunchConfiguration("ground_config"),
                    "sensor_config": LaunchConfiguration("sensor_config"),
                    "sensor_profile": LaunchConfiguration("sensor_profile"),
                },
            )
        )

    if selection.occupancy.static_enabled:
        actions.append(
            _include(
                "occupancy_grid.launch.py",
                {"points_topic": selected_topic},
            )
        )

    if heven_centerpoint:
        actions.append(
            _include(
                "centerpoint_detector.launch.py",
                {
                    "detector_backend": "centerpoint",
                    "checkpoint_path": LaunchConfiguration("checkpoint_path"),
                    "score_threshold": LaunchConfiguration("score_threshold"),
                    "max_detections": LaunchConfiguration("max_detections"),
                    "device": LaunchConfiguration("device"),
                    "point_cloud_range": LaunchConfiguration("point_cloud_range"),
                    "enabled": LaunchConfiguration("centerpoint_enabled"),
                    "mock_mode": LaunchConfiguration("centerpoint_mock_mode"),
                    "openpcdet_root": LaunchConfiguration("openpcdet_root"),
                    "input_topic": selected_topic,
                },
            )
        )
    elif selection.detector.backend == "euclidean_cluster":
        actions.append(
            _include(
                "euclidean_clustering.launch.py",
                {
                    "finite_filter_enabled": LaunchConfiguration(
                        "finite_filter_enabled"
                    ),
                    "densifier_enabled": LaunchConfiguration(
                        "densifier_enabled"
                    ),
                    "cluster_config": LaunchConfiguration("cluster_config"),
                },
            )
        )
    elif selection.detector.backend != "none":
        actions.append(
            _include(
                "object_detection.launch.py",
                {"selection_config": str(selection_path)},
            )
        )

    if tracker_backend == "autoware":
        actions.extend(
            [
                _include(
                    "tracking.launch.py",
                    {"selection_config": str(selection_path)},
                ),
                _include(
                    "prediction.launch.py",
                    {"yaw_rate_source": prediction_yaw_rate_source},
                ),
            ]
        )
    elif tracker_backend == "ab3dmot":
        actions.extend(
            [
                _include(
                    "ab3dmot_tracker.launch.py",
                    {
                        "enabled": "true",
                        "input_topic": "/ad/perception/objects/detected",
                        "output_topic": "/ad/perception/objects/tracked",
                        "target_frame": "odom",
                        "config_path": _tracking_config(
                            "competition_mot_baseline_v1.yaml"
                        ),
                        "ab3dmot_root": LaunchConfiguration(
                            "ab3dmot_root"
                        ),
                        "association_metric": "euclidean",
                        "euclidean_gate_m": "3.0",
                        "matcher": "hungarian",
                        "state_estimator": ab3dmot_state_estimator,
                        "yaw_measurement_mode": "unobserved",
                        "kalmannet_checkpoint": LaunchConfiguration(
                            "ab3dmot_kalmannet_checkpoint", default=""
                        ).perform(context),
                        "kalmannet_device": LaunchConfiguration(
                            "ab3dmot_kalmannet_device", default="cpu"
                        ).perform(context),
                        "velocity_audit_enabled": "true",
                        "defer_until_tf_ready": _perform(
                            context, "ab3dmot_defer_until_tf_ready"
                        ),
                        "max_tf_wait_ms": _perform(
                            context, "ab3dmot_max_tf_wait_ms"
                        ),
                        "max_pending_detections": _perform(
                            context, "ab3dmot_max_pending_detections"
                        ),
                    },
                ),
                _include(
                    "prediction.launch.py",
                    {
                        "runtime_summary_interval_frames": "180",
                        "yaw_rate_source": prediction_yaw_rate_source,
                    },
                ),
            ]
        )

    if selection.occupancy.dynamic_enabled:
        dynamic_arguments = {
            "use_predicted_future_sweep": (
                "true" if use_predicted_future_sweep else "false"
            ),
            "future_sweep_horizon_s": future_sweep_horizon_s,
        }
        if tracker_backend == "ab3dmot":
            dynamic_arguments["runtime_summary_interval_frames"] = "180"
        actions.append(
            _include("dynamic_occupancy_grid.launch.py", dynamic_arguments)
        )

    if selection.occupancy.publish_combined:
        actions.append(_include("combined_occupancy_grid.launch.py"))

    # Opt-in, backend-agnostic planner-facing metric node. Default off: it only
    # observes the shared PredictedObjectArray + canonical odometry and adds no
    # behaviour. Works identically for the autoware and ab3dmot prediction paths.
    if _perform(context, "dynamic_object_risk").strip().lower() in {"1", "true"}:
        actions.append(_include("dynamic_object_risk.launch.py"))
    if start_visualization or start_rviz:
        actions.append(
            _include(
                "perception_visualization.launch.py",
                {
                    "start_rviz": "true" if start_rviz else "false",
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                },
            )
        )
    return actions


def generate_launch_description():
    package_share = Path(get_package_share_directory("ad_lidar_perception"))
    description_share = Path(get_package_share_directory("ad_description"))

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "platform_profile", default_value="morai"
            ),
            DeclareLaunchArgument(
                "composition_config",
                default_value=str(
                    package_share / "config" / "lidar_perception.yaml"
                ),
            ),
            DeclareLaunchArgument("detector_backend", default_value="euclidean"),
            DeclareLaunchArgument(
                "autoware_selection_config",
                default_value=str(
                    package_share
                    / "config"
                    / "experiments"
                    / "autoware_centerpoint_competition_candidate_v1.yaml"
                ),
                description=(
                    "Explicit pinned Autoware selection used only by "
                    "detector_backend:=autoware_centerpoint."
                ),
            ),
            DeclareLaunchArgument(
                "tracker_backend",
                default_value="",
                description=(
                    "Blank preserves the composition config (Autoware in the checked-in "
                    "default); explicitly select 'autoware' or opt-in 'ab3dmot'."
                ),
            ),
            DeclareLaunchArgument("checkpoint_path", default_value=""),
            DeclareLaunchArgument("score_threshold", default_value="0.1"),
            DeclareLaunchArgument("max_detections", default_value="500"),
            DeclareLaunchArgument("device", default_value="cuda:0"),
            DeclareLaunchArgument(
                "point_cloud_range",
                default_value="[-4.0,-25.0,-3.0,100.0,25.0,5.0]",
            ),
            DeclareLaunchArgument("centerpoint_enabled", default_value="true"),
            DeclareLaunchArgument("centerpoint_mock_mode", default_value="false"),
            DeclareLaunchArgument("openpcdet_root", default_value=""),
            DeclareLaunchArgument("raw_input_topic", default_value=RAW_TOPIC),
            DeclareLaunchArgument(
                "cluster_config",
                default_value=str(
                    package_share
                    / "config"
                    / "clustering"
                    / "adaptive_euclidean_cluster.yaml"
                ),
            ),
            DeclareLaunchArgument("crop_clearance_m", default_value="0.20"),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument(
                "start_ground_segmentation", default_value="true"
            ),
            DeclareLaunchArgument("deskew_enabled", default_value="false"),
            DeclareLaunchArgument("deskew_mode", default_value="3d"),
            DeclareLaunchArgument("self_crop_enabled", default_value="true"),
            DeclareLaunchArgument(
                "self_crop_input_reliable", default_value="false"
            ),
            DeclareLaunchArgument(
                "patchwork_leveling_enabled", default_value="false"
            ),
            DeclareLaunchArgument(
                "finite_filter_enabled", default_value="true"
            ),
            DeclareLaunchArgument("densifier_enabled", default_value="false"),
            DeclareLaunchArgument(
                "start_visualization", default_value="false"
            ),
            DeclareLaunchArgument("start_rviz", default_value="false"),
            DeclareLaunchArgument(
                "ab3dmot_defer_until_tf_ready",
                default_value="true",
                description=(
                    "ab3dmot_tracker_backend only: AB3DMOT Exact-Stamp TF "
                    "Deferred Processing v1 pass-through. See "
                    "ab3dmot_tracker.launch.py's 'defer_until_tf_ready' arg."
                ),
            ),
            DeclareLaunchArgument(
                "ab3dmot_max_tf_wait_ms", default_value="500"
            ),
            DeclareLaunchArgument(
                "ab3dmot_max_pending_detections", default_value="8"
            ),
            DeclareLaunchArgument(
                "ab3dmot_state_estimator",
                default_value="linear_kf",
                description=(
                    "Opt-in AB3DMOT estimator selector; the study launch "
                    "uses only linear_kf or the checkpoint-required kalmannet"
                ),
            ),
            DeclareLaunchArgument(
                "prediction_yaw_rate_source",
                default_value="tracker",
                description=(
                    "Curve-Aware Prediction v1: IMM yaw-rate source, forwarded "
                    "to prediction.launch.py for both tracker backends. "
                    "'tracker' (default) reproduces prior behaviour; "
                    "'motion_history' derives the turn rate from tracked "
                    "velocity history."
                ),
            ),
            DeclareLaunchArgument(
                "use_predicted_future_sweep",
                default_value="false",
                description=(
                    "Dynamic OGM Future Sweep v1: forward the predicted "
                    "trajectory keyframes into the dynamic occupancy grid so "
                    "moving-object occupancy covers future swept space. "
                    "Default false reproduces prior behaviour."
                ),
            ),
            DeclareLaunchArgument(
                "future_sweep_horizon_s",
                default_value="3.0",
                description=(
                    "Forward horizon in seconds of the Dynamic OGM future "
                    "sweep; 0.0 also reproduces the current-footprint-only "
                    "behaviour."
                ),
            ),
            DeclareLaunchArgument(
                "ab3dmot_kalmannet_checkpoint", default_value=""
            ),
            DeclareLaunchArgument(
                "ab3dmot_kalmannet_device", default_value="cpu"
            ),
            DeclareLaunchArgument("ab3dmot_root", default_value=""),
            DeclareLaunchArgument(
                "dynamic_object_risk",
                default_value="false",
                description=(
                    "Opt-in: start the planner-facing Dynamic Object Risk "
                    "Interface node (observational, no behaviour change)."
                ),
            ),
            DeclareLaunchArgument(
                "point_layout_adapter_enabled",
                default_value="true",
                description=(
                    "Capability gate for learned detector XYZIRC conversion; "
                    "unused for euclidean_cluster and none."
                ),
            ),
            DeclareLaunchArgument(
                "ground_config",
                default_value=str(
                    package_share
                    / "config"
                    / "preprocessing"
                    / "ground_segmentation.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "sensor_config",
                default_value=str(
                    description_share / "config" / "sensor_mounts.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "sensor_profile",
                default_value="",
                description="Blank selects active_profile from sensor_mounts.yaml.",
            ),
            SetParameter(
                name="use_sim_time",
                value=ParameterValue(
                    LaunchConfiguration("use_sim_time"), value_type=bool
                ),
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
