import math
from pathlib import Path, PurePosixPath

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import SetParameter
import yaml


SOURCE_TOPICS = (
    "/ad/sensors/lidar/points",
    "/tf",
    "/tf_static",
    "/ad/localization/odometry",
    "/ad/localization/input/wheel_speed",
    "/ad/sensors/imu/data",
)
FRONT_CAMERA_TOPIC = "/ad/sensors/camera/front/compressed"
# Raw ego-state topics consumed by ad_localization's gnss_imu backend (the
# repository's own MORAI ego -> odometry/TF converter). Only replayed when
# enable_localization is opted in, for a bag that has no recorded
# /ad/localization/odometry but does have these raw sensor topics.
LOCALIZATION_SOURCE_TOPICS = (
    "/ad/sensors/gps/fix",
    "/ad/vehicle/status",
)
_MCAP_MAGIC = b"\x89MCAP0\r\n"


def _perform(context, name):
    return LaunchConfiguration(name).perform(context)


def _parse_bool(name, value):
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise RuntimeError(f"{name} must be exactly 'true' or 'false'")


def _parse_detector_backend(value):
    normalized = str(value).strip()
    if normalized not in {"euclidean", "centerpoint", "autoware_centerpoint"}:
        raise RuntimeError(
            "detector_backend must be euclidean, centerpoint, or "
            "autoware_centerpoint"
        )
    return normalized


def _parse_number(name, value):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{name} must be numeric") from error
    if not math.isfinite(result):
        raise RuntimeError(f"{name} must be finite")
    return result


def _parse_rate(value):
    rate = _parse_number("rate", value)
    if rate <= 0.0:
        raise RuntimeError("rate must be positive")
    return rate


def _parse_startup_delay(value):
    delay = _parse_number("startup_delay_sec", value)
    if delay < 0.0:
        raise RuntimeError("startup_delay_sec must be nonnegative")
    if delay > 60.0:
        raise RuntimeError("startup_delay_sec must be at most 60 seconds")
    return delay


def _parse_crop_clearance(value):
    clearance = _parse_number("crop_clearance_m", value)
    if clearance < 0.0:
        raise RuntimeError("crop_clearance_m must be nonnegative")
    if clearance > 2.0:
        raise RuntimeError("crop_clearance_m must be at most 2 meters")
    return clearance


def _safe_metadata_path(raw_path):
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError("metadata relative_file_paths must be strings")
    relative = PurePosixPath(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"unsafe MCAP path in metadata: {raw_path!r}")
    if relative.suffix.lower() != ".mcap":
        raise RuntimeError(
            f"metadata MCAP path must end in .mcap: {raw_path!r}"
        )
    return relative


def _validate_bag_path(value):
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise RuntimeError("bag_path is required")
    try:
        requested = Path(value).expanduser()
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError("bag_path is invalid") from error
    if not requested.is_absolute():
        raise RuntimeError("bag_path must be absolute")
    try:
        if not requested.exists():
            raise RuntimeError(f"bag_path does not exist: {requested}")
        if not requested.is_dir():
            raise RuntimeError(f"bag_path must be a directory: {requested}")
        bag_path = requested.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"cannot inspect bag_path: {requested}") from error

    metadata_path = bag_path / "metadata.yaml"
    if not metadata_path.is_file():
        raise RuntimeError(f"bag_path is missing metadata.yaml: {bag_path}")
    try:
        document = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        information = document["rosbag2_bagfile_information"]
    except (KeyError, OSError, TypeError, yaml.YAMLError) as error:
        raise RuntimeError(f"invalid rosbag metadata.yaml: {error}") from error
    if not isinstance(information, dict):
        raise RuntimeError(
            "invalid rosbag metadata.yaml: bagfile information must be a map"
        )

    if information.get("storage_identifier") != "mcap":
        raise RuntimeError("metadata storage_identifier must be mcap")
    relative_paths = information.get("relative_file_paths")
    if not isinstance(relative_paths, list) or not relative_paths:
        raise RuntimeError(
            "metadata relative_file_paths must be a nonempty list"
        )

    seen = set()
    for raw_path in relative_paths:
        relative = _safe_metadata_path(raw_path)
        if relative in seen:
            raise RuntimeError(
                f"duplicate MCAP path in metadata: {raw_path!r}"
            )
        seen.add(relative)
        candidate = bag_path.joinpath(*relative.parts)
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as error:
            raise RuntimeError(f"missing MCAP file: {candidate}") from error
        except OSError as error:
            raise RuntimeError(
                f"cannot inspect MCAP file: {candidate}"
            ) from error
        try:
            resolved.relative_to(bag_path)
        except ValueError as error:
            raise RuntimeError(
                f"unsafe MCAP path outside bag directory: {raw_path!r}"
            ) from error
        if not resolved.is_file():
            raise RuntimeError(f"MCAP path must be a regular file: {resolved}")
        try:
            with resolved.open("rb") as stream:
                magic = stream.read(len(_MCAP_MAGIC))
        except OSError as error:
            raise RuntimeError(f"cannot read MCAP file: {resolved}") from error
        if magic != _MCAP_MAGIC:
            raise RuntimeError(f"invalid MCAP magic: {resolved}")
    return bag_path


def _validate_yaml_path(name, value):
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise RuntimeError(f"{name} is required")
    try:
        requested = Path(value).expanduser()
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(f"{name} path is invalid") from error
    if not requested.is_absolute():
        raise RuntimeError(f"{name} must be absolute")
    if requested.suffix.lower() not in {".yaml", ".yml"}:
        raise RuntimeError(f"{name} must be a YAML file")
    try:
        resolved = requested.resolve(strict=True)
    except FileNotFoundError as error:
        raise RuntimeError(f"{name} does not exist: {requested}") from error
    except OSError as error:
        raise RuntimeError(f"cannot inspect {name}: {requested}") from error
    if not resolved.is_file():
        raise RuntimeError(f"{name} must be a regular YAML file")
    return resolved


def _validate_composition_path(value):
    return _validate_yaml_path("composition_config", value)


def _launch_file(package, name):
    share = Path(get_package_share_directory(package))
    return PythonLaunchDescriptionSource(str(share / "launch" / name))


def _launch_setup(context):
    bag_path = _validate_bag_path(_perform(context, "bag_path"))
    composition_config = _validate_composition_path(
        _perform(context, "composition_config")
    )
    cluster_config = _validate_yaml_path(
        "cluster_config", _perform(context, "cluster_config")
    )
    ground_config = _validate_yaml_path(
        "ground_config", _perform(context, "ground_config")
    )
    qos_overrides = _validate_yaml_path(
        "qos_overrides", _perform(context, "qos_overrides")
    )
    crop_clearance = _parse_crop_clearance(
        _perform(context, "crop_clearance_m")
    )
    raw_rate = _perform(context, "rate")
    rate = _parse_rate(raw_rate)
    delay = _parse_startup_delay(_perform(context, "startup_delay_sec"))
    start_paused = _parse_bool(
        "start_paused", _perform(context, "start_paused")
    )
    loop = _parse_bool("loop", _perform(context, "loop"))
    include_front_camera = _parse_bool(
        "include_front_camera", _perform(context, "include_front_camera")
    )
    enable_localization = _parse_bool(
        "enable_localization", _perform(context, "enable_localization")
    )
    detector_backend = _parse_detector_backend(
        _perform(context, "detector_backend")
    )
    tracker_backend = _perform(context, "tracker_backend").strip()
    if tracker_backend not in {"", "autoware", "ab3dmot"}:
        raise RuntimeError(
            "tracker_backend must be empty, autoware, or ab3dmot"
        )
    dynamic_object_risk = _parse_bool(
        "dynamic_object_risk", _perform(context, "dynamic_object_risk")
    )

    # Reformat the validated number to prevent passing non-numeric shell-like
    # input through to the subprocess while retaining a readable command line.
    rate_argument = format(rate, ".15g")
    command = [
        "ros2",
        "bag",
        "play",
        "--storage",
        "mcap",
        "--clock",
        "100",
        "--rate",
        rate_argument,
        "--qos-profile-overrides-path",
        str(qos_overrides),
        "--wait-for-all-acked",
        "10000",
        "--disable-keyboard-controls",
        str(bag_path),
    ]
    if start_paused:
        command.append("--start-paused")
    if loop:
        command.append("--loop")
    replay_topics = list(SOURCE_TOPICS)
    if include_front_camera:
        replay_topics.append(FRONT_CAMERA_TOPIC)
    if enable_localization:
        replay_topics.extend(LOCALIZATION_SOURCE_TOPICS)
    command.extend(["--topics", *replay_topics])

    description = IncludeLaunchDescription(
        _launch_file("ad_description", "description.launch.py")
    )
    localization = (
        IncludeLaunchDescription(
            _launch_file("ad_localization", "localization.launch.py")
        )
        if enable_localization
        else None
    )
    perception = IncludeLaunchDescription(
        _launch_file("ad_lidar_perception", "lidar_perception.launch.py"),
        launch_arguments={
            "composition_config": str(composition_config),
            "detector_backend": detector_backend,
            "autoware_selection_config": _perform(
                context, "autoware_selection_config"
            ),
            "tracker_backend": tracker_backend,
            "dynamic_object_risk": "true" if dynamic_object_risk else "false",
            "checkpoint_path": _perform(context, "checkpoint_path"),
            "device": _perform(context, "device"),
            "openpcdet_root": _perform(context, "openpcdet_root"),
            "cluster_config": str(cluster_config),
            "ground_config": str(ground_config),
            "crop_clearance_m": format(crop_clearance, ".15g"),
            "use_sim_time": "true",
            "platform_profile": "morai",
            "deskew_enabled": "false",
            "deskew_mode": "3d",
            "self_crop_enabled": "true",
            "self_crop_input_reliable": "true",
            "patchwork_leveling_enabled": "false",
            "finite_filter_enabled": "true",
            "densifier_enabled": "false",
            # The pinned Autoware CenterPoint node accepts only exact XYZIRC.
            # The existing adapter fail-closes on an unsupported MORAI source
            # layout; historical OpenPCDet and Euclidean replay behaviour is
            # unchanged.
            "point_layout_adapter_enabled": (
                "true" if detector_backend == "autoware_centerpoint" else "false"
            ),
            "ab3dmot_defer_until_tf_ready": _perform(
                context, "ab3dmot_defer_until_tf_ready"
            ),
            "ab3dmot_max_tf_wait_ms": _perform(context, "ab3dmot_max_tf_wait_ms"),
            "ab3dmot_max_pending_detections": _perform(
                context, "ab3dmot_max_pending_detections"
            ),
            "ab3dmot_state_estimator": LaunchConfiguration(
                "ab3dmot_state_estimator", default="linear_kf"
            ).perform(context),
            "prediction_yaw_rate_source": LaunchConfiguration(
                "prediction_yaw_rate_source", default="tracker"
            ).perform(context),
            "use_predicted_future_sweep": LaunchConfiguration(
                "use_predicted_future_sweep", default="false"
            ).perform(context),
            "future_sweep_horizon_s": LaunchConfiguration(
                "future_sweep_horizon_s", default="3.0"
            ).perform(context),
            "ab3dmot_kalmannet_checkpoint": LaunchConfiguration(
                "ab3dmot_kalmannet_checkpoint", default=""
            ).perform(context),
            "ab3dmot_kalmannet_device": LaunchConfiguration(
                "ab3dmot_kalmannet_device", default="cpu"
            ).perform(context),
            "ab3dmot_root": LaunchConfiguration(
                "ab3dmot_root", default=""
            ).perform(context),
            # Keep replay input ownership separate from visualization.  These
            # must be explicit because an enclosing launch can otherwise leak
            # identically named start_* configurations into this include.
            "start_visualization": "false",
            "start_rviz": "false",
            # CenterPoint's validated contract is cropped-only; the classical
            # Euclidean detector consumes the post-ground stream.
            "start_ground_segmentation": (
                "false" if detector_backend == "centerpoint" else "true"
            ),
        }.items(),
    )
    player = ExecuteProcess(
        cmd=command,
        output="screen",
        emulate_tty=True,
    )

    group_actions = [
        SetParameter(name="use_sim_time", value=True),
        description,
        perception,
        TimerAction(period=delay, actions=[player]),
    ]

    actions = [
        GroupAction(
            scoped=True,
            actions=group_actions,
        )
    ]
    if localization is not None:
        # A separate, UNSCOPED group: localization.launch.py uses
        # RegisterEventHandler/EmitEvent for its lifecycle autostart, whose
        # IfCondition is evaluated asynchronously when the event fires -
        # after a *scoped* GroupAction has already been popped from the
        # launch context, which makes that later lookup fail with
        # "launch configuration 'autostart' does not exist". This group is
        # the last action in this launch's action list, so an unscoped
        # SetParameter here cannot leak into any preceding node.
        actions.append(
            GroupAction(
                scoped=False,
                actions=[
                    SetParameter(name="use_sim_time", value=True),
                    localization,
                ],
            )
        )
    return actions


def generate_launch_description():
    perception_share = Path(
        get_package_share_directory("ad_lidar_perception")
    )
    default_composition = (
        perception_share
        / "config"
        / "lidar_perception_morai_classical.yaml"
    )
    default_cluster = (
        perception_share
        / "config"
        / "clustering"
        / "adaptive_euclidean_cluster.yaml"
    )
    default_ground = (
        perception_share
        / "config"
        / "preprocessing"
        / "ground_segmentation.yaml"
    )
    default_qos_overrides = (
        perception_share / "config" / "replay_qos_overrides.yaml"
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "bag_path",
                description=(
                    "Required absolute path to an extracted ROS 2 MCAP bag "
                    "directory containing metadata.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "rate",
                default_value="0.5",
                description="Positive finite rosbag playback rate",
            ),
            DeclareLaunchArgument(
                "startup_delay_sec",
                default_value="2.0",
                description=(
                    "Finite 0..60 second delay before starting rosbag play"
                ),
            ),
            DeclareLaunchArgument(
                "start_paused",
                default_value="false",
                description="Start rosbag paused; must be true or false",
            ),
            DeclareLaunchArgument(
                "loop",
                default_value="true",
                description="Replay the bag repeatedly; must be true or false",
            ),
            DeclareLaunchArgument(
                "include_front_camera",
                default_value="false",
                description=(
                    "Also replay the front compressed-camera topic; opt-in "
                    "so the LiDAR-only replay contract stays unchanged"
                ),
            ),
            DeclareLaunchArgument(
                "enable_localization",
                default_value="false",
                description=(
                    "Opt-in: also replay /ad/sensors/gps/fix + "
                    "/ad/vehicle/status and start ad_localization's "
                    "gnss_imu backend (existing MORAI ego -> "
                    "/ad/localization/odometry + odom->base_link TF "
                    "converter), scoped to this replay's sim time. For a "
                    "bag with raw ego sensors but no recorded odometry."
                ),
            ),
            DeclareLaunchArgument(
                "detector_backend",
                default_value="euclidean",
                description=(
                    "euclidean (default), historical OpenPCDet centerpoint, "
                    "or opt-in pinned Autoware autoware_centerpoint"
                ),
            ),
            DeclareLaunchArgument(
                "autoware_selection_config",
                default_value=str(
                    perception_share
                    / "config"
                    / "experiments"
                    / "autoware_centerpoint_competition_candidate_v1.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "tracker_backend",
                default_value="",
                description=(
                    "Blank preserves the composition config tracker; "
                    "'autoware' or opt-in 'ab3dmot' overrides it"
                ),
            ),
            DeclareLaunchArgument(
                "dynamic_object_risk",
                default_value="false",
                description=(
                    "Opt-in: also start the Dynamic Object Risk node "
                    "(observational, no behaviour change)"
                ),
            ),
            DeclareLaunchArgument(
                "ab3dmot_defer_until_tf_ready",
                default_value="true",
                description=(
                    "tracker_backend:=ab3dmot only: AB3DMOT Exact-Stamp TF "
                    "Deferred Processing v1 pass-through -- see "
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
                "ab3dmot_state_estimator", default_value="linear_kf"
            ),
            DeclareLaunchArgument(
                "prediction_yaw_rate_source", default_value="tracker"
            ),
            DeclareLaunchArgument(
                "use_predicted_future_sweep", default_value="false"
            ),
            DeclareLaunchArgument(
                "future_sweep_horizon_s", default_value="3.0"
            ),
            DeclareLaunchArgument(
                "ab3dmot_kalmannet_checkpoint", default_value=""
            ),
            DeclareLaunchArgument(
                "ab3dmot_kalmannet_device", default_value="cpu"
            ),
            DeclareLaunchArgument("ab3dmot_root", default_value=""),
            DeclareLaunchArgument("checkpoint_path", default_value=""),
            DeclareLaunchArgument("device", default_value="cuda:0"),
            DeclareLaunchArgument("openpcdet_root", default_value=""),
            DeclareLaunchArgument(
                "composition_config",
                default_value=str(default_composition),
                description=(
                    "Absolute classical LiDAR composition YAML to tune"
                ),
            ),
            DeclareLaunchArgument(
                "cluster_config",
                default_value=str(default_cluster),
                description="Absolute classical clustering parameter YAML",
            ),
            DeclareLaunchArgument(
                "ground_config",
                default_value=str(default_ground),
                description="Absolute ground segmentation parameter YAML",
            ),
            DeclareLaunchArgument(
                "qos_overrides",
                default_value=str(default_qos_overrides),
                description="Absolute rosbag player QoS override YAML",
            ),
            DeclareLaunchArgument(
                "crop_clearance_m",
                default_value="0.20",
                description="Finite self-crop clearance from 0 to 2 meters",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
