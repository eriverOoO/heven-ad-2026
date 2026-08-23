"""Preset-selectable MORAI camera + LiDAR tracking replay.

The production Euclidean/Autoware path remains unchanged. Exactly one of ten
explicit AB3DMOT comparison presets is added as an opt-in experimental arm.
The identity odom anchor is replay-only: camera bags without localization TF
cannot support global-frame accuracy claims.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter

from ad_lidar_perception.tracking_replay_presets import (
    PresetConfigError,
    select_tracking_replay_preset,
)


def _launch_file(name: str) -> PythonLaunchDescriptionSource:
    package_share = Path(get_package_share_directory("ad_lidar_perception"))
    return PythonLaunchDescriptionSource(str(package_share / "launch" / name))


def _perform(context, name: str) -> str:
    return LaunchConfiguration(name).perform(context)


def _required_file(name: str, value: str) -> str:
    requested = Path(value).expanduser()
    if not value.strip() or not requested.is_absolute():
        raise RuntimeError(f"{name} must be an absolute file path")
    try:
        resolved = requested.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise RuntimeError(f"{name} does not exist: {requested}") from exc
    if not resolved.is_file():
        raise RuntimeError(f"{name} must be a regular file: {resolved}")
    return str(resolved)


def _required_directory(name: str, value: str) -> str:
    requested = Path(value).expanduser()
    if not value.strip() or not requested.is_absolute():
        raise RuntimeError(f"{name} must be an absolute directory path")
    try:
        resolved = requested.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise RuntimeError(f"{name} does not exist: {requested}") from exc
    if not resolved.is_dir():
        raise RuntimeError(f"{name} must be a directory: {resolved}")
    return str(resolved)


def _launch_setup(context):
    preset_config = _required_file(
        "preset_config", _perform(context, "preset_config")
    )
    try:
        preset = select_tracking_replay_preset(
            preset_config, _perform(context, "mode")
        )
    except PresetConfigError as exc:
        raise RuntimeError(str(exc)) from exc

    centerpoint_checkpoint = ""
    openpcdet_root = ""
    centerpoint_device = _perform(context, "centerpoint_device")
    if preset.needs_centerpoint:
        centerpoint_checkpoint = _required_file(
            "centerpoint_checkpoint",
            _perform(context, "centerpoint_checkpoint"),
        )
        openpcdet_root = _required_directory(
            "openpcdet_root", _perform(context, "openpcdet_root")
        )
        if not centerpoint_device.startswith("cuda"):
            raise RuntimeError("CenterPoint presets require a CUDA device")

    kalmannet_checkpoint = ""
    kalmannet_device = _perform(context, "kalmannet_device")
    if preset.needs_kalmannet:
        kalmannet_checkpoint = _required_file(
            "kalmannet_checkpoint", _perform(context, "kalmannet_checkpoint")
        )
        if kalmannet_device != "cpu" and not kalmannet_device.startswith("cuda"):
            raise RuntimeError("kalmannet_device must be cpu or a CUDA device")

    package_share = Path(get_package_share_directory("ad_lidar_perception"))
    rviz_config = package_share / "rviz" / "heven_camera_tracking.rviz"
    replay = IncludeLaunchDescription(
        _launch_file("lidar_bag_replay.launch.py"),
        launch_arguments={
            "bag_path": LaunchConfiguration("bag_path"),
            "rate": LaunchConfiguration("rate"),
            "startup_delay_sec": LaunchConfiguration("startup_delay_sec"),
            "start_paused": LaunchConfiguration("start_paused"),
            "include_front_camera": "true",
            "detector_backend": preset.detector,
            "checkpoint_path": centerpoint_checkpoint,
            "device": centerpoint_device,
            "openpcdet_root": openpcdet_root,
        }.items(),
    )
    experimental_tracker = IncludeLaunchDescription(
        _launch_file("ab3dmot_tracker.launch.py"),
        launch_arguments={
            "enabled": "true",
            "matcher": preset.matcher,
            "association_metric": preset.association,
            "euclidean_gate_m": format(preset.euclidean_gate_m, ".15g"),
            "mahalanobis_gate": format(preset.mahalanobis_gate, ".15g"),
            "mahalanobis_max_distance_m": format(
                preset.mahalanobis_max_distance_m, ".15g"
            ),
            "state_estimator": preset.estimator,
            "yaw_measurement_mode": preset.yaw_measurement_mode,
            "kalmannet_checkpoint": kalmannet_checkpoint,
            "kalmannet_device": kalmannet_device,
        }.items(),
    )
    visualization = IncludeLaunchDescription(
        _launch_file("perception_visualization.launch.py"),
        launch_arguments={
            "start_rviz": "true",
            "use_sim_time": "true",
            "rviz_config": str(rviz_config),
        }.items(),
    )
    replay_odom_anchor = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="camera_bag_replay_odom_anchor",
        output="screen",
        arguments=[
            "--x", "0", "--y", "0", "--z", "0",
            "--roll", "0", "--pitch", "0", "--yaw", "0",
            "--frame-id", "odom", "--child-frame-id", "base_link",
        ],
    )
    return [
        LogInfo(msg=f"tracking replay mode {preset.mode}: {preset.label}"),
        SetParameter(name="use_sim_time", value=True),
        replay_odom_anchor,
        experimental_tracker,
        visualization,
        replay,
    ]


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("ad_lidar_perception"))
    default_presets = (
        package_share / "config" / "tracking" / "camera_replay_presets.yaml"
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "bag_path",
                description="Absolute path to the extracted MCAP bag folder",
            ),
            DeclareLaunchArgument("mode", default_value="3"),
            DeclareLaunchArgument(
                "preset_config", default_value=str(default_presets)
            ),
            DeclareLaunchArgument("rate", default_value="0.5"),
            DeclareLaunchArgument("startup_delay_sec", default_value="4.0"),
            DeclareLaunchArgument("start_paused", default_value="false"),
            DeclareLaunchArgument("centerpoint_checkpoint", default_value=""),
            DeclareLaunchArgument("openpcdet_root", default_value=""),
            DeclareLaunchArgument("centerpoint_device", default_value="cuda:0"),
            DeclareLaunchArgument("kalmannet_checkpoint", default_value=""),
            DeclareLaunchArgument("kalmannet_device", default_value="cpu"),
            OpaqueFunction(function=_launch_setup),
        ]
    )
