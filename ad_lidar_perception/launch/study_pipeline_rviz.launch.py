"""Opt-in, same-bag A/B/C/D perception study demo.

The selector changes only detector and/or AB3DMOT state estimator. Association
(BEV Euclidean 3 m + Hungarian), lifecycle, prediction and occupancy remain on
the existing canonical path. Production defaults are not changed.
"""

from hashlib import sha256
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node, SetParameter
import yaml


CENTERPOINT_SHA256 = "466c8181a377682e032bb32579c8ddb65807b5feebd8625a750b1d5538ddbc95"
KALMANNET_SHA256 = "956604975e5204b2c584c3e8fe16a7ba9346097980566ecbbb22c2001fbf7d48"
VARIANTS = {
    "training_free": ("PIPELINE A — Euclidean + Hungarian + Linear KF", "euclidean", "linear_kf"),
    "centerpoint_kf": ("PIPELINE B — CenterPoint + Hungarian + Linear KF", "centerpoint", "linear_kf"),
    "euclidean_knet": ("PIPELINE C — Euclidean + Hungarian + KalmanNet", "euclidean", "kalmannet"),
    "centerpoint_knet": ("PIPELINE D — CenterPoint + Hungarian + KalmanNet", "centerpoint", "kalmannet"),
}


def _launch_file(package, name):
    share = Path(get_package_share_directory(package))
    return PythonLaunchDescriptionSource(str(share / "launch" / name))


def _perform(context, name):
    return LaunchConfiguration(name).perform(context)


def _boolean(name, value):
    normalized = str(value).strip().lower()
    if normalized not in {"true", "false"}:
        raise RuntimeError(f"{name} must be exactly 'true' or 'false'")
    return normalized == "true"


def _artifact(name, value, expected_hash, kind):
    requested = Path(value).expanduser()
    if not value.strip() or not requested.is_absolute() or not requested.is_file():
        raise RuntimeError(f"{name} must be an absolute existing file")
    resolved = requested.resolve(strict=True)
    actual = sha256(resolved.read_bytes()).hexdigest()
    if actual != expected_hash:
        raise RuntimeError(
            f"{name} SHA-256 mismatch: {actual}; expected frozen {kind} artifact {expected_hash}"
        )
    return str(resolved)


def _launch_setup(context):
    variant_name = _perform(context, "pipeline_variant").strip()
    try:
        label, detector, estimator = VARIANTS[variant_name]
    except KeyError as exc:
        raise RuntimeError(
            "pipeline_variant must be training_free, centerpoint_kf, "
            "euclidean_knet, or centerpoint_knet"
        ) from exc

    needs_centerpoint = detector == "centerpoint"
    needs_kalmannet = estimator == "kalmannet"
    centerpoint_checkpoint = ""
    openpcdet_root = ""
    kalmannet_checkpoint = ""
    if needs_centerpoint:
        centerpoint_checkpoint = _artifact(
            "centerpoint_checkpoint", _perform(context, "centerpoint_checkpoint"),
            CENTERPOINT_SHA256, "T14 CenterPoint",
        )
        root = Path(_perform(context, "openpcdet_root")).expanduser()
        if not root.is_absolute() or not root.is_dir():
            raise RuntimeError("openpcdet_root must be an absolute existing directory")
        openpcdet_root = str(root.resolve(strict=True))
    if needs_kalmannet:
        kalmannet_checkpoint = _artifact(
            "kalmannet_checkpoint", _perform(context, "kalmannet_checkpoint"),
            KALMANNET_SHA256, "DENSE-KALMANNET-v2",
        )

    start_rviz = _boolean("start_rviz", _perform(context, "start_rviz"))
    enable_drivable_mask = _boolean(
        "enable_drivable_mask", _perform(context, "enable_drivable_mask")
    )
    share = Path(get_package_share_directory("ad_lidar_perception"))
    replay = IncludeLaunchDescription(
        _launch_file("ad_lidar_perception", "lidar_bag_replay.launch.py"),
        launch_arguments={
            "bag_path": _perform(context, "bag_path"),
            "rate": _perform(context, "rate"),
            "loop": "false",
            "start_paused": _perform(context, "start_paused"),
            "include_front_camera": "true",
            "enable_localization": "true",
            "composition_config": str(share / "config" / "lidar_perception.yaml"),
            "detector_backend": detector,
            "tracker_backend": "ab3dmot",
            "checkpoint_path": centerpoint_checkpoint,
            "device": _perform(context, "centerpoint_device"),
            "openpcdet_root": openpcdet_root,
            "ab3dmot_state_estimator": estimator,
            "ab3dmot_kalmannet_checkpoint": kalmannet_checkpoint,
            "ab3dmot_kalmannet_device": _perform(context, "kalmannet_device"),
            "ab3dmot_root": _perform(context, "ab3dmot_root"),
        }.items(),
    )
    visualization = IncludeLaunchDescription(
        _launch_file("ad_lidar_perception", "perception_visualization.launch.py"),
        launch_arguments={
            "start_rviz": "true" if start_rviz else "false",
            "use_sim_time": "true",
            "rviz_config": str(share / "rviz" / "training_free_perception_camera.rviz"),
        }.items(),
    )
    actions = [
        LogInfo(msg=f"STUDY DEMO: {label}"),
        SetParameter(name="use_sim_time", value=True),
        Node(
            package="ad_lidar_perception",
            executable="ad_study_pipeline_label",
            name="ad_study_pipeline_label",
            output="screen",
            parameters=[{"label": label, "frame_id": "base_link"}],
        ),
        visualization,
        replay,
    ]
    if enable_drivable_mask:
        data_dir = Path(_perform(context, "data_dir")).expanduser()
        path_file = data_dir / "path" / "2026_molit_comp_global_path.txt"
        corridor_file = data_dir / "map" / "route_corridor.json"
        if not path_file.is_file() or not corridor_file.is_file():
            raise RuntimeError("enable_drivable_mask requires the committed AD_DATA_DIR path and corridor files")
        planner_share = Path(get_package_share_directory("ad_planner"))
        config_file = planner_share / "config" / "road_corridor_mask.yaml"
        parameters = yaml.safe_load(config_file.read_text(encoding="utf-8"))[
            "ad_road_corridor_mask"
        ]["ros__parameters"]
        parameters = dict(parameters)
        parameters.update({
            "data_dir": str(data_dir.resolve(strict=True)),
            "route_corridor_file": "map/route_corridor.json",
            "route_corridor.expected_global_path_sha256": sha256(path_file.read_bytes()).hexdigest(),
        })
        actions.append(GroupAction(scoped=True, actions=[
            SetParameter(name="use_sim_time", value=True),
            Node(
                package="ad_planner",
                executable="ad_road_corridor_mask_node",
                name="ad_road_corridor_mask",
                output="screen",
                parameters=[parameters],
            ),
        ]))
    metrics_output = _perform(context, "metrics_output").strip()
    if metrics_output:
        actions.append(GroupAction(scoped=True, actions=[
            SetParameter(name="use_sim_time", value=True),
            Node(
                package="ad_lidar_perception",
                executable="ad_study_runtime_recorder",
                name="ad_study_runtime_recorder",
                output="screen",
                parameters=[{
                    "variant": variant_name,
                    "output_path": metrics_output,
                    "duration_sec": float(_perform(context, "metrics_duration_sec")),
                    "source_start_sec": float(_perform(context, "metrics_source_start_sec")),
                }],
            ),
        ]))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("pipeline_variant", default_value="training_free"),
        DeclareLaunchArgument("bag_path"),
        DeclareLaunchArgument("rate", default_value="0.5"),
        DeclareLaunchArgument("start_paused", default_value="false"),
        DeclareLaunchArgument("start_rviz", default_value="true"),
        DeclareLaunchArgument("enable_drivable_mask", default_value="true"),
        DeclareLaunchArgument("data_dir", default_value=EnvironmentVariable("AD_DATA_DIR", default_value="")),
        DeclareLaunchArgument("centerpoint_checkpoint", default_value=""),
        DeclareLaunchArgument("openpcdet_root", default_value=""),
        DeclareLaunchArgument("centerpoint_device", default_value="cuda:0"),
        DeclareLaunchArgument("kalmannet_checkpoint", default_value=""),
        DeclareLaunchArgument("kalmannet_device", default_value="cpu"),
        DeclareLaunchArgument("ab3dmot_root", default_value=""),
        DeclareLaunchArgument("metrics_output", default_value=""),
        DeclareLaunchArgument("metrics_duration_sec", default_value="60.0"),
        DeclareLaunchArgument("metrics_source_start_sec", default_value="0.0"),
        OpaqueFunction(function=_launch_setup),
    ])
