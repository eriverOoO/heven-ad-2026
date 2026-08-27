"""Opt-in Extension-2 camera/LiDAR late-fusion research launch.

With defaults this launch creates no nodes.  It never modifies or remaps the
production detector/tracker outputs.  The Stage-3 semantic tracker remains an
offline diagnostic because the existing DetectedObjects message cannot carry
the validated camera-LiDAR IoU and evidence timestamp without loss.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


EXPECTED_YOLO_SHA256 = (
    "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"
)


def _perform(context, name: str) -> str:
    return LaunchConfiguration(name).perform(context)


def _parse_bool(name: str, value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise RuntimeError(f"{name} must be exactly 'true' or 'false'")


def _required_file(name: str, value: str) -> Path:
    requested = Path(value).expanduser()
    if not value.strip() or not requested.is_absolute():
        raise RuntimeError(f"{name} must be an absolute file path")
    try:
        resolved = requested.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise RuntimeError(f"{name} does not exist: {requested}") from exc
    if not resolved.is_file():
        raise RuntimeError(f"{name} must be a regular file: {resolved}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _launch_setup(context):
    fusion_enabled = _parse_bool(
        "camera_lidar_fusion_enabled", _perform(context, "camera_lidar_fusion_enabled")
    )
    semantic_enabled = _parse_bool(
        "semantic_association_enabled", _perform(context, "semantic_association_enabled")
    )
    start_camera_detector = _parse_bool(
        "start_camera_detector", _perform(context, "start_camera_detector")
    )
    if semantic_enabled:
        raise RuntimeError(
            "semantic_association_enabled has no lossless ROS evidence transport: "
            "use the documented offline Stage-3 diagnostic replay"
        )
    if not fusion_enabled:
        if start_camera_detector:
            raise RuntimeError(
                "start_camera_detector requires camera_lidar_fusion_enabled:=true"
            )
        return [LogInfo(msg="Extension-2 Mode 0: inert baseline (no research nodes)")]

    tf_static_json = _required_file("tf_static_json", _perform(context, "tf_static_json"))
    fusion_params = _required_file("fusion_params", _perform(context, "fusion_params"))
    actions = []
    if start_camera_detector:
        camera_params = _required_file("camera_params", _perform(context, "camera_params"))
        weight_path = _required_file("yolo_weight", _perform(context, "yolo_weight"))
        actual_sha = _sha256(weight_path)
        if actual_sha != EXPECTED_YOLO_SHA256:
            raise RuntimeError(
                f"yolo_weight SHA256 mismatch: expected {EXPECTED_YOLO_SHA256}, "
                f"got {actual_sha}"
            )
        actions.append(
            Node(
                package="ad_camera_perception",
                executable="ad_dynamic_obstacle_detector_node",
                name="dynamic_obstacle_detector",
                output="screen",
                parameters=[
                    str(camera_params),
                    {
                        "model_path": str(weight_path),
                        "image_topic": _perform(context, "camera_image_topic"),
                        "detections_topic": _perform(context, "camera_detections_topic"),
                        "device": _perform(context, "camera_device"),
                        "use_sim_time": _parse_bool(
                            "use_sim_time", _perform(context, "use_sim_time")
                        ),
                    },
                ],
            )
        )

    actions.append(
        Node(
            package="ad_lidar_perception",
            executable="ad_camera_lidar_fusion",
            name="camera_lidar_fusion",
            output="screen",
            parameters=[
                str(fusion_params),
                {
                    "enabled": True,
                    "tf_static_json": str(tf_static_json),
                    "camera_detections_topic": _perform(
                        context, "camera_detections_topic"
                    ),
                    "lidar_detections_topic": _perform(
                        context, "lidar_detections_topic"
                    ),
                    "fused_topic": _perform(context, "fused_topic"),
                    "use_sim_time": _parse_bool(
                        "use_sim_time", _perform(context, "use_sim_time")
                    ),
                },
            ],
        )
    )
    return [LogInfo(msg="Extension-2 Mode 1: opt-in IoU late fusion"), *actions]


def generate_launch_description() -> LaunchDescription:
    lidar_share = Path(get_package_share_directory("ad_lidar_perception"))
    camera_share = Path(get_package_share_directory("ad_camera_perception"))
    return LaunchDescription(
        [
            DeclareLaunchArgument("camera_lidar_fusion_enabled", default_value="false"),
            DeclareLaunchArgument("semantic_association_enabled", default_value="false"),
            DeclareLaunchArgument("start_camera_detector", default_value="false"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument(
                "fusion_params",
                default_value=str(lidar_share / "config/camera_lidar_fusion_experiment.yaml"),
            ),
            DeclareLaunchArgument(
                "camera_params",
                default_value=str(camera_share / "config/dynamic_obstacle.yaml"),
            ),
            DeclareLaunchArgument("tf_static_json", default_value=""),
            DeclareLaunchArgument("yolo_weight", default_value=""),
            DeclareLaunchArgument("camera_device", default_value="auto"),
            DeclareLaunchArgument(
                "camera_image_topic",
                default_value="/ad/sensors/camera/front/compressed",
            ),
            DeclareLaunchArgument(
                "camera_detections_topic",
                default_value="/vision/dynamic_obstacle/detections",
            ),
            DeclareLaunchArgument(
                "lidar_detections_topic",
                default_value="/ad/perception/objects/detected",
            ),
            DeclareLaunchArgument(
                "fused_topic",
                default_value="/experiment/perception/objects/camera_lidar_fused",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
