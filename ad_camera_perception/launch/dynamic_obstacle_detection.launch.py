"""Launch dynamic-obstacle YOLO detection and bbox visualization."""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    """Build a configurable detector plus visualizer launch graph."""
    default_config = PathJoinSubstitution(
        [FindPackageShare("ad_camera_perception"), "config", "dynamic_obstacle.yaml"]
    )
    cache_root = Path(
        os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    )
    model_directory = Path(
        os.environ.get(
            "AD_CAMERA_PERCEPTION_MODEL_DIR",
            cache_root / "ad_camera_perception" / "models",
        )
    )
    default_model = str(model_directory / "yolo26s.pt")

    arguments = [
        DeclareLaunchArgument("params_file", default_value=default_config),
        DeclareLaunchArgument(
            "image_topic",
            default_value="/ad/sensors/camera/front/compressed",
        ),
        DeclareLaunchArgument("image_transport", default_value="compressed"),
        DeclareLaunchArgument("model_path", default_value=default_model),
        DeclareLaunchArgument("device", default_value="auto"),
        DeclareLaunchArgument(
            "detections_topic",
            default_value="/vision/dynamic_obstacle/detections",
        ),
        DeclareLaunchArgument(
            "visualization_image_topic",
            default_value="/ad/viz/perception/camera/dynamic_obstacle",
        ),
        DeclareLaunchArgument("show_window", default_value="false"),
        DeclareLaunchArgument(
            "window_name", default_value="dynamic_obstacle_debug"
        ),
    ]

    detector = Node(
        package="ad_camera_perception",
        executable="ad_dynamic_obstacle_detector_node",
        name="dynamic_obstacle_detector",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            {
                "image_topic": LaunchConfiguration("image_topic"),
                "image_transport": LaunchConfiguration("image_transport"),
                "model_path": LaunchConfiguration("model_path"),
                "device": LaunchConfiguration("device"),
                "detections_topic": LaunchConfiguration("detections_topic"),
            },
        ],
    )
    visualizer = Node(
        package="ad_camera_perception",
        executable="ad_vision_visualizer_node",
        name="dynamic_obstacle_visualizer",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            {
                "image_topic": LaunchConfiguration("image_topic"),
                "image_transport": LaunchConfiguration("image_transport"),
                "detections_topic": LaunchConfiguration("detections_topic"),
                "output_image_topic": LaunchConfiguration(
                    "visualization_image_topic"
                ),
                "show_window": ParameterValue(
                    LaunchConfiguration("show_window"), value_type=bool
                ),
                "window_name": LaunchConfiguration("window_name"),
            },
        ],
    )

    return LaunchDescription(arguments + [detector, visualizer])
