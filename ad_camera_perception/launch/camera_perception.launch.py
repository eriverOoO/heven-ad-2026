from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share_directory = Path(
        get_package_share_directory("ad_camera_perception")
    )
    traffic_config = str(
        package_share_directory
        / "config"
        / "traffic_light.yaml"
    )
    dynamic_obstacle_config = str(
        package_share_directory
        / "config"
        / "dynamic_obstacle.yaml"
    )

    arguments = [
        DeclareLaunchArgument(
            "traffic_params_file", default_value=traffic_config
        ),
        DeclareLaunchArgument(
            "traffic_light_image_topic",
            default_value="/ad/sensors/camera/traffic_light/compressed",
        ),
        DeclareLaunchArgument(
            "traffic_light_detections_topic",
            default_value="/vision/traffic_light/detections",
        ),
        DeclareLaunchArgument(
            "traffic_light_status_topic",
            default_value="/vision/traffic_light/status",
        ),
        DeclareLaunchArgument(
            "traffic_light_model_path",
            default_value="models/yolov7_best.pt",
        ),
        DeclareLaunchArgument(
            "traffic_light_yolov7_repository_path", default_value=""
        ),
        DeclareLaunchArgument(
            "traffic_light_device", default_value="cuda:0"
        ),
        DeclareLaunchArgument(
            "traffic_light_enable_visualizer", default_value="true"
        ),
        DeclareLaunchArgument(
            "traffic_light_show_window", default_value="true"
        ),
        DeclareLaunchArgument(
            "dynamic_obstacle_params_file",
            default_value=dynamic_obstacle_config,
        ),
        DeclareLaunchArgument(
            "dynamic_obstacle_image_topic",
            default_value="/ad/sensors/camera/front/compressed",
        ),
        DeclareLaunchArgument(
            "dynamic_obstacle_image_transport", default_value="compressed"
        ),
        DeclareLaunchArgument(
            "dynamic_obstacle_detections_topic",
            default_value="/vision/dynamic_obstacle/detections",
        ),
        DeclareLaunchArgument(
            "dynamic_obstacle_model_path", default_value="yolo26s.pt"
        ),
        DeclareLaunchArgument("dynamic_obstacle_device", default_value="auto"),
    ]
    traffic_pipeline = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(package_share_directory / "launch" / "traffic_signal.launch.py")
        ),
        launch_arguments={
            "params_file": LaunchConfiguration("traffic_params_file"),
            "image_topic": LaunchConfiguration("traffic_light_image_topic"),
            "detections_topic": LaunchConfiguration(
                "traffic_light_detections_topic"
            ),
            "status_topic": LaunchConfiguration("traffic_light_status_topic"),
            "model_path": LaunchConfiguration("traffic_light_model_path"),
            "yolov7_repository_path": LaunchConfiguration(
                "traffic_light_yolov7_repository_path"
            ),
            "device": LaunchConfiguration("traffic_light_device"),
            "enable_visualizer": LaunchConfiguration(
                "traffic_light_enable_visualizer"
            ),
            "show_window": LaunchConfiguration("traffic_light_show_window"),
        }.items(),
    )
    dynamic_obstacle_detector = Node(
        package="ad_camera_perception",
        executable="ad_dynamic_obstacle_detector_node",
        name="dynamic_obstacle_detector",
        output="screen",
        parameters=[
            LaunchConfiguration("dynamic_obstacle_params_file"),
            {
                "image_topic": LaunchConfiguration(
                    "dynamic_obstacle_image_topic"
                ),
                "image_transport": LaunchConfiguration(
                    "dynamic_obstacle_image_transport"
                ),
                "detections_topic": LaunchConfiguration(
                    "dynamic_obstacle_detections_topic"
                ),
                "model_path": LaunchConfiguration("dynamic_obstacle_model_path"),
                "device": LaunchConfiguration("dynamic_obstacle_device"),
            },
        ],
    )
    return LaunchDescription(
        arguments + [traffic_pipeline, dynamic_obstacle_detector]
    )
