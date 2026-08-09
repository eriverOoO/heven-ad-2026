"""Launch the mm-2025 ROS 2 detector, evaluator, and OpenCV visualizer."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = Path(get_package_share_directory("ad_camera_perception"))
    default_config = str(package_share / "config" / "traffic_light.yaml")

    params_file = LaunchConfiguration("params_file")
    image_topic = LaunchConfiguration("image_topic")
    detections_topic = LaunchConfiguration("detections_topic")
    status_topic = LaunchConfiguration("status_topic")
    model_path = LaunchConfiguration("model_path")
    repository_path = LaunchConfiguration("yolov7_repository_path")
    device = LaunchConfiguration("device")
    show_window = LaunchConfiguration("show_window")
    enable_visualizer = LaunchConfiguration("enable_visualizer")

    arguments = [
        DeclareLaunchArgument("params_file", default_value=default_config),
        DeclareLaunchArgument(
            "image_topic",
            default_value="/ad/sensors/camera/traffic_light/compressed",
        ),
        DeclareLaunchArgument(
            "detections_topic",
            default_value="/vision/traffic_light/detections",
        ),
        DeclareLaunchArgument(
            "status_topic", default_value="/vision/traffic_light/status"
        ),
        DeclareLaunchArgument(
            "model_path", default_value="models/yolov7_best.pt"
        ),
        DeclareLaunchArgument("yolov7_repository_path", default_value=""),
        DeclareLaunchArgument("device", default_value="cuda:0"),
        DeclareLaunchArgument("enable_visualizer", default_value="true"),
        DeclareLaunchArgument("show_window", default_value="true"),
    ]
    detector = Node(
        package="ad_camera_perception",
        executable="ad_traffic_light_detector_node",
        name="traffic_light_detector",
        output="screen",
        parameters=[
            params_file,
            {
                "image_topic": image_topic,
                "detections_topic": detections_topic,
                "model_path": model_path,
                "yolov7_repository_path": repository_path,
                "device": device,
            },
        ],
    )
    evaluator = Node(
        package="ad_camera_perception",
        executable="ad_traffic_light_evaluator_node",
        name="traffic_light_evaluator",
        output="screen",
        parameters=[
            params_file,
            {
                "detections_topic": detections_topic,
                "status_topic": status_topic,
            },
        ],
    )
    visualizer = Node(
        package="ad_camera_perception",
        executable="ad_traffic_light_visualizer_node",
        name="traffic_light_visualizer",
        output="screen",
        condition=IfCondition(enable_visualizer),
        parameters=[
            params_file,
            {
                "image_topic": image_topic,
                "detections_topic": detections_topic,
                "status_topic": status_topic,
                "show_window": ParameterValue(show_window, value_type=bool),
            },
        ],
    )
    return LaunchDescription(arguments + [detector, evaluator, visualizer])
