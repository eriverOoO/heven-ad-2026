from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument("detector_backend", default_value="euclidean"),
        DeclareLaunchArgument("checkpoint_path", default_value=""),
        DeclareLaunchArgument("score_threshold", default_value="0.1"),
        DeclareLaunchArgument("max_detections", default_value="500"),
        DeclareLaunchArgument("device", default_value="cuda:0"),
        DeclareLaunchArgument("point_cloud_range", default_value="[-4.0,-25.0,-3.0,100.0,25.0,5.0]"),
        DeclareLaunchArgument("enabled", default_value="true"),
        DeclareLaunchArgument("mock_mode", default_value="false"),
        DeclareLaunchArgument("openpcdet_root", default_value=""),
        DeclareLaunchArgument("input_topic", default_value="/ad/perception/lidar/cropped"),
    ]
    node = Node(
        package="ad_lidar_perception",
        executable="ad_centerpoint_detector",
        name="ad_centerpoint_detector",
        output="screen",
        parameters=[{
            "detector_backend": LaunchConfiguration("detector_backend"),
            "checkpoint_path": LaunchConfiguration("checkpoint_path"),
            "score_threshold": ParameterValue(LaunchConfiguration("score_threshold"), value_type=float),
            "max_detections": ParameterValue(LaunchConfiguration("max_detections"), value_type=int),
            "device": LaunchConfiguration("device"),
            "point_cloud_range": ParameterValue(LaunchConfiguration("point_cloud_range")),
            "enabled": ParameterValue(LaunchConfiguration("enabled"), value_type=bool),
            "mock_mode": ParameterValue(LaunchConfiguration("mock_mode"), value_type=bool),
            "openpcdet_root": LaunchConfiguration("openpcdet_root"),
            "input_topic": LaunchConfiguration("input_topic"),
            "output_topic": "/ad/perception/objects/detected",
        }],
    )
    return LaunchDescription([*arguments, node])
