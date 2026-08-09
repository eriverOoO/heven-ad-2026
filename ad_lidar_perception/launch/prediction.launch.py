from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [
            FindPackageShare("ad_lidar_perception"),
            "config",
            "tracking",
            "prediction.yaml",
        ]
    )
    config_file = LaunchConfiguration("config_file")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config,
                description="Autoware prediction adapter parameter file",
            ),
            Node(
                package="ad_lidar_perception",
                executable="ad_autoware_prediction_node",
                name="ad_autoware_prediction",
                output="screen",
                parameters=[config_file],
            ),
        ]
    )
