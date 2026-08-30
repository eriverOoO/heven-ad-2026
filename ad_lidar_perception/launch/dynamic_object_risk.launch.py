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
            "planning",
            "dynamic_object_risk.yaml",
        ]
    )
    config_file = LaunchConfiguration("config_file")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config,
                description="Dynamic Object Risk Interface parameter file",
            ),
            Node(
                package="ad_lidar_perception",
                executable="ad_dynamic_object_risk_node",
                name="ad_dynamic_object_risk",
                output="screen",
                parameters=[config_file],
            ),
        ]
    )
