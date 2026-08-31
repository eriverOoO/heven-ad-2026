import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _create_node(context):
    config_file = LaunchConfiguration("config_file").perform(context)
    return [
        Node(
            package="ad_planner",
            executable="ad_highway_merge_gap_response_node",
            name="ad_highway_merge_gap_response",
            output="screen",
            parameters=[config_file],
        )
    ]


def generate_launch_description():
    share = get_package_share_directory("ad_planner")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=os.path.join(
                    share, "config", "highway_merge_gap_response.yaml"
                ),
            ),
            OpaqueFunction(function=_create_node),
        ]
    )
