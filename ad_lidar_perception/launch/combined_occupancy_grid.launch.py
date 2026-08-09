from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = str(
        Path(get_package_share_directory("ad_lidar_perception"))
        / "config"
        / "occupancy_grid"
        / "combined.yaml"
    )
    return LaunchDescription(
        [
            Node(
                package="ad_lidar_perception",
                executable="ad_combined_occupancy_grid_node",
                name="ad_combined_occupancy_grid",
                output="screen",
                parameters=[config],
            ),
        ]
    )
