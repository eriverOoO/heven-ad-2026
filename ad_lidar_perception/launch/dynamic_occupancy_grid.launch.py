from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    config = str(
        Path(get_package_share_directory("ad_lidar_perception"))
        / "config"
        / "occupancy_grid"
        / "dynamic.yaml"
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "drivable_mask_topic",
                default_value="/ad/planning/drivable_mask",
            ),
            DeclareLaunchArgument(
                "runtime_summary_interval_frames",
                default_value="0",
            ),
            Node(
                package="ad_lidar_perception",
                executable="ad_dynamic_occupancy_grid_node",
                name="ad_dynamic_occupancy_grid",
                output="screen",
                parameters=[
                    config,
                    {
                        "topics.drivable_mask": ParameterValue(
                            LaunchConfiguration("drivable_mask_topic"),
                            value_type=str,
                        ),
                        "runtime_summary_interval_frames": ParameterValue(
                            LaunchConfiguration(
                                "runtime_summary_interval_frames"
                            ),
                            value_type=int,
                        ),
                    },
                ],
            ),
        ]
    )
