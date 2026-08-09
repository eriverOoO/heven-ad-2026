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
        / "static.yaml"
    )
    points_topic = LaunchConfiguration("points_topic")
    drivable_mask_topic = LaunchConfiguration("drivable_mask_topic")
    static_ungated_topic = LaunchConfiguration(
        "static_ungated_topic"
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "points_topic", default_value="/ad/sensors/lidar/points"
            ),
            DeclareLaunchArgument(
                "drivable_mask_topic",
                default_value="/ad/planning/drivable_mask",
            ),
            DeclareLaunchArgument(
                "static_ungated_topic",
                default_value="/ad/viz/perception/occupancy/static_ungated",
                description=(
                    "Visualization-only static OGM before road-mask gating."
                ),
            ),
            Node(
                package="ad_lidar_perception",
                executable="ad_lidar_perception_node",
                name="ad_lidar_perception",
                output="screen",
                parameters=[
                    config,
                    {
                        "topics.points": ParameterValue(
                            points_topic, value_type=str
                        ),
                        "topics.drivable_mask": ParameterValue(
                            drivable_mask_topic, value_type=str
                        ),
                        "visualization.topics.static_ungated": ParameterValue(
                            static_ungated_topic, value_type=str
                        ),
                    },
                ],
            ),
        ]
    )
