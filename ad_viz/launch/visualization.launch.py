from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import SetParameter
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_config = str(
        Path(get_package_share_directory("ad_viz")) / "rviz" / "ad.rviz"
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "viz_rviz_config", default_value=default_config
            ),
            DeclareLaunchArgument(
                "viz_clock_rollback_reset_sec", default_value="5.0"
            ),
            SetParameter(name="use_sim_time", value=True),
            Node(
                package="ad_viz",
                executable="ad_perception_marker_node",
                name="ad_perception_marker",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": True,
                        "clock_rollback_reset_sec": LaunchConfiguration(
                            "viz_clock_rollback_reset_sec"
                        )
                    }
                ],
            ),
            Node(
                package="ad_viz",
                executable="ad_localization_ground_odometry",
                name="ad_localization_ground_odometry",
                output="screen",
            ),
            Node(
                package="ad_viz",
                executable="ad_localization_route_elevation_odometry",
                name="ad_localization_route_elevation_odometry",
                output="screen",
            ),
            Node(
                package="ad_viz",
                executable="ad_visualization_tf_relay",
                name="ad_visualization_tf_relay",
                output="screen",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="ad_rviz",
                output="screen",
                arguments=["-d", LaunchConfiguration("viz_rviz_config")],
                remappings=[
                    ("/tf", "/ad/viz/tf"),
                    ("/tf_static", "/ad/viz/tf_static"),
                ],
            ),
        ]
    )
