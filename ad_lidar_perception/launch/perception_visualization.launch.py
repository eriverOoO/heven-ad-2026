from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = Path(
        get_package_share_directory("ad_lidar_perception")
    )
    rviz_config = LaunchConfiguration("rviz_config")
    use_sim_time = ParameterValue(
        LaunchConfiguration("use_sim_time"), value_type=bool
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("start_rviz", default_value="false"),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument(
                "visualize_predictions", default_value="true"
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=str(
                    package_share / "rviz" / "heven_perception.rviz"
                ),
            ),
            Node(
                package="ad_viz",
                executable="perception_visualizer_node",
                name="perception_visualizer",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "visualize_predictions": ParameterValue(
                            LaunchConfiguration("visualize_predictions"),
                            value_type=bool,
                        ),
                    }
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="heven_perception_rviz",
                output="screen",
                arguments=["-d", rviz_config],
                parameters=[{"use_sim_time": use_sim_time}],
                condition=IfCondition(LaunchConfiguration("start_rviz")),
            ),
        ]
    )
