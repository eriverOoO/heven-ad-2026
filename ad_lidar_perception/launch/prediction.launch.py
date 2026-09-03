from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
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
            DeclareLaunchArgument(
                "runtime_summary_interval_frames",
                default_value="0",
            ),
            DeclareLaunchArgument(
                "yaw_rate_source",
                default_value="tracker",
                description=(
                    "IMM yaw-rate source: 'tracker' (default, reproduces the "
                    "pre-Curve-Aware-Prediction behaviour) or 'motion_history' "
                    "(derive turn rate from recent velocity history). "
                    "Authoritative over the config file's yaw_rate_source key."
                ),
            ),
            Node(
                package="ad_lidar_perception",
                executable="ad_autoware_prediction_node",
                name="ad_autoware_prediction",
                output="screen",
                parameters=[
                    config_file,
                    {
                        "runtime_summary_interval_frames": ParameterValue(
                            LaunchConfiguration(
                                "runtime_summary_interval_frames"
                            ),
                            value_type=int,
                        ),
                        "yaw_rate_source": LaunchConfiguration(
                            "yaw_rate_source"
                        ),
                    },
                ],
            ),
        ]
    )
