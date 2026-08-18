from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = Path(get_package_share_directory("ad_lidar_perception"))
    default_config = package_share / "config" / "tracking" / "ab3dmot.yaml"

    arguments = [
        DeclareLaunchArgument("enabled", default_value="true"),
        DeclareLaunchArgument(
            "input_topic", default_value="/ad/perception/objects/detected"
        ),
        DeclareLaunchArgument(
            "output_topic", default_value="/experiment/tracked/ab3dmot"
        ),
        DeclareLaunchArgument("target_frame", default_value="odom"),
        DeclareLaunchArgument(
            "ab3dmot_root",
            default_value="",
            description=(
                "Absolute path to the references/ab3dmot submodule checkout; "
                "blank uses the source-tree-relative default."
            ),
        ),
        DeclareLaunchArgument("config_path", default_value=str(default_config)),
    ]
    node = Node(
        package="ad_lidar_perception",
        executable="ad_ab3dmot_tracker",
        name="ad_ab3dmot_tracker",
        output="screen",
        parameters=[
            LaunchConfiguration("config_path"),
            {
                "enabled": ParameterValue(LaunchConfiguration("enabled"), value_type=bool),
                "input_topic": LaunchConfiguration("input_topic"),
                "output_topic": LaunchConfiguration("output_topic"),
                "target_frame": LaunchConfiguration("target_frame"),
                "ab3dmot_root": LaunchConfiguration("ab3dmot_root"),
            },
        ],
    )
    return LaunchDescription([*arguments, node])
