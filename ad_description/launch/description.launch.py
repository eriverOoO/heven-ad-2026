from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro
import yaml


def _launch_setup(context):
    package_share = Path(get_package_share_directory("ad_description"))
    sensor_config = Path(LaunchConfiguration("sensor_config").perform(context))
    requested_profile = LaunchConfiguration("sensor_profile").perform(context)
    if requested_profile:
        sensor_profile = requested_profile
    else:
        sensor_document = yaml.safe_load(sensor_config.read_text(encoding="utf-8"))
        sensor_profile = sensor_document["active_profile"]

    robot_description = xacro.process_file(
        str(package_share / "urdf" / "ad_ioniq5.urdf.xacro"),
        mappings={
            "sensor_config": str(sensor_config),
            "sensor_profile": sensor_profile,
        },
    ).toxml()

    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description}],
        )
    ]


def generate_launch_description():
    package_share = Path(get_package_share_directory("ad_description"))
    default_sensor_config = str(package_share / "config" / "sensor_mounts.yaml")
    return LaunchDescription(
        [
            DeclareLaunchArgument("sensor_config", default_value=default_sensor_config),
            DeclareLaunchArgument("sensor_profile", default_value="",
                                  description="Blank selects active_profile from sensor_mounts.yaml."),
            OpaqueFunction(function=_launch_setup),
        ]
    )
