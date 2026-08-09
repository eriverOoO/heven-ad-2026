from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bridge_share = Path(get_package_share_directory("ad_morai_bridge"))
    dev_bridge_share = Path(
        get_package_share_directory("ad_morai_bridge_dev")
    )
    competition_config = LaunchConfiguration("competition_config")
    development_config = LaunchConfiguration("development_config")
    control_enabled = LaunchConfiguration("control_enabled")
    enable_velodyne_points = LaunchConfiguration("enable_velodyne_points")
    platform_profile = LaunchConfiguration("platform_profile")
    grpc_target = LaunchConfiguration("grpc_target")
    scenario_reset_enabled = LaunchConfiguration("scenario_reset_enabled")
    scenario_reset_file = LaunchConfiguration("scenario_reset_file")

    competition = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(bridge_share / "launch" / "bridge.launch.py")
        ),
        launch_arguments={
            "config": competition_config,
            "control_enabled": control_enabled,
            "enable_velodyne_points": enable_velodyne_points,
            "platform_profile": platform_profile,
        }.items(),
    )
    development = Node(
        package="ad_morai_bridge_dev",
        executable="ad_morai_bridge_dev_node",
        name="ad_morai_bridge_dev",
        output="screen",
        parameters=[development_config, {"grpc.target": grpc_target}],
    )
    scenario_reset = Node(
        package="ad_morai_bridge_dev",
        executable="ad_morai_scenario_reset",
        name="ad_morai_scenario_reset",
        output="screen",
        condition=IfCondition(scenario_reset_enabled),
        parameters=[
            {
                "scenario_file": scenario_reset_file,
                "grpc.target": grpc_target,
                "grpc.timeout_sec": 5.0,
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "competition_config",
                default_value=str(
                    bridge_share / "config" / "competition.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "development_config",
                default_value=str(
                    dev_bridge_share / "config" / "development.yaml"
                ),
            ),
            DeclareLaunchArgument("control_enabled", default_value="true"),
            DeclareLaunchArgument(
                "enable_velodyne_points",
                default_value="false",
            ),
            DeclareLaunchArgument(
                "platform_profile",
                default_value="morai",
                description="exactly morai or real_hardware",
            ),
            DeclareLaunchArgument(
                "grpc_target",
                default_value="127.0.0.1:7789",
            ),
            DeclareLaunchArgument(
                "scenario_reset_enabled",
                default_value="false",
            ),
            DeclareLaunchArgument(
                "scenario_reset_file",
                default_value="",
            ),
            competition,
            development,
            scenario_reset,
        ]
    )
