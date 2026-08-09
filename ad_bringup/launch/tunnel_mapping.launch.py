from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
)

from ad_bringup.tunnel_stack import build_tunnel_stack


def generate_launch_description():
    localization_share = Path(get_package_share_directory("ad_localization"))
    data_dir = LaunchConfiguration("data_dir")
    control_enabled = LaunchConfiguration("control_enabled")
    map_output_path = LaunchConfiguration("map_output_path")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "data_dir",
                default_value=EnvironmentVariable(
                    "AD_DATA_DIR", default_value=""
                ),
            ),
            DeclareLaunchArgument("control_enabled", default_value="false"),
            DeclareLaunchArgument(
                "map_output_path",
                default_value=str(
                    localization_share / "maps" / "cp14_to_cp15.pcd"
                ),
            ),
            *build_tunnel_stack(
                mode="mapping",
                data_dir=data_dir,
                control_enabled=control_enabled,
                map_path=map_output_path,
            ),
        ]
    )
