from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
)

from ad_bringup.course_stack import build_course_stack


def generate_launch_description():
    localization_share = Path(get_package_share_directory("ad_localization"))
    data_dir = LaunchConfiguration("data_dir")
    control_enabled = LaunchConfiguration("control_enabled")
    map_path = LaunchConfiguration("map_path")
    path_file = LaunchConfiguration("path_file")
    traffic_light_enabled = LaunchConfiguration("traffic_light_enabled")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "data_dir",
                default_value=EnvironmentVariable("AD_DATA_DIR", default_value=""),
            ),
            DeclareLaunchArgument("control_enabled", default_value="false"),
            DeclareLaunchArgument(
                "map_path",
                default_value=str(
                    localization_share / "maps" / "cp14_to_cp15.pcd"
                ),
            ),
            DeclareLaunchArgument(
                "path_file",
                default_value="path/2026_molit_comp_global_path.txt",
            ),
            DeclareLaunchArgument(
                "traffic_light_enabled", default_value="false"
            ),
            *build_course_stack(
                data_dir=data_dir,
                control_enabled=control_enabled,
                map_path=map_path,
                path_file=path_file,
                traffic_light_enabled=traffic_light_enabled,
            ),
        ]
    )
