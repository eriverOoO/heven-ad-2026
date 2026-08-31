import hashlib
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node


def _create_node(context):
    data_dir = LaunchConfiguration("data_dir").perform(context)
    path_file = LaunchConfiguration("path_file").perform(context)
    corridor_file = LaunchConfiguration("route_corridor_file").perform(context)
    config_file = LaunchConfiguration("config_file").perform(context)
    if not data_dir or not path_file or not corridor_file:
        raise RuntimeError(
            "highway merge gap risk requires data_dir, path_file, and "
            "route_corridor_file"
        )
    path = os.path.join(data_dir, path_file)
    try:
        with open(path, "rb") as stream:
            digest = hashlib.sha256(stream.read()).hexdigest()
    except OSError as error:
        raise RuntimeError(
            f"highway merge gap risk requires global path: {path}"
        ) from error
    return [
        Node(
            package="ad_planner",
            executable="ad_highway_merge_gap_risk_node",
            name="ad_highway_merge_gap_risk",
            output="screen",
            parameters=[
                config_file,
                {
                    "data_dir": data_dir,
                    "route_corridor_file": corridor_file,
                    "route_corridor.expected_global_path_sha256": digest,
                },
            ],
        )
    ]


def generate_launch_description():
    share = get_package_share_directory("ad_planner")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "data_dir",
                default_value=EnvironmentVariable(
                    "AD_DATA_DIR", default_value=""
                ),
            ),
            DeclareLaunchArgument(
                "config_file",
                default_value=os.path.join(
                    share, "config", "highway_merge_gap_risk.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "path_file",
                default_value="path/2026_molit_comp_global_path.txt",
            ),
            DeclareLaunchArgument(
                "route_corridor_file",
                default_value="map/route_corridor.json",
            ),
            OpaqueFunction(function=_create_node),
        ]
    )
