from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration

from ad_lidar_perception.autoware_provenance import verify_selection
from ad_lidar_perception.selection import load_selection


def _tracker_arguments(runtime, package_share: Path) -> dict[str, str]:
    arguments = {
        "input/detection01/objects":
            "/ad/perception/objects/detected",
        "input/detection01/channel": "lidar_clustering",
        "output/objects": "/ad/perception/objects/tracked",
        "tracker_setting_path": str(
            package_share / "config" / "tracking" / "autoware.yaml"
        ),
        "publish_merged_objects": "false",
    }
    for channel in range(2, 13):
        arguments[f"input/detection{channel:02d}/channel"] = "none"
    return arguments


def _launch_setup(context):
    package_share = Path(
        get_package_share_directory("ad_lidar_perception")
    )
    selection_path = Path(
        LaunchConfiguration("selection_config").perform(context)
    )
    lock_path = (
        package_share / "config" / "autoware_perception.lock.yaml"
    )
    data_root_text = LaunchConfiguration("data_root").perform(context)
    data_root = Path(data_root_text) if data_root_text else None
    selection = load_selection(selection_path)
    verified = verify_selection(
        selection,
        lock_path=lock_path,
        data_root=data_root,
    )
    if verified.tracker is None:
        raise RuntimeError(
            "tracking.launch.py requires tracker backend autoware"
        )
    arguments = _tracker_arguments(verified.tracker, package_share)
    return [
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(str(verified.tracker.launch_path)),
            launch_arguments=arguments.items(),
        )
    ]


def generate_launch_description():
    package_share = Path(
        get_package_share_directory("ad_lidar_perception")
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "selection_config",
                default_value=str(
                    package_share / "config" / "lidar_perception.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "data_root",
                default_value=EnvironmentVariable(
                    "AD_DATA_DIR", default_value=""
                ),
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
