from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from ad_morai_bridge_dev.perception.validation_contract import (
    canonical_launch_arguments,
)


def _lidar_arguments(
    package_share: Path, description_share: Path
) -> dict[str, str]:
    return canonical_launch_arguments(package_share, description_share)


def generate_launch_description():
    package_share = Path(
        get_package_share_directory("ad_lidar_perception")
    )
    description_share = Path(
        get_package_share_directory("ad_description")
    )
    lidar_launch = package_share / "launch" / "lidar_perception.launch.py"
    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(lidar_launch)),
                launch_arguments=_lidar_arguments(
                    package_share, description_share
                ).items(),
            )
        ]
    )
