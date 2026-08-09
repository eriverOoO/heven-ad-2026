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


def _detector_arguments(runtime, package_share: Path) -> dict[str, str]:
    return {
        "input/pointcloud": "/ad/perception/lidar/points_xyzirc",
        "output/objects": "/ad/perception/objects/detected",
        "model_name": runtime.backend,
        "model_path": str(runtime.model_path),
        "model_param_path": str(
            package_share
            / "config"
            / "detectors"
            / f"{runtime.backend}.yaml"
        ),
        "ml_package_param_path": str(runtime.ml_package_path),
        "class_remapper_param_path": str(runtime.class_remapper_path),
        "build_only": "true" if runtime.build_only else "false",
    }


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
    if verified.detector is None:
        raise RuntimeError(
            "object_detection.launch.py requires a selected detector"
        )
    arguments = _detector_arguments(verified.detector, package_share)
    return [
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(str(verified.detector.launch_path)),
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
