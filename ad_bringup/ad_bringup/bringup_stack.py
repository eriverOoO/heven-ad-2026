from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from ad_bringup.component_config import load_components


_COMPONENT_NAMES = (
    "description",
    "localization",
    "planner",
    "lidar_perception",
    "camera_perception",
    "bridge",
    "visualization",
)


def _launch_file(package, name):
    return PythonLaunchDescriptionSource(
        str(Path(get_package_share_directory(package)) / "launch" / name)
    )


def _include(package, launch_file, arguments=None):
    return IncludeLaunchDescription(
        _launch_file(package, launch_file),
        launch_arguments=(arguments or {}).items(),
    )


def registered_component_names():
    return _COMPONENT_NAMES


def build_bringup_stack(*, components_file, status_topic, bridge_config=None):
    """Build enabled stack actions with plain, readable launch conditionals."""
    enabled = load_components(components_file, _COMPONENT_NAMES)
    actions = []

    if enabled["description"]:
        actions.append(_include("ad_description", "description.launch.py"))
    if enabled["localization"]:
        actions.append(
            _include(
                "ad_localization",
                "localization.launch.py",
                {
                    "status_topic": status_topic,
                    "localization_backend": LaunchConfiguration(
                        "localization_backend"
                    ),
                },
            )
        )
    if enabled["planner"]:
        actions.append(
            _include(
                "ad_planner",
                "planner.launch.py",
                {
                    "data_dir": LaunchConfiguration("data_dir"),
                    "path_file": LaunchConfiguration("path_file"),
                    "route_corridor_file": LaunchConfiguration(
                        "route_corridor_file", default=""
                    ),
                    "path_tracking_backend": LaunchConfiguration(
                        "path_tracking_backend", default=""
                    ),
                    "perception_enabled": LaunchConfiguration(
                        "perception_enabled", default=""
                    ),
                    "local_motion_prediction_mode": LaunchConfiguration(
                        "local_motion_prediction_mode", default=""
                    ),
                    "tuning_lease_required": LaunchConfiguration(
                        "tuning_lease_required", default=""
                    ),
                },
            )
        )
    if enabled["lidar_perception"]:
        actions.append(
            _include(
                "ad_lidar_perception",
                "lidar_perception.launch.py",
                {"platform_profile": "morai"},
            )
        )
    if enabled["camera_perception"]:
        actions.append(
            _include("ad_camera_perception", "camera_perception.launch.py")
        )
    if enabled["bridge"]:
        bridge_arguments = {
            "control_enabled": LaunchConfiguration("control_enabled"),
            "enable_velodyne_points": str(
                enabled["lidar_perception"]
            ).lower(),
        }
        if bridge_config is not None:
            bridge_arguments["config"] = str(bridge_config)
        actions.append(
            _include(
                "ad_morai_bridge",
                "bridge.launch.py",
                bridge_arguments,
            )
        )
    if enabled["visualization"]:
        actions.append(_include("ad_viz", "visualization.launch.py"))

    return actions
