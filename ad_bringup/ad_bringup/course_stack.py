from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource


VEHICLE_STATUS_TOPIC = "/ad/vehicle/status"


def _launch_file(package: str, name: str):
    share = Path(get_package_share_directory(package))
    return PythonLaunchDescriptionSource(str(share / "launch" / name))


def _include(package: str, name: str, arguments=None, condition=None):
    options = {"launch_arguments": (arguments or {}).items()}
    if condition is not None:
        options["condition"] = condition
    return IncludeLaunchDescription(
        _launch_file(package, name),
        **options,
    )


def build_course_stack(
    *,
    data_dir,
    control_enabled,
    map_path,
    path_file,
    traffic_light_enabled,
):
    """Build the obstacle-free CP13-to-route-end development stack."""

    bridge_config = (
        Path(get_package_share_directory("ad_morai_bridge"))
        / "config"
        / "tunnel_fastlio.yaml"
    )
    bridge = _include(
        "ad_morai_bridge",
        "bridge.launch.py",
        {
            "config": str(bridge_config),
            "control_enabled": control_enabled,
            "enable_velodyne_points": "true",
            # MORAI emits one-pose snapshot scans; never synthesize rolling time.
            "velodyne_point_timing_mode": "zero",
            "velodyne_organize_cloud": "false",
            "enable_traffic_light_camera": traffic_light_enabled,
        },
    )
    description = _include("ad_description", "description.launch.py")
    localization = _include(
        "ad_localization",
        "hybrid_localization.launch.py",
        {
            "platform_profile": "morai",
            "status_topic": VEHICLE_STATUS_TOPIC,
            "autostart": "true",
            "map_path": map_path,
        },
    )
    planner = _include(
        "ad_planner",
        "planner.launch.py",
        {
            "data_dir": data_dir,
            "path_file": path_file,
            "path_tracking_backend": "profile_stanley",
            "perception_enabled": "false",
        },
    )
    traffic_signal = _include(
        "ad_camera_perception",
        "traffic_signal.launch.py",
        condition=IfCondition(traffic_light_enabled),
    )
    visualization = _include("ad_viz", "visualization.launch.py")
    return [
        bridge,
        description,
        localization,
        planner,
        traffic_signal,
        visualization,
    ]
