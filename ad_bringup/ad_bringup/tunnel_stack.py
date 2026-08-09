from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


VEHICLE_STATUS_TOPIC = "/ad/vehicle/status"
TUNNEL_ROUTE_FILE = "paths/cp14_to_cp15.txt"
TUNNEL_INITIAL_POSITION_XY_M = "38.868875371112615,-480.68740975673563"


def _launch_file(package: str, name: str):
    share = Path(get_package_share_directory(package))
    return PythonLaunchDescriptionSource(str(share / "launch" / name))


def _include(package: str, name: str, arguments=None):
    return IncludeLaunchDescription(
        _launch_file(package, name),
        launch_arguments=(arguments or {}).items(),
    )


def build_tunnel_stack(
    *,
    mode: str,
    data_dir,
    control_enabled,
    map_path,
):
    """Build one development-only CP14-to-CP15 FastLIO stack."""

    if mode not in {"mapping", "localization"}:
        raise ValueError("mode must be exactly 'mapping' or 'localization'")

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
            # MORAI emits the revolution from one Unity-frame pose.  FAST-LIO's
            # MORAI profile independently bypasses its internal undistortion.
            "velodyne_point_timing_mode": "zero",
            "velodyne_organize_cloud": "false",
        },
    )
    description = _include("ad_description", "description.launch.py")
    fastlio_argument = "map_output_path" if mode == "mapping" else "map_path"
    fastlio = _include(
        "ad_localization",
        f"fastlio_{mode}.launch.py",
        {
            "platform_profile": "morai",
            "status_topic": VEHICLE_STATUS_TOPIC,
            "autostart": "true",
            "initial_position_override_xy_m": TUNNEL_INITIAL_POSITION_XY_M,
            fastlio_argument: map_path,
        },
    )
    planner = _include(
        "ad_planner",
        "planner.launch.py",
        {
            "data_dir": data_dir,
            "path_file": TUNNEL_ROUTE_FILE,
            "path_tracking_backend": "profile_stanley",
            # Premapping remains crawl-speed: a bad LIO estimate must not
            # turn a map-quality experiment into a runaway.
            "target_speed_mps": "1.0",
            "perception_enabled": "false",
        },
    )
    visualization = _include("ad_viz", "visualization.launch.py")
    return [bridge, description, fastlio, planner, visualization]
