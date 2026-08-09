import os
from pathlib import Path
import stat
import xml.etree.ElementTree as ET

from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_prefix,
    get_package_share_directory,
)
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from nav2_common.launch import RewrittenYaml


_PINNED_NAV2_VERSION = "1.1.20"
_PROXY_REQUIREMENT = "AD_PLANNER_ENABLE_NAV2_MPPI=ON"
_PROXY_EXECUTABLE = "ad_planner_mppi_follow_path_proxy"
_NAV2_REQUIREMENTS = (
    ("nav2_controller", ("lib/nav2_controller/controller_server",)),
    (
        "nav2_lifecycle_manager",
        ("lib/nav2_lifecycle_manager/lifecycle_manager",),
    ),
    (
        "nav2_mppi_controller",
        (
            "share/nav2_mppi_controller/mppic.xml",
            "share/nav2_mppi_controller/critics.xml",
        ),
    ),
    ("nav2_costmap_2d", ()),
    ("nav2_msgs", ()),
)


def _strict_regular_file(path, executable=False):
    try:
        file_stat = path.lstat()
    except OSError:
        return False
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size <= 0:
        return False
    return not executable or os.access(path, os.X_OK)


def _installed_proxy_is_usable(planner_prefix):
    executable = (
        Path(planner_prefix)
        / "lib"
        / "ad_planner"
        / _PROXY_EXECUTABLE
    )
    try:
        mode = executable.lstat().st_mode
    except OSError:
        return False

    if stat.S_ISREG(mode):
        return _strict_regular_file(executable, executable=True)
    if not stat.S_ISLNK(mode):
        return False

    try:
        target = executable.resolve(strict=True)
    except OSError:
        return False
    if target.name != _PROXY_EXECUTABLE:
        return False
    if not _strict_regular_file(target, executable=True):
        return False
    if target.parent.name != "ad_planner":
        return False
    cache = target.parent / "CMakeCache.txt"
    if not _strict_regular_file(cache):
        return False
    try:
        cache_text = cache.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError):
        return False
    cache_lines = set(cache_text.splitlines())
    return {
        "CMAKE_PROJECT_NAME:STATIC=ad_planner",
        "AD_PLANNER_ENABLE_NAV2_MPPI:BOOL=ON",
    }.issubset(cache_lines)


def _nav2_package_is_usable(prefix, package, leaves):
    manifest = Path(prefix) / "share" / package / "package.xml"
    if not _strict_regular_file(manifest):
        return False
    try:
        root = ET.parse(manifest).getroot()
    except (OSError, ET.ParseError):
        return False
    if root.findtext("name") != package:
        return False
    if root.findtext("version") != _PINNED_NAV2_VERSION:
        return False

    for relative in leaves:
        leaf = Path(prefix) / relative
        if not _strict_regular_file(
            leaf, executable=relative.startswith("lib/")
        ):
            return False
    return True


def _missing_runtime_requirements(prefix_resolver=None):
    resolve_prefix = prefix_resolver or get_package_prefix
    missing = []
    try:
        planner_prefix = resolve_prefix("ad_planner")
    except (PackageNotFoundError, ValueError):
        planner_prefix = None
    if planner_prefix is None or not _installed_proxy_is_usable(
        planner_prefix
    ):
        missing.append(_PROXY_REQUIREMENT)

    for package, leaves in _NAV2_REQUIREMENTS:
        try:
            prefix = resolve_prefix(package)
        except (PackageNotFoundError, ValueError):
            prefix = None
        if prefix is None or not _nav2_package_is_usable(
            prefix, package, leaves
        ):
            missing.append(package)
    return missing


def _create_mppi_nav2_actions(context):
    missing = _missing_runtime_requirements()
    if missing:
        diagnostic = (
            "Nav2 MPPI runtime requirements are not satisfied:\n"
            + "\n".join(f"- {requirement}" for requirement in missing)
        )
        raise RuntimeError(diagnostic)

    config_file = LaunchConfiguration("config_file")
    controller_config = RewrittenYaml(
        source_file=config_file,
        param_rewrites={},
        convert_types=True,
    )
    visualize = ParameterValue(
        LaunchConfiguration("visualize"), value_type=bool
    )

    reprojector = Node(
        package="ad_planner",
        executable="ad_occupancy_grid_reprojector_node",
        name="occupancy_grid_reprojector_mppi",
        output="screen",
    )
    controller_server = Node(
        package="nav2_controller",
        executable="controller_server",
        name="controller_server",
        output="screen",
        parameters=[
            controller_config,
            {"FollowPath.visualize": visualize},
        ],
        remappings=[
            ("cmd_vel", "/ad/planner/mppi/cmd_vel"),
            ("/trajectories", "/ad/viz/planner/mppi/trajectories"),
            (
                "transformed_global_plan",
                "/ad/viz/planner/mppi/transformed_reference",
            ),
        ],
    )
    lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_mppi",
        output="screen",
        parameters=[
            {
                "autostart": True,
                "node_names": ["controller_server"],
            }
        ],
    )
    follow_path_proxy = Node(
        package="ad_planner",
        executable=_PROXY_EXECUTABLE,
        name="mppi_follow_path_proxy",
        output="screen",
    )
    return [
        reprojector,
        controller_server,
        lifecycle_manager,
        follow_path_proxy,
    ]


def generate_launch_description():
    package_share = get_package_share_directory("ad_planner")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=os.path.join(
                    package_share,
                    "config",
                    "local_planning",
                    "mppi_nav2.yaml",
                ),
            ),
            DeclareLaunchArgument(
                "visualize",
                default_value="false",
                description="Publish MPPI candidate trajectory markers",
            ),
            OpaqueFunction(function=_create_mppi_nav2_actions),
        ]
    )
