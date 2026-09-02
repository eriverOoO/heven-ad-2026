"""Standalone launch of the existing road-corridor drivable-mask producer.

Starts only ``ad_road_corridor_mask_node`` (package ``ad_planner``) against the
committed, checksum-verified competition route corridor
(``ad_data/map/route_corridor.json``) and global path
(``ad_data/path/2026_molit_comp_global_path.txt`` by default). This is the
same node ``planner.launch.py`` already starts as part of the full planner
graph (see ``_create_road_corridor_mask_node``); this file exists so a
perception-only demo (no planner, no controller, no CtrlCmd publisher) can
reuse it standalone. No new mask rasterization logic - the node, its
geometry contract, and its route-corridor source are unchanged.

The node publishes ``/ad/planning/drivable_mask`` (``nav_msgs/OccupancyGrid``,
``base_link`` frame, 0=drivable/100=non-drivable) whenever a
``/ad/sensors/lidar/points`` or ``/ad/perception/objects/predicted`` message
triggers it and an exact-stamp ``map -> base_link`` transform is available. It
never fabricates a mask: no route data or no transform means no publish for
that stamp (see ``road_corridor_mask_node.cpp::on_trigger``).

Parameters are passed as ONE merged dict, not ``[config_file, overrides]``
(the pattern ``cut_in_risk.launch.py`` and ``planner.launch.py`` use). Passing
``config_file`` and the ``data_dir``/``route_corridor_file``/
``route_corridor.expected_global_path_sha256`` override as two separate
``--params-file`` arguments for the same node was found, empirically, to
non-deterministically fail to apply the override for those specific keys
(observed here as intermittent ``FATAL: set data_dir or AD_DATA_DIR`` /
``expected SHA-256 for 'global_path' is malformed`` node crashes, reproduced
with byte-identical params files across repeated runs of the SAME command,
in an otherwise clean, single-process environment) - a real ``rcl_yaml_
param_parser`` merge-race across two files that both declare the same keys
for one node under different specificity (a wildcard ``/**:`` override file
vs. the checked-in YAML's node-name-scoped placeholder). Reading the
checked-in YAML in Python and overlaying the override on top of it *before*
launch, so the process receives exactly one parameter source per key, was
verified (5/5 clean runs, byte-identical inputs) to remove the race. This is
launch-wiring-only; the node's own parameter-declaration code is untouched.
The same underlying two-file pattern is still used elsewhere in this
package (``cut_in_risk.launch.py``, ``planner.launch.py``); fixing those is
out of scope here and is left as a follow-up.
"""

import hashlib
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
import yaml


def _load_node_parameters(config_file, node_name):
    with open(config_file, encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    try:
        parameters = document[node_name]["ros__parameters"]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            f"{config_file} must declare {node_name}.ros__parameters"
        ) from error
    if not isinstance(parameters, dict):
        raise RuntimeError(f"{config_file}: {node_name}.ros__parameters must be a mapping")
    return dict(parameters)


def _create_node(context):
    data_dir = LaunchConfiguration("data_dir").perform(context)
    path_file = LaunchConfiguration("path_file").perform(context)
    corridor_file = LaunchConfiguration("route_corridor_file").perform(context)
    config_file = LaunchConfiguration("config_file").perform(context)
    if not data_dir or not path_file or not corridor_file:
        raise RuntimeError(
            "road corridor mask requires data_dir, path_file, and "
            "route_corridor_file"
        )
    path = os.path.join(data_dir, path_file)
    try:
        with open(path, "rb") as stream:
            digest = hashlib.sha256(stream.read()).hexdigest()
    except OSError as error:
        raise RuntimeError(
            f"road corridor mask requires global path: {path}"
        ) from error
    parameters = _load_node_parameters(config_file, "ad_road_corridor_mask")
    parameters.update(
        {
            "data_dir": data_dir,
            "route_corridor_file": corridor_file,
            "route_corridor.expected_global_path_sha256": digest,
        }
    )
    return [
        Node(
            package="ad_planner",
            executable="ad_road_corridor_mask_node",
            name="ad_road_corridor_mask",
            output="screen",
            parameters=[parameters],
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
                description=(
                    "Directory containing checksum-verified competition data "
                    "(path_file and route_corridor_file are resolved "
                    "relative to it)"
                ),
            ),
            DeclareLaunchArgument(
                "config_file",
                default_value=os.path.join(
                    share, "config", "road_corridor_mask.yaml"
                ),
            ),
            DeclareLaunchArgument(
                "path_file",
                default_value="path/2026_molit_comp_global_path.txt",
                description=(
                    "Committed global path whose SHA-256 guards the "
                    "route corridor against staleness"
                ),
            ),
            DeclareLaunchArgument(
                "route_corridor_file",
                default_value="map/route_corridor.json",
            ),
            OpaqueFunction(function=_create_node),
        ]
    )
