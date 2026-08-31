"""Opt-in Highway Merge Ego-on-Ramp Scenario v1 launch.

Brings up the highway-merge stack (gap risk -> gap response -> planner with the
response integration + mission primitive enabled) driven by the VALIDATION-ONLY
route fixture `ad_data/path/test_highway_merge_ego_ramp_path.txt`, so the ego is
represented on the real acceleration lane `route:0:left:1` and merges onto
`route:0`.

This launch is never included by `planner.launch.py` and changes no production
default. The production competition route
`ad_data/path/2026_molit_comp_global_path.txt` is untouched. It generates a
route corridor for the fixture at launch time -- a byte copy of the real
`route_corridor.json` with only `source_sha256.global_path` rewritten to the
fixture's digest (the lane geometry, including route:0 and route:0:left:1, is
identical).

It commands no steering / lane change / CtrlCmd beyond the single planner
`/ad/control/command` publisher, and it does not generate a lateral merge path.

Usage:
    ros2 launch ad_planner highway_merge_ego_ramp_scenario.launch.py \\
        data_dir:=/abs/path/to/ad_data
"""

import hashlib
import json
import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

FIXTURE_PATH_FILE = "path/test_highway_merge_ego_ramp_path.txt"
PRODUCTION_CORRIDOR_FILE = "map/route_corridor.json"


def _prepare(context):
    data_dir = LaunchConfiguration("data_dir").perform(context)
    if not data_dir or not os.path.isdir(data_dir):
        raise RuntimeError(
            "highway_merge_ego_ramp_scenario requires data_dir:=<abs path to ad_data>"
        )
    fixture_path = os.path.join(data_dir, FIXTURE_PATH_FILE)
    corridor_path = os.path.join(data_dir, PRODUCTION_CORRIDOR_FILE)
    for required in (fixture_path, corridor_path):
        if not os.path.isfile(required):
            raise RuntimeError(f"missing scenario input: {required}")

    with open(fixture_path, "rb") as stream:
        fixture_digest = hashlib.sha256(stream.read()).hexdigest()

    corridor = json.loads(open(corridor_path, encoding="utf-8").read())
    # Only the recorded global-path digest changes; every lane stays identical.
    corridor["source_sha256"]["global_path"] = fixture_digest
    scratch_dir = tempfile.mkdtemp(prefix="heven_ego_ramp_scenario_")
    fixture_corridor_path = os.path.join(scratch_dir, "route_corridor.json")
    with open(fixture_corridor_path, "w", encoding="utf-8") as stream:
        json.dump(corridor, stream)

    planner_share = get_package_share_directory("ad_planner")
    common = {
        "data_dir": data_dir,
        "path_file": FIXTURE_PATH_FILE,
        "route_corridor_file": fixture_corridor_path,
        "route_corridor.expected_global_path_sha256": fixture_digest,
    }
    return [
        Node(
            package="ad_planner",
            executable="ad_highway_merge_gap_risk_node",
            name="ad_highway_merge_gap_risk",
            output="screen",
            parameters=[
                os.path.join(planner_share, "config", "highway_merge_gap_risk.yaml"),
                common,
            ],
        ),
        Node(
            package="ad_planner",
            executable="ad_highway_merge_gap_response_node",
            name="ad_highway_merge_gap_response",
            output="screen",
            parameters=[
                os.path.join(planner_share, "config", "highway_merge_gap_response.yaml"),
            ],
        ),
        Node(
            package="ad_planner",
            executable="ad_planner_node",
            name="ad_planner",
            output="screen",
            parameters=[
                os.path.join(planner_share, "config", "planner.yaml"),
                common,
                {
                    "enable_highway_merge_response_integration": True,
                    "enable_highway_merge_mission": True,
                    "enable_highway_merge_reference_path": True,
                },
            ],
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("data_dir", default_value=""),
            OpaqueFunction(function=_prepare),
        ]
    )
