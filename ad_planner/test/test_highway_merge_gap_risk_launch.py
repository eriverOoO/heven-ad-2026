import importlib.util
import json
from pathlib import Path

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node
import yaml


PACKAGE = Path(__file__).resolve().parents[1]

_FORBIDDEN_TOKENS = (
    "merge_allowed",
    "merge_safe",
    "merge_ready",
    "safe_to_merge",
    "safe_gap",
    "accepted_gap",
    "gap_accepted",
    "lane_change",
    "change_lane",
    "requested_speed",
    "risk_score",
    "brake",
    "throttle",
    "steer",
)


def _load_module(name, relative):
    path = PACKAGE / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_is_geometric_and_uses_canonical_topics():
    parameters = yaml.safe_load(
        (PACKAGE / "config" / "highway_merge_gap_risk.yaml").read_text(
            encoding="utf-8"
        )
    )["ad_highway_merge_gap_risk"]["ros__parameters"]
    assert (
        parameters["topics.dynamic_object_risks"]
        == "/ad/planning/dynamic_object_risks"
    )
    assert parameters["topics.odometry"] == "/ad/localization/odometry"
    assert parameters["topics.output"] == "/ad/planning/highway_merge_gap_risks"
    assert parameters["merge_zone_id"] == "kcity_highway_onramp"
    assert parameters["ego_speed_epsilon_mps"] > 0.0
    assert parameters["maximum_ego_approach_distance_m"] > 0.0
    assert parameters["relevant_rear_window_m"] >= 0.0
    assert parameters["relevant_front_window_m"] >= 0.0
    assert not any(
        token in key.lower()
        for key in parameters
        for token in _FORBIDDEN_TOKENS
    )


def test_merge_geometry_is_well_formed_map_frame_and_provenanced():
    document = json.loads(
        (PACKAGE / "config" / "highway_merge.json").read_text(encoding="utf-8")
    )
    assert document["schema_version"] == 1
    assert document["frame_id"] == "map"
    assert "provenance" in document
    assert (
        document["route_corridor_link_set_sha256"]
        == "5ce0fd57da07b83221c0a4d117b0606b2d9c094c8de476a7d1176c78a4dc5b0f"
    )
    zones = document["merge_zones"]
    assert len(zones) >= 1
    zone = next(z for z in zones if z["merge_zone_id"] == "kcity_highway_onramp")
    assert zone["route_s_zone_entry_m"] < zone["route_s_merge_complete_m"]
    assert zone["source_lane_sequence_id"] == "route:0:left:1"
    assert zone["target_lane_sequence_id"] == "route:0"
    text = json.dumps(document).lower()
    assert not any(token in text for token in _FORBIDDEN_TOKENS)


def test_merge_geometry_matches_route_corridor_source_lane():
    corridor = json.loads(
        (PACKAGE.parent / "ad_data" / "map" / "route_corridor.json").read_text(
            encoding="utf-8"
        )
    )
    lanes = {lane["lane_sequence_id"]: lane for lane in corridor["lanes"]}
    document = json.loads(
        (PACKAGE / "config" / "highway_merge.json").read_text(encoding="utf-8")
    )
    zone = document["merge_zones"][0]
    source = lanes[zone["source_lane_sequence_id"]]
    assert (
        abs(source["points"][0]["route_s_m"] - zone["route_s_zone_entry_m"]) < 2.0
    )
    assert (
        abs(source["points"][-1]["route_s_m"] - zone["route_s_merge_complete_m"])
        < 2.0
    )
    # The pinned link set is shared between the corridor and the actor preset.
    assert (
        corridor["source_sha256"]["link_set.json"]
        == document["route_corridor_link_set_sha256"]
    )
    assert zone["target_lane_sequence_id"] == corridor["primary_lane_sequence_id"]


def test_standalone_launch_starts_only_the_merge_node(monkeypatch):
    module = _load_module(
        "highway_merge_gap_risk_launch", "launch/highway_merge_gap_risk.launch.py"
    )
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda package: str(PACKAGE.parent / package),
    )
    description = module.generate_launch_description()
    arguments = {
        action.name
        for action in description.entities
        if isinstance(action, DeclareLaunchArgument)
    }
    assert arguments == {
        "data_dir",
        "config_file",
        "path_file",
        "route_corridor_file",
    }
    opaque = [
        action for action in description.entities if isinstance(action, OpaqueFunction)
    ]
    assert len(opaque) == 1
    context = LaunchContext()
    context.launch_configurations["data_dir"] = str(PACKAGE.parent / "ad_data")
    context.launch_configurations["path_file"] = (
        "path/2026_molit_comp_global_path.txt"
    )
    context.launch_configurations["route_corridor_file"] = "map/route_corridor.json"
    context.launch_configurations["config_file"] = str(
        PACKAGE / "config" / "highway_merge_gap_risk.yaml"
    )
    nodes = opaque[0].execute(context)
    assert len(nodes) == 1
    assert isinstance(nodes[0], Node)
    assert nodes[0]._Node__node_executable == "ad_highway_merge_gap_risk_node"


def test_planner_launch_opt_in_defaults_off(monkeypatch):
    module = _load_module("planner_launch", "launch/planner.launch.py")
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda package: str(PACKAGE.parent / package),
    )
    arguments = {
        action.name: action
        for action in module.generate_launch_description().entities
        if isinstance(action, DeclareLaunchArgument)
    }
    assert "highway_merge_gap_risk" in arguments
    assert arguments["highway_merge_gap_risk"].default_value[0].text == "false"

    context = LaunchContext()
    context.launch_configurations["config_file"] = str(
        PACKAGE / "config" / "planner.yaml"
    )
    context.launch_configurations["data_dir"] = str(PACKAGE.parent / "ad_data")
    context.launch_configurations["path_file"] = ""
    context.launch_configurations["route_corridor_file"] = ""

    context.launch_configurations["highway_merge_gap_risk"] = "false"
    assert module._create_highway_merge_gap_risk_node(context) is None

    context.launch_configurations["highway_merge_gap_risk"] = "true"
    node = module._create_highway_merge_gap_risk_node(context)
    assert node is not None
    assert node._Node__node_executable == "ad_highway_merge_gap_risk_node"
