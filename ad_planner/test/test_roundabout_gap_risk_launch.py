import importlib.util
import json
from pathlib import Path

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node
import yaml


PACKAGE = Path(__file__).resolve().parents[1]

_FORBIDDEN_TOKENS = (
    "safe_gap",
    "yield_gap",
    "go_gap",
    "emergency_gap",
    "accepted_gap",
    "gap_accepted",
    "safe_to_enter",
    "yield_required",
    "risk_score",
    "score_weight",
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
        (PACKAGE / "config" / "roundabout_gap_risk.yaml").read_text(encoding="utf-8")
    )["ad_roundabout_gap_risk"]["ros__parameters"]
    assert parameters["topics.dynamic_object_risks"] == "/ad/planning/dynamic_object_risks"
    assert parameters["topics.odometry"] == "/ad/localization/odometry"
    assert parameters["topics.output"] == "/ad/planning/roundabout_gap_risks"
    assert parameters["conflict_zone_id"] == "kcity_roundabout"
    assert parameters["ego_speed_epsilon_mps"] > 0.0
    assert parameters["maximum_ego_approach_distance_m"] > 0.0
    assert not any(
        token in key.lower()
        for key in parameters
        for token in _FORBIDDEN_TOKENS
    )


def test_conflict_geometry_is_well_formed_and_map_frame():
    document = json.loads(
        (PACKAGE / "config" / "roundabout_conflicts.json").read_text(encoding="utf-8")
    )
    assert document["schema_version"] == 1
    assert document["frame_id"] == "map"
    assert "provenance" in document
    zones = document["conflict_zones"]
    assert len(zones) >= 1
    zone = next(z for z in zones if z["conflict_zone_id"] == "kcity_roundabout")
    assert zone["route_s_enter_m"] < zone["route_s_exit_m"]
    polygon = zone["polygon_m"]
    assert len(polygon) >= 3
    for vertex in polygon:
        assert len(vertex) == 2
        assert all(isinstance(value, (int, float)) for value in vertex)
    # No policy field leaked into the geometry config.
    text = json.dumps(document).lower()
    assert not any(token in text for token in _FORBIDDEN_TOKENS)


def test_standalone_launch_starts_only_the_roundabout_node(monkeypatch):
    module = _load_module(
        "roundabout_gap_risk_launch", "launch/roundabout_gap_risk.launch.py"
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
    assert arguments == {"data_dir", "config_file", "path_file", "route_corridor_file"}
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
        PACKAGE / "config" / "roundabout_gap_risk.yaml"
    )
    nodes = opaque[0].execute(context)
    assert len(nodes) == 1
    assert isinstance(nodes[0], Node)
    assert nodes[0]._Node__node_executable == "ad_roundabout_gap_risk_node"


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
    assert "roundabout_gap_risk" in arguments
    assert arguments["roundabout_gap_risk"].default_value[0].text == "false"

    context = LaunchContext()
    context.launch_configurations["config_file"] = str(
        PACKAGE / "config" / "planner.yaml"
    )
    context.launch_configurations["data_dir"] = str(PACKAGE.parent / "ad_data")
    context.launch_configurations["path_file"] = ""
    context.launch_configurations["route_corridor_file"] = ""

    context.launch_configurations["roundabout_gap_risk"] = "false"
    assert module._create_roundabout_gap_risk_node(context) is None

    context.launch_configurations["roundabout_gap_risk"] = "true"
    node = module._create_roundabout_gap_risk_node(context)
    assert node is not None
    assert node._Node__node_executable == "ad_roundabout_gap_risk_node"
