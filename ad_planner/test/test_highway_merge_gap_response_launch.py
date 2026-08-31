from pathlib import Path
import importlib.util

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node
import yaml


PACKAGE = Path(__file__).resolve().parents[1]


def _load_module():
    path = PACKAGE / "launch" / "highway_merge_gap_response.launch.py"
    spec = importlib.util.spec_from_file_location(
        "highway_merge_gap_response_launch", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_is_small_physical_and_has_no_actuator_or_speed_request():
    parameters = yaml.safe_load(
        (PACKAGE / "config" / "highway_merge_gap_response.yaml").read_text(
            encoding="utf-8"
        )
    )["ad_highway_merge_gap_response"]["ros__parameters"]
    assert (
        parameters["topics.highway_merge_gap_risks"]
        == "/ad/planning/highway_merge_gap_risks"
    )
    assert parameters["topics.output"] == "/ad/planning/highway_merge_gap_response"
    assert parameters["expected_frame_id"] == "map"
    assert parameters["merge_zone_id"] == "kcity_highway_onramp"
    assert parameters["minimum_front_time_headway_s"] == 1.5
    assert parameters["minimum_rear_time_headway_s"] == 2.0
    assert parameters["minimum_rear_closing_time_s"] == 3.0
    assert parameters["minimum_predicted_route_gap_m"] == 6.0
    assert parameters["merge_standoff_m"] == 6.0
    assert parameters["comfortable_deceleration_mps2"] == 1.8
    # The applicability bound must not exceed the risk node's own approach bound.
    assert parameters["maximum_approach_distance_m"] <= 400.0
    forbidden = (
        "brake",
        "throttle",
        "steer",
        "ctrl",
        "cmd_vel",
        "requested_speed",
        "target_speed",
        "lane_change",
        "acceleration",
        "risk_score",
        "weight",
        "score",
        "hyster",
    )
    assert not any(
        token in key.lower() for key in parameters for token in forbidden
    )


def test_standalone_launch_starts_only_response_node(monkeypatch):
    module = _load_module()
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
    assert arguments == {"config_file"}
    opaque = [
        action
        for action in description.entities
        if isinstance(action, OpaqueFunction)
    ]
    assert len(opaque) == 1
    context = LaunchContext()
    context.launch_configurations["config_file"] = str(
        PACKAGE / "config" / "highway_merge_gap_response.yaml"
    )
    nodes = opaque[0].execute(context)
    assert len(nodes) == 1
    assert isinstance(nodes[0], Node)
    assert nodes[0]._Node__node_executable == "ad_highway_merge_gap_response_node"
