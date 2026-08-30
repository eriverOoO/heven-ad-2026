from pathlib import Path
import importlib.util

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node
import yaml


PACKAGE = Path(__file__).resolve().parents[1]


def _load_module(name, relative):
    path = PACKAGE / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_is_physical_and_uses_canonical_topics():
    parameters = yaml.safe_load(
        (PACKAGE / "config" / "cut_in_response.yaml").read_text(encoding="utf-8")
    )["ad_cut_in_response"]["ros__parameters"]
    assert parameters["topics.cut_in_risks"] == "/ad/planning/cut_in_risks"
    assert parameters["topics.odometry"] == "/ad/localization/odometry"
    assert parameters["topics.output"] == "/ad/planning/cut_in_response"
    # No actuator or opaque-score parameters.
    forbidden = (
        "brake",
        "throttle",
        "steer",
        "pedal",
        "dbw",
        "risk_score",
        "score_weight",
        "risk_low",
        "risk_medium",
        "risk_high",
    )
    assert not any(
        token in key.lower() for key in parameters for token in forbidden
    )
    # Deceleration parameters are ordered and physical.
    assert (
        parameters["maximum_deceleration_mps2"]
        >= parameters["comfortable_deceleration_mps2"]
        > 0.0
    )
    assert parameters["longitudinal_standoff_m"] > 0.0
    assert parameters["minimum_response_speed_mps"] >= 0.0
    # The policy is stateless: no release / latch parameter.
    assert "release_frames" not in parameters


def test_standalone_launch_starts_only_the_response_node(monkeypatch):
    module = _load_module("cut_in_response_launch", "launch/cut_in_response.launch.py")
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
        action for action in description.entities if isinstance(action, OpaqueFunction)
    ]
    assert len(opaque) == 1
    context = LaunchContext()
    context.launch_configurations["config_file"] = str(
        PACKAGE / "config" / "cut_in_response.yaml"
    )
    nodes = opaque[0].execute(context)
    assert len(nodes) == 1
    assert isinstance(nodes[0], Node)
    assert nodes[0]._Node__node_executable == "ad_cut_in_response_node"


def test_planner_launch_opt_in_defaults_off_and_pulls_in_cut_in_risk(monkeypatch):
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
    assert "cut_in_response" in arguments
    assert arguments["cut_in_response"].default_value[0].text == "false"

    context = LaunchContext()
    context.launch_configurations["cut_in_response"] = "true"
    assert module._create_cut_in_response_node(context) is not None

    context.launch_configurations["cut_in_response"] = "false"
    assert module._create_cut_in_response_node(context) is None

    context.launch_configurations["cut_in_response"] = "bogus"
    try:
        module._create_cut_in_response_node(context)
        raise AssertionError("expected RuntimeError for invalid cut_in_response")
    except RuntimeError:
        pass
