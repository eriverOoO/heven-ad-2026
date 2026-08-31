from pathlib import Path
import importlib.util

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node
import yaml


PACKAGE = Path(__file__).resolve().parents[1]


def _load_module():
    path = PACKAGE / "launch" / "roundabout_gap_response.launch.py"
    spec = importlib.util.spec_from_file_location("roundabout_gap_response_launch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_is_small_physical_and_has_no_actuator_or_speed_request():
    parameters = yaml.safe_load(
        (PACKAGE / "config" / "roundabout_gap_response.yaml").read_text(encoding="utf-8")
    )["ad_roundabout_gap_response"]["ros__parameters"]
    assert parameters["topics.roundabout_gap_risks"] == "/ad/planning/roundabout_gap_risks"
    assert parameters["topics.output"] == "/ad/planning/roundabout_gap_response"
    assert parameters["minimum_release_gap_s"] == 2.0
    assert parameters["comfortable_deceleration_mps2"] == 1.8
    assert parameters["entry_standoff_m"] == 6.0
    forbidden = ("brake", "throttle", "steer", "ctrl", "cmd_vel", "requested_speed")
    assert not any(token in key.lower() for key in parameters for token in forbidden)
    assert not any("hyster" in key.lower() for key in parameters)


def test_standalone_launch_starts_only_response_node(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(
        module, "get_package_share_directory", lambda package: str(PACKAGE.parent / package)
    )
    description = module.generate_launch_description()
    arguments = {
        action.name for action in description.entities
        if isinstance(action, DeclareLaunchArgument)
    }
    assert arguments == {"config_file"}
    opaque = [action for action in description.entities if isinstance(action, OpaqueFunction)]
    assert len(opaque) == 1
    context = LaunchContext()
    context.launch_configurations["config_file"] = str(
        PACKAGE / "config" / "roundabout_gap_response.yaml"
    )
    nodes = opaque[0].execute(context)
    assert len(nodes) == 1
    assert isinstance(nodes[0], Node)
    assert nodes[0]._Node__node_executable == "ad_roundabout_gap_response_node"
