from pathlib import Path
import importlib.util

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.utilities import perform_substitutions
from launch_ros.actions import Node
import pytest
import yaml


PACKAGE = Path(__file__).resolve().parents[1]


def _load_module():
    path = PACKAGE / "launch" / "cut_in_risk.launch.py"
    spec = importlib.util.spec_from_file_location("cut_in_risk_launch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_is_geometry_only_and_uses_canonical_topics():
    parameters = yaml.safe_load(
        (PACKAGE / "config" / "cut_in_risk.yaml").read_text(encoding="utf-8")
    )["ad_cut_in_risk"]["ros__parameters"]
    assert parameters["topics.dynamic_object_risks"] == (
        "/ad/planning/dynamic_object_risks"
    )
    assert parameters["topics.route_mask"] == "/ad/planning/drivable_mask"
    assert parameters["topics.odometry"] == "/ad/localization/odometry"
    assert parameters["topics.output"] == "/ad/planning/cut_in_risks"
    assert parameters["risk_frame_id"] == "base_link"
    assert parameters["route_mask_frame_id"] == "base_link"
    forbidden = ("brake", "stop_required", "steer", "yield", "risk_score")
    assert not any(
        token in key.lower() for key in parameters for token in forbidden
    )


def test_standalone_launch_requires_and_hashes_active_route(monkeypatch, tmp_path):
    module = _load_module()
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda package: str(PACKAGE.parent / package),
    )
    description = module.generate_launch_description()
    arguments = {
        action.name: action
        for action in description.entities
        if isinstance(action, DeclareLaunchArgument)
    }
    assert set(arguments) == {
        "data_dir", "config_file", "path_file", "route_corridor_file"
    }
    opaque = [
        action for action in description.entities if isinstance(action, OpaqueFunction)
    ]
    assert len(opaque) == 1

    route = tmp_path / "active.txt"
    route.write_bytes(b"route\n")
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "data_dir": str(tmp_path),
            "config_file": str(PACKAGE / "config" / "cut_in_risk.yaml"),
            "path_file": "active.txt",
            "route_corridor_file": "map/route_corridor.json",
        }
    )
    nodes = opaque[0].execute(context)
    assert len(nodes) == 1
    assert isinstance(nodes[0], Node)
    assert nodes[0]._Node__node_executable == "ad_cut_in_risk_node"

    context.launch_configurations["path_file"] = "missing.txt"
    with pytest.raises(RuntimeError, match="requires global path"):
        opaque[0].execute(context)


def test_standalone_launch_defaults_are_repository_consistent(monkeypatch):
    module = _load_module()
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
    context = LaunchContext()
    assert perform_substitutions(
        context, arguments["path_file"].default_value
    ) == "path/2026_molit_comp_global_path.txt"
    assert perform_substitutions(
        context, arguments["route_corridor_file"].default_value
    ) == "map/route_corridor.json"
