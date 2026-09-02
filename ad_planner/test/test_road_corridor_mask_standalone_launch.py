from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import hashlib

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.utilities import perform_substitutions
from launch_ros.actions import Node
import pytest
import yaml


PACKAGE = Path(__file__).resolve().parents[1]
LAUNCH = PACKAGE / "launch" / "road_corridor_mask.launch.py"


def load_launch_module():
    spec = spec_from_file_location("ad_road_corridor_mask_standalone", LAUNCH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _node_identity(node):
    return (
        node._Node__package,
        node._Node__node_executable,
        node._Node__node_name,
    )


def _parameter_mapping(parameter, context):
    return {
        perform_substitutions(context, names): (
            yaml.safe_load(perform_substitutions(context, value))
            if isinstance(value, tuple)
            else value
        )
        for names, value in parameter.items()
    }


def test_declares_data_dir_from_ad_data_dir_and_committed_defaults(
    monkeypatch,
):
    monkeypatch.delenv("AD_DATA_DIR", raising=False)
    module = load_launch_module()
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda _package: str(PACKAGE),
    )
    description = module.generate_launch_description()
    arguments = {
        entity.name: entity.default_value
        for entity in description.entities
        if isinstance(entity, DeclareLaunchArgument)
    }
    assert set(arguments) == {
        "data_dir",
        "config_file",
        "path_file",
        "route_corridor_file",
    }
    context = LaunchContext()
    assert perform_substitutions(context, arguments["data_dir"]) == ""
    assert (
        perform_substitutions(context, arguments["path_file"])
        == "path/2026_molit_comp_global_path.txt"
    )
    assert (
        perform_substitutions(context, arguments["route_corridor_file"])
        == "map/route_corridor.json"
    )
    assert perform_substitutions(
        context, arguments["config_file"]
    ).endswith("ad_planner/config/road_corridor_mask.yaml")


def test_starts_exactly_the_existing_mask_node_with_a_real_global_path_sha256(
    tmp_path,
):
    module = load_launch_module()
    path_file = tmp_path / "path" / "route.txt"
    path_file.parent.mkdir()
    path_file.write_bytes(b"committed global path used by the mask node\n")

    config_file = tmp_path / "road_corridor_mask.yaml"
    config_file.write_text(
        yaml.safe_dump(
            {
                "ad_road_corridor_mask": {
                    "ros__parameters": {
                        "data_dir": "",
                        "route_corridor_file": "map/route_corridor.json",
                        "route_corridor.expected_global_path_sha256": "",
                        "base_frame": "base_link",
                        "topics.drivable_mask": "/ad/planning/drivable_mask",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    context = LaunchContext()
    context.launch_configurations.update(
        {
            "data_dir": str(tmp_path),
            "path_file": "path/route.txt",
            "route_corridor_file": "map/route_corridor.json",
            "config_file": str(config_file),
        }
    )
    actions = module._create_node(context)

    assert len(actions) == 1
    node = actions[0]
    assert isinstance(node, Node)
    assert _node_identity(node) == (
        "ad_planner",
        "ad_road_corridor_mask_node",
        "ad_road_corridor_mask",
    )
    # A single merged dict - never a [config_file, overrides] pair - so the
    # process never has to reconcile two --params-file sources for the same
    # node (see the module docstring for the merge-race this avoids).
    assert len(node._Node__parameters) == 1
    merged = _parameter_mapping(node._Node__parameters[0], context)
    assert merged["base_frame"] == "base_link"
    assert merged["topics.drivable_mask"] == "/ad/planning/drivable_mask"
    assert merged["data_dir"] == str(tmp_path)
    assert merged["route_corridor_file"] == "map/route_corridor.json"
    assert merged["route_corridor.expected_global_path_sha256"] == (
        hashlib.sha256(path_file.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    "missing",
    ["data_dir", "path_file", "route_corridor_file"],
)
def test_missing_required_argument_is_rejected(tmp_path, missing):
    module = load_launch_module()
    values = {
        "data_dir": str(tmp_path),
        "path_file": "path/route.txt",
        "route_corridor_file": "map/route_corridor.json",
        "config_file": "/abs/road_corridor_mask.yaml",
    }
    values[missing] = ""
    context = LaunchContext()
    context.launch_configurations.update(values)
    with pytest.raises(RuntimeError, match="data_dir.*path_file.*route_corridor_file"):
        module._create_node(context)


def test_config_file_missing_node_section_is_rejected(tmp_path):
    module = load_launch_module()
    path_file = tmp_path / "path" / "route.txt"
    path_file.parent.mkdir()
    path_file.write_bytes(b"committed global path\n")
    config_file = tmp_path / "road_corridor_mask.yaml"
    config_file.write_text(
        yaml.safe_dump({"some_other_node": {"ros__parameters": {}}}),
        encoding="utf-8",
    )
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "data_dir": str(tmp_path),
            "path_file": "path/route.txt",
            "route_corridor_file": "map/route_corridor.json",
            "config_file": str(config_file),
        }
    )
    with pytest.raises(RuntimeError, match="ad_road_corridor_mask.ros__parameters"):
        module._create_node(context)


def test_missing_global_path_file_is_rejected(tmp_path):
    module = load_launch_module()
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "data_dir": str(tmp_path),
            "path_file": "path/does_not_exist.txt",
            "route_corridor_file": "map/route_corridor.json",
            "config_file": "/abs/road_corridor_mask.yaml",
        }
    )
    with pytest.raises(RuntimeError, match="requires global path"):
        module._create_node(context)
