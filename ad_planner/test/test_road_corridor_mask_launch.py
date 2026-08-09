from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import hashlib

from launch import LaunchContext
from launch.utilities import perform_substitutions
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile
import pytest
import yaml


PACKAGE = Path(__file__).resolve().parents[1]
REPOSITORY = PACKAGE.parent
LAUNCH = PACKAGE / "launch" / "planner.launch.py"
CONFIG = PACKAGE / "config" / "road_corridor_mask.yaml"


def _load_launch_module(name):
    spec = spec_from_file_location(name, LAUNCH)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _context(tmp_path, backend):
    planner = tmp_path / f"{backend}.yaml"
    planner.write_text(
        "ad_planner:\n  ros__parameters:\n"
        f"    local_motion.backend: {backend}\n"
        "    path_file: path/route.txt\n"
        "    route_corridor_file: map/active-corridor.json\n",
        encoding="utf-8",
    )
    route = tmp_path / "path" / "route.txt"
    route.parent.mkdir()
    route.write_bytes(b"route used by checksum contract\n")
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "config_file": str(planner),
            "data_dir": str(tmp_path),
            "path_file": "",
            "route_corridor_file": "",
        }
    )
    return context, route


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


def test_mask_config_matches_lidar_grid_and_strict_topics():
    parameters = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))[
        "ad_road_corridor_mask"
    ]["ros__parameters"]

    assert parameters == {
        "data_dir": "",
        "route_corridor_file": "map/route_corridor.json",
        "route_corridor.expected_global_path_sha256": "",
        "base_frame": "base_link",
        "transform_timeout_sec": 0.05,
        "deduplication_cache_size": 64,
        "grid.minimum_x_m": -4.0,
        "grid.maximum_x_m": 100.0,
        "grid.minimum_y_m": -10.0,
        "grid.maximum_y_m": 10.0,
        "grid.resolution_m": 0.1,
        "topics.lidar_points": "/ad/sensors/lidar/points",
        "topics.predicted_objects": "/ad/perception/objects/predicted",
        "topics.drivable_mask": "/ad/planning/drivable_mask",
    }


@pytest.mark.parametrize(
    "backend", ["dwa", "frenet_lattice", "mppi_nav2"]
)
def test_every_planner_backend_launches_one_separate_mask_node(
    monkeypatch, tmp_path, backend
):
    module = _load_launch_module(f"road_mask_{backend}")
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda package: str(REPOSITORY / package),
    )
    context, route = _context(tmp_path, backend)

    actions = module._create_planner_actions(context)
    mask_nodes = [
        action
        for action in actions
        if isinstance(action, Node)
        and action._Node__node_executable == "ad_road_corridor_mask_node"
    ]

    assert len(mask_nodes) == 1
    mask = mask_nodes[0]
    assert _node_identity(mask) == (
        "ad_planner",
        "ad_road_corridor_mask_node",
        "ad_road_corridor_mask",
    )
    assert mask._Node__node_namespace is None
    assert len(mask._Node__parameters) == 2
    assert isinstance(mask._Node__parameters[0], ParameterFile)
    assert perform_substitutions(
        context, mask._Node__parameters[0].param_file
    ).endswith(
        "ad_planner/config/road_corridor_mask.yaml"
    )
    assert _parameter_mapping(mask._Node__parameters[1], context) == {
        "data_dir": str(tmp_path),
        "route_corridor_file": "map/active-corridor.json",
        "route_corridor.expected_global_path_sha256": hashlib.sha256(
            route.read_bytes()
        ).hexdigest(),
    }
