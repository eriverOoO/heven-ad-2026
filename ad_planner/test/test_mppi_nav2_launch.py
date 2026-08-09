from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import re
import stat
import xml.etree.ElementTree as ET

from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_prefix,
)
from launch import LaunchContext
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.utilities import perform_substitutions
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile
import pytest
import yaml
from yaml.constructor import ConstructorError


PACKAGE = Path(__file__).resolve().parents[1]
REPOSITORY = PACKAGE.parent
CONFIG = PACKAGE / "config" / "local_planning" / "mppi_nav2.yaml"
LAUNCH = PACKAGE / "launch" / "mppi_nav2.launch.py"
EXPECTED_REQUIREMENTS = [
    "AD_PLANNER_ENABLE_NAV2_MPPI=ON",
    "nav2_controller",
    "nav2_lifecycle_manager",
    "nav2_mppi_controller",
    "nav2_costmap_2d",
    "nav2_msgs",
]
NAV2_LEAVES = {
    "nav2_controller": ["lib/nav2_controller/controller_server"],
    "nav2_lifecycle_manager": [
        "lib/nav2_lifecycle_manager/lifecycle_manager"
    ],
    "nav2_mppi_controller": [
        "share/nav2_mppi_controller/mppic.xml",
        "share/nav2_mppi_controller/critics.xml",
    ],
    "nav2_costmap_2d": [],
    "nav2_msgs": [],
}
BACKEND_PARAMETERS = {
    "mppi_nav2.cmd_vel_topic": "/ad/planner/mppi/cmd_vel",
    "mppi_nav2.command_timeout_s": 0.20,
    "mppi_nav2.path_refresh_period_s": 0.25,
    "mppi_nav2.diagnostic_rollout_dt_s": 0.10,
    "mppi_nav2.diagnostic_rollout_horizon_s": 3.0,
    "mppi_nav2.wheelbase_m": 3.0,
    "mppi_nav2.maximum_road_wheel_angle_rad": 0.588,
    "mppi_nav2.steering_rate_limit_rad_s": 0.35,
    "mppi_nav2.near_zero_speed_mps": 0.05,
}


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate key: {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _load_unique_yaml(path):
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.load(stream, Loader=UniqueKeyLoader)


def _load_launch_module(path, name):
    spec = spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mppi_context(**values):
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "config_file": str(CONFIG),
            "visualize": "false",
            **values,
        }
    )
    return context


def _node_identity(node):
    return node._Node__package, node._Node__node_executable, node._Node__node_name


def _node_remappings(node, context):
    return {
        perform_substitutions(context, source): perform_substitutions(
            context, destination
        )
        for source, destination in node._Node__remappings
    }


def _parameter_value(value, context):
    if hasattr(value, "evaluate"):
        return value.evaluate(context)
    if hasattr(value, "perform"):
        return yaml.safe_load(value.perform(context))
    if isinstance(value, list):
        return [_parameter_value(item, context) for item in value]
    if isinstance(value, tuple):
        if len(value) == 1 and isinstance(value[0], list):
            return _parameter_value(value[0], context)
        return yaml.safe_load(perform_substitutions(context, value))
    return value


def _parameter_mapping(mapping, context):
    return {
        perform_substitutions(context, name): _parameter_value(value, context)
        for name, value in mapping.items()
    }


def _write_manifest(prefix, package, version="1.1.20"):
    share = prefix / "share" / package
    share.mkdir(parents=True, exist_ok=True)
    (share / "package.xml").write_text(
        "<package format=\"3\">"
        f"<name>{package}</name><version>{version}</version>"
        "<description>fixture</description>"
        "<maintainer email=\"fixture@example.com\">Fixture</maintainer>"
        "<license>Apache-2.0</license></package>",
        encoding="utf-8",
    )


def _make_runtime_tree(tmp_path):
    planner_prefix = tmp_path / "install" / "ad_planner"
    proxy = (
        planner_prefix
        / "lib"
        / "ad_planner"
        / "ad_planner_mppi_follow_path_proxy"
    )
    proxy.parent.mkdir(parents=True)
    proxy_target = (
        tmp_path
        / "build"
        / "ad_planner"
        / "ad_planner_mppi_follow_path_proxy"
    )
    proxy_target.parent.mkdir(parents=True)
    (proxy_target.parent / "CMakeCache.txt").write_text(
        "CMAKE_PROJECT_NAME:STATIC=ad_planner\n"
        "AD_PLANNER_ENABLE_NAV2_MPPI:BOOL=ON\n",
        encoding="utf-8",
    )
    proxy_target.write_bytes(b"fixture")
    proxy_target.chmod(0o755)
    proxy.symlink_to(proxy_target)

    nav2_prefix = tmp_path / "nav2"
    for package, leaves in NAV2_LEAVES.items():
        _write_manifest(nav2_prefix, package)
        for relative in leaves:
            leaf = nav2_prefix / relative
            leaf.parent.mkdir(parents=True, exist_ok=True)
            leaf.write_bytes(b"fixture")
            if relative.startswith("lib/"):
                leaf.chmod(0o755)

    def prefix(package):
        if package == "ad_planner":
            return str(planner_prefix)
        if package in NAV2_LEAVES:
            return str(nav2_prefix)
        raise PackageNotFoundError(package)

    return planner_prefix, nav2_prefix, prefix


def test_mppi_yaml_preserves_backend_parameters_and_has_exact_three_roots():
    document = _load_unique_yaml(CONFIG)
    assert set(document) == {"ad_planner", "controller_server", "local_costmap"}
    assert document["ad_planner"] == {"ros__parameters": BACKEND_PARAMETERS}


def test_controller_server_uses_exact_humble_mppi_contract():
    parameters = _load_unique_yaml(CONFIG)["controller_server"]["ros__parameters"]
    assert parameters["controller_frequency"] == 20.0
    assert parameters["odom_topic"] == "/ad/localization/odometry"
    assert parameters["controller_plugins"] == ["FollowPath"]

    follow_path = parameters["FollowPath"]
    assert follow_path["plugin"] == "nav2_mppi_controller::MPPIController"
    assert {
        name: follow_path[name]
        for name in (
            "time_steps",
            "model_dt",
            "batch_size",
            "iteration_count",
            "motion_model",
            "vx_min",
            "vx_max",
            "visualize",
        )
    } == {
        "time_steps": 60,
        "model_dt": 0.05,
        "batch_size": 1000,
        "iteration_count": 1,
        "motion_model": "Ackermann",
        "vx_min": 0.0,
        "vx_max": 6.1,
        "visualize": False,
    }
    assert follow_path["critics"] == [
        "ConstraintCritic",
        "CostCritic",
        "GoalCritic",
        "GoalAngleCritic",
        "PathAlignCritic",
        "PathFollowCritic",
        "PathAngleCritic",
        "PreferForwardCritic",
    ]
    assert follow_path["AckermannConstraints"] == {"min_turning_r": 4.50}
    assert follow_path["CostCritic"] == {
        "enabled": True,
        "cost_power": 1,
        "cost_weight": 3.81,
        "critical_cost": 300.0,
        "consider_footprint": True,
        "collision_cost": 1000000.0,
        "near_goal_distance": 1.0,
    }
    assert not any(
        isinstance(value, dict) and "plugin" in value
        for name, value in follow_path.items()
        if name.endswith("Critic")
    )


def test_local_costmap_uses_only_volatile_static_layer_and_exact_footprint():
    parameters = _load_unique_yaml(CONFIG)["local_costmap"]["local_costmap"][
        "ros__parameters"
    ]
    assert {
        name: parameters[name]
        for name in (
            "global_frame",
            "robot_base_frame",
            "rolling_window",
            "width",
            "height",
            "resolution",
            "track_unknown_space",
            "trinary_costmap",
            "lethal_cost_threshold",
            "footprint",
            "plugins",
        )
    } == {
        "global_frame": "odom",
        "robot_base_frame": "base_link",
        "rolling_window": True,
        "width": 54,
        "height": 54,
        "resolution": 0.1,
        "track_unknown_space": True,
        "trinary_costmap": False,
        "lethal_cost_threshold": 100,
        "footprint": (
            "[[-0.790, -0.945], [-0.790, 0.945], "
            "[3.845, 0.945], [3.845, -0.945]]"
        ),
        "plugins": ["static_layer"],
    }
    assert isinstance(parameters["footprint"], str)
    assert parameters["static_layer"] == {
        "plugin": "nav2_costmap_2d::StaticLayer",
        "map_topic": "/ad/planner/mppi/occupancy_grid_odom",
        "map_subscribe_transient_local": False,
        "subscribe_to_updates": False,
    }

    serialized = CONFIG.read_text(encoding="utf-8")
    for forbidden in ("InflationLayer", "ObstaclesCritic", "map_server"):
        assert forbidden not in serialized


def test_test_loader_rejects_malformed_and_duplicate_key_yaml(tmp_path):
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("controller_server: [\n", encoding="utf-8")
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        "controller_server: {}\ncontroller_server: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(yaml.YAMLError):
        _load_unique_yaml(malformed)
    with pytest.raises(ConstructorError, match="duplicate key"):
        _load_unique_yaml(duplicate)


def test_installed_nav2_packages_are_exact_1_1_20_with_regular_leaves():
    for package, leaves in NAV2_LEAVES.items():
        prefix = Path(get_package_prefix(package))
        manifest = prefix / "share" / package / "package.xml"
        assert stat.S_ISREG(manifest.lstat().st_mode)
        assert ET.parse(manifest).getroot().findtext("version") == "1.1.20"
        for relative in leaves:
            leaf = prefix / relative
            assert stat.S_ISREG(leaf.lstat().st_mode), leaf


def test_checker_accepts_valid_symlink_install_and_exact_runtime_tree(tmp_path):
    module = _load_launch_module(LAUNCH, "mppi_nav2_launch_valid")
    _, _, prefix = _make_runtime_tree(tmp_path)
    assert module._missing_runtime_requirements(prefix_resolver=prefix) == []


def test_checker_aggregates_every_missing_wrong_version_or_leaf_requirement(
    tmp_path,
):
    module = _load_launch_module(LAUNCH, "mppi_nav2_launch_invalid")
    planner_prefix, nav2_prefix, prefix = _make_runtime_tree(tmp_path)
    (
        planner_prefix
        / "lib"
        / "ad_planner"
        / "ad_planner_mppi_follow_path_proxy"
    ).unlink()
    _write_manifest(nav2_prefix, "nav2_controller", version="1.1.19")
    (nav2_prefix / NAV2_LEAVES["nav2_lifecycle_manager"][0]).unlink()
    (nav2_prefix / "share" / "nav2_mppi_controller" / "mppic.xml").unlink()
    _write_manifest(nav2_prefix, "nav2_costmap_2d", version="1.2.0")
    (nav2_prefix / "share" / "nav2_msgs" / "package.xml").write_text(
        "<package>",
        encoding="utf-8",
    )

    assert module._missing_runtime_requirements(
        prefix_resolver=prefix
    ) == EXPECTED_REQUIREMENTS


def test_proxy_checker_rejects_broken_special_nonexec_and_escaped_symlinks(
    tmp_path,
):
    module = _load_launch_module(LAUNCH, "mppi_nav2_launch_proxy")
    planner_prefix, _, _ = _make_runtime_tree(tmp_path)
    proxy = (
        planner_prefix
        / "lib"
        / "ad_planner"
        / "ad_planner_mppi_follow_path_proxy"
    )
    target = proxy.resolve()

    proxy.unlink()
    proxy.symlink_to(tmp_path / "missing")
    assert not module._installed_proxy_is_usable(planner_prefix)

    special = tmp_path / "build" / "ad_planner" / "special"
    special.mkdir()
    proxy.unlink()
    proxy.symlink_to(special)
    assert not module._installed_proxy_is_usable(planner_prefix)

    proxy.unlink()
    target.chmod(0o644)
    proxy.symlink_to(target)
    assert not module._installed_proxy_is_usable(planner_prefix)

    proxy.unlink()
    proxy.symlink_to("/bin/sh")
    assert not module._installed_proxy_is_usable(planner_prefix)


def test_checker_rejects_empty_runtime_artifacts_and_off_symlink_build(
    tmp_path,
):
    module = _load_launch_module(LAUNCH, "mppi_nav2_launch_empty")
    planner_prefix, nav2_prefix, prefix = _make_runtime_tree(tmp_path)
    proxy = (
        planner_prefix
        / "lib"
        / "ad_planner"
        / "ad_planner_mppi_follow_path_proxy"
    )
    proxy_target = proxy.resolve()
    proxy_target.write_bytes(b"")
    assert not module._installed_proxy_is_usable(planner_prefix)

    proxy_target.write_bytes(b"fixture")
    cache = proxy_target.parent / "CMakeCache.txt"
    cache.write_text(
        "CMAKE_PROJECT_NAME:STATIC=ad_planner\n"
        "AD_PLANNER_ENABLE_NAV2_MPPI:BOOL=OFF\n",
        encoding="utf-8",
    )
    assert not module._installed_proxy_is_usable(planner_prefix)

    cache.write_text(
        "CMAKE_PROJECT_NAME:STATIC=ad_planner\n"
        "AD_PLANNER_ENABLE_NAV2_MPPI:BOOL=ON\n",
        encoding="utf-8",
    )
    (
        nav2_prefix
        / "share"
        / "nav2_mppi_controller"
        / "mppic.xml"
    ).write_bytes(b"")
    assert module._missing_runtime_requirements(
        prefix_resolver=prefix
    ) == ["nav2_mppi_controller"]


def test_runtime_failure_is_deterministic_and_occurs_before_node_creation(
    monkeypatch,
):
    module = _load_launch_module(LAUNCH, "mppi_nav2_launch_failure")
    monkeypatch.setattr(
        module,
        "_missing_runtime_requirements",
        lambda **_kwargs: EXPECTED_REQUIREMENTS,
    )
    monkeypatch.setattr(
        module,
        "Node",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("node created before runtime validation")
        ),
    )
    expected = (
        "Nav2 MPPI runtime requirements are not satisfied:\n"
        + "\n".join(f"- {requirement}" for requirement in EXPECTED_REQUIREMENTS)
    )
    with pytest.raises(RuntimeError, match=re.escape(expected)):
        module._create_mppi_nav2_actions(_mppi_context())


def test_package_launch_owns_exact_nodes_remaps_lifecycle_and_visualize_override(
    monkeypatch,
):
    module = _load_launch_module(LAUNCH, "mppi_nav2_launch_actions")
    monkeypatch.setattr(
        module, "_missing_runtime_requirements", lambda **_kwargs: []
    )

    description = module.generate_launch_description()
    arguments = {
        action.name: action
        for action in description.entities
        if isinstance(action, DeclareLaunchArgument)
    }
    assert set(arguments) == {"config_file", "visualize"}
    defaults = {
        name: perform_substitutions(LaunchContext(), action.default_value)
        for name, action in arguments.items()
    }
    assert defaults["visualize"] == "false"
    assert defaults["config_file"].endswith(
        "ad_planner/config/local_planning/mppi_nav2.yaml"
    )

    opaque = next(
        action
        for action in description.entities
        if isinstance(action, OpaqueFunction)
    )
    context = _mppi_context(visualize="true")
    actions = opaque.execute(context)
    assert len(actions) == 4
    assert all(isinstance(action, Node) for action in actions)
    assert [_node_identity(action) for action in actions] == [
        (
            "ad_planner",
            "ad_occupancy_grid_reprojector_node",
            "occupancy_grid_reprojector_mppi",
        ),
        ("nav2_controller", "controller_server", "controller_server"),
        (
            "nav2_lifecycle_manager",
            "lifecycle_manager",
            "lifecycle_manager_mppi",
        ),
        (
            "ad_planner",
            "ad_planner_mppi_follow_path_proxy",
            "mppi_follow_path_proxy",
        ),
    ]
    assert all(action._Node__node_namespace is None for action in actions)

    controller = actions[1]
    assert _node_remappings(controller, context) == {
        "cmd_vel": "/ad/planner/mppi/cmd_vel",
        "/trajectories": "/ad/viz/planner/mppi/trajectories",
        "transformed_global_plan": (
            "/ad/viz/planner/mppi/transformed_reference"
        ),
    }
    assert len(controller._Node__parameters) == 2
    assert isinstance(controller._Node__parameters[0], ParameterFile)
    rewritten_config = Path(perform_substitutions(
        context, controller._Node__parameters[0].param_file
    ))
    assert rewritten_config != CONFIG
    assert _load_unique_yaml(rewritten_config) == _load_unique_yaml(CONFIG)
    assert _parameter_mapping(
        controller._Node__parameters[1], context
    ) == {"FollowPath.visualize": True}

    lifecycle = actions[2]
    assert _parameter_mapping(lifecycle._Node__parameters[0], context) == {
        "autostart": True,
        "node_names": ["controller_server"],
    }

@pytest.mark.parametrize("backend", ["dwa", "frenet_lattice"])
def test_non_mppi_nav2_selection_keeps_planner_and_mask_outside_nav2(
    monkeypatch, tmp_path, backend
):
    module = _load_launch_module(
        PACKAGE / "launch" / "planner.launch.py",
        f"planner_launch_{backend}",
    )
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda package: str(REPOSITORY / package),
    )
    planner = tmp_path / f"{backend}.yaml"
    planner.write_text(
        "ad_planner:\n  ros__parameters:\n"
        f"    local_motion.backend: {backend}\n"
        "    path_file: path/route.txt\n",
        encoding="utf-8",
    )
    route = tmp_path / "path" / "route.txt"
    route.parent.mkdir()
    route.write_text("0 0\n1 0\n", encoding="utf-8")
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "config_file": str(planner),
            "data_dir": str(tmp_path),
        }
    )

    actions = module._create_planner_actions(context)
    assert len(actions) == 2
    assert all(isinstance(action, Node) for action in actions)
    assert [_node_identity(action) for action in actions] == [
        ("ad_planner", "ad_planner_node", "ad_planner"),
        (
            "ad_planner",
            "ad_road_corridor_mask_node",
            "ad_road_corridor_mask",
        ),
    ]


def test_mppi_nav2_selection_includes_package_runtime_before_planner(
    monkeypatch, tmp_path
):
    module = _load_launch_module(
        PACKAGE / "launch" / "planner.launch.py",
        "planner_launch_mppi_nav2",
    )
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda package: str(REPOSITORY / package),
    )
    planner = tmp_path / "mppi_nav2.yaml"
    planner.write_text(
        "ad_planner:\n  ros__parameters:\n"
        "    local_motion.backend: mppi_nav2\n"
        "    path_file: path/route.txt\n",
        encoding="utf-8",
    )
    route = tmp_path / "path" / "route.txt"
    route.parent.mkdir()
    route.write_text("0 0\n1 0\n", encoding="utf-8")
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "config_file": str(planner),
            "data_dir": str(tmp_path),
        }
    )

    actions = module._create_planner_actions(context)
    assert len(actions) == 3
    assert isinstance(actions[0], IncludeLaunchDescription)
    assert perform_substitutions(
        context,
        actions[0]
        .launch_description_source
        ._LaunchDescriptionSource__location,
    ).endswith("ad_planner/launch/mppi_nav2.launch.py")
    assert dict(actions[0].launch_arguments) == {
        "config_file": str(PACKAGE / "config" / "local_planning" / "mppi_nav2.yaml"),
    }
    assert isinstance(actions[1], Node)
    assert isinstance(actions[2], Node)
    assert _node_identity(actions[2]) == (
        "ad_planner",
        "ad_road_corridor_mask_node",
        "ad_road_corridor_mask",
    )


def test_launch_checker_has_no_shell_network_install_or_silent_fallback():
    source = LAUNCH.read_text(encoding="utf-8")
    for forbidden in (
        "subprocess",
        "os.system",
        "os.popen",
        "socket",
        "requests",
        "urllib",
        "apt",
        "fallback",
    ):
        assert forbidden not in source
    assert "optimal" not in source.lower()
