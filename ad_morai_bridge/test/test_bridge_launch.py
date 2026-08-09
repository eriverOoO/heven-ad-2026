from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.utilities import perform_substitutions
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters
import pytest


PACKAGE = Path(__file__).resolve().parents[1]
LAUNCH_FILE = PACKAGE / "launch" / "bridge.launch.py"


def load_launch_module():
    spec = spec_from_file_location("bridge_launch", LAUNCH_FILE)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def velodyne_context(
    enabled: str,
    *,
    platform_profile: str = "morai",
    point_timing_mode: str = "auto",
    measurement_compatibility_enabled: str = "false",
) -> LaunchContext:
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "namespace": "must_be_ignored",
            "enable_velodyne_points": enabled,
            "platform_profile": platform_profile,
            "velodyne_point_timing_mode": point_timing_mode,
            "velodyne_organize_cloud": "false",
            "measurement_compatibility_enabled": (
                measurement_compatibility_enabled
            ),
            "measurement_compatibility_config": str(
                PACKAGE / "config" / "measurement_compatibility.yaml"
            ),
        }
    )
    return context


def node_remappings(node: Node, context: LaunchContext) -> dict[str, str]:
    return {
        perform_substitutions(context, source): perform_substitutions(
            context, destination
        )
        for source, destination in node._Node__remappings
    }


def node_parameters(node: Node, context: LaunchContext) -> dict[str, object]:
    result = {}
    for parameters in evaluate_parameters(context, node._Node__parameters):
        if isinstance(parameters, dict):
            result.update(parameters)
    return result


def test_generate_launch_description_smoke():
    description = load_launch_module().generate_launch_description()
    assert isinstance(description, LaunchDescription)
    nodes = [
        action for action in description.entities if isinstance(action, Node)
    ]
    declared_arguments = {
        action.name
        for action in description.entities
        if isinstance(action, DeclareLaunchArgument)
    }
    assert nodes
    assert "namespace" not in declared_arguments
    assert all(node._Node__node_namespace is None for node in nodes)


def test_control_stays_disabled_by_default_and_wires_the_launch_argument():
    from launch.actions import DeclareLaunchArgument
    from launch.utilities import perform_substitutions

    description = load_launch_module().generate_launch_description()
    context = LaunchContext()
    defaults = {
        action.name: perform_substitutions(context, action.default_value)
        for action in description.entities
        if isinstance(action, DeclareLaunchArgument)
    }
    # Safety default: launching the bridge must never arm control by itself,
    # and the default profile is the competition allowlist.
    assert defaults["control_enabled"] == "false"
    assert defaults["enable_traffic_light_camera"] == "false"
    assert defaults["config"].endswith("competition.yaml")
    assert defaults["platform_profile"] == "morai"
    assert defaults["velodyne_point_timing_mode"] == "auto"
    assert defaults["measurement_compatibility_enabled"] == "false"
    assert defaults["measurement_compatibility_config"].endswith(
        "measurement_compatibility.yaml"
    )

    # The control.enabled override is the winning parameter layer (it comes
    # after the config file), so it must be wired to the launch argument.
    bridge = next(
        action for action in description.entities if isinstance(action, Node)
    )
    override = bridge._Node__parameters[-1]
    names = {
        perform_substitutions(context, list(name)) for name in override
    }
    assert "control.enabled" in names
    assert "camera_traffic_light.enabled" in names


def test_velodyne_chain_is_absent_when_point_conversion_is_disabled():
    module = load_launch_module()

    assert module._launch_velodyne_chain(velodyne_context("false")) == []


def test_measurement_compatibility_is_absent_unless_explicitly_enabled():
    module = load_launch_module()

    assert module._launch_measurement_compatibility(
        velodyne_context("false")
    ) == []


def test_morai_profile_can_launch_the_optional_measurement_compatibility_node():
    module = load_launch_module()
    context = velodyne_context(
        "false",
        platform_profile="morai",
        measurement_compatibility_enabled="true",
    )

    actions = module._launch_measurement_compatibility(context)

    assert len(actions) == 1
    node = actions[0]
    assert isinstance(node, Node)
    assert node.node_package == "ad_morai_bridge"
    assert node.node_executable == "ad_measurement_compatibility"
    assert node._Node__node_name == "ad_measurement_compatibility"
    parameter_files = [
        item
        for item in evaluate_parameters(context, node._Node__parameters)
        if isinstance(item, Path)
    ]
    assert parameter_files == [
        PACKAGE / "config" / "measurement_compatibility.yaml"
    ]


def test_real_hardware_rejects_measurement_compatibility_enablement():
    module = load_launch_module()
    context = velodyne_context(
        "false",
        platform_profile="real_hardware",
        measurement_compatibility_enabled="true",
    )

    with pytest.raises(ValueError, match="MORAI-only"):
        module._launch_measurement_compatibility(context)


def test_morai_chain_zeroes_packet_and_final_point_times_before_publication():
    module = load_launch_module()
    context = velodyne_context("true", platform_profile="morai")

    actions = module._launch_velodyne_chain(context)
    assert len(actions) == 3
    assert all(isinstance(action, Node) for action in actions)
    assert {action.node_executable for action in actions} == {
        "ad_velodyne_adapter",
        "velodyne_transform_node",
        "ad_point_time_zero_boundary",
    }
    assert all(action._Node__node_namespace is None for action in actions)

    adapter = next(
        action
        for action in actions
        if action.node_executable == "ad_velodyne_adapter"
    )
    transform = next(
        action
        for action in actions
        if action.node_executable == "velodyne_transform_node"
    )
    assert node_parameters(adapter, context)["point_timing_mode"] == "zero"
    assert node_remappings(transform, context) == {
        "velodyne_packets": "/ad/sensors/lidar/packets",
        "velodyne_points": "/ad/sensors/lidar/points_with_synthetic_time",
    }


def test_disabled_measurement_compatibility_does_not_change_morai_lidar_chain():
    module = load_launch_module()
    context = velodyne_context(
        "true",
        platform_profile="morai",
        measurement_compatibility_enabled="false",
    )

    lidar_actions = module._launch_velodyne_chain(context)

    assert module._launch_measurement_compatibility(context) == []
    assert {action.node_executable for action in lidar_actions} == {
        "ad_velodyne_adapter",
        "velodyne_transform_node",
        "ad_point_time_zero_boundary",
    }


def test_real_hardware_chain_keeps_stock_rolling_point_times():
    module = load_launch_module()
    context = velodyne_context(
        "true", platform_profile="real_hardware"
    )

    actions = module._launch_velodyne_chain(context)
    assert len(actions) == 2
    assert all(isinstance(action, Node) for action in actions)
    assert {action.node_executable for action in actions} == {
        "ad_velodyne_adapter",
        "velodyne_transform_node",
    }
    assert all(action._Node__node_namespace is None for action in actions)

    adapter = next(
        action
        for action in actions
        if action.node_executable == "ad_velodyne_adapter"
    )
    transform = next(
        action
        for action in actions
        if action.node_executable == "velodyne_transform_node"
    )
    assert node_parameters(adapter, context)["point_timing_mode"] == "azimuth"
    assert node_remappings(transform, context) == {
        "velodyne_packets": "/ad/sensors/lidar/packets",
        "velodyne_points": "/ad/sensors/lidar/points",
    }


@pytest.mark.parametrize("profile", ["", "sim", "hardware", "MORAI"])
def test_unknown_platform_profile_is_rejected_even_when_points_are_disabled(
    profile,
):
    module = load_launch_module()

    with pytest.raises(ValueError, match="platform_profile"):
        module._launch_velodyne_chain(
            velodyne_context("false", platform_profile=profile)
        )


@pytest.mark.parametrize(
    ("profile", "requested"),
    [("morai", "azimuth"), ("real_hardware", "zero")],
)
def test_profile_cannot_be_overridden_with_incompatible_point_timing(
    profile, requested
):
    module = load_launch_module()

    with pytest.raises(ValueError, match="incompatible"):
        module._launch_velodyne_chain(
            velodyne_context(
                "true",
                platform_profile=profile,
                point_timing_mode=requested,
            )
        )


@pytest.mark.parametrize("mode", ["", "rolling", "legacy", "arrival"])
def test_unknown_point_timing_request_is_rejected(mode):
    module = load_launch_module()

    with pytest.raises(ValueError, match="velodyne_point_timing_mode"):
        module._launch_velodyne_chain(
            velodyne_context("true", point_timing_mode=mode)
        )
