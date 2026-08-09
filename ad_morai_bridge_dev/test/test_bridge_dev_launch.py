from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


LAUNCH = (
    Path(__file__).resolve().parents[1] / "launch" / "bridge_dev.launch.py"
)


def test_development_launch_composes_competition_bridge_and_dev_nodes():
    spec = spec_from_file_location("bridge_dev_launch", LAUNCH)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    description = module.generate_launch_description()
    assert isinstance(description, LaunchDescription)

    includes = [
        action
        for action in description.entities
        if isinstance(action, IncludeLaunchDescription)
    ]
    nodes = [
        action for action in description.entities if isinstance(action, Node)
    ]
    declared_arguments = {
        action.name
        for action in description.entities
        if isinstance(action, DeclareLaunchArgument)
    }
    # The competition bridge is included; the dev bridge node and the
    # (condition-gated) scenario reset node are started directly.
    assert len(includes) == 1
    assert len(nodes) == 2
    assert "namespace" not in declared_arguments
    assert "namespace" not in dict(includes[0].launch_arguments)
    assert "platform_profile" in declared_arguments
    forwarded = dict(includes[0].launch_arguments)
    assert isinstance(forwarded["platform_profile"], LaunchConfiguration)
    context = LaunchContext()
    context.launch_configurations["platform_profile"] = "real_hardware"
    assert forwarded["platform_profile"].perform(context) == "real_hardware"
    assert all(node._Node__node_namespace is None for node in nodes)
