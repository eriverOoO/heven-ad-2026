from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from launch import LaunchContext
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    LogInfo,
    RegisterEventHandler,
)
from launch.events import Shutdown
from launch.utilities import perform_substitutions
from launch_ros.actions import Node

from ad_bringup import bringup_stack
from ad_bringup.bringup_stack import registered_component_names
from ad_bringup.component_config import load_components


PACKAGE = Path(__file__).resolve().parents[1]
LAUNCH = PACKAGE / "launch" / "morai_global_path.launch.py"
COMPONENTS = PACKAGE / "config" / "morai_global_path_components.yaml"


class RecordingInclude:
    calls = []

    def __init__(self, source, **kwargs):
        self.source = source
        self.kwargs = kwargs
        self.calls.append(self)


def _load_launch_module():
    spec = spec_from_file_location("ad_bringup_morai_global_path", LAUNCH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_launch_description_constructs_static_scenario_gate(monkeypatch):
    module = _load_launch_module()
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda _package: str(PACKAGE),
    )

    description = module.generate_launch_description()

    scenario_nodes = [
        action for action in description.entities if isinstance(action, Node)
    ]
    handlers = [
        action
        for action in description.entities
        if isinstance(action, RegisterEventHandler)
    ]
    # The stack action is owned by the scenario-success handler, so it must not
    # execute as a top-level action before scenario setup succeeds.
    assert len(scenario_nodes) == 1
    assert scenario_nodes[0].node_executable == "ad_morai_scenario_setup"
    assert len(handlers) == 1


def test_trial_removes_config_and_component_switch_arguments(monkeypatch):
    module = _load_launch_module()
    monkeypatch.delenv("AD_DATA_DIR", raising=False)
    monkeypatch.delenv("MORAI_SCENARIO_FILE", raising=False)
    monkeypatch.setattr(
        module, "get_package_share_directory", lambda _package: str(PACKAGE)
    )
    description = module.generate_launch_description()
    arguments = {
        action.name: action
        for action in description.entities
        if isinstance(action, DeclareLaunchArgument)
    }
    context = LaunchContext()
    defaults = {
        name: perform_substitutions(context, action.default_value)
        for name, action in arguments.items()
    }

    assert defaults == {
        "data_dir": "",
        "scenario_file": module.DEFAULT_SCENARIO_FILE,
        "grpc_target": "127.0.0.1:7789",
        "control_enabled": "true",
        "localization_backend": "gnss_imu",
        "path_file": "",
        "local_motion_prediction_mode": "required",
    }
    assert "autonomy_config" not in arguments
    assert "local_planner_backend" not in arguments
    assert "path_tracking_backend" not in arguments
    assert not [name for name in arguments if name.startswith("start_")]


class ScenarioExit:
    def __init__(self, returncode):
        self.returncode = returncode


def test_trial_success_starts_both_runtime_actions():
    module = _load_launch_module()
    runtime_actions = [object(), object()]

    actions = module._after_scenario_setup(
        ScenarioExit(0), LaunchContext(), runtime_actions
    )

    assert isinstance(actions[0], LogInfo)
    assert actions[1:] == runtime_actions


def test_trial_failure_shuts_down_instead_of_starting_control_stack():
    module = _load_launch_module()
    runtime_actions = [object(), object()]

    actions = module._after_scenario_setup(
        ScenarioExit(17), LaunchContext(), runtime_actions
    )

    assert len(actions) == 2
    assert isinstance(actions[0], LogInfo)
    assert isinstance(actions[1], EmitEvent)
    assert isinstance(actions[1].event, Shutdown)
    assert all(action not in actions for action in runtime_actions)


def test_registered_handler_gates_the_actual_runtime_actions(monkeypatch):
    module = _load_launch_module()
    captured = {}
    original_on_process_exit = module.OnProcessExit

    def record_on_process_exit(**kwargs):
        captured.update(kwargs)
        return original_on_process_exit(**kwargs)

    monkeypatch.setattr(module, "OnProcessExit", record_on_process_exit)
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda _package: str(PACKAGE),
    )
    description = module.generate_launch_description()
    scenario = next(
        action
        for action in description.entities
        if isinstance(action, Node)
    )

    assert captured["target_action"] is scenario
    success = captured["on_exit"](ScenarioExit(0), LaunchContext())
    assert isinstance(success[0], LogInfo)
    assert all(
        isinstance(action, IncludeLaunchDescription) for action in success[1:]
    )

    failure = captured["on_exit"](ScenarioExit(17), LaunchContext())
    assert isinstance(failure[0], LogInfo)
    assert isinstance(failure[1], EmitEvent)
    assert isinstance(failure[1].event, Shutdown)
    assert not any(
        isinstance(action, IncludeLaunchDescription)
        for action in failure
    )


def test_trial_forwards_only_the_dev_bridge_runtime_contract(monkeypatch):
    module = _load_launch_module()
    RecordingInclude.calls.clear()
    monkeypatch.setattr(module, "IncludeLaunchDescription", RecordingInclude)
    monkeypatch.setattr(
        module, "_launch_file", lambda package, name: (package, name)
    )
    monkeypatch.setattr(
        module, "get_package_share_directory", lambda _package: str(PACKAGE)
    )
    module.generate_launch_description()
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "data_dir": "/tmp/data",
            "scenario_file": "sample.json",
            "grpc_target": "10.0.0.1:7789",
            "control_enabled": "true",
            "localization_backend": "gnss_imu",
            "path_file": "path/derived/example.txt",
        }
    )

    def evaluated(include):
        return {
            name: (
                value.perform(context)
                if hasattr(value, "perform")
                else value
            )
            for name, value in dict(include.kwargs["launch_arguments"]).items()
        }

    assert len(RecordingInclude.calls) == 1
    development = RecordingInclude.calls[0]
    assert development.source == (
        "ad_morai_bridge_dev",
        "bridge_dev.launch.py",
    )
    assert evaluated(development) == {
        "control_enabled": "true",
        "enable_velodyne_points": "true",
        "grpc_target": "10.0.0.1:7789",
    }


def test_trial_invokes_shared_stack_with_fixed_morai_config(monkeypatch):
    module = _load_launch_module()
    recorded = []

    def record_stack(**kwargs):
        recorded.append(kwargs)
        return []

    monkeypatch.setattr(module, "build_bringup_stack", record_stack)
    monkeypatch.setattr(
        module, "get_package_share_directory", lambda _package: str(PACKAGE)
    )
    module.generate_launch_description()
    assert recorded == [
        {
            "components_file": str(COMPONENTS),
            "status_topic": "/ad/dev/vehicle/ego_status",
        }
    ]


def test_morai_shared_stack_forwards_explicit_path_to_planner(monkeypatch):
    RecordingInclude.calls.clear()
    monkeypatch.setattr(
        bringup_stack, "IncludeLaunchDescription", RecordingInclude
    )
    monkeypatch.setattr(
        bringup_stack, "_launch_file", lambda package, name: (package, name)
    )
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "data_dir": "/tmp/ad-data",
            "control_enabled": "true",
            "localization_backend": "gnss_imu",
            "path_file": "path/derived/example.txt",
        }
    )

    bringup_stack.build_bringup_stack(
        components_file=COMPONENTS,
        status_topic="/ad/dev/vehicle/ego_status",
    )

    planner = next(
        include
        for include in RecordingInclude.calls
        if include.source == ("ad_planner", "planner.launch.py")
    )
    assert {
        name: value.perform(context) if hasattr(value, "perform") else value
        for name, value in dict(
            planner.kwargs["launch_arguments"]
        ).items()
    } == {
        "data_dir": "/tmp/ad-data",
        "path_file": "path/derived/example.txt",
        "route_corridor_file": "",
        "path_tracking_backend": "",
        "perception_enabled": "",
        "local_motion_prediction_mode": "",
        "tuning_lease_required": "",
    }


def test_morai_component_profile_enables_planner_but_not_competition_bridge():
    assert load_components(COMPONENTS, registered_component_names()) == {
        "description": True,
        "localization": True,
        "planner": True,
        "lidar_perception": True,
        "camera_perception": False,
        "bridge": False,
        "visualization": True,
    }
