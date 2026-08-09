from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import time

from launch import LaunchContext, LaunchDescription, LaunchService
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    OpaqueFunction,
    SetLaunchConfiguration,
    TimerAction,
)
from launch.events import Shutdown
from launch.utilities import perform_substitutions
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
LAUNCH_PATH = PACKAGE_ROOT / "launch" / "eskf_parallel_ab.launch.py"
CONFIG_PATH = PACKAGE_ROOT / "config" / "eskf_parallel_ab.yaml"
DRIVE_CONFIG_PATH = PACKAGE_ROOT / "config" / "eskf_parallel_ab_drive.yaml"
ESKF_PATH = PACKAGE_ROOT.parent / "ad_localization" / "config" / "eskf.yaml"


def _load_launch_module():
    spec = spec_from_file_location("eskf_parallel_ab_launch", LAUNCH_PATH)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _node_parameters(context: LaunchContext, node: Node) -> dict[str, object]:
    evaluated = evaluate_parameters(context, node._Node__parameters)
    combined: dict[str, object] = {}
    for value in evaluated:
        if isinstance(value, dict):
            combined.update(value)
    return combined


def _run_setup_with_fake_processes(
    module, monkeypatch, *, harness_enabled: bool, harness_exit_code: int = 0
) -> float:
    """Exercise the launch lifecycle without starting ROS or contacting MORAI."""

    def fake_node(*, executable, condition=None, **_kwargs):
        if executable == "eskf_ab_harness":
            command = f"import sys; sys.exit({harness_exit_code})"
        else:
            command = "import time; time.sleep(5.0)"
        return ExecuteProcess(
            cmd=[sys.executable, "-c", command],
            condition=condition,
            output="log",
        )

    monkeypatch.setattr(module, "Node", fake_node)
    settings = {
        "experiment_config": str(CONFIG_PATH),
        "eskf_config": str(ESKF_PATH),
        "run_id": "lifecycle_test",
        "grpc_target": "127.0.0.1:7789",
        "grpc_timeout_sec": "2.0",
        "drive_enabled": "false",
        "harness_enabled": "true" if harness_enabled else "false",
        "profile": "stationary",
        "stationary_duration_sec": "1.0",
        "maximum_pair_age_sec": "0.1",
        "data_root": "",
        "repository_root": "",
        "active_sensor_file": "",
    }
    description = LaunchDescription(
        [
            *(SetLaunchConfiguration(name, value) for name, value in settings.items()),
            OpaqueFunction(function=module._launch_setup),
            TimerAction(
                period=0.5,
                actions=[EmitEvent(event=Shutdown(reason="lifecycle test timeout"))],
            ),
        ]
    )
    service = LaunchService()
    service.include_launch_description(description)
    started = time.monotonic()
    assert service.run() == 0
    return time.monotonic() - started


def test_stationary_config_is_a_standard_gravity_two_by_two_bias_matrix():
    module = _load_launch_module()
    experiment = module.load_experiment_config(CONFIG_PATH)

    assert experiment.drive_enabled is False
    assert experiment.profile == "stationary"
    assert set(experiment.allowed_candidate_parameters) == {
        "gravity_mps2",
        "initial_imu_acc_bias_covariance",
        "stationary_initialization_estimate_accel_bias",
        "var_imu_acc_bias",
    }

    candidates = {item.name: item for item in experiment.candidates if item.enabled}
    assert set(candidates) == {
        "baseline",
        "bias_covariance",
        "observable_bias",
        "combined_bias",
    }
    combinations = {
        (
            item.parameters["stationary_initialization_estimate_accel_bias"],
            item.parameters["initial_imu_acc_bias_covariance"],
        )
        for item in candidates.values()
    }
    assert combinations == {
        (False, 0.0),
        (False, 0.01),
        (True, 0.0),
        (True, 0.01),
    }
    assert {
        item.parameters["gravity_mps2"] for item in candidates.values()
    } == {9.80665}
    assert {
        item.parameters["var_imu_acc_bias"] for item in candidates.values()
    } == {0.0}


def test_reviewed_drive_config_is_explicit_two_candidate_ab():
    module = _load_launch_module()
    experiment = module.load_experiment_config(DRIVE_CONFIG_PATH)

    assert experiment.drive_enabled is True
    assert experiment.profile == "closed_loop_pulse"
    enabled = {
        item.name: item
        for item in experiment.candidates
        if item.enabled
    }
    assert set(enabled) == {"baseline", "production_bias"}
    assert enabled["baseline"].parameters == {
        "gravity_mps2": 9.80665,
        "initial_imu_acc_bias_covariance": 0.0,
        "stationary_initialization_estimate_accel_bias": False,
        "var_imu_acc_bias": 0.0,
    }
    assert enabled["production_bias"].parameters == {
        "gravity_mps2": 9.80665,
        "initial_imu_acc_bias_covariance": 0.01,
        "stationary_initialization_estimate_accel_bias": False,
        "var_imu_acc_bias": 0.0,
    }
    assert experiment.safety["maximum_speed_mps"] == pytest.approx(0.50)
    assert experiment.safety["maximum_travel_m"] == pytest.approx(0.25)

    document = module.yaml.safe_load(DRIVE_CONFIG_PATH.read_text(encoding="utf-8"))
    controller = document["closed_loop_pulse"]
    assert controller["target_speed_mps"] < controller["soft_speed_limit_mps"]
    assert controller["soft_speed_limit_mps"] < experiment.safety["maximum_speed_mps"]
    assert controller["soft_travel_limit_m"] < experiment.safety["maximum_travel_m"]
    assert controller["maximum_throttle"] <= 0.10


def test_config_rejects_unknown_candidate_parameter(tmp_path: Path):
    module = _load_launch_module()
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """
schema_version: 1
drive_enabled: false
profile: stationary
candidates:
  unsafe:
    enabled: true
    parameters:
      output_odometry_topic: /ad/localization/odometry
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="candidate parameter"):
        module.load_experiment_config(bad)


@pytest.mark.parametrize(
    "body",
    (
        "drive_enabled: 'false'\nprofile: stationary\ncandidates:\n"
        "  c:\n    enabled: true\n    parameters: {}\n",
        "drive_enabled: false\nprofile: stationary\ncandidates:\n"
        "  c:\n    enabled: 'false'\n    parameters: {}\n",
    ),
)
def test_config_rejects_string_booleans_that_could_fail_open(tmp_path: Path, body):
    module = _load_launch_module()
    path = tmp_path / "string_bool.yaml"
    path.write_text("schema_version: 1\n" + body, encoding="utf-8")

    with pytest.raises(ValueError, match="boolean"):
        module.load_experiment_config(path)


def test_eskf_parameter_loader_rejects_multiple_node_roots(tmp_path: Path):
    module = _load_launch_module()
    path = tmp_path / "multi.yaml"
    path.write_text(
        "a:\n  ros__parameters: {}\nb:\n  ros__parameters: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly one"):
        module.load_eskf_parameters(path)


def test_candidate_nodes_share_inputs_but_isolate_initial_pose_outputs_and_tf():
    module = _load_launch_module()
    experiment = module.load_experiment_config(CONFIG_PATH)
    base_parameters = module.load_eskf_parameters(ESKF_PATH)
    context = LaunchContext()
    nodes = [
        module.make_candidate_node(
            run_id="test_run",
            candidate=candidate,
            base_parameters=base_parameters,
        )
        for candidate in experiment.candidates
        if candidate.enabled
    ]

    assert nodes
    assert all(isinstance(node, Node) for node in nodes)
    evaluated = [_node_parameters(context, node) for node in nodes]
    assert {p["imu_topic"] for p in evaluated} == {
        "/ad/localization/input/eskf_imu"
    }
    assert {p["gnss_pose_topic"] for p in evaluated} == {
        "/ad/localization/input/gnss_pose"
    }
    assert {p["wheel_speed_topic"] for p in evaluated} == {
        "/ad/localization/input/wheel_speed"
    }
    assert len({p["initial_pose_topic"] for p in evaluated}) == len(nodes)
    assert len({p["output_odometry_topic"] for p in evaluated}) == len(nodes)
    assert all(p["publish_tf"] is False for p in evaluated)
    assert all(p["publish_debug_topics"] is True for p in evaluated)
    assert all(
        p["output_odometry_topic"] != "/ad/localization/odometry"
        for p in evaluated
    )
    assert all(
        str(node._Node__node_namespace) == "/ad/experiment/eskf/test_run"
        for node in nodes
    )


def test_forced_safety_parameters_override_candidate_values():
    module = _load_launch_module()
    candidate = module.CandidateConfig(
        name="attempt",
        enabled=True,
        parameters={"gravity_mps2": 9.7},
    )
    base = module.load_eskf_parameters(ESKF_PATH)
    node = module.make_candidate_node(
        run_id="test_run", candidate=candidate, base_parameters=base
    )
    parameters = _node_parameters(LaunchContext(), node)

    assert parameters["publish_tf"] is False
    assert parameters["publish_debug_topics"] is True
    assert parameters["output_odometry_topic"].startswith(
        "/ad/experiment/eskf/test_run/attempt/"
    )


def test_launch_starts_one_restricted_harness_and_drive_defaults_off(tmp_path: Path):
    module = _load_launch_module()
    description = module.generate_launch_description()
    arguments = {
        action.name: action
        for action in description.entities
        if isinstance(action, DeclareLaunchArgument)
    }
    assert perform_substitutions(
        LaunchContext(), arguments["drive_enabled"].default_value
    ) == "false"
    assert perform_substitutions(
        LaunchContext(), arguments["profile"].default_value
    ) == ""

    sensor_file = tmp_path / "sensor.json"
    sensor_file.write_text("{}", encoding="utf-8")
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "experiment_config": str(CONFIG_PATH),
            "eskf_config": str(ESKF_PATH),
            "run_id": "test_run",
            "grpc_target": "127.0.0.1:7789",
            "grpc_timeout_sec": "2.0",
            "drive_enabled": "false",
            "profile": "stationary",
            "stationary_duration_sec": "1.0",
            "maximum_pair_age_sec": "0.1",
            "data_root": str(tmp_path),
            "repository_root": str(PACKAGE_ROOT.parent),
            "active_sensor_file": str(sensor_file),
        }
    )
    actions = module._launch_setup(context)
    harnesses = [
        action
        for action in actions
        if isinstance(action, Node)
        and action._Node__node_executable == "eskf_ab_harness"
    ]
    assert len(harnesses) == 1
    assert str(harnesses[0]._Node__node_namespace) == (
        "/ad/experiment/eskf/test_run"
    )


@pytest.mark.parametrize("exit_code", (0, 7))
def test_harness_exit_shuts_down_candidate_processes(monkeypatch, exit_code):
    module = _load_launch_module()

    elapsed = _run_setup_with_fake_processes(
        module,
        monkeypatch,
        harness_enabled=True,
        harness_exit_code=exit_code,
    )

    assert elapsed < 0.4


def test_disabled_harness_keeps_candidates_alive_until_external_shutdown(monkeypatch):
    module = _load_launch_module()

    elapsed = _run_setup_with_fake_processes(
        module, monkeypatch, harness_enabled=False
    )

    assert elapsed >= 0.4
