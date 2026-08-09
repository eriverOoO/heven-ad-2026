from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import runpy
import xml.etree.ElementTree as ET

import pytest
import yaml

from ad_vehicle_profiling.controller import ControlDecision, ProfilerPhase
from ad_vehicle_profiling.experiment import DEFAULT_SPEEDS_KPH


PACKAGE = Path(__file__).resolve().parents[1]
REPOSITORY = PACKAGE.parent


def _capture_setup(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "setuptools.setup",
        lambda **kwargs: captured.update(kwargs),
    )
    runpy.run_path(str(PACKAGE / "setup.py"), run_name="__main__")
    return captured


def _load_launch_module():
    path = PACKAGE / "launch" / "profiling.launch.py"
    spec = spec_from_file_location("vehicle_profiling_launch", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    module.get_package_share_directory = lambda name: str(
        PACKAGE if name == "ad_vehicle_profiling" else REPOSITORY / name
    )
    return module


def test_package_declares_runtime_dependencies_and_entrypoints(monkeypatch):
    root = ET.parse(PACKAGE / "package.xml").getroot()
    dependencies = {
        element.text for element in root.findall("exec_depend")
    }
    setup = _capture_setup(monkeypatch)

    assert {
        "rclpy",
        "ad_morai_interfaces",
        "ad_morai_bridge_dev",
        "python3-yaml",
    } <= dependencies
    assert setup["entry_points"]["console_scripts"] == [
        "ad_vehicle_profiler = ad_vehicle_profiling.profiler_node:main",
        "ad_vehicle_profile_report = ad_vehicle_profiling.report:main",
        "ad_vehicle_profile_loop_guard = ad_vehicle_profiling.loop_guard_node:main",
    ]


def test_default_configuration_uses_competition_topics_and_complete_axes():
    document = yaml.safe_load(
        (PACKAGE / "config" / "profiling.yaml").read_text()
    )
    parameters = document["/**"]["ros__parameters"]

    assert parameters["topics.command"] == "/ad/control/command"
    assert parameters["topics.vehicle_status"] == "/ad/vehicle/status"
    assert parameters["topics.collisions"] == "/ad/safety/collisions"
    assert parameters["command_kinds"] == ["brake"]
    assert parameters["speed_bins_kph"] == list(DEFAULT_SPEEDS_KPH)
    assert parameters["command_percentages"] == list(range(0, 101, 10))
    assert parameters["minimum_valid_trials"] == 3
    assert parameters["maximum_attempts"] == 7
    assert parameters["repeatability_mad_limit_mps2"] == 0.5
    assert parameters["baseline_duration_sec"] == 0.75
    assert parameters["minimum_measurement_samples"] == 2
    assert parameters["controller.maximum_reach_duration_sec"] == 120.0
    assert parameters["controller.stale_status_sec"] == 1.0
    assert parameters["controller.test_duration_sec"] == 1.5
    assert parameters["controller.scale_brake_duration_by_command"] is True
    assert parameters["controller.brake_duration_reference_command"] == 0.1
    assert parameters["controller.speed_control_ki"] == 0.08
    assert parameters["controller.maximum_abs_jerk_mps3"] == 1000.0


def test_smoke_configuration_is_low_speed_and_single_trial():
    document = yaml.safe_load(
        (PACKAGE / "config" / "profiling_smoke.yaml").read_text()
    )
    parameters = document["/**"]["ros__parameters"]

    assert parameters["speed_bins_kph"] == [0, 10]
    assert parameters["command_percentages"] == [0, 10]
    assert parameters["minimum_valid_trials"] == 1
    assert parameters["maximum_attempts"] == 1
    assert parameters["controller.maximum_speed_mps"] <= 15.0 / 3.6


def test_ctrl_message_maps_normalized_throttle_without_mixing_commands():
    from ad_morai_interfaces.msg import CtrlCmd
    from ad_vehicle_profiling.profiler_node import make_ctrl_cmd

    decision = ControlDecision(
        phase=ProfilerPhase.APPLY_TEST_COMMAND,
        accelerator=0.35,
        brake=0.0,
        steering=0.0,
        gear=CtrlCmd.GEAR_DRIVE,
    )

    message = make_ctrl_cmd(decision)

    assert message.ctrl_mode == CtrlCmd.CTRL_MODE_AUTO
    assert message.gear == CtrlCmd.GEAR_DRIVE
    assert message.long_cmd_type == CtrlCmd.LONG_CMD_THROTTLE
    assert message.velocity == 0.0
    assert message.acceleration == 0.0
    assert message.accel == pytest.approx(0.35)
    assert message.brake == 0.0
    assert message.steering == 0.0


def test_output_root_requires_ad_data_dir_unless_explicit(monkeypatch):
    from ad_vehicle_profiling.profiler_node import (
        resolve_output_root,
        resolve_run_id,
    )

    monkeypatch.delenv("AD_DATA_DIR", raising=False)
    with pytest.raises(ValueError, match="AD_DATA_DIR"):
        resolve_output_root("")

    assert resolve_output_root("/tmp/profile-output") == Path(
        "/tmp/profile-output"
    )
    assert resolve_run_id("", now=0.0) == (
        "19700101-000000-ioniq5-longitudinal"
    )
    assert resolve_run_id("fixed-run", now=0.0) == "fixed-run"


def test_launch_starts_only_bridge_and_profiler():
    from launch import LaunchContext
    from launch.actions import (
        DeclareLaunchArgument,
        IncludeLaunchDescription,
        RegisterEventHandler,
    )
    from launch.utilities import perform_substitutions
    from launch_ros.actions import Node

    description = _load_launch_module().generate_launch_description()
    includes = [
        entity
        for entity in description.entities
        if isinstance(entity, IncludeLaunchDescription)
    ]
    nodes = [
        entity for entity in description.entities if isinstance(entity, Node)
    ]
    exit_handlers = [
        entity
        for entity in description.entities
        if isinstance(entity, RegisterEventHandler)
    ]

    assert len(includes) == 1
    assert len(nodes) == 2
    assert len(exit_handlers) == 1
    profiler = next(
        node
        for node in nodes
        if node.node_executable == "ad_vehicle_profiler"
    )
    guard = next(
        node
        for node in nodes
        if node.node_executable == "ad_vehicle_profile_loop_guard"
    )
    assert profiler.node_package == "ad_vehicle_profiling"
    assert guard.node_package == "ad_vehicle_profiling"
    context = LaunchContext()
    defaults = {
        entity.name: perform_substitutions(context, entity.default_value)
        for entity in description.entities
        if isinstance(entity, DeclareLaunchArgument)
    }
    assert defaults["profiler_config"].endswith("profiling.yaml")
    assert defaults["bridge_config"].endswith("bridge_profiling.yaml")
    assert defaults["loop_guard_enabled"] == "true"
    include_arguments = dict(includes[0].launch_arguments)
    assert include_arguments["config"].describe() == (
        "LaunchConfig('bridge_config')"
    )


def test_profiling_bridge_disables_unneeded_high_bandwidth_sensors():
    document = yaml.safe_load(
        (PACKAGE / "config" / "bridge_profiling.yaml").read_text()
    )
    parameters = document["/**"]["ros__parameters"]

    assert parameters["competition_status"]["enabled"]
    assert parameters["collisions"]["enabled"]
    assert parameters["timestamp_mode"] == "source_preferred"
    assert parameters["source_stamp_tolerance_sec"] == 1.0
    assert all(
        parameters[f"{name}.enabled"] is False
        for name in (
            "camera_front",
            "camera_left",
            "camera_right",
            "camera_traffic_light",
            "gps",
            "imu",
            "velodyne",
        )
    )
    assert parameters["control"]["target_port"] == 9093
