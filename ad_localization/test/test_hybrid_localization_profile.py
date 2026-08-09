"""Behavior test for the GNSS/FastLIO hybrid localization launch."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.utilities import perform_substitutions
from launch_ros.actions import LifecycleNode, Node
from launch_ros.utilities import evaluate_parameters
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parent
HYBRID_LAUNCH = PACKAGE_ROOT / "launch" / "hybrid_localization.launch.py"
SENSOR_CONFIG = (
    REPOSITORY_ROOT / "ad_description" / "config" / "sensor_mounts.yaml"
)


def _load_launch():
    spec = importlib.util.spec_from_file_location(
        "hybrid_localization", HYBRID_LAUNCH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hybrid_launch_runs_private_backends_behind_one_canonical_owner():
    module = _load_launch()
    description = module.generate_launch_description()
    declared_arguments = {
        action.name
        for action in description.entities
        if isinstance(action, DeclareLaunchArgument)
    }
    assert "namespace" not in declared_arguments
    assert "platform_profile" in declared_arguments
    defaults = {
        action.name: perform_substitutions(LaunchContext(), action.default_value)
        for action in description.entities
        if isinstance(action, DeclareLaunchArgument)
    }
    assert defaults["imu_topic"] == "/ad/sensors/imu/data"

    context = LaunchContext()
    context.launch_configurations.update(
        {
            "namespace": "must_be_ignored",
            "adapter_config": str(PACKAGE_ROOT / "config" / "localization.yaml"),
            "gnss_imu_config": str(PACKAGE_ROOT / "config" / "gnss_imu.yaml"),
            "fastlio_config": str(PACKAGE_ROOT / "config" / "fastlio.yaml"),
            "hybrid_config": str(PACKAGE_ROOT / "config" / "hybrid.yaml"),
            "map_path": "/tmp/cp14_to_cp15.pcd",
            "status_topic": "/ad/vehicle/status",
            "autostart": "true",
            "sensor_config": str(SENSOR_CONFIG),
            "sensor_profile": "planned_centered_sensor_mounts",
            "platform_profile": "morai",
            "imu_topic": "/ad/localization/input/imu_compatible",
        }
    )
    actions = module._launch_setup(context)

    adapters = [a for a in actions if isinstance(a, LifecycleNode)]
    nodes = [
        a for a in actions
        if isinstance(a, Node) and not isinstance(a, LifecycleNode)
    ]
    assert len(adapters) == 1
    assert adapters[0].node_executable == "ad_localization_node"
    assert all(
        action._Node__node_namespace in (None, "", "/")
        for action in [*adapters, *nodes]
    )
    assert {(node.node_package, node.node_executable) for node in nodes} == {
        ("ad_localization", "gnss_imu_localization_node"),
        ("fast_lio", "fastlio_mapping"),
        ("ad_localization", "localization_handoff_node"),
    }

    # Backends publish on private topics without TF; the handoff node owns
    # the canonical output and /tf.
    geometry = module._common._load_fastlio_geometry(
        SENSOR_CONFIG, "planned_centered_sensor_mounts"
    )
    selected_imu = LaunchConfiguration("imu_topic")
    gnss = module._gnss_overrides(geometry, selected_imu)
    assert gnss["output_odometry_topic"] == (
        "/ad/localization/backends/gnss_imu/odometry"
    )
    assert gnss["publish_tf"] is False

    fastlio = module._hybrid_fastlio_parameters(
        geometry, "morai", "/tmp/cp14_to_cp15.pcd", selected_imu
    )
    assert fastlio["mode"] == "localization"
    assert fastlio["map_path"] == "/tmp/cp14_to_cp15.pcd"
    assert fastlio["odometry_topic"] == (
        "/ad/localization/backends/fastlio/odometry"
    )
    assert fastlio["publish_tf"] is False
    assert fastlio["preprocess.scan_timing_mode"] == "instantaneous"

    overrides = module._adapter_overrides(geometry, selected_imu)
    assert overrides["localization_backend"] == "hybrid"
    assert overrides["initial_pose_topic"] == (
        "/ad/localization/input/adapter_initial_pose"
    )
    assert overrides["imu_topic"] is selected_imu
    assert gnss["imu_topic"] is selected_imu
    assert fastlio["common.imu_topic"] is selected_imu

    adapter = adapters[0]
    adapter_parameters = evaluate_parameters(context, adapter._Node__parameters)
    adapter_runtime = next(
        item for item in adapter_parameters if isinstance(item, dict)
    )
    assert adapter_runtime["imu_topic"] == (
        "/ad/localization/input/imu_compatible"
    )


def test_hybrid_real_hardware_profile_keeps_rolling_scan_timing():
    module = _load_launch()
    geometry = module._common._load_fastlio_geometry(
        SENSOR_CONFIG, "planned_centered_sensor_mounts"
    )

    parameters = module._hybrid_fastlio_parameters(
        geometry,
        "real_hardware",
        "/tmp/cp14_to_cp15.pcd",
        LaunchConfiguration("imu_topic"),
    )

    assert parameters["preprocess.scan_timing_mode"] == "rolling"


@pytest.mark.parametrize("profile", ["simulator", "hardware", ""])
def test_hybrid_launch_rejects_unknown_platform_profile(profile):
    module = _load_launch()
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "adapter_config": str(PACKAGE_ROOT / "config" / "localization.yaml"),
            "gnss_imu_config": str(PACKAGE_ROOT / "config" / "gnss_imu.yaml"),
            "fastlio_config": str(PACKAGE_ROOT / "config" / "fastlio.yaml"),
            "hybrid_config": str(PACKAGE_ROOT / "config" / "hybrid.yaml"),
            "map_path": "/tmp/cp14_to_cp15.pcd",
            "status_topic": "/ad/vehicle/status",
            "autostart": "true",
            "sensor_config": str(SENSOR_CONFIG),
            "sensor_profile": "planned_centered_sensor_mounts",
            "platform_profile": profile,
            "imu_topic": "/ad/sensors/imu/data",
        }
    )

    with pytest.raises(RuntimeError, match="platform_profile"):
        module._launch_setup(context)
