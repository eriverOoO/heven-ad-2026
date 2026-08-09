"""Contracts for the quaternion-wheel-GNSS EKF profile and launch selection."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.utilities import perform_substitutions
from launch_ros.actions import LifecycleNode, Node
from launch_ros.utilities import evaluate_parameters
import pytest
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parent
LAUNCH_PATH = PACKAGE_ROOT / "launch" / "localization.launch.py"
PROFILE_PATH = PACKAGE_ROOT / "config" / "quaternion_wheel_gnss_ekf.yaml"
SENSOR_CONFIG_PATH = (
    REPOSITORY_ROOT / "ad_description" / "config" / "sensor_mounts.yaml"
)
PRIVATE_TOPIC = (
    "/ad/localization/backends/quaternion_wheel_gnss_ekf/odometry"
)


def _load_launch():
    spec = importlib.util.spec_from_file_location(
        "ad_localization_launch_quaternion_ekf", LAUNCH_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_context() -> dict[str, str]:
    return {
        "adapter_config": str(PACKAGE_ROOT / "config" / "localization.yaml"),
        "gnss_imu_config": str(PACKAGE_ROOT / "config" / "gnss_imu.yaml"),
        "eskf_config": str(PACKAGE_ROOT / "config" / "eskf.yaml"),
        "imu_quaternion_encoder_config": str(
            PACKAGE_ROOT / "config" / "imu_quaternion_encoder.yaml"
        ),
        "quaternion_wheel_gnss_ekf_config": str(PROFILE_PATH),
        "localization_manager_config": str(
            PACKAGE_ROOT / "config" / "localization_manager.yaml"
        ),
        "imu_quaternion_encoder_mode": "status_pose",
        "imu_topic": "/ad/localization/test/compatible_imu",
        "eskf_imu_topic": "/ad/localization/input/eskf_imu",
        "status_topic": "/ad/vehicle/status",
        "autostart": "true",
        "sensor_config": str(SENSOR_CONFIG_PATH),
        "sensor_profile": "current_front_sensor_mounts",
        "localization_backend": "quaternion_wheel_gnss_ekf",
    }


def test_profile_keeps_xy_only_high_wheel_trust_and_low_gnss_gain():
    document = yaml.safe_load(PROFILE_PATH.read_text(encoding="utf-8"))
    parameters = document["quaternion_wheel_gnss_ekf"]["ros__parameters"]

    assert parameters["wheel_topic"] == "/ad/localization/input/wheel_speed"
    assert parameters["gnss_topic"] == "/ad/localization/input/gnss_pose"
    assert parameters["output_odometry_topic"] == PRIVATE_TOPIC
    assert parameters["reference_frame"] == "odom"
    assert parameters["base_frame"] == "base_link"
    assert parameters["fixed_output_z_m"] == 0.0
    assert parameters["unobserved_variance"] >= 1.0e6
    assert 0.0 < parameters["wheel_speed_variance_floor_m2ps2"] < 0.1
    assert parameters["gnss_variance_m2"] >= 4.0
    assert parameters["gnss_mahalanobis_threshold"] == pytest.approx(9.21)
    assert parameters["initialization.sample_count"] > 1
    assert parameters["teleport.confirmation_samples"] >= 3
    assert parameters["publish_tf"] is False
    assert not any(
        "acceleration" in name or "angular_velocity" in name
        for name in parameters
    )


def test_backend_launch_uses_private_inputs_and_manager_only_canonical_tf():
    module = _load_launch()
    assert module.BACKEND_ODOMETRY_TOPICS["quaternion_wheel_gnss_ekf"] == (
        PRIVATE_TOPIC
    )
    context = LaunchContext()
    context.launch_configurations.update(_base_context())

    actions = module._launch_setup(context)
    adapter = next(action for action in actions if isinstance(action, LifecycleNode))
    nodes = [action for action in actions if isinstance(action, Node)]
    estimators = [
        action
        for action in nodes
        if not isinstance(action, LifecycleNode)
        and action.node_executable != "localization_manager_node"
    ]
    managers = [
        action
        for action in nodes
        if action.node_executable == "localization_manager_node"
    ]

    assert len(estimators) == 1
    assert len(managers) == 1
    estimator = estimators[0]
    assert estimator.node_package == "ad_localization"
    assert estimator.node_executable == "quaternion_wheel_gnss_ekf_node"
    estimator_parameters = evaluate_parameters(
        context, estimator._Node__parameters
    )
    estimator_overrides = next(
        item for item in estimator_parameters if isinstance(item, dict)
    )
    assert estimator_overrides == {
        "imu_topic": "/ad/localization/test/compatible_imu",
        "imu_frame": "imu_link",
        "gnss_lever_arm_m": (0.0, 0.0, 1.5685),
        "imu_mount_rpy_rad": (0.0, 0.0, 0.0),
        "output_odometry_topic": PRIVATE_TOPIC,
        "publish_tf": False,
    }
    manager_parameters = evaluate_parameters(context, managers[0]._Node__parameters)
    manager_overrides = next(
        item for item in manager_parameters if isinstance(item, dict)
    )
    assert manager_overrides == {
        "input_odometry_topic": PRIVATE_TOPIC,
        "canonical_odometry_topic": "/ad/localization/odometry",
        "publish_tf": True,
    }
    adapter_parameters = evaluate_parameters(context, adapter._Node__parameters)
    adapter_overrides = next(
        item for item in adapter_parameters if isinstance(item, dict)
    )
    assert adapter_overrides["localization_backend"] == (
        "quaternion_wheel_gnss_ekf"
    )
    assert adapter_overrides["publish_map_to_odom_tf"] is False


def test_launch_declares_backend_config_and_preserves_default_backend():
    description = _load_launch().generate_launch_description()
    arguments = {
        action.name: action
        for action in description.entities
        if isinstance(action, DeclareLaunchArgument)
    }

    assert "quaternion_wheel_gnss_ekf_config" in arguments
    assert perform_substitutions(
        LaunchContext(), arguments["localization_backend"].default_value
    ) == "gnss_imu"


@pytest.mark.parametrize(
    "backend",
    ["quaternion_wheel_gnss_ekf_typo", "quaternion_wheel_gnss_ekf/extra"],
)
def test_backend_name_validation_remains_exact(backend):
    module = _load_launch()
    context = LaunchContext()
    context.launch_configurations.update(
        {**_base_context(), "localization_backend": backend}
    )

    with pytest.raises(RuntimeError, match="localization_backend"):
        module._launch_setup(context)
