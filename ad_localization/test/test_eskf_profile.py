"""Behavior tests for the generic localization launch and the ESKF profile."""

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
PROFILE_PATH = PACKAGE_ROOT / "config" / "eskf.yaml"
SENSOR_CONFIG_PATH = (
    REPOSITORY_ROOT / "ad_description" / "config" / "sensor_mounts.yaml"
)
LOCALIZATION_CONFIG_PATH = PACKAGE_ROOT / "config" / "localization.yaml"
DIRECT_PROFILE_PATH = PACKAGE_ROOT / "config" / "gnss_imu.yaml"
ENCODER_PROFILE_PATH = PACKAGE_ROOT / "config" / "imu_quaternion_encoder.yaml"
QUATERNION_EKF_PROFILE_PATH = (
    PACKAGE_ROOT / "config" / "quaternion_wheel_gnss_ekf.yaml"
)
GRID_YAW_CORRECTION_RAD = -0.02350724531030645
ESKF_IMU_TOPIC = "/ad/localization/input/eskf_imu"
RAW_IMU_TOPIC = "/ad/sensors/imu/data"


def _load_launch():
    spec = importlib.util.spec_from_file_location(
        "ad_localization_launch", LAUNCH_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _profile_parameters() -> dict[str, object]:
    document = yaml.safe_load(PROFILE_PATH.read_text(encoding="utf-8"))
    return document["ekf_localization"]["ros__parameters"]


def _localization_parameters() -> dict[str, object]:
    document = yaml.safe_load(LOCALIZATION_CONFIG_PATH.read_text(encoding="utf-8"))
    return document["ad_localization"]["ros__parameters"]


def test_noise_profile_metadata_and_continuous_density_values_are_exact():
    sensor_document = yaml.safe_load(
        SENSOR_CONFIG_PATH.read_text(encoding="utf-8")
    )
    assert sensor_document["noise_profile_file"] == (
        "noise_SensorInfo_2023_Hyundai_Ioniq5.json"
    )

    parameters = _profile_parameters()
    assert parameters["propagation_model"] == "fast"
    assert parameters["use_continuous_process_noise_density"] is True
    assert parameters["use_second_order_state_transition"] is True
    assert parameters["use_second_order_process_noise"] is True
    assert parameters["var_imu_w"] == pytest.approx(
        2.738777777777778e-9, rel=0.0, abs=1e-24
    )
    assert parameters["var_imu_acc"] == pytest.approx(
        4.669444444444444e-7, rel=0.0, abs=1e-22
    )
    assert parameters["var_imu_gyro_bias"] == 0.0
    assert parameters["var_imu_acc_bias"] == 0.0
    assert parameters["var_gnss_xy"] == 0.25
    assert parameters["var_gnss_z"] == 0.25


def test_production_profile_learns_accel_bias_from_initial_uncertainty():
    """Select the simpler A/B winner, not a MORAI gravity or bias fit."""
    parameters = _profile_parameters()

    assert parameters["gravity_mps2"] == pytest.approx(
        9.80665, rel=0.0, abs=1e-12
    )
    assert parameters["stationary_initialization_estimate_accel_bias"] is False
    # 0.01 (m/s^2)^2 is a conservative 1-sigma prior of 0.1 m/s^2.  It is
    # initial state uncertainty, not continuous process noise.
    assert parameters["initial_imu_acc_bias_covariance"] == pytest.approx(
        0.01, rel=0.0, abs=1e-15
    )
    assert parameters["var_imu_acc_bias"] == 0.0
    assert parameters["tau_acc_bias_sec"] == pytest.approx(
        3600.0, rel=0.0, abs=1e-12
    )


def test_stationary_initialization_keeps_a_time_span_and_supports_20hz_imu():
    parameters = _profile_parameters()

    assert parameters["enable_stationary_initialization"] is True
    assert parameters["stationary_initialization_window_sec"] == pytest.approx(1.5)
    # The pinned initializer independently requires 95% of the time window.
    # At 20 Hz that span contains 30 possible unique samples. Requiring 25
    # retains 24 degrees of freedom for the variance gates and tolerates five
    # missing samples without coupling initialization to a 50 Hz source rate.
    assert parameters["stationary_initialization_min_samples"] == 25


def test_production_profile_uses_fail_closed_wheel_gated_zupt():
    parameters = _profile_parameters()
    adapter = _localization_parameters()

    assert parameters["use_wheel_speed"] is True
    # Preserve measured creep above 5 mm/s instead of quantizing every value
    # below the ZUPT gate to zero before the estimator can veto stationarity.
    assert adapter["wheel_standstill_threshold_mps"] == pytest.approx(0.005)
    assert adapter["wheel_standstill_threshold_mps"] < parameters[
        "zupt_max_wheel_speed_mps"
    ]
    assert parameters["use_zupt"] is True
    assert parameters["zupt_require_wheel_speed"] is True
    assert parameters["zupt_max_wheel_speed_mps"] == pytest.approx(0.02)
    assert parameters["zupt_max_wheel_age_sec"] == pytest.approx(0.1)
    # A 3.3 m/s full-brake pulse left about 0.39 m/s of 3-D state error after
    # the vehicle was physically stopped.  The gate must admit that bounded
    # estimator residual once wheel and IMU have independently confirmed rest.
    assert parameters["zupt_max_speed_mps"] == pytest.approx(0.5)
    assert parameters["zupt_min_stationary_duration_sec"] == pytest.approx(0.5)
    # On the first confirmed-stationary sample only, remove stale velocity
    # overconfidence so the ordinary ZUPT has useful gain.  This is not a
    # direct nominal-state snap and is re-armed only after motion resumes.
    assert parameters["zupt_reinitialize_velocity_covariance"] is True
    assert parameters["zupt_reinitialization_max_speed_mps"] == pytest.approx(0.5)
    assert parameters["var_zupt_reinitialized_velocity"] == pytest.approx(0.25)
    assert parameters["var_zupt_velocity"] == pytest.approx(0.000001)


def test_launched_eskf_queues_a_150ms_future_gnss_sample():
    """Keep normal MORAI scheduling bursts inside the replay queue window."""
    module = _load_launch()
    context = LaunchContext()
    context.launch_configurations.update(
        {**_base_context(), "localization_backend": "eskf"}
    )

    estimator = module._launch_setup(context)[1]
    evaluated = evaluate_parameters(context, estimator._Node__parameters)
    parameter_file = next(item for item in evaluated if isinstance(item, Path))
    parameters = yaml.safe_load(parameter_file.read_text(encoding="utf-8"))[
        "ekf_localization"
    ]["ros__parameters"]

    observed_future_gap_sec = 0.15
    adapter = _localization_parameters()
    assert 0.0 < adapter["input_future_tolerance_sec"] <= 0.05
    assert parameters["enable_measurement_replay"] is True
    assert parameters["max_future_measurement_wait_sec"] >= observed_future_gap_sec
    assert parameters["measurement_history_duration_sec"] > (
        parameters["max_future_measurement_wait_sec"]
    )


def test_profile_enables_conventional_gnss_outlier_and_robust_gates():
    parameters = _profile_parameters()

    assert parameters["max_gnss_position_innovation_m"] == 30.0
    assert parameters["max_gnss_position_nis"] == 0.0
    assert parameters["gnss_position_robust_loss"] == "huber"
    assert parameters["gnss_position_robust_tuning"] == 2.5
    assert parameters["max_gnss_position_robust_variance_scale"] == 100.0
    assert parameters.get("gnss_position_reacquisition_dt_sec", 0.0) == 0.0
    assert parameters["post_gap_gnss_required_consistent_fixes"] == 3
    assert parameters["post_gap_gnss_max_fix_interval_sec"] == 2.0
    assert parameters["post_gap_gnss_min_consistency_span_sec"] == 0.1
    assert parameters["post_gap_gnss_max_speed_mps"] == 70.0
    assert parameters["post_gap_gnss_consistency_tolerance_m"] == 5.0


def test_backend_topics_apply_the_grid_yaw_exactly_once():
    adapter = _localization_parameters()
    eskf = _profile_parameters()
    direct_document = yaml.safe_load(DIRECT_PROFILE_PATH.read_text(encoding="utf-8"))
    direct = direct_document["gnss_imu_localization"]["ros__parameters"]
    encoder_document = yaml.safe_load(
        ENCODER_PROFILE_PATH.read_text(encoding="utf-8")
    )
    encoder = encoder_document["imu_quaternion_encoder"]["ros__parameters"]

    assert adapter["imu_topic"] == RAW_IMU_TOPIC
    assert adapter["eskf_imu_topic"] == ESKF_IMU_TOPIC
    assert adapter["eskf_world_yaw_offset_rad"] == pytest.approx(
        GRID_YAW_CORRECTION_RAD, rel=0.0, abs=1e-17
    )
    assert eskf["imu_topic"] == ESKF_IMU_TOPIC
    assert direct["imu_topic"] == RAW_IMU_TOPIC
    assert direct["world_yaw_offset_rad"] == pytest.approx(
        GRID_YAW_CORRECTION_RAD, rel=0.0, abs=1e-17
    )
    assert encoder["world_yaw_offset_rad"] == pytest.approx(
        GRID_YAW_CORRECTION_RAD, rel=0.0, abs=1e-17
    )
    assert encoder["reject_zero_status_position"] is True


def test_profile_mount_matches_active_sensor_profile():
    parameters = _profile_parameters()
    sensor_document = yaml.safe_load(
        SENSOR_CONFIG_PATH.read_text(encoding="utf-8")
    )
    profile = sensor_document["profiles"][sensor_document["active_profile"]]
    position = profile["sensors"]["gps"]["position_m"]
    base_to_rear = sensor_document["coordinate_convention"]["static_frames"][
        "rear_axle"
    ]["position_m"]
    profile_xyz = tuple(
        parameters[name]
        for name in ("gnss_lever_arm_x", "gnss_lever_arm_y", "gnss_lever_arm_z")
    )
    assert profile_xyz == pytest.approx(
        tuple(position[axis] + base_to_rear[axis] for axis in ("x", "y", "z")),
        abs=1e-12,
    )


def _base_context() -> dict[str, str]:
    return {
        "adapter_config": str(PACKAGE_ROOT / "config" / "localization.yaml"),
        "gnss_imu_config": str(PACKAGE_ROOT / "config" / "gnss_imu.yaml"),
        "eskf_config": str(PACKAGE_ROOT / "config" / "eskf.yaml"),
        "imu_quaternion_encoder_config": str(
            PACKAGE_ROOT / "config" / "imu_quaternion_encoder.yaml"
        ),
        "quaternion_wheel_gnss_ekf_config": str(
            QUATERNION_EKF_PROFILE_PATH
        ),
        "localization_manager_config": str(
            PACKAGE_ROOT / "config" / "localization_manager.yaml"
        ),
        "imu_quaternion_encoder_mode": "status_pose",
        "imu_topic": RAW_IMU_TOPIC,
        "eskf_imu_topic": ESKF_IMU_TOPIC,
        "status_topic": "/ad/vehicle/status",
        "autostart": "true",
        "sensor_config": str(SENSOR_CONFIG_PATH),
        "sensor_profile": "current_front_sensor_mounts",
    }


def test_launch_exposes_only_the_single_vehicle_interface():
    description = _load_launch().generate_launch_description()
    declared_arguments = {
        action.name
        for action in description.entities
        if isinstance(action, DeclareLaunchArgument)
    }

    assert "namespace" not in declared_arguments


def test_eskf_imu_launch_argument_keeps_the_exact_default():
    description = _load_launch().generate_launch_description()
    arguments = {
        action.name: action
        for action in description.entities
        if isinstance(action, DeclareLaunchArgument)
    }

    assert perform_substitutions(
        LaunchContext(), arguments["eskf_imu_topic"].default_value
    ) == ESKF_IMU_TOPIC


def test_selected_imu_launch_argument_keeps_the_raw_default():
    description = _load_launch().generate_launch_description()
    arguments = {
        action.name: action
        for action in description.entities
        if isinstance(action, DeclareLaunchArgument)
    }

    assert perform_substitutions(
        LaunchContext(), arguments["imu_topic"].default_value
    ) == RAW_IMU_TOPIC


@pytest.mark.parametrize(
    ("backend", "estimator_topic"),
    [
        ("gnss_imu", "/ad/localization/test/compatible_imu"),
        ("eskf", ESKF_IMU_TOPIC),
        ("imu_quaternion_encoder", "/ad/localization/test/compatible_imu"),
        ("quaternion_wheel_gnss_ekf", "/ad/localization/test/compatible_imu"),
    ],
)
def test_custom_selected_imu_reaches_adapter_and_each_backend_input(
    backend, estimator_topic
):
    module = _load_launch()
    context = LaunchContext()
    context.launch_configurations.update(
        {
            **_base_context(),
            "localization_backend": backend,
            "imu_topic": "/ad/localization/test/compatible_imu",
        }
    )

    adapter, estimator = module._launch_setup(context)[:2]
    adapter_parameters = evaluate_parameters(context, adapter._Node__parameters)
    estimator_parameters = evaluate_parameters(context, estimator._Node__parameters)
    adapter_overrides = next(
        item for item in adapter_parameters if isinstance(item, dict)
    )
    estimator_overrides = next(
        item for item in estimator_parameters if isinstance(item, dict)
    )

    assert adapter_overrides["imu_topic"] == (
        "/ad/localization/test/compatible_imu"
    )
    assert estimator_overrides["imu_topic"] == estimator_topic


def test_custom_eskf_imu_topic_reaches_adapter_and_estimator():
    module = _load_launch()
    context = LaunchContext()
    context.launch_configurations.update(
        {
            **_base_context(),
            "localization_backend": "eskf",
            "eskf_imu_topic": "/ad/localization/test/custom_eskf_imu",
        }
    )

    adapter, estimator = module._launch_setup(context)[:2]
    adapter_parameters = evaluate_parameters(context, adapter._Node__parameters)
    estimator_parameters = evaluate_parameters(context, estimator._Node__parameters)
    adapter_overrides = next(
        item for item in adapter_parameters if isinstance(item, dict)
    )
    estimator_overrides = next(
        item for item in estimator_parameters if isinstance(item, dict)
    )

    assert adapter_overrides["eskf_imu_topic"] == (
        "/ad/localization/test/custom_eskf_imu"
    )
    assert estimator_overrides["imu_topic"] == (
        "/ad/localization/test/custom_eskf_imu"
    )


def test_launch_selects_exactly_one_estimator_and_rejects_unknown_backend():
    module = _load_launch()

    def estimator_for(backend):
        context = LaunchContext()
        context.launch_configurations.update(
            {
                **_base_context(),
                "localization_backend": backend,
                "namespace": "must_be_ignored",
            }
        )
        actions = module._launch_setup(context)
        launched_nodes = [action for action in actions if isinstance(action, Node)]
        assert all(
            action._Node__node_namespace in (None, "", "/")
            for action in launched_nodes
        )
        estimators = [
            action
            for action in actions
            if isinstance(action, Node)
            and not isinstance(action, LifecycleNode)
            and action.node_executable != "localization_manager_node"
        ]
        managers = [
            action
            for action in actions
            if isinstance(action, Node)
            and action.node_executable == "localization_manager_node"
        ]
        assert len(estimators) == 1
        assert len(managers) == 1
        evaluated = evaluate_parameters(context, estimators[0]._Node__parameters)
        overrides = next(item for item in evaluated if isinstance(item, dict))
        manager_parameters = evaluate_parameters(
            context, managers[0]._Node__parameters
        )
        manager_overrides = next(
            item for item in manager_parameters if isinstance(item, dict)
        )
        adapter = next(
            action for action in actions if isinstance(action, LifecycleNode)
        )
        adapter_parameters = evaluate_parameters(
            context, adapter._Node__parameters
        )
        adapter_overrides = next(
            item for item in adapter_parameters if isinstance(item, dict)
        )
        return (
            estimators[0],
            overrides,
            manager_overrides,
            adapter_overrides,
        )

    direct, direct_overrides, direct_manager, direct_adapter = estimator_for(
        "gnss_imu"
    )
    assert direct.node_package == "ad_localization"
    assert direct.node_executable == "gnss_imu_localization_node"
    assert direct_overrides["imu_topic"] == RAW_IMU_TOPIC
    assert direct_overrides["output_odometry_topic"] == (
        "/ad/localization/backends/gnss_imu/odometry"
    )
    assert direct_overrides["publish_tf"] is False
    assert direct_manager == {
        "input_odometry_topic": "/ad/localization/backends/gnss_imu/odometry",
        "canonical_odometry_topic": "/ad/localization/odometry",
        "publish_tf": True,
    }
    assert direct_adapter["publish_map_to_odom_tf"] is False

    eskf, eskf_overrides, eskf_manager, eskf_adapter = estimator_for("eskf")
    assert eskf.node_package == "kalman_filter_localization"
    assert eskf.node_executable == "ekf_localization_node"
    assert eskf_overrides["imu_topic"] == ESKF_IMU_TOPIC
    assert eskf_overrides["output_odometry_topic"] == (
        "/ad/localization/backends/eskf/odometry"
    )
    assert eskf_overrides["publish_tf"] is False
    assert eskf_manager["input_odometry_topic"] == (
        "/ad/localization/backends/eskf/odometry"
    )
    assert eskf_adapter["publish_map_to_odom_tf"] is False

    encoder, encoder_overrides, encoder_manager, encoder_adapter = (
        estimator_for("imu_quaternion_encoder")
    )
    assert encoder.node_package == "ad_localization"
    assert encoder.node_executable == "imu_quaternion_encoder_node"
    assert encoder_overrides == {
        "mode": "status_pose",
        "status_topic": "/ad/vehicle/status",
        "imu_topic": RAW_IMU_TOPIC,
        "imu_frame": "imu_link",
        "gnss_lever_arm_m": (0.0, 0.0, 1.5685),
        "imu_mount_rpy_rad": (0.0, 0.0, 0.0),
        "output_odometry_topic": (
            "/ad/localization/backends/imu_quaternion_encoder/odometry"
        ),
        "publish_tf": False,
    }
    assert encoder_manager["input_odometry_topic"] == (
        "/ad/localization/backends/imu_quaternion_encoder/odometry"
    )
    assert encoder_adapter["publish_map_to_odom_tf"] is False

    quaternion_ekf, quaternion_ekf_overrides, quaternion_ekf_manager, (
        quaternion_ekf_adapter
    ) = estimator_for("quaternion_wheel_gnss_ekf")
    assert quaternion_ekf.node_package == "ad_localization"
    assert quaternion_ekf.node_executable == "quaternion_wheel_gnss_ekf_node"
    assert quaternion_ekf_overrides == {
        "imu_topic": RAW_IMU_TOPIC,
        "imu_frame": "imu_link",
        "gnss_lever_arm_m": (0.0, 0.0, 1.5685),
        "imu_mount_rpy_rad": (0.0, 0.0, 0.0),
        "output_odometry_topic": (
            "/ad/localization/backends/quaternion_wheel_gnss_ekf/odometry"
        ),
        "publish_tf": False,
    }
    assert quaternion_ekf_manager["input_odometry_topic"] == (
        "/ad/localization/backends/quaternion_wheel_gnss_ekf/odometry"
    )
    assert quaternion_ekf_adapter["publish_map_to_odom_tf"] is False

    context = LaunchContext()
    context.launch_configurations.update(
        {**_base_context(), "localization_backend": "invalid"}
    )
    with pytest.raises(RuntimeError, match="localization_backend"):
        module._launch_setup(context)


def test_launch_resolves_gnss_lever_arms_per_sensor_profile():
    module = _load_launch()

    def lever_arm(profile):
        return module._load_sensor_geometry(SENSOR_CONFIG_PATH, profile)[
            "gnss_lever_arm_m"
        ]

    assert lever_arm("current_front_sensor_mounts") == [0.0, 0.0, 1.5685]
    assert lever_arm("planned_centered_sensor_mounts") == [0.0, 0.0, 1.0685]
    # Blank selects the active profile.
    assert lever_arm("") == [0.0, 0.0, 1.5685]
