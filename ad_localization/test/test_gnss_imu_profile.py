"""The cross-package topics other nodes rely on stay stable in gnss_imu.yaml."""

import math
import importlib.util
from pathlib import Path

from launch import LaunchContext
from launch.substitutions import LaunchConfiguration
from launch_ros.utilities import evaluate_parameters
import yaml


PROFILE = Path(__file__).resolve().parents[1] / "config" / "gnss_imu.yaml"
LAUNCH = PROFILE.parents[1] / "launch" / "localization.launch.py"


def test_profile_keeps_the_shared_topic_names():
    parameters = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))[
        "gnss_imu_localization"
    ]["ros__parameters"]

    assert parameters["imu_topic"] == "/ad/sensors/imu/data"
    assert parameters["gnss_pose_topic"] == "/ad/localization/input/gnss_pose"
    assert parameters["wheel_speed_topic"] == "/ad/localization/input/wheel_speed"
    assert parameters["output_odometry_topic"] == (
        "/ad/localization/backends/gnss_imu/odometry"
    )
    assert parameters["publish_tf"] is False
    assert math.isclose(
        parameters["world_yaw_offset_rad"],
        math.radians(-1.346865944),
        abs_tol=1.0e-9,
    )


def test_gnss_imu_backend_uses_the_selected_imu_launch_configuration():
    spec = importlib.util.spec_from_file_location("localization_launch", LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    geometry = {
        "gnss_lever_arm_m": [0.0, 0.0, 1.0],
        "imu_frame": "imu_link",
        "imu_mount_rpy_rad": [0.0, 0.0, 0.0],
    }
    selected_topic = LaunchConfiguration("imu_topic")
    context = LaunchContext()
    context.launch_configurations["gnss_imu_config"] = str(PROFILE)
    context.launch_configurations["imu_topic"] = (
        "/ad/localization/input/imu_compatible"
    )

    node = module._make_estimator(
        "gnss_imu", geometry, selected_topic, LaunchConfiguration("eskf_imu_topic")
    )

    overrides = next(
        item
        for item in evaluate_parameters(context, node._Node__parameters)
        if isinstance(item, dict)
    )
    assert overrides["imu_topic"] == "/ad/localization/input/imu_compatible"
