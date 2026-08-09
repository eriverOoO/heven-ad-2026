"""Behavior tests for the FastLIO launch helpers and leaf launches."""

from __future__ import annotations

import importlib.util
import math
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
FIXED_MAP_PATH = PACKAGE_ROOT / "maps" / "cp14_to_cp15.pcd"
SENSOR_CONFIG_PATH = (
    REPOSITORY_ROOT / "ad_description" / "config" / "sensor_mounts.yaml"
)


def _load_launch(name: str):
    path = PACKAGE_ROOT / "launch" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_common = _load_launch("fastlio_launch.py")


def test_fastlio_fixed_map_is_a_packaged_localization_resource():
    assert FIXED_MAP_PATH.is_file()
    assert FIXED_MAP_PATH.stat().st_size > 0


def test_fastlio_launch_parses_optional_known_checkpoint_xy():
    parse_optional_xy = _common._parse_optional_xy

    assert parse_optional_xy("") == []
    assert parse_optional_xy("38.5,-480.25") == [38.5, -480.25]
    with pytest.raises(RuntimeError, match="two comma-separated"):
        parse_optional_xy("38.5")
    with pytest.raises(RuntimeError, match="finite"):
        parse_optional_xy("nan,-480.25")


def test_fastlio_launch_geometry_uses_full_sensor_pose_not_a_hard_coded_offset():
    geometry = _common._load_fastlio_geometry(
        SENSOR_CONFIG_PATH, "planned_centered_sensor_mounts"
    )

    assert geometry["imu_frame"] == "imu_link"
    assert geometry["lidar_frame"] == "lidar_link"
    assert geometry["base_frame"] == "base_link"
    assert geometry["gnss_lever_arm_m"] == pytest.approx(
        [0.0, 0.0, 1.0685], abs=1e-12
    )
    assert geometry["base_to_imu_T"] == pytest.approx(
        [0.0, 0.0, 1.0685], abs=1e-12
    )
    assert geometry["extrinsic_T"] == pytest.approx([1.5275, 0.0, 0.3], abs=1e-12)
    assert geometry["extrinsic_R"] == pytest.approx(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], abs=1e-12
    )


def test_fastlio_checked_in_default_config_uses_canonical_lidar_frame():
    parameters = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "fastlio.yaml").read_text(encoding="utf-8")
    )["/**"]["ros__parameters"]

    assert parameters["lidar_frame"] == "lidar_link"


def test_fastlio_launch_geometry_rotates_the_lidar_pose_into_the_imu_frame(tmp_path):
    sensor_config = tmp_path / "sensor_mounts.yaml"
    sensor_config.write_text(
        yaml.safe_dump(
            {
                "active_profile": "rotated",
                "coordinate_convention": {"base_frame": "base_link"},
                "profiles": {
                    "rotated": {
                        "sensors": {
                            "gps": {
                                "frame_id": "gps_link",
                                "position_m": {"x": 0.0, "y": 0.0, "z": 0.0},
                            },
                            "imu": {
                                "frame_id": "imu_link",
                                "position_m": {"x": 1.0, "y": 2.0, "z": 3.0},
                                "rpy_rad": {
                                    "roll": 0.0,
                                    "pitch": 0.0,
                                    "yaw": math.pi / 2.0,
                                },
                            },
                            "lidar": {
                                "frame_id": "lidar_link",
                                "position_m": {"x": 2.0, "y": 4.0, "z": 6.0},
                                "rpy_rad": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
                            },
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    geometry = _common._load_fastlio_geometry(sensor_config, "")

    assert geometry["base_to_imu_T"] == pytest.approx([1.0, 2.0, 3.0], abs=1e-12)
    assert geometry["base_to_imu_R"] == pytest.approx(
        [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0], abs=1e-12
    )
    assert geometry["extrinsic_T"] == pytest.approx([2.0, -1.0, 3.0], abs=1e-12)
    assert geometry["extrinsic_R"] == pytest.approx(
        [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0], abs=1e-12
    )


def _fastlio_context(
    mode: str,
    platform_profile: str = "morai",
    imu_topic: str = "/ad/sensors/imu/data",
) -> LaunchContext:
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "adapter_config": str(PACKAGE_ROOT / "config" / "localization.yaml"),
            "fastlio_config": str(PACKAGE_ROOT / "config" / "fastlio.yaml"),
            "status_topic": "/ad/vehicle/status",
            "autostart": "true",
            "initial_position_override_xy_m": "",
            "sensor_config": str(SENSOR_CONFIG_PATH),
            "sensor_profile": "",
            "platform_profile": platform_profile,
            "imu_topic": imu_topic,
            "map_output_path" if mode == "mapping" else "map_path": str(
                FIXED_MAP_PATH
            ),
        }
    )
    return context


@pytest.mark.parametrize(
    "launch_name",
    ["fastlio_mapping.launch.py", "fastlio_localization.launch.py"],
)
def test_fastlio_leaf_launches_expose_only_the_single_vehicle_interface(
    launch_name,
):
    description = _load_launch(launch_name).generate_launch_description()
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


@pytest.mark.parametrize("mode", ["mapping", "localization"])
def test_fastlio_launch_starts_the_adapter_and_one_fastlio_node(mode):
    context = _fastlio_context(mode)
    context.launch_configurations["namespace"] = "must_be_ignored"
    actions = _common._launch_setup(context, mode)

    adapter, fastlio = actions[0], actions[1]
    assert isinstance(adapter, LifecycleNode)
    assert isinstance(fastlio, Node)
    assert adapter._Node__node_namespace in ("", "/")
    assert fastlio._Node__node_namespace is None
    # Two lifecycle handlers auto-configure/activate the adapter.
    assert len(actions) == 4


@pytest.mark.parametrize("mode", ["mapping", "localization"])
def test_fastlio_launch_routes_selected_imu_to_adapter_and_backend(mode):
    context = _fastlio_context(
        mode, imu_topic="/ad/localization/input/imu_compatible"
    )

    adapter, fastlio = _common._launch_setup(context, mode)[:2]
    adapter_overrides = next(
        item
        for item in evaluate_parameters(context, adapter._Node__parameters)
        if isinstance(item, dict)
    )
    fastlio_overrides = next(
        item
        for item in evaluate_parameters(context, fastlio._Node__parameters)
        if isinstance(item, dict)
    )

    assert adapter_overrides["imu_topic"] == (
        "/ad/localization/input/imu_compatible"
    )
    assert fastlio_overrides["common.imu_topic"] == (
        "/ad/localization/input/imu_compatible"
    )


@pytest.mark.parametrize(
    ("platform_profile", "scan_timing_mode"),
    [("morai", "instantaneous"), ("real_hardware", "rolling")],
)
def test_fastlio_platform_profile_derives_scan_timing_mode(
    platform_profile, scan_timing_mode
):
    geometry = _common._load_fastlio_geometry(SENSOR_CONFIG_PATH, "")
    parameters = _common._fastlio_parameters(
        "mapping", geometry, platform_profile, "/tmp/x.pcd"
    )

    assert parameters["mode"] == "mapping"
    assert parameters["map_path"] == "/tmp/x.pcd"
    assert parameters["base_frame"] == geometry["base_frame"]
    assert parameters["extrinsic_T"] == geometry["extrinsic_T"]
    assert parameters["extrinsic_R"] == geometry["extrinsic_R"]
    assert parameters["base_to_imu_T"] == geometry["base_to_imu_T"]
    assert parameters["base_to_imu_R"] == geometry["base_to_imu_R"]
    assert parameters["preprocess.scan_timing_mode"] == scan_timing_mode


@pytest.mark.parametrize("profile", ["simulator", "hardware", ""])
def test_fastlio_launch_rejects_unknown_platform_profile(profile):
    with pytest.raises(RuntimeError, match="platform_profile"):
        _common._launch_setup(
            _fastlio_context("mapping", platform_profile=profile), "mapping"
        )


def test_generic_launch_rejects_the_fastlio_backend():
    generic_launch = _load_launch("localization.launch.py")
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "adapter_config": str(PACKAGE_ROOT / "config" / "localization.yaml"),
            "gnss_imu_config": str(PACKAGE_ROOT / "config" / "gnss_imu.yaml"),
            "eskf_config": str(PACKAGE_ROOT / "config" / "eskf.yaml"),
            "imu_quaternion_encoder_config": str(
                PACKAGE_ROOT / "config" / "imu_quaternion_encoder.yaml"
            ),
            "localization_manager_config": str(
                PACKAGE_ROOT / "config" / "localization_manager.yaml"
            ),
            "imu_quaternion_encoder_mode": "status_pose",
            "imu_topic": "/ad/sensors/imu/data",
            "eskf_imu_topic": "/ad/localization/input/eskf_imu",
            "localization_backend": "fastlio",
            "status_topic": "/ad/vehicle/status",
            "autostart": "true",
            "sensor_config": str(SENSOR_CONFIG_PATH),
            "sensor_profile": "current_front_sensor_mounts",
        }
    )
    with pytest.raises(RuntimeError, match="localization_backend"):
        generic_launch._launch_setup(context)
