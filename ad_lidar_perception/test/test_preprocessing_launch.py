from importlib.util import module_from_spec, spec_from_file_location
import math
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_prefix,
)
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.utilities import perform_substitutions
import pytest
import yaml


PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parent
LAUNCH = PACKAGE / "launch" / "preprocessing.launch.py"
ADAPTER_CONFIG = (
    PACKAGE / "config" / "preprocessing" / "point_layout_adapter.yaml"
)
DESKEW_CONFIG = (
    PACKAGE / "config" / "preprocessing" / "motion_deskew.yaml"
)
SELF_CROP_CONFIG = (
    PACKAGE / "config" / "preprocessing" / "self_crop.yaml"
)
VEHICLE_CONFIG = ROOT / "ad_description" / "config" / "vehicle_parameters.yaml"


def installed_prefix():
    try:
        return Path(get_package_prefix("ad_lidar_perception"))
    except PackageNotFoundError:
        candidates = (
            ROOT.parents[1] / "install" / "ad_lidar_perception",
            ROOT / "install" / "ad_lidar_perception",
        )
        return next(
            (candidate for candidate in candidates if candidate.is_dir()),
            candidates[0],
        )


INSTALLED_PREFIX = installed_prefix()
ADAPTER_EXECUTABLE = (
    INSTALLED_PREFIX
    / "lib"
    / "ad_lidar_perception"
    / "ad_point_layout_adapter_node"
)


class RecordingNode:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls.append(self)


def load_launch_module():
    spec = spec_from_file_location("ad_preprocessing_launch", LAUNCH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def launch_context(**overrides):
    values = {
        "platform_profile": "real_hardware",
        "deskew_enabled": "true",
        "deskew_mode": "3d",
        "self_crop_enabled": "true",
        "self_crop_input_reliable": "false",
        "point_layout_adapter_enabled": "true",
        "raw_input_topic": "/raw",
        "deskew_output_topic": "/deskewed",
        "self_crop_output_topic": "/cropped",
        "adapter_output_topic": "/xyzirc",
        "motion_deskew_config": str(DESKEW_CONFIG),
        "self_crop_config": str(SELF_CROP_CONFIG),
        "point_layout_adapter_config": str(ADAPTER_CONFIG),
        "vehicle_config": str(VEHICLE_CONFIG),
        "crop_clearance_m": "0.20",
    }
    values.update({key: str(value) for key, value in overrides.items()})
    context = LaunchContext()
    context.launch_configurations.update(values)
    return context


def launch_nodes(monkeypatch, **overrides):
    module = load_launch_module()
    RecordingNode.calls.clear()
    monkeypatch.setattr(module, "Node", RecordingNode)
    actions = module._launch_setup(launch_context(**overrides))
    assert actions == RecordingNode.calls
    return [action.kwargs for action in actions]


def write_vehicle(tmp_path, mutate=None):
    document = yaml.safe_load(VEHICLE_CONFIG.read_text(encoding="utf-8"))
    if mutate is not None:
        mutate(document)
    path = tmp_path / "vehicle_parameters.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def test_launch_declares_pipeline_arguments_and_installed_share_defaults(
    monkeypatch,
):
    module = load_launch_module()
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda package: str(
            PACKAGE if package == "ad_lidar_perception" else ROOT / package
        ),
    )

    description = module.generate_launch_description()
    arguments = {
        entity.name: entity.default_value
        for entity in description.entities
        if isinstance(entity, DeclareLaunchArgument)
    }

    assert set(arguments) == {
        "platform_profile",
        "deskew_enabled",
        "deskew_mode",
        "self_crop_enabled",
        "self_crop_input_reliable",
        "point_layout_adapter_enabled",
        "raw_input_topic",
        "deskew_output_topic",
        "self_crop_output_topic",
        "adapter_output_topic",
        "motion_deskew_config",
        "self_crop_config",
        "point_layout_adapter_config",
        "vehicle_config",
        "crop_clearance_m",
    }
    context = LaunchContext()
    expected = {
        "platform_profile": "morai",
        "deskew_enabled": "false",
        "deskew_mode": "3d",
        "self_crop_enabled": "true",
        "self_crop_input_reliable": "false",
        "point_layout_adapter_enabled": "true",
        "raw_input_topic": "/ad/sensors/lidar/points",
        "deskew_output_topic": "/ad/perception/lidar/deskewed",
        "self_crop_output_topic": "/ad/perception/lidar/cropped",
        "adapter_output_topic": "/ad/perception/lidar/points_xyzirc",
        "motion_deskew_config": str(DESKEW_CONFIG),
        "self_crop_config": str(SELF_CROP_CONFIG),
        "point_layout_adapter_config": str(ADAPTER_CONFIG),
        "vehicle_config": str(VEHICLE_CONFIG),
        "crop_clearance_m": "0.20",
    }
    assert {
        name: perform_substitutions(context, value)
        for name, value in arguments.items()
    } == expected


def test_default_pipeline_launches_deskew_crop_and_adapter_with_exact_routing(
    monkeypatch,
):
    nodes = launch_nodes(
        monkeypatch, platform_profile="real_hardware", deskew_enabled="true"
    )

    assert [node["executable"] for node in nodes] == [
        "ad_motion_deskew_node",
        "ad_self_crop_filter_node",
        "ad_point_layout_adapter_node",
    ]
    assert nodes[0]["parameters"][1] == {
        "topics.input": "/raw",
        "topics.output": "/deskewed",
        "deskew_mode": "3d",
    }
    assert nodes[1]["parameters"][2] == {
        "topics.input": "/deskewed",
        "topics.output": "/cropped",
        "input_reliable": False,
    }
    assert nodes[2]["parameters"][1] == {
        "topics.input": "/cropped",
        "topics.output": "/xyzirc",
    }
    assert all(node["output"] == "screen" for node in nodes)
    assert all("namespace" not in node for node in nodes)


@pytest.mark.parametrize(
    ("deskew_enabled", "crop_enabled", "executables", "adapter_input"),
    [
        (
            "false",
            "true",
            ["ad_self_crop_filter_node", "ad_point_layout_adapter_node"],
            "/cropped",
        ),
        (
            "true",
            "false",
            ["ad_motion_deskew_node", "ad_point_layout_adapter_node"],
            "/deskewed",
        ),
        ("false", "false", ["ad_point_layout_adapter_node"], "/raw"),
    ],
)
def test_disabled_stages_launch_no_node_and_route_from_previous_real_topic(
    monkeypatch, deskew_enabled, crop_enabled, executables, adapter_input
):
    nodes = launch_nodes(
        monkeypatch,
        platform_profile="real_hardware",
        deskew_enabled=deskew_enabled,
        self_crop_enabled=crop_enabled,
        deskew_mode="2d",
    )

    assert [node["executable"] for node in nodes] == executables
    assert nodes[-1]["parameters"][1]["topics.input"] == adapter_input
    if crop_enabled == "true":
        crop = next(
            node
            for node in nodes
            if node["executable"] == "ad_self_crop_filter_node"
        )
        expected_crop_input = "/deskewed" if deskew_enabled == "true" else "/raw"
        assert crop["parameters"][2]["topics.input"] == expected_crop_input
        assert crop["parameters"][2]["input_reliable"] is False


def test_replay_can_request_reliable_self_crop_input(monkeypatch):
    nodes = launch_nodes(
        monkeypatch,
        deskew_enabled="false",
        self_crop_input_reliable="true",
    )
    crop = next(
        node
        for node in nodes
        if node["executable"] == "ad_self_crop_filter_node"
    )

    assert crop["parameters"][2]["input_reliable"] is True


def test_morai_profile_prohibits_motion_deskew_before_launching_any_node(
    monkeypatch,
):
    module = load_launch_module()
    RecordingNode.calls.clear()
    monkeypatch.setattr(module, "Node", RecordingNode)

    with pytest.raises(RuntimeError, match="MORAI.*deskew.*prohibited"):
        module._launch_setup(
            launch_context(platform_profile="morai", deskew_enabled="true")
        )

    assert RecordingNode.calls == []


@pytest.mark.parametrize("profile", ["simulator", "hardware", ""])
def test_unknown_platform_profile_is_rejected(profile, monkeypatch):
    module = load_launch_module()
    RecordingNode.calls.clear()
    monkeypatch.setattr(module, "Node", RecordingNode)

    with pytest.raises(RuntimeError, match="platform_profile"):
        module._launch_setup(launch_context(platform_profile=profile))

    assert RecordingNode.calls == []


@pytest.mark.parametrize(
    ("overrides", "topic_argument"),
    [
        ({"raw_input_topic": "relative"}, "raw_input_topic"),
        ({"deskew_output_topic": "/bad topic"}, "deskew_output_topic"),
        ({"self_crop_output_topic": "/bad//topic"}, "self_crop_output_topic"),
        ({"adapter_output_topic": "/9invalid"}, "adapter_output_topic"),
    ],
)
def test_launch_rejects_invalid_active_full_topic_names(
    monkeypatch, overrides, topic_argument
):
    module = load_launch_module()
    monkeypatch.setattr(module, "Node", RecordingNode)

    with pytest.raises(RuntimeError, match=topic_argument):
        module._launch_setup(launch_context(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"deskew_output_topic": "/raw"},
        {"self_crop_output_topic": "/deskewed"},
        {"adapter_output_topic": "/cropped"},
    ],
)
def test_launch_rejects_adjacent_duplicate_topics(monkeypatch, overrides):
    module = load_launch_module()
    monkeypatch.setattr(module, "Node", RecordingNode)

    with pytest.raises(RuntimeError, match="duplicate active topic"):
        module._launch_setup(launch_context(**overrides))


def test_launch_rejects_nonadjacent_raw_to_adapter_feedback(monkeypatch):
    module = load_launch_module()
    monkeypatch.setattr(module, "Node", RecordingNode)

    with pytest.raises(RuntimeError, match="duplicate active topic"):
        module._launch_setup(launch_context(adapter_output_topic="/raw"))


def test_disabled_stage_outputs_are_irrelevant_to_active_topic_validation(
    monkeypatch,
):
    nodes = launch_nodes(
        monkeypatch,
        deskew_enabled="false",
        self_crop_enabled="false",
        deskew_output_topic="not a full topic",
        self_crop_output_topic="/raw",
    )

    assert [node["executable"] for node in nodes] == [
        "ad_point_layout_adapter_node"
    ]
    assert nodes[0]["parameters"][1] == {
        "topics.input": "/raw",
        "topics.output": "/xyzirc",
    }


def test_disabled_layout_adapter_launches_no_adapter_node(monkeypatch):
    nodes = launch_nodes(
        monkeypatch,
        deskew_enabled="false",
        self_crop_enabled="true",
        point_layout_adapter_enabled="false",
        adapter_output_topic="ignored invalid topic",
    )

    assert [node["executable"] for node in nodes] == [
        "ad_self_crop_filter_node"
    ]
    assert nodes[0]["parameters"][2] == {
        "topics.input": "/raw",
        "topics.output": "/cropped",
        "input_reliable": False,
    }


def test_all_disabled_preprocessing_stages_launch_no_nodes(monkeypatch):
    nodes = launch_nodes(
        monkeypatch,
        deskew_enabled="false",
        self_crop_enabled="false",
        point_layout_adapter_enabled="false",
        deskew_output_topic="ignored invalid topic",
        self_crop_output_topic="also ignored",
        adapter_output_topic="also ignored",
    )

    assert nodes == []


def test_vehicle_description_bounds_feed_only_self_crop(monkeypatch):
    nodes = launch_nodes(monkeypatch)
    crop_parameters = nodes[1]["parameters"][1]

    assert crop_parameters == {
        "base_frame": "base_link",
        "bounds.min_x_m": pytest.approx(-0.990),
        "bounds.max_x_m": pytest.approx(4.045),
        "bounds.min_y_m": pytest.approx(-1.145),
        "bounds.max_y_m": pytest.approx(1.145),
        "bounds.min_z_m": pytest.approx(-0.200),
        "bounds.max_z_m": pytest.approx(1.805),
    }
    assert len(nodes[2]["parameters"]) == 2
    assert not any(
        "bounds." in key or "ego_crop" in key
        for key in nodes[2]["parameters"][1]
    )


def test_changed_vehicle_geometry_changes_all_crop_bounds(monkeypatch, tmp_path):
    def mutate(document):
        geometry = document["vehicle"]["geometry"]
        geometry["front_bumper_x_m"] = 5.0
        geometry["rear_bumper_x_m"] = -2.0
        geometry["width_m"] = 2.4
        geometry["height_m"] = 2.0

    vehicle = write_vehicle(tmp_path, mutate)
    crop = launch_nodes(
        monkeypatch, vehicle_config=vehicle, crop_clearance_m="0.30"
    )[1]["parameters"][1]
    assert crop == {
        "base_frame": "base_link",
        "bounds.min_x_m": pytest.approx(-2.3),
        "bounds.max_x_m": pytest.approx(5.3),
        "bounds.min_y_m": pytest.approx(-1.5),
        "bounds.max_y_m": pytest.approx(1.5),
        "bounds.min_z_m": pytest.approx(-0.3),
        "bounds.max_z_m": pytest.approx(2.3),
    }


@pytest.mark.parametrize("mode", ["planar", "", "3D"])
def test_launch_rejects_invalid_deskew_mode_even_when_deskew_is_disabled(
    monkeypatch, mode
):
    module = load_launch_module()
    monkeypatch.setattr(module, "Node", RecordingNode)
    with pytest.raises(RuntimeError, match="deskew_mode"):
        module._launch_setup(
            launch_context(deskew_enabled="false", deskew_mode=mode)
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("deskew_enabled", "yes"),
        ("deskew_enabled", "1"),
        ("self_crop_enabled", "enabled"),
        ("point_layout_adapter_enabled", "auto"),
    ],
)
def test_launch_rejects_ambiguous_enable_values(monkeypatch, name, value):
    module = load_launch_module()
    monkeypatch.setattr(module, "Node", RecordingNode)
    with pytest.raises(RuntimeError, match=name):
        module._launch_setup(launch_context(**{name: value}))


@pytest.mark.parametrize(
    ("clearance", "message"),
    [
        ("-0.1", "nonnegative"),
        ("nan", "finite"),
        ("inf", "finite"),
        ("true", "numeric"),
        ("not-a-number", "numeric"),
    ],
)
def test_invalid_clearance_fails_closed_when_crop_enabled(
    monkeypatch, clearance, message
):
    module = load_launch_module()
    monkeypatch.setattr(module, "Node", RecordingNode)
    with pytest.raises(RuntimeError, match=message):
        module._launch_setup(launch_context(crop_clearance_m=clearance))


def test_disabled_crop_does_not_require_vehicle_file(monkeypatch, tmp_path):
    nodes = launch_nodes(
        monkeypatch,
        self_crop_enabled="false",
        vehicle_config=tmp_path / "missing.yaml",
    )
    assert "ad_self_crop_filter_node" not in {
        node["executable"] for node in nodes
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d["vehicle"]["geometry"].pop("width_m"), "vehicle"),
        (
            lambda d: d["vehicle"]["geometry"].__setitem__("width_m", True),
            "numeric",
        ),
        (
            lambda d: d["vehicle"]["geometry"].__setitem__(
                "height_m", math.inf
            ),
            "finite",
        ),
        (
            lambda d: d["vehicle"]["geometry"].__setitem__("width_m", -1.0),
            "positive",
        ),
        (
            lambda d: d["vehicle"]["coordinate_convention"].__setitem__(
                "base_frame", "vehicle"
            ),
            "base_link",
        ),
    ],
)
def test_invalid_vehicle_values_fail_closed(
    monkeypatch, tmp_path, mutate, message
):
    module = load_launch_module()
    monkeypatch.setattr(module, "Node", RecordingNode)
    vehicle = write_vehicle(tmp_path, mutate)
    with pytest.raises(RuntimeError, match=message):
        module._launch_setup(launch_context(vehicle_config=vehicle))


def test_preprocessing_configs_separate_crop_from_adapter():
    adapter = yaml.safe_load(ADAPTER_CONFIG.read_text(encoding="utf-8"))[
        "/**"
    ]["ros__parameters"]
    crop = yaml.safe_load(SELF_CROP_CONFIG.read_text(encoding="utf-8"))[
        "/**"
    ]["ros__parameters"]

    assert adapter == {
        "topics": {
            "input": "/ad/perception/lidar/cropped",
            "output": "/ad/perception/lidar/points_xyzirc",
        },
        "intensity": {
            "scale": 1.0,
            "offset": 0.0,
            "nonfinite_value": 0,
        },
        "return_type": 0,
    }
    assert crop == {
        "topics": {
            "input": "/ad/perception/lidar/deskewed",
            "output": "/ad/perception/lidar/cropped",
        },
        "base_frame": "base_link",
        "bounds": {
            "min_x_m": -0.990,
            "max_x_m": 4.045,
            "min_y_m": -1.145,
            "max_y_m": 1.145,
            "min_z_m": -0.200,
            "max_z_m": 1.805,
        },
        "transform_timeout_sec": 0.1,
    }


def test_motion_deskew_configuration_locks_topics_modes_and_safety_limits():
    parameters = yaml.safe_load(DESKEW_CONFIG.read_text(encoding="utf-8"))[
        "/**"
    ]["ros__parameters"]
    assert parameters["topics"] == {
        "input": "/ad/sensors/lidar/points",
        "output": "/ad/perception/lidar/deskewed",
        "imu": "/ad/sensors/imu/data",
        "wheel": "/ad/localization/input/wheel_speed",
    }
    assert parameters["base_frame"] == "base_link"
    assert parameters["deskew_mode"] == "3d"
    assert parameters["limits"] == {
        "maximum_scan_duration_sec": 0.20,
        "maximum_point_count": 300000,
        "maximum_imu_gap_sec": 0.12,
        "maximum_wheel_gap_sec": 0.20,
        "history_age_sec": 1.0,
        "pending_depth": 4,
        "pending_timeout_sec": 0.15,
        "integration_substep_sec": 0.005,
    }


def test_normalized_sensor_inputs_share_the_receipt_time_policy():
    for name in ("competition.yaml", "loopback.yaml", "tunnel_fastlio.yaml"):
        bridge = yaml.safe_load(
            (ROOT / "ad_morai_bridge" / "config" / name).read_text(
                encoding="utf-8"
            )
        )
        parameters = bridge["/**"]["ros__parameters"]
        assert parameters["timestamp_mode"] == "arrival"
        assert parameters["source_stamp_tolerance_sec"] == pytest.approx(1.0)
    localization = yaml.safe_load(
        (ROOT / "ad_localization" / "config" / "localization.yaml").read_text(
            encoding="utf-8"
        )
    )
    # The adapter consumes the bridge-normalized header. It must not bypass the
    # common clock-domain gate by substituting the raw device field again.
    assert (
        localization["ad_localization"]["ros__parameters"][
            "wheel_use_device_timestamp"
        ]
        is False
    )


def test_manifest_retains_launch_vehicle_and_tf_dependencies():
    root = ET.parse(PACKAGE / "package.xml").getroot()
    common_dependencies = {element.text for element in root.findall("depend")}
    runtime_dependencies = {element.text for element in root.findall("exec_depend")}
    test_dependencies = {element.text for element in root.findall("test_depend")}
    assert {
        "ad_description",
        "ament_index_python",
        "launch",
        "launch_ros",
        "python3-yaml",
        "rclpy",
    } <= runtime_dependencies
    assert {"tf2", "tf2_ros"} <= common_dependencies
    assert {"ament_cmake_gtest", "ament_cmake_pytest", "python3-yaml"} <= (
        test_dependencies
    )


def test_installed_package_exports_self_crop_to_downstream_consumer(tmp_path):
    source = tmp_path / "main.cpp"
    source.write_text(
        """
#include <ad_lidar_perception/preprocessing/point_layout_converter.hpp>
#include <ad_lidar_perception/preprocessing/self_crop_filter.hpp>
#include <ad_lidar_perception/preprocessing/xyzirt_layout.hpp>

#include <stdexcept>

int main()
{
  ad_lidar_perception::preprocessing::ConverterConfig converter;
  ad_lidar_perception::preprocessing::SelfCropBounds bounds;
  sensor_msgs::msg::PointCloud2 cloud;
  try {
    const auto result = ad_lidar_perception::preprocessing::crop_self_points(
      cloud, bounds, ad_lidar_perception::preprocessing::RigidTransform3{});
    return static_cast<int>(result.cloud.width);
  } catch (const std::invalid_argument &) {
    return converter.return_type;
  }
}
""".lstrip(),
        encoding="utf-8",
    )
    cmake_lists = tmp_path / "CMakeLists.txt"
    cmake_lists.write_text(
        """
cmake_minimum_required(VERSION 3.8)
project(ad_lidar_perception_downstream LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(ad_lidar_perception REQUIRED)
if(NOT "sensor_msgs" IN_LIST ad_lidar_perception_DEPENDENCIES)
  message(FATAL_ERROR "ad_lidar_perception does not export sensor_msgs")
endif()
add_executable(downstream main.cpp)
ament_target_dependencies(downstream ad_lidar_perception)
""".lstrip(),
        encoding="utf-8",
    )
    build = tmp_path / "build"
    environment = os.environ.copy()
    environment["CMAKE_PREFIX_PATH"] = os.pathsep.join(
        filter(
            None,
            [str(INSTALLED_PREFIX), environment.get("CMAKE_PREFIX_PATH", "")],
        )
    )
    configure = subprocess.run(
        ["cmake", "-S", str(tmp_path), "-B", str(build)],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    assert configure.returncode == 0, configure.stdout + configure.stderr
    compile_result = subprocess.run(
        ["cmake", "--build", str(build), "--parallel", "2"],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )
    assert compile_result.returncode == 0, (
        compile_result.stdout + compile_result.stderr
    )


@pytest.mark.parametrize(
    ("input_topic", "output_topic", "remap_arguments"),
    [
        ("points", "/points", []),
        ("/input", "/output", ["-r", "/output:=/input"]),
    ],
)
def test_adapter_rejects_topics_resolving_to_same_endpoint_without_crop_params(
    input_topic, output_topic, remap_arguments
):
    command = [
        str(ADAPTER_EXECUTABLE),
        "--ros-args",
        *remap_arguments,
        "-p",
        f"topics.input:={input_topic}",
        "-p",
        f"topics.output:={output_topic}",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("adapter remained running with colliding resolved topics")
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "topics must differ after ROS name resolution" in output
