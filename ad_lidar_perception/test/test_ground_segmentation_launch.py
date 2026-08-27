from importlib.util import module_from_spec, spec_from_file_location
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.utilities import perform_substitutions
import pytest
import yaml


PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parent
LAUNCH = PACKAGE / "launch" / "ground_segmentation.launch.py"
SENSOR_CONFIG = ROOT / "ad_description" / "config" / "sensor_mounts.yaml"
GROUND_CONFIG = (
    PACKAGE / "config" / "preprocessing" / "ground_segmentation.yaml"
)
LEVELER_CONFIG = (
    PACKAGE / "config" / "preprocessing" / "gravity_leveler.yaml"
)
RANSAC_CONFIG = (
    PACKAGE / "config" / "preprocessing" / "ransac_ground_filter.yaml"
)
LEVELER_SOURCE = PACKAGE / "src" / "preprocessing" / "gravity_leveler_node.cpp"
PATCHWORK_COMMIT = "3e6903a1d5537a4cc2ace897b0bbb98a92d6014c"


class RecordingNode:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls.append(self)


def load_launch_module():
    spec = spec_from_file_location("ad_ground_segmentation_launch", LAUNCH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def launch_context(**overrides):
    values = {
        "backend": "patchwork",
        "patchwork_leveling_enabled": "true",
        "cropped_input_topic": "/cropped",
        "leveled_output_topic": "/leveled",
        "gravity_leveler_config": str(LEVELER_CONFIG),
        "ground_config": str(GROUND_CONFIG),
        "ransac_config": str(RANSAC_CONFIG),
        "sensor_config": str(SENSOR_CONFIG),
        "sensor_profile": "",
        "odom_frame": "odom",
        "base_frame": "base_link",
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


def write_sensor_config(tmp_path, mutate):
    document = yaml.safe_load(SENSOR_CONFIG.read_text(encoding="utf-8"))
    mutate(document)
    path = tmp_path / "sensor_mounts.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def test_default_launch_levels_before_patchwork_with_exact_frame_and_topics(
    monkeypatch,
):
    nodes = launch_nodes(monkeypatch)

    assert [node["executable"] for node in nodes] == [
        "ad_gravity_leveler_node",
        "patchworkpp_node",
    ]
    leveler, patchwork = nodes
    assert leveler["package"] == "ad_lidar_perception"
    assert leveler["name"] == "ad_gravity_leveler"
    assert leveler["parameters"][0].perform(launch_context()) == str(
        LEVELER_CONFIG
    )
    assert leveler["parameters"][1] == {
        "topics.input": "/cropped",
        "topics.output": "/leveled",
        "expected_input_frame": "lidar_link",
        "output_frame": "lidar_leveled_frame",
        "odom_frame": "odom",
        "base_frame": "base_link",
    }
    assert patchwork["package"] == "patchworkpp"
    assert patchwork["name"] == "ad_ground_segmentation"
    assert patchwork["remappings"] == [
        ("pointcloud_topic", "/leveled"),
        ("/patchworkpp/cloud", "/ad/perception/lidar/cloud"),
        ("/patchworkpp/ground", "/ad/perception/lidar/ground"),
        ("/patchworkpp/nonground", "/ad/perception/lidar/nonground"),
    ]
    assert patchwork["parameters"][1] == {
        "base_frame": "lidar_leveled_frame",
        "sensor_height": 1.7685,
    }
    assert all(node["output"] == "screen" for node in nodes)
    assert all("namespace" not in node for node in nodes)


def test_bypass_omits_leveler_and_routes_patchwork_to_truthful_sensor_frame(
    monkeypatch,
):
    nodes = launch_nodes(monkeypatch, patchwork_leveling_enabled="false")

    assert [node["executable"] for node in nodes] == ["patchworkpp_node"]
    patchwork = nodes[0]
    assert patchwork["remappings"][0] == (
        "pointcloud_topic",
        "/cropped",
    )
    assert patchwork["parameters"][1] == {
        "base_frame": "lidar_link",
        "sensor_height": 1.7685,
    }


def test_bypass_ignores_unused_leveler_topic_and_frames(monkeypatch):
    nodes = launch_nodes(
        monkeypatch,
        patchwork_leveling_enabled="false",
        leveled_output_topic="not a ROS topic",
        odom_frame="/unused odom",
        base_frame="unused base",
    )

    assert [node["executable"] for node in nodes] == ["patchworkpp_node"]
    assert nodes[0]["remappings"][0] == ("pointcloud_topic", "/cropped")


def test_generic_profile_frame_drives_leveler_output_and_patchwork(
    monkeypatch, tmp_path
):
    sensor_config = write_sensor_config(
        tmp_path,
        lambda document: document["profiles"][document["active_profile"]][
            "sensors"
        ]["lidar"].__setitem__("frame_id", "roof_lidar_link"),
    )
    nodes = launch_nodes(monkeypatch, sensor_config=sensor_config)

    assert nodes[0]["parameters"][1]["expected_input_frame"] == "roof_lidar_link"
    assert nodes[0]["parameters"][1]["output_frame"] == "roof_lidar_leveled_frame"
    assert nodes[1]["parameters"][1]["base_frame"] == "roof_lidar_leveled_frame"


@pytest.mark.parametrize(
    ("profile", "expected_height"),
    [
        ("", 1.7685),
        ("current_front_sensor_mounts", 1.7685),
        ("planned_centered_sensor_mounts", 1.3685),
    ],
)
@pytest.mark.parametrize("enabled", ["true", "false"])
def test_selected_profile_height_is_exact_on_enabled_and_bypass_routes(
    monkeypatch, profile, expected_height, enabled
):
    nodes = launch_nodes(
        monkeypatch,
        sensor_profile=profile,
        patchwork_leveling_enabled=enabled,
    )
    assert nodes[-1]["parameters"][1]["sensor_height"] == expected_height


def test_nonzero_finite_mount_roll_pitch_and_yaw_are_supported(monkeypatch, tmp_path):
    def mutate(document):
        rpy = document["profiles"][document["active_profile"]]["sensors"][
            "lidar"
        ]["rpy_rad"]
        rpy.update({"roll": 0.12, "pitch": -0.23, "yaw": 0.34})

    nodes = launch_nodes(
        monkeypatch, sensor_config=write_sensor_config(tmp_path, mutate)
    )
    assert nodes[-1]["parameters"][1]["sensor_height"] == 1.7685


def test_launch_declares_leveling_arguments_and_installed_share_defaults(monkeypatch):
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
        "backend",
        "patchwork_leveling_enabled",
        "cropped_input_topic",
        "leveled_output_topic",
        "gravity_leveler_config",
        "ground_config",
        "ransac_config",
        "sensor_config",
        "sensor_profile",
        "odom_frame",
        "base_frame",
    }
    context = LaunchContext()
    assert {
        name: perform_substitutions(context, default)
        for name, default in arguments.items()
    } == {
        "backend": "patchwork",
        "patchwork_leveling_enabled": "false",
        "cropped_input_topic": "/ad/perception/lidar/cropped",
        "leveled_output_topic": "/ad/perception/lidar/leveled",
        "gravity_leveler_config": str(LEVELER_CONFIG),
        "ground_config": str(GROUND_CONFIG),
        "ransac_config": str(RANSAC_CONFIG),
        "sensor_config": str(SENSOR_CONFIG),
        "sensor_profile": "",
        "odom_frame": "odom",
        "base_frame": "base_link",
    }


@pytest.mark.parametrize("value", ["yes", "1", "enabled", "", "TRUE-ish"])
def test_launch_rejects_ambiguous_leveling_boolean(monkeypatch, value):
    module = load_launch_module()
    monkeypatch.setattr(module, "Node", RecordingNode)
    with pytest.raises(RuntimeError, match="patchwork_leveling_enabled"):
        module._launch_setup(
            launch_context(patchwork_leveling_enabled=value)
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"cropped_input_topic": ""}, "cropped_input_topic"),
        ({"leveled_output_topic": ""}, "leveled_output_topic"),
        ({"cropped_input_topic": "relative"}, "valid full ROS topic"),
        ({"leveled_output_topic": "/bad topic"}, "valid full ROS topic"),
        (
            {
                "cropped_input_topic": "/same",
                "leveled_output_topic": "/same",
            },
            "duplicate active topic",
        ),
        (
            {"cropped_input_topic": "/ad/perception/lidar/ground"},
            "duplicate active topic",
        ),
        (
            {"leveled_output_topic": "/ad/perception/lidar/nonground"},
            "duplicate active topic",
        ),
        ({"odom_frame": "/odom"}, "relative frame"),
        ({"base_frame": "base link"}, "relative frame"),
        ({"base_frame": "vehicle"}, "sensor profile base frame"),
    ],
)
def test_launch_rejects_invalid_topics_and_frames(monkeypatch, overrides, message):
    module = load_launch_module()
    monkeypatch.setattr(module, "Node", RecordingNode)
    with pytest.raises(RuntimeError, match=message):
        module._launch_setup(launch_context(**overrides))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda d: d["profiles"][d["active_profile"]]["sensors"][
                "lidar"
            ].__setitem__("frame_id", "/lidar"),
            "relative",
        ),
        (
            lambda d: d["profiles"][d["active_profile"]]["sensors"][
                "lidar"
            ].__setitem__("frame_id", None),
            "relative",
        ),
        (
            lambda d: d["profiles"][d["active_profile"]]["sensors"][
                "lidar"
            ]["position_m"].__setitem__("x", math.inf),
            "finite",
        ),
        (
            lambda d: d["profiles"][d["active_profile"]]["sensors"][
                "lidar"
            ]["position_m"].__setitem__("z", -0.3685),
            "positive finite",
        ),
        (
            lambda d: d["profiles"][d["active_profile"]]["sensors"][
                "lidar"
            ]["rpy_rad"].__setitem__("pitch", math.nan),
            "finite",
        ),
        (
            lambda d: d.__setitem__("active_profile", "missing"),
            "invalid Patchwork",
        ),
    ],
)
def test_invalid_sensor_profile_fails_closed(monkeypatch, tmp_path, mutate, message):
    module = load_launch_module()
    monkeypatch.setattr(module, "Node", RecordingNode)
    sensor_config = write_sensor_config(tmp_path, mutate)
    with pytest.raises(RuntimeError, match=message):
        module._launch_setup(launch_context(sensor_config=sensor_config))


def test_ground_segmentation_defaults_to_patchworkpp_with_pinned_thresholds():
    parameters = yaml.safe_load(GROUND_CONFIG.read_text(encoding="utf-8"))[
        "/**"
    ]["ros__parameters"]

    assert parameters == {
        "algorithm": "patchworkpp",
        "num_iter": 3,
        "num_lpr": 20,
        "num_min_pts": 5,
        "th_seeds": 0.35,
        "th_dist": 0.18,
        "th_seeds_v": 0.25,
        "th_dist_v": 0.1,
        "max_range": 80.0,
        "min_range": 3.0,
        "uprightness_thr": 0.707,
        "verbose": False,
    }


def test_ransac_backend_replaces_patchwork_and_keeps_the_split_topics(
    monkeypatch,
):
    nodes = launch_nodes(
        monkeypatch, backend="ransac", patchwork_leveling_enabled="false"
    )

    assert [node["executable"] for node in nodes] == [
        "ransac_ground_filter_node"
    ]
    ransac = nodes[0]
    assert ransac["package"] == "autoware_ground_segmentation"
    assert ransac["name"] == "ad_ground_segmentation"
    assert ransac["parameters"][0].perform(launch_context()) == str(
        RANSAC_CONFIG
    )
    assert ransac["parameters"][1] == {"base_frame": "lidar_link"}
    assert ransac["remappings"] == [
        ("input", "/cropped"),
        ("output", "/ad/perception/lidar/nonground"),
        ("debug/ground/pointcloud", "/ad/perception/lidar/ground"),
    ]


def test_ransac_backend_still_consumes_the_leveled_cloud(monkeypatch):
    nodes = launch_nodes(monkeypatch, backend="ransac")

    assert [node["executable"] for node in nodes] == [
        "ad_gravity_leveler_node",
        "ransac_ground_filter_node",
    ]
    assert nodes[-1]["parameters"][1] == {
        "base_frame": "lidar_leveled_frame"
    }
    assert nodes[-1]["remappings"][0] == ("input", "/leveled")


@pytest.mark.parametrize("value", ["patchworkpp", "", "RANSAC", "pcl"])
def test_launch_rejects_an_unknown_backend(monkeypatch, value):
    module = load_launch_module()
    monkeypatch.setattr(module, "Node", RecordingNode)

    with pytest.raises(RuntimeError, match="backend must be one of"):
        module._launch_setup(launch_context(backend=value))


def test_ransac_defaults_match_pinned_autoware_contract():
    parameters = yaml.safe_load(RANSAC_CONFIG.read_text(encoding="utf-8"))[
        "/**"
    ]["ros__parameters"]

    assert parameters == {
        "unit_axis": "z",
        "max_iterations": 200,
        "min_trial": 1000,
        "min_points": 500,
        "outlier_threshold": 0.15,
        "plane_slope_threshold": 12.0,
        "voxel_size_x": 0.10,
        "voxel_size_y": 0.10,
        "voxel_size_z": 0.10,
        "height_threshold": 0.18,
        "debug": True,
        "publish_processing_time_detail": False,
    }
    # Launch owns base_frame and all topics so the enabled/bypass route cannot
    # be contradicted by the shared parameter file.
    assert "base_frame" not in parameters


def test_gravity_leveler_defaults_do_not_claim_a_sensor_frame():
    parameters = yaml.safe_load(LEVELER_CONFIG.read_text(encoding="utf-8"))[
        "/**"
    ]["ros__parameters"]

    assert parameters == {
        "topics": {
            "input": "/ad/perception/lidar/cropped",
            "output": "/ad/perception/lidar/leveled",
        },
        "odom_frame": "odom",
        "base_frame": "base_link",
        "expected_input_frame": "",
        "output_frame": "",
        "transform_timeout_sec": 0.1,
    }


def test_leveler_node_contract_is_timestamped_fail_closed_and_broadcasts_first():
    source = LEVELER_SOURCE.read_text(encoding="utf-8")

    assert "lookupTransform" in source
    assert "rclcpp::Time(input.header.stamp)" in source
    assert source.count("lookupTransform") == 2
    assert "input.header.frame_id != expected_input_frame_" in source
    assert "cloud dropped" in source
    assert "tf2_ros::TransformBroadcaster" in source
    assert source.index("sendTransform") < source.index("publisher_->publish")


def test_package_declares_sensor_profile_runtime_dependency():
    root = ET.parse(PACKAGE / "package.xml").getroot()
    dependencies = {element.text for element in root.findall("exec_depend")}

    assert {
        "ad_description",
        "autoware_ground_segmentation",
        "python3-yaml",
    } <= dependencies


def test_external_repository_is_pinned_to_a_release():
    manifest = yaml.safe_load(
        (ROOT / "dependencies.repos").read_text(encoding="utf-8")
    )

    assert manifest["repositories"]["patchwork-plusplus"] == {
        "type": "git",
        "url": "https://github.com/url-kaist/patchwork-plusplus.git",
        "version": PATCHWORK_COMMIT,
    }
