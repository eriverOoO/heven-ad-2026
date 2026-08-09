from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import xml.etree.ElementTree as ET

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.utilities import perform_substitutions
from launch_ros.actions import Node
import yaml


PACKAGE = Path(__file__).resolve().parents[1]


def _load_launch_module(name):
    launch_path = PACKAGE / "launch" / name
    spec = spec_from_file_location(name.replace(".", "_"), launch_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_camera_launch_aggregates_traffic_and_dynamic_obstacle_controls(monkeypatch):
    module = _load_launch_module("camera_perception.launch.py")
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda _package: str(PACKAGE),
    )

    description = module.generate_launch_description()
    arguments = [
        entity
        for entity in description.entities
        if isinstance(entity, DeclareLaunchArgument)
    ]
    nodes = [
        entity for entity in description.entities if isinstance(entity, Node)
    ]

    assert {
        argument.name for argument in arguments
    } >= {
        "traffic_params_file",
        "traffic_light_image_topic",
        "traffic_light_detections_topic",
        "traffic_light_status_topic",
        "traffic_light_model_path",
        "traffic_light_yolov7_repository_path",
        "traffic_light_enable_visualizer",
        "traffic_light_show_window",
        "dynamic_obstacle_params_file",
        "dynamic_obstacle_image_topic",
        "dynamic_obstacle_detections_topic",
        "dynamic_obstacle_model_path",
        "dynamic_obstacle_device",
    }
    assert {node._Node__node_executable for node in nodes} == {
        "ad_dynamic_obstacle_detector_node"
    }
    assert all(node._Node__node_namespace is None for node in nodes)
    includes = [
        entity
        for entity in description.entities
        if isinstance(entity, IncludeLaunchDescription)
    ]
    assert len(includes) == 1


def test_traffic_signal_launch_starts_detector_evaluator_and_visualizer(monkeypatch):
    module = _load_launch_module("traffic_signal.launch.py")
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda _package: str(PACKAGE),
    )

    description = module.generate_launch_description()
    nodes = [
        entity for entity in description.entities if isinstance(entity, Node)
    ]

    assert [node._Node__node_executable for node in nodes] == [
        "ad_traffic_light_detector_node",
        "ad_traffic_light_evaluator_node",
        "ad_traffic_light_visualizer_node",
    ]


def test_dynamic_obstacle_launch_preserves_detections_and_names_visualization():
    module = _load_launch_module("dynamic_obstacle_detection.launch.py")
    arguments = {
        entity.name: perform_substitutions(
            LaunchContext(), entity.default_value
        )
        for entity in module.generate_launch_description().entities
        if isinstance(entity, DeclareLaunchArgument)
    }

    assert set(arguments) == {
        "params_file",
        "image_topic",
        "image_transport",
        "model_path",
        "device",
        "detections_topic",
        "visualization_image_topic",
        "show_window",
        "window_name",
    }
    assert (
        arguments["detections_topic"]
        == "/vision/dynamic_obstacle/detections"
    )
    assert (
        arguments["visualization_image_topic"]
        == "/ad/viz/perception/camera/dynamic_obstacle"
    )
    visualizer_parameters = yaml.safe_load(
        (PACKAGE / "config" / "dynamic_obstacle.yaml").read_text(
            encoding="utf-8"
        )
    )["dynamic_obstacle_visualizer"]["ros__parameters"]
    assert (
        visualizer_parameters["detections_topic"]
        == "/vision/dynamic_obstacle/detections"
    )
    assert (
        visualizer_parameters["output_image_topic"]
        == "/ad/viz/perception/camera/dynamic_obstacle"
    )


def test_launch_runtime_dependencies_are_declared():
    root = ET.parse(PACKAGE / "package.xml").getroot()
    dependencies = {item.text for item in root.findall("exec_depend")}

    assert {
        "ad_interfaces",
        "ament_index_python",
        "cv_bridge",
        "image_transport",
        "launch",
        "launch_ros",
        "vision_msgs",
    } <= dependencies


def test_package_exports_ament_python_build_type_and_traffic_entry_points():
    root = ET.parse(PACKAGE / "package.xml").getroot()
    buildtools = {item.text for item in root.findall("buildtool_depend")}
    build_type = root.findtext("export/build_type")
    setup_text = (PACKAGE / "setup.py").read_text()

    assert build_type == "ament_python"
    assert "ament_python" not in buildtools
    assert "ad_traffic_signal_node" in setup_text
    assert "ad_traffic_light_detector_node" in setup_text
    assert "ad_traffic_light_evaluator_node" in setup_text
    assert "ad_traffic_light_visualizer_node" in setup_text


def test_local_traffic_logic_uses_mm2025_mapping_and_opencv_window():
    python_package = PACKAGE / "ad_camera_perception"
    classifier = python_package / "traffic_light" / "traffic.py"
    legacy_classifier = python_package / "traffic.py"
    detector = python_package / "nodes" / "traffic_light_detector_node.py"
    evaluator = python_package / "nodes" / "traffic_light_evaluator_node.py"
    visualizer = python_package / "nodes" / "traffic_light_visualizer_node.py"

    assert classifier.is_file()
    assert not legacy_classifier.exists()
    assert "MM2025_CLASS_ASPECTS" in classifier.read_text(encoding="utf-8")
    assert "YoloV7Backend" in detector.read_text(encoding="utf-8")
    assert "TrafficLightStatus" in evaluator.read_text(encoding="utf-8")
    visualizer_text = visualizer.read_text(encoding="utf-8")
    assert "cv2.imshow" in visualizer_text
    assert "create_publisher" not in visualizer_text
