from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch_ros.actions import Node
import yaml


PACKAGE = Path(__file__).resolve().parents[1]
LAUNCH = PACKAGE / "launch" / "perception_visualization.launch.py"
RVIZ = PACKAGE / "rviz" / "heven_perception.rviz"


def load_launch_module():
    spec = spec_from_file_location("perception_visualization_launch", LAUNCH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_launch_starts_diagnostic_node_and_keeps_rviz_opt_in(monkeypatch):
    module = load_launch_module()
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda _package: str(PACKAGE),
    )
    description = module.generate_launch_description()
    arguments = {
        action.name: action
        for action in description.entities
        if isinstance(action, DeclareLaunchArgument)
    }
    assert set(arguments) == {
        "start_rviz",
        "use_sim_time",
        "visualize_predictions",
        "rviz_config",
    }
    nodes = [
        action for action in description.entities if isinstance(action, Node)
    ]
    assert len(nodes) == 2
    visualizer = next(
        node for node in nodes if node.node_package == "ad_viz"
    )
    assert visualizer.node_executable == "perception_visualizer_node"
    assert visualizer.condition is None

    rviz = next(node for node in nodes if node.node_package == "rviz2")
    assert rviz.node_executable == "rviz2"
    assert isinstance(rviz.condition, IfCondition)
    context = LaunchContext()
    context.launch_configurations["start_rviz"] = "false"
    assert rviz.condition.evaluate(context) is False
    context.launch_configurations["start_rviz"] = "true"
    assert rviz.condition.evaluate(context) is True


def test_focused_rviz_config_has_exact_runtime_topics_and_frames():
    config = yaml.safe_load(RVIZ.read_text(encoding="utf-8"))
    manager = config["Visualization Manager"]
    assert manager["Global Options"]["Fixed Frame"] == "odom"
    displays = {display["Name"]: display for display in manager["Displays"]}
    assert displays["TF"]["Class"] == "rviz_default_plugins/TF"
    assert displays["CenterPoint Input (Cropped)"]["Topic"]["Value"] == (
        "/ad/perception/lidar/cropped"
    )
    assert displays["CenterPoint Input (Cropped)"]["Topic"][
        "Reliability Policy"
    ] == "Best Effort"
    expected_markers = {
        "Detected Objects": "/ad/visualization/detected_objects",
        "Tracked Objects": "/ad/visualization/tracked_objects",
        "Predicted Objects": "/ad/visualization/predicted_objects",
    }
    for name, topic in expected_markers.items():
        display = displays[name]
        assert display["Class"] == "rviz_default_plugins/MarkerArray"
        assert display["Topic"]["Value"] == topic
        assert display["Topic"]["Reliability Policy"] == "Reliable"


def test_top_level_visualization_is_opt_in_and_rviz_implies_node(
    tmp_path, monkeypatch
):
    module_path = PACKAGE / "launch" / "lidar_perception.launch.py"
    spec = spec_from_file_location(
        "lidar_perception_visualization_test", module_path
    )
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    composition = tmp_path / "composition.yaml"
    composition.write_text(
        """schema_version: 1
detector:
  backend: none
  model_subdir: models/autoware
  build_only: false
tracker:
  backend: none
occupancy:
  static_enabled: false
  dynamic_enabled: false
  publish_combined: false
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "_launch_file", lambda name: name)

    class Include:
        def __init__(self, source, **kwargs):
            self.source = source
            self.kwargs = kwargs

    monkeypatch.setattr(module, "IncludeLaunchDescription", Include)

    def context(start_visualization="false", start_rviz="false"):
        result = LaunchContext()
        result.launch_configurations.update(
            {
                "composition_config": str(composition),
                "detector_backend": "euclidean",
                "platform_profile": "morai",
                "start_ground_segmentation": "false",
                "deskew_enabled": "false",
                "self_crop_enabled": "true",
                "self_crop_input_reliable": "false",
                "point_layout_adapter_enabled": "true",
                "patchwork_leveling_enabled": "false",
                "finite_filter_enabled": "true",
                "densifier_enabled": "false",
                "deskew_mode": "3d",
                "raw_input_topic": "/ad/sensors/lidar/points",
                "crop_clearance_m": "0.2",
                "cluster_config": "/tmp/cluster.yaml",
                "ground_config": "/tmp/ground.yaml",
                "sensor_config": "/tmp/sensors.yaml",
                "sensor_profile": "",
                "checkpoint_path": "",
                "score_threshold": "0.1",
                "max_detections": "500",
                "device": "cuda:0",
                "point_cloud_range": "[-4,-25,-3,100,25,5]",
                "centerpoint_enabled": "true",
                "centerpoint_mock_mode": "false",
                "openpcdet_root": "",
                "start_visualization": start_visualization,
                "start_rviz": start_rviz,
                "use_sim_time": "false",
            }
        )
        return result

    default_names = [
        action.source for action in module._launch_setup(context())
    ]
    assert "perception_visualization.launch.py" not in default_names

    actions = module._launch_setup(context(start_rviz="true"))
    visualization = next(
        action
        for action in actions
        if action.source == "perception_visualization.launch.py"
    )
    forwarded = dict(visualization.kwargs["launch_arguments"])
    assert forwarded["start_rviz"] == "true"
