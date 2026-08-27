from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.utilities import perform_substitutions
import pytest
import yaml


PACKAGE = Path(__file__).resolve().parents[1]
LAUNCH = PACKAGE / "launch" / "camera_lidar_fusion_experiment.launch.py"
PARAMS = PACKAGE / "config" / "camera_lidar_fusion_experiment.yaml"


def load_launch_module():
    spec = spec_from_file_location("camera_lidar_fusion_experiment_launch", LAUNCH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RecordingAction:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def context(**overrides):
    values = {
        "camera_lidar_fusion_enabled": "false",
        "semantic_association_enabled": "false",
        "start_camera_detector": "false",
        "use_sim_time": "true",
        "fusion_params": str(PARAMS),
        "camera_params": str(PACKAGE.parent / "ad_camera_perception/config/dynamic_obstacle.yaml"),
        "tf_static_json": "",
        "yolo_weight": "",
        "camera_device": "auto",
        "camera_image_topic": "/ad/sensors/camera/front/compressed",
        "camera_detections_topic": "/vision/dynamic_obstacle/detections",
        "lidar_detections_topic": "/ad/perception/objects/detected",
        "fused_topic": "/experiment/perception/objects/camera_lidar_fused",
    }
    values.update({name: str(value) for name, value in overrides.items()})
    result = LaunchContext()
    result.launch_configurations.update(values)
    return result


def test_defaults_are_fully_inert(monkeypatch):
    module = load_launch_module()
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda package: str(PACKAGE if package == "ad_lidar_perception" else PACKAGE.parent / "ad_camera_perception"),
    )
    description = module.generate_launch_description()
    arguments = {
        entity.name: entity.default_value
        for entity in description.entities
        if isinstance(entity, DeclareLaunchArgument)
    }
    launch_context = LaunchContext()
    assert perform_substitutions(
        launch_context, arguments["camera_lidar_fusion_enabled"]
    ) == "false"
    assert perform_substitutions(
        launch_context, arguments["semantic_association_enabled"]
    ) == "false"
    assert perform_substitutions(
        launch_context, arguments["start_camera_detector"]
    ) == "false"

    monkeypatch.setattr(module, "LogInfo", RecordingAction)
    monkeypatch.setattr(module, "Node", RecordingAction)
    actions = module._launch_setup(context())
    assert len(actions) == 1
    assert "inert baseline" in actions[0].kwargs["msg"]


def test_enabled_fusion_requires_explicit_calibration(tmp_path, monkeypatch):
    module = load_launch_module()
    monkeypatch.setattr(module, "LogInfo", RecordingAction)
    monkeypatch.setattr(module, "Node", RecordingAction)
    with pytest.raises(RuntimeError, match="tf_static_json"):
        module._launch_setup(
            context(camera_lidar_fusion_enabled="true", tf_static_json="")
        )


def test_mode1_wires_only_experimental_output(tmp_path, monkeypatch):
    module = load_launch_module()
    tf_static = tmp_path / "tf_static.json"
    tf_static.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(module, "LogInfo", RecordingAction)
    monkeypatch.setattr(module, "Node", RecordingAction)
    actions = module._launch_setup(
        context(camera_lidar_fusion_enabled="true", tf_static_json=tf_static)
    )
    node = actions[-1]
    assert node.kwargs["executable"] == "ad_camera_lidar_fusion"
    overrides = node.kwargs["parameters"][-1]
    assert overrides["enabled"] is True
    assert overrides["fused_topic"].startswith("/experiment/")
    assert overrides["lidar_detections_topic"] == "/ad/perception/objects/detected"


def test_semantic_ros_mode_fails_instead_of_degrading_evidence(monkeypatch):
    module = load_launch_module()
    monkeypatch.setattr(module, "LogInfo", RecordingAction)
    monkeypatch.setattr(module, "Node", RecordingAction)
    with pytest.raises(RuntimeError, match="no lossless ROS evidence transport"):
        module._launch_setup(context(semantic_association_enabled="true"))


def test_camera_detector_requires_exact_external_weight_hash(tmp_path, monkeypatch):
    module = load_launch_module()
    tf_static = tmp_path / "tf_static.json"
    tf_static.write_text("[]", encoding="utf-8")
    weight = tmp_path / "yolo26s.pt"
    weight.write_bytes(b"wrong")
    monkeypatch.setattr(module, "LogInfo", RecordingAction)
    monkeypatch.setattr(module, "Node", RecordingAction)
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        module._launch_setup(
            context(
                camera_lidar_fusion_enabled="true",
                start_camera_detector="true",
                tf_static_json=tf_static,
                yolo_weight=weight,
            )
        )


def test_checked_in_ros_config_keeps_feature_disabled():
    config = yaml.safe_load(PARAMS.read_text(encoding="utf-8"))
    parameters = config["camera_lidar_fusion"]["ros__parameters"]
    assert parameters["enabled"] is False
    assert parameters["association_measure"] == "iou"
    assert parameters["conservative_geometry_filter_enabled"] is True
    assert parameters["geometry_min_dimension_m"] == 0.15
    assert parameters["calibration_quality"] == "B_reconstructed"
