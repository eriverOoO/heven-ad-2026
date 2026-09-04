from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.utilities import perform_substitutions
import yaml


PACKAGE = Path(__file__).resolve().parents[1]
LAUNCH = PACKAGE / "launch" / "study_pipeline_rviz.launch.py"
RVIZ = PACKAGE / "rviz" / "training_free_perception_camera.rviz"


def load_module():
    spec = spec_from_file_location("study_pipeline_rviz_launch", LAUNCH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_study_matrix_is_the_causal_abcd_comparison():
    module = load_module()
    assert module.VARIANTS == {
        "training_free": ("PIPELINE A — Euclidean + Hungarian + Linear KF", "euclidean", "linear_kf"),
        "centerpoint_kf": ("PIPELINE B — CenterPoint + Hungarian + Linear KF", "centerpoint", "linear_kf"),
        "euclidean_knet": ("PIPELINE C — Euclidean + Hungarian + KalmanNet", "euclidean", "kalmannet"),
        "centerpoint_knet": ("PIPELINE D — CenterPoint + Hungarian + KalmanNet", "centerpoint", "kalmannet"),
    }


def test_study_launch_is_opt_in_and_has_no_model_default(monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "get_package_share_directory", lambda _name: str(PACKAGE))
    description = module.generate_launch_description()
    arguments = {
        entity.name: entity.default_value for entity in description.entities
        if isinstance(entity, DeclareLaunchArgument)
    }
    assert "pipeline_variant" in arguments
    assert "bag_path" in arguments
    assert "centerpoint_checkpoint" in arguments
    assert "kalmannet_checkpoint" in arguments
    assert "metrics_source_start_sec" in arguments
    assert "use_predicted_future_sweep" in arguments
    assert "future_sweep_horizon_s" in arguments
    context = LaunchContext()
    assert perform_substitutions(context, arguments["centerpoint_checkpoint"]) == ""
    assert perform_substitutions(context, arguments["kalmannet_checkpoint"]) == ""
    assert (
        perform_substitutions(context, arguments["use_predicted_future_sweep"])
        == "false"
    )


class _RecordingInclude:
    calls = []

    def __init__(self, source, launch_arguments=None, **kwargs):
        self.source = getattr(source, "location", source)
        self.launch_arguments = dict(launch_arguments or {})
        type(self).calls.append(self)


def test_prediction_yaw_rate_source_override_reaches_bag_replay_include(
    monkeypatch,
):
    module = load_module()
    monkeypatch.setattr(
        module, "get_package_share_directory", lambda _name: str(PACKAGE)
    )
    monkeypatch.setattr(
        module,
        "_launch_file",
        lambda package, name: type("S", (), {"location": name})(),
    )
    _RecordingInclude.calls.clear()
    monkeypatch.setattr(module, "IncludeLaunchDescription", _RecordingInclude)

    context = LaunchContext()
    context.launch_configurations.update(
        {
            "pipeline_variant": "training_free",
            "bag_path": "/tmp/bag",
            "rate": "0.5",
            "start_paused": "false",
            "start_rviz": "false",
            "enable_drivable_mask": "false",
            "centerpoint_checkpoint": "",
            "openpcdet_root": "",
            "kalmannet_checkpoint": "",
            "centerpoint_device": "cpu",
            "kalmannet_device": "cpu",
            "ab3dmot_root": "",
            "prediction_yaw_rate_source": "motion_history",
            "use_predicted_future_sweep": "true",
            "future_sweep_horizon_s": "3.0",
            "data_dir": "",
            "metrics_output": "",
            "metrics_duration_sec": "0",
            "metrics_source_start_sec": "0",
        }
    )
    module._launch_setup(context)

    replay = next(
        call
        for call in _RecordingInclude.calls
        if call.source == "lidar_bag_replay.launch.py"
    )
    assert replay.launch_arguments["prediction_yaw_rate_source"] == (
        "motion_history"
    )
    assert replay.launch_arguments["use_predicted_future_sweep"] == "true"


def test_rviz_layout_has_all_study_layers_and_camera():
    displays = {
        item["Name"]: item
        for item in yaml.safe_load(RVIZ.read_text(encoding="utf-8"))["Visualization Manager"]["Displays"]
    }
    for name in (
        "LiDAR (Cropped)", "Dynamic Occupancy", "Detected Objects",
        "Tracked Objects", "Predicted Objects", "Front Camera", "Study Pipeline Label",
    ):
        assert displays[name]["Enabled"] is True
    assert displays["Front Camera"]["Topic"]["Value"] == "/ad/sensors/camera/front/compressed"
    assert displays["Study Pipeline Label"]["Topic"]["Value"] == "/ad/study/pipeline_label"
