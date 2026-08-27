from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.utilities import perform_substitutions
import pytest
import yaml

from ad_lidar_perception.tracking_replay_presets import (
    PresetConfigError,
    load_tracking_replay_presets,
    select_tracking_replay_preset,
)


PACKAGE = Path(__file__).resolve().parents[1]
LAUNCH = PACKAGE / "launch" / "camera_lidar_tracking_replay.launch.py"
RVIZ = PACKAGE / "rviz" / "heven_camera_tracking.rviz"
PRESETS = PACKAGE / "config" / "tracking" / "camera_replay_presets.yaml"


def load_launch_module():
    spec = spec_from_file_location(
        "camera_lidar_tracking_replay_launch", LAUNCH
    )
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RecordingAction:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def launch_context(tmp_path, mode):
    tmp_path.mkdir(parents=True, exist_ok=True)
    centerpoint_checkpoint = tmp_path / "centerpoint.pth"
    centerpoint_checkpoint.write_bytes(b"centerpoint")
    kalmannet_checkpoint = tmp_path / "kalmannet.pt"
    kalmannet_checkpoint.write_bytes(b"kalmannet")
    openpcdet = tmp_path / "OpenPCDet"
    openpcdet.mkdir()
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "bag_path": str(tmp_path / "bag"),
            "mode": str(mode),
            "preset_config": str(PRESETS),
            "rate": "0.5",
            "startup_delay_sec": "4.0",
            "start_paused": "false",
            "centerpoint_checkpoint": str(centerpoint_checkpoint),
            "openpcdet_root": str(openpcdet),
            "centerpoint_device": "cuda:0",
            "kalmannet_checkpoint": str(kalmannet_checkpoint),
            "kalmannet_device": "cpu",
        }
    )
    return context


def record_setup(module, monkeypatch, context):
    for name in (
        "IncludeLaunchDescription",
        "LogInfo",
        "Node",
        "SetParameter",
    ):
        monkeypatch.setattr(module, name, RecordingAction)
    monkeypatch.setattr(module, "_launch_file", lambda name: name)
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda _package: str(PACKAGE),
    )
    return module._launch_setup(context)


def included_arguments(actions, launch_name):
    action = next(
        item for item in actions
        if item.args and item.args[0] == launch_name
    )
    return dict(action.kwargs["launch_arguments"])


def test_declares_safe_defaults_and_keeps_mode_3_as_default(monkeypatch):
    module = load_launch_module()
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda _package: str(PACKAGE),
    )
    description = module.generate_launch_description()
    arguments = {
        entity.name: entity.default_value
        for entity in description.entities
        if isinstance(entity, DeclareLaunchArgument)
    }
    assert set(arguments) == {
        "bag_path",
        "mode",
        "preset_config",
        "rate",
        "startup_delay_sec",
        "start_paused",
        "centerpoint_checkpoint",
        "openpcdet_root",
        "centerpoint_device",
        "kalmannet_checkpoint",
        "kalmannet_device",
    }
    context = LaunchContext()
    assert perform_substitutions(context, arguments["mode"]) == "3"
    assert perform_substitutions(context, arguments["rate"]) == "0.5"
    assert perform_substitutions(
        context, arguments["preset_config"]
    ) == str(PRESETS)


def test_checked_in_matrix_defines_exactly_the_requested_ten_modes():
    default_mode, presets = load_tracking_replay_presets(PRESETS)
    assert default_mode == "3"
    assert list(presets) == [str(index) for index in range(1, 11)]
    expected = {
        "1": ("euclidean", "giou_3d", "greedy", "linear_kf"),
        "2": ("euclidean", "giou_3d", "hungarian", "linear_kf"),
        "3": ("euclidean", "euclidean", "hungarian", "linear_kf"),
        "4": ("euclidean", "mahalanobis", "hungarian", "linear_kf"),
        "5": ("euclidean", "euclidean", "hungarian", "ekf"),
        "6": ("euclidean", "euclidean", "hungarian", "imm"),
        "7": ("euclidean", "euclidean", "hungarian", "kalmannet"),
        "8": ("centerpoint", "euclidean", "hungarian", "linear_kf"),
        "9": ("centerpoint", "giou_3d", "hungarian", "linear_kf"),
        "10": ("centerpoint", "mahalanobis", "hungarian", "linear_kf"),
    }
    for mode, preset in presets.items():
        assert (
            preset.detector,
            preset.association,
            preset.matcher,
            preset.estimator,
        ) == expected[mode]
        assert preset.yaw_measurement_mode == "unobserved"
        expected_cap = 10.0 if mode in {"4", "10"} else 0.0
        assert preset.mahalanobis_max_distance_m == expected_cap


@pytest.mark.parametrize("mode", [str(index) for index in range(1, 11)])
def test_each_mode_wires_detector_and_experimental_tracker(
    tmp_path, monkeypatch, mode
):
    module = load_launch_module()
    actions = record_setup(module, monkeypatch, launch_context(tmp_path, mode))
    preset = select_tracking_replay_preset(PRESETS, mode)
    replay = included_arguments(actions, "lidar_bag_replay.launch.py")
    tracker = included_arguments(actions, "ab3dmot_tracker.launch.py")
    visualization = included_arguments(
        actions, "perception_visualization.launch.py"
    )

    assert replay["include_front_camera"] == "true"
    assert replay["detector_backend"] == preset.detector
    assert tracker["matcher"] == preset.matcher
    assert tracker["association_metric"] == preset.association
    assert tracker["state_estimator"] == preset.estimator
    assert tracker["yaw_measurement_mode"] == "unobserved"
    assert float(tracker["euclidean_gate_m"]) == 3.0
    assert float(tracker["mahalanobis_gate"]) == 11.62
    assert float(tracker["mahalanobis_max_distance_m"]) == (
        10.0 if mode in {"4", "10"} else 0.0
    )
    assert bool(replay["checkpoint_path"]) is preset.needs_centerpoint
    assert bool(tracker["kalmannet_checkpoint"]) is preset.needs_kalmannet
    assert visualization == {
        "start_rviz": "true",
        "use_sim_time": "true",
        "rviz_config": str(RVIZ),
    }


def test_model_modes_fail_early_without_external_artifacts(tmp_path, monkeypatch):
    module = load_launch_module()
    for mode, diagnostic, key in (
        ("7", "kalmannet_checkpoint", "kalmannet_checkpoint"),
        ("8", "centerpoint_checkpoint", "centerpoint_checkpoint"),
    ):
        context = launch_context(tmp_path / mode, mode)
        context.launch_configurations[key] = ""
        with pytest.raises(RuntimeError, match=diagnostic):
            record_setup(module, monkeypatch, context)


@pytest.mark.parametrize("mode", ["0", "11", "ekf"])
def test_unknown_modes_are_rejected(mode):
    with pytest.raises(PresetConfigError, match="1 through 10"):
        select_tracking_replay_preset(PRESETS, mode)


def test_camera_tracking_rviz_has_camera_lidar_and_both_trackers():
    config = yaml.safe_load(RVIZ.read_text(encoding="utf-8"))
    manager = config["Visualization Manager"]
    assert manager["Global Options"]["Fixed Frame"] == "odom"
    displays = {
        display["Name"]: display for display in manager["Displays"]
    }
    assert displays["Front Camera"]["Topic"]["Value"] == (
        "/ad/sensors/camera/front/compressed"
    )
    assert displays["LiDAR (Cropped)"]["Topic"]["Value"] == (
        "/ad/perception/lidar/cropped"
    )
    assert displays["Tracked Objects (Autoware)"]["Topic"]["Value"] == (
        "/ad/visualization/tracked_objects"
    )
    assert displays["Tracked Objects (AB3DMOT)"]["Topic"]["Value"] == (
        "/experiment/visualization/tracked_objects_ab3dmot"
    )
