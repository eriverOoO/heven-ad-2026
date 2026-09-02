from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.utilities import perform_substitutions
import pytest
import yaml


PACKAGE = Path(__file__).resolve().parents[1]
LAUNCH = PACKAGE / "launch" / "training_free_perception_rviz.launch.py"
RVIZ = PACKAGE / "rviz" / "training_free_perception_camera.rviz"


def load_launch_module():
    spec = spec_from_file_location("training_free_perception_rviz_launch", LAUNCH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RecordingAction:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def launch_context(**overrides):
    values = {
        "input_mode": "live",
        "bag_path": "",
        "rate": "0.5",
        "loop": "true",
        "start_paused": "false",
        "enable_camera": "false",
        "enable_camera_perception": "false",
        "enable_dynamic_object_risk": "true",
        "enable_localization": "false",
        "enable_drivable_mask": "false",
        "data_dir": "",
        "start_rviz": "true",
        "rviz_config": str(RVIZ),
    }
    values.update({name: str(value) for name, value in overrides.items()})
    context = LaunchContext()
    context.launch_configurations.update(values)
    return context


def record_setup(module, monkeypatch, context):
    for name in ("GroupAction", "IncludeLaunchDescription", "SetParameter"):
        monkeypatch.setattr(module, name, RecordingAction)
    monkeypatch.setattr(module, "_launch_file", lambda package, name: name)
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda _package: str(PACKAGE),
    )
    return module._launch_setup(context)


def included(actions, launch_name):
    return next(
        item for item in actions
        if item.args and item.args[0] == launch_name
    )


def included_arguments(actions, launch_name):
    return dict(included(actions, launch_name).kwargs["launch_arguments"])


def test_declares_opt_in_defaults(monkeypatch):
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
        "input_mode",
        "bag_path",
        "rate",
        "loop",
        "start_paused",
        "enable_camera",
        "enable_camera_perception",
        "enable_dynamic_object_risk",
        "enable_localization",
        "enable_drivable_mask",
        "data_dir",
        "start_rviz",
        "rviz_config",
    }
    context = LaunchContext()
    assert perform_substitutions(context, arguments["input_mode"]) == "live"
    assert perform_substitutions(context, arguments["start_rviz"]) == "true"
    assert perform_substitutions(
        context, arguments["enable_camera"]
    ) == "false"
    assert perform_substitutions(
        context, arguments["enable_camera_perception"]
    ) == "false"


def test_live_mode_wires_the_training_free_backend(monkeypatch):
    module = load_launch_module()
    actions = record_setup(module, monkeypatch, launch_context())

    names = [item.args[0] for item in actions if item.args]
    assert "description.launch.py" in names
    assert "lidar_perception.launch.py" in names
    assert "lidar_bag_replay.launch.py" not in names
    assert "dynamic_obstacle_detection.launch.py" not in names

    perception = included_arguments(actions, "lidar_perception.launch.py")
    assert perception["detector_backend"] == "euclidean"
    assert perception["tracker_backend"] == "ab3dmot"
    assert perception["dynamic_object_risk"] == "true"
    assert perception["start_visualization"] == "false"
    assert perception["start_rviz"] == "false"
    assert perception["use_sim_time"] == "false"
    assert perception["composition_config"].endswith(
        "config/lidar_perception.yaml"
    )

    visualization = included_arguments(
        actions, "perception_visualization.launch.py"
    )
    assert visualization["start_rviz"] == "true"
    assert visualization["use_sim_time"] == "false"
    assert visualization["enable_experiment_tracker_view"] == "false"
    assert visualization["rviz_config"] == str(RVIZ)


def test_replay_mode_uses_bag_replay_on_sim_time(monkeypatch):
    module = load_launch_module()
    actions = record_setup(
        module,
        monkeypatch,
        launch_context(
            input_mode="replay",
            bag_path="/abs/bag",
            enable_camera="true",
        ),
    )

    names = [item.args[0] for item in actions if item.args]
    assert "lidar_bag_replay.launch.py" in names
    # lidar_bag_replay owns ad_description; the demo must not double-include it.
    assert "description.launch.py" not in names

    replay = included_arguments(actions, "lidar_bag_replay.launch.py")
    assert replay["bag_path"] == "/abs/bag"
    assert replay["detector_backend"] == "euclidean"
    assert replay["tracker_backend"] == "ab3dmot"
    assert replay["dynamic_object_risk"] == "true"
    assert replay["include_front_camera"] == "true"
    assert replay["composition_config"].endswith(
        "config/lidar_perception.yaml"
    )

    visualization = included_arguments(
        actions, "perception_visualization.launch.py"
    )
    assert visualization["use_sim_time"] == "true"
    assert visualization["start_rviz"] == "true"
    assert visualization["enable_experiment_tracker_view"] == "false"


def test_replay_without_bag_path_is_rejected(monkeypatch):
    module = load_launch_module()
    with pytest.raises(RuntimeError, match="bag_path"):
        record_setup(
            module, monkeypatch, launch_context(input_mode="replay")
        )


@pytest.mark.parametrize("mode", ["", "bag", "LIVE-ish"])
def test_invalid_input_mode_is_rejected(monkeypatch, mode):
    module = load_launch_module()
    with pytest.raises(RuntimeError, match="input_mode"):
        record_setup(module, monkeypatch, launch_context(input_mode=mode))


def test_camera_perception_overlay_is_opt_in(monkeypatch):
    module = load_launch_module()

    off = record_setup(module, monkeypatch, launch_context())
    assert not any(
        item.args and item.args[0] == "dynamic_obstacle_detection.launch.py"
        for item in off
    )

    on = record_setup(
        module,
        monkeypatch,
        launch_context(enable_camera_perception="true"),
    )
    flat = []
    for item in on:
        flat.append(item)
        flat.extend(item.kwargs.get("actions", []))
    assert any(
        item.args and item.args[0] == "dynamic_obstacle_detection.launch.py"
        for item in flat
    )


def test_drivable_mask_producer_is_opt_in_and_reuses_the_existing_node(
    monkeypatch,
):
    # Regression coverage for Training-Free Dynamic OGM Completion v1: the
    # existing, already-tested ad_road_corridor_mask_node (ad_planner) is
    # reused unmodified, never a new mask implementation, and stays off by
    # default so it never changes the demo's existing behaviour.
    module = load_launch_module()

    off = record_setup(module, monkeypatch, launch_context())
    assert not any(
        item.args and item.args[0] == "road_corridor_mask.launch.py"
        for item in off
    )

    on = record_setup(
        module,
        monkeypatch,
        launch_context(enable_drivable_mask="true", data_dir="/abs/ad_data"),
    )
    mask_group = next(
        item
        for item in on
        if item.kwargs.get("actions")
        and any(
            getattr(action, "args", None)
            and action.args[0] == "road_corridor_mask.launch.py"
            for action in item.kwargs["actions"]
        )
    )
    # Its own scoped group, not nested inside the perception/replay group -
    # a plain rclcpp::Node (no lifecycle autostart), so scoping is safe here,
    # unlike the ad_localization case this same launch file already fixed.
    assert mask_group.kwargs["scoped"] is True
    mask_include = next(
        action
        for action in mask_group.kwargs["actions"]
        if getattr(action, "args", None)
        and action.args[0] == "road_corridor_mask.launch.py"
    )
    assert dict(mask_include.kwargs["launch_arguments"])["data_dir"] == (
        "/abs/ad_data"
    )


def test_launch_never_wires_a_learned_model_backend(monkeypatch):
    # The executable body (docstring excluded) must not name a learned
    # detector/estimator backend or a weights file in any mode.
    module = load_launch_module()
    body = "\n".join(
        LAUNCH.read_text(encoding="utf-8").split('"""')[2:]
    ).lower()
    for forbidden in ("centerpoint", "kalmannet", "openpcdet", "checkpoint"):
        assert forbidden not in body

    for context in (
        launch_context(),
        launch_context(input_mode="replay", bag_path="/abs/bag"),
    ):
        actions = record_setup(module, monkeypatch, context)
        flat = []
        for item in actions:
            flat.append(item)
            flat.extend(item.kwargs.get("actions", []))
        for item in flat:
            name = item.args[0] if item.args else ""
            assert "centerpoint" not in name.lower()
            assert "kalmannet" not in name.lower()
            for key, value in item.kwargs.get("launch_arguments", []):
                assert "checkpoint" not in f"{key}{value}".lower()
                assert "centerpoint" not in f"{key}{value}".lower()


def test_demo_rviz_config_shows_lidar_camera_objects_and_occupancy():
    config = yaml.safe_load(RVIZ.read_text(encoding="utf-8"))
    manager = config["Visualization Manager"]
    assert manager["Global Options"]["Fixed Frame"] == "odom"
    displays = {display["Name"]: display for display in manager["Displays"]}

    assert displays["LiDAR (Cropped)"]["Topic"]["Value"] == (
        "/ad/perception/lidar/cropped"
    )
    assert displays["LiDAR (Cropped)"]["Enabled"] is True
    for name in ("LiDAR (Raw)", "Ground Points", "Non-Ground Points"):
        assert displays[name]["Enabled"] is False

    assert displays["Front Camera"]["Class"] == "rviz_default_plugins/Image"
    assert displays["Front Camera"]["Topic"]["Value"] == (
        "/ad/sensors/camera/front/compressed"
    )

    markers = {
        "Detected Objects": "/ad/visualization/detected_objects",
        "Tracked Objects": "/ad/visualization/tracked_objects",
        "Predicted Objects": "/ad/visualization/predicted_objects",
    }
    for name, topic in markers.items():
        assert displays[name]["Class"] == "rviz_default_plugins/MarkerArray"
        assert displays[name]["Topic"]["Value"] == topic

    assert displays["Dynamic Occupancy"]["Class"] == (
        "rviz_default_plugins/Map"
    )
    assert displays["Dynamic Occupancy"]["Topic"]["Value"] == (
        "/ad/perception/occupancy/dynamic"
    )
    assert displays["Dynamic Occupancy"]["Enabled"] is True
    assert displays["Combined Occupancy"]["Topic"]["Value"] == (
        "/ad/perception/occupancy/combined"
    )

    assert displays["Drivable Mask (debug)"]["Class"] == (
        "rviz_default_plugins/Map"
    )
    assert displays["Drivable Mask (debug)"]["Topic"]["Value"] == (
        "/ad/planning/drivable_mask"
    )
    # Toggleable but off by default: enable_drivable_mask itself defaults to
    # false, so this display should not silently imply the mask is running.
    assert displays["Drivable Mask (debug)"]["Enabled"] is False

    names = {display["Name"] for display in manager["Displays"]}
    assert not any("CenterPoint" in name for name in names)
    assert not any("AB3DMOT" in name for name in names)
