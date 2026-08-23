from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.utilities import perform_substitutions
from launch_ros.actions import Node, SetParameter
import yaml


PACKAGE = Path(__file__).resolve().parents[1]
LAUNCH = PACKAGE / "launch" / "camera_lidar_tracking_replay.launch.py"
RVIZ = PACKAGE / "rviz" / "heven_camera_tracking.rviz"


def load_launch_module():
    spec = spec_from_file_location(
        "camera_lidar_tracking_replay_launch", LAUNCH
    )
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_one_command_launch_wires_replay_trackers_visualization_and_tf(
    monkeypatch,
):
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
        "rate",
        "startup_delay_sec",
        "start_paused",
    }
    context = LaunchContext()
    assert arguments["bag_path"] is None
    assert perform_substitutions(context, arguments["rate"]) == "0.5"
    assert (
        perform_substitutions(context, arguments["startup_delay_sec"])
        == "4.0"
    )

    includes = [
        entity
        for entity in description.entities
        if isinstance(entity, IncludeLaunchDescription)
    ]
    assert len(includes) == 3
    by_name = dict(
        zip(
            (
                "ab3dmot_tracker.launch.py",
                "perception_visualization.launch.py",
                "lidar_bag_replay.launch.py",
            ),
            includes,
            strict=True,
        )
    )

    replay = dict(by_name["lidar_bag_replay.launch.py"].launch_arguments)
    assert replay["include_front_camera"] == "true"
    tracker = dict(by_name["ab3dmot_tracker.launch.py"].launch_arguments)
    assert tracker == {
        "enabled": "true",
        "matcher": "hungarian",
        "association_metric": "euclidean",
        "euclidean_gate_m": "3.0",
        "state_estimator": "linear_kf",
        "yaw_measurement_mode": "unobserved",
    }
    visualization = dict(
        by_name["perception_visualization.launch.py"].launch_arguments
    )
    assert visualization == {
        "start_rviz": "true",
        "use_sim_time": "true",
        "rviz_config": str(RVIZ),
    }

    parameters = [
        entity
        for entity in description.entities
        if isinstance(entity, SetParameter)
    ]
    assert len(parameters) == 1
    nodes = [
        entity for entity in description.entities if isinstance(entity, Node)
    ]
    assert len(nodes) == 1
    anchor = nodes[0]
    assert anchor.node_package == "tf2_ros"
    assert anchor.node_executable == "static_transform_publisher"
    source = LAUNCH.read_text(encoding="utf-8")
    assert '"--frame-id",\n            "odom"' in source
    assert '"--child-frame-id",\n            "base_link"' in source


def test_camera_tracking_rviz_has_camera_lidar_and_both_trackers():
    config = yaml.safe_load(RVIZ.read_text(encoding="utf-8"))
    manager = config["Visualization Manager"]
    assert manager["Global Options"]["Fixed Frame"] == "odom"
    displays = {
        display["Name"]: display for display in manager["Displays"]
    }
    assert displays["Front Camera"]["Class"] == (
        "rviz_default_plugins/Image"
    )
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
