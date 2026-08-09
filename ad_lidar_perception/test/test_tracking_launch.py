from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction
import yaml


PACKAGE = Path(__file__).resolve().parents[1]
LAUNCH = PACKAGE / "launch" / "tracking.launch.py"


def load_launch_module():
    spec = spec_from_file_location("tracking_launch", LAUNCH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_tracker_overlay_is_complete_pinned_schema_with_four_heven_values():
    document = yaml.safe_load(
        (
            PACKAGE / "config" / "tracking" / "autoware.yaml"
        ).read_text(encoding="utf-8")
    )
    assert set(document) == {"/**"}
    assert set(document["/**"]) == {"ros__parameters"}
    parameters = document["/**"]["ros__parameters"]
    assert parameters == {
        "initial_tracker": {
            "car": "multi_vehicle_tracker",
            "truck": "multi_vehicle_tracker",
            "bus": "multi_vehicle_tracker",
            "trailer": "multi_vehicle_tracker",
            "pedestrian": "pedestrian_and_bicycle_tracker",
            "bicycle": "pedestrian_and_bicycle_tracker",
            "motorcycle": "pedestrian_and_bicycle_tracker",
        },
        "publish_rate": 10.0,
        "world_frame_id": "odom",
        "ego_frame_id": "base_link",
        "enable_delay_compensation": False,
        "consider_odometry_uncertainty": False,
        "enable_unknown_object_velocity_estimation": True,
        "enable_unknown_object_motion_output": True,
        "min_known_object_removal_iou": 0.1,
        "min_unknown_object_removal_iou": 0.001,
        "pruning_generalized_iou_thresholds": {
            "unknown": -0.3,
            "car": -0.4,
            "truck": -0.6,
            "bus": -0.6,
            "trailer": -0.6,
            "motorcycle": -0.1,
            "bicycle": -0.1,
            "pedestrian": -0.1,
        },
        "pruning_distance_thresholds": {
            "unknown": 9.0,
            "car": 5.0,
            "truck": 9.0,
            "bus": 9.0,
            "trailer": 9.0,
            "motorcycle": 4.0,
            "bicycle": 3.0,
            "pedestrian": 2.0,
        },
        "pruning_static_object_speed": 1.38,
        "pruning_moving_object_speed": 5.5,
        "pruning_static_iou_threshold": 0.0,
        "publish_processing_time": False,
        "publish_processing_time_detail": False,
        "publish_tentative_objects": False,
        "publish_debug_markers": False,
        "diagnostics_warn_delay": 0.5,
        "diagnostics_error_delay": 1.0,
        "diagnostics_warn_extrapolation": 0.5,
        "diagnostics_error_extrapolation": 1.0,
    }


def tracker_runtime(tmp_path):
    return SimpleNamespace(
        package="autoware_multi_object_tracker",
        executable="multi_object_tracker_node",
        launch_path=(
            tmp_path
            / "install"
            / "share"
            / "autoware_multi_object_tracker"
            / "launch"
            / "multi_object_tracker.launch.xml"
        ),
    )


def test_tracker_passes_exact_active_channel_and_disables_02_through_12(
    tmp_path,
):
    module = load_launch_module()
    package_share = tmp_path / "share" / "ad_lidar_perception"
    arguments = module._tracker_arguments(
        tracker_runtime(tmp_path), package_share
    )

    assert arguments["input/detection01/objects"] == (
        "/ad/perception/objects/detected"
    )
    assert arguments["input/detection01/channel"] == "lidar_clustering"
    for channel in range(2, 13):
        assert arguments[f"input/detection{channel:02d}/channel"] == "none"
    assert arguments["output/objects"] == "/ad/perception/objects/tracked"
    assert arguments["publish_merged_objects"] == "false"
    assert arguments["tracker_setting_path"] == str(
        package_share / "config" / "tracking" / "autoware.yaml"
    )

    object_overrides = {
        key for key in arguments if key.endswith("/objects")
    }
    assert object_overrides == {
        "input/detection01/objects",
        "output/objects",
    }
    channel_overrides = {
        key for key in arguments if key.endswith("/channel")
    }
    assert channel_overrides == {
        f"input/detection{channel:02d}/channel"
        for channel in range(1, 13)
    }


def test_tracking_launch_runs_preflight_and_includes_pinned_xml(
    tmp_path, monkeypatch
):
    module = load_launch_module()
    verified_tracker = tracker_runtime(tmp_path)
    selection = SimpleNamespace(
        detector=SimpleNamespace(backend="centerpoint_tiny"),
        tracker=SimpleNamespace(backend="autoware"),
    )
    calls = []

    monkeypatch.setattr(module, "load_selection", lambda path: selection)
    monkeypatch.setattr(
        module,
        "verify_selection",
        lambda selected, **kwargs: (
            calls.append((selected, kwargs))
            or SimpleNamespace(tracker=verified_tracker)
        ),
    )
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda name: str(tmp_path / "share" / name),
    )
    monkeypatch.setattr(
        module,
        "AnyLaunchDescriptionSource",
        lambda location: SimpleNamespace(location=location),
    )
    monkeypatch.setattr(
        module,
        "IncludeLaunchDescription",
        lambda source, **kwargs: SimpleNamespace(
            launch_description_source=source,
            launch_arguments=list(kwargs["launch_arguments"]),
        ),
    )

    context = LaunchContext()
    context.launch_configurations.update(
        {
            "selection_config": str(tmp_path / "selection.yaml"),
            "data_root": str(tmp_path / "data-root"),
        }
    )
    actions = module._launch_setup(context)
    assert len(calls) == 1
    assert calls[0][1] == {
        "lock_path": (
            tmp_path
            / "share"
            / "ad_lidar_perception"
            / "config"
            / "autoware_perception.lock.yaml"
        ),
        "data_root": tmp_path / "data-root",
    }
    assert len(actions) == 1
    include = actions[0]
    assert include.launch_description_source.location == str(
        verified_tracker.launch_path
    )
    arguments = dict(include.launch_arguments)
    assert arguments["input/detection01/channel"] == "lidar_clustering"
    assert arguments["input/detection12/channel"] == "none"


def test_tracking_launch_interface_uses_installed_config_and_opaque_setup():
    module = load_launch_module()
    description = module.generate_launch_description()
    arguments = {
        entity.name
        for entity in description.entities
        if isinstance(entity, DeclareLaunchArgument)
    }
    assert arguments == {"selection_config", "data_root"}
    assert sum(
        isinstance(entity, OpaqueFunction)
        for entity in description.entities
    ) == 1
