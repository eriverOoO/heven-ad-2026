from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch.utilities import perform_substitutions

from ad_bringup import course_stack


PACKAGE = Path(__file__).resolve().parents[1]
LAUNCH_PATH = PACKAGE / "launch/course_trial.launch.py"


class RecordingInclude:
    calls = []

    def __init__(self, source, **kwargs):
        self.source = source
        self.kwargs = kwargs
        self.calls.append(self)


def _load_launch(path):
    spec = spec_from_file_location(path.stem, path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _evaluate(arguments, context):
    return {
        name: value.perform(context) if hasattr(value, "perform") else value
        for name, value in dict(arguments).items()
    }


def test_course_trial_launch_has_small_explicit_defaults(monkeypatch):
    monkeypatch.setenv("AD_DATA_DIR", "/tmp/ad-data")
    module = _load_launch(LAUNCH_PATH)
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda package: f"/opt/ros/share/{package}",
    )
    recorded = {}

    def record_stack(**kwargs):
        recorded.update(kwargs)
        return []

    monkeypatch.setattr(module, "build_course_stack", record_stack)

    description = module.generate_launch_description()
    context = LaunchContext()
    defaults = {
        action.name: perform_substitutions(context, action.default_value)
        for action in description.entities
        if isinstance(action, DeclareLaunchArgument)
    }
    assert defaults == {
        "data_dir": "/tmp/ad-data",
        "control_enabled": "false",
        "map_path": "/opt/ros/share/ad_localization/maps/cp14_to_cp15.pcd",
        "path_file": "path/2026_molit_comp_global_path.txt",
        "traffic_light_enabled": "false",
    }
    assert not [
        action
        for action in description.entities
        if isinstance(action, OpaqueFunction)
    ]
    forwarding_context = LaunchContext()
    forwarding_context.launch_configurations["path_file"] = (
        "path/derived/example.txt"
    )
    assert recorded["path_file"].perform(forwarding_context) == (
        "path/derived/example.txt"
    )


def test_course_stack_uses_profile_stanley_planner_and_unified_rviz(
    monkeypatch,
):
    RecordingInclude.calls.clear()
    monkeypatch.setattr(
        course_stack, "IncludeLaunchDescription", RecordingInclude
    )
    monkeypatch.setattr(
        course_stack, "_launch_file", lambda package, name: (package, name)
    )
    monkeypatch.setattr(
        course_stack,
        "get_package_share_directory",
        lambda package: f"/opt/ros/share/{package}",
    )
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "data_dir": "/tmp/ad-data",
            "control_enabled": "true",
            "map_path": "/tmp/ad-data/maps/cp14_to_cp15.pcd",
            "path_file": "path/derived/example.txt",
            "traffic_light_enabled": "true",
        }
    )

    actions = course_stack.build_course_stack(
        data_dir=LaunchConfiguration("data_dir"),
        control_enabled=LaunchConfiguration("control_enabled"),
        map_path=LaunchConfiguration("map_path"),
        path_file=LaunchConfiguration("path_file"),
        traffic_light_enabled=LaunchConfiguration("traffic_light_enabled"),
    )

    assert actions == RecordingInclude.calls
    assert [include.source for include in RecordingInclude.calls] == [
        ("ad_morai_bridge", "bridge.launch.py"),
        ("ad_description", "description.launch.py"),
        ("ad_localization", "hybrid_localization.launch.py"),
        ("ad_planner", "planner.launch.py"),
        ("ad_camera_perception", "traffic_signal.launch.py"),
        ("ad_viz", "visualization.launch.py"),
    ]
    assert _evaluate(
        RecordingInclude.calls[0].kwargs["launch_arguments"], context
    ) == {
        "config": "/opt/ros/share/ad_morai_bridge/config/tunnel_fastlio.yaml",
        "control_enabled": "true",
        "enable_velodyne_points": "true",
        "velodyne_point_timing_mode": "zero",
        "velodyne_organize_cloud": "false",
        "enable_traffic_light_camera": "true",
    }
    assert _evaluate(
        RecordingInclude.calls[2].kwargs["launch_arguments"], context
    ) == {
        "platform_profile": "morai",
        "status_topic": "/ad/vehicle/status",
        "autostart": "true",
        "map_path": "/tmp/ad-data/maps/cp14_to_cp15.pcd",
    }
    assert _evaluate(
        RecordingInclude.calls[3].kwargs["launch_arguments"], context
    ) == {
        "data_dir": "/tmp/ad-data",
        "path_file": "path/derived/example.txt",
        "path_tracking_backend": "profile_stanley",
        "perception_enabled": "false",
    }
    assert not any(
        forbidden in str(include.source).lower()
        for include in RecordingInclude.calls
        for forbidden in ("patchwork", "ogm", "dynamic_obstacle")
    )
