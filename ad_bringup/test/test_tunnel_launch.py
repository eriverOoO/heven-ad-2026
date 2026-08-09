from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.utilities import perform_substitutions

from ad_bringup import tunnel_stack


PACKAGE = Path(__file__).resolve().parents[1]
MAPPING_LAUNCH = PACKAGE / "launch/tunnel_mapping.launch.py"
LOCALIZATION_LAUNCH = PACKAGE / "launch/tunnel_localization.launch.py"


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


def _render_arguments(description, context):
    return {
        action.name: perform_substitutions(context, action.default_value)
        for action in description.entities
        if isinstance(action, DeclareLaunchArgument)
    }


def _evaluate(arguments, context):
    return {
        name: value.perform(context) if hasattr(value, "perform") else value
        for name, value in dict(arguments).items()
    }


def test_tunnel_launches_keep_separate_small_development_contracts(
    monkeypatch,
):
    monkeypatch.setenv("AD_DATA_DIR", "/tmp/ad-data")
    expected = {
        MAPPING_LAUNCH: {
            "data_dir": "/tmp/ad-data",
            "control_enabled": "false",
            "map_output_path": (
                "/opt/ros/share/ad_localization/maps/cp14_to_cp15.pcd"
            ),
        },
        LOCALIZATION_LAUNCH: {
            "data_dir": "/tmp/ad-data",
            "control_enabled": "false",
            "map_path": "/opt/ros/share/ad_localization/maps/cp14_to_cp15.pcd",
        },
    }

    for path, defaults in expected.items():
        module = _load_launch(path)
        monkeypatch.setattr(
            module,
            "get_package_share_directory",
            lambda package: f"/opt/ros/share/{package}",
            raising=False,
        )
        monkeypatch.setattr(module, "build_tunnel_stack", lambda **_kwargs: [])
        description = module.generate_launch_description()
        assert _render_arguments(description, LaunchContext()) == defaults


def test_mapping_stack_uses_profile_stanley_planner_and_one_tunnel_rviz(
    monkeypatch,
):
    RecordingInclude.calls.clear()
    monkeypatch.setattr(
        tunnel_stack, "IncludeLaunchDescription", RecordingInclude
    )
    monkeypatch.setattr(
        tunnel_stack, "_launch_file", lambda package, name: (package, name)
    )
    monkeypatch.setattr(
        tunnel_stack,
        "get_package_share_directory",
        lambda package: f"/opt/ros/share/{package}",
    )
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "data_dir": "/tmp/ad-data",
            "control_enabled": "false",
            "map_output_path": "/tmp/ad-data/maps/run.pcd",
        }
    )

    actions = tunnel_stack.build_tunnel_stack(
        mode="mapping",
        data_dir=LaunchConfiguration("data_dir"),
        control_enabled=LaunchConfiguration("control_enabled"),
        map_path=LaunchConfiguration("map_output_path"),
    )

    assert actions == RecordingInclude.calls
    assert [include.source for include in RecordingInclude.calls] == [
        ("ad_morai_bridge", "bridge.launch.py"),
        ("ad_description", "description.launch.py"),
        ("ad_localization", "fastlio_mapping.launch.py"),
        ("ad_planner", "planner.launch.py"),
        ("ad_viz", "visualization.launch.py"),
    ]
    assert _evaluate(
        RecordingInclude.calls[0].kwargs["launch_arguments"], context
    ) == {
        "config": "/opt/ros/share/ad_morai_bridge/config/tunnel_fastlio.yaml",
        "control_enabled": "false",
        "enable_velodyne_points": "true",
        "velodyne_point_timing_mode": "zero",
        "velodyne_organize_cloud": "false",
    }
    assert _evaluate(
        RecordingInclude.calls[2].kwargs["launch_arguments"], context
    ) == {
        "platform_profile": "morai",
        "status_topic": "/ad/vehicle/status",
        "autostart": "true",
        "initial_position_override_xy_m": (
            "38.868875371112615,-480.68740975673563"
        ),
        "map_output_path": "/tmp/ad-data/maps/run.pcd",
    }
    assert _evaluate(
        RecordingInclude.calls[3].kwargs["launch_arguments"], context
    ) == {
        "data_dir": "/tmp/ad-data",
        "path_file": "paths/cp14_to_cp15.txt",
        "path_tracking_backend": "profile_stanley",
        "target_speed_mps": "1.0",
        "perception_enabled": "false",
    }
    assert not any(
        "perception" in str(include.source).lower()
        for include in RecordingInclude.calls
    )


def test_localization_stack_selects_fixed_map_leaf(monkeypatch):
    RecordingInclude.calls.clear()
    monkeypatch.setattr(
        tunnel_stack, "IncludeLaunchDescription", RecordingInclude
    )
    monkeypatch.setattr(
        tunnel_stack, "_launch_file", lambda package, name: (package, name)
    )
    monkeypatch.setattr(
        tunnel_stack,
        "get_package_share_directory",
        lambda package: f"/opt/ros/share/{package}",
    )
    tunnel_stack.build_tunnel_stack(
        mode="localization",
        data_dir="/tmp/ad-data",
        control_enabled="false",
        map_path="/tmp/ad-data/maps/fixed.pcd",
    )

    fixed_map = RecordingInclude.calls[2]
    assert fixed_map.source == (
        "ad_localization",
        "fastlio_localization.launch.py",
    )
    assert dict(fixed_map.kwargs["launch_arguments"]) == {
        "platform_profile": "morai",
        "status_topic": "/ad/vehicle/status",
        "autostart": "true",
        "initial_position_override_xy_m": (
            "38.868875371112615,-480.68740975673563"
        ),
        "map_path": "/tmp/ad-data/maps/fixed.pcd",
    }
