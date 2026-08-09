from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.utilities import perform_substitutions
import yaml

from ad_bringup import bringup_stack


PACKAGE = Path(__file__).resolve().parents[1]
LAUNCH = PACKAGE / "launch" / "bringup.launch.py"
COMPONENT_NAMES = set(bringup_stack.registered_component_names())


def load_launch_module():
    spec = spec_from_file_location("ad_bringup_launch", LAUNCH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RecordingInclude:
    calls = []

    def __init__(self, source, **kwargs):
        self.source = source
        self.kwargs = kwargs
        self.calls.append(self)


def component_file(tmp_path, enabled):
    path = tmp_path / "components.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "components": {
                    name: name in enabled for name in COMPONENT_NAMES
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def launch_context(**overrides):
    values = {
        "data_dir": "/tmp/ad-data",
        "control_enabled": "false",
        "localization_backend": "gnss_imu",
        "path_file": "",
        "path_tracking_backend": "",
        "perception_enabled": "",
    }
    values.update(overrides)
    context = LaunchContext()
    context.launch_configurations.update(values)
    return context


def evaluated_arguments(include, context):
    return {
        name: value.perform(context) if hasattr(value, "perform") else value
        for name, value in dict(include.kwargs["launch_arguments"]).items()
    }


def test_bringup_exposes_exactly_seven_runtime_arguments(monkeypatch):
    module = load_launch_module()
    monkeypatch.delenv("AD_DATA_DIR", raising=False)
    monkeypatch.setattr(
        module, "get_package_share_directory", lambda _package: str(PACKAGE)
    )

    description = module.generate_launch_description()
    arguments = {
        action.name: action
        for action in description.entities
        if isinstance(action, DeclareLaunchArgument)
    }
    context = LaunchContext()

    assert {
        name: perform_substitutions(context, action.default_value)
        for name, action in arguments.items()
    } == {
        "data_dir": "",
        "control_enabled": "false",
        "localization_backend": "gnss_imu",
        "path_file": "",
        "path_tracking_backend": "",
        "perception_enabled": "",
        "local_motion_prediction_mode": "",
    }


def test_old_autonomy_launch_is_not_kept_as_an_alias():
    assert not (PACKAGE / "launch" / "autonomy.launch.py").exists()


def test_bringup_entrypoint_invokes_shared_stack_with_fixed_contract(
    monkeypatch,
):
    module = load_launch_module()
    recorded = []

    def record_stack(**kwargs):
        recorded.append(kwargs)
        return []

    monkeypatch.setattr(module, "build_bringup_stack", record_stack)
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda _package: str(PACKAGE),
    )

    module.generate_launch_description()
    assert recorded == [
        {
            "components_file": str(PACKAGE / "config" / "components.yaml"),
            "status_topic": "/ad/vehicle/status",
        }
    ]


def test_default_component_profile_matches_registered_components():
    assert bringup_stack.load_components(
        PACKAGE / "config" / "components.yaml",
        bringup_stack.registered_component_names(),
    ) == {
        "description": True,
        "localization": True,
        "planner": True,
        "lidar_perception": True,
        "camera_perception": False,
        "bridge": True,
        "visualization": True,
    }


def test_all_enabled_components_use_leaf_launches_and_minimal_forwarding(
    monkeypatch, tmp_path
):
    RecordingInclude.calls.clear()
    monkeypatch.setattr(
        bringup_stack, "IncludeLaunchDescription", RecordingInclude
    )
    monkeypatch.setattr(
        bringup_stack, "_launch_file", lambda package, name: (package, name)
    )
    context = launch_context(
        control_enabled="true",
        path_file="path/derived/example.txt",
        path_tracking_backend="profile_stanley",
    )

    actions = bringup_stack.build_bringup_stack(
        components_file=component_file(tmp_path, COMPONENT_NAMES),
        status_topic="/ad/vehicle/status",
    )

    assert actions == RecordingInclude.calls
    assert [action.source for action in actions] == [
        ("ad_description", "description.launch.py"),
        ("ad_localization", "localization.launch.py"),
        ("ad_planner", "planner.launch.py"),
        ("ad_lidar_perception", "lidar_perception.launch.py"),
        ("ad_camera_perception", "camera_perception.launch.py"),
        ("ad_morai_bridge", "bridge.launch.py"),
        ("ad_viz", "visualization.launch.py"),
    ]
    assert {
        action.source: evaluated_arguments(action, context)
        for action in actions
    } == {
        ("ad_description", "description.launch.py"): {},
        ("ad_localization", "localization.launch.py"): {
            "status_topic": "/ad/vehicle/status",
            "localization_backend": "gnss_imu",
        },
        ("ad_planner", "planner.launch.py"): {
            "data_dir": "/tmp/ad-data",
            "path_file": "path/derived/example.txt",
            "route_corridor_file": "",
            "path_tracking_backend": "profile_stanley",
            "perception_enabled": "",
            "local_motion_prediction_mode": "",
            "tuning_lease_required": "",
        },
        ("ad_lidar_perception", "lidar_perception.launch.py"): {
            "platform_profile": "morai",
        },
        ("ad_camera_perception", "camera_perception.launch.py"): {},
        ("ad_morai_bridge", "bridge.launch.py"): {
            "control_enabled": "true",
            "enable_velodyne_points": "true",
        },
        ("ad_viz", "visualization.launch.py"): {},
    }


def test_bringup_declares_visualization_runtime_dependency():
    package_xml = (PACKAGE / "package.xml").read_text(encoding="utf-8")
    assert "<exec_depend>ad_viz</exec_depend>" in package_xml


def test_disabled_components_do_not_lookup_their_packages(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        bringup_stack,
        "get_package_share_directory",
        lambda package: (_ for _ in ()).throw(
            AssertionError(f"disabled component looked up {package}")
        ),
    )

    assert bringup_stack.build_bringup_stack(
        components_file=component_file(tmp_path, set()),
        status_topic="/ad/vehicle/status",
    ) == []


def test_bridge_disables_point_conversion_when_lidar_component_is_off(
    monkeypatch, tmp_path
):
    RecordingInclude.calls.clear()
    monkeypatch.setattr(
        bringup_stack, "IncludeLaunchDescription", RecordingInclude
    )
    monkeypatch.setattr(
        bringup_stack, "_launch_file", lambda package, name: (package, name)
    )
    context = launch_context()

    actions = bringup_stack.build_bringup_stack(
        components_file=component_file(tmp_path, {"bridge"}),
        status_topic="/ad/vehicle/status",
    )

    assert len(actions) == 1
    assert evaluated_arguments(actions[0], context) == {
        "control_enabled": "false",
        "enable_velodyne_points": "false",
    }


def test_bridge_accepts_a_dedicated_runtime_config(monkeypatch, tmp_path):
    RecordingInclude.calls.clear()
    monkeypatch.setattr(
        bringup_stack, "IncludeLaunchDescription", RecordingInclude
    )
    monkeypatch.setattr(
        bringup_stack, "_launch_file", lambda package, name: (package, name)
    )
    bridge_config = tmp_path / "minimal_bridge.yaml"

    actions = bringup_stack.build_bringup_stack(
        components_file=component_file(tmp_path, {"bridge"}),
        status_topic="/ad/vehicle/status",
        bridge_config=bridge_config,
    )

    assert len(actions) == 1
    assert evaluated_arguments(actions[0], launch_context()) == {
        "control_enabled": "false",
        "enable_velodyne_points": "false",
        "config": str(bridge_config),
    }
