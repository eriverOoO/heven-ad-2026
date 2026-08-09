from importlib.util import module_from_spec, spec_from_file_location
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.utilities import perform_substitutions
from launch_ros.actions import SetParameter
import pytest
import yaml

from ad_lidar_perception.selection import SelectionError


PACKAGE = Path(__file__).resolve().parents[1]
LAUNCH = PACKAGE / "launch" / "lidar_perception.launch.py"
DENSIFIER_CONFIG = PACKAGE / "config" / "preprocessing" / "densifier.yaml"
DENSIFIER_SOURCE = (
    PACKAGE / "src" / "preprocessing" / "pointcloud_densifier_node.cpp"
)


class RecordingInclude:
    calls = []

    def __init__(self, source, **kwargs):
        self.source = source
        self.kwargs = kwargs
        self.calls.append(self)


class RecordingNode:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls.append(self)


class RecordingRemap:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls.append(self)


def load_launch_module(name="lidar_perception.launch.py"):
    path = PACKAGE / "launch" / name
    spec = spec_from_file_location(name.replace(".", "_"), path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def composition_text(
    *,
    detector="none",
    tracker="none",
    static=True,
    dynamic=False,
    combined=True,
    build_only=False,
):
    return yaml.safe_dump(
        {
            "schema_version": 1,
            "detector": {
                "backend": detector,
                "model_subdir": "models/autoware",
                "build_only": build_only,
            },
            "tracker": {"backend": tracker},
            "occupancy": {
                "static_enabled": static,
                "dynamic_enabled": dynamic,
                "publish_combined": combined,
            },
        },
        sort_keys=False,
    )


def write_composition(tmp_path, text):
    path = tmp_path / "composition.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def launch_context(config, **overrides):
    values = {
        "composition_config": str(config),
        "raw_input_topic": "/ad/sensors/lidar/points",
        "cluster_config": "/tmp/clustering.yaml",
        "crop_clearance_m": "0.20",
        "start_ground_segmentation": "true",
        "ground_config": "/tmp/ground.yaml",
        "sensor_config": "/tmp/sensors.yaml",
        "sensor_profile": "",
        "platform_profile": "real_hardware",
        "deskew_enabled": "true",
        "deskew_mode": "3d",
        "self_crop_enabled": "true",
        "self_crop_input_reliable": "false",
        "patchwork_leveling_enabled": "true",
        "finite_filter_enabled": "true",
        "densifier_enabled": "false",
        "point_layout_adapter_enabled": "true",
    }
    values.update(overrides)
    context = LaunchContext()
    context.launch_configurations.update(values)
    return context


def record_setup(monkeypatch, config, **overrides):
    module = load_launch_module()
    RecordingInclude.calls.clear()
    monkeypatch.setattr(module, "IncludeLaunchDescription", RecordingInclude)
    monkeypatch.setattr(module, "_launch_file", lambda name: name)
    actions = module._launch_setup(launch_context(config, **overrides))
    assert actions == RecordingInclude.calls
    return module, actions


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            composition_text(),
            [
                "preprocessing.launch.py",
                "ground_segmentation.launch.py",
                "occupancy_grid.launch.py",
                "combined_occupancy_grid.launch.py",
            ],
        ),
        (
            composition_text(detector="centerpoint_tiny"),
            [
                "preprocessing.launch.py",
                "ground_segmentation.launch.py",
                "occupancy_grid.launch.py",
                "object_detection.launch.py",
                "combined_occupancy_grid.launch.py",
            ],
        ),
        (
            composition_text(detector="euclidean_cluster"),
            [
                "preprocessing.launch.py",
                "ground_segmentation.launch.py",
                "occupancy_grid.launch.py",
                "euclidean_clustering.launch.py",
                "combined_occupancy_grid.launch.py",
            ],
        ),
        (
            composition_text(
                detector="euclidean_cluster",
                tracker="autoware",
                dynamic=True,
            ),
            [
                "preprocessing.launch.py",
                "ground_segmentation.launch.py",
                "occupancy_grid.launch.py",
                "euclidean_clustering.launch.py",
                "tracking.launch.py",
                "prediction.launch.py",
                "dynamic_occupancy_grid.launch.py",
                "combined_occupancy_grid.launch.py",
            ],
        ),
        (
            composition_text(
                detector="centerpoint", tracker="autoware"
            ),
            [
                "preprocessing.launch.py",
                "ground_segmentation.launch.py",
                "occupancy_grid.launch.py",
                "object_detection.launch.py",
                "tracking.launch.py",
                "prediction.launch.py",
                "combined_occupancy_grid.launch.py",
            ],
        ),
        (
            composition_text(
                detector="transfusion",
                tracker="autoware",
                dynamic=True,
            ),
            [
                "preprocessing.launch.py",
                "ground_segmentation.launch.py",
                "occupancy_grid.launch.py",
                "object_detection.launch.py",
                "tracking.launch.py",
                "prediction.launch.py",
                "dynamic_occupancy_grid.launch.py",
                "combined_occupancy_grid.launch.py",
            ],
        ),
    ],
)
def test_exact_ordered_leaf_graphs_for_every_composition(
    tmp_path, monkeypatch, text, expected
):
    config = write_composition(tmp_path, text)
    _module, actions = record_setup(monkeypatch, config)
    assert [action.source for action in actions] == expected
    assert "ground_segmentation.launch.py" in expected
    assert "occupancy_grid.launch.py" in expected


def test_optional_branches_forward_only_their_owned_inputs(
    tmp_path, monkeypatch
):
    config = write_composition(
        tmp_path,
        composition_text(
            detector="bevfusion_lidar",
            tracker="autoware",
            dynamic=True,
        ),
    )
    _module, actions = record_setup(
        monkeypatch,
        config,
        platform_profile="real_hardware",
        ground_config="/tmp/custom-ground.yaml",
        sensor_config="/tmp/custom-sensors.yaml",
        sensor_profile="centered",
        raw_input_topic="/bag/lidar/points",
        crop_clearance_m="0.35",
    )
    context = launch_context(
        config,
        platform_profile="real_hardware",
        ground_config="/tmp/custom-ground.yaml",
        sensor_config="/tmp/custom-sensors.yaml",
        sensor_profile="centered",
        raw_input_topic="/bag/lidar/points",
        crop_clearance_m="0.35",
    )

    arguments = {
        action.source: {
            name: value.perform(context) if hasattr(value, "perform") else value
            for name, value in dict(
                action.kwargs.get("launch_arguments", [])
            ).items()
        }
        for action in actions
    }
    assert arguments == {
        "preprocessing.launch.py": {
            "platform_profile": "real_hardware",
            "deskew_enabled": "true",
            "deskew_mode": "3d",
            "self_crop_enabled": "true",
            "self_crop_input_reliable": "false",
            "raw_input_topic": "/bag/lidar/points",
            "crop_clearance_m": "0.35",
            "point_layout_adapter_enabled": "true",
        },
        "ground_segmentation.launch.py": {
            "patchwork_leveling_enabled": "true",
            "cropped_input_topic": "/ad/perception/lidar/cropped",
            "ground_config": "/tmp/custom-ground.yaml",
            "sensor_config": "/tmp/custom-sensors.yaml",
            "sensor_profile": "centered",
        },
        "occupancy_grid.launch.py": {
            "points_topic": "/ad/perception/lidar/cropped",
        },
        "object_detection.launch.py": {
            "selection_config": str(config),
        },
        "tracking.launch.py": {"selection_config": str(config)},
        "prediction.launch.py": {},
        "dynamic_occupancy_grid.launch.py": {},
        "combined_occupancy_grid.launch.py": {},
    }


def test_ground_switch_and_occupancy_switches_have_one_source_of_truth(
    tmp_path, monkeypatch
):
    config = write_composition(
        tmp_path,
        composition_text(
            detector="centerpoint_tiny",
            static=False,
            combined=False,
        ),
    )
    _module, actions = record_setup(
        monkeypatch, config, start_ground_segmentation="false"
    )
    assert [action.source for action in actions] == [
        "preprocessing.launch.py",
        "object_detection.launch.py",
    ]

    config = write_composition(
        tmp_path,
        composition_text(
            detector="centerpoint_tiny",
            tracker="autoware",
            static=False,
            dynamic=True,
            combined=False,
        ),
    )
    _module, actions = record_setup(monkeypatch, config)
    names = [action.source for action in actions]
    assert "occupancy_grid.launch.py" not in names
    assert "combined_occupancy_grid.launch.py" not in names
    assert "dynamic_occupancy_grid.launch.py" in names


@pytest.mark.parametrize(
    "text, diagnostic",
    [
        (
            composition_text(detector="none", tracker="autoware"),
            "tracker requires a non-none detector",
        ),
        (
            composition_text(
                detector="centerpoint_tiny",
                tracker="none",
                dynamic=True,
            ),
            "dynamic occupancy requires tracker autoware",
        ),
        (
            composition_text(static=False, combined=True),
            "combined occupancy requires static occupancy",
        ),
        (
            composition_text().replace(
                "schema_version: 1\n", "schema_version: 1\nunknown: true\n"
            ),
            "unknown keys",
        ),
        (
            composition_text().replace(
                "backend: none", "backend: none\n  backend: centerpoint", 1
            ),
            "duplicate key",
        ),
        (
            composition_text().replace(
                "static_enabled: true", 'static_enabled: "true"'
            ),
            "must be a boolean",
        ),
    ],
)
def test_composition_rejects_invalid_documents(
    tmp_path, monkeypatch, text, diagnostic
):
    config = write_composition(tmp_path, text)
    with pytest.raises(SelectionError, match=diagnostic):
        record_setup(monkeypatch, config)


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_composition_rejects_missing_or_unreadable_config(
    tmp_path, monkeypatch, kind
):
    config = tmp_path / "missing.yaml"
    if kind == "directory":
        config.mkdir()
    with pytest.raises(SelectionError, match="could not read"):
        record_setup(monkeypatch, config)


def test_build_only_selection_fails_closed_before_any_runtime_action(
    tmp_path, monkeypatch
):
    config = write_composition(
        tmp_path,
        composition_text(detector="centerpoint_tiny", build_only=True),
    )
    module = load_launch_module()
    RecordingInclude.calls.clear()
    monkeypatch.setattr(module, "IncludeLaunchDescription", RecordingInclude)
    monkeypatch.setattr(module, "_launch_file", lambda name: name)

    with pytest.raises(RuntimeError, match="build_only"):
        module._launch_setup(launch_context(config))
    assert RecordingInclude.calls == []


def test_default_graph_has_one_common_preprocessing_and_no_optional_backends(
    tmp_path, monkeypatch
):
    config = write_composition(tmp_path, composition_text())
    module, actions = record_setup(monkeypatch, config)
    names = [action.source.lower() for action in actions]
    assert not any(
        token in "\n".join(names)
        for token in (
            "object_detection",
            "tracking",
            "prediction",
            "dynamic",
            "autoware",
        )
    )
    source = LAUNCH.read_text(encoding="utf-8")
    assert names.count("preprocessing.launch.py") == 1
    assert source.count("load_selection(") == 1
    assert "yaml.safe_load" not in source
    assert "verify_selection" not in source
    assert "get_package_share_directory" not in module._launch_setup.__code__.co_names


def test_launch_interface_is_small_and_owns_composition_config(monkeypatch):
    module = load_launch_module()
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda package: str(PACKAGE)
        if package == "ad_lidar_perception"
        else str(PACKAGE.parent / package),
    )
    description = module.generate_launch_description()
    declared_arguments = {
        action.name: action
        for action in description.entities
        if isinstance(action, DeclareLaunchArgument)
    }
    assert set(declared_arguments) == {
        "platform_profile",
        "composition_config",
        "start_ground_segmentation",
        "ground_config",
        "sensor_config",
        "sensor_profile",
        "deskew_enabled",
        "deskew_mode",
        "self_crop_enabled",
        "self_crop_input_reliable",
        "patchwork_leveling_enabled",
        "finite_filter_enabled",
        "densifier_enabled",
        "point_layout_adapter_enabled",
        "raw_input_topic",
        "cluster_config",
        "crop_clearance_m",
        "use_sim_time",
    }
    default_context = LaunchContext()
    assert perform_substitutions(
        default_context,
        declared_arguments["raw_input_topic"].default_value,
    ) == "/ad/sensors/lidar/points"
    assert perform_substitutions(
        default_context,
        declared_arguments["cluster_config"].default_value,
    ).endswith("config/clustering/adaptive_euclidean_cluster.yaml")
    assert perform_substitutions(
        default_context,
        declared_arguments["crop_clearance_m"].default_value,
    ) == "0.20"
    assert perform_substitutions(
        default_context,
        declared_arguments["use_sim_time"].default_value,
    ) == "false"
    assert perform_substitutions(
        default_context,
        declared_arguments["self_crop_input_reliable"].default_value,
    ) == "false"
    assert sum(
        isinstance(action, OpaqueFunction)
        for action in description.entities
    ) == 1
    assert sum(
        isinstance(action, SetParameter)
        for action in description.entities
    ) == 1
    sim_time = next(
        action
        for action in description.entities
        if isinstance(action, SetParameter)
    )
    context = LaunchContext()
    context.launch_configurations["use_sim_time"] = "true"
    assert perform_substitutions(context, sim_time.name) == "use_sim_time"
    assert sim_time.value.evaluate(context) is True
    assert description.entities.index(sim_time) < next(
        index
        for index, action in enumerate(description.entities)
        if isinstance(action, OpaqueFunction)
    )

    source = LAUNCH.read_text(encoding="utf-8")
    assert "start_occupancy_grid" not in source
    assert 'LaunchConfiguration("points_topic")' not in source


def test_full_pipeline_disables_legacy_ground_leveling_by_default(monkeypatch):
    module = load_launch_module()
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda package: str(PACKAGE)
        if package == "ad_lidar_perception"
        else str(PACKAGE.parent / package),
    )
    description = module.generate_launch_description()
    argument = next(
        action
        for action in description.entities
        if isinstance(action, DeclareLaunchArgument)
        and action.name == "patchwork_leveling_enabled"
    )

    assert (
        perform_substitutions(LaunchContext(), argument.default_value)
        == "false"
    )


def test_checked_in_default_tracks_model_free_clusters_for_dynamic_safety():
    from ad_lidar_perception.selection import load_selection

    selection = load_selection(PACKAGE / "config" / "lidar_perception.yaml")
    assert selection.schema_version == 1
    assert selection.detector.backend == "euclidean_cluster"
    assert selection.detector.model_subdir == Path("models/autoware")
    assert selection.detector.build_only is False
    assert selection.tracker.backend == "autoware"
    assert selection.occupancy.static_enabled is True
    assert selection.occupancy.dynamic_enabled is True
    assert selection.occupancy.publish_combined is True


def test_euclidean_cluster_leaf_receives_optional_stage_toggles(
    tmp_path, monkeypatch
):
    config = write_composition(
        tmp_path, composition_text(detector="euclidean_cluster")
    )
    _module, actions = record_setup(monkeypatch, config)
    clustering = next(
        action
        for action in actions
        if action.source == "euclidean_clustering.launch.py"
    )
    context = launch_context(config)
    arguments = {
        name: value.perform(context) if hasattr(value, "perform") else value
        for name, value in dict(clustering.kwargs["launch_arguments"]).items()
    }
    assert arguments == {
        "finite_filter_enabled": "true",
        "densifier_enabled": "false",
        "cluster_config": "/tmp/clustering.yaml",
    }

    dependencies = {
        element.text
        for element in ET.parse(PACKAGE / "package.xml")
        .getroot()
        .findall("exec_depend")
    }
    assert "autoware_euclidean_cluster_object_detector" not in dependencies


def test_euclidean_cluster_leaf_runs_at_root_with_absolute_topics(monkeypatch):
    module = load_launch_module("euclidean_clustering.launch.py")
    RecordingNode.calls.clear()
    monkeypatch.setattr(module, "Node", RecordingNode)
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda package: str(PACKAGE / package),
    )

    description = module.generate_launch_description()
    arguments = {
        action.name
        for action in description.entities
        if isinstance(action, DeclareLaunchArgument)
    }

    assert arguments == {
        "finite_filter_enabled",
        "densifier_enabled",
        "finite_input_topic",
        "finite_output_topic",
        "densified_output_topic",
        "densifier_config",
        "cluster_config",
    }
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "finite_filter_enabled": "true",
            "densifier_enabled": "false",
            "finite_input_topic": "/ad/perception/lidar/nonground",
            "finite_output_topic": "/ad/perception/lidar/nonground_finite",
            "densified_output_topic": "/ad/perception/lidar/nonground_densified",
            "densifier_config": str(
                PACKAGE / "config" / "preprocessing" / "densifier.yaml"
            ),
            "cluster_config": "/tmp/custom-clustering.yaml",
        }
    )
    actions = module._launch_setup(context)
    assert actions == RecordingNode.calls
    assert len(RecordingNode.calls) == 2
    finite_filter = RecordingNode.calls[0].kwargs
    assert finite_filter["name"] == "ad_finite_point_filter"
    assert "namespace" not in finite_filter
    assert str(finite_filter["parameters"][0]).endswith(
        "config/preprocessing/finite_point_filter.yaml"
    )
    assert finite_filter["parameters"][1] == {
        "topics.input": "/ad/perception/lidar/nonground",
        "topics.output": "/ad/perception/lidar/nonground_finite",
    }
    cluster = RecordingNode.calls[1].kwargs
    assert cluster["name"] == "ad_adaptive_euclidean_cluster"
    assert cluster["executable"] == "ad_adaptive_euclidean_cluster_node"
    assert cluster["parameters"][0].perform(context) == (
        "/tmp/custom-clustering.yaml"
    )
    assert cluster["parameters"][1] == {
        "input_topic": "/ad/perception/lidar/nonground_finite"
    }


def euclidean_context(**overrides):
    values = {
        "finite_filter_enabled": "true",
        "densifier_enabled": "false",
        "finite_input_topic": "/ad/perception/lidar/nonground",
        "finite_output_topic": "/ad/perception/lidar/nonground_finite",
        "densified_output_topic": "/ad/perception/lidar/nonground_densified",
        "densifier_config": str(DENSIFIER_CONFIG),
        "cluster_config": str(
            PACKAGE / "config" / "clustering" / "adaptive_euclidean_cluster.yaml"
        ),
    }
    values.update({key: str(value) for key, value in overrides.items()})
    context = LaunchContext()
    context.launch_configurations.update(values)
    return context


def euclidean_nodes(monkeypatch, **overrides):
    module = load_launch_module("euclidean_clustering.launch.py")
    RecordingNode.calls.clear()
    monkeypatch.setattr(module, "Node", RecordingNode)
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda _package: str(PACKAGE),
    )
    actions = module._launch_setup(euclidean_context(**overrides))
    assert actions == RecordingNode.calls
    return [action.kwargs for action in actions]


def test_enabled_densifier_changes_only_finite_to_cluster_edge(monkeypatch):
    nodes = euclidean_nodes(monkeypatch, densifier_enabled="true")

    assert [node["executable"] for node in nodes] == [
        "ad_finite_point_filter_node",
        "ad_pointcloud_densifier_node",
        "ad_adaptive_euclidean_cluster_node",
    ]
    assert nodes[0]["parameters"][1] == {
        "topics.input": "/ad/perception/lidar/nonground",
        "topics.output": "/ad/perception/lidar/nonground_finite",
    }
    assert nodes[1]["parameters"][1] == {
        "topics.input": "/ad/perception/lidar/nonground_finite",
        "topics.output": "/ad/perception/lidar/nonground_densified",
    }
    assert nodes[2]["parameters"][1] == {
        "input_topic": "/ad/perception/lidar/nonground_densified"
    }


@pytest.mark.parametrize(
    ("densifier_enabled", "executables", "densifier_input", "cluster_input"),
    [
        (
            "false",
            ["ad_adaptive_euclidean_cluster_node"],
            None,
            "/ad/perception/lidar/nonground",
        ),
        (
            "true",
            [
                "ad_pointcloud_densifier_node",
                "ad_adaptive_euclidean_cluster_node",
            ],
            "/ad/perception/lidar/nonground",
            "/ad/perception/lidar/nonground_densified",
        ),
    ],
)
def test_disabled_finite_filter_launches_no_node_and_routes_its_predecessor(
    monkeypatch,
    densifier_enabled,
    executables,
    densifier_input,
    cluster_input,
):
    nodes = euclidean_nodes(
        monkeypatch,
        finite_filter_enabled="false",
        densifier_enabled=densifier_enabled,
        finite_output_topic="ignored invalid topic",
    )

    assert [node["executable"] for node in nodes] == executables
    if densifier_input is not None:
        assert nodes[0]["parameters"][1]["topics.input"] == densifier_input
    assert nodes[-1]["parameters"][1] == {"input_topic": cluster_input}


@pytest.mark.parametrize("value", ["yes", "1", "enabled", "", "TRUE-ish"])
def test_euclidean_launch_strictly_rejects_ambiguous_densifier_boolean(
    monkeypatch, value
):
    module = load_launch_module("euclidean_clustering.launch.py")
    monkeypatch.setattr(module, "Node", RecordingNode)
    with pytest.raises(RuntimeError, match="densifier_enabled"):
        module._launch_setup(euclidean_context(densifier_enabled=value))


@pytest.mark.parametrize("value", ["yes", "1", "enabled", "", "TRUE-ish"])
def test_euclidean_launch_strictly_rejects_ambiguous_finite_filter_boolean(
    monkeypatch, value
):
    module = load_launch_module("euclidean_clustering.launch.py")
    monkeypatch.setattr(module, "Node", RecordingNode)
    with pytest.raises(RuntimeError, match="finite_filter_enabled"):
        module._launch_setup(euclidean_context(finite_filter_enabled=value))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"finite_input_topic": "relative"}, "finite_input_topic"),
        ({"finite_output_topic": "/bad topic"}, "finite_output_topic"),
        (
            {"finite_input_topic": "/same", "finite_output_topic": "/same"},
            "duplicate active topic",
        ),
        (
            {
                "densifier_enabled": "true",
                "densified_output_topic": "/bad//topic",
            },
            "densified_output_topic",
        ),
    ],
)
def test_euclidean_launch_validates_active_full_topics(
    monkeypatch, overrides, message
):
    module = load_launch_module("euclidean_clustering.launch.py")
    monkeypatch.setattr(module, "Node", RecordingNode)
    with pytest.raises(RuntimeError, match=message):
        module._launch_setup(euclidean_context(**overrides))


def test_disabled_densifier_ignores_unused_output_topic(monkeypatch):
    nodes = euclidean_nodes(
        monkeypatch,
        densifier_enabled="false",
        densified_output_topic="ignored invalid topic",
    )
    assert [node["executable"] for node in nodes] == [
        "ad_finite_point_filter_node",
        "ad_adaptive_euclidean_cluster_node",
    ]


@pytest.mark.parametrize(
    ("deskew", "crop", "selected"),
    [
        ("true", "true", "/ad/perception/lidar/cropped"),
        ("true", "false", "/ad/perception/lidar/deskewed"),
        ("false", "true", "/ad/perception/lidar/cropped"),
        ("false", "false", "/bag/lidar/points"),
    ],
)
def test_top_level_forwards_toggles_and_routes_selected_common_topic_once(
    tmp_path, monkeypatch, deskew, crop, selected
):
    config = write_composition(
        tmp_path, composition_text(detector="euclidean_cluster")
    )
    _module, actions = record_setup(
        monkeypatch,
        config,
        deskew_enabled=deskew,
        deskew_mode="2d",
        self_crop_enabled=crop,
        patchwork_leveling_enabled="false",
        finite_filter_enabled="true",
        densifier_enabled="true",
        raw_input_topic="/bag/lidar/points",
        crop_clearance_m="0.35",
        cluster_config="/tmp/custom-clustering.yaml",
    )
    context = launch_context(
        config,
        deskew_enabled=deskew,
        deskew_mode="2d",
        self_crop_enabled=crop,
        patchwork_leveling_enabled="false",
        finite_filter_enabled="true",
        densifier_enabled="true",
        raw_input_topic="/bag/lidar/points",
        crop_clearance_m="0.35",
        cluster_config="/tmp/custom-clustering.yaml",
    )
    by_name = {action.source: action for action in actions}
    assert (
        [action.source for action in actions].count(
            "preprocessing.launch.py"
        )
        == 1
    )
    assert {
        key: value.perform(context) if hasattr(value, "perform") else value
        for key, value in dict(
            by_name["preprocessing.launch.py"].kwargs["launch_arguments"]
        ).items()
    } == {
        "platform_profile": "real_hardware",
        "deskew_enabled": deskew,
        "deskew_mode": "2d",
        "self_crop_enabled": crop,
        "self_crop_input_reliable": "false",
        "raw_input_topic": "/bag/lidar/points",
        "crop_clearance_m": "0.35",
        "point_layout_adapter_enabled": "false",
    }
    ground_arguments = dict(
        by_name["ground_segmentation.launch.py"].kwargs["launch_arguments"]
    )
    occupancy_arguments = dict(
        by_name["occupancy_grid.launch.py"].kwargs["launch_arguments"]
    )
    cluster_arguments = dict(
        by_name["euclidean_clustering.launch.py"].kwargs["launch_arguments"]
    )
    assert ground_arguments["cropped_input_topic"] == selected
    assert (
        ground_arguments["patchwork_leveling_enabled"].perform(context)
        == "false"
    )
    assert occupancy_arguments["points_topic"] == selected
    assert cluster_arguments["finite_filter_enabled"].perform(context) == "true"
    assert cluster_arguments["densifier_enabled"].perform(context) == "true"
    assert cluster_arguments["cluster_config"].perform(context) == (
        "/tmp/custom-clustering.yaml"
    )


def test_top_level_morai_profile_prohibits_deskew_before_any_include(
    tmp_path, monkeypatch
):
    config = write_composition(tmp_path, composition_text())
    module = load_launch_module()
    RecordingInclude.calls.clear()
    monkeypatch.setattr(module, "IncludeLaunchDescription", RecordingInclude)
    monkeypatch.setattr(module, "_launch_file", lambda launch_name: launch_name)

    with pytest.raises(RuntimeError, match="MORAI.*deskew.*prohibited"):
        module._launch_setup(
            launch_context(
                config, platform_profile="morai", deskew_enabled="true"
            )
        )

    assert RecordingInclude.calls == []


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("start_ground_segmentation", "yes"),
        ("deskew_enabled", "1"),
        ("self_crop_enabled", "enabled"),
        ("patchwork_leveling_enabled", ""),
        ("finite_filter_enabled", "enabled"),
        ("densifier_enabled", "TRUE-ish"),
        ("point_layout_adapter_enabled", "auto"),
        ("deskew_mode", "planar"),
    ],
)
def test_top_level_strictly_rejects_invalid_toggles_and_mode(
    tmp_path, monkeypatch, name, value
):
    config = write_composition(tmp_path, composition_text())
    module = load_launch_module()
    RecordingInclude.calls.clear()
    monkeypatch.setattr(module, "IncludeLaunchDescription", RecordingInclude)
    monkeypatch.setattr(module, "_launch_file", lambda launch_name: launch_name)
    with pytest.raises(RuntimeError, match=name):
        module._launch_setup(launch_context(config, **{name: value}))
    assert RecordingInclude.calls == []


def test_euclidean_requires_ground_before_any_include_is_created(
    tmp_path, monkeypatch
):
    config = write_composition(
        tmp_path, composition_text(detector="euclidean_cluster")
    )
    module = load_launch_module()
    RecordingInclude.calls.clear()
    monkeypatch.setattr(module, "IncludeLaunchDescription", RecordingInclude)
    monkeypatch.setattr(module, "_launch_file", lambda launch_name: launch_name)
    with pytest.raises(RuntimeError, match="euclidean_cluster.*ground"):
        module._launch_setup(
            launch_context(config, start_ground_segmentation="false")
        )
    assert RecordingInclude.calls == []


@pytest.mark.parametrize("detector", ["none", "euclidean_cluster"])
def test_nonlearned_compositions_disable_unused_point_layout_conversion(
    tmp_path, monkeypatch, detector
):
    config = write_composition(tmp_path, composition_text(detector=detector))
    _module, actions = record_setup(monkeypatch, config)
    preprocessing = next(
        action
        for action in actions
        if action.source == "preprocessing.launch.py"
    )

    assert dict(preprocessing.kwargs["launch_arguments"])[
        "point_layout_adapter_enabled"
    ] == "false"


def test_learned_detector_requires_enabled_point_layout_conversion(
    tmp_path, monkeypatch
):
    config = write_composition(
        tmp_path, composition_text(detector="centerpoint_tiny")
    )
    module = load_launch_module()
    RecordingInclude.calls.clear()
    monkeypatch.setattr(module, "IncludeLaunchDescription", RecordingInclude)
    monkeypatch.setattr(module, "_launch_file", lambda launch_name: launch_name)

    with pytest.raises(RuntimeError, match="layout adapter.*required"):
        module._launch_setup(
            launch_context(config, point_layout_adapter_enabled="false")
        )

    assert RecordingInclude.calls == []


def test_learned_detector_has_no_second_preprocessing_and_fast_lio_is_absent(
    tmp_path, monkeypatch
):
    config = write_composition(
        tmp_path, composition_text(detector="centerpoint_tiny")
    )
    _module, actions = record_setup(monkeypatch, config)
    names = [action.source for action in actions]
    preprocessing = next(
        action
        for action in actions
        if action.source == "preprocessing.launch.py"
    )
    assert names.count("preprocessing.launch.py") == 1
    assert dict(preprocessing.kwargs["launch_arguments"])[
        "point_layout_adapter_enabled"
    ] == "true"
    assert "object_detection.launch.py" in names
    assert not any("fast_lio" in name.lower() for name in names)
    assert "fast_lio" not in LAUNCH.read_text(encoding="utf-8").lower()


def test_densifier_defaults_and_node_contract_are_conservative():
    parameters = yaml.safe_load(DENSIFIER_CONFIG.read_text(encoding="utf-8"))[
        "/**"
    ]["ros__parameters"]
    assert parameters == {
        "topics": {
            "input": "/ad/perception/lidar/nonground_finite",
            "output": "/ad/perception/lidar/nonground_densified",
        },
        "fixed_frame": "odom",
        "voxel_size_m": 0.30,
        "roi": {
            "min_x_m": 20.0,
            "max_x_m": 100.0,
            "min_y_m": -12.0,
            "max_y_m": 12.0,
        },
        "maximum_history_age_sec": 0.25,
        "maximum_translation_jump_m": 5.0,
        "maximum_rotation_jump_rad": 0.35,
        "transform_timeout_sec": 0.05,
    }

    source = DENSIFIER_SOURCE.read_text(encoding="utf-8")
    assert source.count("rclcpp::SensorDataQoS()") == 2
    assert "lookupTransform" in source
    assert "fixed_frame_" in source
    assert "rclcpp::Time(input.header.stamp)" in source
    assert "rclcpp::Time(previous_observation_->stamp)" in source
    assert "std::nullopt" in source
    assert "catch (const std::exception &" in source
    assert "std_msgs/msg/header.hpp" not in source


def test_lidar_config_covers_ioniq5_body_and_local_planner_horizon():
    parameters = yaml.safe_load(
        (PACKAGE / "config" / "occupancy_grid" / "static.yaml").read_text(
            encoding="utf-8"
        )
    )["ad_lidar_perception"]["ros__parameters"]

    assert parameters["x_min"] == -4.0
    assert parameters["x_max"] == 100.0
    assert parameters["y_min"] == -10.0
    assert parameters["y_max"] == 10.0
    assert parameters["z_min"] == 0.1
    assert parameters["z_max"] == 2.0
    assert parameters["resolution"] == 0.1
    assert parameters["inflation_radius_m"] == 1.8
    assert parameters["inflation_cost_scaling_factor"] == 2.0
    assert parameters["ego_clear_x_min"] == -1.0
    assert parameters["ego_clear_x_max"] == 4.05
    assert parameters["ego_clear_y_min"] == -1.15
    assert parameters["ego_clear_y_max"] == 1.15
    assert parameters["transform_timeout_sec"] == 0.05
    assert parameters["road_gate.enabled"] is True
    assert parameters["road_gate.maximum_pending_messages"] == 8
    assert (
        parameters["topics.drivable_mask"]
        == "/ad/planning/drivable_mask"
    )


def test_prediction_horizon_covers_dwa_rollout_braking_and_timeout():
    prediction_parameters = yaml.safe_load(
        (PACKAGE / "config" / "tracking" / "prediction.yaml").read_text(
            encoding="utf-8"
        )
    )["ad_autoware_prediction"]["ros__parameters"]
    planner_parameters = yaml.safe_load(
        (PACKAGE.parent / "ad_planner" / "config" / "planner.yaml").read_text(
            encoding="utf-8"
        )
    )["ad_planner"]["ros__parameters"]
    dwa_parameters = yaml.safe_load(
        (
            PACKAGE.parent
            / "ad_planner"
            / "config"
            / "local_planning"
            / "dwa.yaml"
        ).read_text(encoding="utf-8")
    )["ad_planner"]["ros__parameters"]

    dt_s = dwa_parameters["dwa.simulation_dt"]
    rollout_steps = math.ceil(dwa_parameters["dwa.horizon_sec"] / dt_s)
    braking_steps = math.ceil(
        (
            dwa_parameters["dwa.maximum_speed_mps"]
            / dwa_parameters["dwa.emergency_deceleration_mps2"]
            + dt_s
        )
        / dt_s
    )
    required_horizon_s = (rollout_steps + braking_steps) * dt_s
    timeout_s = planner_parameters["local_motion.prediction_timeout_sec"]

    assert prediction_parameters["horizons_s"][-1] >= (
        required_horizon_s + timeout_s
    )
