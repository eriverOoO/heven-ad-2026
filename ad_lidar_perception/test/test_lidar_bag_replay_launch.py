from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.utilities import perform_substitutions
import pytest
import yaml


PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parent
LAUNCH = PACKAGE / "launch" / "lidar_bag_replay.launch.py"


def load_launch_module():
    spec = spec_from_file_location("ad_lidar_bag_replay_launch", LAUNCH)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_bag(tmp_path, *, mutate=None):
    bag = tmp_path / "sample_bag"
    bag.mkdir()
    metadata = {
        "rosbag2_bagfile_information": {
            "storage_identifier": "mcap",
            "relative_file_paths": ["sample_0.mcap"],
            "topics_with_message_count": [
                {
                    "topic_metadata": {
                        "name": "/ad/sensors/lidar/points",
                        "type": "sensor_msgs/msg/PointCloud2",
                        "serialization_format": "cdr",
                    },
                    "message_count": 1,
                }
            ],
        }
    }
    if mutate is not None:
        mutate(metadata)
    (bag / "metadata.yaml").write_text(
        yaml.safe_dump(metadata), encoding="utf-8"
    )
    (bag / "sample_0.mcap").write_bytes(b"\x89MCAP0\r\n")
    return bag


def launch_context(bag, **overrides):
    values = {
        "bag_path": str(bag),
        "rate": "1.0",
        "startup_delay_sec": "2.0",
        "start_paused": "false",
        "composition_config": str(
            PACKAGE / "config" / "lidar_perception_morai_classical.yaml"
        ),
        "cluster_config": str(
            PACKAGE
            / "config"
            / "clustering"
            / "adaptive_euclidean_cluster.yaml"
        ),
        "ground_config": str(
            PACKAGE
            / "config"
            / "preprocessing"
            / "ground_segmentation.yaml"
        ),
        "qos_overrides": str(
            PACKAGE / "config" / "replay_qos_overrides.yaml"
        ),
        "crop_clearance_m": "0.20",
    }
    values.update({name: str(value) for name, value in overrides.items()})
    context = LaunchContext()
    context.launch_configurations.update(values)
    return context


class RecordingAction:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def record_setup(module, monkeypatch, context):
    for name in (
        "ExecuteProcess",
        "GroupAction",
        "IncludeLaunchDescription",
        "SetParameter",
        "TimerAction",
    ):
        monkeypatch.setattr(module, name, RecordingAction)
    monkeypatch.setattr(
        module,
        "_launch_file",
        lambda package, name: f"{package}/{name}",
    )
    return module._launch_setup(context)


def test_declares_only_safe_replay_controls_and_installed_defaults(
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
        "composition_config",
        "cluster_config",
        "ground_config",
        "qos_overrides",
        "crop_clearance_m",
    }
    assert arguments["bag_path"] is None
    context = LaunchContext()
    assert {
        name: perform_substitutions(context, value)
        for name, value in arguments.items()
        if name != "bag_path"
    } == {
        "rate": "0.5",
        "startup_delay_sec": "2.0",
        "start_paused": "false",
        "composition_config": str(
            PACKAGE / "config" / "lidar_perception_morai_classical.yaml"
        ),
        "cluster_config": str(
            PACKAGE
            / "config"
            / "clustering"
            / "adaptive_euclidean_cluster.yaml"
        ),
        "ground_config": str(
            PACKAGE
            / "config"
            / "preprocessing"
            / "ground_segmentation.yaml"
        ),
        "qos_overrides": str(
            PACKAGE / "config" / "replay_qos_overrides.yaml"
        ),
        "crop_clearance_m": "0.20",
    }


def test_validates_metadata_and_all_referenced_mcap_files(tmp_path):
    module = load_launch_module()
    bag = write_bag(tmp_path)

    assert module._validate_bag_path(str(bag)) == bag.resolve()

    second = bag / "sample_1.mcap"
    second.write_bytes(b"\x89MCAP0\r\n")
    metadata = yaml.safe_load(
        (bag / "metadata.yaml").read_text(encoding="utf-8")
    )
    metadata["rosbag2_bagfile_information"]["relative_file_paths"].append(
        second.name
    )
    (bag / "metadata.yaml").write_text(
        yaml.safe_dump(metadata), encoding="utf-8"
    )
    assert module._validate_bag_path(str(bag)) == bag.resolve()


@pytest.mark.parametrize(
    ("mutation", "diagnostic"),
    [
        (
            lambda data: data["rosbag2_bagfile_information"].update(
                storage_identifier="sqlite3"
            ),
            "storage_identifier.*mcap",
        ),
        (
            lambda data: data["rosbag2_bagfile_information"].update(
                relative_file_paths=[]
            ),
            "relative_file_paths",
        ),
        (
            lambda data: data["rosbag2_bagfile_information"].update(
                relative_file_paths=["../escape.mcap"]
            ),
            "unsafe MCAP path",
        ),
        (
            lambda data: data["rosbag2_bagfile_information"].update(
                relative_file_paths=["sample_0.db3"]
            ),
            "must end in .mcap",
        ),
        (
            lambda data: data["rosbag2_bagfile_information"].update(
                relative_file_paths=["missing.mcap"]
            ),
            "missing MCAP file",
        ),
    ],
)
def test_rejects_invalid_or_unsafe_metadata(
    tmp_path, mutation, diagnostic
):
    module = load_launch_module()
    bag = write_bag(tmp_path, mutate=mutation)
    with pytest.raises(RuntimeError, match=diagnostic):
        module._validate_bag_path(str(bag))


@pytest.mark.parametrize(
    ("path_value", "diagnostic"),
    [
        ("", "bag_path is required"),
        ("relative/bag", "absolute"),
    ],
)
def test_rejects_empty_and_relative_bag_paths(path_value, diagnostic):
    module = load_launch_module()
    with pytest.raises(RuntimeError, match=diagnostic):
        module._validate_bag_path(path_value)


def test_rejects_missing_metadata_and_non_directory_paths(tmp_path):
    module = load_launch_module()
    missing = tmp_path / "missing"
    with pytest.raises(RuntimeError, match="does not exist"):
        module._validate_bag_path(str(missing))

    file_path = tmp_path / "not_a_bag"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeError, match="must be a directory"):
        module._validate_bag_path(str(file_path))

    directory = tmp_path / "no_metadata"
    directory.mkdir()
    with pytest.raises(RuntimeError, match="metadata.yaml"):
        module._validate_bag_path(str(directory))


@pytest.mark.parametrize(
    ("value", "diagnostic"),
    [
        ("0", "positive"),
        ("-1", "positive"),
        ("nan", "finite"),
        ("inf", "finite"),
        ("fast", "numeric"),
    ],
)
def test_rejects_unsafe_replay_rates(value, diagnostic):
    module = load_launch_module()
    with pytest.raises(RuntimeError, match=diagnostic):
        module._parse_rate(value)


@pytest.mark.parametrize(
    ("value", "diagnostic"),
    [
        ("-0.1", "nonnegative"),
        ("61", "at most 60"),
        ("nan", "finite"),
        ("soon", "numeric"),
    ],
)
def test_rejects_unsafe_startup_delays(value, diagnostic):
    module = load_launch_module()
    with pytest.raises(RuntimeError, match=diagnostic):
        module._parse_startup_delay(value)


@pytest.mark.parametrize("value", ["yes", "1", "", "falsee"])
def test_start_paused_is_a_strict_boolean(value):
    module = load_launch_module()
    with pytest.raises(RuntimeError, match="start_paused.*true.*false"):
        module._parse_bool("start_paused", value)


def test_graph_scopes_sim_time_and_replays_only_source_whitelist(
    tmp_path, monkeypatch
):
    module = load_launch_module()
    bag = write_bag(tmp_path)
    actions = record_setup(
        module,
        monkeypatch,
        launch_context(
            bag,
            rate="0.5",
            startup_delay_sec="3.25",
            start_paused="false",
        ),
    )

    assert len(actions) == 1
    group = actions[0]
    scoped = group.kwargs["actions"]
    assert scoped[0].kwargs == {"name": "use_sim_time", "value": True}
    assert scoped[1].args == (
        "ad_description/description.launch.py",
    )
    assert scoped[2].args == (
        "ad_lidar_perception/lidar_perception.launch.py",
    )
    perception_arguments = dict(scoped[2].kwargs["launch_arguments"])
    assert perception_arguments == {
        "composition_config": str(
            PACKAGE / "config" / "lidar_perception_morai_classical.yaml"
        ),
        "cluster_config": str(
            PACKAGE
            / "config"
            / "clustering"
            / "adaptive_euclidean_cluster.yaml"
        ),
        "ground_config": str(
            PACKAGE
            / "config"
            / "preprocessing"
            / "ground_segmentation.yaml"
        ),
        "crop_clearance_m": "0.2",
        "use_sim_time": "true",
        "platform_profile": "morai",
        "deskew_enabled": "false",
        "deskew_mode": "3d",
        "self_crop_enabled": "true",
        "self_crop_input_reliable": "true",
        "patchwork_leveling_enabled": "false",
        "finite_filter_enabled": "true",
        "densifier_enabled": "false",
        "point_layout_adapter_enabled": "false",
        "start_ground_segmentation": "true",
    }

    timer = scoped[3]
    assert timer.kwargs["period"] == 3.25
    assert len(timer.kwargs["actions"]) == 1
    player = timer.kwargs["actions"][0]
    command = player.kwargs["cmd"]
    assert command == [
        "ros2",
        "bag",
        "play",
        "--storage",
        "mcap",
        "--clock",
        "100",
        "--rate",
        "0.5",
        "--qos-profile-overrides-path",
        str(PACKAGE / "config" / "replay_qos_overrides.yaml"),
        "--wait-for-all-acked",
        "10000",
        "--disable-keyboard-controls",
        str(bag.resolve()),
        "--topics",
        *module.SOURCE_TOPICS,
    ]
    assert player.kwargs["output"] == "screen"
    assert player.kwargs["emulate_tty"] is True
    assert not any(
        "/ad/perception/" in token for token in command
    )


def test_start_paused_flag_is_explicit_and_precedes_topics(
    tmp_path, monkeypatch
):
    module = load_launch_module()
    bag = write_bag(tmp_path)
    group = record_setup(
        module, monkeypatch, launch_context(bag, start_paused="true")
    )[0]
    command = group.kwargs["actions"][3].kwargs["actions"][0].kwargs["cmd"]

    assert command.count("--start-paused") == 1
    assert command.index("--start-paused") < command.index("--topics")


def test_composition_config_must_be_an_absolute_regular_yaml(tmp_path):
    module = load_launch_module()
    valid = tmp_path / "selection.yaml"
    valid.write_text("schema_version: 1\n", encoding="utf-8")
    assert module._validate_composition_path(str(valid)) == valid.resolve()

    for invalid in ("relative.yaml", str(tmp_path), str(tmp_path / "x.txt")):
        with pytest.raises(RuntimeError, match="composition_config"):
            module._validate_composition_path(invalid)


@pytest.mark.parametrize(
    ("value", "diagnostic"),
    [
        ("-0.01", "nonnegative"),
        ("2.01", "at most 2"),
        ("nan", "finite"),
        ("inf", "finite"),
        ("wide", "numeric"),
    ],
)
def test_crop_clearance_rejects_unsafe_values(value, diagnostic):
    module = load_launch_module()
    with pytest.raises(RuntimeError, match=diagnostic):
        module._parse_crop_clearance(value)


@pytest.mark.parametrize("argument", ["cluster_config", "ground_config"])
def test_tuning_configs_must_be_absolute_existing_regular_yaml(
    tmp_path, argument
):
    module = load_launch_module()
    valid = tmp_path / f"{argument}.yaml"
    valid.write_text("/**:\n  ros__parameters: {}\n", encoding="utf-8")
    assert module._validate_yaml_path(argument, str(valid)) == valid.resolve()

    invalid_values = (
        "relative.yaml",
        str(tmp_path),
        str(tmp_path / f"missing-{argument}.yaml"),
        str(tmp_path / "wrong.txt"),
    )
    for invalid in invalid_values:
        with pytest.raises(RuntimeError, match=argument):
            module._validate_yaml_path(argument, invalid)
