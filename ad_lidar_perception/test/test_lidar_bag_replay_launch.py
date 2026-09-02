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
        "loop": "true",
        "include_front_camera": "false",
        "enable_localization": "false",
        "detector_backend": "euclidean",
        "tracker_backend": "",
        "dynamic_object_risk": "false",
        "checkpoint_path": "",
        "device": "cuda:0",
        "openpcdet_root": "",
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
        "ab3dmot_defer_until_tf_ready": "true",
        "ab3dmot_max_tf_wait_ms": "500",
        "ab3dmot_max_pending_detections": "8",
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
        "loop",
        "include_front_camera",
        "enable_localization",
        "detector_backend",
        "tracker_backend",
        "dynamic_object_risk",
        "checkpoint_path",
        "device",
        "openpcdet_root",
        "composition_config",
        "cluster_config",
        "ground_config",
        "qos_overrides",
        "crop_clearance_m",
        "ab3dmot_defer_until_tf_ready",
        "ab3dmot_max_tf_wait_ms",
        "ab3dmot_max_pending_detections",
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
        "loop": "true",
        "include_front_camera": "false",
        "enable_localization": "false",
        "detector_backend": "euclidean",
        "tracker_backend": "",
        "dynamic_object_risk": "false",
        "checkpoint_path": "",
        "device": "cuda:0",
        "openpcdet_root": "",
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
        "ab3dmot_defer_until_tf_ready": "true",
        "ab3dmot_max_tf_wait_ms": "500",
        "ab3dmot_max_pending_detections": "8",
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


@pytest.mark.parametrize("value", ["yes", "1", "", "falsee"])
def test_include_front_camera_is_a_strict_boolean(value):
    module = load_launch_module()
    with pytest.raises(
        RuntimeError, match="include_front_camera.*true.*false"
    ):
        module._parse_bool("include_front_camera", value)


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
        "detector_backend": "euclidean",
        "tracker_backend": "",
        "dynamic_object_risk": "false",
        "checkpoint_path": "",
        "device": "cuda:0",
        "openpcdet_root": "",
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
        "start_visualization": "false",
        "start_rviz": "false",
        "start_ground_segmentation": "true",
        "ab3dmot_defer_until_tf_ready": "true",
        "ab3dmot_max_tf_wait_ms": "500",
        "ab3dmot_max_pending_detections": "8",
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
        "--loop",
        "--topics",
        *module.SOURCE_TOPICS,
    ]
    assert player.kwargs["output"] == "screen"
    assert player.kwargs["emulate_tty"] is True
    assert not any(
        "/ad/perception/" in token for token in command
    )


def test_centerpoint_replay_uses_cropped_only_detector_contract(
    tmp_path, monkeypatch
):
    module = load_launch_module()
    bag = write_bag(tmp_path)
    checkpoint = tmp_path / "centerpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    openpcdet = tmp_path / "OpenPCDet"
    openpcdet.mkdir()
    group = record_setup(
        module,
        monkeypatch,
        launch_context(
            bag,
            detector_backend="centerpoint",
            checkpoint_path=checkpoint,
            openpcdet_root=openpcdet,
        ),
    )[0]
    perception_arguments = dict(
        group.kwargs["actions"][2].kwargs["launch_arguments"]
    )

    assert perception_arguments["detector_backend"] == "centerpoint"
    assert perception_arguments["checkpoint_path"] == str(checkpoint)
    assert perception_arguments["openpcdet_root"] == str(openpcdet)
    assert perception_arguments["start_ground_segmentation"] == "false"


@pytest.mark.parametrize("value", ["", "center_point", "both"])
def test_rejects_unknown_detector_backend(value):
    module = load_launch_module()
    with pytest.raises(RuntimeError, match="detector_backend"):
        module._parse_detector_backend(value)


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


def test_front_camera_replay_is_explicitly_opt_in(tmp_path, monkeypatch):
    module = load_launch_module()
    bag = write_bag(tmp_path)
    group = record_setup(
        module,
        monkeypatch,
        launch_context(bag, include_front_camera="true"),
    )[0]
    command = group.kwargs["actions"][3].kwargs["actions"][0].kwargs["cmd"]

    assert module.FRONT_CAMERA_TOPIC in command
    assert command.count(module.FRONT_CAMERA_TOPIC) == 1


def test_enable_localization_false_starts_no_localization_group(
    tmp_path, monkeypatch
):
    # Regression coverage for the default (unchanged) replay contract: no
    # second GroupAction, no raw ego-sensor topics, no ad_localization
    # include - byte-identical to the pre-enable_localization behaviour.
    module = load_launch_module()
    bag = write_bag(tmp_path)
    actions = record_setup(
        module, monkeypatch, launch_context(bag, enable_localization="false")
    )

    assert len(actions) == 1
    command = actions[0].kwargs["actions"][3].kwargs["actions"][0].kwargs[
        "cmd"
    ]
    for topic in module.LOCALIZATION_SOURCE_TOPICS:
        assert topic not in command


def test_enable_localization_true_replays_raw_ego_topics_and_starts_localization_unscoped(
    tmp_path, monkeypatch
):
    # Regression test for the real launch-scoping bug found while wiring
    # this feature: nesting the ad_localization include inside the existing
    # scoped=True GroupAction raised, at launch time,
    # "launch configuration 'autostart' does not exist" - localization.
    # launch.py's RegisterEventHandler/EmitEvent lifecycle-autostart
    # condition is evaluated asynchronously, after a *scoped* group has
    # already been popped from the launch context. The fix is a second,
    # separate, scoped=False GroupAction. This test locks that structure:
    # any regression back to nesting it inside the scoped group, or to
    # scoped=True on its own group, must fail this assertion.
    module = load_launch_module()
    bag = write_bag(tmp_path)
    actions = record_setup(
        module, monkeypatch, launch_context(bag, enable_localization="true")
    )

    assert len(actions) == 2
    perception_group, localization_group = actions

    # The original perception/replay group is unaffected: still one scoped
    # group with the same four actions in the same order.
    assert perception_group.kwargs["scoped"] is True
    assert len(perception_group.kwargs["actions"]) == 4

    # The localization group is separate and, critically, NOT scoped.
    assert localization_group.kwargs["scoped"] is False
    loc_actions = localization_group.kwargs["actions"]
    set_param, include = loc_actions
    assert set_param.args == ()
    assert set_param.kwargs == {"name": "use_sim_time", "value": True}
    assert include.args == ("ad_localization/localization.launch.py",)

    # The raw ego-sensor topics ad_localization's gnss_imu backend needs
    # are replayed alongside the existing whitelist, never in place of it.
    command = perception_group.kwargs["actions"][3].kwargs["actions"][
        0
    ].kwargs["cmd"]
    for topic in module.SOURCE_TOPICS:
        assert topic in command
    for topic in module.LOCALIZATION_SOURCE_TOPICS:
        assert command.count(topic) == 1


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
