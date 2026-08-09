import ast
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from ad_morai_bridge_dev.perception import mcap_audit


class Value:
    def __init__(self, key, value):
        self.key = key
        self.value = value


class Status:
    def __init__(self, level, name, message, values=()):
        self.level = level
        self.name = name
        self.message = message
        self.values = list(values)


class Stamp:
    def __init__(self, nanoseconds):
        self.sec, self.nanosec = divmod(nanoseconds, 1_000_000_000)


class Header:
    def __init__(self, nanoseconds):
        self.stamp = Stamp(nanoseconds)


class Cloud:
    def __init__(self, stamp_ns, width, height=1):
        self.header = Header(stamp_ns)
        self.width = width
        self.height = height


class Objects:
    def __init__(self, stamp_ns, count):
        self.header = Header(stamp_ns)
        self.objects = [object() for _ in range(count)]


class Diagnostics:
    def __init__(self, stamp_ns, statuses):
        self.header = Header(stamp_ns)
        self.status = list(statuses)


class TopicMetadata:
    def __init__(self, name, type_name):
        self.name = name
        self.type = type_name


class Options:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeReader:
    def __init__(self, records, topics):
        self.records = list(records)
        self.topics = list(topics)
        self.opened = None
        self.filter = None

    def open(self, storage, converter):
        self.opened = (storage, converter)

    def get_all_topics_and_types(self):
        return self.topics

    def set_filter(self, selected_filter):
        self.filter = selected_filter

    def has_next(self):
        return bool(self.records)

    def read_next(self):
        return self.records.pop(0)


class FakeRuntime:
    StorageOptions = Options
    ConverterOptions = Options
    StorageFilter = Options

    def __init__(self, reader):
        self.reader = reader
        self.resolved_types = []
        self.deserialized_types = []

    def SequentialReader(self):
        return self.reader

    def get_message(self, type_name):
        self.resolved_types.append(type_name)
        return type_name

    def deserialize_message(self, serialized, message_type):
        self.deserialized_types.append(message_type)
        return serialized


def _topics():
    return [
        TopicMetadata(spec.topic, spec.type_name)
        for spec in mcap_audit.STAGES.values()
    ] + [
        TopicMetadata(
            mcap_audit.DIAGNOSTIC_TOPIC,
            "diagnostic_msgs/msg/DiagnosticArray",
        ),
        TopicMetadata("/unselected/camera", "sensor_msgs/msg/Image"),
    ]


def _record(topic, message, storage_stamp=0):
    return (topic, message, storage_stamp)


def test_fake_reader_filters_selected_topics_and_scores_exact_stamp_coverage(
    tmp_path,
):
    bag = tmp_path / "bag"
    bag.mkdir()
    records = [
        _record("/ad/sensors/lidar/points", Cloud(100, 10)),
        _record("/ad/sensors/lidar/points", Cloud(200, 20)),
        _record("/ad/sensors/lidar/points", Cloud(300, 30)),
        _record("/ad/perception/lidar/cropped", Cloud(100, 8)),
        _record("/ad/perception/lidar/cropped", Cloud(300, 24)),
        _record("/ad/perception/lidar/cropped", Cloud(400, 4)),
        _record("/ad/perception/lidar/nonground", Cloud(100, 5)),
        _record("/ad/perception/lidar/nonground", Cloud(200, 6)),
        _record("/ad/perception/lidar/nonground", Cloud(300, 7)),
        _record("/ad/perception/lidar/nonground_finite", Cloud(100, 5)),
        _record("/ad/perception/lidar/nonground_finite", Cloud(200, 6)),
        _record("/ad/perception/lidar/nonground_finite", Cloud(200, 6)),
        _record("/ad/perception/objects/detected", Objects(100, 2)),
        _record("/ad/perception/objects/detected", Objects(300, 1)),
        _record("/ad/perception/objects/tracked", Objects(100, 1)),
        _record("/ad/perception/objects/predicted", Objects(100, 1)),
        _record(
            mcap_audit.DIAGNOSTIC_TOPIC,
            Diagnostics(
                100,
                [
                    Status(
                        0,
                        "track-a",
                        "IMM measurement accepted",
                        [Value("reset_or_gating_reason", "track_initialized")],
                    ),
                    Status(
                        2,
                        "tracked_object_array",
                        "Tracked-object prediction input rejected",
                        [
                            Value(
                                "reset_or_gating_reason",
                                "rejected_stale_array_gate",
                            )
                        ],
                    ),
                ],
            ),
        ),
    ]
    reader = FakeReader(records, _topics())
    runtime = FakeRuntime(reader)

    report = mcap_audit.audit_bag(bag, runtime=runtime)

    assert reader.opened[0].uri == str(bag.resolve())
    assert reader.opened[0].storage_id == "mcap"
    assert set(reader.filter.topics) == set(mcap_audit.SELECTED_TOPICS)
    assert "/unselected/camera" not in reader.filter.topics
    assert set(runtime.resolved_types) == {
        metadata.type for metadata in _topics()[:-1]
    }
    assert report["stages"]["input"]["message_count"] == 3
    assert report["stages"]["input"]["point_count"] == {
        "total": 60,
        "minimum": 10,
        "maximum": 30,
        "mean": 20.0,
    }
    assert report["stages"]["detected"]["object_count"]["total"] == 3
    assert report["stages"]["finite"]["header_stamp"][
        "duplicate_count"
    ] == 1
    assert report["stages"]["finite"]["header_stamp"][
        "non_increasing_count"
    ] == 1

    crop = report["exact_stamp_coverage"]["crop"]
    assert crop == {
        "topic": "/ad/perception/lidar/cropped",
        "matching_unique_stamps": 2,
        "missing_input_stamps": 1,
        "unexpected_output_stamps": 1,
        "coverage_ratio": pytest.approx(2.0 / 3.0),
    }
    assert report["exact_stamp_coverage"]["predicted"][
        "coverage_ratio"
    ] == pytest.approx(1.0 / 3.0)
    assert report["diagnostics"] == {
        "message_count": 1,
        "status_count": 2,
        "levels": {"ERROR": 1, "OK": 1},
        "names": {"track-a": 1, "tracked_object_array": 1},
        "messages": {
            "IMM measurement accepted": 1,
            "Tracked-object prediction input rejected": 1,
        },
        "reasons": {
            "reset_or_gating_reason": {
                "rejected_stale_array_gate": 1,
                "track_initialized": 1,
            }
        },
    }


@pytest.mark.parametrize("level", [b"\x01", bytearray(b"\x02")])
def test_diagnostic_level_accepts_humble_uint8_byte_representation(level):
    expected = "WARN" if level[0] == 1 else "ERROR"

    assert mcap_audit._level_name(level) == expected


@pytest.mark.parametrize("level", [b"", b"\x01\x02"])
def test_diagnostic_level_rejects_malformed_byte_sequences(level):
    with pytest.raises(ValueError, match="must have length one"):
        mcap_audit._level_name(level)


def test_reader_rejects_recorded_type_drift_before_deserializing(tmp_path):
    bag = tmp_path / "bag"
    bag.mkdir()
    topics = _topics()
    topics[0] = TopicMetadata(
        "/ad/sensors/lidar/points", "sensor_msgs/msg/Image"
    )
    runtime = FakeRuntime(FakeReader([], topics))

    with pytest.raises(ValueError, match="recorded topic type mismatch"):
        mcap_audit.audit_bag(bag, runtime=runtime)

    assert runtime.resolved_types == []
    assert runtime.deserialized_types == []


def test_missing_stages_are_explicit_without_dividing_by_zero(tmp_path):
    bag = tmp_path / "bag"
    bag.mkdir()
    runtime = FakeRuntime(FakeReader([], []))

    report = mcap_audit.audit_bag(bag, runtime=runtime)

    assert report["missing_topics"] == list(mcap_audit.SELECTED_TOPICS)
    assert report["stages"]["input"]["message_count"] == 0
    assert report["stages"]["input"]["point_count"]["mean"] is None
    assert all(
        item["coverage_ratio"] is None
        for item in report["exact_stamp_coverage"].values()
    )


def test_reports_are_finite_json_and_markdown_with_diagnostic_reasons(tmp_path):
    report = {
        "schema_version": 1,
        "bag_path": "/read-only/bag",
        "selected_topics": list(mcap_audit.SELECTED_TOPICS),
        "missing_topics": [],
        "stages": {
            "input": {
                "topic": "/input",
                "message_count": 1,
                "header_stamp": {
                    "first_ns": 10,
                    "last_ns": 10,
                    "unique_count": 1,
                    "duplicate_count": 0,
                    "non_increasing_count": 0,
                },
                "point_count": {
                    "total": 4,
                    "minimum": 4,
                    "maximum": 4,
                    "mean": 4.0,
                },
            }
        },
        "exact_stamp_coverage": {},
        "diagnostics": {
            "message_count": 1,
            "status_count": 1,
            "levels": {"WARN": 1},
            "names": {"track": 1},
            "messages": {"warning": 1},
            "reasons": {"reset_or_gating_reason": {"clock_rollback": 1}},
        },
    }

    json_path, markdown_path = mcap_audit.write_reports(report, tmp_path)

    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "MCAP Replay Audit" in markdown
    assert "clock_rollback" in markdown
    assert "| input | /input | 1 | 1 |" in markdown


def test_run_audit_never_places_outputs_inside_original_bag(tmp_path):
    bag = tmp_path / "bag"
    bag.mkdir()
    runtime = FakeRuntime(FakeReader([], _topics()))

    with pytest.raises(ValueError, match="outside the bag directory"):
        mcap_audit.run_audit(bag, bag / "audit", runtime=runtime)

    assert list(bag.iterdir()) == []


def test_cli_routes_paths_and_prints_both_reports(tmp_path, monkeypatch, capsys):
    bag = tmp_path / "bag"
    output = tmp_path / "audit"
    json_path = output / "mcap_replay_audit.json"
    markdown_path = output / "mcap_replay_audit.md"
    calls = []
    monkeypatch.setattr(
        mcap_audit,
        "run_audit",
        lambda bag_path, output_dir: (
            calls.append((bag_path, output_dir)),
            (json_path, markdown_path),
        )[1],
    )

    assert mcap_audit.main([str(bag), "--output-dir", str(output)]) == 0

    assert calls == [(bag, output)]
    assert capsys.readouterr().out.splitlines() == [
        str(json_path),
        str(markdown_path),
    ]


def test_console_entry_and_runtime_dependencies_are_declared():
    package = Path(__file__).resolve().parents[1]
    setup_text = (package / "setup.py").read_text(encoding="utf-8")
    setup_strings = {
        item.value
        for item in ast.walk(ast.parse(setup_text))
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    }
    assert (
        "ad_morai_perception_mcap_audit = "
        "ad_morai_bridge_dev.perception.mcap_audit:main"
    ) in setup_strings

    dependencies = {
        element.text
        for element in ET.parse(package / "package.xml")
        .getroot()
        .findall("exec_depend")
    }
    assert {
        "ad_interfaces",
        "autoware_perception_msgs",
        "diagnostic_msgs",
        "rosbag2_py",
        "rosbag2_storage_mcap",
        "rosidl_runtime_py",
    } <= dependencies


def test_rosbag2_is_imported_only_when_default_runtime_is_requested():
    source = Path(mcap_audit.__file__).read_text(encoding="utf-8")
    load_start = source.index("def _load_rosbag_runtime")
    assert "import rosbag2_py" not in source[:load_start]
    assert "import rosbag2_py" in source[load_start:]
