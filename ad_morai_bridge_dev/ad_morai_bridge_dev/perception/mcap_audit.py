"""Read-only audit of recorded LiDAR perception stages in an MCAP rosbag."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


_NANOSECONDS_PER_SECOND = 1_000_000_000


@dataclass(frozen=True)
class StageSpec:
    topic: str
    type_name: str
    count_field: str


STAGES: Mapping[str, StageSpec] = {
    "input": StageSpec(
        "/ad/sensors/lidar/points",
        "sensor_msgs/msg/PointCloud2",
        "point_count",
    ),
    "crop": StageSpec(
        "/ad/perception/lidar/cropped",
        "sensor_msgs/msg/PointCloud2",
        "point_count",
    ),
    "nonground": StageSpec(
        "/ad/perception/lidar/nonground",
        "sensor_msgs/msg/PointCloud2",
        "point_count",
    ),
    "finite": StageSpec(
        "/ad/perception/lidar/nonground_finite",
        "sensor_msgs/msg/PointCloud2",
        "point_count",
    ),
    "detected": StageSpec(
        "/ad/perception/objects/detected",
        "autoware_perception_msgs/msg/DetectedObjects",
        "object_count",
    ),
    "tracked": StageSpec(
        "/ad/perception/objects/tracked",
        "autoware_perception_msgs/msg/TrackedObjects",
        "object_count",
    ),
    "predicted": StageSpec(
        "/ad/perception/objects/predicted",
        "ad_interfaces/msg/PredictedObjectArray",
        "object_count",
    ),
}

DIAGNOSTIC_TOPIC = "/ad/perception/objects/prediction_debug"
DIAGNOSTIC_TYPE = "diagnostic_msgs/msg/DiagnosticArray"
SELECTED_TOPICS = tuple(
    [spec.topic for spec in STAGES.values()] + [DIAGNOSTIC_TOPIC]
)


@dataclass(frozen=True)
class _RosbagRuntime:
    SequentialReader: Any
    StorageOptions: Any
    ConverterOptions: Any
    StorageFilter: Any
    deserialize_message: Any
    get_message: Any


def _load_rosbag_runtime() -> _RosbagRuntime:
    """Import rosbag support only for an actual audit invocation."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    return _RosbagRuntime(
        SequentialReader=rosbag2_py.SequentialReader,
        StorageOptions=rosbag2_py.StorageOptions,
        ConverterOptions=rosbag2_py.ConverterOptions,
        StorageFilter=rosbag2_py.StorageFilter,
        deserialize_message=deserialize_message,
        get_message=get_message,
    )


def _validated_bag_directory(path: Path) -> Path:
    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"bag path does not exist: {candidate}") from error
    if not resolved.is_dir():
        raise ValueError(f"bag path must be a directory: {candidate}")
    return resolved


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _header_stamp_ns(message: object, topic: str) -> int:
    try:
        stamp = message.header.stamp
        seconds = stamp.sec
        nanoseconds = stamp.nanosec
    except AttributeError as error:
        raise ValueError(f"{topic} message has no header stamp") from error
    seconds = _integer(seconds, f"{topic} header seconds")
    nanoseconds = _integer(nanoseconds, f"{topic} header nanoseconds")
    if nanoseconds >= _NANOSECONDS_PER_SECOND:
        raise ValueError(f"{topic} header nanoseconds are malformed")
    return seconds * _NANOSECONDS_PER_SECOND + nanoseconds


def _point_count(message: object, topic: str) -> int:
    try:
        width = _integer(message.width, f"{topic} width")
        height = _integer(message.height, f"{topic} height")
    except AttributeError as error:
        raise ValueError(f"{topic} message is not PointCloud2-like") from error
    return width * height


def _object_count(message: object, topic: str) -> int:
    try:
        return len(message.objects)
    except (AttributeError, TypeError) as error:
        raise ValueError(f"{topic} message has no object sequence") from error


class _StageAccumulator:
    def __init__(self, spec: StageSpec):
        self.spec = spec
        self.stamps: list[int] = []
        self.item_counts: list[int] = []
        self.non_increasing_count = 0

    def observe(self, message: object) -> None:
        stamp_ns = _header_stamp_ns(message, self.spec.topic)
        if self.stamps and stamp_ns <= self.stamps[-1]:
            self.non_increasing_count += 1
        self.stamps.append(stamp_ns)
        if self.spec.count_field == "point_count":
            count = _point_count(message, self.spec.topic)
        elif self.spec.count_field == "object_count":
            count = _object_count(message, self.spec.topic)
        else:
            raise RuntimeError(f"unknown count field: {self.spec.count_field}")
        self.item_counts.append(count)

    def summary(self) -> dict[str, object]:
        unique_stamps = set(self.stamps)
        if self.stamps:
            first_ns = min(self.stamps)
            last_ns = max(self.stamps)
        else:
            first_ns = None
            last_ns = None
        if self.item_counts:
            count_summary = {
                "total": sum(self.item_counts),
                "minimum": min(self.item_counts),
                "maximum": max(self.item_counts),
                "mean": sum(self.item_counts) / len(self.item_counts),
            }
        else:
            count_summary = {
                "total": 0,
                "minimum": None,
                "maximum": None,
                "mean": None,
            }
        return {
            "topic": self.spec.topic,
            "message_count": len(self.stamps),
            "header_stamp": {
                "first_ns": first_ns,
                "last_ns": last_ns,
                "unique_count": len(unique_stamps),
                "duplicate_count": len(self.stamps) - len(unique_stamps),
                "non_increasing_count": self.non_increasing_count,
            },
            self.spec.count_field: count_summary,
        }


def _level_name(level: object) -> str:
    # Humble's Python CDR deserializer represents ``uint8`` fields as a
    # single-byte ``bytes`` value on some rosidl runtime versions.
    if isinstance(level, (bytes, bytearray)):
        if len(level) != 1:
            raise ValueError("diagnostic level byte sequence must have length one")
        level = level[0]
    if isinstance(level, bool) or not isinstance(level, int):
        raise ValueError("diagnostic level must be an integer")
    return {0: "OK", 1: "WARN", 2: "ERROR", 3: "STALE"}.get(
        level, f"UNKNOWN_{level}"
    )


class _DiagnosticAccumulator:
    def __init__(self):
        self.message_count = 0
        self.status_count = 0
        self.levels: Counter[str] = Counter()
        self.names: Counter[str] = Counter()
        self.messages: Counter[str] = Counter()
        self.reasons: dict[str, Counter[str]] = {}

    def observe(self, message: object) -> None:
        _header_stamp_ns(message, DIAGNOSTIC_TOPIC)
        try:
            statuses = message.status
        except AttributeError as error:
            raise ValueError("prediction diagnostic has no status sequence") from error
        self.message_count += 1
        for status in statuses:
            self.status_count += 1
            self.levels[_level_name(status.level)] += 1
            self.names[str(status.name)] += 1
            self.messages[str(status.message)] += 1
            for item in status.values:
                key = str(item.key)
                if key == "reason" or key.endswith("_reason"):
                    self.reasons.setdefault(key, Counter())[str(item.value)] += 1

    def summary(self) -> dict[str, object]:
        return {
            "message_count": self.message_count,
            "status_count": self.status_count,
            "levels": _sorted_counter(self.levels),
            "names": _sorted_counter(self.names),
            "messages": _sorted_counter(self.messages),
            "reasons": {
                key: _sorted_counter(values)
                for key, values in sorted(self.reasons.items())
            },
        }


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _expected_types() -> dict[str, str]:
    result = {spec.topic: spec.type_name for spec in STAGES.values()}
    result[DIAGNOSTIC_TOPIC] = DIAGNOSTIC_TYPE
    return result


def _recorded_types(metadata: Sequence[object]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in metadata:
        name = str(item.name)
        type_name = str(item.type)
        previous = result.get(name)
        if previous is not None and previous != type_name:
            raise ValueError(f"recorded topic has conflicting types: {name}")
        result[name] = type_name
    return result


def _coverage(
    input_stamps: set[int], output_stamps: set[int], topic: str
) -> dict[str, object]:
    matching = input_stamps & output_stamps
    ratio = len(matching) / len(input_stamps) if input_stamps else None
    return {
        "topic": topic,
        "matching_unique_stamps": len(matching),
        "missing_input_stamps": len(input_stamps - output_stamps),
        "unexpected_output_stamps": len(output_stamps - input_stamps),
        "coverage_ratio": ratio,
    }


def audit_bag(path: Path, *, runtime: object | None = None) -> dict[str, object]:
    """Read selected baseline topics and return deterministic primitive metrics."""
    bag = _validated_bag_directory(path)
    ros = runtime if runtime is not None else _load_rosbag_runtime()
    reader = ros.SequentialReader()
    reader.open(
        ros.StorageOptions(uri=str(bag), storage_id="mcap"),
        ros.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )

    recorded_types = _recorded_types(reader.get_all_topics_and_types())
    expected_types = _expected_types()
    for topic, expected in expected_types.items():
        actual = recorded_types.get(topic)
        if actual is not None and actual != expected:
            raise ValueError(
                "recorded topic type mismatch for "
                f"{topic}: expected {expected}, got {actual}"
            )

    present_topics = tuple(
        topic for topic in SELECTED_TOPICS if topic in recorded_types
    )
    missing_topics = [
        topic for topic in SELECTED_TOPICS if topic not in recorded_types
    ]
    message_types = {
        topic: ros.get_message(recorded_types[topic])
        for topic in present_topics
    }
    reader.set_filter(ros.StorageFilter(topics=list(SELECTED_TOPICS)))

    by_topic = {spec.topic: name for name, spec in STAGES.items()}
    stage_accumulators = {
        name: _StageAccumulator(spec) for name, spec in STAGES.items()
    }
    diagnostics = _DiagnosticAccumulator()
    while reader.has_next():
        topic, serialized, _storage_stamp = reader.read_next()
        if topic not in message_types:
            raise RuntimeError(f"filtered reader returned unexpected topic: {topic}")
        try:
            message = ros.deserialize_message(serialized, message_types[topic])
        except Exception as error:
            raise ValueError(f"could not deserialize selected topic {topic}") from error
        if topic == DIAGNOSTIC_TOPIC:
            diagnostics.observe(message)
        else:
            stage_accumulators[by_topic[topic]].observe(message)

    stages = {
        name: accumulator.summary()
        for name, accumulator in stage_accumulators.items()
    }
    input_stamps = set(stage_accumulators["input"].stamps)
    exact_stamp_coverage = {
        name: _coverage(
            input_stamps, set(stage_accumulators[name].stamps), spec.topic
        )
        for name, spec in STAGES.items()
        if name != "input"
    }
    return {
        "schema_version": 1,
        "bag_path": str(bag),
        "selected_topics": list(SELECTED_TOPICS),
        "missing_topics": missing_topics,
        "stages": stages,
        "exact_stamp_coverage": exact_stamp_coverage,
        "diagnostics": diagnostics.summary(),
    }


def _markdown_text(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("markdown metric must be finite")
        return f"{value:.6f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _counter_section(title: str, values: Mapping[str, int]) -> list[str]:
    lines = [f"### {title}", "", "| Value | Count |", "|---|---:|"]
    if values:
        lines.extend(
            f"| {_markdown_text(key)} | {count} |"
            for key, count in values.items()
        )
    else:
        lines.append("| _none_ | 0 |")
    lines.append("")
    return lines


def render_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# MCAP Replay Audit",
        "",
        f"- Bag: `{_markdown_text(report['bag_path'])}`",
        f"- Missing selected topics: {len(report['missing_topics'])}",
        "",
        "## Stage statistics",
        "",
        "| Stage | Topic | Messages | Unique stamps | Items | Min | Max | Mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, stage in report["stages"].items():
        count_field = "point_count" if "point_count" in stage else "object_count"
        counts = stage[count_field]
        lines.append(
            "| "
            + " | ".join(
                _markdown_text(value)
                for value in (
                    name,
                    stage["topic"],
                    stage["message_count"],
                    stage["header_stamp"]["unique_count"],
                    counts["total"],
                    counts["minimum"],
                    counts["maximum"],
                    counts["mean"],
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Exact-stamp coverage against LiDAR input",
            "",
            "| Stage | Matches | Missing | Unexpected | Coverage |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, coverage in report["exact_stamp_coverage"].items():
        lines.append(
            "| "
            + " | ".join(
                _markdown_text(value)
                for value in (
                    name,
                    coverage["matching_unique_stamps"],
                    coverage["missing_input_stamps"],
                    coverage["unexpected_output_stamps"],
                    coverage["coverage_ratio"],
                )
            )
            + " |"
        )

    diagnostic = report["diagnostics"]
    lines.extend(
        [
            "",
            "## Prediction diagnostics",
            "",
            f"Messages: {diagnostic['message_count']}; "
            f"statuses: {diagnostic['status_count']}.",
            "",
        ]
    )
    lines.extend(_counter_section("Levels", diagnostic["levels"]))
    lines.extend(_counter_section("Names", diagnostic["names"]))
    lines.extend(_counter_section("Messages", diagnostic["messages"]))
    for key, values in diagnostic["reasons"].items():
        lines.extend(_counter_section(f"Reasons: {key}", values))
    return "\n".join(lines).rstrip() + "\n"


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_reports(
    report: Mapping[str, object], output_dir: Path
) -> tuple[Path, Path]:
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve(strict=True)
    if not output.is_dir():
        raise ValueError("audit output path must be a directory")
    json_path = output / "mcap_replay_audit.json"
    markdown_path = output / "mcap_replay_audit.md"
    json_text = json.dumps(
        report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    _atomic_write(json_path, json_text)
    _atomic_write(markdown_path, render_markdown(report))
    return json_path, markdown_path


def run_audit(
    bag_path: Path, output_dir: Path, *, runtime: object | None = None
) -> tuple[Path, Path]:
    bag = _validated_bag_directory(bag_path)
    output = Path(output_dir).expanduser().resolve(strict=False)
    try:
        output.relative_to(bag)
    except ValueError:
        pass
    else:
        raise ValueError("audit output must be outside the bag directory")
    report = audit_bag(bag, runtime=runtime)
    return write_reports(report, output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="rosbag2 MCAP directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    json_path, markdown_path = run_audit(arguments.bag, arguments.output_dir)
    print(json_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
