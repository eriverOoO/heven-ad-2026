#!/usr/bin/env python3
"""Validate a sequence-disjoint MORAI rosbag2 before an Autoware audit.

The default check is metadata-only and never opens a bag payload.  It is thus
safe to run before the optional Autoware runtime is provisioned.  A PASS means
the recording has the required topic/type/count contract; it is not a claim
that detector or tracking metrics have been evaluated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


REQUIRED_TOPICS = {
    "/ad/sensors/lidar/points": "sensor_msgs/msg/PointCloud2",
    "/ad/dev/objects": "ad_morai_interfaces_dev/msg/ObjectStatusArray",
    "/ad/dev/vehicle/ego_status": "ad_morai_interfaces/msg/EgoVehicleStatus",
    "/tf": "tf2_msgs/msg/TFMessage",
    "/tf_static": "tf2_msgs/msg/TFMessage",
    "/clock": "rosgraph_msgs/msg/Clock",
}


def _topic_index(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    info = metadata.get("rosbag2_bagfile_information", {})
    indexed = {}
    for entry in info.get("topics_with_message_count", []):
        topic = entry.get("topic_metadata", {})
        name = topic.get("name")
        if isinstance(name, str):
            indexed[name] = {"type": topic.get("type"), "count": entry.get("message_count", 0)}
    return indexed


def validate_metadata(metadata: dict[str, Any], bag_path: Path) -> dict[str, Any]:
    """Return deterministic PASS/WARN/FAIL checks from rosbag2 metadata."""
    checks: list[dict[str, str]] = []
    info = metadata.get("rosbag2_bagfile_information")
    if not isinstance(info, dict):
        return {"status": "FAIL", "checks": [{"status": "FAIL", "name": "metadata", "detail": "rosbag2_bagfile_information missing"}]}
    topic_index = _topic_index(metadata)
    for name, expected_type in REQUIRED_TOPICS.items():
        observed = topic_index.get(name)
        if observed is None:
            checks.append({"status": "FAIL", "name": name, "detail": "required topic missing"})
        elif observed["type"] != expected_type:
            checks.append({"status": "FAIL", "name": name, "detail": f"type {observed['type']!r} != {expected_type!r}"})
        elif not isinstance(observed["count"], int) or observed["count"] <= 0:
            checks.append({"status": "FAIL", "name": name, "detail": "zero message count"})
        else:
            checks.append({"status": "PASS", "name": name, "detail": f"{observed['count']} messages"})
    duration_ns = info.get("duration", {}).get("nanoseconds", 0)
    if not isinstance(duration_ns, int) or duration_ns <= 0:
        checks.append({"status": "FAIL", "name": "duration", "detail": "missing or non-positive duration"})
    elif duration_ns < 120_000_000_000:
        checks.append({"status": "WARN", "name": "duration", "detail": f"{duration_ns / 1e9:.1f}s; recommended minimum is 120s"})
    else:
        checks.append({"status": "PASS", "name": "duration", "detail": f"{duration_ns / 1e9:.1f}s"})
    paths = info.get("relative_file_paths", [])
    if not isinstance(paths, list) or not paths:
        checks.append({"status": "FAIL", "name": "payload", "detail": "relative_file_paths missing"})
    else:
        missing = [str(path) for path in paths if not (bag_path / str(path)).is_file()]
        status = "FAIL" if missing else "PASS"
        detail = f"missing payload files: {', '.join(missing)}" if missing else f"{len(paths)} payload file(s) present"
        checks.append({"status": status, "name": "payload", "detail": detail})
    # Metadata does not contain individual stamps or actor-array contents.
    checks.append({"status": "WARN", "name": "deep_content", "detail": "timestamp monotonicity and non-empty actor arrays require --deep with rosbag2_py"})
    overall = "FAIL" if any(item["status"] == "FAIL" for item in checks) else "WARN" if any(item["status"] == "WARN" for item in checks) else "PASS"
    return {"status": overall, "duration_sec": duration_ns / 1e9 if isinstance(duration_ns, int) else None, "topics": topic_index, "checks": checks}


def deep_validate(bag_path: Path, result: dict[str, Any]) -> None:
    """Check per-topic recorded timestamps and that actor arrays are non-empty.

    This is optional because it requires the ROS message packages used at
    recording time. It deliberately fails closed when deserialization cannot
    be performed rather than claiming metadata counts are message validity.
    """
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as error:
        result["checks"].append({"status": "FAIL", "name": "deep_content", "detail": f"ROS bag reader unavailable: {error.name}"})
        result["status"] = "FAIL"
        return
    info = yaml.safe_load((bag_path / "metadata.yaml").read_text())["rosbag2_bagfile_information"]
    storage = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id=info["storage_identifier"])
    converter = rosbag2_py.ConverterOptions("cdr", "cdr")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage, converter)
    types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    previous: dict[str, int] = {}
    actor_nonempty = 0
    timestamp_failure = False
    while reader.has_next():
        topic, raw, timestamp = reader.read_next()
        if topic not in REQUIRED_TOPICS:
            continue
        if timestamp < previous.get(topic, timestamp):
            timestamp_failure = True
        previous[topic] = timestamp
        if topic == "/ad/dev/objects":
            message = deserialize_message(raw, get_message(types[topic]))
            objects = getattr(message, "objects", getattr(message, "object_list", ()))
            actor_nonempty += int(bool(objects))
    if timestamp_failure:
        result["checks"].append({"status": "FAIL", "name": "timestamp_monotonicity", "detail": "per-topic recorded timestamp regression"})
    else:
        result["checks"].append({"status": "PASS", "name": "timestamp_monotonicity", "detail": "per-topic recorded timestamps monotonic"})
    actor_status = "PASS" if actor_nonempty else "FAIL"
    result["checks"].append({"status": actor_status, "name": "non_empty_gt_frames", "detail": str(actor_nonempty) + " non-empty actor messages"})
    result["status"] = "FAIL" if any(check["status"] == "FAIL" for check in result["checks"]) else "WARN" if any(check["status"] == "WARN" for check in result["checks"]) else "PASS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--traffic-configuration", required=True)
    parser.add_argument("--spawn-configuration", required=True)
    parser.add_argument("--used-for-training", choices=("yes", "no"), default="no")
    parser.add_argument("--deep", action="store_true", help="deserialize the payload to validate timestamps and actor-array content")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata_path = args.bag / "metadata.yaml"
    if not metadata_path.is_file():
        result = {"status": "FAIL", "bag": str(args.bag), "checks": [{"status": "FAIL", "name": "metadata", "detail": "metadata.yaml missing"}]}
    else:
        result = validate_metadata(yaml.safe_load(metadata_path.read_text(encoding="utf-8")), args.bag)
        result["bag"] = str(args.bag.resolve())
        if args.deep and result["status"] != "FAIL":
            deep_validate(args.bag, result)
    result["held_out_metadata"] = {
        "scene": args.scene, "route": args.route,
        "traffic_configuration": args.traffic_configuration,
        "spawn_configuration": args.spawn_configuration,
        "used_for_training": args.used_for_training == "yes",
        "purpose": "held_out_centerpoint_audit",
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["status"] != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
