#!/usr/bin/env python3
"""Record DetectedObjects on a topic to JSONL for Euclidean/CenterPoint comparison.

Run once per backend (each pointed at the same replayed MORAI frames from
``ad_publish_morai_frames``) to collect comparable ``heven.ros_detection_
comparison.v1`` records: object count, per-object class/score/box, and
publish-to-receipt latency. Purely descriptive -- no accuracy/mAP claim.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Sequence

import rclpy
from autoware_perception_msgs.msg import DetectedObjects, ObjectClassification
from rclpy.node import Node

from ad_lidar_perception.detection_recording import summarize_detected_objects

LABEL_NAMES = {
    ObjectClassification.UNKNOWN: "unknown",
    ObjectClassification.CAR: "vehicle",
    ObjectClassification.PEDESTRIAN: "pedestrian",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("euclidean", "centerpoint"))
    parser.add_argument("--topic", default="/ad/perception/objects/detected")
    parser.add_argument("--count", type=int, default=5, help="Messages to record before exiting.")
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.count < 1:
        parser.error("--count must be positive")
    if args.timeout_sec <= 0:
        parser.error("--timeout-sec must be positive")
    return args


def write_jsonl_atomic(records: list[dict], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records: list[dict] = []

    rclpy.init()
    node = Node("ad_record_detected_objects")

    def _on_message(message: DetectedObjects) -> None:
        received_ns = node.get_clock().now().nanoseconds
        record = summarize_detected_objects(
            message,
            backend=args.backend,
            received_wall_time_ns=received_ns,
            label_names=LABEL_NAMES,
        )
        records.append(record)
        node.get_logger().info(
            f"recorded backend={args.backend} objects={record['object_count']} "
            f"latency_ms={record['latency_ms']:.3f} ({len(records)}/{args.count})"
        )

    node.create_subscription(DetectedObjects, args.topic, _on_message, 10)
    deadline = time.monotonic() + args.timeout_sec
    try:
        while len(records) < args.count and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.5)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

    if not records:
        raise RuntimeError(f"no {args.topic} messages received within {args.timeout_sec}s")
    write_jsonl_atomic(records, args.output.resolve())
    summary = {
        "backend": args.backend,
        "messages_recorded": len(records),
        "mean_object_count": sum(r["object_count"] for r in records) / len(records),
        "mean_latency_ms": sum(r["latency_ms"] for r in records) / len(records),
        "output_path": str(args.output.resolve()),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
