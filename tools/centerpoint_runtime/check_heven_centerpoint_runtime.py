#!/usr/bin/env python3
"""Bounded ROS-CLI health check for the opt-in HEVEN CenterPoint candidate.

The checker intentionally uses only ``ros2`` CLI calls, so it can run from the
same sourced overlay as the candidate launch without importing a Python venv.
Missing live sensor data is reported as ``WAITING_FOR_SENSOR`` rather than as a
detector failure.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, asdict


TOPICS = {
    "points_xyzirc": (
        "/ad/perception/lidar/points_xyzirc", "sensor_msgs/msg/PointCloud2"
    ),
    "detected": (
        "/ad/perception/objects/detected",
        "autoware_perception_msgs/msg/DetectedObjects",
    ),
    "tracked": (
        "/ad/perception/objects/tracked",
        "autoware_perception_msgs/msg/TrackedObjects",
    ),
    "predicted": (
        "/ad/perception/objects/predicted",
        "ad_interfaces/msg/PredictedObjectArray",
    ),
}


@dataclass(frozen=True)
class TopicHealth:
    name: str
    topic: str
    expected_type: str
    observed_type: str | None
    publishers: int | None
    subscribers: int | None
    status: str
    detail: str


def parse_topic_info(text: str) -> tuple[str | None, int | None, int | None]:
    """Parse stable fields from ``ros2 topic info -v`` without locale guesses."""
    type_match = re.search(r"^Type:\s*(\S+)", text, flags=re.MULTILINE)
    pub_match = re.search(r"^Publisher count:\s*(\d+)", text, flags=re.MULTILINE)
    sub_match = re.search(r"^Subscription count:\s*(\d+)", text, flags=re.MULTILINE)
    return (
        type_match.group(1) if type_match else None,
        int(pub_match.group(1)) if pub_match else None,
        int(sub_match.group(1)) if sub_match else None,
    )


def classify_topic(
    name: str,
    topic: str,
    expected_type: str,
    observed_type: str | None,
    publishers: int | None,
    subscribers: int | None,
) -> TopicHealth:
    if observed_type is None:
        status = "WAITING_FOR_SENSOR" if name == "points_xyzirc" else "BLOCKED"
        return TopicHealth(name, topic, expected_type, None, publishers, subscribers, status, "topic unavailable")
    if observed_type != expected_type:
        return TopicHealth(name, topic, expected_type, observed_type, publishers, subscribers, "FAIL", "message type mismatch")
    if publishers == 0:
        status = "WAITING_FOR_SENSOR" if name == "points_xyzirc" else "BLOCKED"
        return TopicHealth(name, topic, expected_type, observed_type, publishers, subscribers, status, "no publisher")
    return TopicHealth(name, topic, expected_type, observed_type, publishers, subscribers, "PASS", "type and publisher contract satisfied")


def ros2_topic_info(topic: str, timeout_sec: float) -> str:
    completed = subprocess.run(
        ["ros2", "topic", "info", "-v", topic],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_sec,
        check=False,
    )
    return completed.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-sec", type=float, default=3.0)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.timeout_sec <= 0.0:
        parser.error("--timeout-sec must be positive")
    if shutil.which("ros2") is None:
        result = {"status": "FAIL", "detail": "ros2 executable unavailable", "topics": []}
    else:
        health = []
        for name, (topic, expected_type) in TOPICS.items():
            try:
                observed_type, publishers, subscribers = parse_topic_info(
                    ros2_topic_info(topic, args.timeout_sec)
                )
            except subprocess.TimeoutExpired:
                observed_type = publishers = subscribers = None
            health.append(classify_topic(name, topic, expected_type, observed_type, publishers, subscribers))
        statuses = {item.status for item in health}
        overall = "FAIL" if "FAIL" in statuses else "WAITING_FOR_SENSOR" if "WAITING_FOR_SENSOR" in statuses else "BLOCKED" if "BLOCKED" in statuses else "PASS"
        result = {"status": overall, "topics": [asdict(item) for item in health]}
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(payload)
    print(payload, end="")
    return 0 if result["status"] in {"PASS", "WAITING_FOR_SENSOR"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
