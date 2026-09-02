#!/usr/bin/env python3
"""Read-only sensor-bag / live-topic readiness preflight.

Classifies whether a rosbag2 (MCAP) recording - or a currently-running ROS
graph - has enough of the canonical LiDAR + camera + localization + TF
contract (see docs/perception/real_sensor_bag_capture_v1.md) to replay
through ``training_free_perception_rviz.launch.py`` directly, to replay it
with ``enable_localization:=true`` (real ego-sensor recompute via the
existing ``ad_localization`` stack), or neither.

Never rewrites, reindexes, or deletes a bag. Never fabricates a message,
timestamp, or transform. Never trains or runs a detector/tracker.

Two independent tiers:

1. ``classify_bag_metadata()`` - pure ``metadata.yaml`` parsing (PyYAML
   only, no ROS import at all). This is the CORE classification and is
   fully unit-testable without a sourced ROS environment, RViz, CUDA,
   torch, CenterPoint, or KalmanNet.
2. ``inspect_bag_messages()`` - OPTIONAL deeper diagnostics (timestamp
   monotonicity/duplicates, nearest camera<->LiDAR and LiDAR<->odometry
   stamp offsets) that read real message content via ``rosbag2_py``.
   Needs a sourced ROS Python environment (for the message packages) but
   nothing else - no RViz, no GPU.

``check_live_topics()`` is a separate, THIRD tier for a currently-running
ROS graph (needs ``rclpy`` + ``tf2_ros``); it is not exercised by the core
test suite.

    # bag mode (classification only, no ROS needed)
    python3 tools/runtime/check_sensor_bag_ready.py --bag <bag_dir>

    # bag mode + deep timestamp/timing diagnostics (needs sourced ROS)
    python3 tools/runtime/check_sensor_bag_ready.py --bag <bag_dir> --deep

    # live mode (needs a sourced ROS environment against a running graph)
    python3 tools/runtime/check_sensor_bag_ready.py --live
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------
# Canonical topic contract (see Phase 0/1/2 audit in
# docs/perception/real_sensor_bag_capture_v1.md - grounded directly in
# ad_lidar_perception/launch/lidar_bag_replay.launch.py's SOURCE_TOPICS /
# LOCALIZATION_SOURCE_TOPICS and ad_localization/src/adapter/
# localization_node.cpp's declared subscription types).
# --------------------------------------------------------------------------

LIDAR_TOPIC = "/ad/sensors/lidar/points"
LIDAR_TYPE = "sensor_msgs/msg/PointCloud2"

CAMERA_TOPIC = "/ad/sensors/camera/front/compressed"
CAMERA_TYPE = "sensor_msgs/msg/CompressedImage"

ODOMETRY_TOPIC = "/ad/localization/odometry"
ODOMETRY_TYPE = "nav_msgs/msg/Odometry"

TF_TOPIC = "/tf"
TF_STATIC_TOPIC = "/tf_static"
TF_TYPE = "tf2_msgs/msg/TFMessage"

GPS_TOPIC = "/ad/sensors/gps/fix"
GPS_TYPE = "sensor_msgs/msg/NavSatFix"

IMU_TOPIC = "/ad/sensors/imu/data"
IMU_TYPE = "sensor_msgs/msg/Imu"

# ad_localization's adapter (localization_node.cpp) subscribes this exact
# MORAI-defined message type unconditionally, regardless of backend. A real
# (non-MORAI) vehicle needs a bridge that publishes this same message type
# on this topic for enable_localization:=true to work as validated; there is
# no generic non-MORAI vehicle-status type in this repository today.
VEHICLE_STATUS_TOPIC = "/ad/vehicle/status"
VEHICLE_STATUS_TYPE = "ad_morai_interfaces/msg/EgoVehicleStatus"

CANONICAL_RECORD_TOPICS = (
    LIDAR_TOPIC,
    CAMERA_TOPIC,
    GPS_TOPIC,
    IMU_TOPIC,
    VEHICLE_STATUS_TOPIC,
    ODOMETRY_TOPIC,
    TF_TOPIC,
    TF_STATIC_TOPIC,
)

# --------------------------------------------------------------------------
# Readiness classification
# --------------------------------------------------------------------------

STATUS_BROKEN = "BROKEN"
STATUS_MISSING_LIDAR = "MISSING_LIDAR"
STATUS_MISSING_TF = "MISSING_TF"
STATUS_MISSING_LOCALIZATION = "MISSING_LOCALIZATION"
STATUS_PARTIAL_SENSOR_ONLY = "PARTIAL_SENSOR_ONLY"
STATUS_READY_REPLAY_WITH_LOCALIZATION_RECOMPUTE = (
    "READY_REPLAY_WITH_LOCALIZATION_RECOMPUTE"
)
STATUS_READY_FULL_REPLAY = "READY_FULL_REPLAY"

_ALL_STATUSES = (
    STATUS_BROKEN,
    STATUS_MISSING_LIDAR,
    STATUS_MISSING_TF,
    STATUS_MISSING_LOCALIZATION,
    STATUS_PARTIAL_SENSOR_ONLY,
    STATUS_READY_REPLAY_WITH_LOCALIZATION_RECOMPUTE,
    STATUS_READY_FULL_REPLAY,
)


@dataclass
class TopicInfo:
    present: bool = False
    type: str | None = None
    message_count: int = 0
    type_matches_expected: bool | None = None


@dataclass
class BagReadiness:
    bag_path: str
    status: str
    reasons: list[str] = field(default_factory=list)
    duration_sec: float | None = None
    topics: dict[str, TopicInfo] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bag_path": self.bag_path,
            "status": self.status,
            "reasons": list(self.reasons),
            "duration_sec": self.duration_sec,
            "topics": {
                name: {
                    "present": info.present,
                    "type": info.type,
                    "message_count": info.message_count,
                    "type_matches_expected": info.type_matches_expected,
                }
                for name, info in sorted(self.topics.items())
            },
        }


class BagMetadataError(RuntimeError):
    """Raised for a structurally unusable bag (never for a merely-partial one)."""


def _topic_info(
    topics_with_count: list[dict[str, Any]], topic_name: str, expected_type: str | None
) -> TopicInfo:
    for entry in topics_with_count:
        metadata = entry.get("topic_metadata", {})
        if metadata.get("name") == topic_name:
            actual_type = metadata.get("type")
            return TopicInfo(
                present=True,
                type=actual_type,
                message_count=int(entry.get("message_count", 0)),
                type_matches_expected=(
                    None if expected_type is None else actual_type == expected_type
                ),
            )
    return TopicInfo(present=False, type=None, message_count=0, type_matches_expected=None)


def _has_messages(info: TopicInfo) -> bool:
    return info.present and info.message_count > 0


def classify_bag_metadata(bag_path: str | Path) -> BagReadiness:
    """Classify a bag from its ``metadata.yaml`` alone.

    Pure-Python: no ROS import, no message deserialization, no bag
    rewriting. Detects a metadata-only ("BROKEN") bag by checking that
    every ``relative_file_paths`` entry actually exists on disk - this is
    exactly the failure mode found for ``bags/static_20260805_003151`` in
    an earlier session (a real, non-fabricated metadata.yaml with no
    underlying .mcap file).
    """
    root = Path(bag_path).expanduser()
    reasons: list[str] = []
    metadata_path = root / "metadata.yaml"

    if not root.is_dir():
        return BagReadiness(
            bag_path=str(root),
            status=STATUS_BROKEN,
            reasons=[f"bag_path is not a directory: {root}"],
        )
    if not metadata_path.is_file():
        return BagReadiness(
            bag_path=str(root),
            status=STATUS_BROKEN,
            reasons=[f"missing metadata.yaml: {metadata_path}"],
        )
    try:
        document = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        info = document["rosbag2_bagfile_information"]
    except (OSError, TypeError, KeyError, yaml.YAMLError) as error:
        return BagReadiness(
            bag_path=str(root),
            status=STATUS_BROKEN,
            reasons=[f"invalid metadata.yaml: {error}"],
        )
    if not isinstance(info, dict):
        return BagReadiness(
            bag_path=str(root),
            status=STATUS_BROKEN,
            reasons=["metadata.yaml: rosbag2_bagfile_information must be a mapping"],
        )

    relative_paths = info.get("relative_file_paths") or []
    missing_files = [
        path for path in relative_paths if not (root / path).is_file()
    ]
    if not relative_paths:
        missing_files = ["<no relative_file_paths declared>"]
    if missing_files:
        return BagReadiness(
            bag_path=str(root),
            status=STATUS_BROKEN,
            reasons=[
                "metadata.yaml declares a storage file that does not exist "
                f"on disk (metadata-only bag, unusable): {missing_files}"
            ],
        )

    topics_with_count = info.get("topics_with_message_count") or []
    duration_ns = (info.get("duration") or {}).get("nanoseconds")
    duration_sec = duration_ns / 1e9 if isinstance(duration_ns, (int, float)) else None

    lidar = _topic_info(topics_with_count, LIDAR_TOPIC, LIDAR_TYPE)
    camera = _topic_info(topics_with_count, CAMERA_TOPIC, CAMERA_TYPE)
    odometry = _topic_info(topics_with_count, ODOMETRY_TOPIC, ODOMETRY_TYPE)
    tf_dynamic = _topic_info(topics_with_count, TF_TOPIC, TF_TYPE)
    tf_static = _topic_info(topics_with_count, TF_STATIC_TOPIC, TF_TYPE)
    gps = _topic_info(topics_with_count, GPS_TOPIC, GPS_TYPE)
    imu = _topic_info(topics_with_count, IMU_TOPIC, IMU_TYPE)
    status_topic = _topic_info(
        topics_with_count, VEHICLE_STATUS_TOPIC, VEHICLE_STATUS_TYPE
    )

    topics = {
        LIDAR_TOPIC: lidar,
        CAMERA_TOPIC: camera,
        ODOMETRY_TOPIC: odometry,
        TF_TOPIC: tf_dynamic,
        TF_STATIC_TOPIC: tf_static,
        GPS_TOPIC: gps,
        IMU_TOPIC: imu,
        VEHICLE_STATUS_TOPIC: status_topic,
    }

    for name, expected_type, info_obj in (
        (LIDAR_TOPIC, LIDAR_TYPE, lidar),
        (CAMERA_TOPIC, CAMERA_TYPE, camera),
        (ODOMETRY_TOPIC, ODOMETRY_TYPE, odometry),
        (TF_TOPIC, TF_TYPE, tf_dynamic),
        (TF_STATIC_TOPIC, TF_TYPE, tf_static),
        (GPS_TOPIC, GPS_TYPE, gps),
        (IMU_TOPIC, IMU_TYPE, imu),
        (VEHICLE_STATUS_TOPIC, VEHICLE_STATUS_TYPE, status_topic),
    ):
        if info_obj.present and info_obj.type_matches_expected is False:
            reasons.append(
                f"{name}: recorded type {info_obj.type!r} does not match "
                f"expected {expected_type!r}"
            )

    if not _has_messages(lidar):
        status = STATUS_MISSING_LIDAR
        reasons.append(f"{LIDAR_TOPIC} absent or has 0 messages")
    elif _has_messages(odometry) and _has_messages(tf_dynamic):
        status = STATUS_READY_FULL_REPLAY
        reasons.append(
            f"{ODOMETRY_TOPIC} and {TF_TOPIC} both recorded with real messages"
        )
    elif _has_messages(odometry) and not _has_messages(tf_dynamic):
        status = STATUS_MISSING_TF
        reasons.append(
            f"{ODOMETRY_TOPIC} recorded but {TF_TOPIC} absent/empty - "
            "cannot do exact-stamp TF lookups from this bag alone"
        )
    elif _has_messages(gps) and _has_messages(imu) and _has_messages(status_topic):
        status = STATUS_READY_REPLAY_WITH_LOCALIZATION_RECOMPUTE
        reasons.append(
            f"{GPS_TOPIC} + {IMU_TOPIC} + {VEHICLE_STATUS_TOPIC} all recorded - "
            "ad_localization's gnss_imu backend can recompute odometry/TF "
            "(enable_localization:=true)"
        )
    elif _has_messages(gps) or _has_messages(imu) or _has_messages(status_topic):
        status = STATUS_PARTIAL_SENSOR_ONLY
        present = [
            name
            for name, info_obj in (
                (GPS_TOPIC, gps),
                (IMU_TOPIC, imu),
                (VEHICLE_STATUS_TOPIC, status_topic),
            )
            if _has_messages(info_obj)
        ]
        reasons.append(
            "incomplete raw ego-sensor set for localization recompute - "
            f"present: {present}, need all of gps+imu+vehicle_status"
        )
    else:
        status = STATUS_MISSING_LOCALIZATION
        reasons.append(
            "no odometry, no dynamic TF, and no raw ego sensors (gps/imu/"
            "vehicle_status) - LiDAR-only (raw sensor-domain use only, "
            "tracking replay is not possible from this bag alone)"
        )

    if status in (
        STATUS_READY_FULL_REPLAY,
        STATUS_READY_REPLAY_WITH_LOCALIZATION_RECOMPUTE,
    ) and not _has_messages(camera):
        reasons.append(f"{CAMERA_TOPIC} absent - camera panel will stay blank")

    return BagReadiness(
        bag_path=str(root),
        status=status,
        reasons=reasons,
        duration_sec=duration_sec,
        topics=topics,
    )


# --------------------------------------------------------------------------
# Optional deeper diagnostics (needs a sourced ROS environment: rosbag2_py
# + the message packages listed above - no RViz, no GPU).
# --------------------------------------------------------------------------


@dataclass
class TimestampDiagnostics:
    topic: str
    count: int = 0
    nonzero: bool = True
    monotonic: bool = True
    duplicate_count: int = 0
    backward_count: int = 0


@dataclass
class NearestStampOffsetStats:
    reference_topic: str
    target_topic: str
    pairs: int = 0
    median_ms: float | None = None
    p95_ms: float | None = None
    max_ms: float | None = None


def _read_stamps_ns(bag_path: Path, topic: str, type_name: str) -> list[int]:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    storage_options = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    message_type = get_message(type_name)

    stamps: list[int] = []
    while reader.has_next():
        current_topic, data, _receive_time = reader.read_next()
        if current_topic != topic:
            continue
        message = deserialize_message(data, message_type)
        header = getattr(message, "header", None)
        if header is None:
            continue
        stamp = header.stamp
        stamps.append(int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec))
    return stamps


def timestamp_diagnostics(bag_path: str | Path, topic: str, type_name: str) -> TimestampDiagnostics:
    """Sample header stamps for one topic: nonzero / monotonic / duplicate / backward.

    Needs a sourced ROS environment (``rosbag2_py`` + the message package
    for ``type_name``). Read-only; the bag is never modified.
    """
    stamps = _read_stamps_ns(Path(bag_path), topic, type_name)
    diag = TimestampDiagnostics(topic=topic, count=len(stamps))
    if not stamps:
        return diag
    diag.nonzero = all(stamp > 0 for stamp in stamps)
    previous = stamps[0]
    for stamp in stamps[1:]:
        if stamp == previous:
            diag.duplicate_count += 1
        elif stamp < previous:
            diag.backward_count += 1
            diag.monotonic = False
        previous = stamp
    return diag


def nearest_stamp_offset_stats(
    bag_path: str | Path,
    reference_topic: str,
    reference_type: str,
    target_topic: str,
    target_type: str,
) -> NearestStampOffsetStats:
    """Nearest-stamp |dt| between two topics' header stamps (timing only).

    This characterizes recorded timing; it is not a hardware-synchronization
    claim unless independently proven for the specific sensors involved.
    """
    import bisect

    reference_stamps = sorted(_read_stamps_ns(Path(bag_path), reference_topic, reference_type))
    target_stamps = sorted(_read_stamps_ns(Path(bag_path), target_topic, target_type))
    result = NearestStampOffsetStats(
        reference_topic=reference_topic, target_topic=target_topic
    )
    if not reference_stamps or not target_stamps:
        return result

    diffs_ns = []
    for stamp in reference_stamps:
        index = bisect.bisect_left(target_stamps, stamp)
        best = None
        for candidate_index in (index - 1, index):
            if 0 <= candidate_index < len(target_stamps):
                diff = abs(target_stamps[candidate_index] - stamp)
                if best is None or diff < best:
                    best = diff
        if best is not None:
            diffs_ns.append(best)

    if not diffs_ns:
        return result
    diffs_ns.sort()
    count = len(diffs_ns)

    def _percentile_ms(fraction: float) -> float:
        index = min(count - 1, int(fraction * count))
        return diffs_ns[index] / 1e6

    result.pairs = count
    result.median_ms = _percentile_ms(0.5)
    result.p95_ms = _percentile_ms(0.95)
    result.max_ms = diffs_ns[-1] / 1e6
    return result


# --------------------------------------------------------------------------
# LIVE mode (needs rclpy + tf2_ros against a running ROS graph)
# --------------------------------------------------------------------------


def check_live_topics(timeout_sec: float = 5.0) -> dict[str, Any]:
    """Check the currently-running ROS graph for the canonical topic set
    and the odom->base_link->rear_axle_link->lidar_link TF chain.

    Needs a sourced ROS environment and a running graph; not exercised by
    the core (metadata-only) test suite. Read-only: subscribes only, never
    publishes.
    """
    import rclpy
    from rclpy.node import Node
    from tf2_ros import Buffer, TransformListener

    rclpy.init(args=None)
    node = Node("check_sensor_bag_ready_live_probe")
    try:
        buffer = Buffer()
        listener = TransformListener(buffer, node)
        assert listener is not None  # keeps the listener alive for this scope
        deadline = node.get_clock().now().nanoseconds + int(timeout_sec * 1e9)
        chain_results = {}
        chains = [
            ("odom", "base_link"),
            ("base_link", "rear_axle_link"),
            ("rear_axle_link", "lidar_link"),
            ("rear_axle_link", "camera_front_optical_frame"),
        ]
        # DDS discovery of currently-advertised topics is not instantaneous
        # after node creation, so topic presence is re-queried on every spin
        # (not just once at the start) and the loop always runs at least
        # once even if the TF chain resolves immediately.
        topic_types: dict[str, Any] = {}
        while node.get_clock().now().nanoseconds < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
            topic_types = dict(node.get_topic_names_and_types())
            for parent, child in chains:
                key = f"{parent}->{child}"
                if key in chain_results:
                    continue
                if buffer.can_transform(parent, child, rclpy.time.Time()):
                    chain_results[key] = True
            if len(chain_results) == len(chains):
                break
        for parent, child in chains:
            key = f"{parent}->{child}"
            chain_results.setdefault(key, False)

        result: dict[str, Any] = {"topics": {}}
        for name in CANONICAL_RECORD_TOPICS:
            types = topic_types.get(name)
            result["topics"][name] = {
                "present": types is not None,
                "types": types or [],
            }
        result["tf_chain"] = chain_results
        return result
    finally:
        node.destroy_node()
        rclpy.shutdown()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bag", type=str, help="Bag directory containing metadata.yaml")
    mode.add_argument("--live", action="store_true", help="Check a running ROS graph")
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Also run message-content diagnostics (needs sourced ROS)",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON only"
    )
    args = parser.parse_args(argv)

    if args.live:
        result = check_live_topics()
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("LIVE topic check:")
            for name, info in result["topics"].items():
                print(f"  {name}: present={info['present']} types={info['types']}")
            print("TF chain:")
            for key, ok in result["tf_chain"].items():
                print(f"  {key}: {'OK' if ok else 'MISSING'}")
        return 0

    readiness = classify_bag_metadata(args.bag)
    output: dict[str, Any] = readiness.to_dict()

    if args.deep and readiness.status != STATUS_BROKEN:
        deep: dict[str, Any] = {}
        lidar_diag = timestamp_diagnostics(args.bag, LIDAR_TOPIC, LIDAR_TYPE)
        deep["lidar_timestamps"] = vars(lidar_diag)
        if readiness.topics[CAMERA_TOPIC].present:
            camera_lidar = nearest_stamp_offset_stats(
                args.bag, LIDAR_TOPIC, LIDAR_TYPE, CAMERA_TOPIC, CAMERA_TYPE
            )
            deep["camera_lidar_offset_ms"] = vars(camera_lidar)
        if readiness.topics[ODOMETRY_TOPIC].present:
            odom_lidar = nearest_stamp_offset_stats(
                args.bag, LIDAR_TOPIC, LIDAR_TYPE, ODOMETRY_TOPIC, ODOMETRY_TYPE
            )
            deep["odometry_lidar_offset_ms"] = vars(odom_lidar)
        output["deep"] = deep

    if args.json:
        print(json.dumps(output, indent=2))
    else:
        print(f"bag_path: {readiness.bag_path}")
        print(f"status: {readiness.status}")
        print(f"duration_sec: {readiness.duration_sec}")
        print("reasons:")
        for reason in readiness.reasons:
            print(f"  - {reason}")
        print("topics:")
        for name, info in sorted(readiness.topics.items()):
            print(
                f"  {name}: present={info.present} type={info.type} "
                f"count={info.message_count} "
                f"type_ok={info.type_matches_expected}"
            )
        if "deep" in output:
            print("deep diagnostics:")
            print(json.dumps(output["deep"], indent=2))

    return 1 if readiness.status == STATUS_BROKEN else 0


if __name__ == "__main__":
    sys.exit(main())
