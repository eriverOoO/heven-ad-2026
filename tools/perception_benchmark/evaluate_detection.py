#!/usr/bin/env python3
"""Evaluate a detector output recorded in a ROS 2 bag without running it."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from actor_gt import TransformHistory, build_dynamic_indices, edge_from_message
from frame_alignment import NearestIndex, TimedSample, stamp_ns
from metrics import distance_bin, greedy_center_matches, percentile


@dataclass(frozen=True)
class DetectionFrame:
    frame_id: str
    centers: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class ActorFrame:
    frame_id: str
    centers: tuple[tuple[float, float, float], ...]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/perception"))
    parser.add_argument("--experiment", help="override config experiment name")
    return parser.parse_args()


def _read_bag(bag: Path, config: dict[str, Any]) -> dict[str, Any]:
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as error:
        raise RuntimeError(
            "ROS 2 Python packages are unavailable; source /opt/ros/humble/setup.bash "
            "and the workspace install/setup.bash"
        ) from error

    topics = config["topics"]
    selected = set(topics.values())
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    available_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    missing = selected - available_types.keys()
    if missing:
        raise RuntimeError(f"bag is missing configured topics: {sorted(missing)}")
    for key, expected in config.get("expected_types", {}).items():
        actual = available_types.get(topics[key])
        if actual != expected:
            raise RuntimeError(f"{topics[key]} type is {actual!r}, expected {expected!r}")
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(selected)))
    message_types = {topic: get_message(available_types[topic]) for topic in selected}

    source_times: dict[int, int] = {}
    detections: list[TimedSample[DetectionFrame]] = []
    actors: list[TimedSample[ActorFrame]] = []
    dynamic_transforms = []
    static_transforms = []
    while reader.has_next():
        topic, serialized, record_ns = reader.read_next()
        message = deserialize_message(serialized, message_types[topic])
        if topic == topics["source"]:
            source_ns = stamp_ns(message.header.stamp)
            source_times[source_ns] = min(record_ns, source_times.get(source_ns, record_ns))
        elif topic == topics["detections"]:
            centers = tuple(
                (
                    float(item.kinematics.pose_with_covariance.pose.position.x),
                    float(item.kinematics.pose_with_covariance.pose.position.y),
                )
                for item in message.objects
            )
            detections.append(
                TimedSample(
                    stamp_ns(message.header.stamp),
                    record_ns,
                    DetectionFrame(message.header.frame_id, centers),
                )
            )
        elif topic == topics["actor_gt"]:
            actors.append(
                TimedSample(
                    stamp_ns(message.header.stamp),
                    record_ns,
                    ActorFrame(
                        message.header.frame_id,
                        tuple(
                            (
                                float(item.position.x),
                                float(item.position.y),
                                float(item.position.z),
                            )
                            for item in message.objects
                        ),
                    ),
                )
            )
        elif topic == topics["tf"]:
            for transform in message.transforms:
                dynamic_transforms.append(
                    TimedSample(
                        stamp_ns(transform.header.stamp),
                        record_ns,
                        edge_from_message(transform),
                    )
                )
        elif topic == topics["tf_static"]:
            static_transforms.extend(edge_from_message(item) for item in message.transforms)
    return {
        "source_times": source_times,
        "detections": detections,
        "actors": actors,
        "dynamic_transforms": dynamic_transforms,
        "static_transforms": static_transforms,
        "topic_types": {topic: available_types[topic] for topic in selected},
    }


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return 100.0 * numerator / denominator if denominator else None


def _evaluate(data: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    actor_index = NearestIndex(data["actors"])
    alignment = config["alignment"]
    actor_max_ns = round(float(alignment["actor_max_delta_ms"]) * 1_000_000)
    tf_max_ns = round(float(alignment["tf_max_delta_ms"]) * 1_000_000)
    transform_history = TransformHistory(
        data["static_transforms"],
        build_dynamic_indices(data["dynamic_transforms"]),
        [tuple(pair) for pair in alignment.get("identity_frame_aliases", [])],
    )
    roi = config["evaluation"]["roi"]
    thresholds = [float(value) for value in config["evaluation"]["center_thresholds_m"]]
    primary = float(config["evaluation"]["primary_threshold_m"])
    edges = [float(value) for value in config["evaluation"]["distance_bin_edges_m"]]
    bin_names = [
        distance_bin(0.0 if index == 0 else edges[index - 1], edges)
        for index in range(len(edges) + 1)
    ]
    totals = {threshold: 0 for threshold in thresholds}
    bin_gt = {name: 0 for name in bin_names}
    bin_matches = {name: 0 for name in bin_names}
    primary_distances: list[float] = []
    latencies_ms: list[float] = []
    exact_source_pairs = 0
    negative_latency_samples = 0
    actor_deltas_ms: list[float] = []
    compared_frames = 0
    total_detections = 0
    total_gt = 0
    skipped_source = 0
    skipped_actor = 0
    skipped_tf = 0

    for detection in data["detections"]:
        source_record_ns = data["source_times"].get(detection.source_ns)
        if source_record_ns is None:
            skipped_source += 1
            continue
        exact_source_pairs += 1
        if detection.record_ns >= source_record_ns:
            latencies_ms.append(
                (detection.record_ns - source_record_ns) / 1_000_000.0
            )
        else:
            negative_latency_samples += 1
        actor = actor_index.nearest(detection.source_ns, actor_max_ns)
        if actor is None:
            skipped_actor += 1
            continue
        transformed = []
        transform_failed = False
        for point in actor.value.centers:
            value = transform_history.transform_point(
                point,
                actor.value.frame_id,
                detection.value.frame_id,
                detection.source_ns,
                tf_max_ns,
            )
            if value is None:
                transform_failed = True
                break
            if roi["x"][0] <= value[0] <= roi["x"][1] and roi["y"][0] <= value[1] <= roi["y"][1]:
                transformed.append((value[0], value[1]))
        if transform_failed:
            skipped_tf += 1
            continue
        compared_frames += 1
        actor_deltas_ms.append(abs(actor.source_ns - detection.source_ns) / 1_000_000.0)
        total_detections += len(detection.value.centers)
        total_gt += len(transformed)
        actor_bins = [distance_bin(math.hypot(*point), edges) for point in transformed]
        for name in actor_bins:
            bin_gt[name] += 1
        for threshold in thresholds:
            matches = greedy_center_matches(detection.value.centers, transformed, threshold)
            totals[threshold] += len(matches)
            if threshold == primary:
                primary_distances.extend(match[2] for match in matches)
                for _, actor_position, _ in matches:
                    bin_matches[actor_bins[actor_position]] += 1

    result: dict[str, Any] = {
        "frames_compared": compared_frames,
        "source_frames": len(data["source_times"]),
        "detection_frames": len(data["detections"]),
        "frames_skipped_source_alignment": skipped_source,
        "frames_skipped_actor_alignment": skipped_actor,
        "frames_skipped_tf": skipped_tf,
        "detections_per_frame": total_detections / compared_frames if compared_frames else None,
        "total_detections": total_detections,
        "actor_frame_gt_count": total_gt,
        "actor_match_count": totals[primary],
        "recall_proxy_pct": _safe_ratio(totals[primary], total_gt),
        "matched_center_distance_mean_m": (
            sum(primary_distances) / len(primary_distances)
            if primary_distances
            else None
        ),
        "matched_center_distance_p50_m": percentile(primary_distances, 50),
        "matched_center_distance_p95_m": percentile(primary_distances, 95),
        "exact_source_pairs": exact_source_pairs,
        "latency_samples": len(latencies_ms),
        "negative_latency_samples_excluded": negative_latency_samples,
        "latency_p50_ms": percentile(latencies_ms, 50),
        "latency_p95_ms": percentile(latencies_ms, 95),
        "actor_alignment_delta_p95_ms": percentile(actor_deltas_ms, 95),
    }
    for threshold in thresholds:
        suffix = str(threshold).replace(".", "_")
        result[f"matches_{suffix}m"] = totals[threshold]
        result[f"recall_{suffix}m_pct"] = _safe_ratio(totals[threshold], total_gt)
    for name in bin_names:
        suffix = name.replace("-", "_").replace("+", "_plus").replace("m", "m")
        result[f"gt_{suffix}"] = bin_gt[name]
        result[f"matches_{suffix}"] = bin_matches[name]
        result[f"recall_{suffix}_pct"] = _safe_ratio(bin_matches[name], bin_gt[name])
    return result


def _git_metadata(repository: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True, text=True, capture_output=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    )
    return commit, dirty


def _format(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_outputs(
    output_dir: Path,
    experiment: str,
    row: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{experiment}.csv"
    markdown_path = output_dir / f"{experiment}.md"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    bin_rows = []
    for name in metadata["distance_bins"]:
        suffix = name.replace("-", "_").replace("+", "_plus")
        recall = _format(row[f"recall_{suffix}_pct"], 2)
        bin_rows.append(
            f"| {name} | {row[f'gt_{suffix}']} | "
            f"{row[f'matches_{suffix}']} | {recall} |"
        )
    threshold_rows = []
    for threshold in metadata["thresholds_m"]:
        suffix = str(float(threshold)).replace(".", "_")
        recall = _format(row[f"recall_{suffix}m_pct"], 2)
        threshold_rows.append(
            f"| {threshold:g} | {row[f'matches_{suffix}m']} | {recall} |"
        )
    content = f"""# Detection benchmark: {experiment}

## Summary

| Metric | Value |
|---|---:|
| Frames compared | {row['frames_compared']} |
| Recorded source frames | {row['source_frames']} |
| Recorded detection frames | {row['detection_frames']} |
| Detections/frame | {_format(row['detections_per_frame'])} |
| Total detections | {row['total_detections']} |
| Actor-frame GT count | {row['actor_frame_gt_count']} |
| Actor matches ({metadata['primary_threshold_m']:g} m) | {row['actor_match_count']} |
| Recall proxy | {_format(row['recall_proxy_pct'], 2)}% |
| Matched center distance mean / p50 / p95 | {_format(row['matched_center_distance_mean_m'])} / {_format(row['matched_center_distance_p50_m'])} / {_format(row['matched_center_distance_p95_m'])} m |
| Detection latency p50 / p95 | {_format(row['latency_p50_ms'])} / {_format(row['latency_p95_ms'])} ms |

## Threshold sensitivity

| Center threshold (m) | Matches | Recall proxy (%) |
|---:|---:|---:|
{os.linesep.join(threshold_rows)}

## Distance-binned recall

| Actor center range | Actor-frame GT | Matches | Recall proxy (%) |
|---|---:|---:|---:|
{os.linesep.join(bin_rows)}

## Alignment and interpretation

- Detection-to-source alignment: exact source `header.stamp` equality. Frames without a recorded exact-stamp source are excluded from every comparison metric.
- Actor GT alignment: nearest source `header.stamp`, maximum {metadata['actor_max_delta_ms']:g} ms; observed p95 delta {_format(row['actor_alignment_delta_p95_ms'])} ms.
- TF alignment: nearest transform source `header.stamp`, maximum {metadata['tf_max_delta_ms']:g} ms; static TF is timeless.
- Identity frame aliases from config: `{metadata['identity_frame_aliases']}`.
- Latency is the MCAP recorder receive-time difference for source and detection messages carrying the same source stamp. It is an end-to-end pipeline proxy, not isolated detector CPU time. Exact source pairs: {row['exact_source_pairs']}; usable samples: {row['latency_samples']}; negative recorder-order samples excluded: {row['negative_latency_samples_excluded']}.
- Recall is a center-distance proxy. It does not model occlusion or visibility, and MORAI actor GT omits useful static-obstacle detections.
- Frames skipped for missing exact-stamp source: {row['frames_skipped_source_alignment']}; actor alignment: {row['frames_skipped_actor_alignment']}; TF: {row['frames_skipped_tf']}.

## Reproducibility metadata

```json
{json.dumps(metadata, indent=2, sort_keys=True)}
```
"""
    markdown_path.write_text(content, encoding="utf-8")
    return csv_path, markdown_path


def main() -> int:
    arguments = parse_arguments()
    config_bytes = arguments.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    experiment = arguments.experiment or config["experiment"]
    repository = Path(__file__).resolve().parents[2]
    commit, dirty = _git_metadata(repository)
    data = _read_bag(arguments.bag.resolve(), config)
    metrics = _evaluate(data, config)
    bag_files = sorted(path.name for path in arguments.bag.resolve().glob("*.mcap"))
    edges = [float(value) for value in config["evaluation"]["distance_bin_edges_m"]]
    distance_bins = [
        distance_bin(0.0 if index == 0 else edges[index - 1], edges)
        for index in range(len(edges) + 1)
    ]
    metadata = {
        "algorithm": config["algorithm"],
        "bag": str(arguments.bag.resolve()),
        "bag_files": bag_files,
        "config": str(arguments.config.resolve()),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "distance_bins": distance_bins,
        "thresholds_m": config["evaluation"]["center_thresholds_m"],
        "primary_threshold_m": float(config["evaluation"]["primary_threshold_m"]),
        "actor_max_delta_ms": float(config["alignment"]["actor_max_delta_ms"]),
        "tf_max_delta_ms": float(config["alignment"]["tf_max_delta_ms"]),
        "identity_frame_aliases": config["alignment"].get("identity_frame_aliases", []),
        "git_commit": commit,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "ros_distro": os.environ.get("ROS_DISTRO", "unknown"),
        "topic_types": data["topic_types"],
    }
    row = {
        "experiment": experiment,
        "algorithm": config["algorithm"],
        "git_commit": commit,
        "git_dirty": dirty,
        "bag": str(arguments.bag.resolve()),
        "bag_files": ";".join(bag_files),
        "config": str(arguments.config.resolve()),
        "config_sha256": metadata["config_sha256"],
        "python": metadata["python"],
        "platform": metadata["platform"],
        "ros_distro": metadata["ros_distro"],
        **metrics,
    }
    csv_path, markdown_path = _write_outputs(arguments.output_dir, experiment, row, metadata)
    print(csv_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
