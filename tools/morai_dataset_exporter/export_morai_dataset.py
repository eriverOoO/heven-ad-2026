#!/usr/bin/env python3
"""Export timestamp-aligned MORAI LiDAR frames and actor 3D boxes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Iterable, TypeVar

import numpy as np
import yaml

from geometry import (
    RigidTransform,
    normalize_angle,
    quaternion_from_rpy,
    transform_from_message,
)


T = TypeVar("T")


def stamp_ns(stamp: object) -> int:
    return int(getattr(stamp, "sec")) * 1_000_000_000 + int(
        getattr(stamp, "nanosec")
    )


@dataclass(frozen=True)
class TimedSample(Generic[T]):
    source_ns: int
    record_ns: int
    value: T


class NearestIndex(Generic[T]):
    def __init__(self, samples: Iterable[TimedSample[T]]) -> None:
        self.samples = sorted(samples, key=lambda sample: sample.source_ns)
        self.stamps = [sample.source_ns for sample in self.samples]

    def nearest(self, source_ns: int, maximum_delta_ns: int) -> TimedSample[T] | None:
        position = bisect_left(self.stamps, source_ns)
        candidates = []
        if position < len(self.samples):
            candidates.append(self.samples[position])
        if position:
            candidates.append(self.samples[position - 1])
        if not candidates:
            return None
        result = min(
            candidates,
            key=lambda sample: (abs(sample.source_ns - source_ns), sample.source_ns),
        )
        return (
            result
            if abs(result.source_ns - source_ns) <= maximum_delta_ns
            else None
        )


@dataclass(frozen=True)
class Actor:
    unique_id: int
    object_type: int
    position: tuple[float, float, float]
    heading: float
    size: tuple[float, float, float]
    overhang: float
    wheelbase: float
    rear_overhang: float


@dataclass(frozen=True)
class ActorFrame:
    frame_id: str
    actors: tuple[Actor, ...]


@dataclass(frozen=True)
class EgoPose:
    frame_id: str
    pose: RigidTransform


@dataclass(frozen=True)
class DynamicTransform:
    parent: str
    child: str
    transform: RigidTransform


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", action="append", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing exporter-owned dataset directory",
    )
    return parser.parse_args()


def _message_support() -> tuple[Any, Any, Any, Any]:
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as error:
        raise RuntimeError(
            "source ROS 2 Humble and the workspace install/setup.bash first"
        ) from error
    return rosbag2_py, deserialize_message, get_message, rosbag2_py.StorageFilter


def _reader(bag: Path, selected_topics: Iterable[str]) -> tuple[Any, dict[str, str]]:
    rosbag2_py, _, _, storage_filter = _message_support()
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    reader.set_filter(storage_filter(topics=list(selected_topics)))
    return reader, topic_types


def _verify_topics(
    topic_types: dict[str, str], config: dict[str, Any]
) -> dict[str, str]:
    topics = config["topics"]
    missing = set(topics.values()) - topic_types.keys()
    if missing:
        raise RuntimeError(f"bag is missing configured topics: {sorted(missing)}")
    for key, expected in config["expected_types"].items():
        actual = topic_types[topics[key]]
        if actual != expected:
            raise RuntimeError(f"{topics[key]} type is {actual!r}, expected {expected!r}")
    return {topic: topic_types[topic] for topic in topics.values()}


def _read_alignment_data(bag: Path, config: dict[str, Any]) -> dict[str, Any]:
    _, deserialize_message, get_message, _ = _message_support()
    topics = config["topics"]
    selected = [
        topics["actor_gt"],
        topics["ego_status"],
        topics["tf"],
        topics["tf_static"],
    ]
    reader, topic_types = _reader(bag, selected)
    verified_types = _verify_topics(topic_types, config)
    types = {topic: get_message(topic_types[topic]) for topic in selected}
    actors: list[TimedSample[ActorFrame]] = []
    egos: list[TimedSample[EgoPose]] = []
    transforms: dict[tuple[str, str], list[TimedSample[DynamicTransform]]] = {}
    static: dict[tuple[str, str], DynamicTransform] = {}
    while reader.has_next():
        topic, serialized, record_ns = reader.read_next()
        message = deserialize_message(serialized, types[topic])
        if topic == topics["actor_gt"]:
            actors.append(
                TimedSample(
                    stamp_ns(message.header.stamp),
                    record_ns,
                    ActorFrame(
                        str(message.header.frame_id).lstrip("/"),
                        tuple(
                            Actor(
                                int(item.unique_id),
                                int(item.object_type),
                                (
                                    float(item.position.x),
                                    float(item.position.y),
                                    float(item.position.z),
                                ),
                                float(item.heading),
                                (
                                    float(item.size.x),
                                    float(item.size.y),
                                    float(item.size.z),
                                ),
                                float(item.overhang),
                                float(item.wheelbase),
                                float(item.rear_overhang),
                            )
                            for item in message.objects
                        ),
                    ),
                )
            )
        elif topic == topics["ego_status"]:
            egos.append(
                TimedSample(
                    stamp_ns(message.header.stamp),
                    record_ns,
                    EgoPose(
                        str(message.header.frame_id).lstrip("/"),
                        RigidTransform(
                            (
                                float(message.position.x),
                                float(message.position.y),
                                float(message.position.z),
                            ),
                            quaternion_from_rpy(
                                float(message.rpy.x),
                                float(message.rpy.y),
                                float(message.rpy.z),
                            ),
                        ),
                    ),
                )
            )
        elif topic in (topics["tf"], topics["tf_static"]):
            for item in message.transforms:
                value = DynamicTransform(
                    str(item.header.frame_id).lstrip("/"),
                    str(item.child_frame_id).lstrip("/"),
                    transform_from_message(item),
                )
                key = (value.parent, value.child)
                if topic == topics["tf_static"]:
                    static[key] = value
                else:
                    transforms.setdefault(key, []).append(
                        TimedSample(stamp_ns(item.header.stamp), record_ns, value)
                    )
    return {
        "actor_index": NearestIndex(actors),
        "ego_index": NearestIndex(egos),
        "tf_indices": {key: NearestIndex(values) for key, values in transforms.items()},
        "static": static,
        "topic_types": verified_types,
    }


def _point_array(message: object) -> tuple[np.ndarray, int, int]:
    if bool(message.is_bigendian):
        raise ValueError("big-endian PointCloud2 is unsupported")
    fields = {field.name: field for field in message.fields}
    required = ("x", "y", "z", "intensity")
    if any(name not in fields for name in required):
        raise ValueError("PointCloud2 lacks x/y/z/intensity")
    if any(fields[name].datatype != 7 or fields[name].count != 1 for name in required):
        raise ValueError("x/y/z/intensity must each be one FLOAT32")
    dtype = np.dtype(
        {
            "names": list(required),
            "formats": ["<f4"] * 4,
            "offsets": [int(fields[name].offset) for name in required],
            "itemsize": int(message.point_step),
        }
    )
    view = np.ndarray(
        shape=(int(message.height), int(message.width)),
        dtype=dtype,
        buffer=message.data,
        strides=(int(message.row_step), int(message.point_step)),
    )
    points = np.column_stack([view[name].reshape(-1) for name in required])
    finite = np.isfinite(points).all(axis=1)
    result = np.asarray(points[finite], dtype="<f4")
    return result, int(points.shape[0]), int((~finite).sum())


def _num_points_inside_box(points: np.ndarray, box: dict[str, Any]) -> int:
    """Count finite lidar points inside an oriented 3D box, including edges."""
    dx = points[:, 0] - float(box["x"])
    dy = points[:, 1] - float(box["y"])
    cosine = math.cos(float(box["yaw"]))
    sine = math.sin(float(box["yaw"]))
    local_x = cosine * dx + sine * dy
    local_y = -sine * dx + cosine * dy
    local_z = points[:, 2] - float(box["z"])
    inside = (
        (np.abs(local_x) <= float(box["length"]) / 2.0)
        & (np.abs(local_y) <= float(box["width"]) / 2.0)
        & (np.abs(local_z) <= float(box["height"]) / 2.0)
    )
    return int(inside.sum())


def _scene_definitions(config: dict[str, Any]) -> dict[str, dict[str, str]]:
    scenes = config.get("scenes")
    if not isinstance(scenes, dict) or not scenes:
        raise RuntimeError("config must define a non-empty scenes mapping")
    result: dict[str, dict[str, str]] = {}
    for name, value in scenes.items():
        if not isinstance(name, str) or not name or "/" in name:
            raise RuntimeError(f"invalid scene name: {name!r}")
        if not isinstance(value, dict):
            raise RuntimeError(f"scene {name!r} must be a mapping")
        split = value.get("split")
        scenario = value.get("scenario")
        if split not in ("train", "val", "test"):
            raise RuntimeError(
                f"scene {name!r} has no explicit train/val/test assignment"
            )
        if not isinstance(scenario, str) or not scenario:
            raise RuntimeError(f"scene {name!r} has no scenario evidence path")
        result[name] = {"split": split, "scenario": scenario}
    return result


def _preflight_bags(
    bag_arguments: Iterable[Path],
    config: dict[str, Any],
    scenario_root: Path | None = None,
) -> list[tuple[Path, str, dict[str, str]]]:
    dataset = config.get("dataset")
    if not isinstance(dataset, dict) or not str(dataset.get("version", "")).strip():
        raise RuntimeError("config must define a non-empty dataset.version")
    scene_definitions = _scene_definitions(config)
    result = []
    seen_paths: set[Path] = set()
    seen_names: set[str] = set()
    for argument in bag_arguments:
        bag = argument.resolve()
        scene_name = bag.name
        if not bag.is_dir():
            raise RuntimeError(f"bag directory does not exist: {bag}")
        if bag in seen_paths:
            raise RuntimeError(f"duplicate bag argument: {bag}")
        if scene_name in seen_names:
            raise RuntimeError(f"duplicate scene name from bag basename: {scene_name}")
        scene = scene_definitions.get(scene_name)
        if scene is None:
            raise RuntimeError(f"scene {scene_name!r} is absent from config scenes")
        scenario_path = Path(scene["scenario"])
        if scenario_root is not None and not scenario_path.is_absolute():
            scenario_path = scenario_root / scenario_path
        if scenario_root is not None and not scenario_path.is_file():
            raise RuntimeError(
                f"scene {scene_name!r} scenario evidence does not exist: "
                f"{scenario_path}"
            )
        seen_paths.add(bag)
        seen_names.add(scene_name)
        result.append((bag, scene_name, scene))
    return result


def _preflight_topic_contract(
    bags: Iterable[tuple[Path, str, dict[str, str]]], config: dict[str, Any]
) -> dict[str, str]:
    expected: dict[str, str] | None = None
    for bag, scene_name, _ in bags:
        _, topic_types = _reader(bag, config["topics"].values())
        verified = _verify_topics(topic_types, config)
        if expected is not None and verified != expected:
            raise RuntimeError(
                f"scene {scene_name!r} topic types differ from earlier scenes"
            )
        expected = verified
    return expected or {}


def _static_transform_contract(
    keys: Iterable[tuple[str, str]],
    values: dict[tuple[str, str], DynamicTransform],
) -> dict[str, dict[str, list[float]]]:
    return {
        f"{parent}->{child}": {
            "translation": list(values[(parent, child)].transform.translation),
            "quaternion_xyzw": list(values[(parent, child)].transform.rotation),
        }
        for parent, child in keys
    }


def _same_static_transform_contract(
    left: dict[str, dict[str, list[float]]],
    right: dict[str, dict[str, list[float]]],
    tolerance: float = 1e-9,
) -> bool:
    if left.keys() != right.keys():
        return False
    for edge in left:
        if not np.allclose(
            left[edge]["translation"],
            right[edge]["translation"],
            rtol=0.0,
            atol=tolerance,
        ):
            return False
        left_quaternion = np.asarray(left[edge]["quaternion_xyzw"], dtype=float)
        right_quaternion = np.asarray(right[edge]["quaternion_xyzw"], dtype=float)
        left_quaternion /= np.linalg.norm(left_quaternion)
        right_quaternion /= np.linalg.norm(right_quaternion)
        if not math.isclose(
            abs(float(np.dot(left_quaternion, right_quaternion))),
            1.0,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            return False
    return True


def _transform_box(
    actor: Actor,
    class_config: dict[str, Any],
    map_to_lidar: RigidTransform,
) -> dict[str, Any]:
    values = (*actor.position, actor.heading, *actor.size)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite_actor")
    length, width, height = actor.size
    if min(length, width, height) <= 0.0:
        raise ValueError("invalid_dimensions")
    policy = class_config["center_policy"]
    forward_offset = 0.0
    if policy == "rear_axle_ground_to_box_center":
        geometry = (actor.overhang, actor.wheelbase, actor.rear_overhang)
        if not all(math.isfinite(value) and value >= 0.0 for value in geometry):
            raise ValueError("invalid_vehicle_geometry")
        expected_length = sum(geometry)
        if abs(expected_length - length) > float(class_config["length_tolerance_m"]):
            raise ValueError("vehicle_length_geometry_mismatch")
        forward_offset = (actor.wheelbase + actor.overhang - actor.rear_overhang) / 2.0
    elif policy != "ground_center_to_box_center":
        raise ValueError("unknown_center_policy")
    center_map = (
        actor.position[0] + forward_offset * math.cos(actor.heading),
        actor.position[1] + forward_offset * math.sin(actor.heading),
        actor.position[2] + height / 2.0,
    )
    center_lidar = map_to_lidar.apply(center_map)
    forward_lidar = map_to_lidar.apply(
        (
            center_map[0] + math.cos(actor.heading),
            center_map[1] + math.sin(actor.heading),
            center_map[2],
        )
    )
    yaw = normalize_angle(
        math.atan2(
            forward_lidar[1] - center_lidar[1],
            forward_lidar[0] - center_lidar[0],
        )
    )
    return {
        "actor_id": actor.unique_id,
        "raw_object_type": actor.object_type,
        "class_name": class_config["name"],
        "x": center_lidar[0],
        "y": center_lidar[1],
        "z": center_lidar[2],
        "length": length,
        "width": width,
        "height": height,
        "yaw": yaw,
        "source_position_map": list(actor.position),
        "source_heading_rad": actor.heading,
        "source_size_xyz": list(actor.size),
        "center_policy": policy,
        "forward_center_offset_m": forward_offset,
    }


def _inside_roi(box: dict[str, Any], roi: dict[str, list[float]]) -> bool:
    return all(
        float(roi[axis][0]) <= float(box[axis]) <= float(roi[axis][1])
        for axis in ("x", "y", "z")
    )


def _prepare_output(output: Path, overwrite: bool) -> None:
    marker = output / ".morai_dataset_export"
    if output.exists():
        if not overwrite:
            raise RuntimeError(f"output already exists: {output}; pass --overwrite")
        if not marker.is_file():
            raise RuntimeError(f"refusing to replace unrecognized directory: {output}")
        shutil.rmtree(output)
    for directory in ("points", "labels", "splits"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    marker.write_text("HEVEN MORAI dataset exporter\n", encoding="utf-8")


def _git_metadata(repository: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        text=True,
        capture_output=True,
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


def _export_bag(
    bag: Path,
    output: Path,
    config: dict[str, Any],
    scene_name: str,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]], dict[str, str]]:
    _, deserialize_message, get_message, _ = _message_support()
    alignment_data = _read_alignment_data(bag, config)
    topics = config["topics"]
    reader, topic_types = _reader(bag, [topics["points"]])
    _verify_topics(topic_types, config)
    point_type = get_message(topic_types[topics["points"]])
    maximum = {
        key: round(float(config["alignment"][f"{key}_max_delta_ms"]) * 1_000_000)
        for key in ("actor", "ego", "tf")
    }
    tf_key = tuple(config["transform"]["dynamic_edge"])
    tf_index = alignment_data["tf_indices"].get(tf_key)
    static_keys = [tuple(edge) for edge in config["transform"]["static_edges"]]
    static_values = alignment_data["static"]
    if tf_index is None:
        raise RuntimeError(f"dynamic TF edge is absent: {tf_key}")
    if any(key not in static_values for key in static_keys):
        missing = [key for key in static_keys if key not in static_values]
        raise RuntimeError(f"static TF edges are absent: {missing}")
    base_to_lidar = RigidTransform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    for key in static_keys:
        base_to_lidar = base_to_lidar.compose(static_values[key].transform)
    static_transforms = _static_transform_contract(static_keys, static_values)

    class_mapping = {int(key): value for key, value in config["classes"].items()}
    roi = config["evaluation_roi_lidar_m"]
    skipped = Counter()
    invalid_objects = Counter()
    classes = Counter()
    exported_ids: list[str] = []
    skipped_rows: list[dict[str, Any]] = []
    point_raw_total = 0
    point_finite_total = 0
    gt_total = 0
    visibility = Counter()
    visibility_by_class: dict[str, Counter[str]] = {}
    source_frames = 0
    maximum_observed_delta_ns = {"actor": 0, "ego": 0, "tf": 0}
    while reader.has_next():
        _, serialized, record_ns = reader.read_next()
        cloud = deserialize_message(serialized, point_type)
        source_frames += 1
        source_ns = stamp_ns(cloud.header.stamp)
        sample_id = f"{scene_name}_{source_ns}"

        def skip(reason: str, detail: str = "") -> None:
            skipped[reason] += 1
            skipped_rows.append(
                {"sample_id": sample_id, "source_stamp_ns": source_ns, "reason": reason, "detail": detail}
            )

        if source_ns <= 0 or str(cloud.header.frame_id).lstrip("/") != config["lidar_frame"]:
            skip("invalid_source_header", str(cloud.header.frame_id))
            continue
        actor_sample = alignment_data["actor_index"].nearest(source_ns, maximum["actor"])
        if actor_sample is None:
            nearest = alignment_data["actor_index"].nearest(source_ns, 2**63 - 1)
            detail = (
                f"nearest_delta_ms={abs(nearest.source_ns - source_ns) / 1e6:.6f}"
                if nearest is not None
                else "no_actor_samples"
            )
            skip("actor_timestamp_gap", detail)
            continue
        ego_sample = alignment_data["ego_index"].nearest(source_ns, maximum["ego"])
        if ego_sample is None:
            skip("ego_timestamp_gap")
            continue
        tf_sample = tf_index.nearest(source_ns, maximum["tf"])
        if tf_sample is None:
            skip("tf_timestamp_gap")
            continue
        if (
            actor_sample.value.frame_id != config["map_frame"]
            or ego_sample.value.frame_id != config["map_frame"]
        ):
            skip("frame_contract_mismatch")
            continue
        try:
            points, raw_count, nonfinite_count = _point_array(cloud)
        except (TypeError, ValueError) as error:
            skip("invalid_pointcloud", str(error))
            continue
        if not len(points):
            skip("no_finite_points")
            continue

        maximum_observed_delta_ns["actor"] = max(
            maximum_observed_delta_ns["actor"],
            abs(actor_sample.source_ns - source_ns),
        )
        maximum_observed_delta_ns["ego"] = max(
            maximum_observed_delta_ns["ego"], abs(ego_sample.source_ns - source_ns)
        )
        maximum_observed_delta_ns["tf"] = max(
            maximum_observed_delta_ns["tf"], abs(tf_sample.source_ns - source_ns)
        )

        map_to_base = ego_sample.value.pose
        odom_to_base = tf_sample.value.transform
        map_to_odom = odom_to_base.compose(map_to_base.inverse())
        map_to_lidar = base_to_lidar.inverse().compose(odom_to_base.inverse()).compose(map_to_odom)
        boxes = []
        frame_invalid = None
        for actor in actor_sample.value.actors:
            class_config = class_mapping.get(actor.object_type)
            if class_config is None:
                frame_invalid = f"unknown object_type {actor.object_type}"
                invalid_objects["unknown_class"] += 1
                break
            try:
                box = _transform_box(actor, class_config, map_to_lidar)
            except ValueError as error:
                frame_invalid = f"actor {actor.unique_id}: {error}"
                invalid_objects[str(error)] += 1
                break
            if _inside_roi(box, roi):
                box["num_lidar_points_inside_box"] = _num_points_inside_box(
                    points, box
                )
                boxes.append(box)
        if frame_invalid is not None:
            skip("invalid_gt_object", frame_invalid)
            continue

        point_path = output / "points" / f"{sample_id}.bin"
        label_path = output / "labels" / f"{sample_id}.json"
        points.tofile(point_path)
        label = {
            "schema_version": 1,
            "sample_id": sample_id,
            "scene": scene_name,
            "source": {
                "bag": str(bag),
                "topic": topics["points"],
                "header_stamp_ns": source_ns,
                "record_timestamp_ns": int(record_ns),
                "frame_id": config["lidar_frame"],
            },
            "points": {
                "path": f"points/{sample_id}.bin",
                "dtype": "float32_little_endian",
                "fields": ["x", "y", "z", "intensity"],
                "raw_count": raw_count,
                "finite_count": int(len(points)),
                "nonfinite_removed": nonfinite_count,
            },
            "ground_truth": {
                "topic": topics["actor_gt"],
                "header_stamp_ns": actor_sample.source_ns,
                "source_delta_ns": actor_sample.source_ns - source_ns,
                "source_frame": config["map_frame"],
                "target_frame": config["lidar_frame"],
                "boxes": boxes,
            },
            "transform_alignment": {
                "ego_header_stamp_ns": ego_sample.source_ns,
                "ego_delta_ns": ego_sample.source_ns - source_ns,
                "tf_header_stamp_ns": tf_sample.source_ns,
                "tf_delta_ns": tf_sample.source_ns - source_ns,
                "chain": ["map", "odom", "base_link", "rear_axle_link", "lidar_link"],
                "map_to_odom": {
                    "translation": list(map_to_odom.translation),
                    "quaternion_xyzw": list(map_to_odom.rotation),
                },
            },
        }
        label_path.write_text(json.dumps(label, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        exported_ids.append(sample_id)
        point_raw_total += raw_count
        point_finite_total += len(points)
        gt_total += len(boxes)
        classes.update(box["class_name"] for box in boxes)
        for box in boxes:
            count = int(box["num_lidar_points_inside_box"])
            class_name = str(box["class_name"])
            class_visibility = visibility_by_class.setdefault(
                class_name, Counter()
            )
            visibility["total_points_inside_boxes"] += count
            class_visibility["objects"] += 1
            class_visibility["total_points_inside_boxes"] += count
            if count == 0:
                visibility["zero_point_objects"] += 1
                class_visibility["zero_point_objects"] += 1
            elif count <= 5:
                visibility["one_to_five_point_objects"] += 1
                class_visibility["one_to_five_point_objects"] += 1

    summary = {
        "source_frames": source_frames,
        "exported_frames": len(exported_ids),
        "skipped_frames": sum(skipped.values()),
        "skip_reasons": dict(sorted(skipped.items())),
        "invalid_objects": dict(sorted(invalid_objects.items())),
        "gt_objects": gt_total,
        "class_distribution": dict(sorted(classes.items())),
        "static_transforms": static_transforms,
        "visibility": {
            "zero_point_objects": visibility["zero_point_objects"],
            "one_to_five_point_objects": visibility["one_to_five_point_objects"],
            "total_points_inside_boxes": visibility["total_points_inside_boxes"],
            "by_class": {
                class_name: {
                    "objects": values["objects"],
                    "zero_point_objects": values["zero_point_objects"],
                    "one_to_five_point_objects": values[
                        "one_to_five_point_objects"
                    ],
                    "total_points_inside_boxes": values[
                        "total_points_inside_boxes"
                    ],
                }
                for class_name, values in sorted(visibility_by_class.items())
            },
        },
        "raw_points": point_raw_total,
        "finite_points": point_finite_total,
        "nonfinite_points_removed": point_raw_total - point_finite_total,
        "maximum_observed_delta_ms": {
            key: value / 1_000_000.0
            for key, value in maximum_observed_delta_ns.items()
        },
    }
    return summary, exported_ids, skipped_rows, alignment_data["topic_types"]


def _write_dataset_documents(
    output: Path,
    config: dict[str, Any],
    metadata: dict[str, Any],
    split_ids: dict[str, list[str]],
    skipped_rows: list[dict[str, Any]],
) -> None:
    for split, identifiers in split_ids.items():
        (output / "splits" / f"{split}.txt").write_text(
            "".join(f"{identifier}\n" for identifier in identifiers), encoding="utf-8"
        )
    with (output / "skipped_frames.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["sample_id", "source_stamp_ns", "reason", "detail"],
        )
        writer.writeheader()
        writer.writerows(skipped_rows)
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = metadata["summary"]
    scene_table = "\n".join(
        f"| `{scene['name']}` | `{scene['split']}` | `{scene['scenario']}` | "
        f"{scene['summary']['exported_frames']} |"
        for scene in metadata["scenes"]
    )
    readme = f"""# MORAI HEVEN 3D detection dataset

This dataset was exported from recorded HEVEN topics. It contains no model,
training output, or CenterPoint integration.

- Dataset name: `{metadata['dataset_name']}`
- Dataset version: `{metadata['dataset_version']}`

## Contents

- `points/*.bin`: little-endian float32 XYZI, one finite point per row.
- `labels/*.json`: source timestamps, point counts, transform provenance, and
  lidar-frame 3D boxes with exact `num_lidar_points_inside_box` visibility.
- `splits/*.txt`: sample IDs.
- `skipped_frames.csv`: every rejected source frame and reason.
- `metadata.json`: source/config/environment and aggregate statistics.

## Export summary

- Source frames: {summary['source_frames']}
- Exported frames: {summary['exported_frames']}
- Skipped frames: {summary['skipped_frames']}
- GT objects in configured lidar ROI: {summary['gt_objects']}
- Class distribution: `{json.dumps(summary['class_distribution'], sort_keys=True)}`
- Skip reasons: `{json.dumps(summary['skip_reasons'], sort_keys=True)}`
- Raw / finite / removed-nonfinite points: {summary['raw_points']} /
  {summary['finite_points']} / {summary['nonfinite_points_removed']}
- Zero-point / 1--5-point GT objects: {summary['visibility']['zero_point_objects']} /
  {summary['visibility']['one_to_five_point_objects']}
- Total point memberships across GT boxes: {summary['visibility']['total_points_inside_boxes']}
- Maximum accepted actor / ego / TF gap: {summary['maximum_observed_delta_ms']['actor']:.6f} /
  {summary['maximum_observed_delta_ms']['ego']:.6f} /
  {summary['maximum_observed_delta_ms']['tf']:.6f} ms

## Timestamp policy

The PointCloud2 `header.stamp` is the sample timestamp. Actor GT, ego status,
and dynamic TF are independently selected by nearest source `header.stamp`,
with no interpolation. Maximum absolute gaps are actor
{config['alignment']['actor_max_delta_ms']:g} ms, ego
{config['alignment']['ego_max_delta_ms']:g} ms, and TF
{config['alignment']['tf_max_delta_ms']:g} ms. A missing or larger-gap input
rejects the entire frame; zero transforms are never substituted.

## Transform convention

All output points and boxes use `lidar_link`: +X forward, +Y left, +Z up.
The evaluated chain is `map -> odom -> base_link -> rear_axle_link ->
lidar_link`. The bag has no direct map-to-odom TF. For every cloud, the exporter
derives it from the nearest ego-status map pose (`map -> base_link`, as used by
the repository's zero-offset status-pose localizer) and recorded `odom ->
base_link`; static rear-axle and lidar transforms come from `/tf_static`.

## Box convention

Each label stores center `(x,y,z)`, `(length,width,height)`, and yaw in radians
in `lidar_link`. Yaw is counter-clockwise from lidar +X and normalized to
`[-pi, pi]`. MORAI bridge code converts ObjectInfo heading degrees to radians;
the map convention is ENU, heading zero at +X/East and positive CCW.

`ObjectStatus.size.x/y/z` is exported as length/width/height without swapping.
Vehicle length is validated against `overhang + wheelbase + rear_overhang`.
The repository defines vehicle/status origin as rear-axle center at ground, so
vehicle centers shift forward by `(wheelbase + overhang - rear_overhang)/2`
and upward by `height/2`. Pedestrian and obstacle scenario positions are their
ground-centered origins and shift only upward by `height/2`. Raw source fields
and the applied center policy remain in every label.

The configured class mapping evidence is
`{config['class_evidence']['scenario']}` with audited matches
`{json.dumps(config['class_evidence']['matches'], sort_keys=True)}`. Raw
`object_type` is not assumed to be the gRPC ObjectType enum. Every additional
scene's scenario evidence must independently confirm the same mapping; otherwise
it belongs in a new dataset version.

Only centers within the configured lidar ROI are labeled. Visibility and
occlusion are not inferred. `num_lidar_points_inside_box` counts finite exported
points inside the exact oriented box, including its boundary and using no
margin. It is a visibility statistic, not an occlusion label or correctness
score; zero-point objects remain valid GT review candidates.

## Scene manifest

| scene | split | scenario evidence | exported frames |
|---|---|---|---:|
{scene_table}

## Splits

Splits are bag/scene-level, never random frame-level. Every input bag basename
must have an explicit `train`, `val`, or `test` assignment in the versioned
config before export. Empty split files are intentional; the exporter never
creates a random fallback split. Multiple bags are merged only by rebuilding a
new exporter-owned output from the complete repeated `--bag` list.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")


def main() -> int:
    arguments = parse_arguments()
    config_bytes = arguments.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    repository = Path(__file__).resolve().parents[2]
    bags = _preflight_bags(arguments.bag, config, repository)
    preflight_topic_types = _preflight_topic_contract(bags, config)
    output = arguments.output.resolve()
    _prepare_output(output, arguments.overwrite)
    commit, dirty = _git_metadata(repository)
    aggregate = Counter()
    aggregate_classes = Counter()
    aggregate_skips = Counter()
    aggregate_visibility = Counter()
    aggregate_visibility_by_class: dict[str, Counter[str]] = {}
    aggregate_maximum_deltas = {"actor": 0.0, "ego": 0.0, "tf": 0.0}
    all_skipped = []
    split_ids = {"train": [], "val": [], "test": []}
    scenes = []
    topic_types = preflight_topic_types
    common_static_transforms: dict[str, dict[str, list[float]]] | None = None
    for bag, scene_name, scene_definition in bags:
        split = scene_definition["split"]
        summary, identifiers, skipped, types = _export_bag(
            bag, output, config, scene_name
        )
        scenes.append(
            {
                "name": scene_name,
                "bag": str(bag),
                "split": split,
                "scenario": scene_definition["scenario"],
                "summary": summary,
            }
        )
        split_ids[split].extend(identifiers)
        all_skipped.extend(skipped)
        if types != topic_types:
            raise RuntimeError(
                f"scene {scene_name!r} topic types differ from earlier scenes"
            )
        scene_static_transforms = summary["static_transforms"]
        if common_static_transforms is None:
            common_static_transforms = scene_static_transforms
        elif not _same_static_transform_contract(
            common_static_transforms, scene_static_transforms
        ):
            raise RuntimeError(
                f"scene {scene_name!r} static sensor transforms differ from "
                "earlier scenes"
            )
        for key in (
            "source_frames",
            "exported_frames",
            "skipped_frames",
            "gt_objects",
            "raw_points",
            "finite_points",
            "nonfinite_points_removed",
        ):
            aggregate[key] += summary[key]
        aggregate_classes.update(summary["class_distribution"])
        aggregate_skips.update(summary["skip_reasons"])
        for key in (
            "zero_point_objects",
            "one_to_five_point_objects",
            "total_points_inside_boxes",
        ):
            aggregate_visibility[key] += summary["visibility"][key]
        for class_name, values in summary["visibility"]["by_class"].items():
            class_visibility = aggregate_visibility_by_class.setdefault(
                class_name, Counter()
            )
            class_visibility.update(values)
        for key, value in summary["maximum_observed_delta_ms"].items():
            aggregate_maximum_deltas[key] = max(
                aggregate_maximum_deltas[key], value
            )
    summary = dict(aggregate)
    summary["class_distribution"] = dict(sorted(aggregate_classes.items()))
    summary["skip_reasons"] = dict(sorted(aggregate_skips.items()))
    summary["maximum_observed_delta_ms"] = aggregate_maximum_deltas
    summary["visibility"] = {
        "zero_point_objects": aggregate_visibility["zero_point_objects"],
        "one_to_five_point_objects": aggregate_visibility[
            "one_to_five_point_objects"
        ],
        "total_points_inside_boxes": aggregate_visibility[
            "total_points_inside_boxes"
        ],
        "by_class": {
            class_name: {
                key: values[key]
                for key in (
                    "objects",
                    "zero_point_objects",
                    "one_to_five_point_objects",
                    "total_points_inside_boxes",
                )
            }
            for class_name, values in sorted(
                aggregate_visibility_by_class.items()
            )
        },
    }
    metadata = {
        "schema_version": 1,
        "dataset_name": config["dataset"].get("name", "morai_heven"),
        "dataset_version": str(config["dataset"]["version"]),
        "git_commit": commit,
        "git_dirty": dirty,
        "config": str(arguments.config.resolve()),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "ros_distro": os.environ.get("ROS_DISTRO", "unknown"),
        "topic_types": topic_types,
        "alignment": config["alignment"],
        "transform": config["transform"],
        "static_transforms": common_static_transforms or {},
        "classes": config["classes"],
        "class_evidence": config["class_evidence"],
        "lidar_frame": config["lidar_frame"],
        "map_frame": config["map_frame"],
        "evaluation_roi_lidar_m": config["evaluation_roi_lidar_m"],
        "visibility_statistic": {
            "field": "num_lidar_points_inside_box",
            "points": "finite exported XYZI points",
            "box": "exact oriented 3D box with inclusive boundaries",
            "margin_m": 0.0,
        },
        "merge_policy": {
            "mode": "single_pass_all_bags",
            "scene_identity": "bag directory basename",
            "duplicate_scene_names": "rejected",
            "split_unit": "whole scene",
        },
        "scenes": scenes,
        "summary": summary,
    }
    _write_dataset_documents(output, config, metadata, split_ids, all_skipped)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
