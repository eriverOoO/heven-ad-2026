"""Pure conversion helpers for the optional CenterPoint ROS adapter."""

from __future__ import annotations

import math
import struct
from typing import Any, Iterable

import numpy as np


CLASS_NAMES = ("vehicle", "pedestrian", "obstacle")


def validate_backend(backend: str) -> str:
    if backend not in {"euclidean", "centerpoint"}:
        raise ValueError("detector_backend must be euclidean or centerpoint")
    return backend


def pointcloud2_to_xyzi(message: Any) -> np.ndarray:
    """Decode finite FLOAT32 x/y/z/intensity fields without changing axes."""
    fields = {field.name: field for field in message.fields}
    required = ("x", "y", "z", "intensity")
    missing = [name for name in required if name not in fields]
    if missing:
        raise ValueError(f"PointCloud2 is missing fields: {missing}")
    if any(fields[name].datatype != 7 or fields[name].count != 1 for name in required):
        raise ValueError("x/y/z/intensity must be scalar FLOAT32 fields")
    if message.point_step <= 0 or message.row_step < message.point_step * message.width:
        raise ValueError("invalid PointCloud2 point/row stride")
    endian = ">" if message.is_bigendian else "<"
    rows = []
    for row in range(message.height):
        row_start = row * message.row_step
        for column in range(message.width):
            point_start = row_start + column * message.point_step
            values = tuple(
                struct.unpack_from(f"{endian}f", message.data, point_start + fields[name].offset)[0]
                for name in required
            )
            if all(math.isfinite(value) for value in values):
                rows.append(values)
    return np.asarray(rows, dtype=np.float32).reshape(-1, 4)


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    if not math.isfinite(yaw):
        raise ValueError("yaw must be finite")
    return 0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)


def filter_detections(
    detections: Iterable[dict[str, Any]], score_threshold: float, max_detections: int
) -> list[dict[str, Any]]:
    if not 0.0 <= score_threshold <= 1.0 or max_detections < 1:
        raise ValueError("invalid detection filter configuration")
    accepted = [item for item in detections if float(item["score"]) >= score_threshold]
    accepted.sort(key=lambda item: -float(item["score"]))
    return accepted[:max_detections]


def detections_to_message(
    *, header: Any, detections: Iterable[dict[str, Any]], message_types: dict[str, Any]
) -> Any:
    """Map lidar boxes to the exact Autoware DetectedObjects tracker input."""
    output = message_types["DetectedObjects"]()
    output.header = header
    labels = {
        "vehicle": message_types["ObjectClassification"].CAR,
        "pedestrian": message_types["ObjectClassification"].PEDESTRIAN,
        # The established Euclidean path uses UNKNOWN; no semantic category is invented.
        "obstacle": message_types["ObjectClassification"].UNKNOWN,
    }
    for item in detections:
        name = str(item["class_name"])
        if name not in labels:
            raise ValueError(f"unknown MORAI class: {name!r}")
        score = float(item["score"])
        box = tuple(float(value) for value in item["box_lidar"])
        if len(box) != 7 or not all(math.isfinite(value) for value in box):
            raise ValueError("box_lidar must contain seven finite values")
        if not 0.0 <= score <= 1.0 or min(box[3:6]) <= 0.0:
            raise ValueError("invalid score or box dimensions")
        detected = message_types["DetectedObject"]()
        detected.existence_probability = score
        classification = message_types["ObjectClassification"]()
        classification.label = labels[name]
        classification.probability = score
        detected.classification.append(classification)
        pose = detected.kinematics.pose_with_covariance.pose
        pose.position.x, pose.position.y, pose.position.z = box[:3]
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = yaw_to_quaternion(box[6])
        detected.kinematics.orientation_availability = message_types["DetectedObjectKinematics"].AVAILABLE
        detected.kinematics.has_twist = False
        detected.kinematics.has_twist_covariance = False
        detected.shape.type = message_types["Shape"].BOUNDING_BOX
        detected.shape.dimensions.x, detected.shape.dimensions.y, detected.shape.dimensions.z = box[3:6]
        output.objects.append(detected)
    return output
