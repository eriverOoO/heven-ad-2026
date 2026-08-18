"""Pure helper for recording DetectedObjects into a comparable JSONL record."""

from __future__ import annotations

from typing import Any


def summarize_detected_objects(
    message: Any,
    *,
    backend: str,
    received_wall_time_ns: int,
    label_names: dict[int, str],
) -> dict[str, Any]:
    """Build one ``heven.ros_detection_comparison.v1`` record from a DetectedObjects message."""
    if not isinstance(backend, str) or not backend:
        raise ValueError("backend must be a non-empty string")
    header_stamp_ns = (
        int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)
    )
    objects: list[dict[str, Any]] = []
    for detected in message.objects:
        if not detected.classification:
            raise ValueError("DetectedObject has no classification")
        label = int(detected.classification[0].label)
        position = detected.kinematics.pose_with_covariance.pose.position
        dimensions = detected.shape.dimensions
        objects.append(
            {
                "class_name": label_names.get(label, f"label_{label}"),
                "score": float(detected.existence_probability),
                "position": [float(position.x), float(position.y), float(position.z)],
                "dimensions": [
                    float(dimensions.x),
                    float(dimensions.y),
                    float(dimensions.z),
                ],
            }
        )
    return {
        "schema": "heven.ros_detection_comparison.v1",
        "backend": backend,
        "frame_id": message.header.frame_id,
        "header_stamp_ns": header_stamp_ns,
        "received_wall_time_ns": int(received_wall_time_ns),
        "latency_ms": (int(received_wall_time_ns) - header_stamp_ns) / 1e6,
        "object_count": len(objects),
        "objects": objects,
    }
