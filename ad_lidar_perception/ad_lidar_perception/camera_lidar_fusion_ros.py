"""POST-FREEZE EXTENSION 2, Stage 1: pure message<->core adapter helpers for
the opt-in camera+LiDAR late-fusion path. Message classes are always injected
(same pattern as `ab3dmot_ros.py` / `centerpoint_ros.py`) so this module has
no hard ROS import and its logic is unit-testable without a ROS environment.

Publish contract (documented, adapter unit-tested, NOT exercised live in this
stage -- see limitations.md, blockers B1/B2):
  input  : /ad/perception/objects/detected      autoware_perception_msgs/DetectedObjects (unchanged, read-only)
           /vision/dynamic_obstacle/detections  vision_msgs/Detection2DArray (from ad_camera_perception)
  output : /experiment/perception/objects/camera_lidar_fused
           autoware_perception_msgs/DetectedObjects -- a COPY of each fused LiDAR
           object with its `classification` replaced by the camera-derived
           semantic label + probability, and `existence_probability` blended.
           The original /ad/perception/objects/detected stream is never modified.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

from ad_lidar_perception.camera_lidar_fusion_core import (
    CameraDetection,
    FusionResult,
    LidarDetection,
    map_coco_name_to_semantic,
)

__all__ = [
    "SEMANTIC_TO_AUTOWARE_LABEL",
    "AUTOWARE_LABEL_TO_SEMANTIC",
    "detected_objects_to_lidar_detections",
    "detection2d_array_to_camera_detections",
    "fused_result_to_detected_objects",
]

# autoware_perception_msgs/ObjectClassification integer labels.
_AW_UNKNOWN, _AW_CAR, _AW_TRUCK, _AW_BUS = 0, 1, 2, 3
_AW_TRAILER, _AW_MOTORCYCLE, _AW_BICYCLE, _AW_PEDESTRIAN = 4, 5, 6, 7

SEMANTIC_TO_AUTOWARE_LABEL = {
    "car": _AW_CAR,
    "truck": _AW_TRUCK,
    "bus": _AW_BUS,
    "motorcycle": _AW_MOTORCYCLE,
    "bicycle": _AW_BICYCLE,
    "person": _AW_PEDESTRIAN,
    "unknown": _AW_UNKNOWN,
}
AUTOWARE_LABEL_TO_SEMANTIC = {v: k for k, v in SEMANTIC_TO_AUTOWARE_LABEL.items()}


def _stamp_ns(header) -> int:
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def detected_objects_to_lidar_detections(message) -> list:
    """autoware_perception_msgs/DetectedObjects -> list[LidarDetection].

    Reads only bounding-box objects. Non-finite geometry is skipped (logged by
    the caller if desired). Dimensions are passed through verbatim, INCLUDING
    the Euclidean detector's near-degenerate 0.1 m clutter boxes -- the fusion
    core represents and flags those rather than dropping them.
    """
    out = []
    for index, obj in enumerate(message.objects):
        shape = obj.shape
        # 0 == BOUNDING_BOX in autoware_perception_msgs/Shape
        if getattr(shape, "type", 0) != 0:
            continue
        pose = obj.kinematics.pose_with_covariance.pose
        dims = shape.dimensions
        q = pose.orientation
        yaw = _yaw_from_quat(q.x, q.y, q.z, q.w)
        values = (pose.position.x, pose.position.y, pose.position.z,
                  dims.x, dims.y, dims.z, yaw)
        if not all(math.isfinite(v) for v in values):
            continue
        out.append(LidarDetection(
            x=float(pose.position.x), y=float(pose.position.y), z=float(pose.position.z),
            length=float(dims.x), width=float(dims.y), height=float(dims.z),
            yaw=float(yaw),
            score=float(obj.classification[0].probability) if obj.classification else 1.0,
            source_index=index,
        ))
    return out


def _yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def detection2d_array_to_camera_detections(message, *, default_confidence: float = 0.0) -> list:
    """vision_msgs/Detection2DArray -> list[CameraDetection].

    Each Detection2D carries a center (bbox.center.position.x/y) + size
    (bbox.size_x/size_y) and one ObjectHypothesisWithPose whose
    `hypothesis.class_id` is the external (COCO) class NAME string and
    `hypothesis.score` the confidence -- exactly what
    `ad_camera_perception.inference.detection_converter.detections_to_message`
    produces. Classes outside the vehicle/person set are dropped.
    """
    out = []
    for index, det in enumerate(message.detections):
        if not det.results:
            continue
        hypo = det.results[0].hypothesis
        raw_name = str(hypo.class_id)
        semantic = map_coco_name_to_semantic(raw_name)
        if semantic is None:
            continue
        cx = float(det.bbox.center.position.x)
        cy = float(det.bbox.center.position.y)
        half_w = float(det.bbox.size_x) / 2.0
        half_h = float(det.bbox.size_y) / 2.0
        conf = float(hypo.score) if hypo.score else default_confidence
        out.append(CameraDetection(
            x1=cx - half_w, y1=cy - half_h, x2=cx + half_w, y2=cy + half_h,
            confidence=conf, semantic_class=semantic, raw_class_name=raw_name,
            source_index=index,
        ))
    return out


def fused_result_to_detected_objects(
    result: FusionResult,
    source_header,
    *,
    detected_objects_cls,
    detected_object_cls,
    object_classification_cls,
    shape_cls,
    include_lidar_only: bool = True,
):
    """Build the experimental /experiment/perception/objects/camera_lidar_fused
    message: one DetectedObject per fused (and optionally lidar-only) LiDAR
    detection, geometry preserved, classification set from the camera semantics.

    All message classes are injected so this stays importable without ROS.
    """
    message = detected_objects_cls()
    message.header = source_header

    entries = list(result.fused)
    if include_lidar_only:
        entries += list(result.lidar_only)

    for fused in entries:
        det = fused.lidar
        obj = detected_object_cls()
        # geometry: LiDAR stays authoritative
        obj.kinematics.pose_with_covariance.pose.position.x = float(det.x)
        obj.kinematics.pose_with_covariance.pose.position.y = float(det.y)
        obj.kinematics.pose_with_covariance.pose.position.z = float(det.z)
        obj.shape.type = 0  # BOUNDING_BOX
        obj.shape.dimensions.x = float(det.length)
        obj.shape.dimensions.y = float(det.width)
        obj.shape.dimensions.z = float(det.height)

        classification = object_classification_cls()
        classification.label = SEMANTIC_TO_AUTOWARE_LABEL.get(fused.semantic_class, _AW_UNKNOWN)
        classification.probability = float(fused.semantic_confidence)
        obj.classification = [classification]

        # existence: blend the LiDAR detector's own score with the camera's,
        # weighted by whether the camera visually confirmed it. Never > 1.0.
        base = float(det.score) if math.isfinite(det.score) else 1.0
        if fused.match_state == "fused":
            obj.existence_probability = float(min(1.0, 0.5 * base + 0.5 * fused.semantic_confidence))
        else:
            obj.existence_probability = float(min(1.0, base))
        message.objects.append(obj)

    return message
