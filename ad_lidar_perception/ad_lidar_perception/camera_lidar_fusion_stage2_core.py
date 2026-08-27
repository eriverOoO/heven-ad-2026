"""Reusable, analysis-only helpers for Camera/LiDAR Fusion Stage 2.

Nothing in this module is wired into ROS or a production launch.  It provides
transparent geometry filtering, deterministic adjacent-frame linking, camera
intrinsic perturbation, match-set comparison, and descriptive semantic
stability statistics for the GT-free Stage-2 experiment.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from .camera_lidar_fusion_core import CameraIntrinsics, LidarDetection

__all__ = [
    "GeometryFilterRule",
    "TemporalDetection",
    "TemporalLinkConfig",
    "SemanticObservation",
    "passes_geometry_filter",
    "link_adjacent_frames",
    "perturb_intrinsics",
    "compare_match_sets",
    "semantic_stability_statistics",
]


@dataclass(frozen=True)
class GeometryFilterRule:
    """LiDAR-only box thresholds; ``None`` means that threshold is disabled."""

    name: str
    min_dimension_m: Optional[float] = None
    min_footprint_m2: Optional[float] = None
    min_volume_m3: Optional[float] = None
    max_footprint_aspect_ratio: Optional[float] = None

    def __post_init__(self) -> None:
        for field_name in (
            "min_dimension_m", "min_footprint_m2", "min_volume_m3",
            "max_footprint_aspect_ratio",
        ):
            value = getattr(self, field_name)
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(f"{field_name} must be finite and non-negative")


def passes_geometry_filter(detection: LidarDetection, rule: GeometryFilterRule) -> bool:
    """Return whether a LiDAR box passes a declared geometry-only rule."""

    dimensions = (detection.length, detection.width, detection.height)
    if not all(math.isfinite(value) and value > 0.0 for value in dimensions):
        return False
    footprint = detection.length * detection.width
    volume = footprint * detection.height
    short_side = min(detection.length, detection.width)
    aspect_ratio = max(detection.length, detection.width) / short_side
    return not (
        (rule.min_dimension_m is not None and min(dimensions) <= rule.min_dimension_m)
        or (rule.min_footprint_m2 is not None and footprint < rule.min_footprint_m2)
        or (rule.min_volume_m3 is not None and volume < rule.min_volume_m3)
        or (
            rule.max_footprint_aspect_ratio is not None
            and aspect_ratio > rule.max_footprint_aspect_ratio
        )
    )


@dataclass(frozen=True)
class TemporalDetection:
    frame_index: int
    detection_index: int
    x: float
    y: float
    length: float
    width: float
    height: float


@dataclass(frozen=True)
class TemporalLinkConfig:
    """Strict adjacent-frame gates for the analysis-only LiDAR linker."""

    center_distance_gate_m: float
    max_log_size_ratio: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.center_distance_gate_m) or self.center_distance_gate_m <= 0.0:
            raise ValueError("center_distance_gate_m must be finite and positive")
        if not math.isfinite(self.max_log_size_ratio) or self.max_log_size_ratio <= 0.0:
            raise ValueError("max_log_size_ratio must be finite and positive")


def _size_log_ratio(first: TemporalDetection, second: TemporalDetection) -> float:
    # Sorting horizontal sides makes the comparison invariant to an axis swap
    # in an oriented detector box between adjacent frames.
    first_size = np.asarray(sorted((first.length, first.width)) + [first.height], dtype=float)
    second_size = np.asarray(sorted((second.length, second.width)) + [second.height], dtype=float)
    if np.any(first_size <= 0.0) or np.any(second_size <= 0.0):
        return math.inf
    return float(np.max(np.abs(np.log(second_size / first_size))))


def link_adjacent_frames(
    frames: Sequence[Sequence[TemporalDetection]],
    config: TemporalLinkConfig,
) -> dict[tuple[int, int], int]:
    """Link detections across adjacent frames with gated one-to-one Hungarian.

    Tracks never coast: a detection can inherit an ``analysis_track_id`` only
    from a detection in the immediately preceding frame.  The returned key is
    ``(frame_index, detection_index)``.  These IDs are analysis links, not GT.
    """

    from .ab3dmot_core import _hungarian_matching

    links: dict[tuple[int, int], int] = {}
    previous: list[TemporalDetection] = []
    previous_ids: list[int] = []
    next_track_id = 1
    invalid_cost = 1e6

    for expected_frame, frame in enumerate(frames):
        current = sorted(frame, key=lambda detection: detection.detection_index)
        if any(detection.frame_index != expected_frame for detection in current):
            raise ValueError("TemporalDetection.frame_index must match its frame position")
        current_ids: list[Optional[int]] = [None] * len(current)
        if previous and current:
            costs = np.full((len(previous), len(current)), invalid_cost, dtype=float)
            for row, prior in enumerate(previous):
                for column, now in enumerate(current):
                    distance = math.hypot(now.x - prior.x, now.y - prior.y)
                    size_ratio = _size_log_ratio(prior, now)
                    if (
                        distance <= config.center_distance_gate_m
                        and size_ratio <= config.max_log_size_ratio
                    ):
                        costs[row, column] = (
                            distance / config.center_distance_gate_m
                            + size_ratio / config.max_log_size_ratio
                        )
            for row, column in _hungarian_matching(costs):
                row, column = int(row), int(column)
                if costs[row, column] < invalid_cost:
                    current_ids[column] = previous_ids[row]

        for index, detection in enumerate(current):
            if current_ids[index] is None:
                current_ids[index] = next_track_id
                next_track_id += 1
            links[(detection.frame_index, detection.detection_index)] = int(current_ids[index])
        previous = current
        previous_ids = [int(track_id) for track_id in current_ids]
    return links


def perturb_intrinsics(
    baseline: CameraIntrinsics,
    *,
    focal_scale: float = 1.0,
    cx_offset_fraction: float = 0.0,
    cy_offset_fraction: float = 0.0,
) -> CameraIntrinsics:
    """Create one bounded sensitivity-sweep K from a baseline camera model."""

    values = (focal_scale, cx_offset_fraction, cy_offset_fraction)
    if not all(math.isfinite(value) for value in values) or focal_scale <= 0.0:
        raise ValueError("intrinsic perturbations must be finite and focal_scale positive")
    return CameraIntrinsics(
        fx=baseline.fx * focal_scale,
        fy=baseline.fy * focal_scale,
        cx=baseline.cx + cx_offset_fraction * baseline.width,
        cy=baseline.cy + cy_offset_fraction * baseline.height,
        width=baseline.width,
        height=baseline.height,
        calibration_quality=baseline.calibration_quality,
        distortion_model=baseline.distortion_model,
    )


def compare_match_sets(
    baseline: Iterable[tuple[int, ...]],
    condition: Iterable[tuple[int, ...]],
) -> dict[str, float | int]:
    """Compare accepted correspondence sets without implying correctness."""

    baseline_set = set(baseline)
    condition_set = set(condition)
    intersection = baseline_set & condition_set
    union = baseline_set | condition_set
    return {
        "baseline_matches": len(baseline_set),
        "condition_matches": len(condition_set),
        "baseline_retained": len(intersection),
        "new_matches": len(condition_set - baseline_set),
        "lost_matches": len(baseline_set - condition_set),
        "jaccard_overlap": len(intersection) / len(union) if union else 1.0,
        "baseline_retention_ratio": (
            len(intersection) / len(baseline_set) if baseline_set else 1.0
        ),
    }


@dataclass(frozen=True)
class SemanticObservation:
    frame_index: int
    camera_class: Optional[str]
    camera_confidence: Optional[float] = None


def semantic_stability_statistics(observations: Sequence[SemanticObservation]) -> dict:
    """Describe semantic availability/consistency on one analysis trajectory."""

    ordered = sorted(observations, key=lambda observation: observation.frame_index)
    if len({observation.frame_index for observation in ordered}) != len(ordered):
        raise ValueError("semantic observations must have unique frame indices")
    matched = [observation for observation in ordered if observation.camera_class is not None]
    counts = Counter(observation.camera_class for observation in matched)
    modal_class = None
    if counts:
        modal_class = min(counts, key=lambda label: (-counts[label], label))

    class_flips = 0
    matched_unmatched_transitions = 0
    longest_run = 0
    current_label = None
    current_run = 0
    previous = None
    for observation in ordered:
        if previous is not None and (
            (previous.camera_class is None) != (observation.camera_class is None)
        ):
            matched_unmatched_transitions += 1
        if (
            previous is not None
            and previous.camera_class is not None
            and observation.camera_class is not None
            and previous.camera_class != observation.camera_class
        ):
            class_flips += 1
        if observation.camera_class is not None and observation.camera_class == current_label:
            current_run += 1
        elif observation.camera_class is not None:
            current_label = observation.camera_class
            current_run = 1
        else:
            current_label = None
            current_run = 0
        longest_run = max(longest_run, current_run)
        previous = observation

    confidences = np.asarray(
        [
            observation.camera_confidence
            for observation in matched
            if observation.camera_confidence is not None
        ],
        dtype=float,
    )
    return {
        "observed_frames": len(ordered),
        "camera_matched_frames": len(matched),
        "camera_match_ratio": len(matched) / len(ordered) if ordered else 0.0,
        "camera_class_sequence": [
            observation.camera_class if observation.camera_class is not None else "unmatched"
            for observation in ordered
        ],
        "modal_semantic_class": modal_class,
        "modal_class_consistency_ratio": (
            counts[modal_class] / len(matched) if modal_class is not None else 0.0
        ),
        "confidence_mean": float(np.mean(confidences)) if len(confidences) else None,
        "confidence_std": float(np.std(confidences)) if len(confidences) else None,
        "longest_consecutive_semantic_consistent_run": longest_run,
        "class_flips": class_flips,
        "matched_unmatched_transitions": matched_unmatched_transitions,
    }
