#!/usr/bin/env python3
"""Read and reproduce Autoware CenterPoint 0.51 raw-head gate semantics."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np


BINDING_SUFFIXES = {
    "heatmap": "heatmap.f32",
    "reg": "reg.f32",
    "height": "height.f32",
    "dim": "dim.f32",
    "rot": "rot.f32",
    "vel": "vel.f32",
}
REASON_NAMES = ("accepted", "below_score", "outside_distance_bins", "invalid_yaw")


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    output = np.empty_like(logits)
    positive = logits >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponential = np.exp(logits[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


@dataclass
class RawHeadSnapshot:
    metadata: dict
    arrays: dict[str, np.ndarray]

    @property
    def height(self) -> int:
        return int(self.metadata["height"])

    @property
    def width(self) -> int:
        return int(self.metadata["width"])

    @property
    def class_size(self) -> int:
        return int(self.metadata["class_size"])

    @classmethod
    def load(cls, raw_dir: Path, timestamp_ns: int) -> "RawHeadSnapshot":
        prefix = raw_dir / str(timestamp_ns)
        metadata = json.loads(Path(f"{prefix}_meta.json").read_text())
        if int(metadata["timestamp_ns"]) != timestamp_ns:
            raise ValueError("raw-head metadata timestamp mismatch")
        if metadata.get("dtype") != "float32" or metadata.get("layout") != "NCHW_without_batch":
            raise ValueError("unsupported raw-head dtype/layout")
        height, width = int(metadata["height"]), int(metadata["width"])
        arrays = {}
        for binding, suffix in BINDING_SUFFIXES.items():
            channels = int(metadata["bindings"][binding])
            path = Path(f"{prefix}_{suffix}")
            expected_bytes = channels * height * width * np.dtype(np.float32).itemsize
            if path.stat().st_size != expected_bytes:
                raise ValueError(
                    f"raw binding {binding} size mismatch: expected {expected_bytes}, "
                    f"got {path.stat().st_size}"
                )
            arrays[binding] = np.memmap(
                path, mode="r", dtype=np.float32, shape=(channels, height, width)
            )
        return cls(metadata=metadata, arrays=arrays)


@dataclass
class GateResult:
    scores: np.ndarray
    winner_class: np.ndarray
    winner_score: np.ndarray
    decoded_x: np.ndarray
    decoded_y: np.ndarray
    radial_distance: np.ndarray
    distance_bucket: np.ndarray
    actual_threshold: np.ndarray
    yaw_norm: np.ndarray
    accepted: np.ndarray
    reason: np.ndarray


def reproduce_gate(snapshot: RawHeadSnapshot) -> GateResult:
    """Reproduce generateBoxes3D_kernel decisions without changing inference."""
    scores = sigmoid(snapshot.arrays["heatmap"])
    winner_class = np.argmax(scores, axis=0).astype(np.int16)
    winner_score = np.take_along_axis(scores, winner_class[None, ...], axis=0)[0]
    yy, xx = np.indices((snapshot.height, snapshot.width), dtype=np.float32)
    scale_x = float(snapshot.metadata["voxel_size_x"]) * int(
        snapshot.metadata["downsample_factor"]
    )
    scale_y = float(snapshot.metadata["voxel_size_y"]) * int(
        snapshot.metadata["downsample_factor"]
    )
    decoded_x = scale_x * (xx + snapshot.arrays["reg"][0]) + float(
        snapshot.metadata["range_min_x"]
    )
    decoded_y = scale_y * (yy + snapshot.arrays["reg"][1]) + float(
        snapshot.metadata["range_min_y"]
    )
    radial_distance = np.hypot(decoded_x, decoded_y)
    upper = np.asarray(snapshot.metadata["distance_bin_upper_limits"], dtype=np.float32)
    distance_bucket = np.searchsorted(upper, radial_distance, side="right").astype(np.int16)
    inside = distance_bucket < len(upper)
    threshold_matrix = np.asarray(snapshot.metadata["score_thresholds"], dtype=np.float32).reshape(
        len(upper), snapshot.class_size
    )
    actual_threshold = np.full((snapshot.height, snapshot.width), np.nan, dtype=np.float32)
    actual_threshold[inside] = threshold_matrix[
        distance_bucket[inside], winner_class[inside]
    ]
    yaw_norm = np.hypot(snapshot.arrays["rot"][0], snapshot.arrays["rot"][1])
    yaw_thresholds = np.asarray(snapshot.metadata["yaw_norm_thresholds"], dtype=np.float32)
    score_pass = inside & (winner_score >= actual_threshold)
    yaw_pass = yaw_norm >= yaw_thresholds[winner_class]
    accepted = score_pass & yaw_pass
    reason = np.full((snapshot.height, snapshot.width), 2, dtype=np.uint8)
    reason[inside & ~score_pass] = 1
    reason[score_pass & ~yaw_pass] = 3
    reason[accepted] = 0
    return GateResult(
        scores=scores,
        winner_class=winner_class,
        winner_score=winner_score,
        decoded_x=decoded_x,
        decoded_y=decoded_y,
        radial_distance=radial_distance,
        distance_bucket=distance_bucket,
        actual_threshold=actual_threshold,
        yaw_norm=yaw_norm,
        accepted=accepted,
        reason=reason,
    )


def describe(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "p99_9": None,
        }
    return {
        "count": len(values),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "p99_9": float(np.percentile(values, 99.9)),
    }


def heatmap_class_statistics(snapshot: RawHeadSnapshot, gate: GateResult) -> list[dict]:
    upper = np.asarray(snapshot.metadata["distance_bin_upper_limits"], dtype=np.float32)
    thresholds = np.asarray(snapshot.metadata["score_thresholds"], dtype=np.float32).reshape(
        len(upper), snapshot.class_size
    )
    rows = []
    for class_id in range(snapshot.class_size):
        values = gate.scores[class_id]
        per_cell_threshold = np.full(values.shape, np.inf, dtype=np.float32)
        inside = gate.distance_bucket < len(upper)
        per_cell_threshold[inside] = thresholds[gate.distance_bucket[inside], class_id]
        rows.append(
            {
                "class_id": class_id,
                **describe(values.ravel()),
                "count_gt_0_10": int(np.count_nonzero(values > 0.10)),
                "count_gt_0_20": int(np.count_nonzero(values > 0.20)),
                "count_gt_0_30": int(np.count_nonzero(values > 0.30)),
                "count_ge_actual_threshold": int(
                    np.count_nonzero(values >= per_cell_threshold)
                ),
                "winner_cells": int(np.count_nonzero(gate.winner_class == class_id)),
                "accepted_cells": int(
                    np.count_nonzero(gate.accepted & (gate.winner_class == class_id))
                ),
            }
        )
    return rows


def top_k_rows(
    snapshot: RawHeadSnapshot, gate: GateResult, *, top_k: int = 100
) -> list[dict]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    rows = []
    for class_id in range(snapshot.class_size):
        scores = gate.scores[class_id].ravel()
        for eligible_name, eligible in (
            ("raw_class", np.ones(scores.shape, dtype=bool)),
            ("decoder_winner", (gate.winner_class == class_id).ravel()),
        ):
            indices = np.flatnonzero(eligible)
            count = min(top_k, len(indices))
            if not count:
                continue
            chosen = indices[np.argpartition(scores[indices], -count)[-count:]]
            chosen = chosen[np.argsort(scores[chosen])[::-1]]
            for rank, flat_index in enumerate(chosen, start=1):
                grid_y, grid_x = divmod(int(flat_index), snapshot.width)
                threshold = float(gate.actual_threshold[grid_y, grid_x])
                rows.append(
                    {
                        "class_id": class_id,
                        "kind": eligible_name,
                        "rank": rank,
                        "grid_x": grid_x,
                        "grid_y": grid_y,
                        "raw_logit": float(snapshot.arrays["heatmap"][class_id, grid_y, grid_x]),
                        "score": float(scores[flat_index]),
                        "decoded_x": float(gate.decoded_x[grid_y, grid_x]),
                        "decoded_y": float(gate.decoded_y[grid_y, grid_x]),
                        "distance_m": float(gate.radial_distance[grid_y, grid_x]),
                        "actual_winner_class": int(gate.winner_class[grid_y, grid_x]),
                        "actual_threshold": threshold if math.isfinite(threshold) else None,
                        "threshold_margin": (
                            float(scores[flat_index] - threshold)
                            if math.isfinite(threshold)
                            else None
                        ),
                        "yaw_norm": float(gate.yaw_norm[grid_y, grid_x]),
                        "gate_accepted": bool(gate.accepted[grid_y, grid_x]),
                        "gate_reason": REASON_NAMES[int(gate.reason[grid_y, grid_x])],
                    }
                )
    return rows


def project_xy_to_grid(snapshot: RawHeadSnapshot, x: float, y: float) -> tuple[int, int]:
    scale_x = float(snapshot.metadata["voxel_size_x"]) * int(
        snapshot.metadata["downsample_factor"]
    )
    scale_y = float(snapshot.metadata["voxel_size_y"]) * int(
        snapshot.metadata["downsample_factor"]
    )
    grid_x = math.floor((x - float(snapshot.metadata["range_min_x"])) / scale_x)
    grid_y = math.floor((y - float(snapshot.metadata["range_min_y"])) / scale_y)
    return grid_x, grid_y


def gt_neighborhood_evidence(
    snapshot: RawHeadSnapshot,
    gate: GateResult,
    x: float,
    y: float,
    *,
    class_id: int = 0,
    radius_cells: int = 5,
) -> dict:
    """Return the strongest class heatmap evidence within a bounded GT neighborhood."""
    center_x, center_y = project_xy_to_grid(snapshot, x, y)
    x0, x1 = max(0, center_x - radius_cells), min(snapshot.width, center_x + radius_cells + 1)
    y0, y1 = max(0, center_y - radius_cells), min(snapshot.height, center_y + radius_cells + 1)
    if x0 >= x1 or y0 >= y1:
        return {"in_grid": False, "grid_x": center_x, "grid_y": center_y}
    neighborhood = gate.scores[class_id, y0:y1, x0:x1]
    local_flat = int(np.argmax(neighborhood))
    local_y, local_x = np.unravel_index(local_flat, neighborhood.shape)
    grid_x, grid_y = x0 + int(local_x), y0 + int(local_y)
    score = float(gate.scores[class_id, grid_y, grid_x])
    distance_bucket = int(gate.distance_bucket[grid_y, grid_x])
    upper = snapshot.metadata["distance_bin_upper_limits"]
    if distance_bucket >= len(upper):
        threshold = None
        margin = None
    else:
        thresholds = np.asarray(snapshot.metadata["score_thresholds"]).reshape(
            len(upper), snapshot.class_size
        )
        threshold = float(thresholds[distance_bucket, class_id])
        margin = score - threshold
    return {
        "in_grid": True,
        "query_grid_x": center_x,
        "query_grid_y": center_y,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "radius_cells": radius_cells,
        "radius_m": radius_cells
        * float(snapshot.metadata["voxel_size_x"])
        * int(snapshot.metadata["downsample_factor"]),
        "raw_logit": float(snapshot.arrays["heatmap"][class_id, grid_y, grid_x]),
        "score": score,
        "decoded_x": float(gate.decoded_x[grid_y, grid_x]),
        "decoded_y": float(gate.decoded_y[grid_y, grid_x]),
        "decoded_z": float(snapshot.arrays["height"][0, grid_y, grid_x]),
        "decoded_length": float(math.exp(snapshot.arrays["dim"][1, grid_y, grid_x])),
        "decoded_width": float(math.exp(snapshot.arrays["dim"][0, grid_y, grid_x])),
        "decoded_height": float(math.exp(snapshot.arrays["dim"][2, grid_y, grid_x])),
        "decoded_yaw": float(
            math.atan2(
                snapshot.arrays["rot"][0, grid_y, grid_x],
                snapshot.arrays["rot"][1, grid_y, grid_x],
            )
        ),
        "decoded_vx": float(snapshot.arrays["vel"][0, grid_y, grid_x]),
        "decoded_vy": float(snapshot.arrays["vel"][1, grid_y, grid_x]),
        "distance_from_gt_m": float(
            math.hypot(gate.decoded_x[grid_y, grid_x] - x, gate.decoded_y[grid_y, grid_x] - y)
        ),
        "radial_distance_m": float(gate.radial_distance[grid_y, grid_x]),
        "actual_threshold": threshold,
        "threshold_margin": margin,
        "winner_class": int(gate.winner_class[grid_y, grid_x]),
        "winner_score": float(gate.winner_score[grid_y, grid_x]),
        "yaw_norm": float(gate.yaw_norm[grid_y, grid_x]),
        "gate_accepted_at_cell": bool(gate.accepted[grid_y, grid_x]),
        "gate_reason_at_cell": REASON_NAMES[int(gate.reason[grid_y, grid_x])],
    }


def gate_reason_summary(snapshot: RawHeadSnapshot, gate: GateResult) -> dict:
    result = {
        name: int(np.count_nonzero(gate.reason == index))
        for index, name in enumerate(REASON_NAMES)
    }
    result["total_cells"] = snapshot.height * snapshot.width
    result["by_winner_class"] = {
        str(class_id): {
            name: int(
                np.count_nonzero((gate.winner_class == class_id) & (gate.reason == index))
            )
            for index, name in enumerate(REASON_NAMES)
        }
        for class_id in range(snapshot.class_size)
    }
    return result


def classify_native_hit_sparse_miss(
    *,
    corrected_local_score: float,
    corrected_threshold: float,
    native_margin: float,
    corrected_s1_matched: bool,
    corrected_final_matched: bool,
) -> str:
    """Classify a native-final hit / corrected-final miss without forced pairing."""
    if corrected_final_matched:
        raise ValueError("classification requires a corrected final miss")
    if corrected_s1_matched:
        return "B4_lost_after_S1"
    corrected_margin = corrected_local_score - corrected_threshold
    if corrected_margin < 0.0:
        if native_margin >= 0.0 and corrected_margin < -0.10:
            return "B1_peak_strongly_weakened"
        return "B2_below_threshold"
    return "B3_above_threshold_without_S1_match"
