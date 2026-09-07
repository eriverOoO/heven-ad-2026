"""Deterministic, dependency-light diagnostics for CenterPoint stability.

This deliberately uses a many-to-one GT candidate count in addition to the
usual one-to-one match, so duplicate detections cannot be hidden by matching.
Boxes are dictionaries with x/y/z/length/width/height/yaw, score and class.
"""
from __future__ import annotations

import math
from collections import defaultdict


DISTANCE_BINS = ((0.0, 20.0), (20.0, 40.0), (40.0, 60.0), (60.0, 80.0))


def distance_bin(distance: float) -> str:
    for lo, hi in DISTANCE_BINS:
        if lo <= distance < hi:
            return f"{int(lo)}-{int(hi)}m"
    return "80m+"


def center_distance(a: dict, b: dict) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def compatible(gt: dict, pred: dict) -> bool:
    # Historical T14's defensible GT domain is vehicle-only.  Do not silently
    # count a pedestrian/obstacle prediction as a vehicle duplicate.
    return str(gt.get("class_name", "vehicle")) == str(pred.get("class_label", pred.get("class_name", "vehicle")))


def filter_score(predictions: list[dict], threshold: float) -> list[dict]:
    return [p for p in predictions if float(p.get("score", 0.0)) >= threshold]


def one_to_one(gt: list[dict], predictions: list[dict], gate_m: float = 3.0) -> list[tuple[int, int, float]]:
    """GT-priority nearest available match; matches the historical protocol."""
    used: set[int] = set()
    matches = []
    for gi, g in enumerate(gt):
        choices = [(center_distance(g, p), pi) for pi, p in enumerate(predictions)
                   if pi not in used and compatible(g, p) and center_distance(g, p) <= gate_m]
        if choices:
            d, pi = min(choices)
            used.add(pi)
            matches.append((gi, pi, d))
    return matches


def candidate_counts(gt: list[dict], predictions: list[dict], gate_m: float = 3.0) -> list[int]:
    return [sum(compatible(g, p) and center_distance(g, p) <= gate_m for p in predictions) for g in gt]


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * p / 100.0
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def describe(values: list[float]) -> dict:
    return {"n": len(values), "mean": sum(values) / len(values) if values else None,
            "median": percentile(values, 50), "p90": percentile(values, 90),
            "p95": percentile(values, 95), "max": max(values) if values else None}


def temporal_residual_jitter(primary_rows: list[dict]) -> dict:
    """Absolute frame-to-frame change in prediction-minus-GT residual.

    Rows must contain actor_id, frame_idx, gt_x, gt_y, pred_x, pred_y and may
    contain score/yaw/length/width/height. Consecutive frame indices only.
    """
    by_actor = defaultdict(list)
    for row in primary_rows:
        by_actor[row["actor_id"]].append(row)
    center, score, yaw, size = [], [], [], []
    for rows in by_actor.values():
        rows.sort(key=lambda r: r["frame_idx"])
        for a, b in zip(rows, rows[1:]):
            if b["frame_idx"] != a["frame_idx"] + 1:
                continue
            ar = (a["pred_x"] - a["gt_x"], a["pred_y"] - a["gt_y"])
            br = (b["pred_x"] - b["gt_x"], b["pred_y"] - b["gt_y"])
            center.append(math.hypot(br[0] - ar[0], br[1] - ar[1]))
            if "score" in a and "score" in b:
                score.append(abs(b["score"] - a["score"]))
            if "yaw" in a and "yaw" in b:
                yaw.append(abs(math.atan2(math.sin(b["yaw"] - a["yaw"]), math.cos(b["yaw"] - a["yaw"]))))
            if "length" in a and "length" in b:
                size.append(abs(b["length"] - a["length"]))
    return {"center_residual_delta_m": describe(center), "confidence_delta": describe(score),
            "yaw_delta_rad": describe(yaw), "length_delta_m": describe(size)}
