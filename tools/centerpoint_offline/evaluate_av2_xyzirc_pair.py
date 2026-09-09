#!/usr/bin/env python3
"""Evaluate paired native/adapted CenterPoint ROS outputs against AV2 GT.

This offline-only tool intentionally depends on PyYAML, PyArrow and SciPy. It
must run in the AV2/pyarrow-capable environment, never in the ROS system
Python used by ``publish_av2_xyzirc.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from publish_av2_xyzirc import load_npz


DEFAULT_POINT_CLOUD_RANGE = (-76.8, -76.8, -4.0, 76.8, 76.8, 6.0)


def detections_from_ros_dict(message: dict, label: int | None = 1) -> list[dict]:
    """Extract vehicle centers and scores from a DetectedObjects YAML dict."""
    result = []
    for obj in message.get("objects", []):
        classifications = obj.get("classification", [])
        if not classifications:
            continue
        object_label = int(classifications[0].get("label", -1))
        if label is not None and object_label != label:
            continue
        position = obj["kinematics"]["pose_with_covariance"]["pose"]["position"]
        result.append(
            {
                "x": float(position["x"]),
                "y": float(position["y"]),
                "z": float(position["z"]),
                "score": float(obj.get("existence_probability", 0.0)),
                "label": object_label,
            }
        )
    return result


def select_gt_rows(
    annotation_columns: dict[str, list],
    timestamp_ns: int,
    category: str = "REGULAR_VEHICLE",
    point_cloud_range: tuple[float, ...] = DEFAULT_POINT_CLOUD_RANGE,
) -> list[dict]:
    """Select exact-timestamp GT whose centers lie inside the model ROI."""
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    result = []
    row_count = len(annotation_columns["timestamp_ns"])
    for index in range(row_count):
        if int(annotation_columns["timestamp_ns"][index]) != timestamp_ns:
            continue
        if str(annotation_columns["category"][index]) != category:
            continue
        x = float(annotation_columns["tx_m"][index])
        y = float(annotation_columns["ty_m"][index])
        z = float(annotation_columns["tz_m"][index])
        if x_min <= x < x_max and y_min <= y < y_max and z_min <= z < z_max:
            result.append({"x": x, "y": y, "z": z})
    return result


def gated_hungarian_matches(
    gt: list[dict], detections: list[dict], gate_m: float
) -> list[tuple[int, int, float]]:
    """Maximum-cardinality, minimum-distance matching within a BEV gate."""
    if not gt or not detections:
        return []

    from scipy.optimize import linear_sum_assignment

    gt_xy = np.asarray([(row["x"], row["y"]) for row in gt], dtype=np.float64)
    det_xy = np.asarray([(row["x"], row["y"]) for row in detections], dtype=np.float64)
    distances = np.linalg.norm(gt_xy[:, None, :] - det_xy[None, :, :], axis=2)
    gt_count, det_count = distances.shape
    size = gt_count + det_count
    forbidden = 1.0e9
    unmatched = gate_m + 1.0
    cost = np.full((size, size), forbidden, dtype=np.float64)
    cost[:gt_count, :det_count] = np.where(distances <= gate_m, distances, forbidden)
    cost[np.arange(gt_count), det_count + np.arange(gt_count)] = unmatched
    cost[gt_count + np.arange(det_count), np.arange(det_count)] = unmatched
    cost[gt_count:, det_count:] = 0.0
    rows, columns = linear_sum_assignment(cost)
    return [
        (int(row), int(column), float(distances[row, column]))
        for row, column in zip(rows, columns)
        if row < gt_count and column < det_count and distances[row, column] <= gate_m
    ]


def evaluate(gt: list[dict], detections: list[dict], gate_m: float = 3.0) -> dict:
    matches = gated_hungarian_matches(gt, detections, gate_m)
    return {
        "gt": len(gt),
        "evaluated_detections": len(detections),
        "matched": len(matches),
        "recall": len(matches) / len(gt) if gt else None,
        "false_positives": len(detections) - len(matches),
        "mean_center_error_m": (
            float(np.mean([match[2] for match in matches])) if matches else None
        ),
        "mean_score": (
            float(np.mean([row["score"] for row in detections])) if detections else None
        ),
    }


def roi_point_count(cloud: np.ndarray, point_cloud_range: tuple[float, ...]) -> int:
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    keep = (
        (cloud["x"] >= x_min)
        & (cloud["x"] < x_max)
        & (cloud["y"] >= y_min)
        & (cloud["y"] < y_max)
        & (cloud["z"] >= z_min)
        & (cloud["z"] < z_max)
    )
    return int(np.count_nonzero(keep))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--timestamp-ns", type=int, required=True)
    parser.add_argument("--native-yaml", type=Path, required=True)
    parser.add_argument("--adapted-yaml", type=Path, required=True)
    parser.add_argument("--native-npz", type=Path, required=True)
    parser.add_argument("--adapted-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gate-m", type=float, default=3.0)
    args = parser.parse_args()

    import pyarrow.feather as feather
    import yaml

    annotation_columns = feather.read_table(args.annotations).to_pydict()
    gt = select_gt_rows(annotation_columns, args.timestamp_ns)
    native_messages = [row for row in yaml.safe_load_all(args.native_yaml.read_text()) if row]
    adapted_messages = [row for row in yaml.safe_load_all(args.adapted_yaml.read_text()) if row]
    if not native_messages or not adapted_messages:
        raise ValueError("each ROS YAML capture must contain at least one message")
    # The publisher deliberately repeats one immutable cloud so a subscriber
    # can connect. Evaluate the first complete response rather than combining
    # repeated callbacks into one synthetic frame.
    native_message = native_messages[0]
    adapted_message = adapted_messages[0]
    native_all_detections = detections_from_ros_dict(native_message, label=None)
    adapted_all_detections = detections_from_ros_dict(adapted_message, label=None)
    native_detections = [row for row in native_all_detections if row["label"] == 1]
    adapted_detections = [row for row in adapted_all_detections if row["label"] == 1]
    native_cloud, native_metadata = load_npz(args.native_npz)
    adapted_cloud, adapted_metadata = load_npz(args.adapted_npz)
    if native_metadata.get("timestamp_ns") != args.timestamp_ns:
        raise ValueError("native NPZ timestamp does not match --timestamp-ns")
    if adapted_metadata.get("timestamp_ns") != args.timestamp_ns:
        raise ValueError("adapted NPZ timestamp does not match --timestamp-ns")

    result = {
        "timestamp_ns": args.timestamp_ns,
        "coordinate_frame": "av2_egovehicle",
        "matching": {"method": "Hungarian BEV center distance", "gate_m": args.gate_m},
        "point_cloud_range": list(DEFAULT_POINT_CLOUD_RANGE),
        "native": {
            "captured_messages": len(native_messages),
            "input_points": len(native_cloud),
            "roi_points": roi_point_count(native_cloud, DEFAULT_POINT_CLOUD_RANGE),
            "final_detections_total": len(native_all_detections),
            "mean_final_score": (
                float(np.mean([row["score"] for row in native_all_detections]))
                if native_all_detections
                else None
            ),
            **evaluate(gt, native_detections, args.gate_m),
        },
        "vlp16_like": {
            "captured_messages": len(adapted_messages),
            "input_points": len(adapted_cloud),
            "roi_points": roi_point_count(adapted_cloud, DEFAULT_POINT_CLOUD_RANGE),
            "final_detections_total": len(adapted_all_detections),
            "mean_final_score": (
                float(np.mean([row["score"] for row in adapted_all_detections]))
                if adapted_all_detections
                else None
            ),
            **evaluate(gt, adapted_detections, args.gate_m),
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
