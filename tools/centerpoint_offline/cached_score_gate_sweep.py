#!/usr/bin/env python3
"""Faithful offline score-gate replay for cached Autoware CenterPoint R0 tensors.

This module is experimental evaluation code. It never edits runtime parameters and
does not invoke TensorRT. The constants below mirror the pinned 0.51.0 node config.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np

from centerpoint_raw_head import RawHeadSnapshot, gt_neighborhood_evidence, reproduce_gate
from evaluate_av2_xyzirc_pair import gated_hungarian_matches
from stage_postprocess import bev_iou
from summarize_av2_stage_audit import gt_for_timestamp, points_in_box, stage_rows
from publish_av2_xyzirc import load_npz


MODES = ("native", "source_ring_vlp16_v2")
PRIMARY_THRESHOLDS = (0.20, 0.225, 0.25, 0.275, 0.30, 0.325, 0.35, 0.375, 0.40, 0.45)
CAR_ONLY_THRESHOLDS = (0.20, 0.25, 0.275, 0.30, 0.325, 0.35)
CLASS_NAMES = ("CAR", "TRUCK", "BUS", "BICYCLE", "PEDESTRIAN")
SEMANTIC_LABELS = (1, 2, 3, 6, 7)
CAR_LABEL = 1
CIRCLE_NMS_DISTANCE_M = 0.5
IOU_SEARCH_DISTANCE_M = 10.0
IOU_THRESHOLD = 0.1
DISTANCE_BINS = ((0.0, 20.0), (20.0, 40.0), (40.0, 60.0), (60.0, 80.0))


def threshold_vector(class_size: int, threshold: float, policy: str) -> np.ndarray:
    """Return a fresh per-class vector; cached metadata is never mutated."""
    if policy == "global":
        return np.full(class_size, threshold, dtype=np.float32)
    if policy == "car_only":
        result = np.full(class_size, 0.35, dtype=np.float32)
        result[0] = threshold
        return result
    raise ValueError(f"unsupported threshold policy: {policy}")


def decode_s1(snapshot: RawHeadSnapshot, thresholds: np.ndarray) -> list[dict]:
    """Reproduce CUDA decode, validity gate, compaction, and score sorting."""
    gate = reproduce_gate(snapshot, thresholds)
    flat_indices = np.flatnonzero(gate.accepted.ravel())
    # CUDA starts with flat raster order and sorts descending by score. A stable
    # sort makes tied-score behavior deterministic for offline diagnostics.
    order = np.argsort(-gate.winner_score.ravel()[flat_indices], kind="stable")
    rows = []
    for flat_index in flat_indices[order]:
        gy, gx = divmod(int(flat_index), snapshot.width)
        raw_label = int(gate.winner_class[gy, gx])
        raw_yaw = float(math.atan2(snapshot.arrays["rot"][0, gy, gx], snapshot.arrays["rot"][1, gy, gx]))
        rows.append(
            {
                "source_index": int(flat_index),
                "raw_label": raw_label,
                "label": SEMANTIC_LABELS[raw_label],
                "class_name": CLASS_NAMES[raw_label],
                "score": float(gate.winner_score[gy, gx]),
                "x": float(gate.decoded_x[gy, gx]),
                "y": float(gate.decoded_y[gy, gx]),
                "z": float(snapshot.arrays["height"][0, gy, gx]),
                "length": float(math.exp(float(snapshot.arrays["dim"][1, gy, gx]))),
                "width": float(math.exp(float(snapshot.arrays["dim"][0, gy, gx]))),
                "height": float(math.exp(float(snapshot.arrays["dim"][2, gy, gx]))),
                # box3DToDetectedObject converts MMDet3D yaw to ROS yaw.
                "yaw": -raw_yaw - math.pi / 2.0,
                "vx": float(snapshot.arrays["vel"][0, gy, gx]),
                "vy": float(snapshot.arrays["vel"][1, gy, gx]),
            }
        )
    return rows


def circle_nms(rows: list[dict], distance_m: float = CIRCLE_NMS_DISTANCE_M) -> list[dict]:
    """Score-ordered, class-agnostic CenterPoint circle NMS (`distance < threshold`)."""
    distance_sq = distance_m * distance_m
    kept: list[dict] = []
    for row in rows:
        if all((row["x"] - prior["x"]) ** 2 + (row["y"] - prior["y"]) ** 2 >= distance_sq for prior in kept):
            kept.append(row)
    return kept


def iou_nms(rows: list[dict]) -> list[dict]:
    """Mirror pinned NonMaximumSuppression ordering and target-pair rules."""
    kept: list[dict] = []
    search_sq = IOU_SEARCH_DISTANCE_M * IOU_SEARCH_DISTANCE_M
    for index, row in enumerate(rows):
        suppress = False
        # The pinned implementation compares against every earlier score-ordered
        # input, including an earlier object that was itself suppressed.
        for prior in rows[:index]:
            if row["label"] != prior["label"] and (row["label"] == 7 or prior["label"] == 7):
                continue
            if (row["x"] - prior["x"]) ** 2 + (row["y"] - prior["y"]) ** 2 > search_sq:
                continue
            if bev_iou(row, prior) > IOU_THRESHOLD:
                suppress = True
                break
        if not suppress:
            kept.append(row)
    return kept


def remap_class_by_area(row: dict) -> dict:
    """Apply the pinned model package's nonzero class-remapping rules."""
    result = dict(row)
    area = result["length"] * result["width"]
    if result["label"] == 1:
        if 12.1 <= area <= 36.0:
            result["label"] = 2
        elif 36.0 <= area <= 999.999:
            result["label"] = 4
    elif result["label"] in (2, 3) and 36.0 <= area <= 999.999:
        result["label"] = 4
    return result


def offline_postprocess(snapshot: RawHeadSnapshot, threshold: float, policy: str) -> dict[str, list[dict]]:
    thresholds = threshold_vector(snapshot.class_size, threshold, policy)
    s1 = decode_s1(snapshot, thresholds)
    s2 = circle_nms(s1)
    s3 = list(s2)
    s4 = iou_nms(s3)
    s5 = [remap_class_by_area(row) for row in s4]
    return {"post_score": s1, "post_circle_nms": s2, "pre_iou": s3, "post_iou": s4, "final": s5}


def compare_rows(expected: list[dict], actual: list[dict], tolerance: float = 2e-4) -> dict:
    """Compare stage output by source-independent geometry matching."""
    if len(expected) != len(actual):
        return {"equal": False, "count_equal": False, "expected": len(expected), "actual": len(actual)}
    matches = gated_hungarian_matches(expected, actual, gate_m=tolerance)
    if len(matches) != len(expected):
        return {"equal": False, "count_equal": True, "matched": len(matches), "count": len(expected)}
    max_diffs = defaultdict(float)
    for ei, ai, center_diff in matches:
        left, right = expected[ei], actual[ai]
        max_diffs["center"] = max(max_diffs["center"], center_diff)
        if left["label"] != right["label"]:
            return {"equal": False, "count_equal": True, "label_mismatch": True}
        for field in ("score", "z", "length", "width", "height"):
            max_diffs[field] = max(max_diffs[field], abs(left[field] - right[field]))
        yaw_delta = abs(math.atan2(math.sin(left["yaw"] - right["yaw"]), math.cos(left["yaw"] - right["yaw"])))
        max_diffs["yaw"] = max(max_diffs["yaw"], yaw_delta)
    equal = all(value <= tolerance for value in max_diffs.values())
    return {"equal": equal, "count_equal": True, "count": len(expected), "max_diffs": dict(max_diffs)}


def evaluate_threshold(gt: list[dict], stages: dict[str, list[dict]], gate_m: float = 3.0) -> dict:
    final_all = stages["final"]
    vehicles = [row for row in final_all if row["label"] == CAR_LABEL]
    matches = gated_hungarian_matches(gt, vehicles, gate_m)
    errors = [distance for _, _, distance in matches]
    scores = [vehicles[di]["score"] for _, di, _ in matches]
    fp = len(vehicles) - len(matches)
    precision = len(matches) / len(vehicles) if vehicles else None
    recall = len(matches) / len(gt) if gt else None
    return {
        "gt": len(gt),
        "matched": len(matches),
        "vehicle_detections": len(vehicles),
        "all_detections": len(final_all),
        "false_positives": fp,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision and recall else 0.0,
        "mean_center_error_m": float(np.mean(errors)) if errors else None,
        "median_center_error_m": float(np.median(errors)) if errors else None,
        "mean_matched_score": float(np.mean(scores)) if scores else None,
        "matches": matches,
    }


def distance_bin(distance: float) -> str | None:
    for low, high in DISTANCE_BINS:
        if low <= distance < high:
            return f"{int(low)}-{int(high)}m"
    return None


def parse_thresholds(text: str) -> tuple[float, ...]:
    values = tuple(float(value) for value in text.split(",") if value.strip())
    if not values or any(not 0.0 < value < 1.0 for value in values):
        raise ValueError("thresholds must be comma-separated values strictly between zero and one")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--thresholds", default=",".join(str(x) for x in PRIMARY_THRESHOLDS))
    parser.add_argument("--car-only-thresholds", default=",".join(str(x) for x in CAR_ONLY_THRESHOLDS))
    parser.add_argument("--gate-m", type=float, default=3.0)
    args = parser.parse_args()

    import pyarrow.feather as feather

    thresholds = parse_thresholds(args.thresholds)
    car_only_thresholds = parse_thresholds(args.car_only_thresholds)
    annotation_columns = feather.read_table(args.annotations).to_pydict()
    timestamps = sorted(int(path.stem) for path in (args.derived_root / "native").glob("*.npz"))
    if not timestamps:
        raise ValueError("no cached frames found")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache: dict[tuple[int, str], dict] = {}
    reproduction: dict[str, dict[str, list[dict]]] = {mode: defaultdict(list) for mode in MODES}
    for timestamp in timestamps:
        gt = gt_for_timestamp(annotation_columns, timestamp)
        for mode in MODES:
            run_dir = args.run_root / str(timestamp) / mode
            snapshot = RawHeadSnapshot.load(run_dir / "raw_head", timestamp)
            reference = stage_rows(run_dir, expected_timestamp_ns=timestamp)
            baseline = offline_postprocess(snapshot, 0.35, "global")
            for stage in baseline:
                result = compare_rows(reference[stage], baseline[stage])
                reproduction[mode][stage].append({"timestamp_ns": timestamp, **result})
                if not result["equal"]:
                    raise ValueError(f"cached 0.35 reproduction failed: {timestamp}/{mode}/{stage}: {result}")
            native_cloud, _ = load_npz(args.derived_root / mode / f"{timestamp}.npz")
            cache[(timestamp, mode)] = {
                "snapshot": snapshot,
                "gt": gt,
                "point_counts": [points_in_box(native_cloud, actor) for actor in gt],
                "local_car_evidence": [
                    gt_neighborhood_evidence(snapshot, reproduce_gate(snapshot), actor["x"], actor["y"], class_id=0, radius_cells=5)
                    for actor in gt
                ],
            }

    policies = [("global", threshold) for threshold in thresholds] + [
        ("car_only", threshold) for threshold in car_only_thresholds
    ]
    aggregate_rows = []
    per_threshold: dict[str, dict] = {}
    per_gt_rows = []
    fp_rows = []
    distance_rows = []
    baseline_matches: dict[tuple[int, str], set[int]] = {}
    for timestamp in timestamps:
        for mode in MODES:
            stages = offline_postprocess(cache[(timestamp, mode)]["snapshot"], 0.35, "global")
            metrics = evaluate_threshold(cache[(timestamp, mode)]["gt"], stages, args.gate_m)
            baseline_matches[(timestamp, mode)] = {gi for gi, _, _ in metrics["matches"]}

    for policy, threshold in policies:
        key = f"{policy}/{threshold:g}"
        per_threshold[key] = {mode: {"frames": []} for mode in MODES}
        for mode in MODES:
            totals = defaultdict(float)
            all_errors, all_scores = [], []
            gt_bin = defaultdict(int)
            hit_bin = defaultdict(int)
            for timestamp in timestamps:
                item = cache[(timestamp, mode)]
                stages = offline_postprocess(item["snapshot"], threshold, policy)
                gate = reproduce_gate(
                    item["snapshot"],
                    threshold_vector(item["snapshot"].class_size, threshold, policy),
                )
                metrics = evaluate_threshold(item["gt"], stages, args.gate_m)
                match_by_gt = {gi: (di, error) for gi, di, error in metrics.pop("matches")}
                s1_vehicles = [row for row in stages["post_score"] if row["label"] == CAR_LABEL]
                s1_match_by_gt = {
                    gi: (di, error)
                    for gi, di, error in gated_hungarian_matches(item["gt"], s1_vehicles, args.gate_m)
                }
                final_vehicles = [row for row in stages["final"] if row["label"] == CAR_LABEL]
                frame_row = {
                    "timestamp_ns": timestamp,
                    "r0_score_pass_count": int(
                        np.count_nonzero((gate.reason == 0) | (gate.reason == 3))
                    ),
                    **{f"{stage}_count": len(rows) for stage, rows in stages.items()},
                    **metrics,
                }
                per_threshold[key][mode]["frames"].append(frame_row)
                for field in ("gt", "matched", "vehicle_detections", "all_detections", "false_positives"):
                    totals[field] += metrics[field]
                for gi, (di, error) in match_by_gt.items():
                    all_errors.append(error)
                    all_scores.append(final_vehicles[di]["score"])
                for gi, actor in enumerate(item["gt"]):
                    name = distance_bin(math.hypot(actor["x"], actor["y"]))
                    if name:
                        gt_bin[name] += 1
                        hit_bin[name] += int(gi in match_by_gt)
                    local_score = item["local_car_evidence"][gi]["score"]
                    final_match = match_by_gt.get(gi)
                    per_gt_rows.append(
                        {
                            "policy": policy,
                            "threshold": threshold,
                            "mode": mode,
                            "timestamp_ns": timestamp,
                            "track_uuid": actor["track_uuid"],
                            "distance_m": math.hypot(actor["x"], actor["y"]),
                            "points_in_box": item["point_counts"][gi],
                            "local_car_score": local_score,
                            "evidence_crosses_threshold": local_score >= threshold,
                            "baseline_matched": gi in baseline_matches[(timestamp, mode)],
                            "s1_matched": gi in s1_match_by_gt,
                            "matched": gi in match_by_gt,
                            "center_error_m": final_match[1] if final_match else None,
                            "final_score": final_vehicles[final_match[0]]["score"] if final_match else None,
                        }
                    )
                matched_det = {di for di, _ in match_by_gt.values()}
                vehicle_index = 0
                for row in stages["final"]:
                    is_supported_match = row["label"] == CAR_LABEL and vehicle_index in matched_det
                    if not is_supported_match:
                        fp_rows.append(
                            {
                                "policy": policy, "threshold": threshold, "mode": mode,
                                "timestamp_ns": timestamp, "class": row["class_name"],
                                "semantic_label": row["label"],
                                "distance_m": math.hypot(row["x"], row["y"]), "score": row["score"],
                                "is_supported_vehicle_fp": row["label"] == CAR_LABEL,
                            }
                        )
                    if row["label"] == CAR_LABEL:
                        vehicle_index += 1
            frame_count = len(timestamps)
            gt_total, matched = int(totals["gt"]), int(totals["matched"])
            vehicle_detections = int(totals["vehicle_detections"])
            summary = {
                "frames": frame_count,
                "gt": gt_total,
                "matched": matched,
                "recall": matched / gt_total,
                "vehicle_detections": vehicle_detections,
                "false_positives_total": int(totals["false_positives"]),
                "false_positives_per_frame": totals["false_positives"] / frame_count,
                "precision": matched / vehicle_detections if vehicle_detections else None,
                "f1": (2 * matched / (gt_total + vehicle_detections)) if (gt_total + vehicle_detections) else None,
                "mean_center_error_m": float(np.mean(all_errors)) if all_errors else None,
                "median_center_error_m": float(np.median(all_errors)) if all_errors else None,
                "mean_matched_score": float(np.mean(all_scores)) if all_scores else None,
            }
            for stage in ("post_score", "post_circle_nms", "pre_iou", "post_iou", "final"):
                counts = [row[f"{stage}_count"] for row in per_threshold[key][mode]["frames"]]
                summary[f"{stage}_total"] = sum(counts)
                summary[f"{stage}_mean_per_frame"] = float(np.mean(counts))
                summary[f"{stage}_max_per_frame"] = max(counts)
            score_pass_counts = [
                row["r0_score_pass_count"] for row in per_threshold[key][mode]["frames"]
            ]
            summary["r0_score_pass_total"] = sum(score_pass_counts)
            summary["r0_score_pass_mean_per_frame"] = float(np.mean(score_pass_counts))
            per_threshold[key][mode]["summary"] = summary
            aggregate_rows.append({"policy": policy, "threshold": threshold, "mode": mode, **summary})
            for name in ("0-20m", "20-40m", "40-60m", "60-80m"):
                distance_rows.append(
                    {"policy": policy, "threshold": threshold, "mode": mode, "distance_bin": name,
                     "gt": gt_bin[name], "matched": hit_bin[name],
                     "recall": hit_bin[name] / gt_bin[name] if gt_bin[name] else None}
                )

    baselines = {
        mode: next(
            row for row in aggregate_rows
            if row["policy"] == "global" and row["threshold"] == 0.35 and row["mode"] == mode
        )
        for mode in MODES
    }
    for row in aggregate_rows:
        baseline = baselines[row["mode"]]
        recovered = row["matched"] - baseline["matched"]
        row["additional_gt_recovered"] = recovered
        row["additional_fp"] = row["false_positives_total"] - baseline["false_positives_total"]
        row["additional_s1"] = row["post_score_total"] - baseline["post_score_total"]
        row["s1_multiplier_vs_035"] = row["post_score_total"] / baseline["post_score_total"]
        row["s5_multiplier_vs_035"] = row["final_total"] / baseline["final_total"]
        row["extra_fp_per_recovered_gt"] = row["additional_fp"] / recovered if recovered > 0 else None
        row["extra_s1_per_recovered_gt"] = row["additional_s1"] / recovered if recovered > 0 else None

    for policy in {row["policy"] for row in aggregate_rows}:
        for mode in MODES:
            ordered = sorted(
                (row for row in aggregate_rows if row["policy"] == policy and row["mode"] == mode),
                key=lambda row: row["threshold"],
            )
            counts = [row["post_score_total"] for row in ordered]
            if any(left < right for left, right in zip(counts, counts[1:])):
                raise ValueError(f"non-monotonic S1 candidate count for {policy}/{mode}")

    plot_tradeoffs(args.output_dir, aggregate_rows, distance_rows)

    def write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            path.write_text("")
            return
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(args.output_dir / "summary.csv", aggregate_rows)
    write_csv(args.output_dir / "per_gt_recovery.csv", per_gt_rows)
    write_csv(args.output_dir / "fp_breakdown.csv", fp_rows)
    write_csv(args.output_dir / "distance_tradeoff.csv", distance_rows)
    payload = {
        "provenance": {"run_root": str(args.run_root), "timestamps": timestamps, "r0_reused": True},
        "runtime_constants": {"circle_nms_distance_m": CIRCLE_NMS_DISTANCE_M, "iou_search_distance_m": IOU_SEARCH_DISTANCE_M, "iou_threshold": IOU_THRESHOLD},
        "baseline_reproduction": reproduction,
        "results": per_threshold,
    }
    (args.output_dir / "per_threshold.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"frames": len(timestamps), "output_dir": str(args.output_dir), "baseline_reproduced": True}, indent=2))
    return 0


def plot_tradeoffs(output_dir: Path, rows: list[dict], distance_rows: list[dict]) -> None:
    """Write compact diagnostic plots; these are not production tuning recommendations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = sorted(
        (row for row in rows if row["policy"] == "global" and row["mode"] == "source_ring_vlp16_v2"),
        key=lambda row: row["threshold"],
    )
    plots = (
        ("recall_vs_threshold.png", "recall", "Recall"),
        ("fp_per_frame_vs_threshold.png", "false_positives_per_frame", "Vehicle FP / frame"),
        ("s1_per_frame_vs_threshold.png", "post_score_mean_per_frame", "S1 candidates / frame"),
    )
    for filename, field, ylabel in plots:
        fig, axis = plt.subplots(figsize=(6.4, 4.0))
        axis.plot([row["threshold"] for row in selected], [row[field] for row in selected], marker="o")
        axis.axvline(0.35, color="gray", linestyle="--", label="runtime baseline")
        axis.set(xlabel="Offline score threshold", ylabel=ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=140)
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.plot([row["recall"] for row in selected], [row["precision"] for row in selected], marker="o")
    axis.set(xlabel="Recall", ylabel="Precision")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall.png", dpi=140)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.plot(
        [row["additional_gt_recovered"] for row in selected],
        [row["additional_fp"] for row in selected],
        marker="o",
    )
    axis.set(xlabel="Additional GT recovered vs 0.35", ylabel="Additional vehicle FP")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "recovered_gt_vs_extra_fp.png", dpi=140)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    for name in ("0-20m", "20-40m", "40-60m", "60-80m"):
        values = sorted(
            (row for row in distance_rows if row["policy"] == "global" and row["mode"] == "source_ring_vlp16_v2" and row["distance_bin"] == name),
            key=lambda row: row["threshold"],
        )
        axis.plot([row["threshold"] for row in values], [row["recall"] for row in values], marker="o", label=name)
    axis.axvline(0.35, color="gray", linestyle="--")
    axis.set(xlabel="Offline score threshold", ylabel="Recall")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "distance_recall_vs_threshold.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
