#!/usr/bin/env python3
"""Summarize the dual/up32/random/ring16 controlled replay.

This reader never invokes ROS or TensorRT.  It verifies one fixed threshold
and compares only saved stage dumps and R0 tensors from the controlled runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit_cached_regression_geometry import distance_bin, geometry_error
from centerpoint_raw_head import RawHeadSnapshot, decode_cell_geometry, gt_neighborhood_evidence, project_xy_to_grid, reproduce_gate
from evaluate_av2_xyzirc_pair import gated_hungarian_matches, roi_point_count
from publish_av2_xyzirc import load_npz
from summarize_av2_stage_audit import gt_for_timestamp, points_in_box, stage_rows


ARMS = {
    "A_native_dual64": ("native", "baseline"),
    "B_up_lidar_full32": ("up_lidar_full", "control"),
    "D_up_ring16": ("source_ring_vlp16_v2", "baseline"),
}
SEEDS = (11, 29, 47)
VEHICLE_LABEL = 1


def describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "median": None, "p25": None, "p75": None, "min": None, "max": None}
    data = np.asarray(values, dtype=np.float64)
    return {"count": len(data), "mean": float(data.mean()), "std": float(data.std()), "median": float(np.median(data)), "p25": float(np.percentile(data, 25)), "p75": float(np.percentile(data, 75)), "min": float(data.min()), "max": float(data.max())}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def run_dir_for(root: Path, timestamp: int, mode: str) -> Path:
    return root / str(timestamp) / mode


def vehicle_matches(gt: list[dict], final: list[dict], gate_m: float) -> list[tuple[int, int, float]]:
    vehicles = [row for row in final if row["label"] == VEHICLE_LABEL]
    return gated_hungarian_matches(gt, vehicles, gate_m)


def arm_rows(name: str, mode: str, run_root: Path, derived_root: Path, annotations: dict, gate_m: float) -> tuple[list[dict], list[dict]]:
    per_frame, per_gt = [], []
    for path in sorted((derived_root / "native").glob("*.npz")):
        timestamp = int(path.stem)
        cloud, metadata = load_npz(derived_root / mode / path.name)
        if int(metadata["timestamp_ns"]) != timestamp:
            raise ValueError(f"timestamp mismatch for {name}/{timestamp}")
        run_dir = run_dir_for(run_root, timestamp, mode)
        snapshot = RawHeadSnapshot.load(run_dir / "raw_head", timestamp)
        gate = reproduce_gate(snapshot)
        stages = stage_rows(run_dir, expected_timestamp_ns=timestamp)
        gt = gt_for_timestamp(annotations, timestamp)
        matches = vehicle_matches(gt, stages["final"], gate_m)
        matched = {index for index, _, _ in matches}
        final_vehicle = [row for row in stages["final"] if row["label"] == VEHICLE_LABEL]
        frame = {
            "arm": name, "mode": mode, "timestamp_ns": timestamp,
            "input_points": len(cloud), "roi_points": roi_point_count(cloud, (-76.8, -76.8, -4.0, 76.8, 76.8, 6.0)),
            "gt": len(gt), "matched": len(matches), "vehicle_fp": len(final_vehicle) - len(matches),
            "final_count": len(stages["final"]), "vehicle_final_count": len(final_vehicle),
            "mean_matched_center_error_m": float(np.mean([value for _, _, value in matches])) if matches else math.nan,
        }
        for stage in ("post_score", "post_circle_nms", "pre_iou", "post_iou", "final"):
            frame[f"{stage}_count"] = len(stages[stage])
        per_frame.append(frame)
        for index, actor in enumerate(gt):
            ref_x, ref_y = project_xy_to_grid(snapshot, actor["x"], actor["y"])
            evidence = gt_neighborhood_evidence(snapshot, gate, actor["x"], actor["y"], class_id=0, radius_cells=5)
            same = decode_cell_geometry(snapshot, gate, ref_x, ref_y)
            winner = decode_cell_geometry(snapshot, gate, evidence["grid_x"], evidence["grid_y"])
            same_error, winner_error = geometry_error(same, actor), geometry_error(winner, actor)
            scale_x = float(snapshot.metadata["voxel_size_x"]) * int(snapshot.metadata["downsample_factor"])
            scale_y = float(snapshot.metadata["voxel_size_y"]) * int(snapshot.metadata["downsample_factor"])
            peak_drift = math.hypot((winner["grid_x"] - ref_x) * scale_x, (winner["grid_y"] - ref_y) * scale_y)
            distance = math.hypot(actor["x"], actor["y"])
            per_gt.append({
                "arm": name, "mode": mode, "timestamp_ns": timestamp, "track_uuid": actor["track_uuid"],
                "distance_m": distance, "distance_bin": distance_bin(distance), "points_in_gt": points_in_box(cloud, actor),
                "r0_car_score": evidence["score"], "r0_score_ge_035": evidence["score"] >= 0.35,
                "final_matched": index in matched, "same_center_xy_error_m": same_error["center_xy_error_m"],
                "winner_center_xy_error_m": winner_error["center_xy_error_m"], "winner_dimension_relative_error": winner_error["dimension_relative_error_mean"],
                "winner_yaw_axis_error_rad": winner_error["yaw_axis_error_rad"], "winner_z_error_m": winner_error["z_error_m"],
                "winner_peak_drift_m": peak_drift,
            })
    return per_frame, per_gt


def aggregate(name: str, frame_rows: list[dict], gt_rows: list[dict]) -> dict:
    total_gt = sum(int(row["gt"]) for row in frame_rows)
    total_matched = sum(int(row["matched"]) for row in frame_rows)
    return {
        "arm": name, "frames": len(frame_rows), "gt": total_gt, "matched": total_matched,
        "recall": total_matched / total_gt if total_gt else None,
        "fp_per_frame": float(np.mean([row["vehicle_fp"] for row in frame_rows])),
        "input_points_per_frame": describe([row["input_points"] for row in frame_rows]),
        "roi_points_per_frame": describe([row["roi_points"] for row in frame_rows]),
        "s1_per_frame": describe([row["post_score_count"] for row in frame_rows]),
        "s2_per_frame": describe([row["post_circle_nms_count"] for row in frame_rows]),
        "s3_per_frame": describe([row["pre_iou_count"] for row in frame_rows]),
        "s4_per_frame": describe([row["post_iou_count"] for row in frame_rows]),
        "s5_per_frame": describe([row["final_count"] for row in frame_rows]),
        "r0_car_score": describe([row["r0_car_score"] for row in gt_rows]),
        "r0_ge_035": int(sum(bool(row["r0_score_ge_035"]) for row in gt_rows)),
        "winner_center_xy_error_m": describe([row["winner_center_xy_error_m"] for row in gt_rows]),
        "winner_peak_drift_m": describe([row["winner_peak_drift_m"] for row in gt_rows]),
        "winner_dimension_relative_error": describe([row["winner_dimension_relative_error"] for row in gt_rows]),
        "winner_yaw_axis_error_rad": describe([row["winner_yaw_axis_error_rad"] for row in gt_rows]),
        "winner_z_error_m": describe([row["winner_z_error_m"] for row in gt_rows]),
        "zero_point_fraction": float(np.mean([row["points_in_gt"] == 0 for row in gt_rows])),
    }


def flatten_summary(summary: dict) -> dict:
    row = {"arm": summary["arm"], "frames": summary["frames"], "gt": summary["gt"], "matched": summary["matched"], "recall": summary["recall"], "fp_per_frame": summary["fp_per_frame"], "r0_ge_035": summary["r0_ge_035"], "zero_point_fraction": summary["zero_point_fraction"]}
    for key in ("input_points_per_frame", "roi_points_per_frame", "s1_per_frame", "s2_per_frame", "s3_per_frame", "s4_per_frame", "s5_per_frame", "r0_car_score", "winner_center_xy_error_m", "winner_peak_drift_m", "winner_dimension_relative_error", "winner_yaw_axis_error_rad", "winner_z_error_m"):
        row[f"{key}_mean"] = summary[key]["mean"]
        row[f"{key}_std"] = summary[key]["std"]
    return row


def distance_rows(name: str, gt_rows: list[dict]) -> list[dict]:
    result = []
    for bucket in ("0-20m", "20-40m", "40-60m", "60-80m"):
        rows = [row for row in gt_rows if row["distance_bin"] == bucket]
        if not rows: continue
        result.append({"arm": name, "distance_bin": bucket, "gt": len(rows), "recall": float(np.mean([row["final_matched"] for row in rows])), "r0_score": describe([row["r0_car_score"] for row in rows]), "points_in_gt": describe([row["points_in_gt"] for row in rows]), "zero_point_fraction": float(np.mean([row["points_in_gt"] == 0 for row in rows])), "winner_center_error": describe([row["winner_center_xy_error_m"] for row in rows]), "winner_peak_drift": describe([row["winner_peak_drift_m"] for row in rows])})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--baseline-run-root", type=Path, required=True)
    parser.add_argument("--control-run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gate-m", type=float, default=3.0)
    args = parser.parse_args()
    import pyarrow.feather as feather
    annotations = feather.read_table(args.annotations).to_pydict()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_frames, all_gt, summaries, distance = [], [], [], []
    for arm, (mode, origin) in ARMS.items():
        frames, gt = arm_rows(arm, mode, args.baseline_run_root if origin == "baseline" else args.control_run_root, args.derived_root, annotations, args.gate_m)
        all_frames += frames; all_gt += gt
        summary = aggregate(arm, frames, gt); summaries.append(summary); distance += distance_rows(arm, gt)
    random_summaries = []
    for seed in SEEDS:
        arm, mode = f"C_up_random_matched_seed{seed}", f"up_random_matched_seed{seed}"
        frames, gt = arm_rows(arm, mode, args.control_run_root, args.derived_root, annotations, args.gate_m)
        all_frames += frames; all_gt += gt
        summary = aggregate(arm, frames, gt); random_summaries.append(summary); summaries.append(summary); distance += distance_rows(arm, gt)
    random_flat = [flatten_summary(row) for row in random_summaries]
    random_mean = {key: describe([float(row[key]) for row in random_flat if row[key] is not None]) for key in ("recall", "fp_per_frame", "s1_per_frame_mean", "r0_car_score_mean", "winner_center_xy_error_m_mean", "winner_peak_drift_m_mean")}
    write_csv(args.output_dir / "per_frame.csv", all_frames)
    write_csv(args.output_dir / "per_gt.csv", all_gt)
    write_csv(args.output_dir / "summary.csv", [flatten_summary(row) for row in summaries])
    write_csv(args.output_dir / "per_seed.csv", random_flat)
    write_csv(args.output_dir / "distance_summary.csv", [{"arm": row["arm"], "distance_bin": row["distance_bin"], "gt": row["gt"], "recall": row["recall"], "r0_score_mean": row["r0_score"]["mean"], "points_in_gt_median": row["points_in_gt"]["median"], "zero_point_fraction": row["zero_point_fraction"], "winner_center_error_mean": row["winner_center_error"]["mean"], "winner_peak_drift_mean": row["winner_peak_drift"]["mean"]} for row in distance])
    payload = {"threshold": 0.35, "gate_m": args.gate_m, "arms": summaries, "random_seed_summary": random_mean, "provenance": {"baseline_run_root": str(args.baseline_run_root), "control_run_root": str(args.control_run_root), "derived_root": str(args.derived_root)}}
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"arms": len(summaries), "frames": len(all_frames), "gt_rows": len(all_gt), "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
