#!/usr/bin/env python3
"""GT-conditioned raw-regression audit over cached Autoware CenterPoint R0.

This is offline-only: it reads existing R0 snapshots and never invokes TensorRT
or changes a runtime threshold/configuration.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from centerpoint_raw_head import (
    RawHeadSnapshot,
    decode_cell_geometry,
    gt_neighborhood_evidence,
    project_xy_to_grid,
    reproduce_gate,
)
from evaluate_av2_xyzirc_pair import gated_hungarian_matches
from publish_av2_xyzirc import load_npz
from summarize_av2_stage_audit import gt_for_timestamp, points_in_box, stage_rows


MODES = ("native", "source_ring_vlp16_v2")
DISTANCE_BINS = ((0.0, 20.0), (20.0, 40.0), (40.0, 60.0), (60.0, 80.0))
VEHICLE_LABEL = 1


def wrapped_angle_difference(left: float, right: float) -> float:
    return abs(math.atan2(math.sin(left - right), math.cos(left - right)))


def yaw_axis_error(left: float, right: float) -> float:
    """Yaw error with front/rear ambiguity removed for a 3D box axis."""
    directional = wrapped_angle_difference(left, right)
    return min(directional, abs(math.pi - directional))


def quaternion_yaw(quaternion: tuple[float, float, float, float]) -> float:
    qw, qx, qy, qz = quaternion
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {key: None for key in ("count", "mean", "median", "p10", "p90", "max")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(array),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def distance_bin(distance: float) -> str | None:
    for low, high in DISTANCE_BINS:
        if low <= distance < high:
            return f"{int(low)}-{int(high)}m"
    return None


def geometry_error(prediction: dict, gt: dict) -> dict[str, float]:
    dx, dy, dz = prediction["x"] - gt["x"], prediction["y"] - gt["y"], prediction["z"] - gt["z"]
    gt_yaw = quaternion_yaw(gt["quaternion"])
    dimensions = ("length", "width", "height")
    result = {
        "center_xy_error_m": math.hypot(dx, dy),
        "center_3d_error_m": math.sqrt(dx * dx + dy * dy + dz * dz),
        "z_error_m": abs(dz),
        "yaw_directional_error_rad": wrapped_angle_difference(prediction["ros_yaw"], gt_yaw),
        "yaw_axis_error_rad": yaw_axis_error(prediction["ros_yaw"], gt_yaw),
    }
    relative = []
    for name in dimensions:
        absolute = abs(prediction[name] - gt[name])
        result[f"{name}_abs_error_m"] = absolute
        result[f"{name}_relative_error"] = absolute / gt[name] if gt[name] else math.nan
        relative.append(result[f"{name}_relative_error"])
    result["dimension_abs_error_l2_m"] = math.sqrt(
        sum(result[f"{name}_abs_error_m"] ** 2 for name in dimensions)
    )
    result["dimension_relative_error_mean"] = float(np.mean(relative))
    return result


def geometry_shift(native: dict, sparse: dict) -> dict[str, float]:
    return {
        "center_shift_xy_m": math.hypot(native["x"] - sparse["x"], native["y"] - sparse["y"]),
        "z_shift_m": abs(native["z"] - sparse["z"]),
        "dimension_shift_l2_m": math.sqrt(
            sum((native[name] - sparse[name]) ** 2 for name in ("length", "width", "height"))
        ),
        "yaw_axis_shift_rad": yaw_axis_error(native["ros_yaw"], sparse["ros_yaw"]),
        "velocity_shift_mps": math.hypot(native["vx"] - sparse["vx"], native["vy"] - sparse["vy"]),
    }


def final_match_map(gt: list[dict], rows: list[dict], gate_m: float) -> dict[int, dict]:
    vehicles = [row for row in rows if row["label"] == VEHICLE_LABEL]
    return {gt_index: vehicles[det_index] for gt_index, det_index, _ in gated_hungarian_matches(gt, vehicles, gate_m)}


def row_summary(rows: list[dict]) -> dict[str, dict]:
    fields = (
        "same_native_center_xy_error_m", "same_sparse_center_xy_error_m",
        "winner_native_center_xy_error_m", "winner_sparse_center_xy_error_m",
        "same_center_shift_xy_m", "winner_center_shift_xy_m",
        "winner_grid_displacement_m", "winner_residual_shift_xy_m",
        "same_dimension_shift_l2_m", "winner_dimension_shift_l2_m",
        "same_yaw_axis_shift_rad", "winner_yaw_axis_shift_rad",
        "same_z_shift_m", "winner_z_shift_m", "same_velocity_shift_mps",
    )
    return {field: describe([float(row[field]) for row in rows]) for field in fields}


def flat_row_summary(group_type: str, name: str, rows: list[dict]) -> dict[str, float | int | str | None]:
    """Emit the detailed summaries in a CSV-friendly, non-nested form."""
    result: dict[str, float | int | str | None] = {
        "group_type": group_type,
        "group": name,
        "count": len(rows),
    }
    for field, values in row_summary(rows).items():
        for statistic in ("mean", "median", "p90"):
            result[f"{field}_{statistic}"] = values[statistic]
    return result


def correlations(rows: list[dict]) -> list[dict]:
    from scipy.stats import spearmanr

    pairs = (
        ("sparse_score_vs_same_center_error", "sparse_score", "same_sparse_center_xy_error_m"),
        ("sparse_score_vs_winner_center_error", "sparse_score", "winner_sparse_center_xy_error_m"),
        ("sparse_score_vs_winner_dimension_error", "sparse_score", "winner_sparse_dimension_relative_error_mean"),
        ("sparse_score_vs_winner_yaw_axis_error", "sparse_score", "winner_sparse_yaw_axis_error_rad"),
        ("sparse_points_vs_same_center_error", "sparse_points", "same_sparse_center_xy_error_m"),
        ("sparse_points_vs_winner_center_error", "sparse_points", "winner_sparse_center_xy_error_m"),
        ("retention_vs_dimension_error_delta", "point_retention", "winner_dimension_error_delta"),
        ("retention_vs_yaw_error_delta", "point_retention", "winner_yaw_axis_error_delta"),
    )
    result = []
    for name, x_name, y_name in pairs:
        xy = [(float(row[x_name]), float(row[y_name])) for row in rows if math.isfinite(float(row[x_name])) and math.isfinite(float(row[y_name]))]
        if len(xy) < 3:
            result.append({"name": name, "count": len(xy), "rho": None, "pvalue": None})
            continue
        x, y = zip(*xy)
        value = spearmanr(x, y)
        result.append({"name": name, "count": len(xy), "rho": float(value.statistic), "pvalue": float(value.pvalue)})
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plots(output_dir: Path, rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sparse_center = [row["winner_sparse_center_xy_error_m"] for row in rows]
    native_center = [row["winner_native_center_xy_error_m"] for row in rows]
    fig, axis = plt.subplots(figsize=(6.0, 4.0))
    axis.scatter(native_center, sparse_center, c=[row["distance_m"] for row in rows], cmap="viridis")
    limit = max(native_center + sparse_center)
    axis.plot((0, limit), (0, limit), "--", color="gray")
    axis.set(xlabel="Native winner center error (m)", ylabel="Sparse winner center error (m)")
    fig.tight_layout(); fig.savefig(output_dir / "native_vs_sparse_center_error.png", dpi=140); plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.0, 4.0))
    axis.scatter([row["score_delta"] for row in rows], [row["winner_center_error_delta"] for row in rows])
    axis.axhline(0, color="gray", linewidth=.8); axis.axvline(0, color="gray", linewidth=.8)
    axis.set(xlabel="Sparse - native CAR score", ylabel="Sparse - native winner center error (m)")
    fig.tight_layout(); fig.savefig(output_dir / "score_delta_vs_center_error_delta.png", dpi=140); plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.0, 4.0))
    axis.scatter([row["sparse_points"] for row in rows], sparse_center)
    axis.set(xlabel="Sparse points / GT cuboid", ylabel="Sparse winner center error (m)")
    fig.tight_layout(); fig.savefig(output_dir / "sparse_points_vs_center_error.png", dpi=140); plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.0, 4.0))
    grouped = defaultdict(list)
    for row in rows: grouped[row["distance_bin"]].append(row["winner_grid_displacement_m"])
    names = [name for name in ("0-20m", "20-40m", "40-60m", "60-80m") if grouped[name]]
    axis.boxplot([grouped[name] for name in names], tick_labels=names)
    axis.set(ylabel="Native-to-sparse winner grid displacement (m)")
    fig.tight_layout(); fig.savefig(output_dir / "winner_grid_displacement_by_distance.png", dpi=140); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gate-m", type=float, default=3.0)
    parser.add_argument("--radius-cells", type=int, default=5)
    args = parser.parse_args()

    import pyarrow.feather as feather

    columns = feather.read_table(args.annotations).to_pydict()
    timestamps = sorted(int(path.stem) for path in (args.derived_root / "native").glob("*.npz"))
    if not timestamps:
        raise ValueError("no cached native NPZ files")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for timestamp in timestamps:
        gt = gt_for_timestamp(columns, timestamp)
        conditions = {}
        for mode in MODES:
            run_dir = args.run_root / str(timestamp) / mode
            snapshot = RawHeadSnapshot.load(run_dir / "raw_head", timestamp)
            gate = reproduce_gate(snapshot)
            final = stage_rows(run_dir, expected_timestamp_ns=timestamp)["final"]
            cloud, metadata = load_npz(args.derived_root / mode / f"{timestamp}.npz")
            if int(metadata["timestamp_ns"]) != timestamp:
                raise ValueError(f"NPZ timestamp mismatch for {timestamp}/{mode}")
            conditions[mode] = {
                "snapshot": snapshot, "gate": gate, "final": final, "cloud": cloud,
                "matches": final_match_map(gt, final, args.gate_m),
                "point_counts": [points_in_box(cloud, actor) for actor in gt],
            }
        native_snapshot = conditions["native"]["snapshot"]
        sparse_snapshot = conditions["source_ring_vlp16_v2"]["snapshot"]
        if (native_snapshot.height, native_snapshot.width) != (sparse_snapshot.height, sparse_snapshot.width):
            raise ValueError("paired R0 grids differ")
        for gt_index, actor in enumerate(gt):
            reference_x, reference_y = project_xy_to_grid(native_snapshot, actor["x"], actor["y"])
            if not (0 <= reference_x < native_snapshot.width and 0 <= reference_y < native_snapshot.height):
                continue
            decoded = {}
            for mode in MODES:
                condition = conditions[mode]
                evidence = gt_neighborhood_evidence(
                    condition["snapshot"], condition["gate"], actor["x"], actor["y"], class_id=0,
                    radius_cells=args.radius_cells,
                )
                decoded[mode] = {
                    "same": decode_cell_geometry(condition["snapshot"], condition["gate"], reference_x, reference_y),
                    "winner": decode_cell_geometry(condition["snapshot"], condition["gate"], evidence["grid_x"], evidence["grid_y"]),
                    "evidence": evidence,
                }
            native_same, sparse_same = decoded["native"]["same"], decoded["source_ring_vlp16_v2"]["same"]
            native_winner, sparse_winner = decoded["native"]["winner"], decoded["source_ring_vlp16_v2"]["winner"]
            same_native_error, same_sparse_error = geometry_error(native_same, actor), geometry_error(sparse_same, actor)
            winner_native_error, winner_sparse_error = geometry_error(native_winner, actor), geometry_error(sparse_winner, actor)
            same_shift, winner_shift = geometry_shift(native_same, sparse_same), geometry_shift(native_winner, sparse_winner)
            scale_x = float(native_snapshot.metadata["voxel_size_x"]) * int(native_snapshot.metadata["downsample_factor"])
            scale_y = float(native_snapshot.metadata["voxel_size_y"]) * int(native_snapshot.metadata["downsample_factor"])
            peak_dx = (sparse_winner["grid_x"] - native_winner["grid_x"]) * scale_x
            peak_dy = (sparse_winner["grid_y"] - native_winner["grid_y"]) * scale_y
            total_dx, total_dy = sparse_winner["x"] - native_winner["x"], sparse_winner["y"] - native_winner["y"]
            native_detected = gt_index in conditions["native"]["matches"]
            sparse_detected = gt_index in conditions["source_ring_vlp16_v2"]["matches"]
            cohort = "A_detected_both" if native_detected and sparse_detected else "B_native_hit_sparse_miss" if native_detected else "C_native_miss_sparse_hit" if sparse_detected else "D_missed_both"
            distance = math.hypot(actor["x"], actor["y"])
            native_points = conditions["native"]["point_counts"][gt_index]
            sparse_points = conditions["source_ring_vlp16_v2"]["point_counts"][gt_index]
            row = {
                "timestamp_ns": timestamp, "track_uuid": actor["track_uuid"], "cohort": cohort,
                "distance_m": distance, "distance_bin": distance_bin(distance),
                "native_points": native_points,
                "sparse_points": sparse_points,
                "native_score": decoded["native"]["evidence"]["score"],
                "sparse_score": decoded["source_ring_vlp16_v2"]["evidence"]["score"],
                "score_delta": decoded["source_ring_vlp16_v2"]["evidence"]["score"] - decoded["native"]["evidence"]["score"],
                "native_detected": native_detected, "sparse_detected": sparse_detected,
                "reference_grid_x": reference_x, "reference_grid_y": reference_y,
                "native_winner_grid_x": native_winner["grid_x"], "native_winner_grid_y": native_winner["grid_y"],
                "sparse_winner_grid_x": sparse_winner["grid_x"], "sparse_winner_grid_y": sparse_winner["grid_y"],
                "winner_peak_dx_m": peak_dx, "winner_peak_dy_m": peak_dy,
                "winner_grid_displacement_m": math.hypot(peak_dx, peak_dy),
                "winner_total_center_shift_xy_m": math.hypot(total_dx, total_dy),
                "winner_residual_shift_xy_m": math.hypot(total_dx - peak_dx, total_dy - peak_dy),
                "point_retention": sparse_points / native_points if native_points else math.nan,
            }
            for prefix, values in (
                ("same_native", same_native_error), ("same_sparse", same_sparse_error),
                ("winner_native", winner_native_error), ("winner_sparse", winner_sparse_error),
                ("same", same_shift), ("winner", winner_shift),
            ):
                for name, value in values.items(): row[f"{prefix}_{name}"] = value
            row["winner_dimension_error_delta"] = winner_sparse_error["dimension_relative_error_mean"] - winner_native_error["dimension_relative_error_mean"]
            row["winner_yaw_axis_error_delta"] = winner_sparse_error["yaw_axis_error_rad"] - winner_native_error["yaw_axis_error_rad"]
            row["winner_center_error_delta"] = winner_sparse_error["center_xy_error_m"] - winner_native_error["center_xy_error_m"]
            rows.append(row)

    if len(rows) != 89:
        raise ValueError(f"expected all 89 supported GT, got {len(rows)}")
    cohort_rows = []
    for cohort in ("A_detected_both", "B_native_hit_sparse_miss", "C_native_miss_sparse_hit", "D_missed_both"):
        group = [row for row in rows if row["cohort"] == cohort]
        cohort_rows.append(flat_row_summary("cohort", cohort, group))
    for name in ("0-20m", "20-40m", "40-60m", "60-80m"):
        group = [row for row in rows if row["distance_bin"] == name]
        cohort_rows.append(flat_row_summary("distance", name, group))
    correlation_rows = correlations(rows)
    write_csv(args.output_dir / "per_gt_geometry.csv", rows)
    write_csv(args.output_dir / "cohort_summary.csv", cohort_rows)
    write_csv(args.output_dir / "correlations.csv", correlation_rows)
    write_plots(args.output_dir, rows)
    payload = {
        "decoder_contract": {
            "reg": "x/y = voxel_size * downsample_factor * (grid + reg) + range_min",
            "height": "raw height head is decoded z",
            "dim": "raw [w, l, h], decoded [width, length, height] = exp(raw)",
            "rot": "raw yaw = atan2(rot[0]=sin, rot[1]=cos); ROS/AV2 comparison yaw = -raw_yaw - pi/2",
            "velocity": "raw [vel_x, vel_y] copied as diagnostic only",
        },
        "provenance": {"r0_reused": True, "run_root": str(args.run_root), "timestamps": timestamps, "gt_count": len(rows)},
        "cohort_counts": dict(Counter(row["cohort"] for row in rows)),
        "all_gt": row_summary(rows),
        "correlations": correlation_rows,
    }
    (args.output_dir / "head_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"gt": len(rows), "output_dir": str(args.output_dir), "r0_reused": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
