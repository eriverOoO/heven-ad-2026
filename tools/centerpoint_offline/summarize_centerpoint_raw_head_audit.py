#!/usr/bin/env python3
"""Decompose native/source-ring CenterPoint loss between R0 and the S1 gate."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np

from centerpoint_raw_head import (
    RawHeadSnapshot,
    classify_native_hit_sparse_miss,
    describe,
    gate_reason_summary,
    gt_neighborhood_evidence,
    heatmap_class_statistics,
    reproduce_gate,
    top_k_rows,
)
from evaluate_av2_xyzirc_pair import gated_hungarian_matches
from publish_av2_xyzirc import load_npz
from summarize_av2_stage_audit import (
    gt_for_timestamp,
    outputs_equivalent,
    points_in_box,
    stage_rows,
)


MODES = ("native", "source_ring_vlp16_v2")
DISTANCE_BINS = ((0.0, 20.0), (20.0, 40.0), (40.0, 60.0), (60.0, 80.0))
VEHICLE_LABEL = 1
VEHICLE_HEAD_CLASS = 0


def bin_name(distance_m: float) -> str | None:
    return next(
        (f"{int(low)}-{int(high)}m" for low, high in DISTANCE_BINS if low <= distance_m < high),
        None,
    )


def margin_bin(margin: float) -> str:
    if margin < -0.10:
        return "far_below"
    if margin < 0.0:
        return "near_below"
    if margin < 0.10:
        return "near_above"
    return "well_above"


def match_map(gt: list[dict], detections: list[dict], gate_m: float) -> dict[int, dict]:
    vehicles = [row for row in detections if row["label"] == VEHICLE_LABEL]
    return {
        gt_index: vehicles[det_index]
        for gt_index, det_index, _ in gated_hungarian_matches(gt, vehicles, gate_m)
    }


def load_frame(
    timestamp_ns: int,
    mode: str,
    run_root: Path,
    derived_root: Path,
) -> dict:
    run_dir = run_root / str(timestamp_ns) / mode
    stages = stage_rows(run_dir, expected_timestamp_ns=timestamp_ns)
    snapshot = RawHeadSnapshot.load(run_dir / "raw_head", timestamp_ns)
    gate = reproduce_gate(snapshot)
    s1_count = len(stages["post_score"])
    accepted = int(np.count_nonzero(gate.accepted))
    if accepted != s1_count:
        raise ValueError(
            f"{timestamp_ns}/{mode}: reproduced R0 gate accepted {accepted}, S1 has {s1_count}"
        )
    raw_scores = np.sort(gate.winner_score[gate.accepted].astype(np.float64))
    s1_scores = np.sort(np.asarray([row["score"] for row in stages["post_score"]]))
    if not np.allclose(raw_scores, s1_scores, atol=2e-5, rtol=2e-5):
        raise ValueError(f"{timestamp_ns}/{mode}: reproduced gate scores differ from S1")
    cloud, metadata = load_npz(derived_root / mode / f"{timestamp_ns}.npz")
    if int(metadata["timestamp_ns"]) != timestamp_ns:
        raise ValueError(f"{timestamp_ns}/{mode}: NPZ timestamp mismatch")
    return {"snapshot": snapshot, "gate": gate, "stages": stages, "cloud": cloud}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--reference-run-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gt-csv", type=Path, required=True)
    parser.add_argument("--top-k-jsonl", type=Path, required=True)
    parser.add_argument("--gate-m", type=float, default=3.0)
    parser.add_argument("--radius-cells", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    import pyarrow.feather as feather
    from scipy.stats import spearmanr

    annotation_columns = feather.read_table(args.annotations).to_pydict()
    timestamps = sorted(int(path.stem) for path in (args.derived_root / "native").glob("*.npz"))
    if not timestamps:
        raise ValueError("no native NPZ frames found")

    heatmap_stats: dict[str, list[dict]] = defaultdict(list)
    gate_totals = {mode: Counter() for mode in MODES}
    gate_by_class = {mode: defaultdict(Counter) for mode in MODES}
    gate_by_distance = {mode: defaultdict(Counter) for mode in MODES}
    vehicle_top_cells_by_distance = {
        mode: defaultdict(list) for mode in MODES
    }
    margin_totals = {mode: Counter() for mode in MODES}
    s1_scores = {mode: [] for mode in MODES}
    gt_rows: list[dict] = []
    outcome_counts = Counter()
    b_counts = Counter()
    b3_reason_counts = Counter()
    far_rows: list[dict] = []
    top_k_output: list[dict] = []
    dump_equivalence = Counter()

    for timestamp_ns in timestamps:
        gt = gt_for_timestamp(annotation_columns, timestamp_ns)
        frame = {
            mode: load_frame(timestamp_ns, mode, args.run_root, args.derived_root)
            for mode in MODES
        }
        if args.reference_run_root:
            for mode in MODES:
                reference = stage_rows(
                    args.reference_run_root / str(timestamp_ns) / mode,
                    expected_timestamp_ns=timestamp_ns,
                )
                for stage, raw_rows in frame[mode]["stages"].items():
                    if not outputs_equivalent(reference[stage], raw_rows):
                        raise ValueError(
                            f"dump OFF/ON output mismatch: {timestamp_ns}/{mode}/{stage}"
                        )
                    dump_equivalence[f"{mode}/{stage}"] += 1
        matches = {
            mode: {
                "s1": match_map(gt, frame[mode]["stages"]["post_score"], args.gate_m),
                "final": match_map(gt, frame[mode]["stages"]["final"], args.gate_m),
            }
            for mode in MODES
        }

        for mode in MODES:
            snapshot, gate = frame[mode]["snapshot"], frame[mode]["gate"]
            s1_scores[mode].extend(
                row["score"] for row in frame[mode]["stages"]["post_score"]
            )
            for row in heatmap_class_statistics(snapshot, gate):
                heatmap_stats[mode].append({"timestamp_ns": timestamp_ns, **row})
            summary = gate_reason_summary(snapshot, gate)
            for reason in ("accepted", "below_score", "outside_distance_bins", "invalid_yaw"):
                gate_totals[mode][reason] += summary[reason]
            for class_id, counts in summary["by_winner_class"].items():
                gate_by_class[mode][class_id].update(counts)
            inside = gate.distance_bucket < len(snapshot.metadata["distance_bin_upper_limits"])
            margins = gate.winner_score[inside] - gate.actual_threshold[inside]
            margin_totals[mode].update(margin_bin(float(value)) for value in margins)
            for distance_name, (low, high) in zip(
                ("0-20m", "20-40m", "40-60m", "60-80m"), DISTANCE_BINS
            ):
                mask = inside & (gate.radial_distance >= low) & (gate.radial_distance < high)
                gate_by_distance[mode][distance_name]["winner_cells"] += int(np.count_nonzero(mask))
                for reason_id, reason in enumerate(
                    ("accepted", "below_score", "outside_distance_bins", "invalid_yaw")
                ):
                    gate_by_distance[mode][distance_name][reason] += int(
                        np.count_nonzero(mask & (gate.reason == reason_id))
                    )
                values = gate.winner_score[mask]
                if len(values):
                    gate_by_distance[mode][distance_name]["score_sum"] += float(values.sum())
                    gate_by_distance[mode][distance_name]["max_score_sum"] += float(values.max())
                    gate_by_distance[mode][distance_name]["frames"] += 1
                vehicle_scores = gate.scores[VEHICLE_HEAD_CLASS][mask]
                count = min(100, len(vehicle_scores))
                if count:
                    selected = np.partition(vehicle_scores, -count)[-count:]
                    vehicle_top_cells_by_distance[mode][distance_name].extend(
                        float(value) for value in selected
                    )
                gate_by_distance[mode][distance_name]["vehicle_cells_above_0_35"] += int(
                    np.count_nonzero(vehicle_scores >= 0.35)
                )
            for row in top_k_rows(snapshot, gate, top_k=args.top_k):
                top_k_output.append({"timestamp_ns": timestamp_ns, "condition": mode, **row})

        for gt_index, actor in enumerate(gt):
            distance_m = math.hypot(actor["x"], actor["y"])
            evidences = {
                mode: gt_neighborhood_evidence(
                    frame[mode]["snapshot"], frame[mode]["gate"], actor["x"], actor["y"],
                    class_id=VEHICLE_HEAD_CLASS, radius_cells=args.radius_cells,
                )
                for mode in MODES
            }
            native_final = gt_index in matches["native"]["final"]
            corrected_final = gt_index in matches["source_ring_vlp16_v2"]["final"]
            outcome = (
                "A_detected_both" if native_final and corrected_final else
                "B_native_hit_corrected_miss" if native_final else
                "C_native_miss_corrected_hit" if corrected_final else
                "D_missed_both"
            )
            outcome_counts[outcome] += 1
            b_subcategory = None
            b3_reason = None
            if outcome == "B_native_hit_corrected_miss":
                b_subcategory = classify_native_hit_sparse_miss(
                    corrected_local_score=evidences["source_ring_vlp16_v2"]["score"],
                    corrected_threshold=evidences["source_ring_vlp16_v2"]["actual_threshold"],
                    native_margin=evidences["native"]["threshold_margin"],
                    corrected_s1_matched=(
                        gt_index in matches["source_ring_vlp16_v2"]["s1"]
                    ),
                    corrected_final_matched=False,
                )
                b_counts[b_subcategory] += 1
                if b_subcategory == "B3_above_threshold_without_S1_match":
                    corrected_evidence = evidences["source_ring_vlp16_v2"]
                    if corrected_evidence["winner_class"] != VEHICLE_HEAD_CLASS:
                        b3_reason = "class_competition"
                    elif corrected_evidence["gate_reason_at_cell"] == "invalid_yaw":
                        b3_reason = "invalid_yaw"
                    elif corrected_evidence["gate_accepted_at_cell"]:
                        b3_reason = "accepted_cell_without_GT_gated_S1_match"
                    else:
                        b3_reason = "other_gate_or_decode_condition"
                    b3_reason_counts[b3_reason] += 1
            row = {
                "timestamp_ns": timestamp_ns,
                "track_uuid": actor["track_uuid"],
                "distance_m": distance_m,
                "distance_bin": bin_name(distance_m),
                "native_points": points_in_box(frame["native"]["cloud"], actor),
                "corrected_points": points_in_box(frame["source_ring_vlp16_v2"]["cloud"], actor),
                "outcome": outcome,
                "b_subcategory": b_subcategory,
                "b3_reason": b3_reason,
                "native_s1_matched": gt_index in matches["native"]["s1"],
                "corrected_s1_matched": gt_index in matches["source_ring_vlp16_v2"]["s1"],
                "native_final_matched": native_final,
                "corrected_final_matched": corrected_final,
            }
            for mode, prefix in (("native", "native"), ("source_ring_vlp16_v2", "corrected")):
                evidence = evidences[mode]
                for field in (
                    "score",
                    "raw_logit",
                    "actual_threshold",
                    "threshold_margin",
                    "distance_from_gt_m",
                    "winner_class",
                    "winner_score",
                    "yaw_norm",
                    "decoded_x",
                    "decoded_y",
                    "decoded_z",
                    "decoded_length",
                    "decoded_width",
                    "decoded_height",
                    "decoded_yaw",
                    "decoded_vx",
                    "decoded_vy",
                    "gate_accepted_at_cell",
                    "gate_reason_at_cell",
                ):
                    row[f"{prefix}_{field}"] = evidence[field]
            gt_rows.append(row)
            if row["distance_bin"] == "60-80m":
                far_rows.append(row)

    corrected_points = np.asarray([row["corrected_points"] for row in gt_rows], dtype=float)
    corrected_scores = np.asarray([row["corrected_score"] for row in gt_rows], dtype=float)
    native_points = np.asarray([row["native_points"] for row in gt_rows], dtype=float)
    native_scores = np.asarray([row["native_score"] for row in gt_rows], dtype=float)
    retention = np.divide(
        corrected_points,
        native_points,
        out=np.zeros_like(corrected_points),
        where=native_points > 0,
    )
    score_delta = corrected_scores - native_scores
    point_score = spearmanr(corrected_points, corrected_scores)
    retention_delta = spearmanr(retention, score_delta)

    result = {
        "frame_count": len(timestamps),
        "gt_count": len(gt_rows),
        "modes": list(MODES),
        "raw_contract": {
            "R0": "dense TensorRT head tensors",
            "R1": "not independently materialized",
            "S1": "accepted cells compacted and score-sorted",
            "vehicle_head_class": VEHICLE_HEAD_CLASS,
            "gt_neighborhood_radius_cells": args.radius_cells,
            "gt_neighborhood_radius_m": 0.32 * args.radius_cells,
        },
        "dump_off_on_equivalence": {
            "reference_run_root": str(args.reference_run_root) if args.reference_run_root else None,
            "verified_frame_pairs": dict(dump_equivalence),
        },
        "heatmap_class_statistics": heatmap_stats,
        "gate": {
            mode: {
                "reason_counts": dict(gate_totals[mode]),
                "s1_count": len(s1_scores[mode]),
                "s1_score": describe(np.asarray(s1_scores[mode])),
                "reason_counts_by_winner_class": {
                    class_id: dict(values)
                    for class_id, values in gate_by_class[mode].items()
                },
                "threshold_margin_bins": dict(margin_totals[mode]),
                "by_distance": {
                    name: {
                        **dict(values),
                        "mean_winner_score": (
                            values["score_sum"] / values["winner_cells"]
                            if values["winner_cells"] else None
                        ),
                        "mean_frame_max_winner_score": (
                            values["max_score_sum"] / values["frames"]
                            if values["frames"] else None
                        ),
                        "vehicle_top100_cells_per_frame": describe(
                            np.asarray(vehicle_top_cells_by_distance[mode][name])
                        ),
                    }
                    for name, values in gate_by_distance[mode].items()
                },
            }
            for mode in MODES
        },
        "gt_outcomes": dict(outcome_counts),
        "native_hit_corrected_miss": dict(b_counts),
        "above_threshold_without_S1_reason": dict(b3_reason_counts),
        "gt_vehicle_heatmap_score": {
            mode: describe(np.asarray([
                row["native_score" if mode == "native" else "corrected_score"] for row in gt_rows
            ]))
            for mode in MODES
        },
        "distance_bins": {},
        "far_60_80m": {
            "gt": len(far_rows),
            "zero_corrected_points": sum(row["corrected_points"] == 0 for row in far_rows),
            "nonzero_corrected_points": sum(row["corrected_points"] > 0 for row in far_rows),
            "corrected_above_threshold": sum(
                row["corrected_threshold_margin"] >= 0 for row in far_rows
            ),
            "corrected_below_threshold": sum(
                row["corrected_threshold_margin"] < 0 for row in far_rows
            ),
        },
        "correlations": {
            "corrected_points_vs_local_heatmap_score_spearman": {
                "rho": float(point_score.statistic), "pvalue": float(point_score.pvalue)
            },
            "point_retention_vs_score_delta_spearman": {
                "rho": float(retention_delta.statistic), "pvalue": float(retention_delta.pvalue)
            },
        },
    }
    for name in ("0-20m", "20-40m", "40-60m", "60-80m"):
        rows = [row for row in gt_rows if row["distance_bin"] == name]
        result["distance_bins"][name] = {
            "gt": len(rows),
            "native_local_score": describe(np.asarray([row["native_score"] for row in rows])),
            "corrected_local_score": describe(np.asarray([row["corrected_score"] for row in rows])),
            "native_above_threshold": sum(row["native_threshold_margin"] >= 0 for row in rows),
            "corrected_above_threshold": sum(
                row["corrected_threshold_margin"] >= 0 for row in rows
            ),
            "native_final_hits": sum(row["native_final_matched"] for row in rows),
            "corrected_final_hits": sum(row["corrected_final_matched"] for row in rows),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.gt_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.gt_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(gt_rows[0]))
        writer.writeheader()
        writer.writerows(gt_rows)
    args.top_k_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.top_k_jsonl.open("w") as stream:
        for row in top_k_output:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
