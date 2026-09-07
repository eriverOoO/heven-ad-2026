#!/usr/bin/env python3
"""Stream a saved CenterPoint JSONL run against MORAI labels for stability audit.

This is intentionally post-NMS: OpenPCDet's saved ``pred_boxes`` are already
postprocessed. It supports reproducible score-threshold sensitivity, but never
pretends to evaluate unavailable pre-NMS heatmap proposals.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from stability_metrics import (candidate_counts, center_distance, describe, distance_bin,
                               filter_score, one_to_one, temporal_residual_jitter)


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--thresholds", type=float, nargs="+", default=[.1, .15, .2, .3, .4, .5])
    p.add_argument("--gate-m", type=float, default=3.0)
    return p.parse_args()


def load_labels(root: Path, sample_id: str):
    payload = json.loads((root / "labels" / f"{sample_id}.json").read_text())
    # Historical experiment's valid class domain: vehicle.  Excluded actor IDs
    # 2/3 are non-vehicle despite inconsistent simulator categories.
    return [dict(b, class_name="vehicle") for b in payload["ground_truth"]["boxes"]
            if b.get("class_name") == "vehicle" and b.get("actor_id") not in {2, 3}]


def pred_box(d):
    b = d["box_lidar"]
    return {"x": b[0], "y": b[1], "z": b[2], "length": b[3], "width": b[4], "height": b[5],
            "yaw": b[6], "score": d["score"], "class_label": d["class_name"]}


def run(records, labels, threshold, gate):
    frame_rows, object_rows, primary = [], [], []
    scores_primary, scores_duplicate, scores_unmatched, errors, candidates = [], [], [], [], []
    total_gt = total_pred = total_match = 0
    for idx, rec in enumerate(records):
        gt = labels[idx]
        preds = filter_score([pred_box(d) for d in rec["detections"]], threshold)
        matches = one_to_one(gt, preds, gate)
        matched_p = {pi for _, pi, _ in matches}
        counts = candidate_counts(gt, preds, gate)
        total_gt += len(gt); total_pred += len(preds); total_match += len(matches)
        candidates.extend(counts)
        for gi, pi, d in matches:
            g, p = gt[gi], preds[pi]
            errors.append(d); scores_primary.append(p["score"])
            primary.append({"actor_id": g["actor_id"], "frame_idx": idx, "gt_x": g["x"], "gt_y": g["y"],
                            "pred_x": p["x"], "pred_y": p["y"], "score": p["score"], "yaw": p["yaw"], "length": p["length"]})
        for gi, (g, count) in enumerate(zip(gt, counts)):
            nearby = [p for p in preds if p["class_label"] == "vehicle" and center_distance(g, p) <= gate]
            if len(nearby) > 1:
                scores_duplicate.extend(sorted([p["score"] for p in nearby], reverse=True)[1:])
            object_rows.append({"frame_idx": idx, "sample_id": rec["sample_id"], "actor_id": g["actor_id"],
                                "range_bin": distance_bin((g["x"]**2 + g["y"]**2)**.5), "predictions_per_gt": count})
        scores_unmatched.extend(p["score"] for pi, p in enumerate(preds) if pi not in matched_p)
        frame_rows.append({"frame_idx": idx, "sample_id": rec["sample_id"], "gt_count": len(gt), "prediction_count": len(preds),
                           "matched_count": len(matches), "unmatched_count": len(preds) - len(matches)})
    ge2 = sum(v >= 2 for v in candidates); ge3 = sum(v >= 3 for v in candidates); ge5 = sum(v >= 5 for v in candidates)
    summary = {"threshold": threshold, "frames_evaluated": len(records), "gt_actor_frame_opportunities": total_gt,
               "detections": total_pred, "recall": total_match / total_gt if total_gt else None,
               "position_error_m": describe(errors), "predictions_per_gt": describe(candidates),
               "gt_frames_duplicate_rate_ge2": ge2 / total_gt if total_gt else None,
               "gt_frames_duplicate_rate_ge3": ge3 / total_gt if total_gt else None,
               "gt_frames_duplicate_rate_ge5": ge5 / total_gt if total_gt else None,
               "unmatched_detections_per_frame": (total_pred-total_match)/len(records) if records else None,
               "confidence": {"matched_primary": describe(scores_primary), "duplicates": describe(scores_duplicate), "unmatched": describe(scores_unmatched)},
               "temporal_residual_jitter": temporal_residual_jitter(primary),
               "raw_pre_nms_available": False, "note": "saved predictions are OpenPCDet post-NMS; score sweep is post-NMS only"}
    return summary, frame_rows, object_rows


def main():
    a = args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in a.predictions.open()]
    labels = [load_labels(a.labels, r["sample_id"]) for r in records]
    all_summaries = []
    for t in a.thresholds:
        summary, frames, objects = run(records, labels, t, a.gate_m); all_summaries.append(summary)
        tag = str(t).replace('.', '_')
        for name, rows in ((f"frames_{tag}.csv", frames), (f"objects_{tag}.csv", objects)):
            with (a.output_dir / name).open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else []); w.writeheader(); w.writerows(rows)
    (a.output_dir / "baseline_summary.json").write_text(json.dumps(all_summaries[0], indent=2) + "\n")
    (a.output_dir / "score_sweep_summary.json").write_text(json.dumps(all_summaries, indent=2) + "\n")
    print(json.dumps(all_summaries, indent=2))

if __name__ == "__main__": main()
