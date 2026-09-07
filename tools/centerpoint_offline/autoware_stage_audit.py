#!/usr/bin/env python3
"""Validate and summarize opt-in Autoware CenterPoint stage dumps.

The TensorRT node has no public topic for decoded, score-filtered, or
circle-NMS candidates.  This tool intentionally consumes only an explicit
debug dump from an instrumented node; it never infers an earlier stage from a
final ``DetectedObjects`` message.  That distinction prevents a final-output
recording from being misreported as a raw-model measurement.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from stability_metrics import candidate_counts, describe, one_to_one


STAGES = ("decoded", "score", "circle_nms", "iou_nms", "final")
REQUIRED_BOX_FIELDS = frozenset(
    {"candidate_id", "class_name", "score", "x", "y", "z", "length", "width", "height", "yaw"}
)


def _ids(candidates: list[dict[str, Any]]) -> set[int]:
    ids = [int(candidate["candidate_id"]) for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate ids must be unique within a stage")
    return set(ids)


def validate_stage_record(record: dict[str, Any]) -> None:
    """Reject incomplete or fabricated stage provenance deterministically."""
    if record.get("schema") != "heven.autoware_centerpoint_stage_dump.v1":
        raise ValueError("unexpected stage-dump schema")
    stages = record.get("stages")
    if not isinstance(stages, dict) or tuple(stages) != STAGES:
        raise ValueError(f"stages must be ordered exactly as {STAGES}")
    stage_ids: dict[str, set[int]] = {}
    for stage in STAGES:
        candidates = stages[stage]
        if not isinstance(candidates, list):
            raise ValueError(f"{stage} must be a list")
        for candidate in candidates:
            if not isinstance(candidate, dict) or not REQUIRED_BOX_FIELDS <= set(candidate):
                raise ValueError(f"{stage} candidate lacks required provenance/box fields")
        stage_ids[stage] = _ids(candidates)
    if not stage_ids["score"] <= stage_ids["decoded"]:
        raise ValueError("score candidates must originate from decoded candidates")
    if not stage_ids["circle_nms"] <= stage_ids["score"]:
        raise ValueError("circle-NMS candidates must originate from score candidates")
    if not stage_ids["iou_nms"] <= stage_ids["circle_nms"]:
        raise ValueError("IoU-NMS candidates must originate from circle-NMS candidates")
    if stage_ids["final"] != stage_ids["iou_nms"]:
        raise ValueError("final objects must preserve the IoU-NMS candidate set")


def stage_metrics(
    records: Iterable[dict[str, Any]], gt_by_frame: dict[int, list[dict[str, Any]]], *, gate_m: float = 3.0
) -> dict[str, dict[str, Any]]:
    """Calculate many-to-one duplicate and one-to-one recall metrics per stage."""
    total: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"frames": 0, "candidates": 0, "gt": 0, "matched": 0, "counts": [], "errors": [], "unmatched": 0}
    )
    for record in records:
        validate_stage_record(record)
        frame_idx = int(record["frame_idx"])
        gt = gt_by_frame.get(frame_idx, [])
        for stage in STAGES:
            predictions = record["stages"][stage]
            metrics = total[stage]
            matches = one_to_one(gt, predictions, gate_m)
            counts = candidate_counts(gt, predictions, gate_m)
            metrics["frames"] += 1
            metrics["candidates"] += len(predictions)
            metrics["gt"] += len(gt)
            metrics["matched"] += len(matches)
            metrics["counts"].extend(counts)
            metrics["errors"].extend(distance for _, _, distance in matches)
            metrics["unmatched"] += len(predictions) - len(matches)
    summary = {}
    for stage in STAGES:
        metrics = total[stage]
        gt = metrics["gt"]
        frames = metrics["frames"]
        summary[stage] = {
            "frames": frames,
            "candidates_per_frame": metrics["candidates"] / frames if frames else None,
            "predictions_per_gt": describe(metrics["counts"]),
            "duplicate_rate_ge2": sum(count >= 2 for count in metrics["counts"]) / gt if gt else None,
            "duplicate_rate_ge5": sum(count >= 5 for count in metrics["counts"]) / gt if gt else None,
            "recall": metrics["matched"] / gt if gt else None,
            "position_error_m": describe(metrics["errors"]),
            "unmatched_per_frame": metrics["unmatched"] / frames if frames else None,
        }
    return summary


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dump", type=Path, required=True)
    parser.add_argument("--gt-json", type=Path, required=True,
                        help="JSON mapping frame index strings to lists of lidar-frame GT boxes")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gt_by_frame = {int(key): value for key, value in json.loads(args.gt_json.read_text()).items()}
    summary = stage_metrics(_load_jsonl(args.stage_dump), gt_by_frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
