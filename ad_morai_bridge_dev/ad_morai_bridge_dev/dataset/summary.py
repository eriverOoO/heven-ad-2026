"""Offline dataset summary / inspector.

Reports counts, timing, point and GT-actor statistics. No model performance.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

from ad_morai_bridge_dev.dataset.schema import DatasetPaths


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p95": ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))],
        "max": ordered[-1],
    }


def _range_regime(distance_m: float) -> str:
    if distance_m <= 20.0:
        return "near"
    if distance_m <= 45.0:
        return "mid"
    return "far"


def summarize_dataset(root: Path) -> dict[str, Any]:
    paths = DatasetPaths(Path(root))
    runs: list[dict[str, Any]] = []
    point_counts: list[float] = []
    actors_per_frame: list[float] = []
    unique_ids: set[int] = set()
    class_counts: dict[str, int] = {}
    range_counts = {"near": 0, "mid": 0, "far": 0}
    invalid_sync = 0
    total_frames = 0
    tracking_frames = 0
    detection_frames = 0

    for scenario_dir in sorted(p for p in paths.runs_dir.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in scenario_dir.iterdir() if p.is_dir()):
            manifest = json.loads((run_dir / "run_manifest.json").read_text("utf-8"))
            rows = [
                json.loads(line)
                for line in (run_dir / "frames.jsonl")
                .read_text("utf-8")
                .splitlines()
                if line.strip()
            ]
            stamps = [r["lidar_header_stamp_ns"] for r in rows]
            duration_s = (max(stamps) - min(stamps)) / 1e9 if len(stamps) > 1 else 0.0
            runs.append(
                {
                    "scenario_id": manifest["scenario_id"],
                    "run_id": manifest["run_id"],
                    "status": manifest["status"],
                    "frames": len(rows),
                    "duration_s": round(duration_s, 3),
                    "termination_reason": manifest.get("termination_reason"),
                }
            )
            for row in rows:
                total_frames += 1
                point_counts.append(row["point_count"])
                actors_per_frame.append(row["actor_gt"]["actor_count"])
                unique_ids.update(row["actor_gt"]["actor_ids"])
                if row["valid_for_tracking_gt"]:
                    tracking_frames += 1
                if row["valid_for_detection_gt"]:
                    detection_frames += 1
                for src in ("ego", "actor_gt", "tf"):
                    if row[src]["status"] not in ("matched", None):
                        invalid_sync += 1
                gt = json.loads(
                    (run_dir / "gt" / f"{row['frame_index']:06d}.json").read_text("utf-8")
                )
                for box in gt.get("boxes", []):
                    name = str(box.get("class_name"))
                    class_counts[name] = class_counts.get(name, 0) + 1
                    geom = box.get("lidar_frame") or box.get("map_frame")
                    if geom:
                        d = math.hypot(geom["center"][0], geom["center"][1])
                        range_counts[_range_regime(d)] += 1

    return {
        "dataset_root": str(paths.root),
        "runs": runs,
        "total_runs": len(runs),
        "total_frames": total_frames,
        "valid_tracking_gt_frames": tracking_frames,
        "valid_detection_gt_frames": detection_frames,
        "invalid_sync_source_frames": invalid_sync,
        "point_count": _percentiles(point_counts),
        "gt_actors_per_frame": _percentiles(actors_per_frame),
        "unique_gt_actor_ids": sorted(unique_ids),
        "class_counts": dict(sorted(class_counts.items())),
        "range_distribution": range_counts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(summarize_dataset(args.dataset_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
