"""Offline structural validator for a captured dataset.

Checks structure and internal consistency only - it makes no model or
accuracy judgement. Exit status is non-zero when any structural failure is
found.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ad_morai_bridge_dev.dataset.pointcloud import read_frame_npz
from ad_morai_bridge_dev.dataset.schema import (
    SCHEMA_VERSION,
    DatasetPaths,
    RUN_STATUS_COMPLETE,
)


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_runs: int = 0
    checked_frames: int = 0

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    @property
    def ok(self) -> bool:
        return not self.errors


def _load_json(path: Path, report: Report) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report.error(f"{path}: unreadable JSON ({exc})")
        return None


def _validate_frame_row(
    run_dir: Path, row: dict[str, Any], seen_ids: set[str], report: Report
) -> None:
    idx = row.get("frame_index")
    sample_id = row.get("sample_id")
    if row.get("schema_version") != SCHEMA_VERSION:
        report.error(f"{run_dir}: frame {idx} schema_version mismatch")
    if sample_id in seen_ids:
        report.error(f"{run_dir}: duplicate sample_id {sample_id}")
    seen_ids.add(sample_id)

    lidar_path = run_dir / row.get("lidar_filename", "")
    gt_path = run_dir / "gt" / f"{idx:06d}.json"
    for path in (lidar_path, gt_path, run_dir / "ego" / f"{idx:06d}.json",
                 run_dir / "tf" / f"{idx:06d}.json"):
        if not path.is_file():
            report.error(f"{run_dir}: frame {idx} missing {path.name}")
            return

    try:
        arrays = read_frame_npz(lidar_path)
    except (OSError, ValueError) as exc:
        report.error(f"{lidar_path}: unreadable npz ({exc})")
        return
    counts = {len(a) for a in arrays.values()}
    if len(counts) != 1:
        report.error(f"{lidar_path}: point field length mismatch {counts}")
    elif next(iter(counts)) != row.get("point_count"):
        report.error(
            f"{lidar_path}: point_count {next(iter(counts))} != index "
            f"{row.get('point_count')}"
        )
    for name, arr in arrays.items():
        if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
            report.error(f"{lidar_path}: field {name!r} has non-finite values")

    gt = _load_json(gt_path, report)
    if isinstance(gt, dict):
        for box in gt.get("boxes", []):
            for section in ("map_frame", "lidar_frame"):
                geom = box.get(section)
                if geom is None:
                    continue
                dims = [geom["length"], geom["width"], geom["height"], *geom["center"], geom["yaw"]]
                if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in dims):
                    report.error(f"{gt_path}: box {box.get('actor_id')} {section} non-finite")
                elif min(geom["length"], geom["width"], geom["height"]) <= 0.0:
                    report.error(f"{gt_path}: box {box.get('actor_id')} {section} non-positive dims")


def _validate_run(run_dir: Path, report: Report) -> dict[str, Any] | None:
    manifest = _load_json(run_dir / "run_manifest.json", report)
    if not isinstance(manifest, dict):
        return None
    report.checked_runs += 1
    if manifest.get("schema_version") != SCHEMA_VERSION:
        report.error(f"{run_dir}: run manifest schema_version mismatch")
    status = manifest.get("status")
    if status == "running":
        report.error(f"{run_dir}: run manifest still marked 'running' (not finalized)")
    elif status != RUN_STATUS_COMPLETE:
        report.warn(f"{run_dir}: run status is {status!r}")

    index_path = run_dir / "frames.jsonl"
    if not index_path.is_file():
        report.error(f"{run_dir}: frames.jsonl missing")
        return manifest
    rows = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    seen_ids: set[str] = set()
    prev_idx = -1
    prev_stamp = -1
    for row in rows:
        idx = row.get("frame_index")
        if idx != prev_idx + 1:
            report.error(f"{run_dir}: frame_index not contiguous at {idx}")
        prev_idx = idx
        stamp = row.get("lidar_header_stamp_ns", 0)
        if stamp <= prev_stamp:
            report.error(f"{run_dir}: LiDAR stamp not strictly increasing at frame {idx}")
        prev_stamp = stamp
        for src in ("ego", "actor_gt", "tf"):
            block = row.get(src, {})
            skew = block.get("skew_ns")
            if block.get("status") == "matched" and (skew is None or skew < 0):
                report.error(f"{run_dir}: frame {idx} {src} matched with bad skew {skew}")
        report.checked_frames += 1
        _validate_frame_row(run_dir, row, seen_ids, report)

    declared = manifest.get("summary", {}).get("captured_lidar_frames")
    if declared is not None and declared != len(rows):
        report.error(
            f"{run_dir}: manifest captured_lidar_frames {declared} != "
            f"{len(rows)} index rows"
        )
    for sub in ("lidar", "gt", "ego", "tf"):
        sub_dir = run_dir / sub
        on_disk = len(list(sub_dir.glob("*"))) if sub_dir.is_dir() else 0
        if on_disk != len(rows):
            report.error(
                f"{run_dir}: {sub}/ has {on_disk} files but {len(rows)} index rows "
                "(orphaned or missing artifact)"
            )
    return manifest


def validate_dataset(root: Path) -> Report:
    report = Report()
    paths = DatasetPaths(Path(root))
    if not (paths.root / DatasetPaths.MARKER).is_file():
        report.error(f"{paths.root}: not a dataset factory root (marker absent)")
        return report
    manifest = _load_json(paths.manifest, report)
    if isinstance(manifest, dict) and manifest.get("schema_version") != SCHEMA_VERSION:
        report.error("dataset_manifest.json schema_version mismatch")
    if not paths.schema.is_file():
        report.error("schema.json missing")
    if not paths.runs_dir.is_dir():
        report.error("runs/ directory missing")
        return report
    for scenario_dir in sorted(p for p in paths.runs_dir.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in scenario_dir.iterdir() if p.is_dir()):
            _validate_run(run_dir, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--json", action="store_true", help="emit a JSON report")
    args = parser.parse_args(argv)
    report = validate_dataset(args.dataset_root)
    if args.json:
        print(
            json.dumps(
                {
                    "ok": report.ok,
                    "checked_runs": report.checked_runs,
                    "checked_frames": report.checked_frames,
                    "errors": report.errors,
                    "warnings": report.warnings,
                },
                indent=2,
            )
        )
    else:
        for warning in report.warnings:
            print(f"WARN  {warning}")
        for error in report.errors:
            print(f"ERROR {error}")
        print(
            f"{'OK' if report.ok else 'FAIL'}: "
            f"{report.checked_runs} runs, {report.checked_frames} frames, "
            f"{len(report.errors)} errors, {len(report.warnings)} warnings"
        )
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
