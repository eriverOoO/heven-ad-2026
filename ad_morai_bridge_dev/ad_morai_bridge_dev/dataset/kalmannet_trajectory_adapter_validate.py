"""Offline structural validator for a KalmanNet MORAI trajectory export.

Checks the derived dataset against the ``kalmannet_morai_trajectory_v1``
contract, re-checks split leakage from the written trajectory index, and
verifies every NPZ is time-consistent and finite. No estimator is run and
no accuracy judgement is made. Non-zero exit on any structural failure.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ad_morai_bridge_dev.dataset.kalmannet_trajectory_adapter import (
    ADAPTER_SCHEMA_VERSION,
    KNET_MEAS_DIM,
    KNET_STATE_DIM,
    NPZ_ARRAY_NAMES,
    KalmanNetTrajectoryExporter,
    check_split_leakage,
)


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_trajectories: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_export(root: Path) -> Report:
    report = Report()
    root = Path(root)
    if not (root / KalmanNetTrajectoryExporter.MARKER).is_file():
        report.errors.append(f"{root}: not an adapter export (marker absent)")
        return report
    manifest_path = root / "export_manifest.json"
    if not manifest_path.is_file():
        report.errors.append("export_manifest.json missing (export not finalized)")
        return report
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if manifest.get("status") != "complete":
        report.errors.append(f"export status is {manifest.get('status')!r}")
    if manifest.get("adapter_schema_version") != ADAPTER_SCHEMA_VERSION:
        report.errors.append("adapter_schema_version mismatch")
    if not manifest.get("source_dataset_manifest_sha256"):
        report.errors.append("source_dataset_manifest_sha256 absent")
    if manifest.get("measurements_present") is not False:
        report.errors.append("v1 export must declare measurements_present = false")
    if manifest.get("training_ready") is not False:
        report.errors.append("v1 export must declare training_ready = false")

    for name in ("metadata.json", "split_manifest.json", "trajectory_index.jsonl"):
        if not (root / name).is_file():
            report.errors.append(f"{name} missing")

    index_rows: dict[str, dict] = {}
    if (root / "trajectory_index.jsonl").is_file():
        for line in (root / "trajectory_index.jsonl").read_text("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            index_rows[row["trajectory_id"]] = row

    traj_to_split: dict[str, str] = {}
    traj_to_group: dict[str, str] = {}
    traj_to_run: dict[str, str] = {}
    split_counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        split_file = root / "splits" / f"{split}.txt"
        if not split_file.is_file():
            report.errors.append(f"splits/{split}.txt missing")
            continue
        ids = [
            line.strip()
            for line in split_file.read_text("utf-8").splitlines()
            if line.strip()
        ]
        split_counts[split] = len(ids)
        if len(set(ids)) != len(ids):
            report.errors.append(f"splits/{split}.txt has duplicate ids")
        for tid in ids:
            if tid in traj_to_split:
                report.errors.append(f"{tid} appears in multiple splits")
            traj_to_split[tid] = split
            row = index_rows.get(tid)
            if row is None:
                report.errors.append(f"{tid} missing from trajectory_index.jsonl")
                continue
            traj_to_group[tid] = str(row.get("group_id"))
            traj_to_run[tid] = str(row.get("run_id"))
            if row.get("split") != split:
                report.errors.append(
                    f"{tid} index split {row.get('split')!r} != {split!r}"
                )
            _validate_trajectory(root, tid, row, report)
            report.checked_trajectories += 1

    leakage = check_split_leakage(traj_to_split, traj_to_group, traj_to_run)
    for issue in leakage:
        report.errors.append(f"split leakage: {issue}")

    declared = manifest.get("split_trajectory_counts", {})
    for split, count in split_counts.items():
        if declared.get(split) != count:
            report.errors.append(
                f"manifest split_trajectory_counts[{split}]={declared.get(split)} "
                f"!= {count}"
            )
    if manifest.get("trajectory_count") != len(traj_to_split):
        report.errors.append(
            f"manifest trajectory_count={manifest.get('trajectory_count')} != "
            f"{len(traj_to_split)} split ids"
        )
    if not traj_to_split:
        report.errors.append("export contains zero trajectories")
    return report


def _validate_trajectory(root: Path, tid: str, row: dict, report: Report) -> None:
    npz_path = root / "trajectories" / f"{tid}.npz"
    if not npz_path.is_file():
        report.errors.append(f"trajectories/{tid}.npz missing")
        return
    with np.load(npz_path, allow_pickle=False) as data:
        keys = set(data.files)
        missing = [k for k in NPZ_ARRAY_NAMES if k not in keys]
        if missing:
            report.errors.append(f"{tid}: npz lacks {missing}")
            return
        stamps = data["timestamps_ns"]
        dt_s = data["dt_s"]
        gt_state = data["gt_state"]
        count = stamps.shape[0]
        if count != row.get("sample_count"):
            report.errors.append(
                f"{tid}: npz T={count} != index sample_count {row.get('sample_count')}"
            )
        if gt_state.shape != (count, KNET_STATE_DIM):
            report.errors.append(f"{tid}: gt_state shape {gt_state.shape}")
        if data["measurement"].shape != (count, KNET_MEAS_DIM):
            report.errors.append(f"{tid}: measurement shape {data['measurement'].shape}")
        if count >= 1 and not math.isnan(float(dt_s[0])):
            report.errors.append(f"{tid}: dt_s[0] must be nan, got {dt_s[0]}")
        if count >= 2:
            if not np.all(stamps[1:] > stamps[:-1]):
                report.errors.append(f"{tid}: timestamps_ns not strictly increasing")
            if not np.all(np.isfinite(dt_s[1:])) or not np.all(dt_s[1:] > 0.0):
                report.errors.append(f"{tid}: dt_s[1:] not all finite positive")
        if not np.all(np.isfinite(gt_state)):
            report.errors.append(f"{tid}: gt_state has non-finite values")
        if not np.all(np.isfinite(data["gt_velocity_map"])):
            report.errors.append(f"{tid}: gt_velocity_map has non-finite values")
        # v1: measurement slots entirely empty.
        if bool(np.any(data["measurement_valid"])):
            report.errors.append(f"{tid}: v1 must have no valid measurements")
        if not np.all(np.isnan(data["measurement"])):
            report.errors.append(f"{tid}: v1 measurement array must be all-nan")
        # measurement mask consistency (holds in v1 and after attachment).
        valid = data["measurement_valid"].astype(bool)
        meas = data["measurement"]
        if np.any(np.isfinite(meas[~valid])):
            report.errors.append(f"{tid}: finite measurement on a masked-out frame")
        if valid.any() and not np.all(np.isfinite(meas[valid])):
            report.errors.append(f"{tid}: non-finite measurement on a valid frame")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_root", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = validate_export(args.export_root)
    if args.json:
        print(
            json.dumps(
                {
                    "ok": report.ok,
                    "checked_trajectories": report.checked_trajectories,
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
            f"{'OK' if report.ok else 'FAIL'}: {report.checked_trajectories} "
            f"trajectories, {len(report.errors)} errors"
        )
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
