"""Offline structural validator for a KalmanNet measurement-attachment export.

Checks the ``kalmannet_morai_measurement_v1`` contract, that the GT arrays are
numerically identical to the source ``kalmannet_morai_trajectory_v1`` export,
and that the measurement mask is consistent. No estimator is run; no accuracy
judgement is made. Non-zero exit on any structural failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ad_morai_bridge_dev.dataset.kalmannet_measurement_attachment import (
    GT_ARRAY_NAMES,
    KNET_MEAS_DIM,
    KNET_STATE_DIM,
    MEASUREMENT_SCHEMA_VERSION,
    KalmanNetMeasurementExporter,
)


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_trajectories: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_export(root: Path, trajectory_root: Path | None = None) -> Report:
    report = Report()
    root = Path(root)
    if not (root / KalmanNetMeasurementExporter.MARKER).is_file():
        report.errors.append(f"{root}: not a measurement export (marker absent)")
        return report
    manifest_path = root / "export_manifest.json"
    if not manifest_path.is_file():
        report.errors.append("export_manifest.json missing (export not finalized)")
        return report
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if manifest.get("status") != "complete":
        report.errors.append(f"export status is {manifest.get('status')!r}")
    if manifest.get("measurement_schema_version") != MEASUREMENT_SCHEMA_VERSION:
        report.errors.append("measurement_schema_version mismatch")
    detector = manifest.get("detector", {})
    if not detector.get("manifest_sha256"):
        report.errors.append("detector.manifest_sha256 absent (no detector provenance)")
    if not detector.get("source"):
        report.errors.append("detector.source absent")
    if not manifest.get("source_trajectory_export", {}).get("content_fingerprint"):
        report.errors.append("source trajectory content_fingerprint not recorded")

    for name in ("metadata.json", "split_manifest.json", "trajectory_index.jsonl"):
        if not (root / name).is_file():
            report.errors.append(f"{name} missing")

    index_rows: dict[str, dict] = {}
    for line in (root / "trajectory_index.jsonl").read_text("utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            index_rows[row["trajectory_id"]] = row

    # split membership must match the source trajectory export verbatim
    if trajectory_root is not None:
        trajectory_root = Path(trajectory_root)
        for split in ("train", "val", "test"):
            here = _read_split(root, split)
            there = _read_split(trajectory_root, split)
            if here != there:
                report.errors.append(
                    f"splits/{split}.txt differs from the source trajectory export"
                )

    seen: set[str] = set()
    split_counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        ids = _read_split(root, split)
        split_counts[split] = len(ids)
        for tid in ids:
            if tid in seen:
                report.errors.append(f"{tid} appears in multiple splits")
            seen.add(tid)
            row = index_rows.get(tid)
            if row is None:
                report.errors.append(f"{tid} missing from trajectory_index.jsonl")
                continue
            _validate_trajectory(root, tid, row, report, trajectory_root)
            report.checked_trajectories += 1

    declared = manifest.get("split_trajectory_counts", {})
    for split, count in split_counts.items():
        if declared.get(split) != count:
            report.errors.append(
                f"manifest split_trajectory_counts[{split}]={declared.get(split)} != {count}"
            )
    if manifest.get("trajectory_count") != len(seen):
        report.errors.append(
            f"manifest trajectory_count={manifest.get('trajectory_count')} != {len(seen)}"
        )
    return report


def _read_split(root: Path, split: str) -> list[str]:
    path = root / "splits" / f"{split}.txt"
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text("utf-8").splitlines() if line.strip()]


def _validate_trajectory(
    root: Path,
    tid: str,
    row: dict,
    report: Report,
    trajectory_root: Path | None,
) -> None:
    npz_path = root / "trajectories" / f"{tid}.npz"
    if not npz_path.is_file():
        report.errors.append(f"trajectories/{tid}.npz missing")
        return
    with np.load(npz_path, allow_pickle=False) as data:
        count = int(data["timestamps_ns"].shape[0])
        meas = data["measurement"]
        valid = data["measurement_valid"].astype(bool)
        if meas.shape != (count, KNET_MEAS_DIM):
            report.errors.append(f"{tid}: measurement shape {meas.shape}")
        if data["measurement_state"].shape != (count, KNET_STATE_DIM):
            report.errors.append(f"{tid}: measurement_state shape")
        if valid.shape != (count,):
            report.errors.append(f"{tid}: measurement_valid shape {valid.shape}")
        # mask invariant: finite iff valid
        if np.any(np.isfinite(meas[~valid])):
            report.errors.append(f"{tid}: finite measurement on a masked-out frame")
        if valid.any() and not np.all(np.isfinite(meas[valid])):
            report.errors.append(f"{tid}: non-finite measurement on a valid frame")
        # KNet z_k is position-only: measurement_state velocity columns are nan
        if not np.all(np.isnan(data["measurement_state"][:, KNET_MEAS_DIM:])):
            report.errors.append(f"{tid}: measurement_state velocity columns must be nan")
        if int(valid.sum()) != row.get("measurement_count"):
            report.errors.append(
                f"{tid}: valid count {int(valid.sum())} != index measurement_count "
                f"{row.get('measurement_count')}"
            )
        # scores present iff valid
        score = data["measurement_score"]
        if np.any(np.isfinite(score[~valid])):
            report.errors.append(f"{tid}: measurement_score on a masked-out frame")

        # GT arrays byte/numeric identical to the source trajectory export
        if trajectory_root is not None:
            src_npz = Path(trajectory_root) / "trajectories" / f"{tid}.npz"
            if not src_npz.is_file():
                report.errors.append(f"{tid}: source trajectory npz missing for GT check")
            else:
                with np.load(src_npz, allow_pickle=False) as src:
                    for name in GT_ARRAY_NAMES:
                        a, b = data[name], src[name]
                        if a.shape != b.shape or not _array_equal_nan(a, b):
                            report.errors.append(f"{tid}: GT array {name!r} changed")


def _array_equal_nan(a: np.ndarray, b: np.ndarray) -> bool:
    if a.dtype.kind in ("U", "S"):
        return bool(np.array_equal(a, b))
    return bool(np.array_equal(a, b, equal_nan=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_root", type=Path)
    parser.add_argument(
        "--trajectory-root",
        type=Path,
        help="source kalmannet_morai_trajectory_v1 export, for GT-immutability + split checks",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = validate_export(args.export_root, args.trajectory_root)
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
