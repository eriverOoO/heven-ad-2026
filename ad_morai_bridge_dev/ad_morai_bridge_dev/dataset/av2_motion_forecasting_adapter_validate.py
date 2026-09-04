"""Offline structural validator for an AV2 KalmanNet dataset export.

Checks the derived dataset against the ``av2_kalmannet_adapter_v1``
contract: schema/version, scenario count, segment count, split, scenario
leakage, tensor shapes, finiteness, monotonic timestamps, positive dt,
metadata presence, coordinate metadata, corruption seed/config presence,
the invalid-measurement representation (NaN + ``measurement_valid=False``,
never ``[0, 0]``, never a repeated previous value), and manifest/content
consistency. No estimator is run and no accuracy judgement is made.
Fails closed: any structural problem is an error, never a warning that
gets silently accepted. Non-zero exit on any structural failure.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ad_morai_bridge_dev.dataset.av2_motion_forecasting_adapter import (
    ADAPTER_SCHEMA_VERSION,
    COARSE_CLASSES,
    KNET_MEAS_DIM,
    KNET_STATE_DIM,
    NPZ_ARRAY_NAMES,
    SUPPORTED_AV2_SPLITS,
    Av2KalmanNetExporter,
    check_scenario_split_leakage,
)


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_segments: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_export(root: Path) -> Report:
    report = Report()
    root = Path(root)

    if not (root / Av2KalmanNetExporter.MARKER).is_file():
        report.errors.append(f"{root}: not an AV2 KalmanNet adapter export (marker absent)")
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
    if manifest.get("split") not in SUPPORTED_AV2_SPLITS:
        report.errors.append(f"manifest split {manifest.get('split')!r} not a supported AV2 split")
    if manifest.get("training_ready") is not False:
        report.errors.append("v0/Stage-0 export must declare training_ready = false (no training occurs here)")

    coord_cfg = manifest.get("coordinate_config") or {}
    if "coordinate_mode" not in coord_cfg:
        report.errors.append("export_manifest.json missing coordinate_config.coordinate_mode")
    corr_cfg = manifest.get("corruption_config")
    if corr_cfg is None:
        report.errors.append("export_manifest.json missing corruption_config")
    elif "seed" not in corr_cfg:
        report.errors.append("corruption_config missing 'seed' (corruption must be deterministic given a seed)")

    prov = manifest.get("scenario_manifest_provenance") or {}
    for key in ("dataset_source", "bucket", "dataset_prefix"):
        if not prov.get(key):
            report.warnings.append(f"scenario_manifest_provenance missing/empty {key!r}")

    for name in ("metadata.json", "split_manifest.json", "trajectory_index.jsonl"):
        if not (root / name).is_file():
            report.errors.append(f"{name} missing")

    metadata = {}
    if (root / "metadata.json").is_file():
        metadata = json.loads((root / "metadata.json").read_text("utf-8"))
        if metadata.get("class_fed_to_kalmannet") is not False:
            report.errors.append("metadata.json must declare class_fed_to_kalmannet = false")
        if metadata.get("state_dim") != KNET_STATE_DIM or metadata.get("measurement_dim") != KNET_MEAS_DIM:
            report.errors.append("metadata.json state_dim/measurement_dim mismatch vs KNET_STATE_DIM/KNET_MEAS_DIM")

    split_manifest = {}
    if (root / "split_manifest.json").is_file():
        split_manifest = json.loads((root / "split_manifest.json").read_text("utf-8"))
        declared_ids = split_manifest.get("scenario_ids") or []
        if len(declared_ids) != len(set(declared_ids)):
            report.errors.append("split_manifest.json scenario_ids contains duplicates")
        leakage = check_scenario_split_leakage({sid: split_manifest.get("split") for sid in declared_ids})
        for issue in leakage:
            report.errors.append(f"scenario split leakage: {issue}")

    index_rows: list[dict] = []
    seen_ids: set[str] = set()
    if (root / "trajectory_index.jsonl").is_file():
        for line in (root / "trajectory_index.jsonl").read_text("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row["segment_id"]
            if sid in seen_ids:
                report.errors.append(f"duplicate segment_id in trajectory_index.jsonl: {sid}")
            seen_ids.add(sid)
            index_rows.append(row)

    declared_scenario_ids = set(split_manifest.get("scenario_ids") or [])
    index_scenario_ids = {row["scenario_id"] for row in index_rows}
    missing_scenarios = index_scenario_ids - declared_scenario_ids
    if missing_scenarios:
        report.errors.append(
            f"trajectory_index.jsonl references scenario_ids not in split_manifest: {sorted(missing_scenarios)[:5]}"
        )

    if manifest.get("counts", {}).get("segments_exported") != len(index_rows):
        report.errors.append(
            f"manifest counts.segments_exported={manifest.get('counts', {}).get('segments_exported')} "
            f"!= {len(index_rows)} index rows"
        )

    for row in index_rows:
        _validate_segment(root, row, report)
        report.checked_segments += 1

    if not index_rows:
        report.errors.append("export contains zero segments")

    if not manifest.get("content_fingerprint"):
        report.errors.append("export_manifest.json missing content_fingerprint")
    for row in index_rows:
        npz_path = root / row.get("npz_path", "")
        if not npz_path.is_file() or npz_path.stat().st_size == 0:
            report.errors.append(f"{row.get('segment_id')}: referenced artifact missing/empty on disk: {npz_path}")

    return report


def _validate_segment(root: Path, row: dict, report: Report) -> None:
    sid = row["segment_id"]
    npz_path = root / row.get("npz_path", f"trajectories/{sid}.npz")
    if not npz_path.is_file():
        report.errors.append(f"{sid}: {npz_path} missing")
        return

    if row.get("coarse_object_type") not in COARSE_CLASSES:
        report.errors.append(f"{sid}: coarse_object_type {row.get('coarse_object_type')!r} not in {COARSE_CLASSES}")
    if not row.get("av2_object_type"):
        report.errors.append(f"{sid}: av2_object_type missing/empty")
    if "origin_x" not in row or "origin_y" not in row:
        report.errors.append(f"{sid}: origin_x/origin_y coordinate metadata missing from index row")
    if row.get("coordinate_mode") != "first_state_relative":
        report.errors.append(f"{sid}: unexpected coordinate_mode {row.get('coordinate_mode')!r}")

    with np.load(npz_path, allow_pickle=False) as data:
        keys = set(data.files)
        missing = [k for k in NPZ_ARRAY_NAMES if k not in keys]
        if missing:
            report.errors.append(f"{sid}: npz lacks {missing}")
            return

        stamps = data["timestamps_ns"]
        dt_s = data["dt_s"]
        gt_state = data["gt_state"]
        measurement = data["measurement"]
        valid = data["measurement_valid"].astype(bool)
        origin_xy = data["origin_xy"]
        count = stamps.shape[0]

        if count != row.get("sample_count"):
            report.errors.append(f"{sid}: npz T={count} != index sample_count {row.get('sample_count')}")
        if gt_state.shape != (count, KNET_STATE_DIM):
            report.errors.append(f"{sid}: gt_state shape {gt_state.shape} != ({count}, {KNET_STATE_DIM})")
        if measurement.shape != (count, KNET_MEAS_DIM):
            report.errors.append(f"{sid}: measurement shape {measurement.shape} != ({count}, {KNET_MEAS_DIM})")
        if origin_xy.shape != (2,):
            report.errors.append(f"{sid}: origin_xy shape {origin_xy.shape} != (2,)")
        if not np.all(np.isfinite(origin_xy)):
            report.errors.append(f"{sid}: origin_xy has non-finite values")

        if count >= 1 and not math.isnan(float(dt_s[0])):
            report.errors.append(f"{sid}: dt_s[0] must be nan, got {dt_s[0]}")
        if count >= 2:
            if not np.all(stamps[1:] > stamps[:-1]):
                report.errors.append(f"{sid}: timestamps_ns not strictly increasing")
            if not np.all(np.isfinite(dt_s[1:])) or not np.all(dt_s[1:] > 0.0):
                report.errors.append(f"{sid}: dt_s[1:] not all finite positive")

        if not np.all(np.isfinite(gt_state)):
            report.errors.append(f"{sid}: gt_state has non-finite values")

        # invalid-measurement representation: exactly NaN + valid=False,
        # never [0, 0], never a finite value on a masked-out frame.
        if np.any(np.isfinite(measurement[~valid])):
            report.errors.append(f"{sid}: finite measurement on a masked-out (measurement_valid=False) frame")
        if valid.any() and not np.all(np.isfinite(measurement[valid])):
            report.errors.append(f"{sid}: non-finite measurement on a valid frame")
        zero_rows = np.all(measurement == 0.0, axis=1) & valid
        if np.any(zero_rows):
            report.warnings.append(
                f"{sid}: {int(np.count_nonzero(zero_rows))} valid measurement rows are exactly [0, 0] "
                "(only expected if the AV2 first-state-relative origin itself lands on [0, 0], i.e. row 0)"
            )

        if row.get("sample_count", 0) < 5 and row.get("valid_for_kalmannet_gt") is not False:
            report.errors.append(f"{sid}: short segment (<5 samples) must be flagged valid_for_kalmannet_gt=false")


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
                    "checked_segments": report.checked_segments,
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
        print(f"{'OK' if report.ok else 'FAIL'}: {report.checked_segments} segments, {len(report.errors)} errors")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
