"""Offline structural validator for an AV2 KalmanNet **sharded** dataset
export (``av2_kalmannet_shard_v1``).

Checks: format/shard-format version, shard index <-> shard-file agreement,
segment-offset validity (monotonic, no gaps, no overlap, in-bounds),
per-segment tensor shapes, per-segment finiteness, per-segment monotonic
timestamps / positive dt (checked per *segment slice*, never across the
whole concatenated shard), scenario uniqueness across shards, segment
uniqueness, scenario-level split leakage, coordinate/corruption-policy
metadata presence, content-fingerprint presence, and full manifest <->
on-disk-shard agreement (declared vs. actual segment/sample counts and
byte sizes). No estimator is run and no accuracy judgement is made.
Fails closed: any structural problem is an error, never silently accepted
as a warning. Non-zero exit on any structural failure.
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
    NPZ_SHARD_ARRAY_NAMES,
    NPZ_SHARD_PER_SEGMENT_ARRAYS,
    SHARD_FORMAT_VERSION,
    SUPPORTED_AV2_SPLITS,
    Av2KalmanNetExporter,
    check_scenario_split_leakage,
)


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_segments: int = 0
    checked_shards: int = 0

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
    if manifest.get("shard_format_version") != SHARD_FORMAT_VERSION:
        report.errors.append("shard_format_version mismatch")
    if manifest.get("split") not in SUPPORTED_AV2_SPLITS:
        report.errors.append(f"manifest split {manifest.get('split')!r} not a supported AV2 split")
    if manifest.get("training_ready") is not False:
        report.errors.append("export must declare training_ready = false (no training occurs here)")
    if manifest.get("corruption_materialized") is not False:
        report.errors.append(
            "canonical sharded export must declare corruption_materialized = false "
            "(corruption is a load-time transform, never baked into shards)"
        )

    coord_cfg = manifest.get("coordinate_config") or {}
    if "coordinate_mode" not in coord_cfg:
        report.errors.append("export_manifest.json missing coordinate_config.coordinate_mode")
    shard_cfg = manifest.get("shard_config")
    if not shard_cfg or "scenarios_per_shard" not in shard_cfg:
        report.errors.append("export_manifest.json missing shard_config.scenarios_per_shard")
    default_corruption_cfg = manifest.get("default_load_time_corruption_config")
    if default_corruption_cfg is None:
        report.errors.append("export_manifest.json missing default_load_time_corruption_config")
    elif "seed" not in default_corruption_cfg:
        report.errors.append("default_load_time_corruption_config missing 'seed'")

    prov = manifest.get("scenario_manifest_provenance") or {}
    for key in ("dataset_source", "bucket", "dataset_prefix"):
        if not prov.get(key):
            report.warnings.append(f"scenario_manifest_provenance missing/empty {key!r}")

    for name in ("metadata.json", "split_manifest.json", "shard_manifest.json", "shard_index.jsonl"):
        if not (root / name).is_file():
            report.errors.append(f"{name} missing")
    if report.errors:
        return report

    metadata = json.loads((root / "metadata.json").read_text("utf-8"))
    if metadata.get("output_format") != "sharded_npz_v1":
        report.errors.append("metadata.json output_format must be 'sharded_npz_v1'")
    if metadata.get("class_fed_to_kalmannet") is not False:
        report.errors.append("metadata.json must declare class_fed_to_kalmannet = false")
    if metadata.get("measurement_data_is_clean_only") is not True:
        report.errors.append("metadata.json must declare measurement_data_is_clean_only = true")
    if metadata.get("corruption_applied_at_load_time_only") is not True:
        report.errors.append("metadata.json must declare corruption_applied_at_load_time_only = true")
    if metadata.get("state_dim") != KNET_STATE_DIM or metadata.get("measurement_dim") != KNET_MEAS_DIM:
        report.errors.append("metadata.json state_dim/measurement_dim mismatch vs KNET_STATE_DIM/KNET_MEAS_DIM")

    split_manifest = json.loads((root / "split_manifest.json").read_text("utf-8"))
    declared_scenario_ids = split_manifest.get("scenario_ids") or []
    if len(declared_scenario_ids) != len(set(declared_scenario_ids)):
        report.errors.append("split_manifest.json scenario_ids contains duplicates")
    leakage = check_scenario_split_leakage({sid: split_manifest.get("split") for sid in declared_scenario_ids})
    for issue in leakage:
        report.errors.append(f"scenario split leakage: {issue}")
    if split_manifest.get("scenario_never_split_across_shards") is not True:
        report.errors.append("split_manifest.json must declare scenario_never_split_across_shards = true")

    shard_manifest = json.loads((root / "shard_manifest.json").read_text("utf-8"))
    if shard_manifest.get("shard_format_version") != SHARD_FORMAT_VERSION:
        report.errors.append("shard_manifest.json shard_format_version mismatch")
    shard_rows = shard_manifest.get("shards") or []
    if shard_manifest.get("shard_count") != len(shard_rows):
        report.errors.append(
            f"shard_manifest.json shard_count={shard_manifest.get('shard_count')} != {len(shard_rows)} shard rows"
        )

    # scenario uniqueness across shards (never split, never duplicated)
    scenario_shard_owner: dict[str, str] = {}
    for shard_row in shard_rows:
        for scenario_id in shard_row.get("scenario_ids", []):
            if scenario_id in scenario_shard_owner:
                report.errors.append(
                    f"scenario_id {scenario_id!r} appears in both shard "
                    f"{scenario_shard_owner[scenario_id]!r} and {shard_row['shard_file']!r}"
                )
            scenario_shard_owner[scenario_id] = shard_row["shard_file"]
    missing_scenarios = set(declared_scenario_ids) - set(scenario_shard_owner)
    if missing_scenarios:
        report.errors.append(
            f"scenario_ids declared in split_manifest but absent from any shard: {sorted(missing_scenarios)[:5]}"
        )

    # shard_index.jsonl: segment uniqueness + per-shard offset agreement
    index_rows: list[dict] = []
    seen_segment_ids: set[str] = set()
    for line in (root / "shard_index.jsonl").read_text("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = row["segment_id"]
        if sid in seen_segment_ids:
            report.errors.append(f"duplicate segment_id in shard_index.jsonl: {sid}")
        seen_segment_ids.add(sid)
        index_rows.append(row)

    if manifest.get("counts", {}).get("segments_exported") != len(index_rows):
        report.errors.append(
            f"manifest counts.segments_exported={manifest.get('counts', {}).get('segments_exported')} "
            f"!= {len(index_rows)} shard_index rows"
        )

    by_shard_file: dict[str, list[dict]] = {}
    for row in index_rows:
        by_shard_file.setdefault(row["shard_file"], []).append(row)

    declared_shard_files = {r["shard_file"] for r in shard_rows}
    index_shard_files = set(by_shard_file)
    if declared_shard_files != index_shard_files:
        report.errors.append(
            f"shard_manifest shard files {sorted(declared_shard_files)} != "
            f"shard_index.jsonl shard files {sorted(index_shard_files)}"
        )

    for shard_row in shard_rows:
        shard_file = shard_row["shard_file"]
        npz_path = root / shard_file
        if not npz_path.is_file():
            report.errors.append(f"{shard_file}: file missing on disk")
            continue
        declared_bytes = shard_row.get("byte_size")
        actual_bytes = npz_path.stat().st_size
        if declared_bytes is not None and declared_bytes != actual_bytes:
            report.errors.append(
                f"{shard_file}: declared byte_size={declared_bytes} != actual {actual_bytes}"
            )
        _validate_one_shard(root, shard_row, by_shard_file.get(shard_file, []), report)
        report.checked_shards += 1
        report.checked_segments += len(by_shard_file.get(shard_file, []))

    if not index_rows:
        report.errors.append("export contains zero segments")
    if not manifest.get("content_fingerprint"):
        report.errors.append("export_manifest.json missing content_fingerprint")

    return report


def _validate_one_shard(root: Path, shard_row: dict, index_rows: list[dict], report: Report) -> None:
    shard_file = shard_row["shard_file"]
    npz_path = root / shard_file
    with np.load(npz_path, allow_pickle=False) as data:
        keys = set(data.files)
        missing = [k for k in NPZ_SHARD_ARRAY_NAMES if k not in keys]
        if missing:
            report.errors.append(f"{shard_file}: npz lacks {missing}")
            return
        arrays = {name: data[name] for name in NPZ_SHARD_ARRAY_NAMES}

    offsets = arrays["segment_offsets"]
    num_segments = len(offsets) - 1
    if num_segments != shard_row.get("num_segments"):
        report.errors.append(
            f"{shard_file}: segment_offsets implies {num_segments} segments, "
            f"shard_manifest declares {shard_row.get('num_segments')}"
        )
    if num_segments != len(index_rows):
        report.errors.append(
            f"{shard_file}: {num_segments} segments in shard vs {len(index_rows)} shard_index.jsonl rows"
        )

    total_frames = int(arrays["timestamps_ns"].shape[0])
    if int(offsets[-1]) != total_frames:
        report.errors.append(
            f"{shard_file}: segment_offsets[-1]={int(offsets[-1])} != {total_frames} total per-frame rows"
        )
    if int(offsets[0]) != 0:
        report.errors.append(f"{shard_file}: segment_offsets[0] must be 0, got {int(offsets[0])}")
    if not np.all(np.diff(offsets) > 0):
        report.errors.append(
            f"{shard_file}: segment_offsets is not strictly increasing (a segment must have >=1 sample)"
        )
    if int(offsets[-1]) != shard_row.get("num_samples"):
        report.errors.append(
            f"{shard_file}: segment_offsets[-1]={int(offsets[-1])} != "
            f"declared num_samples={shard_row.get('num_samples')}"
        )

    for name in NPZ_SHARD_PER_SEGMENT_ARRAYS:
        if arrays[name].shape[0] != num_segments:
            report.errors.append(
                f"{shard_file}: per-segment array {name!r} has length {arrays[name].shape[0]} != {num_segments}"
            )

    state_data = arrays["state_data"]
    measurement_data = arrays["measurement_data"]
    if state_data.shape != (total_frames, KNET_STATE_DIM):
        report.errors.append(
            f"{shard_file}: state_data shape {state_data.shape} != ({total_frames}, {KNET_STATE_DIM})"
        )
    if measurement_data.shape != (total_frames, KNET_MEAS_DIM):
        report.errors.append(
            f"{shard_file}: measurement_data shape {measurement_data.shape} != ({total_frames}, {KNET_MEAS_DIM})"
        )
    if not np.all(np.isfinite(state_data)):
        report.errors.append(f"{shard_file}: state_data has non-finite values")

    valid = arrays["measurement_valid"].astype(bool)
    if np.any(np.isfinite(measurement_data[~valid])):
        report.errors.append(f"{shard_file}: finite measurement on a masked-out (measurement_valid=False) frame")
    if valid.any() and not np.all(np.isfinite(measurement_data[valid])):
        report.errors.append(f"{shard_file}: non-finite measurement on a valid frame")

    index_by_id = {row["segment_id"]: row for row in index_rows}
    for i in range(num_segments):
        segment_id = str(arrays["segment_id"][i])
        index_row = index_by_id.get(segment_id)
        if index_row is None:
            report.errors.append(f"{shard_file}: segment {segment_id!r} in shard but not in shard_index.jsonl")
            continue

        lo, hi = int(offsets[i]), int(offsets[i + 1])
        if index_row["offset_start"] != lo or index_row["offset_end"] != hi:
            report.errors.append(
                f"{segment_id}: shard_index offsets ({index_row['offset_start']}, {index_row['offset_end']}) "
                f"!= actual shard offsets ({lo}, {hi})"
            )
        if lo < 0 or hi > total_frames or lo >= hi:
            report.errors.append(f"{segment_id}: out-of-bounds/invalid slice [{lo}, {hi}) over {total_frames} rows")
            continue

        seg_count = index_row.get("sample_count")
        if seg_count != hi - lo:
            report.errors.append(f"{segment_id}: sample_count={seg_count} != slice length {hi - lo}")

        stamps = arrays["timestamps_ns"][lo:hi]
        dt = arrays["dt_s"][lo:hi]
        if hi - lo >= 1 and not math.isnan(float(dt[0])):
            report.errors.append(f"{segment_id}: dt_s[0] (per-segment) must be nan, got {dt[0]}")
        if hi - lo >= 2:
            if not np.all(stamps[1:] > stamps[:-1]):
                report.errors.append(f"{segment_id}: timestamps_ns not strictly increasing within the segment")
            if not np.all(np.isfinite(dt[1:])) or not np.all(dt[1:] > 0.0):
                report.errors.append(f"{segment_id}: dt_s[1:] (per-segment) not all finite positive")

        if index_row.get("coarse_object_type") not in COARSE_CLASSES:
            report.errors.append(
                f"{segment_id}: coarse_object_type {index_row.get('coarse_object_type')!r} not in {COARSE_CLASSES}"
            )
        if index_row.get("coordinate_mode") != "first_state_relative":
            report.errors.append(f"{segment_id}: unexpected coordinate_mode {index_row.get('coordinate_mode')!r}")
        if seg_count is not None and seg_count < 5 and index_row.get("valid_for_kalmannet_gt") is not False:
            report.errors.append(f"{segment_id}: short segment (<5 samples) must be flagged valid_for_kalmannet_gt=false")


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
                    "checked_shards": report.checked_shards,
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
        print(
            f"{'OK' if report.ok else 'FAIL'}: {report.checked_shards} shards, "
            f"{report.checked_segments} segments, {len(report.errors)} errors"
        )
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
