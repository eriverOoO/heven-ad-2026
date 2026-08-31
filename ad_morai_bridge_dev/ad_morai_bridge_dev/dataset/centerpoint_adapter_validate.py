"""Offline structural validator for a CenterPoint MORAI adapter export.

Verifies the derived dataset against the contract read by
``tools/centerpoint_offline/morai_dataset.py::MoraiHevenDatasetCore`` and
re-checks split leakage from the written labels. No model / accuracy
judgement. Non-zero exit on any structural failure.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ad_morai_bridge_dev.dataset.centerpoint_adapter import (
    ADAPTER_SCHEMA_VERSION,
    TARGET_BOX_FIELDS,
    TARGET_CLASS_NAMES,
    TARGET_GT_FRAME,
    TARGET_POINT_FEATURES,
    CenterPointExporter,
    check_split_leakage,
)


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_samples: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_export(root: Path) -> Report:
    report = Report()
    root = Path(root)
    if not (root / CenterPointExporter.MARKER).is_file():
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

    for name in ("metadata.json", "split_manifest.json", "sample_mapping.jsonl"):
        if not (root / name).is_file():
            report.errors.append(f"{name} missing")

    mapping_ids = set()
    if (root / "sample_mapping.jsonl").is_file():
        for line in (root / "sample_mapping.jsonl").read_text("utf-8").splitlines():
            if line.strip():
                mapping_ids.add(json.loads(line)["target_sample_id"])

    frame_to_split: dict[str, str] = {}
    frame_to_group: dict[str, str] = {}
    frame_to_run: dict[str, str] = {}
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
        for sample_id in ids:
            if sample_id in frame_to_split:
                report.errors.append(f"{sample_id} appears in multiple splits")
            frame_to_split[sample_id] = split
            if sample_id not in mapping_ids:
                report.errors.append(f"{sample_id} missing from sample_mapping.jsonl")
            _validate_sample(root, sample_id, report, frame_to_group, frame_to_run)
            report.checked_samples += 1

    leakage = check_split_leakage(frame_to_split, frame_to_group, frame_to_run)
    for issue in leakage:
        report.errors.append(f"split leakage: {issue}")

    declared = manifest.get("split_frame_counts", {})
    for split, count in split_counts.items():
        if declared.get(split) != count:
            report.errors.append(
                f"manifest split_frame_counts[{split}]={declared.get(split)} != {count}"
            )
    if manifest.get("sample_count") != len(frame_to_split):
        report.errors.append(
            f"manifest sample_count={manifest.get('sample_count')} != "
            f"{len(frame_to_split)} split ids"
        )
    return report


def _validate_sample(
    root: Path,
    sample_id: str,
    report: Report,
    frame_to_group: dict[str, str],
    frame_to_run: dict[str, str],
) -> None:
    label_path = root / "labels" / f"{sample_id}.json"
    bin_path = root / "points" / f"{sample_id}.bin"
    if not label_path.is_file():
        report.errors.append(f"labels/{sample_id}.json missing")
        return
    if not bin_path.is_file():
        report.errors.append(f"points/{sample_id}.bin missing")
        return
    label = json.loads(label_path.read_text("utf-8"))
    if label.get("sample_id") != sample_id:
        report.errors.append(f"{label_path}: sample_id mismatch")
    src = label.get("source", {})
    frame_to_group[sample_id] = (
        f"{src.get('scenario_id')}::seed_{src.get('requested_seed')}"
        if src.get("requested_seed") is not None
        else f"{src.get('run_id')}::{src.get('run_id')}"
    )
    frame_to_run[sample_id] = str(src.get("run_id"))

    points_info = label.get("points", {})
    if list(points_info.get("feature_names", [])) != list(TARGET_POINT_FEATURES):
        report.errors.append(f"{label_path}: point feature_names != {TARGET_POINT_FEATURES}")
    raw = np.fromfile(bin_path, dtype="<f4")
    if raw.size % len(TARGET_POINT_FEATURES):
        report.errors.append(f"{bin_path}: byte count not a multiple of 4 floats")
        return
    points = raw.reshape(-1, len(TARGET_POINT_FEATURES))
    if points.shape[0] != points_info.get("finite_count"):
        report.errors.append(
            f"{bin_path}: {points.shape[0]} points != label finite_count "
            f"{points_info.get('finite_count')}"
        )
    if points.size and not np.isfinite(points).all():
        report.errors.append(f"{bin_path}: non-finite point value")

    gt = label.get("ground_truth", {})
    if gt.get("target_frame") != TARGET_GT_FRAME:
        report.errors.append(f"{label_path}: target_frame != {TARGET_GT_FRAME}")
    if list(gt.get("box_fields", [])) != list(TARGET_BOX_FIELDS):
        report.errors.append(f"{label_path}: box_fields != {TARGET_BOX_FIELDS}")
    for i, box in enumerate(gt.get("boxes", [])):
        if box.get("class_name") not in TARGET_CLASS_NAMES:
            report.errors.append(f"{label_path}: box {i} class {box.get('class_name')!r}")
        values = [box.get(f) for f in TARGET_BOX_FIELDS]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            report.errors.append(f"{label_path}: box {i} non-finite field")
            continue
        if min(box["length"], box["width"], box["height"]) <= 0.0:
            report.errors.append(f"{label_path}: box {i} non-positive dimension")
        if not -math.pi - 1e-6 <= box["yaw"] <= math.pi + 1e-6:
            report.errors.append(f"{label_path}: box {i} yaw {box['yaw']} out of [-pi, pi]")


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
                    "checked_samples": report.checked_samples,
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
            f"{'OK' if report.ok else 'FAIL'}: {report.checked_samples} samples, "
            f"{len(report.errors)} errors"
        )
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
