#!/usr/bin/env python3
"""Writes the small, machine-independent freeze manifest for AV2
KalmanNet Scale-Up Dataset v2 -- 10k TRAIN-source scenarios + 1k
official-VAL scenarios, both freshly acquired and verified to have zero
overlap with every prior AV2 experiment in this project
(Stage-0's 120 + Stage-1 pilot's 2,000).

Run once both TRAIN and official-VAL have been downloaded, sharded, and
validated (this script does not download/shard/validate anything itself
-- it only reads already-produced manifests/export summaries).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
        ).decode().strip()
    except Exception:
        return "unknown"


def _sha256_of_json_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-scenario-manifest", type=Path, required=True)
    parser.add_argument("--train-split-manifest", type=Path, required=True)
    parser.add_argument("--train-export-manifest", type=Path, required=True,
                         help="the export_manifest.json written into the TRAIN shard output root")
    parser.add_argument("--val-scenario-manifest", type=Path, required=True)
    parser.add_argument("--val-split-manifest", type=Path, required=True)
    parser.add_argument("--val-export-manifest", type=Path, required=True,
                         help="the export_manifest.json written into the official-VAL shard output root")
    parser.add_argument("--exclusion-manifests", nargs="+", required=True,
                         help="every prior scenario manifest excluded from this task's selection")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    train_scenarios = json.loads(args.train_scenario_manifest.read_text("utf-8"))
    train_export = json.loads(args.train_export_manifest.read_text("utf-8"))
    val_scenarios = json.loads(args.val_scenario_manifest.read_text("utf-8"))
    val_export = json.loads(args.val_export_manifest.read_text("utf-8"))

    train_ids = {s["scenario_id"] for s in train_scenarios["scenarios"]}
    val_ids = {s["scenario_id"] for s in val_scenarios["scenarios"]}
    overlap_train_val = train_ids & val_ids

    exclusion_ids: set[str] = set()
    exclusion_hashes = {}
    for p in args.exclusion_manifests:
        p = Path(p).expanduser()
        m = json.loads(p.read_text("utf-8"))
        exclusion_ids |= {s["scenario_id"] for s in m["scenarios"]}
        exclusion_hashes[p.name] = _sha256_of_json_file(p)

    manifest = {
        "schema_version": "av2_kalmannet_scaleup_v2_freeze_v1",
        "git_sha": _git_sha(),
        "train": {
            "scenario_count": train_scenarios["scenario_count"],
            "selection_seed": train_scenarios["selection_seed"],
            "selection_pool_size": train_scenarios["selection_pool_size"],
            "source_split": train_scenarios["split"],
            "scenario_manifest_sha256": _sha256_of_json_file(args.train_scenario_manifest),
            "split_manifest_sha256": _sha256_of_json_file(args.train_split_manifest),
            "content_fingerprint": train_export.get("content_fingerprint"),
            "shard_count": train_export.get("shard_count"),
            "counts": train_export.get("counts"),
        },
        "official_val": {
            "scenario_count": val_scenarios["scenario_count"],
            "selection_seed": val_scenarios["selection_seed"],
            "selection_pool_size": val_scenarios["selection_pool_size"],
            "source_split": val_scenarios["split"],
            "scenario_manifest_sha256": _sha256_of_json_file(args.val_scenario_manifest),
            "split_manifest_sha256": _sha256_of_json_file(args.val_split_manifest),
            "content_fingerprint": val_export.get("content_fingerprint"),
            "shard_count": val_export.get("shard_count"),
            "counts": val_export.get("counts"),
        },
        "exclusion_manifests": exclusion_hashes,
        "exclusion_scenario_count": len(exclusion_ids),
        "overlap_train_vs_official_val": len(overlap_train_val),
        "overlap_train_vs_exclusion": len(train_ids & exclusion_ids),
        "overlap_val_vs_exclusion": len(val_ids & exclusion_ids),
        "adapter_shard_format_version": "av2_kalmannet_shard_v1",
        "corruption_config_version": "corruption_v2",
        "corruption_materialized": False,
        "coordinate_mode": "first_state_relative",
    }

    for key, value in (
        ("overlap_train_vs_official_val", manifest["overlap_train_vs_official_val"]),
        ("overlap_train_vs_exclusion", manifest["overlap_train_vs_exclusion"]),
        ("overlap_val_vs_exclusion", manifest["overlap_val_vs_exclusion"]),
    ):
        if value != 0:
            print(f"FATAL: {key} = {value}, expected 0 -- refusing to write a freeze manifest", file=sys.stderr)
            return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
