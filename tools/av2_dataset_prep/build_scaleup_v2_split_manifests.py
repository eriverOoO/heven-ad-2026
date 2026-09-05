#!/usr/bin/env python3
"""Writes the two single-role split manifests the Scale-Up v2 dataset
needs -- one all-``train`` manifest for the 10k TRAIN-source shard root,
one all-``val`` manifest for the physically separate 1k official-VAL
shard root. Deliberately NOT `av2_split.py::assign_split` (that function
enforces a mandatory train/val/test 3-way carve-out of ONE scenario
pool, which is the wrong tool here -- Scale-Up v2 has two INDEPENDENT
scenario pools, each already 100% one role by construction).

Kept as its own tiny script rather than a new `av2_split.py` function so
the existing, shared split module is not touched by this task.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "kalmannet_training"))
from av2_split import SPLIT_MANIFEST_SCHEMA_VERSION  # noqa: E402


def build_single_role_manifest(scenario_ids: list[str], role: str, scenario_manifest_provenance: dict) -> dict:
    if role not in ("train", "val", "test"):
        raise ValueError(f"unsupported role {role!r}")
    unique_sorted = sorted(set(scenario_ids))
    if len(unique_sorted) != len(scenario_ids):
        raise ValueError("duplicate scenario_ids in input")
    splits = {"train": [], "val": [], "test": []}
    splits[role] = unique_sorted
    return {
        "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
        "split_policy": "single_role_physically_separate_dataset",
        "not_av2_official_test_split": True,
        "config": {"role": role},
        "scenario_count": len(unique_sorted),
        "split_counts": {name: len(ids) for name, ids in splits.items()},
        "splits": splits,
        "scenario_manifest_provenance": scenario_manifest_provenance,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-manifest", type=Path, required=True)
    parser.add_argument("--role", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    scenario_manifest = json.loads(args.scenario_manifest.read_text("utf-8"))
    scenario_ids = [s["scenario_id"] for s in scenario_manifest["scenarios"]]
    manifest = build_single_role_manifest(
        scenario_ids, args.role,
        scenario_manifest_provenance={
            "manifest_schema_version": scenario_manifest.get("manifest_schema_version"),
            "selection_seed": scenario_manifest.get("selection_seed"),
            "selection_pool_size": scenario_manifest.get("selection_pool_size"),
            "split": scenario_manifest.get("split"),
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps({"output": str(args.output), "scenario_count": manifest["scenario_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
