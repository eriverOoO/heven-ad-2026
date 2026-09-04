"""Deterministic SCENARIO-level train/val/held-out-test split for the AV2
Stage-0 KalmanNet sanity experiment.

Splits by ``scenario_id`` only -- never by segment/track row -- so no
actor track from one scenario can ever appear in two different buckets
(a scenario's segments are already never split across shards either, per
``av2_motion_forecasting_adapter.assign_scenarios_to_shards``; this module
adds the analogous guarantee one level up, for train/val/test membership).

This is an **internal** split of the 120 scenarios already downloaded
from AV2's own public ``train`` split (see
``ad_morai_bridge_dev/config/dataset_factory/av2_motion_forecasting_adapter/
stage0_scenarios.manifest.json``). It is NOT related to, and never reads,
AV2's own official ``test`` split or its withheld answer-key files --
those are never used anywhere in this project.

The split-*generation logic* (this module, deterministic given a config)
is committed. The *generated* split manifest for a specific local
Stage-0 download is written to a machine-local path (default
``~/datasets/av2/manifests/stage0_kalmannet_split.json``) and is not
committed -- it is fully reproducible from this module + the committed
scenario manifest + the seed recorded in the config.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SPLIT_NAMES = ("train", "val", "test")
SPLIT_MANIFEST_SCHEMA_VERSION = "av2_kalmannet_stage0_split_v1"


class SplitError(RuntimeError):
    pass


@dataclass(frozen=True)
class SplitConfig:
    """~80/10/10 by default. Exact per-scenario counts are derived
    deterministically from ``len(scenario_ids)`` (rounded, remainder
    given to train) and recorded in the manifest -- never re-derived
    differently on a re-run with the same input."""

    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1
    seed: int = 20260905

    def canonical_json(self) -> str:
        return json.dumps(
            {"train_frac": self.train_frac, "val_frac": self.val_frac,
             "test_frac": self.test_frac, "seed": self.seed},
            sort_keys=True,
        )

    def validate(self) -> None:
        if not (0.0 < self.train_frac < 1.0 and 0.0 < self.val_frac < 1.0 and 0.0 < self.test_frac < 1.0):
            raise SplitError("train_frac/val_frac/test_frac must each be in (0, 1)")
        total = self.train_frac + self.val_frac + self.test_frac
        if abs(total - 1.0) > 1e-6:
            raise SplitError(f"train_frac + val_frac + test_frac must sum to 1.0, got {total}")


def assign_split(scenario_ids: list[str], config: SplitConfig) -> dict[str, str]:
    """Deterministic given (the *set* of scenario_ids, config) -- sorts
    first so input traversal order never matters (same discipline as
    ``assign_scenarios_to_shards``), then uses a seeded
    ``random.Random`` shuffle (never Python's process-randomized
    ``hash()``, never unseeded ``random``)."""
    config.validate()
    unique_sorted = sorted(set(scenario_ids))
    if len(unique_sorted) != len(scenario_ids):
        raise SplitError("duplicate scenario_ids in input")
    n = len(unique_sorted)
    if n < 3:
        raise SplitError(f"need at least 3 scenarios to form train/val/test, got {n}")

    n_train = round(n * config.train_frac)
    n_val = round(n * config.val_frac)
    n_train = max(1, min(n_train, n - 2))
    n_val = max(1, min(n_val, n - n_train - 1))
    n_test = n - n_train - n_val
    if n_test < 1:
        raise SplitError(f"degenerate split: n_train={n_train} n_val={n_val} n_test={n_test} for n={n}")

    rng = random.Random(config.seed)
    shuffled = list(unique_sorted)
    rng.shuffle(shuffled)

    assignment: dict[str, str] = {}
    for sid in shuffled[:n_train]:
        assignment[sid] = "train"
    for sid in shuffled[n_train : n_train + n_val]:
        assignment[sid] = "val"
    for sid in shuffled[n_train + n_val :]:
        assignment[sid] = "test"
    return assignment


def check_no_leakage(assignment: dict[str, str]) -> list[str]:
    """A dict can hold only one split per scenario_id by construction, so
    the only reachable failure modes are: a scenario assigned to an
    unknown split name, or (checked by the caller comparing against the
    original scenario_ids) a missing/extra scenario. Kept as an explicit,
    testable function rather than an inline assumption."""
    problems = []
    for sid, split in assignment.items():
        if split not in SPLIT_NAMES:
            problems.append(f"scenario_id {sid!r} assigned to unknown split {split!r}")
    return problems


def build_split_manifest(
    scenario_ids: list[str], config: SplitConfig, scenario_manifest_provenance: dict[str, Any]
) -> dict[str, Any]:
    assignment = assign_split(scenario_ids, config)
    leakage = check_no_leakage(assignment)
    if leakage:
        raise SplitError(f"split leakage: {leakage}")

    by_split: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for sid, split in assignment.items():
        by_split[split].append(sid)
    for name in SPLIT_NAMES:
        by_split[name].sort()

    # pairwise-disjoint + exact-coverage re-check directly from the built
    # lists (not just from the single-valued dict above), belt-and-braces.
    all_ids = [sid for name in SPLIT_NAMES for sid in by_split[name]]
    if sorted(all_ids) != sorted(set(scenario_ids)):
        raise SplitError("split does not exactly cover the input scenario_ids")
    if len(all_ids) != len(set(all_ids)):
        raise SplitError("a scenario_id appears in more than one split")

    return {
        "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
        "split_policy": "scenario_level_never_segment_level",
        "not_av2_official_test_split": True,
        "config": json.loads(config.canonical_json()),
        "scenario_count": len(scenario_ids),
        "split_counts": {name: len(by_split[name]) for name in SPLIT_NAMES},
        "splits": by_split,
        "scenario_manifest_provenance": scenario_manifest_provenance,
    }


def write_split_manifest(manifest: dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_split_manifest(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-manifest", type=Path, required=True,
                         help="the Stage-0 scenario manifest (tools/av2_dataset_prep/fetch_av2_stage0.py output)")
    parser.add_argument("--output", type=Path, required=True, help="split manifest output path (machine-local)")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args(argv)

    scenario_manifest = json.loads(args.scenario_manifest.read_text(encoding="utf-8"))
    scenario_ids = sorted(e["scenario_id"] for e in scenario_manifest.get("scenarios", []))
    config = SplitConfig(train_frac=args.train_frac, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed)
    manifest = build_split_manifest(
        scenario_ids, config,
        scenario_manifest_provenance={
            "manifest_path": str(args.scenario_manifest),
            "dataset_source": scenario_manifest.get("dataset_source"),
            "split": scenario_manifest.get("split"),
            "selection_seed": scenario_manifest.get("selection_seed"),
            "scenario_count": scenario_manifest.get("scenario_count"),
        },
    )
    write_split_manifest(manifest, args.output)
    print(json.dumps({"output": str(args.output), "split_counts": manifest["split_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
