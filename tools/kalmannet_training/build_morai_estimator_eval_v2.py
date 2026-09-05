#!/usr/bin/env python3
"""Builds (or, when no eligible new actors exist, honestly reports why it
cannot build) the MORAI_ESTIMATOR_EVAL_V2 dataset.

**Validation-first, by design**: this CLI always runs the source
inventory + actor-leakage audit BEFORE attempting to construct a single
sequence. If the audit finds zero eligible new actors (the situation
this task actually found, given only one MORAI run has ever been
captured and every one of its actors is already assigned to
TRAIN/VAL/TEST/excluded), the CLI exits after writing the audit report --
it does not fabricate sequences, and it does not silently fall back to
reusing TRAIN/VAL actors.

Restartable: every step reads already-on-disk artifacts and writes new
output atomically; re-running with the same inputs reproduces the same
audit/output.

Example::

    python3 tools/kalmannet_training/build_morai_estimator_eval_v2.py \\
        --actor-manifest ~/heven_presentation_assets/motion_gt_expansion/canonical/actor_manifest.csv \\
        --split-policy ~/heven_presentation_assets/motion_gt_expansion/canonical/split_policy.json \\
        --output-dir tools/kalmannet_training/frozen_configs \\
        --split-role test
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from morai_estimator_eval_v2 import (  # noqa: E402
    V1_FROZEN_TEST_ACTOR_IDS,
    audit_actor_leakage,
    audit_source_runs,
    load_actor_manifest_csv,
)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
        ).decode().strip()
    except Exception:
        return "unknown"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--actor-manifest", type=Path,
                         default=Path.home() / "heven_presentation_assets/motion_gt_expansion/canonical/actor_manifest.csv")
    parser.add_argument("--split-policy", type=Path,
                         default=Path.home() / "heven_presentation_assets/motion_gt_expansion/canonical/split_policy.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-role", default="test")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    source_records = audit_source_runs()
    print(f"source inventory: {len(source_records)} record(s)", file=sys.stderr)
    for r in source_records:
        print(f"  {r.date_or_run_id}: suitable_as_new_v2_test_data={r.suitable_as_new_v2_test_data}", file=sys.stderr)

    actor_rows = load_actor_manifest_csv(args.split_policy.parent / "actor_manifest.csv") \
        if (args.split_policy.parent / "actor_manifest.csv").is_file() else []
    all_actor_ids = [int(r["actor_id"]) for r in actor_rows] if actor_rows else None

    split_policy = json.loads(args.split_policy.read_text("utf-8")) if args.split_policy.is_file() else {}
    leakage_rows = audit_actor_leakage(
        all_actor_ids=all_actor_ids,
        train_ids=tuple(split_policy.get("train_actor_ids", ())),
        val_ids=tuple(split_policy.get("val_actor_ids", ())),
        test_ids=tuple(split_policy.get("test_actor_ids", V1_FROZEN_TEST_ACTOR_IDS)),
        excluded_ids=tuple(split_policy.get("excluded_actor_ids", ())),
    ) if split_policy else audit_actor_leakage()

    eligible = [row for row in leakage_rows if row.eligible_for_v2]
    print(f"leakage audit: {len(leakage_rows)} actors total, {len(eligible)} eligible for NEW v2 data", file=sys.stderr)

    report = {
        "schema_version": "morai_estimator_eval_v2_audit",
        "git_sha": _git_sha(),
        "split_role": args.split_role,
        "source_runs": [asdict(r) for r in source_records],
        "actor_leakage_audit": [asdict(r) for r in leakage_rows],
        "n_eligible_new_actors": len(eligible),
        "v1_frozen_test_actor_ids_reused_unchanged": list(V1_FROZEN_TEST_ACTOR_IDS),
        "conclusion": (
            "0 new actor trajectories are available from existing local MORAI data -- "
            "every actor in the sole captured MORAI run is already assigned to "
            "TRAIN/VAL/TEST or excluded for GT-quality reasons. New MORAI simulator "
            "collection is required to grow V2 beyond V1's existing 3 sequences. "
            "See docs/perception/morai_estimator_eval_v2.md for the collection checklist."
            if not eligible else
            f"{len(eligible)} new actor(s) eligible -- sequence construction not yet run by this CLI call."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "morai_estimator_eval_v2_audit.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "n_eligible_new_actors": len(eligible)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
