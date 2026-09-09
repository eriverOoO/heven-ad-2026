#!/usr/bin/env python3
"""Motion-composition analysis of the frozen 9k AV2 TRAIN split
(docs/perception/kalmannet_av2_motion_composition_v1.md, sections 2-3).

Reads GT-only (CLEAN) sequences via streaming ``iter_segments`` directly
(never materializes the full corrupted 496k-sequence list via
``kalmannet_sequences.load_split_sequences``, which OOM-killed an earlier
attempt on this machine at ~150k/496k segments when run concurrently with
another memory-heavy process; this streaming version peaks at ~1.2 GB
RSS) -- motion composition is a property of ground truth, independent of
any corruption draw. Reports the definition-sensitivity audit (median vs.
mean vs. p90 speed) BEFORE locking in the chosen definition, then the
natural TRAIN distribution under that definition. Never reads official
VAL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from av2_split import load_split_manifest  # noqa: E402
from motion_composition import (  # noqa: E402
    MOVING,
    NEAR_STATIONARY,
    REPRESENTATIVE_SPEED_KEY,
    SPEED_BIN_LABELS,
    classify_motion,
    compute_segment_motion_stats,
    definition_agreement,
    speed_bin,
)

from ad_morai_bridge_dev.dataset.av2_kalmannet_shard_loader import iter_segments, load_shard_index  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    split_manifest = load_split_manifest(args.split_manifest)
    train_scenario_ids = set(split_manifest["splits"]["train"])

    refs = load_shard_index(args.shard_root)
    all_stats = []
    av2_types = []
    n_seen = 0
    for segment in iter_segments(args.shard_root, shard_index=refs):
        if not segment.valid_for_kalmannet_gt or segment.scenario_id not in train_scenario_ids:
            continue
        seq = segment.to_kalmannet_sequence(None)
        all_stats.append(compute_segment_motion_stats(seq))
        av2_types.append(segment.av2_object_type)
        n_seen += 1
        if n_seen % 50_000 == 0:
            print(f"  ...{n_seen} segments processed", flush=True)
    print(f"n_train_segments={n_seen}", flush=True)

    sensitivity = definition_agreement(all_stats)
    print("definition sensitivity (median/mean/p90):", json.dumps(sensitivity, indent=2), flush=True)

    categories = [classify_motion(s, key=REPRESENTATIVE_SPEED_KEY) for s in all_stats]
    n = len(categories)
    n_moving = categories.count(MOVING)
    n_stationary = categories.count(NEAR_STATIONARY)

    speed_bins: dict[str, int] = {label: 0 for label in SPEED_BIN_LABELS}
    for stats, cat in zip(all_stats, categories):
        if cat == MOVING:
            speed_bins[speed_bin(stats[REPRESENTATIVE_SPEED_KEY])] += 1

    class_by_motion: dict[str, dict[str, int]] = {MOVING: {}, NEAR_STATIONARY: {}}
    for cls, cat in zip(av2_types, categories):
        class_by_motion[cat][cls] = class_by_motion[cat].get(cls, 0) + 1

    report = {
        "schema_version": "av2_motion_composition_natural_distribution_v1",
        "representative_speed_key_chosen": REPRESENTATIVE_SPEED_KEY,
        "near_stationary_threshold_mps": 0.5,
        "definition_sensitivity_audit": sensitivity,
        "total_segments": n,
        "near_stationary_count": n_stationary,
        "near_stationary_fraction": n_stationary / n,
        "moving_count": n_moving,
        "moving_fraction": n_moving / n,
        "moving_speed_bins": speed_bins,
        "class_by_motion_group": class_by_motion,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in report.items() if k not in ("definition_sensitivity_audit", "class_by_motion_group")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
