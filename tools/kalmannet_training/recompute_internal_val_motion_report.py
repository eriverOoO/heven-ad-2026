#!/usr/bin/env python3
"""Recomputes the internal-VAL ALL/NEAR_STATIONARY/MOVING motion report
for an ALREADY-TRAINED checkpoint, without retraining -- used once to
recover from a JSON-serialization bug in
``train_kalmannet_motion_composition.py``'s first release (fixed; see
``compute_internal_val_motion_report`` and its regression tests) that
crashed only at the very last write step of an hours-long screening run,
after the checkpoint itself was already safely saved.

Streams only the VAL split (never materializes TRAIN in memory) to keep
this recovery run's own memory footprint small.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import checkpoint_utils  # noqa: E402
from av2_split import load_split_manifest  # noqa: E402
from evaluate_kalmannet import run_knet  # noqa: E402
from train_kalmannet import CONDITION_PRESETS  # noqa: E402
from train_kalmannet_motion_composition import compute_internal_val_motion_report  # noqa: E402

from ad_morai_bridge_dev.dataset.av2_kalmannet_shard_loader import iter_segments, load_shard_index  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--condition", choices=list(CONDITION_PRESETS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    from ad_lidar_perception.kalmannet_core import KalmanNetGRU
    from kalmannet_sequences import _to_training_sequence

    corruption_config = CONDITION_PRESETS[args.condition]
    split_manifest = load_split_manifest(args.split_manifest)
    val_scenario_ids = set(split_manifest["splits"]["val"])

    refs = load_shard_index(args.shard_root)
    val_seqs = []
    for segment in iter_segments(args.shard_root, shard_index=refs):
        if not segment.valid_for_kalmannet_gt or segment.scenario_id not in val_scenario_ids:
            continue
        seq = _to_training_sequence(segment, corruption_config)
        if seq is not None:
            val_seqs.append(seq)
    print(f"n_val_sequences={len(val_seqs)}", file=sys.stderr)

    state_dict, _manifest = checkpoint_utils.load_checkpoint(args.checkpoint)
    net = KalmanNetGRU(hidden_size=args.hidden_size).to(args.device)
    net.load_state_dict(state_dict)
    net.eval()

    report = compute_internal_val_motion_report(val_seqs, lambda s: run_knet(net, s, args.device))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
