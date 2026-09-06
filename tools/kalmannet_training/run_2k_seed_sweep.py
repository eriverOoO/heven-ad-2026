#!/usr/bin/env python3
"""AFTER-FIX 2k Stage-1 five-seed instability sweep, using the real
(fixed) ``batched_trainer.train_one_run_batched``. Reports the same
per-seed summary as the before-fix sweep, plus (for seed 0 only) a
bounded forensic capture via ``ForensicRecorder`` if any instability
event occurs.

TRAIN(1600)/internal-VAL(200) only -- the 200-scenario held-out TEST
split is never read here, and no seed is selected by test/official
performance (this is a numerical-stability diagnostic, not a model-
selection run).

Not wired as a permanent CLI entry point (no setup.py registration) --
a reproducible ad hoc diagnostic script, matching this task's own
"small multi-seed instability sweep" framing rather than a new
permanent training product.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from av2_split import load_split_manifest  # noqa: E402
from batched_trainer import train_one_run_batched  # noqa: E402
from kalmannet_sequences import load_split_sequences  # noqa: E402
from train_kalmannet import CONDITION_PRESETS  # noqa: E402
from trainer_core import set_all_seeds  # noqa: E402
from training_forensics import ForensicRecorder  # noqa: E402

SHARD_ROOT = Path.home() / "datasets/av2/processed/av2_kalmannet_v1_stage1_pilot_sharded"
SPLIT_MANIFEST = Path.home() / "datasets/av2/manifests/stage1_pilot_split.json"
OUTPUT = Path.home() / "datasets/av2/checkpoints/instability_sweep_after_2k.json"
FORENSIC_OUTPUT = Path.home() / "datasets/av2/checkpoints/instability_sweep_after_2k_seed0_forensic.json"

SEEDS = [0, 1, 2, 3, 4]


def main():
    split_manifest = load_split_manifest(SPLIT_MANIFEST)
    corruption = CONDITION_PRESETS["generic_robust"]
    sequences = load_split_sequences(SHARD_ROOT, split_manifest, corruption_config=corruption)
    train_seqs, val_seqs = sequences["train"], sequences["val"]
    print(f"train={len(train_seqs)} val={len(val_seqs)} (test={len(sequences['test'])} NEVER used)",
          file=sys.stderr, flush=True)

    results = {}
    for seed in SEEDS:
        t0 = time.time()
        set_all_seeds(seed)

        recorder = ForensicRecorder(window_size=20) if seed == 0 else None

        def progress(record, elapsed_s, _seed=seed):
            print(f"  seed={_seed} epoch={record.epoch} train_loss={record.train_loss:.4g} "
                  f"val_loss={record.val_loss:.4g} grad_norm_max={record.grad_norm_max} "
                  f"any_nan_train={record.any_nan_train} any_nan_val={record.any_nan_val} "
                  f"elapsed={elapsed_s:.1f}s", file=sys.stderr, flush=True)

        result = train_one_run_batched(
            train_seqs, val_seqs, seed=seed, device="cpu", batch_size=64,
            use_length_bucketing=True, n_buckets=8, lr=0.004, max_epochs=60, patience=15,
            hidden_size=32, grad_clip=10.0, loss_on_predict_only=True, progress_callback=progress,
            forensic_recorder=recorder,
        )
        wall_time = time.time() - t0

        seed_result = {
            "seed": seed,
            "grad_skip_count": result.grad_skip_count,
            "training_unstable": result.training_unstable,
            "training_collapsed": result.training_collapsed,
            "abort_reason": result.abort_reason,
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val,
            "n_epochs_run": result.n_epochs_run,
            "catastrophic": result.catastrophic,
            "nonfinite_step_count": result.nonfinite_step_count,
            "wall_time_s": wall_time,
        }
        results[str(seed)] = seed_result
        print(f"SEED {seed} DONE: {json.dumps(seed_result)}", file=sys.stderr, flush=True)

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(results, indent=2), encoding="utf-8")

        if recorder is not None and recorder.captured:
            FORENSIC_OUTPUT.write_text(
                json.dumps({
                    "first_failure_epoch": recorder.first_failure_epoch,
                    "first_failure_batch_index": recorder.first_failure_batch_index,
                    "window": recorder.capture_as_dicts(),
                }, indent=2, default=str),
                encoding="utf-8",
            )
            print(f"FORENSIC CAPTURE written to {FORENSIC_OUTPUT}", file=sys.stderr, flush=True)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
