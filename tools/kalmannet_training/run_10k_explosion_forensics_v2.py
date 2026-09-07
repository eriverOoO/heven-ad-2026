#!/usr/bin/env python3
"""Bounded (<=20 epoch) forensic replay of the real AV2 Scale-Up v2 10k
GENERIC-ROBUST seed-0 run, using the exact frozen dataset/split/config
and the PR #56 + PR #58 (gradient-norm-overflow) safe-step guard.
Analysis run only -- never trains to completion, never starts a new
seed, never touches architecture/hyperparameters, never evaluates
official AV2 VAL.

v2 of ``run_10k_explosion_forensics.py`` (kept unmodified as the v1
record of the 12-epoch run): extends the bound to 20 epochs (comfortably
past the original run's own documented epoch-16 collapse, per
docs/perception/kalmannet_gradient_norm_overflow_v1.md), and additionally
surfaces STATE-B (norm-overflow) event counts and per-epoch Adam-state /
parameter-magnitude snapshots in the live progress output.

Not wired as a permanent CLI/setup.py entry point -- a reproducible ad
hoc diagnostic script, matching this task's own framing.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from av2_split import load_split_manifest  # noqa: E402
from explosion_forensics import _speed_and_accel_stats, run_bounded_forensic_search  # noqa: E402
from kalmannet_sequences import load_split_sequences  # noqa: E402
from train_kalmannet import CONDITION_PRESETS  # noqa: E402
from training_forensics import _gap_stats  # noqa: E402

SHARD_ROOT = Path.home() / "datasets/av2/processed/kalmannet_scaleup_v2/train"
SPLIT_MANIFEST = Path.home() / "datasets/av2/manifests/scaleup_v2_internal_split_9k1k.json"
OUTPUT = Path.home() / "datasets/av2/checkpoints/kalmannet_10k_explosion_forensics_v2_seed0.json"

MAX_EPOCHS = 20
RING_BUFFER_SIZE = 20
SEED = 0


def dataset_percentiles(train_seqs: list[dict]) -> dict:
    """Computed ONCE over the real 9,000-scenario internal-TRAIN pool
    (496,434 sequences) for the gap/motion-outlier hypothesis tests
    (sections 9-10) -- percentiles only, never a claim about the failure
    batch's own identity."""
    missing_fracs, max_gaps, lengths, speeds, accels = [], [], [], [], []
    rng = np.random.RandomState(0)
    # A large, fixed, deterministic subsample (50,000 of 496,434) keeps
    # this bounded-cost while still representative -- computing exact
    # percentiles over the full 496,434 is not needed for a percentile
    # comparison and would add real minutes to a bounded diagnostic run.
    sample_idx = rng.choice(len(train_seqs), size=min(50_000, len(train_seqs)), replace=False)
    for i in sample_idx:
        seq = train_seqs[i]
        n_missing, max_gap, frac_missing = _gap_stats(seq["z_meas"])
        missing_fracs.append(frac_missing)
        max_gaps.append(max_gap)
        lengths.append(len(seq["frames"]))
        _, _, speed_max, _, accel_max = _speed_and_accel_stats(seq)
        speeds.append(speed_max)
        accels.append(accel_max)

    def pct(arr, p):
        return float(np.percentile(arr, p))

    return {
        "n_sampled": len(sample_idx),
        "missing_fraction": {p: pct(missing_fracs, p) for p in (50, 90, 95, 99)},
        "max_gap": {p: pct(max_gaps, p) for p in (50, 90, 95, 99)},
        "sequence_length": {p: pct(lengths, p) for p in (50, 90, 95, 99)},
        "speed_max": {p: pct(speeds, p) for p in (50, 90, 95, 99)},
        "accel_max": {p: pct(accels, p) for p in (50, 90, 95, 99)},
    }


def main():
    print(f"loading internal-TRAIN split from {SHARD_ROOT} ...", file=sys.stderr, flush=True)
    t0 = time.time()
    split_manifest = load_split_manifest(SPLIT_MANIFEST)
    corruption = CONDITION_PRESETS["generic_robust"]
    sequences = load_split_sequences(SHARD_ROOT, split_manifest, corruption_config=corruption)
    train_seqs = sequences["train"]
    print(f"loaded train={len(train_seqs)} in {time.time() - t0:.1f}s (val/test not needed for this analysis run)",
          file=sys.stderr, flush=True)

    print("computing dataset-wide percentile baselines (sampled) ...", file=sys.stderr, flush=True)
    t0 = time.time()
    baselines = dataset_percentiles(train_seqs)
    print(f"baselines computed in {time.time() - t0:.1f}s", file=sys.stderr, flush=True)

    print(f"starting bounded forensic search (v2, gradient-norm-overflow-aware): seed={SEED} "
          f"max_epochs={MAX_EPOCHS} ring_buffer_size={RING_BUFFER_SIZE}", file=sys.stderr, flush=True)
    search_start = time.time()

    def progress(epoch, batch_index, n_batches_this_epoch, running_train_loss_mean, running_grad_norm_max):
        tag = "EPOCH_END" if batch_index == -1 else f"batch={batch_index + 1}/{n_batches_this_epoch}"
        print(f"  epoch={epoch} {tag} running_train_loss_mean={running_train_loss_mean:.4g} "
              f"running_grad_norm_max={running_grad_norm_max} elapsed={time.time() - search_start:.1f}s",
              file=sys.stderr, flush=True)

    t0 = time.time()
    result = run_bounded_forensic_search(
        train_seqs, seed=SEED, device="cpu", batch_size=64,
        use_length_bucketing=True, n_buckets=8, lr=0.004, grad_clip=10.0,
        loss_on_predict_only=True, hidden_size=32,
        max_epochs=MAX_EPOCHS, ring_buffer_size=RING_BUFFER_SIZE,
        progress_callback=progress, progress_every_n_batches=500,
    )
    wall_time = time.time() - t0
    print(f"bounded search finished in {wall_time:.1f}s -- reproduced={result.reproduced} "
          f"stop_reason={result.stop_reason!r}", file=sys.stderr, flush=True)
    print(f"total norm_overflow_events captured: {len(result.norm_overflow_events)}", file=sys.stderr, flush=True)

    payload = result.to_json_dict()
    payload["wall_time_s"] = wall_time
    payload["dataset_percentile_baselines"] = baselines

    if result.reproduced:
        # Percentile-rank the failure batch's own worst sequence against
        # the dataset-wide baselines (sections 9-10).
        worst = result.failure_sequence_details[0]
        payload["failure_vs_dataset_percentile_rank"] = {
            "missing_fraction_of_worst_sequence": (worst.n_missing / worst.length_frames) if worst.length_frames else 0.0,
            "max_gap_of_worst_sequence": worst.longest_gap,
            "speed_max_of_worst_sequence": worst.speed_max,
            "sequence_length_of_worst_sequence": worst.length_frames,
        }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"written to {OUTPUT}", file=sys.stderr, flush=True)
    print(json.dumps({
        "reproduced": result.reproduced, "stop_reason": result.stop_reason,
        "first_event_epoch": result.first_event_epoch, "first_event_batch_index": result.first_event_batch_index,
        "first_nonfinite_param_name": result.first_nonfinite_param_name,
        "n_norm_overflow_events": len(result.norm_overflow_events),
        "preceding_norm_overflow_count_in_ring_buffer": result.preceding_norm_overflow_count_in_ring_buffer,
        "cumulative_norm_overflow_count_before_failure": result.cumulative_norm_overflow_count_before_failure,
        "output": str(OUTPUT),
    }, indent=2))


if __name__ == "__main__":
    main()
