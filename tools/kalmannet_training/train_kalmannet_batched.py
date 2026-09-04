#!/usr/bin/env python3
"""THROUGHPUT MODE AV2 Stage-0 KalmanNet training entry point.

Mirrors ``train_kalmannet.py``'s CLI shape exactly (same dataset/split/
corruption/hyperparameter arguments), but drives the batched recursion
(``batched_kalmannet.py``/``batched_trainer.py``) instead of the
historical per-sequence loop. At ``--batch-size 1`` this is CORRECTNESS
MODE (numerically equivalent to ``train_kalmannet.py``, see
``test_batched_trainer.py``); at ``--batch-size >1`` it is THROUGHPUT
MODE -- a distinct mini-batch optimization regime, documented in
``docs/perception/kalmannet_batched_training_v1.md``, never claimed
numerically identical to the historical per-sequence optimizer path.

**Stage-0 sanity / throughput-optimization tooling only.** Does not
modify the KalmanNet architecture, does not wire any checkpoint into
ROS/AB3DMOT, does not claim AV2 pretraining improves MORAI performance.

Example::

    python3 tools/kalmannet_training/train_kalmannet_batched.py \\
        --shard-root ~/datasets/av2/processed/av2_kalmannet_v1_sharded \\
        --split-manifest ~/datasets/av2/manifests/stage0_kalmannet_split.json \\
        --condition clean --device cpu --batch-size 32 \\
        --output-checkpoint ~/datasets/av2/checkpoints/av2_stage0_clean_batched32_seed0.pt \\
        --output-history ~/datasets/av2/checkpoints/av2_stage0_clean_batched32_seed0_history.csv \\
        --seed 0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import checkpoint_utils  # noqa: E402
from av2_split import load_split_manifest  # noqa: E402
from batched_trainer import train_one_run_batched  # noqa: E402
from kalmannet_sequences import class_distribution, load_split_sequences  # noqa: E402
from trainer_core import set_all_seeds  # noqa: E402
from train_kalmannet import (  # noqa: E402
    CONDITION_PRESETS,
    HISTORICAL_GRAD_CLIP,
    HISTORICAL_HIDDEN_SIZE,
    HISTORICAL_LR,
    HISTORICAL_MAX_EPOCHS,
    HISTORICAL_PATIENCE,
)

BATCHED_CHECKPOINT_FAMILY = "av2_stage0_kalmannet_batched_v1"
SAMPLER_VERSION = "length_bucketed_v1"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--condition", choices=list(CONDITION_PRESETS), default="clean")
    parser.add_argument("--corruption-seed", type=int, default=None)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=32,
                         help="1 = CORRECTNESS MODE (numerically equivalent to the historical per-sequence "
                              "trainer); >1 = THROUGHPUT MODE (mini-batch gradient averaging, a distinct "
                              "optimization regime -- see docs/perception/kalmannet_batched_training_v1.md)")
    parser.add_argument("--no-length-bucketing", dest="use_length_bucketing", action="store_false", default=True,
                         help="disable length-bucketed batch construction (falls back to a plain shuffled "
                              "contiguous chunking of the epoch order)")
    parser.add_argument("--n-buckets", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=HISTORICAL_HIDDEN_SIZE)
    parser.add_argument("--lr", type=float, default=HISTORICAL_LR)
    parser.add_argument("--grad-clip", type=float, default=HISTORICAL_GRAD_CLIP)
    parser.add_argument("--max-epochs", type=int, default=HISTORICAL_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=HISTORICAL_PATIENCE)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--loss-on-predict-only", dest="loss_on_predict_only", action="store_true", default=True)
    parser.add_argument("--no-loss-on-predict-only", dest="loss_on_predict_only", action="store_false")
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--output-history", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            print("--device cuda requested but CUDA is not available", file=sys.stderr)
            return 2

    corruption_config = CONDITION_PRESETS[args.condition]
    if args.corruption_seed is not None:
        corruption_config = type(corruption_config)(**{**corruption_config.__dict__, "seed": args.corruption_seed})

    split_manifest = load_split_manifest(args.split_manifest)
    split_manifest_sha256 = checkpoint_utils.sha256_json(split_manifest)
    dataset_manifest_path = Path(args.shard_root) / "export_manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text("utf-8"))
    dataset_manifest_sha256 = dataset_manifest.get("content_fingerprint", "unknown")

    sequences = load_split_sequences(args.shard_root, split_manifest, corruption_config=corruption_config)
    train_seqs, val_seqs = sequences["train"], sequences["val"]
    print(f"train={len(train_seqs)} val={len(val_seqs)} test={len(sequences['test'])} sequences", file=sys.stderr)
    print(f"train class distribution: {class_distribution(train_seqs)}", file=sys.stderr)
    print(
        f"batch_size={args.batch_size} use_length_bucketing={args.use_length_bucketing} "
        f"n_buckets={args.n_buckets}", file=sys.stderr,
    )

    def report_progress(record, elapsed_s):
        print(
            f"epoch {record.epoch}: train_loss={record.train_loss:.4f} val_loss={record.val_loss:.4f} "
            f"grad_norm_mean={record.grad_norm_mean:.3f} elapsed={elapsed_s:.1f}s",
            file=sys.stderr, flush=True,
        )

    set_all_seeds(args.seed)
    result = train_one_run_batched(
        train_seqs, val_seqs, seed=args.seed, device=args.device, batch_size=args.batch_size,
        use_length_bucketing=args.use_length_bucketing, n_buckets=args.n_buckets, lr=args.lr,
        max_epochs=args.max_epochs, patience=args.patience, hidden_size=args.hidden_size,
        grad_clip=args.grad_clip, loss_on_predict_only=args.loss_on_predict_only,
        progress_callback=report_progress,
    )

    args.output_history.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_history, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "grad_norm_mean",
                                                 "grad_norm_max", "any_nan_train", "any_nan_val"])
        writer.writeheader()
        for record in result.history:
            writer.writerow(vars(record))

    if result.best_state_dict is None:
        print("training produced no valid checkpoint (catastrophic run) -- not saving", file=sys.stderr)
        print(json.dumps({"catastrophic": True, "best_val": result.best_val}, indent=2))
        return 1

    manifest = checkpoint_utils.build_checkpoint_manifest(
        checkpoint_family=BATCHED_CHECKPOINT_FAMILY,
        model_config={"hidden_size": args.hidden_size, "state_dim": 4, "measurement_dim": 2},
        training_config={
            "lr": args.lr, "grad_clip": args.grad_clip, "max_epochs": args.max_epochs,
            "patience": args.patience, "batch_size": args.batch_size, "optimizer": "Adam",
            "condition": args.condition, "n_train_sequences": len(train_seqs), "n_val_sequences": len(val_seqs),
            "deviation_from_historical_max_epochs": args.max_epochs != HISTORICAL_MAX_EPOCHS,
            "use_length_bucketing": args.use_length_bucketing, "n_buckets": args.n_buckets,
            "sampler_version": SAMPLER_VERSION,
            "mini_batch_semantics": (
                "correctness_mode_equivalent_to_per_sequence_optimizer" if args.batch_size == 1
                else "throughput_mode_mini_batch_gradient_averaging_not_equivalent_to_per_sequence_optimizer"
            ),
        },
        seed_info={"seed": args.seed, "init_seed": result.init_seed, "order_seed": result.order_seed},
        corruption_config=json.loads(corruption_config.canonical_json()),
        dataset_manifest_sha256=dataset_manifest_sha256,
        split_manifest_sha256=split_manifest_sha256,
        coordinate_mode="first_state_relative",
        best_epoch=result.best_epoch,
        best_val_loss=result.best_val,
        loss_on_predict_only=args.loss_on_predict_only,
    )
    checksum = checkpoint_utils.save_checkpoint(result.best_state_dict, manifest, args.output_checkpoint)

    summary = {
        "output_checkpoint": str(args.output_checkpoint),
        "checkpoint_sha256": checksum,
        "best_epoch": result.best_epoch,
        "best_val_loss": result.best_val,
        "n_epochs_run": result.n_epochs_run,
        "catastrophic": result.catastrophic,
        "train_time_s": result.train_time_s,
        "nonfinite_step_count": result.nonfinite_step_count,
        "n_train_sequences": len(train_seqs),
        "n_val_sequences": len(val_seqs),
        "batch_size": args.batch_size,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
