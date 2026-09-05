#!/usr/bin/env python3
"""Reproducible AV2 Stage-0 KalmanNet training entry point.

Trains the EXISTING, UNMODIFIED ``KalmanNetGRU`` architecture
(``ad_lidar_perception/ad_lidar_perception/kalmannet_core.py``) on the
AV2 Stage-0 sharded dataset (``av2_kalmannet_shard_loader.py``, reused as
the single data source -- no second AV2 loader here). See
``trainer_core.py``'s own module docstring for exactly which historical
training behaviors were ported and which were intentionally left out.

**Stage-0 sanity only.** Does not claim AV2 pretraining improves MORAI
performance. Does not wire any checkpoint into ROS/AB3DMOT/the runtime
tracker. Produces an offline checkpoint + reproducibility manifest only.

Example::

    python3 tools/kalmannet_training/train_kalmannet.py \\
        --shard-root ~/datasets/av2/processed/av2_kalmannet_v1_sharded \\
        --split-manifest ~/datasets/av2/manifests/stage0_kalmannet_split.json \\
        --condition clean \\
        --device cuda \\
        --output-checkpoint ~/datasets/av2/checkpoints/av2_stage0_clean_seed0.pt \\
        --output-history ~/datasets/av2/checkpoints/av2_stage0_clean_seed0_history.csv \\
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
from kalmannet_sequences import class_distribution, load_split_sequences  # noqa: E402
from trainer_core import set_all_seeds, train_one_run  # noqa: E402

from ad_morai_bridge_dev.dataset.av2_motion_forecasting_adapter import CorruptionConfig  # noqa: E402

# Frozen dense-v2 historical hyperparameters, verified against
# ~/heven_presentation_assets/kalmannet_training_stability/
# frozen_model_v2_manifest.json (checked, not assumed): hidden_size=32,
# lr=0.001, grad_clip=10.0, max_epochs=60, patience=10.
HISTORICAL_HIDDEN_SIZE = 32
HISTORICAL_LR = 0.001
HISTORICAL_GRAD_CLIP = 10.0
HISTORICAL_MAX_EPOCHS = 60
HISTORICAL_PATIENCE = 10


# MORAI TRAIN+VAL-calibrated corruption (never TEST) -- constants and
# full derivation live in morai_calibration.py, reused here unmodified.
from morai_calibration import build_morai_calibrated_corruption_config  # noqa: E402

CONDITION_PRESETS = {
    "clean": CorruptionConfig(mode="none"),
    "generic_robust": CorruptionConfig(
        mode="custom", gaussian_noise_std_m=0.3, dropout_prob=0.1,
        dropout_burst_enabled=True, dropout_burst_prob=0.02,
        dropout_burst_min_len=2, dropout_burst_max_len=5, seed=20260905,
    ),
    "morai_calibrated_robust": build_morai_calibrated_corruption_config(),
}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--condition", choices=list(CONDITION_PRESETS), default="clean",
                         help="clean = no corruption; generic_robust = modest Gaussian noise + dropout + burst "
                              "(NOT claimed MORAI-realistic -- a generic robustness sanity condition only); "
                              "morai_calibrated_robust = bias+full-covariance noise and dropout/burst rates "
                              "calibrated from MORAI TRAIN+VAL residuals only (never TEST)")
    parser.add_argument("--corruption-seed", type=int, default=None,
                         help="override the condition preset's corruption seed")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    parser.add_argument("--hidden-size", type=int, default=HISTORICAL_HIDDEN_SIZE)
    parser.add_argument("--lr", type=float, default=HISTORICAL_LR)
    parser.add_argument("--grad-clip", type=float, default=HISTORICAL_GRAD_CLIP)
    parser.add_argument("--max-epochs", type=int, default=HISTORICAL_MAX_EPOCHS,
                         help="Stage-0 sanity: may be reduced from the historical 60 if convergence is clearly "
                              "demonstrated sooner -- any deviation is recorded in the checkpoint manifest")
    parser.add_argument("--patience", type=int, default=HISTORICAL_PATIENCE)
    parser.add_argument("--seed", type=int, required=True, help="model init + shuffle-order seed")
    parser.add_argument("--loss-on-predict-only", dest="loss_on_predict_only", action="store_true", default=True,
                         help="DEFAULT (historical-compatible, verified against instrumented_train.py): "
                              "loss on every frame with dt is not None, including missing-measurement "
                              "predict-only frames")
    parser.add_argument("--no-loss-on-predict-only", dest="loss_on_predict_only", action="store_false",
                         help="diagnostic-only alternative: loss restricted to measurement-update frames only")
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--output-history", type=Path, required=True, help="CSV per-epoch history")
    args = parser.parse_args(argv)
    return args


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

    def report_progress(record, elapsed_s):
        print(
            f"epoch {record.epoch}: train_loss={record.train_loss:.4f} val_loss={record.val_loss:.4f} "
            f"grad_norm_mean={record.grad_norm_mean:.3f} elapsed={elapsed_s:.1f}s",
            file=sys.stderr, flush=True,
        )

    set_all_seeds(args.seed)
    result = train_one_run(
        train_seqs, val_seqs, seed=args.seed, device=args.device, lr=args.lr,
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
        checkpoint_family=checkpoint_utils.CHECKPOINT_FAMILY,
        model_config={"hidden_size": args.hidden_size, "state_dim": 4, "measurement_dim": 2},
        training_config={
            "lr": args.lr, "grad_clip": args.grad_clip, "max_epochs": args.max_epochs,
            "patience": args.patience, "batch_size": 1, "optimizer": "Adam",
            "condition": args.condition, "n_train_sequences": len(train_seqs), "n_val_sequences": len(val_seqs),
            "deviation_from_historical_max_epochs": args.max_epochs != HISTORICAL_MAX_EPOCHS,
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
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
