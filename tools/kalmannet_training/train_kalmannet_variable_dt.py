#!/usr/bin/env python3
"""AV2 KalmanNet training entry point with an OPT-IN physically-consistent
variable-dt (temporal-thinning) augmentation (docs/perception/
kalmannet_av2_variable_dt_v1.md).

Mirrors ``train_kalmannet_batched.py``/`train_kalmannet_motion_composition.py`
exactly (same dataset/split/corruption/frozen-hyperparameter CLI contract
-- model/optimizer config is NOT retuned here) and adds exactly one new
axis: ``--dt-policy {fixed,mild,strong}``. ``fixed`` (the default) uses
``variable_dt.FIXED_DT``, an exact thinning no-op -- byte-identical to
the frozen NATURAL baseline's own training path.

**No KalmanNet architecture/optimizer-hyperparameter change. No MORAI
evaluation or claim.**
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import checkpoint_utils  # noqa: E402
from av2_split import load_split_manifest  # noqa: E402
from batched_trainer import train_one_run_batched  # noqa: E402
from kalmannet_sequences import class_distribution  # noqa: E402
from trainer_core import set_all_seeds  # noqa: E402
from train_kalmannet import (  # noqa: E402
    CONDITION_PRESETS,
    HISTORICAL_GRAD_CLIP,
    HISTORICAL_HIDDEN_SIZE,
    HISTORICAL_LR,
    HISTORICAL_MAX_EPOCHS,
    HISTORICAL_PATIENCE,
)
from variable_dt import (  # noqa: E402
    FIXED_DT,
    MILD_VARIABLE_DT,
    STRONG_VARIABLE_DT,
    evaluate_with_dt_buckets,
    load_split_sequences_with_thinning,
)

BATCHED_CHECKPOINT_FAMILY = "av2_stage0_kalmannet_batched_v1"
SAMPLER_VERSION = "length_bucketed_v1"

DT_POLICY_CHOICES = {"fixed": FIXED_DT, "mild": MILD_VARIABLE_DT, "strong": STRONG_VARIABLE_DT}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--condition", choices=list(CONDITION_PRESETS), default="clean")
    parser.add_argument("--corruption-seed", type=int, default=None)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-length-bucketing", dest="use_length_bucketing", action="store_false", default=True)
    parser.add_argument("--n-buckets", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=HISTORICAL_HIDDEN_SIZE)
    parser.add_argument("--lr", type=float, default=HISTORICAL_LR)
    parser.add_argument("--grad-clip", type=float, default=HISTORICAL_GRAD_CLIP)
    parser.add_argument("--max-epochs", type=int, default=HISTORICAL_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=HISTORICAL_PATIENCE)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--loss-on-predict-only", dest="loss_on_predict_only", action="store_true", default=True)
    parser.add_argument("--no-loss-on-predict-only", dest="loss_on_predict_only", action="store_false")
    parser.add_argument("--dt-policy", choices=list(DT_POLICY_CHOICES), default="fixed",
                         help="'fixed' (default) = byte-identical to the frozen baseline (no thinning); "
                              "'mild'/'strong' apply deterministic temporal thinning at load time, per "
                              "variable_dt.py's declared skip-probability distributions.")
    parser.add_argument("--augmentation-seed", type=int, default=None,
                         help="RNG seed for temporal thinning (default: same value as --seed).")
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--output-history", type=Path, required=True)
    parser.add_argument("--output-internal-val-dt-report", type=Path, default=None,
                         help="optional: evaluate the FINAL best checkpoint on internal VAL under all three "
                              "dt policies (fixed/mild/strong, section 13) with dt-bucket breakdown "
                              "(section 14) -- reuses evaluate_with_dt_buckets verbatim, never official VAL.")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--no-resume", dest="resume_enabled", action="store_false", default=True)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=1)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--num-threads", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    augmentation_seed = args.augmentation_seed if args.augmentation_seed is not None else args.seed

    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            print("--device cuda requested but CUDA is not available", file=sys.stderr)
            return 2

    if args.num_threads is not None:
        import torch
        torch.set_num_threads(args.num_threads)
        print(f"torch.set_num_threads({args.num_threads})", file=sys.stderr)

    corruption_config = CONDITION_PRESETS[args.condition]
    if args.corruption_seed is not None:
        corruption_config = type(corruption_config)(**{**corruption_config.__dict__, "seed": args.corruption_seed})

    split_manifest = load_split_manifest(args.split_manifest)
    split_manifest_sha256 = checkpoint_utils.sha256_json(split_manifest)
    dataset_manifest_path = Path(args.shard_root) / "export_manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text("utf-8"))
    dataset_manifest_sha256 = dataset_manifest.get("content_fingerprint", "unknown")

    dt_policy = DT_POLICY_CHOICES[args.dt_policy]
    sequences = load_split_sequences_with_thinning(
        args.shard_root, split_manifest, dt_policy, augmentation_seed, corruption_config,
    )
    train_seqs, val_seqs = sequences["train"], sequences["val"]
    n_thinned = sum(1 for s in train_seqs if s.get("is_thinned"))
    print(f"train={len(train_seqs)} val={len(val_seqs)} test={len(sequences['test'])} sequences", file=sys.stderr)
    print(f"train class distribution: {class_distribution(train_seqs)}", file=sys.stderr)
    print(
        f"dt_policy={dt_policy.name} augmentation_seed={augmentation_seed} "
        f"n_train_thinned={n_thinned} ({100.0 * n_thinned / max(1, len(train_seqs)):.2f}%)",
        file=sys.stderr,
    )

    n_batches_per_epoch = math.ceil(len(train_seqs) / args.batch_size) if args.batch_size else len(train_seqs)
    log_file = open(args.log_file, "a", buffering=1, encoding="utf-8") if args.log_file else None
    if log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
    progress_state = {"best_val": float("inf"), "best_epoch": None, "epochs_since_improve": 0,
                       "prev_cumulative_s": 0.0, "run_start_wall": time.time()}

    def report_progress(record, cumulative_elapsed_s):
        epoch_wall_time_s = cumulative_elapsed_s - progress_state["prev_cumulative_s"]
        progress_state["prev_cumulative_s"] = cumulative_elapsed_s
        improved = math.isfinite(record.val_loss) and record.val_loss < progress_state["best_val"] - 1e-6
        if improved:
            progress_state["best_val"] = record.val_loss
            progress_state["best_epoch"] = record.epoch
            progress_state["epochs_since_improve"] = 0
        else:
            progress_state["epochs_since_improve"] += 1
        patience_remaining = max(0, args.patience - progress_state["epochs_since_improve"])
        line = (
            f"epoch={record.epoch} train_loss={record.train_loss:.6f} val_loss_mse_xy_vxvy={record.val_loss:.6f} "
            f"grad_norm_mean={record.grad_norm_mean:.4f} grad_norm_max={record.grad_norm_max:.4f} "
            f"optimizer_steps_this_epoch~={n_batches_per_epoch} epoch_wall_time_s={epoch_wall_time_s:.1f} "
            f"cumulative_wall_time_s={cumulative_elapsed_s:.1f} best_epoch={progress_state['best_epoch']} "
            f"patience_remaining={patience_remaining} any_nan_train={record.any_nan_train} "
            f"any_nan_val={record.any_nan_val} "
            f"n_norm_overflow_batches={record.n_norm_overflow_batches} "
            f"n_element_nonfinite_batches={record.n_element_nonfinite_batches}"
        )
        print(line, file=sys.stderr, flush=True)
        if log_file is not None:
            log_file.write(line + "\n")

    resume_checkpoint_path = None
    resume_validation_key = None
    if args.resume_enabled:
        resume_checkpoint_path = args.resume_checkpoint or args.output_checkpoint.with_suffix(
            args.output_checkpoint.suffix + ".resume.pt"
        )
        resume_validation_key = {
            "seed": args.seed, "lr": args.lr, "batch_size": args.batch_size, "hidden_size": args.hidden_size,
            "grad_clip": args.grad_clip, "max_epochs": args.max_epochs, "patience": args.patience,
            "loss_on_predict_only": args.loss_on_predict_only, "use_length_bucketing": args.use_length_bucketing,
            "n_buckets": args.n_buckets, "condition": args.condition, "corruption_seed": args.corruption_seed,
            "dataset_manifest_sha256": dataset_manifest_sha256, "split_manifest_sha256": split_manifest_sha256,
            "n_train_sequences": len(train_seqs), "n_val_sequences": len(val_seqs),
            # A resume under a DIFFERENT dt policy/augmentation seed (or a
            # differently-thinned train set) must not silently continue.
            "dt_policy": dt_policy.name,
            "augmentation_seed": augmentation_seed,
        }
        if resume_checkpoint_path.is_file():
            print(f"resume checkpoint found at {resume_checkpoint_path} -- will resume if configuration matches",
                  file=sys.stderr)

    set_all_seeds(args.seed)
    try:
        result = train_one_run_batched(
            train_seqs, val_seqs, seed=args.seed, device=args.device, batch_size=args.batch_size,
            use_length_bucketing=args.use_length_bucketing, n_buckets=args.n_buckets, lr=args.lr,
            max_epochs=args.max_epochs, patience=args.patience, hidden_size=args.hidden_size,
            grad_clip=args.grad_clip, loss_on_predict_only=args.loss_on_predict_only,
            progress_callback=report_progress,
            resume_checkpoint_path=resume_checkpoint_path,
            resume_validation_key=resume_validation_key,
            checkpoint_every_epochs=args.checkpoint_every_epochs,
        )
    finally:
        if log_file is not None:
            log_file.close()

    args.output_history.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_history, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "grad_norm_mean",
                                                 "grad_norm_max", "any_nan_train", "any_nan_val",
                                                 "n_norm_overflow_batches", "n_element_nonfinite_batches"])
        writer.writeheader()
        for record in result.history:
            writer.writerow(vars(record))

    if result.best_state_dict is None:
        print("training produced no valid checkpoint (catastrophic run) -- not saving", file=sys.stderr)
        print(json.dumps({
            "catastrophic": True, "best_val": result.best_val,
            "training_collapsed": result.training_collapsed, "training_unstable": result.training_unstable,
            "abort_reason": result.abort_reason, "n_epochs_run": result.n_epochs_run,
            "dt_policy": dt_policy.name,
        }, indent=2))
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
            "training_unstable": result.training_unstable,
            "abort_reason": result.abort_reason,
            "grad_skip_count": result.grad_skip_count,
            "large_finite_gradient_count": result.large_finite_gradient_count,
            "norm_overflow_count": result.norm_overflow_count,
            "norm_overflow_skip_count": result.norm_overflow_skip_count,
            "per_element_nonfinite_gradient_count": result.per_element_nonfinite_gradient_count,
            "nonfinite_gradient_skip_count": result.nonfinite_gradient_skip_count,
            "parameter_collapse_count": result.parameter_collapse_count,
            "optimizer_state_collapse_count": result.optimizer_state_collapse_count,
            "training_collapsed": result.training_collapsed,
            # Variable-dt augmentation additions (section 17: "record
            # temporal augmentation policy, augmentation seed, dt
            # distribution" in the checkpoint manifest).
            "dt_policy": dt_policy.name,
            "dt_policy_skip_probs": list(dt_policy.skip_probs),
            "augmentation_seed": augmentation_seed,
            "n_train_thinned": n_thinned,
            "n_train_thinned_fraction": n_thinned / max(1, len(train_seqs)),
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
        "training_unstable": result.training_unstable,
        "training_collapsed": result.training_collapsed,
        "abort_reason": result.abort_reason,
        "train_time_s": result.train_time_s,
        "norm_overflow_count": result.norm_overflow_count,
        "per_element_nonfinite_gradient_count": result.per_element_nonfinite_gradient_count,
        "parameter_collapse_count": result.parameter_collapse_count,
        "optimizer_state_collapse_count": result.optimizer_state_collapse_count,
        "large_finite_gradient_count": result.large_finite_gradient_count,
        "n_train_sequences": len(train_seqs),
        "n_val_sequences": len(val_seqs),
        "batch_size": args.batch_size,
        "dt_policy": dt_policy.name,
        "augmentation_seed": augmentation_seed,
        "n_train_thinned_fraction": n_thinned / max(1, len(train_seqs)),
    }
    print(json.dumps(summary, indent=2))

    if args.output_internal_val_dt_report is not None:
        from evaluate_kalmannet import run_knet
        from ad_lidar_perception.kalmannet_core import KalmanNetGRU

        net = KalmanNetGRU(hidden_size=args.hidden_size).to(args.device)
        net.load_state_dict(result.best_state_dict)
        net.eval()

        report: dict[str, Any] = {}
        for eval_policy_name, eval_policy in DT_POLICY_CHOICES.items():
            eval_sequences = load_split_sequences_with_thinning(
                args.shard_root, split_manifest, eval_policy, augmentation_seed, corruption_config,
            )["val"]
            report[eval_policy_name] = evaluate_with_dt_buckets(
                eval_sequences, lambda s: run_knet(net, s, args.device)
            )
        args.output_internal_val_dt_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_internal_val_dt_report.write_text(json.dumps(report, indent=2, sort_keys=True))
        print("internal_val_dt_report:", json.dumps(report, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
