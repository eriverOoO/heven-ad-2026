#!/usr/bin/env python3
"""AV2 10k GENERIC-ROBUST KalmanNet training entry point with an OPT-IN
motion-composition weighted sampler (docs/perception/
kalmannet_av2_motion_composition_v1.md).

Mirrors ``train_kalmannet_batched.py`` exactly (same dataset/split/
corruption/frozen-hyperparameter CLI contract -- model/optimizer config
is NOT retuned here) and adds exactly one new axis: ``--motion-sampler
{natural,balanced,moving_focused}``. ``natural`` (the default) passes
``motion_sampler=None`` through to ``train_one_run_batched`` -- byte-
identical to the frozen 10k baseline's own training path; the NATURAL
condition of this experiment is therefore the EXISTING seed checkpoints,
never retrained here.

Motion categories are computed ONCE from each sequence's own GT
``x_true`` (corruption-independent) before training starts, using the
definition locked in ``motion_composition.py``
(``REPRESENTATIVE_SPEED_KEY="median_speed_mps"``, threshold 0.5 m/s) --
never from official VAL, never re-derived per epoch.

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

sys.path.insert(0, str(Path(__file__).resolve().parent))

import checkpoint_utils  # noqa: E402
from av2_split import load_split_manifest  # noqa: E402
from batched_trainer import train_one_run_batched  # noqa: E402
from kalmannet_sequences import class_distribution, load_split_sequences  # noqa: E402
from motion_composition import (  # noqa: E402
    BALANCED_MOTION,
    MOVING,
    MOVING_FOCUSED,
    NATURAL,
    NEAR_STATIONARY,
    classify_motion,
    compute_segment_motion_stats,
    weight_manifest_fingerprint,
)
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

MOTION_SAMPLER_CHOICES = {"natural": NATURAL, "balanced": BALANCED_MOTION, "moving_focused": MOVING_FOCUSED}

# Frame-level GT-speed threshold for ALL/NEAR_STATIONARY/MOVING internal-VAL
# reporting (section 8) -- kept in exact sync with evaluate_kalmannet_official_val's
# own MOTION_STATIONARY_THRESHOLD_MPS (locked equal by
# test_evaluate_motion_stratified_threshold_matches_locked_definition).
_MOTION_STATIONARY_THRESHOLD_MPS = 0.5


def compute_internal_val_motion_report(sequences: list, estimate_fn) -> dict:
    """Single pass over ``sequences`` (each sequence's own ``estimate_fn``
    call made exactly once, never duplicated) producing ALL / NEAR_STATIONARY
    / MOVING position+velocity RMSE, all values native Python floats (JSON
    serializable) -- the ALL bucket added on top of the exact same
    frame-level GT-speed bucketing ``evaluate_kalmannet_official_val.
    evaluate_motion_stratified`` already uses, so results are directly
    comparable to it (never re-derives a second, inconsistent threshold)."""
    import numpy as np

    buckets: dict[str, dict[str, list]] = {}

    def bucket(name):
        if name not in buckets:
            buckets[name] = {"pos_sq_err": [], "vel_sq_err": [], "n": 0}
        return buckets[name]

    for seq in sequences:
        est = estimate_fn(seq)
        for i in range(len(seq["frames"])):
            x_est = np.asarray(est[i], dtype=np.float64)
            x_gt = np.asarray(seq["x_true"][i], dtype=np.float64)
            if not np.all(np.isfinite(x_est)) or np.any(np.abs(x_est) > 1e6):
                continue
            speed = float(np.hypot(x_gt[2], x_gt[3]))
            name = "near_stationary" if speed < _MOTION_STATIONARY_THRESHOLD_MPS else "moving"
            pos_err = float(np.sum((x_est[:2] - x_gt[:2]) ** 2))
            vel_err = float(np.sum((x_est[2:] - x_gt[2:]) ** 2))
            for bucket_name in (name, "all"):
                b = bucket(bucket_name)
                b["pos_sq_err"].append(pos_err)
                b["vel_sq_err"].append(vel_err)
                b["n"] += 1

    result = {}
    for name, b in buckets.items():
        if b["n"] == 0:
            continue
        result[name] = {
            "n": b["n"],
            "position_rmse_m": float(np.sqrt(np.mean(b["pos_sq_err"]))),
            "velocity_rmse_mps": float(np.sqrt(np.mean(b["vel_sq_err"]))),
        }
    return result


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
    parser.add_argument("--motion-sampler", choices=list(MOTION_SAMPLER_CHOICES), default="natural",
                         help="'natural' (default) = byte-identical to the frozen baseline (motion_sampler=None); "
                              "'balanced' targets ~50%% moving draws/epoch; 'moving_focused' targets ~77.5%% "
                              "moving draws/epoch. Draw count per epoch always equals len(train_seqs) (section 11).")
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--output-history", type=Path, required=True)
    parser.add_argument("--output-internal-val-motion-report", type=Path, default=None,
                         help="optional: write a small JSON with internal-VAL ALL/NEAR_STATIONARY/MOVING "
                              "position+velocity RMSE for the FINAL best checkpoint (section 8) -- frame-level "
                              "GT-speed stratification, reusing evaluate_kalmannet_official_val.evaluate_motion_stratified "
                              "verbatim, never official VAL.")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--no-resume", dest="resume_enabled", action="store_false", default=True)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=1)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--num-threads", type=int, default=None,
                         help="optional torch.set_num_threads() override -- pure infrastructure/performance "
                              "setting, never affects results/comparability (default: torch's own default, "
                              "unchanged from every prior run in this task).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

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

    sequences = load_split_sequences(args.shard_root, split_manifest, corruption_config=corruption_config)
    train_seqs, val_seqs = sequences["train"], sequences["val"]
    print(f"train={len(train_seqs)} val={len(val_seqs)} test={len(sequences['test'])} sequences", file=sys.stderr)
    print(f"train class distribution: {class_distribution(train_seqs)}", file=sys.stderr)

    # Motion categories computed ONCE from GT x_true (corruption-independent),
    # using the locked definition (median speed, 0.5 m/s) -- never from
    # official VAL, never re-derived per epoch.
    motion_sampler_cfg = MOTION_SAMPLER_CHOICES[args.motion_sampler]
    train_categories = [classify_motion(compute_segment_motion_stats(s)) for s in train_seqs]
    n_moving = train_categories.count(MOVING)
    n_stationary = train_categories.count(NEAR_STATIONARY)
    natural_moving_fraction = n_moving / len(train_categories)
    sampler_fingerprint = weight_manifest_fingerprint(train_categories, motion_sampler_cfg)
    print(
        f"motion_sampler={motion_sampler_cfg.name} target_moving_fraction={motion_sampler_cfg.target_moving_fraction} "
        f"natural_moving_fraction={natural_moving_fraction:.6f} n_moving={n_moving} n_stationary={n_stationary} "
        f"fingerprint={sampler_fingerprint}",
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
            # New: a resume under a DIFFERENT motion-sampler policy (or a
            # differently-composed train set) must not silently continue.
            "motion_sampler": motion_sampler_cfg.name,
            "motion_sampler_fingerprint": sampler_fingerprint,
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
            motion_sampler=(None if motion_sampler_cfg.target_moving_fraction is None else motion_sampler_cfg),
            motion_categories=(None if motion_sampler_cfg.target_moving_fraction is None else train_categories),
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
            "motion_sampler": motion_sampler_cfg.name,
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
            # Motion-composition ablation additions (section 10: "sequence
            # sampling weights recorded in checkpoint manifest").
            "motion_sampler": motion_sampler_cfg.name,
            "motion_sampler_target_moving_fraction": motion_sampler_cfg.target_moving_fraction,
            "motion_sampler_fingerprint": sampler_fingerprint,
            "motion_natural_moving_fraction": natural_moving_fraction,
            "motion_n_moving_segments": n_moving,
            "motion_n_stationary_segments": n_stationary,
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
        "motion_sampler": motion_sampler_cfg.name,
        "motion_sampler_target_moving_fraction": motion_sampler_cfg.target_moving_fraction,
        "motion_natural_moving_fraction": natural_moving_fraction,
    }
    print(json.dumps(summary, indent=2))

    if args.output_internal_val_motion_report is not None:
        from evaluate_kalmannet import run_knet
        from ad_lidar_perception.kalmannet_core import KalmanNetGRU

        net = KalmanNetGRU(hidden_size=args.hidden_size).to(args.device)
        net.load_state_dict(result.best_state_dict)
        net.eval()
        motion_report = compute_internal_val_motion_report(val_seqs, lambda s: run_knet(net, s, args.device))
        args.output_internal_val_motion_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_internal_val_motion_report.write_text(json.dumps(motion_report, indent=2, sort_keys=True))
        print("internal_val_motion_stratified:", json.dumps(motion_report, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
