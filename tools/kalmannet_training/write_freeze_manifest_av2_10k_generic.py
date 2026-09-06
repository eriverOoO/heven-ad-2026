#!/usr/bin/env python3
"""Writes the AV2 Scale-Up v2 10k GENERIC-ROBUST freeze manifest, in the
SAME ``kalmannet_batch_calibration_freeze_v1`` schema every prior AV2
freeze manifest in this project uses (``freeze_manifest.py``, unmodified
-- reused verbatim, not re-invented). Must be run AFTER the best
checkpoint has been selected purely from INTERNAL VALIDATION (never
official AV2 VAL) and BEFORE any official AV2 VAL evaluation.

``evaluate_kalmannet_official_val.py`` calls
``freeze_manifest.require_freeze_manifest_exists`` first and refuses to
run if this file does not exist yet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import checkpoint_utils  # noqa: E402
from freeze_manifest import build_freeze_manifest, save_freeze_manifest  # noqa: E402


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
            check=True, text=True, capture_output=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                         help="the selected av2_stage0_kalmannet_batched_v1 checkpoint (best_epoch already chosen "
                              "from internal validation only)")
    parser.add_argument("--train-scenario-manifest-sha256", required=True)
    parser.add_argument("--internal-split-manifest", type=Path, required=True)
    parser.add_argument("--official-val-scenario-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    state_dict, ckpt_manifest = checkpoint_utils.load_checkpoint(args.checkpoint)
    internal_split = json.loads(args.internal_split_manifest.read_text("utf-8"))
    internal_split_sha256 = checkpoint_utils.sha256_json(internal_split)

    if ckpt_manifest.get("split_manifest_sha256") != internal_split_sha256:
        raise ValueError(
            f"checkpoint's own recorded split_manifest_sha256 "
            f"({ckpt_manifest.get('split_manifest_sha256')}) does not match the internal split manifest passed "
            f"here ({internal_split_sha256}) -- wrong checkpoint/split pairing, refusing to freeze"
        )

    training_config = ckpt_manifest["training_config"]
    checkpoint_sha256 = _sha256_file(args.checkpoint)

    manifest = build_freeze_manifest(
        batch_size=training_config["batch_size"],
        learning_rate=training_config["lr"],
        max_epochs=training_config["max_epochs"],
        patience=training_config["patience"],
        gradient_clip=training_config["grad_clip"],
        loss_on_predict_only=ckpt_manifest["loss_on_predict_only"],
        bucket_policy={
            "n_buckets": training_config["n_buckets"],
            "sampler_version": training_config["sampler_version"],
            "use_length_bucketing": training_config["use_length_bucketing"],
        },
        shuffle_seed_policy={
            "seed": ckpt_manifest["seed_info"]["seed"],
            "init_seed": ckpt_manifest["seed_info"]["init_seed"],
            "order_seed": ckpt_manifest["seed_info"]["order_seed"],
        },
        corruption_policy=ckpt_manifest["corruption_config"],
        model_architecture={
            "architecture": ckpt_manifest["architecture"],
            "state_fields": ckpt_manifest["state_fields"],
            "measurement_fields": ckpt_manifest["measurement_fields"],
            **ckpt_manifest["model_config"],
        },
        selection_evidence={
            "dataset": "av2_kalmannet_scaleup_v2 TRAIN-source pool (10,000 scenarios), PR #54",
            "train_scenario_manifest_sha256": args.train_scenario_manifest_sha256,
            "official_val_scenario_manifest_sha256": args.official_val_scenario_manifest_sha256,
            "internal_split_manifest_path_machine_local": str(args.internal_split_manifest),
            "internal_split_manifest_sha256": internal_split_sha256,
            "internal_split_counts": internal_split["split_counts"],
            "internal_split_seed": internal_split["config"]["seed"],
            "verified_overlap_with_official_val": internal_split.get("verified_overlap_with_official_val"),
            "best_epoch": ckpt_manifest["best_epoch"],
            "internal_validation_best_loss": ckpt_manifest["best_val_loss"],
            "internal_validation_metric": "mean per-frame MSE over [x,y,vx,vy], loss_on_predict_only semantics "
                                           "-- NEVER official AV2 VAL",
            "checkpoint_path": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "dataset_manifest_sha256_recorded_in_checkpoint": ckpt_manifest["dataset_manifest_sha256"],
            "n_train_sequences": training_config["n_train_sequences"],
            "n_internal_val_sequences": training_config["n_val_sequences"],
            "torch_version": ckpt_manifest["environment"]["torch_version"],
            "device_used_for_training": ckpt_manifest["environment"].get("cuda_available"),
        },
        three_seed_validation_summary={
            "note": "ONE initialization seed only, per this task's own explicit instruction -- no seed-selection "
                    "decision was made (no ambiguity to protect from official-VAL influence). seed=0.",
            "seeds_run": [ckpt_manifest["seed_info"]["seed"]],
        },
        git_sha=_git_sha(),
        notes=(
            "AV2 KalmanNet Full-Scale 10K Generic-Robust Pretraining v1 -- 10,000 fresh AV2 TRAIN-source "
            "scenarios (9,000 internal-train / 1,000 internal-val), 1 seed, GENERIC-ROBUST corruption. "
            f"Checkpoint SHA-256 {checkpoint_sha256}. No MORAI evaluation is available in this environment -- "
            "this freeze covers AV2 official-VAL evaluation only, never a MORAI/competition-performance claim."
        ),
    )
    save_freeze_manifest(manifest, args.output)
    print(json.dumps({"output": str(args.output), "best_epoch": manifest.selection_evidence["best_epoch"],
                       "checkpoint_sha256": checkpoint_sha256}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
