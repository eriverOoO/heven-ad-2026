#!/usr/bin/env python3
"""Writes the freeze manifest for the MORAI-Calibrated AV2 Corruption v1
checkpoint (av2_stage1_pilot_morai_calibrated_robust_bs64_lr004_seed0.pt).

Uses ONLY the checkpoint's own manifest (built from AV2 TRAIN/VAL
selection -- ``best_epoch``/``best_val_loss`` are AV2-VAL-selected, never
influenced by AV2 TEST or any MORAI split) plus the MORAI TRAIN/VAL
calibration provenance recorded in ``morai_calibration.py``. Run this
BEFORE any AV2 TEST or MORAI TEST evaluation of this checkpoint --
``evaluate_morai_frozen_stream.py`` refuses to run without it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import checkpoint_utils  # noqa: E402
from freeze_manifest import build_freeze_manifest, save_freeze_manifest  # noqa: E402
from morai_calibration import (  # noqa: E402
    MORAI_CALIBRATED_BURST_PROB,
    MORAI_CALIBRATION_TRAIN_ACTOR_IDS,
    MORAI_CALIBRATION_VAL_ACTOR_IDS,
    MORAI_FROZEN_TEST_ACTOR_IDS,
    build_morai_calibrated_corruption_config,
)

CHECKPOINT_PATH = Path.home() / "datasets/av2/checkpoints/av2_stage1_pilot_morai_calibrated_robust_bs64_lr004_seed0.pt"
FREEZE_PATH = Path(__file__).resolve().parent / "frozen_configs" / "kalmannet_morai_calibrated_corruption_v1_freeze.json"


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2]).decode().strip()


def main() -> int:
    _, manifest = checkpoint_utils.load_checkpoint(CHECKPOINT_PATH)
    checkpoint_sha256 = checkpoint_utils.sha256_file(CHECKPOINT_PATH)
    config = build_morai_calibrated_corruption_config()

    freeze = build_freeze_manifest(
        batch_size=manifest["training_config"]["batch_size"],
        learning_rate=manifest["training_config"]["lr"],
        max_epochs=manifest["training_config"]["max_epochs"],
        patience=manifest["training_config"]["patience"],
        gradient_clip=manifest["training_config"]["grad_clip"],
        loss_on_predict_only=manifest["loss_on_predict_only"],
        bucket_policy={
            "use_length_bucketing": manifest["training_config"]["use_length_bucketing"],
            "n_buckets": manifest["training_config"]["n_buckets"],
            "sampler_version": manifest["training_config"]["sampler_version"],
        },
        shuffle_seed_policy={"seed": manifest["seed_info"]["seed"], "init_seed": manifest["seed_info"]["init_seed"],
                              "order_seed": manifest["seed_info"]["order_seed"]},
        corruption_policy=manifest["corruption_config"],
        model_architecture=manifest["model_config"],
        selection_evidence={
            "best_epoch": manifest["best_epoch"],
            "best_val_loss": manifest["best_val_loss"],
            "selected_on": "AV2 VAL only (early stopping) -- AV2 TEST and every MORAI split untouched",
            "morai_calibration_provenance": {
                "corruption_model_version": "corruption_v2",
                "calibration_source": "MORAI TRAIN+VAL sequences.pkl (original, non-dense-filtered)",
                "calibration_train_actor_ids": list(MORAI_CALIBRATION_TRAIN_ACTOR_IDS),
                "calibration_val_actor_ids": list(MORAI_CALIBRATION_VAL_ACTOR_IDS),
                "frozen_test_actor_ids_never_touched": list(MORAI_FROZEN_TEST_ACTOR_IDS),
                "bias_xy": list(config.bias_xy),
                "covariance_xy": [list(row) for row in config.covariance_xy],
                "dropout_prob": config.dropout_prob,
                "dropout_burst_prob_renewal_corrected": MORAI_CALIBRATED_BURST_PROB,
                "dropout_burst_length_samples_n": len(config.dropout_burst_length_samples),
                "corruption_config_canonical_json": config.canonical_json(),
            },
        },
        three_seed_validation_summary={
            "note": "single-seed run per this task's explicit scope "
                     "('train ONE MORAI-CALIBRATED-ROBUST KNet, one seed only') -- "
                     "no multi-seed stability claim is made for this checkpoint",
        },
        git_sha=_git_sha(),
        notes=(
            "MORAI-Calibrated AV2 Corruption v1 -- AV2 Stage-1 pilot (2000 scenarios, seed 0), "
            f"MORAI TRAIN+VAL-calibrated corruption (bias+full-covariance noise, empirical burst-length "
            f"resampling). Checkpoint SHA-256 {checkpoint_sha256}. "
            "Written BEFORE any AV2 TEST or MORAI TEST evaluation of this checkpoint."
        ),
    )
    save_freeze_manifest(freeze, FREEZE_PATH)
    print(f"wrote {FREEZE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
