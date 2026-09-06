#!/usr/bin/env python3
"""Held-out OFFICIAL AV2 VAL evaluation for the AV2 Scale-Up v2 10k
GENERIC-ROBUST KalmanNet checkpoint (and comparison baselines), on the
frozen 1,000-scenario official-VAL shard root (PR #54) -- a physically
separate scenario pool from the 9,000/1,000 internal TRAIN/VAL split,
never read for training, early stopping, or KF calibration.

Reuses ``evaluate_kalmannet.py``'s own ``run_kf``/``run_knet``/
``evaluate_estimator``/``GAP_BUCKETS``/``DENSE_V2_CHECKPOINT_PATH``/
``TUNED_HEVEN_KF_*`` verbatim (no metric/estimator logic duplicated) and
``calibrate_kf.calibrate_kf_on_av2`` for the fair AV2-tuned KF baseline.
Adds only what is new to this task: (1) a physically separate
calibration root (internal TRAIN/VAL, 9k/1k) from the evaluation root
(official VAL, 1k), (2) a mandatory freeze-manifest-exists guard before
any official-VAL sequence is read, (3) a motion-stratified (near-
stationary vs. moving) breakdown official-VAL evaluation did not
previously need.

**No MORAI evaluation. No claim that this checkpoint improves MORAI or
competition performance -- AV2 official-VAL metrics only.**
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import checkpoint_utils  # noqa: E402
from av2_split import load_split_manifest  # noqa: E402
from calibrate_kf import calibrate_kf_on_av2  # noqa: E402
from evaluate_kalmannet import (  # noqa: E402
    DENSE_V2_CHECKPOINT_PATH,
    TUNED_HEVEN_KF_P0_SCALE,
    TUNED_HEVEN_KF_PROVENANCE,
    TUNED_HEVEN_KF_R_STD,
    TUNED_HEVEN_KF_SIGMA_A,
    evaluate_estimator,
    run_kf,
    run_knet,
)
from freeze_manifest import require_freeze_manifest_exists  # noqa: E402
from kalmannet_sequences import load_split_sequences  # noqa: E402

from ad_morai_bridge_dev.dataset.av2_motion_forecasting_adapter import CorruptionConfig  # noqa: E402

_AD_LIDAR_PKG_DIR = Path(__file__).resolve().parents[2] / "ad_lidar_perception"
if str(_AD_LIDAR_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_AD_LIDAR_PKG_DIR))
from ad_lidar_perception.kalmannet_core import KalmanNetGRU  # noqa: E402

MOTION_STATIONARY_THRESHOLD_MPS = 0.5


def evaluate_motion_stratified(sequences: list[dict], estimate_fn) -> dict:
    """Descriptive-only speed-bin breakdown (near-stationary vs. moving,
    from GT [vx,vy] -- never used for model selection). Reuses the exact
    same squared-error accumulation semantics as ``evaluate_estimator``,
    just bucketed by GT speed at each frame instead of by matched/missing
    or class."""
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
            name = "near_stationary" if speed < MOTION_STATIONARY_THRESHOLD_MPS else "moving"
            pos_err = float(np.sum((x_est[:2] - x_gt[:2]) ** 2))
            vel_err = float(np.sum((x_est[2:] - x_gt[2:]) ** 2))
            b = bucket(name)
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
    parser.add_argument("--freeze-manifest", type=Path, required=True,
                         help="must already exist -- see write_freeze_manifest_av2_10k_generic.py")
    parser.add_argument("--checkpoint", type=Path, required=True, help="checkpoint to evaluate (checkpoint A/B/C)")
    parser.add_argument("--training-condition", choices=["clean", "generic_robust", "morai_calibrated_robust"],
                         required=True)
    parser.add_argument("--calibration-shard-root", type=Path, required=True,
                         help="TRAIN-source shard root (10k), used ONLY for the AV2-tuned KF calibration "
                              "(internal TRAIN + internal VAL, never official VAL)")
    parser.add_argument("--calibration-split-manifest", type=Path, required=True,
                         help="the 9,000/1,000 internal TRAIN/VAL split manifest")
    parser.add_argument("--official-val-shard-root", type=Path, required=True)
    parser.add_argument("--official-val-split-manifest", type=Path, required=True)
    parser.add_argument("--official-val-split-name", default="val",
                         help="which bucket of --official-val-split-manifest holds the 1,000 scenarios "
                              "(single-role manifest convention: 'val')")
    parser.add_argument("--eval-corruption-seed-variant", type=int, default=999999)
    parser.add_argument("--add-morai-calibrated-diagnostic", action="store_true",
                         help="add condition D: MORAI-calibrated corruption, evaluation-only diagnostic. "
                              "NOT a MORAI evaluation -- only tests tolerance to a more severe, anisotropic "
                              "measurement-corruption family, using the already-frozen corruption config.")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-dense-v2", action="store_true")
    parser.add_argument("--skip-av2-kf-calibration", action="store_true")
    parser.add_argument("--precomputed-kf-calibration-json", type=Path, default=None,
                         help="reuse an AV2-tuned KF calibration already written by a previous invocation of this "
                              "script (avoids re-running the calibration grid search for every baseline "
                              "checkpoint) instead of recalibrating")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Hard guard: refuse to touch official VAL before a freeze manifest exists.
    require_freeze_manifest_exists(args.freeze_manifest)

    state_dict, ckpt_manifest = checkpoint_utils.load_checkpoint(args.checkpoint)
    hidden_size = ckpt_manifest.get("model_config", {}).get("hidden_size", 32)
    net = KalmanNetGRU(hidden_size=hidden_size).to(args.device)
    net.load_state_dict(state_dict)
    net.eval()

    dense_v2_net = None
    if not args.skip_dense_v2 and DENSE_V2_CHECKPOINT_PATH.is_file():
        raw = torch.load(DENSE_V2_CHECKPOINT_PATH, map_location=args.device)
        sd = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
        dense_v2_net = KalmanNetGRU(hidden_size=32).to(args.device)
        dense_v2_net.load_state_dict(sd)
        dense_v2_net.eval()

    from train_kalmannet import CONDITION_PRESETS

    conditions = {"A_clean_official_val": CorruptionConfig(mode="none")}
    if args.training_condition in ("generic_robust", "morai_calibrated_robust"):
        train_corruption = CONDITION_PRESETS[args.training_condition]
        conditions["B_same_corruption_family"] = train_corruption
        conditions["C_different_corruption_seed"] = type(train_corruption)(
            **{**train_corruption.__dict__, "seed": args.eval_corruption_seed_variant}
        )
    if args.add_morai_calibrated_diagnostic:
        conditions["D_morai_calibrated_diagnostic"] = CONDITION_PRESETS["morai_calibrated_robust"]

    # ---- AV2-tuned KF: calibrated ONLY on the internal TRAIN/VAL split
    # of the 10k TRAIN-source pool -- official VAL is NEVER read here. ----
    av2_kf_calibration = None
    if args.precomputed_kf_calibration_json is not None:
        calib_json = json.loads(args.precomputed_kf_calibration_json.read_text("utf-8"))
        from calibrate_kf import KFCalibrationResult
        av2_kf_calibration = KFCalibrationResult(**calib_json["linear_kf_av2_tuned"]["params"] | {
            "train_residual_sigma_z": calib_json["linear_kf_av2_tuned"]["train_residual_sigma_z"],
            "val_position_rmse": calib_json["linear_kf_av2_tuned"]["internal_val_position_rmse"],
            "val_velocity_rmse": calib_json["linear_kf_av2_tuned"]["internal_val_velocity_rmse"],
            "n_candidates": calib_json["linear_kf_av2_tuned"]["n_candidates"],
            "calibrated_on_condition": "generic_robust",
        })
    elif not args.skip_av2_kf_calibration:
        calib_split_manifest = load_split_manifest(args.calibration_split_manifest)
        calib_corruption = CONDITION_PRESETS["generic_robust"]
        calib_sequences = load_split_sequences(
            args.calibration_shard_root, calib_split_manifest, corruption_config=calib_corruption
        )
        assert calib_sequences["test"] == [], "internal split must never populate a 'test' bucket"
        av2_kf_calibration = calibrate_kf_on_av2(
            calib_sequences["train"], calib_sequences["val"], condition_name="generic_robust"
        )
        print(f"AV2-tuned KF calibration (internal TRAIN/VAL only, official VAL untouched): "
              f"sigma_a={av2_kf_calibration.sigma_a} r_std={av2_kf_calibration.r_std:.6f} "
              f"val_pos_rmse={av2_kf_calibration.val_position_rmse:.4f}", file=sys.stderr)

    # ---- OFFICIAL AV2 VAL sequences (read only here, after calibration). ----
    official_val_split_manifest = load_split_manifest(args.official_val_split_manifest)

    report: dict = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_manifest": ckpt_manifest,
        "official_val_shard_root": str(args.official_val_shard_root),
        "official_val_split_manifest": str(args.official_val_split_manifest),
        "linear_kf_transferred_from_morai": {
            "provenance": TUNED_HEVEN_KF_PROVENANCE,
            "params": {"sigma_a": TUNED_HEVEN_KF_SIGMA_A, "r_std": TUNED_HEVEN_KF_R_STD,
                       "p0_scale": TUNED_HEVEN_KF_P0_SCALE},
        },
        "linear_kf_av2_tuned": (
            {
                "provenance": "calibrated on the internal 9,000-scenario TRAIN + 1,000-scenario internal VAL "
                               "split of the 10k TRAIN-source pool (generic_robust condition) -- official AV2 "
                               "VAL never touched",
                "params": {"sigma_a": av2_kf_calibration.sigma_a, "r_std": av2_kf_calibration.r_std,
                           "p0_scale": av2_kf_calibration.p0_scale},
                "train_residual_sigma_z": av2_kf_calibration.train_residual_sigma_z,
                "internal_val_position_rmse": av2_kf_calibration.val_position_rmse,
                "internal_val_velocity_rmse": av2_kf_calibration.val_velocity_rmse,
                "n_candidates": av2_kf_calibration.n_candidates,
            }
            if av2_kf_calibration is not None else None
        ),
        "dense_v2_evaluated": dense_v2_net is not None,
        "dense_v2_checkpoint_path": str(DENSE_V2_CHECKPOINT_PATH) if dense_v2_net is not None else None,
        "conditions": {},
    }

    for cond_name, corruption_config in conditions.items():
        sequences = load_split_sequences(
            args.official_val_shard_root, official_val_split_manifest, corruption_config=corruption_config
        )
        eval_seqs = sequences[args.official_val_split_name]
        cond_report = {
            "n_official_val_sequences": len(eval_seqs),
            "corruption_config": json.loads(corruption_config.canonical_json()),
            "knet": evaluate_estimator(eval_seqs, lambda s: run_knet(net, s, args.device)),
            "linear_kf_transferred_from_morai": evaluate_estimator(eval_seqs, run_kf),
            "knet_motion_stratified": evaluate_motion_stratified(eval_seqs, lambda s: run_knet(net, s, args.device)),
        }
        if av2_kf_calibration is not None:
            calib = av2_kf_calibration
            cond_report["linear_kf_av2_tuned"] = evaluate_estimator(
                eval_seqs, lambda s: run_kf(s, sigma_a=calib.sigma_a, r_std=calib.r_std, p0_scale=calib.p0_scale)
            )
        if dense_v2_net is not None:
            cond_report["dense_v2_cross_domain_diagnostic"] = evaluate_estimator(
                eval_seqs, lambda s: run_knet(dense_v2_net, s, args.device)
            )
        report["conditions"][cond_name] = cond_report
        knet_overall = cond_report["knet"]["buckets"].get("overall", {})
        print(f"[{cond_name}] n={len(eval_seqs)} knet_pos_rmse={knet_overall.get('position_rmse_m')} "
              f"knet_vel_rmse={knet_overall.get('velocity_rmse_mps')} "
              f"divergence={cond_report['knet']['divergence_count']} "
              f"nonfinite={cond_report['knet']['nonfinite_count']}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
