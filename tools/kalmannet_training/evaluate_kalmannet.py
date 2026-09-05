#!/usr/bin/env python3
"""Held-out AV2 Stage-0 evaluation: trained KalmanNetGRU vs. the tuned
``LinearCVKF`` baseline (and, if locally available, the frozen
DENSE-KALMANNET-v2 checkpoint as a cross-domain diagnostic) on the exact
same held-out sequence stream.

**Stage-0 sanity diagnostic only.** No claim that AV2 pretraining
improves MORAI performance, and no claim that a MORAI-trained checkpoint
evaluated on AV2 (or vice versa) proves anything about simulator
superiority -- both are reported, neither is spun.
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
from kalmannet_sequences import load_split_sequences  # noqa: E402

from ad_morai_bridge_dev.dataset.av2_motion_forecasting_adapter import CorruptionConfig  # noqa: E402

_AD_LIDAR_PKG_DIR = Path(__file__).resolve().parents[2] / "ad_lidar_perception"
if str(_AD_LIDAR_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_AD_LIDAR_PKG_DIR))
from ad_lidar_perception.kalmannet_core import KalmanNetFilter, KalmanNetGRU, LinearCVKF  # noqa: E402

# Tuned HEVEN Linear KF values (T-9A.1's own validation-only selection,
# ~/heven_presentation_assets/state_estimator_gt_comparison/
# selected_kf_config.json) -- **a cross-domain TRANSFERRED baseline**,
# never retuned on any AV2 data (train, val, or test). Tuned on MORAI
# HEVEN actors, a different domain -- reported as provenance, not claimed
# optimal, and not the fair AV2-native comparison point (see
# ``calibrate_kf.py``/``linear_kf_av2_tuned`` below for that).
TUNED_HEVEN_KF_SIGMA_A = 5.0
TUNED_HEVEN_KF_R_STD = 0.2254682076528578
TUNED_HEVEN_KF_P0_SCALE = 10.0
TUNED_HEVEN_KF_PROVENANCE = (
    "T-9A.1 selected_kf_config.json, validation-only selection on MORAI HEVEN "
    "actors 1/13/15/18/35/36/38/40/44/46/48 (train) + 21/23/27/31 (val); "
    "CROSS-DOMAIN TRANSFERRED to AV2 without retuning -- NOT the fair AV2-tuned baseline"
)

DENSE_V2_CHECKPOINT_PATH = (
    Path.home() / "heven_presentation_assets" / "kalmannet_training_stability" / "checkpoint" / "dense_kalmannet_v2.pt"
)

GAP_BUCKETS = [(0, 0, "matched"), (1, 2, "gap_1_2"), (3, 5, "gap_3_5"), (6, None, "gap_6_plus")]


def run_kf(seq, sigma_a=TUNED_HEVEN_KF_SIGMA_A, r_std=TUNED_HEVEN_KF_R_STD, p0_scale=TUNED_HEVEN_KF_P0_SCALE):
    """Default params are the cross-domain TRANSFERRED (MORAI-tuned)
    baseline; pass the AV2-tuned triple from ``calibrate_kf_on_av2`` for
    the fair AV2-native comparison."""
    kf = LinearCVKF(sigma_a=sigma_a, r_std=r_std, p0_scale=p0_scale)
    first = seq["z_meas"][0]
    kf.init_sequence(np.array([first[0], first[1], 0.0, 0.0]))
    out = []
    for i in range(len(seq["frames"])):
        dt, z = seq["dt"][i], seq["z_meas"][i]
        if dt is not None:
            kf.predict(dt)
            if z is not None:
                kf.update(np.array(z))
        out.append(kf.x.copy())
    return out


def run_knet(net, seq, device):
    kf = KalmanNetFilter(net, device=device)
    first = seq["z_meas"][0]
    x0 = torch.tensor([first[0], first[1], 0.0, 0.0], dtype=torch.float32, device=device).unsqueeze(0)
    kf.init_sequence(x0, batch_size=1)
    out = []
    with torch.no_grad():
        for i in range(len(seq["frames"])):
            dt, z = seq["dt"][i], seq["z_meas"][i]
            if dt is not None:
                if z is not None:
                    zt = torch.tensor(z, dtype=torch.float32, device=device).unsqueeze(0)
                    kf.step(zt, dt)
                else:
                    kf.x_post = KalmanNetFilter._f(kf.x_post, dt)
            out.append(kf.x_post[0].detach().cpu().numpy().copy())
    return out


def _gap_length_at(seq, i: int) -> int:
    """How many consecutive frames (including this one) have had a
    missing measurement, counting back from frame i."""
    if seq["z_meas"][i] is not None:
        return 0
    n = 0
    j = i
    while j >= 0 and seq["z_meas"][j] is None:
        n += 1
        j -= 1
    return n


def _bucket_for_gap(gap_len: int) -> str:
    for lo, hi, name in GAP_BUCKETS:
        if hi is None:
            if gap_len >= lo:
                return name
        elif lo <= gap_len <= hi:
            return name
    return "gap_6_plus"


def evaluate_estimator(sequences: list[dict], estimate_fn) -> dict:
    """Returns per-category squared-error accumulators; RMSE computed by
    the caller once all sequences/conditions are pooled or stratified."""
    buckets: dict[str, dict[str, list]] = {}

    def bucket(name):
        if name not in buckets:
            buckets[name] = {"pos_sq_err": [], "vel_sq_err": [], "n": 0}
        return buckets[name]

    divergence_count = 0
    nonfinite_count = 0

    for seq in sequences:
        est = estimate_fn(seq)
        for i in range(len(seq["frames"])):
            x_est = np.asarray(est[i], dtype=np.float64)
            x_gt = np.asarray(seq["x_true"][i], dtype=np.float64)
            if not np.all(np.isfinite(x_est)):
                nonfinite_count += 1
                continue
            if np.any(np.abs(x_est) > 1e6):
                divergence_count += 1
                continue
            pos_err = float(np.sum((x_est[:2] - x_gt[:2]) ** 2))
            vel_err = float(np.sum((x_est[2:] - x_gt[2:]) ** 2))

            b_all = bucket("overall")
            b_all["pos_sq_err"].append(pos_err); b_all["vel_sq_err"].append(vel_err); b_all["n"] += 1

            matched = seq["z_meas"][i] is not None
            b_mm = bucket("matched" if matched else "missing")
            b_mm["pos_sq_err"].append(pos_err); b_mm["vel_sq_err"].append(vel_err); b_mm["n"] += 1

            gap_bucket = _bucket_for_gap(_gap_length_at(seq, i))
            b_gap = bucket(f"gap::{gap_bucket}")
            b_gap["pos_sq_err"].append(pos_err); b_gap["vel_sq_err"].append(vel_err); b_gap["n"] += 1

            coarse = seq.get("coarse_object_type")
            if coarse:
                b_cls = bucket(f"class::{coarse}")
                b_cls["pos_sq_err"].append(pos_err); b_cls["vel_sq_err"].append(vel_err); b_cls["n"] += 1

    result = {"divergence_count": divergence_count, "nonfinite_count": nonfinite_count, "buckets": {}}
    for name, b in buckets.items():
        if b["n"] == 0:
            continue
        result["buckets"][name] = {
            "n": b["n"],
            "position_rmse_m": float(np.sqrt(np.mean(b["pos_sq_err"]))),
            "velocity_rmse_mps": float(np.sqrt(np.mean(b["vel_sq_err"]))),
        }
    return result


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="AV2 Stage-0 KNet checkpoint to evaluate")
    parser.add_argument("--training-condition", choices=["clean", "generic_robust", "morai_calibrated_robust"],
                         required=True,
                         help="which corruption family this checkpoint was trained under (for condition-B eval)")
    parser.add_argument("--eval-corruption-seed-variant", type=int, default=999999,
                         help="a different corruption seed for condition C (must differ from training's own seed)")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-dense-v2", action="store_true")
    parser.add_argument("--skip-av2-kf-calibration", action="store_true",
                         help="skip the fair AV2-tuned KF baseline (report the transferred-from-MORAI KF only)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    split_manifest = load_split_manifest(args.split_manifest)

    state_dict, manifest = checkpoint_utils.load_checkpoint(args.checkpoint)
    hidden_size = manifest.get("model_config", {}).get("hidden_size", 32)
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

    conditions = {
        "A_clean_held_out": CorruptionConfig(mode="none"),
    }
    if args.training_condition in ("generic_robust", "morai_calibrated_robust"):
        from train_kalmannet import CONDITION_PRESETS
        train_corruption = CONDITION_PRESETS[args.training_condition]
        conditions["B_same_corruption_family"] = train_corruption
        conditions["C_different_corruption_seed"] = type(train_corruption)(
            **{**train_corruption.__dict__, "seed": args.eval_corruption_seed_variant}
        )

    # Fair AV2-tuned KF: sigma_a/sigma_z calibrated on AV2 TRAIN+VAL only
    # (never held-out test), under the GENERIC-ROBUST corruption family --
    # the only condition with real, non-degenerate measurement-noise
    # residuals to calibrate against (CLEAN residuals are exactly zero by
    # construction). Calibrated once here, then applied unchanged to
    # every held-out condition below (including CLEAN) -- this is
    # data-only work, independent of which checkpoint is being evaluated.
    av2_kf_calibration = None
    if not args.skip_av2_kf_calibration:
        from train_kalmannet import CONDITION_PRESETS
        calib_corruption = CONDITION_PRESETS["generic_robust"]
        calib_sequences = load_split_sequences(args.shard_root, split_manifest, corruption_config=calib_corruption)
        av2_kf_calibration = calibrate_kf_on_av2(
            calib_sequences["train"], calib_sequences["val"], condition_name="generic_robust"
        )
        print(f"AV2-tuned KF calibration: sigma_a={av2_kf_calibration.sigma_a} "
              f"r_std={av2_kf_calibration.r_std:.6f} (train sigma_z={av2_kf_calibration.train_residual_sigma_z:.6f}) "
              f"val_pos_rmse={av2_kf_calibration.val_position_rmse:.4f} "
              f"[{av2_kf_calibration.n_candidates} candidates, TRAIN+VAL only, test never touched]",
              file=sys.stderr)

    report: dict = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_manifest": manifest,
        "linear_kf_transferred_from_morai": {
            "provenance": TUNED_HEVEN_KF_PROVENANCE,
            "params": {"sigma_a": TUNED_HEVEN_KF_SIGMA_A, "r_std": TUNED_HEVEN_KF_R_STD,
                       "p0_scale": TUNED_HEVEN_KF_P0_SCALE},
        },
        "linear_kf_av2_tuned": (
            {
                "provenance": "calibrated on AV2 TRAIN+VAL only (generic_robust condition), "
                               "held-out TEST never touched",
                "params": {"sigma_a": av2_kf_calibration.sigma_a, "r_std": av2_kf_calibration.r_std,
                           "p0_scale": av2_kf_calibration.p0_scale},
                "train_residual_sigma_z": av2_kf_calibration.train_residual_sigma_z,
                "val_position_rmse": av2_kf_calibration.val_position_rmse,
                "val_velocity_rmse": av2_kf_calibration.val_velocity_rmse,
                "n_candidates": av2_kf_calibration.n_candidates,
            }
            if av2_kf_calibration is not None else None
        ),
        "dense_v2_evaluated": dense_v2_net is not None,
        "dense_v2_checkpoint_path": str(DENSE_V2_CHECKPOINT_PATH) if dense_v2_net is not None else None,
        "conditions": {},
    }

    for cond_name, corruption_config in conditions.items():
        sequences = load_split_sequences(args.shard_root, split_manifest, corruption_config=corruption_config)
        test_seqs = sequences["test"]
        cond_report = {
            "n_test_sequences": len(test_seqs),
            "corruption_config": json.loads(corruption_config.canonical_json()),
            "knet": evaluate_estimator(test_seqs, lambda s: run_knet(net, s, args.device)),
            "linear_kf_transferred_from_morai": evaluate_estimator(test_seqs, run_kf),
        }
        if av2_kf_calibration is not None:
            calib = av2_kf_calibration
            cond_report["linear_kf_av2_tuned"] = evaluate_estimator(
                test_seqs, lambda s: run_kf(s, sigma_a=calib.sigma_a, r_std=calib.r_std, p0_scale=calib.p0_scale)
            )
        if dense_v2_net is not None:
            cond_report["dense_v2_cross_domain_diagnostic"] = evaluate_estimator(
                test_seqs, lambda s: run_knet(dense_v2_net, s, args.device)
            )
        report["conditions"][cond_name] = cond_report
        kf_av2_rmse = (
            cond_report.get("linear_kf_av2_tuned", {}).get("buckets", {}).get("overall", {}).get("position_rmse_m")
        )
        print(f"[{cond_name}] n={len(test_seqs)} "
              f"knet_pos_rmse={cond_report['knet']['buckets'].get('overall', {}).get('position_rmse_m')} "
              f"kf_transferred_pos_rmse="
              f"{cond_report['linear_kf_transferred_from_morai']['buckets'].get('overall', {}).get('position_rmse_m')} "
              f"kf_av2_tuned_pos_rmse={kf_av2_rmse}",
              file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
