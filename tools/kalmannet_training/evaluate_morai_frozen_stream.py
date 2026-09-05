#!/usr/bin/env python3
"""Frozen MORAI zero-shot evaluation: A=Tuned Linear KF, B=Dense-v2 (MORAI-
trained KalmanNet), C=Stage-0 AV2 (generic-robust, calibrated), D=Stage-1
AV2 pilot (generic-robust), E=Stage-1 AV2 pilot (MORAI-calibrated
corruption, this task's own new checkpoint) -- all evaluated on the exact
same frozen MORAI TEST stream (actors 16/20/30,
``morai_frozen_stream.load_morai_frozen_test_sequences``, never touched
by any training or calibration step in this repo).

**MUST be run only after a freeze manifest exists** (see
``freeze_manifest.require_freeze_manifest_exists``) -- this script refuses
to run otherwise, so "evaluate MORAI TEST before freezing" is a hard,
catchable error rather than a silent process discipline.

Reuses ``evaluate_kalmannet.run_kf``/``run_knet`` (byte-for-byte, no
reimplementation) and ``estimator_trace.py``'s prior/posterior/gain
instrumentation for the matched-vs-missing regime comparison. Reports
per-sequence RMSE (Section 17), a gain/prior-posterior comparison of D
vs. E only (Section 18, the two Stage-1-pilot AV2 conditions), and a
per-sequence bootstrap of the mean RMSE difference D-vs-E (Section 19,
n=3 -- explicitly NOT claimed statistically robust).

No claim that AV2 pretraining improves MORAI performance; no claim about
which condition is "best" beyond what the numbers below show.
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
from domain_gap_analysis import per_sequence_bootstrap_rmse_diff  # noqa: E402
from estimator_trace import summarize_traces_by_regime, trace_knet_sequence  # noqa: E402
from evaluate_kalmannet import (  # noqa: E402
    TUNED_HEVEN_KF_P0_SCALE,
    TUNED_HEVEN_KF_PROVENANCE,
    TUNED_HEVEN_KF_R_STD,
    TUNED_HEVEN_KF_SIGMA_A,
    run_kf,
    run_knet,
)
from freeze_manifest import FreezeManifestMissingError, require_freeze_manifest_exists  # noqa: E402
from morai_frozen_stream import FROZEN_TEST_ACTOR_IDS, load_morai_frozen_test_sequences  # noqa: E402

_AD_LIDAR_PKG_DIR = Path(__file__).resolve().parents[2] / "ad_lidar_perception"
if str(_AD_LIDAR_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_AD_LIDAR_PKG_DIR))
from ad_lidar_perception.kalmannet_core import KalmanNetGRU  # noqa: E402


def _load_knet(checkpoint_path: Path, device: str) -> KalmanNetGRU:
    state_dict, manifest = checkpoint_utils.load_checkpoint(checkpoint_path)
    hidden_size = manifest.get("model_config", {}).get("hidden_size", 32)
    net = KalmanNetGRU(hidden_size=hidden_size).to(device)
    net.load_state_dict(state_dict)
    net.eval()
    return net


def _pos_rmse(estimates: list[np.ndarray], gt: list[list[float]]) -> float:
    est = np.stack(estimates)
    truth = np.array(gt)
    return float(np.sqrt(np.mean((est[:, 0] - truth[:, 0]) ** 2 + (est[:, 1] - truth[:, 1]) ** 2)))


def _vel_rmse(estimates: list[np.ndarray], gt: list[list[float]]) -> float:
    est = np.stack(estimates)
    truth = np.array(gt)
    return float(np.sqrt(np.mean((est[:, 2] - truth[:, 2]) ** 2 + (est[:, 3] - truth[:, 3]) ** 2)))


def _matched_missing_split(seq: dict, estimates: list[np.ndarray]) -> dict:
    matched_pos, matched_vel, missing_pos, missing_vel = [], [], [], []
    for i, x_est in enumerate(estimates):
        if seq["dt"][i] is None:
            continue
        gt = seq["x_true"][i]
        pos_err = float(np.hypot(x_est[0] - gt[0], x_est[1] - gt[1]))
        vel_err = float(np.hypot(x_est[2] - gt[2], x_est[3] - gt[3]))
        if seq["z_meas"][i] is not None:
            matched_pos.append(pos_err)
            matched_vel.append(vel_err)
        else:
            missing_pos.append(pos_err)
            missing_vel.append(vel_err)
    return {
        "matched_pos_rmse": (float(np.sqrt(np.mean(np.square(matched_pos)))) if matched_pos else None),
        "matched_vel_rmse": (float(np.sqrt(np.mean(np.square(matched_vel)))) if matched_vel else None),
        "missing_pos_rmse": (float(np.sqrt(np.mean(np.square(missing_pos)))) if missing_pos else None),
        "missing_vel_rmse": (float(np.sqrt(np.mean(np.square(missing_vel)))) if missing_vel else None),
        "n_matched": len(matched_pos), "n_missing": len(missing_pos),
    }


def evaluate_condition(sequences: list[dict], estimate_fn) -> dict:
    per_seq_pos_rmse, per_seq_vel_rmse = [], []
    matched_missing_rows = []
    n_divergent = 0
    for seq in sequences:
        est = estimate_fn(seq)
        pos_rmse = _pos_rmse(est, seq["x_true"])
        per_seq_pos_rmse.append(pos_rmse)
        per_seq_vel_rmse.append(_vel_rmse(est, seq["x_true"]))
        matched_missing_rows.append(_matched_missing_split(seq, est))
        if not all(np.all(np.isfinite(x)) for x in est) or pos_rmse > 1000.0:
            n_divergent += 1
    return {
        "per_sequence_pos_rmse": per_seq_pos_rmse,
        "per_sequence_vel_rmse": per_seq_vel_rmse,
        "overall_pos_rmse": float(np.mean(per_seq_pos_rmse)),
        "overall_vel_rmse": float(np.mean(per_seq_vel_rmse)),
        "matched_missing_per_sequence": matched_missing_rows,
        "n_divergent_sequences": n_divergent,
    }


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--freeze-manifest", type=Path, required=True,
                         help="path to the freeze manifest written BEFORE this evaluation (required guard)")
    parser.add_argument("--sequences-pkl", type=Path,
                         default=Path.home() / "heven_presentation_assets/state_estimator_gt_comparison/sequences.pkl")
    parser.add_argument("--stage0-av2-checkpoint", type=Path,
                         default=Path.home() / "datasets/av2/checkpoints/av2_stage0_generic_robust_calibrated_bs64_lr004_seed0.pt")
    parser.add_argument("--stage1-generic-checkpoint", type=Path,
                         default=Path.home() / "datasets/av2/checkpoints/av2_stage1_pilot_generic_robust_bs64_lr004_seed0.pt")
    parser.add_argument("--stage1-morai-calibrated-checkpoint", type=Path,
                         default=Path.home() / "datasets/av2/checkpoints/av2_stage1_pilot_morai_calibrated_robust_bs64_lr004_seed0.pt")
    parser.add_argument("--dense-v2-checkpoint", type=Path,
                         default=Path.home() / "heven_presentation_assets/kalmannet_training_stability/checkpoint/dense_kalmannet_v2.pt")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        require_freeze_manifest_exists(args.freeze_manifest)
    except FreezeManifestMissingError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    sequences = load_morai_frozen_test_sequences(args.sequences_pkl)
    print(f"loaded {len(sequences)} frozen MORAI TEST sequences, actors="
          f"{sorted({s['actor_id'] for s in sequences})} (expected {FROZEN_TEST_ACTOR_IDS})", file=sys.stderr)

    net_c = _load_knet(args.stage0_av2_checkpoint, args.device)
    net_d = _load_knet(args.stage1_generic_checkpoint, args.device)
    net_e = _load_knet(args.stage1_morai_calibrated_checkpoint, args.device)
    net_b = None
    if args.dense_v2_checkpoint.is_file():
        raw = torch.load(args.dense_v2_checkpoint, map_location=args.device)
        sd = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
        net_b = KalmanNetGRU(hidden_size=32).to(args.device)
        net_b.load_state_dict(sd)
        net_b.eval()

    report: dict = {
        "schema_version": "kalmannet_morai_calibrated_corruption_v1_eval",
        "frozen_test_actor_ids": list(FROZEN_TEST_ACTOR_IDS),
        "conditions": {},
    }

    report["conditions"]["A_tuned_kf"] = evaluate_condition(
        sequences, lambda s: run_kf(s, sigma_a=TUNED_HEVEN_KF_SIGMA_A, r_std=TUNED_HEVEN_KF_R_STD,
                                     p0_scale=TUNED_HEVEN_KF_P0_SCALE))
    report["conditions"]["A_tuned_kf"]["provenance"] = TUNED_HEVEN_KF_PROVENANCE
    if net_b is not None:
        report["conditions"]["B_dense_v2"] = evaluate_condition(sequences, lambda s: run_knet(net_b, s, args.device))
    report["conditions"]["C_stage0_av2"] = evaluate_condition(sequences, lambda s: run_knet(net_c, s, args.device))
    report["conditions"]["D_stage1_generic_av2"] = evaluate_condition(sequences, lambda s: run_knet(net_d, s, args.device))
    report["conditions"]["E_stage1_morai_calibrated_av2"] = evaluate_condition(
        sequences, lambda s: run_knet(net_e, s, args.device))

    # Section 18: gain/prior-posterior comparison, D (generic) vs. E (morai-calibrated).
    d_traces, e_traces = [], []
    for seq in sequences:
        d_traces.extend(trace_knet_sequence(net_d, seq, device=args.device))
        e_traces.extend(trace_knet_sequence(net_e, seq, device=args.device))
    report["gain_prior_posterior_D_vs_E"] = {
        "D_stage1_generic_av2": summarize_traces_by_regime(d_traces),
        "E_stage1_morai_calibrated_av2": summarize_traces_by_regime(e_traces),
    }

    # Section 19: per-sequence bootstrap of the mean position-RMSE difference D-vs-E (n=3, explicitly NOT claimed robust).
    bootstrap = per_sequence_bootstrap_rmse_diff(
        report["conditions"]["D_stage1_generic_av2"]["per_sequence_pos_rmse"],
        report["conditions"]["E_stage1_morai_calibrated_av2"]["per_sequence_pos_rmse"],
    )
    report["bootstrap_D_minus_E_pos_rmse"] = {
        "observed_diff": bootstrap.observed_diff, "ci_low": bootstrap.ci_low, "ci_high": bootstrap.ci_high,
        "fraction_bootstrap_same_sign": bootstrap.fraction_bootstrap_same_sign,
        "n_bootstrap": bootstrap.n_bootstrap, "stable": bootstrap.stable,
        "caveat": "n=3 sequences -- CI/stability are NOT a claim of statistical robustness, per this task's own explicit scope",
    }

    for name, cond in report["conditions"].items():
        print(f"[{name}] overall_pos_rmse={cond['overall_pos_rmse']:.4f} "
              f"overall_vel_rmse={cond['overall_vel_rmse']:.4f} n_divergent={cond['n_divergent_sequences']}",
              file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
