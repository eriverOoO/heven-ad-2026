"""Fair AV2-tuned ``LinearCVKF`` baseline.

Reuses the historical T-9A.1/T-11 calibration methodology exactly
(``~/heven_presentation_assets/state_estimator_gt_comparison/
calibrate_kf.py``, inspected fresh from disk, not recalled from memory):
measure an isotropic ``sigma_z`` directly from real TRAIN measurement
residuals, then a bounded log-grid search over ``sigma_a`` (process
noise) and an ``r_scale`` factor on ``sigma_z``, selected purely by
VALIDATION position RMSE. **The held-out TEST split is never touched by
this module.**

This is a distinct baseline from the transferred-without-retuning
``TUNED_HEVEN_KF_*`` constants in ``evaluate_kalmannet.py`` (which were
tuned on MORAI HEVEN data, a different domain, and never touch AV2 data
at all) -- this module produces the fair, AV2-native comparison point.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_AD_LIDAR_PKG_DIR = Path(__file__).resolve().parents[2] / "ad_lidar_perception"
if str(_AD_LIDAR_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_AD_LIDAR_PKG_DIR))
from ad_lidar_perception.kalmannet_core import LinearCVKF  # noqa: E402

# Identical grid to the historical calibration -- not re-chosen for AV2.
SIGMA_A_GRID = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
R_SCALE_GRID = [0.25, 0.5, 1.0, 2.0, 4.0]
P0_SCALE = 10.0  # unchanged historical default


@dataclass(frozen=True)
class KFCalibrationResult:
    sigma_a: float
    r_std: float
    p0_scale: float
    train_residual_sigma_z: float
    val_position_rmse: float
    val_velocity_rmse: float
    n_candidates: int
    calibrated_on_condition: str


def measure_train_residual_sigma_z(train_seqs: list[dict[str, Any]]) -> float:
    """Isotropic measurement-noise sigma_z from real TRAIN
    measurement-vs-GT residuals -- only over frames with an actual
    measurement (``z_meas[i] is not None``); a dropped frame contributes
    no residual (never fabricated)."""
    residuals = []
    for s in train_seqs:
        for i in range(len(s["z_meas"])):
            z = s["z_meas"][i]
            if z is None:
                continue
            gt = s["x_true"][i]
            residuals.append([z[0] - gt[0], z[1] - gt[1]])
    if not residuals:
        raise ValueError("no measured frames in train_seqs -- cannot estimate sigma_z")
    residuals = np.asarray(residuals, dtype=np.float64)
    return float(np.sqrt(np.mean(residuals**2)))


def _run_kf_on_sequences(seqs: list[dict[str, Any]], sigma_a: float, r_std: float, p0_scale: float):
    pos_errs, vel_errs = [], []
    for s in seqs:
        first_meas = s["z_meas"][0]
        if first_meas is None:
            continue  # defensive -- sequences are already truncated to their first measurement
        kf = LinearCVKF(sigma_a=sigma_a, r_std=r_std, p0_scale=p0_scale)
        kf.init_sequence(np.array([first_meas[0], first_meas[1], 0.0, 0.0]))
        for i in range(len(s["frames"])):
            dt = s["dt"][i]
            if dt is not None:
                kf.predict(dt)
            if s["z_meas"][i] is not None:
                kf.update(np.array(s["z_meas"][i]))
            gt = s["x_true"][i]
            pos_errs.append(float(np.hypot(kf.x[0] - gt[0], kf.x[1] - gt[1])))
            vel_errs.append(float(np.hypot(kf.x[2] - gt[2], kf.x[3] - gt[3])))
    return float(np.sqrt(np.mean(np.square(pos_errs)))), float(np.sqrt(np.mean(np.square(vel_errs))))


def calibrate_kf_on_av2(
    train_seqs: list[dict[str, Any]], val_seqs: list[dict[str, Any]], condition_name: str, p0_scale: float = P0_SCALE
) -> KFCalibrationResult:
    """Bounded log-grid search, selected on VALIDATION position RMSE
    only. ``train_seqs``/``val_seqs`` should be the SAME corruption
    condition (e.g. both GENERIC-ROBUST) so ``sigma_z`` reflects that
    condition's real measurement-noise statistics -- calibrating against
    CLEAN data would be degenerate (residuals are exactly zero by
    construction, nothing to estimate)."""
    sigma_z = measure_train_residual_sigma_z(train_seqs)

    best = None
    n_candidates = 0
    for sigma_a in SIGMA_A_GRID:
        for r_scale in R_SCALE_GRID:
            n_candidates += 1
            r_std = sigma_z * r_scale
            if r_std <= 0.0:
                continue  # a degenerate zero-noise KF is not a valid grid candidate
            val_pos_rmse, val_vel_rmse = _run_kf_on_sequences(val_seqs, sigma_a, r_std, p0_scale)
            if best is None or val_pos_rmse < best[0]:
                best = (val_pos_rmse, val_vel_rmse, sigma_a, r_std)

    if best is None:
        raise ValueError("no valid (sigma_a, r_std) candidate found -- check sigma_z/grid")
    val_pos_rmse, val_vel_rmse, sigma_a, r_std = best
    return KFCalibrationResult(
        sigma_a=sigma_a, r_std=r_std, p0_scale=p0_scale, train_residual_sigma_z=sigma_z,
        val_position_rmse=val_pos_rmse, val_velocity_rmse=val_vel_rmse,
        n_candidates=n_candidates, calibrated_on_condition=condition_name,
    )
