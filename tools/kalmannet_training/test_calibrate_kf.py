"""Unit tests for the fair AV2-tuned LinearCVKF calibration
(``calibrate_kf.py``): TRAIN-only sigma_z measurement, VALIDATION-only
grid selection, and that TEST is never touched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibrate_kf import (  # noqa: E402
    P0_SCALE,
    R_SCALE_GRID,
    SIGMA_A_GRID,
    calibrate_kf_on_av2,
    measure_train_residual_sigma_z,
)


def _seq(n=20, dt=0.1, x0=0.0, y0=0.0, vx=3.0, vy=1.0, noise_std=0.0, missing_indices=(), seed=0):
    import random

    rng = random.Random(seed)
    frames = list(range(n))
    dts = [None] + [dt] * (n - 1)
    x_true, z_meas = [], []
    for i in range(n):
        x, y = x0 + vx * dt * i, y0 + vy * dt * i
        x_true.append([x, y, vx, vy])
        if i in missing_indices:
            z_meas.append(None)
        else:
            nx = x + rng.gauss(0, noise_std)
            ny = y + rng.gauss(0, noise_std)
            z_meas.append([nx, ny])
    return {"actor_id": f"s{x0}", "frames": frames, "dt": dts, "x_true": x_true, "z_meas": z_meas}


def test_measure_train_residual_sigma_z_recovers_known_noise_level():
    seqs = [_seq(n=200, noise_std=0.3, x0=float(i), seed=i) for i in range(10)]
    sigma_z = measure_train_residual_sigma_z(seqs)
    assert 0.2 < sigma_z < 0.4  # recovers ~0.3 within reasonable sampling tolerance


def test_measure_train_residual_sigma_z_ignores_missing_frames():
    seqs = [_seq(n=50, noise_std=0.2, missing_indices=set(range(0, 50, 2)), x0=float(i), seed=i) for i in range(6)]
    sigma_z = measure_train_residual_sigma_z(seqs)
    assert sigma_z > 0.0  # only computed from the real (present) measurements, not fabricated for missing ones


def test_measure_train_residual_sigma_z_raises_on_no_measurements():
    seqs = [_seq(n=10, missing_indices=set(range(10)))]
    with pytest.raises(ValueError):
        measure_train_residual_sigma_z(seqs)


def test_calibrate_kf_on_av2_never_touches_a_third_sequence_set():
    """The function signature itself only accepts train_seqs/val_seqs --
    structurally cannot read a test set it was never given."""
    import inspect

    sig = inspect.signature(calibrate_kf_on_av2)
    assert set(sig.parameters) >= {"train_seqs", "val_seqs", "condition_name"}
    assert "test_seqs" not in sig.parameters
    assert "test" not in sig.parameters


def test_calibrate_kf_on_av2_selects_by_validation_rmse_only():
    train_seqs = [_seq(n=100, noise_std=0.25, x0=float(i), seed=100 + i) for i in range(8)]
    val_seqs = [_seq(n=100, noise_std=0.25, x0=float(i) + 50, seed=200 + i) for i in range(4)]

    result = calibrate_kf_on_av2(train_seqs, val_seqs, condition_name="test_condition")
    assert result.sigma_a in SIGMA_A_GRID
    assert result.p0_scale == P0_SCALE
    assert result.n_candidates == len(SIGMA_A_GRID) * len(R_SCALE_GRID)
    assert result.val_position_rmse > 0.0
    assert result.calibrated_on_condition == "test_condition"
    # r_std must be a positive multiple of the measured train sigma_z
    ratio = result.r_std / result.train_residual_sigma_z
    assert any(abs(ratio - r) < 1e-6 for r in R_SCALE_GRID)


def test_calibrate_kf_on_av2_is_deterministic():
    train_seqs = [_seq(n=60, noise_std=0.2, x0=float(i), seed=10 + i) for i in range(6)]
    val_seqs = [_seq(n=60, noise_std=0.2, x0=float(i) + 20, seed=30 + i) for i in range(3)]

    r1 = calibrate_kf_on_av2(train_seqs, val_seqs, condition_name="c")
    r2 = calibrate_kf_on_av2(train_seqs, val_seqs, condition_name="c")
    assert r1 == r2
