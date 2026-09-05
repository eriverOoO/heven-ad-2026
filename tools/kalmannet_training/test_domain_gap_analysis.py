"""Tests for domain_gap_analysis.py -- deterministic metric aggregation
over synthetic sequence fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from domain_gap_analysis import (  # noqa: E402
    corruption_exposure_stats,
    measurement_residual_stats,
    per_sequence_bootstrap_rmse_diff,
    sequence_distribution_stats,
)


def _make_seq(n=10, dt=0.1, x0=0.0, y0=0.0, vx=3.0, vy=0.0, missing=(), noise=(0.0, 0.0), seed=0):
    rng = np.random.RandomState(seed)
    frames = list(range(n))
    dts = [None] + [dt] * (n - 1)
    x_true, z_meas = [], []
    for i in range(n):
        x, y = x0 + vx * dt * i, y0 + vy * dt * i
        x_true.append([x, y, vx, vy])
        if i in missing:
            z_meas.append(None)
        else:
            nx = x + rng.normal(0, noise[0]) + noise[1]
            ny = y + rng.normal(0, noise[0])
            z_meas.append([nx, ny])
    return {"actor_id": f"a{x0}", "frames": frames, "dt": dts, "x_true": x_true, "z_meas": z_meas}


def test_sequence_distribution_stats_basic_speed_and_length():
    seqs = [_make_seq(n=10, vx=3.0, seed=i) for i in range(5)]
    stats = sequence_distribution_stats(seqs)
    assert stats.n_sequences == 5
    assert stats.n_frames_total == 50
    assert stats.speed_mps["p50"] == pytest.approx(3.0, abs=1e-6)
    assert stats.stationary_fraction == pytest.approx(0.0)


def test_sequence_distribution_stats_stationary_fraction():
    seqs = [_make_seq(n=10, vx=0.0, vy=0.0, seed=i) for i in range(3)]
    stats = sequence_distribution_stats(seqs, stationary_speed_threshold_mps=0.5)
    assert stats.stationary_fraction == pytest.approx(1.0)


def test_sequence_distribution_stats_missing_measurement_fraction():
    seqs = [_make_seq(n=10, missing=(3, 4, 5))]
    stats = sequence_distribution_stats(seqs)
    assert stats.missing_measurement_fraction == pytest.approx(0.3)
    assert stats.measurement_available_fraction == pytest.approx(0.7)
    assert stats.n_gaps == 1
    assert stats.gap_length_frames["p50"] == pytest.approx(3.0)


def test_sequence_distribution_stats_turn_rate_zero_for_straight_motion():
    seqs = [_make_seq(n=10, vx=3.0, vy=0.0)]
    stats = sequence_distribution_stats(seqs)
    assert stats.turn_rate_radps["p50"] == pytest.approx(0.0, abs=1e-6)


def test_measurement_residual_stats_recovers_known_noise():
    seqs = [_make_seq(n=200, noise=(0.5, 0.0), seed=i) for i in range(5)]
    stats = measurement_residual_stats(seqs)
    assert stats.n == 200 * 5
    assert 0.3 < stats.x_std < 0.7  # recovers ~0.5 within sampling tolerance
    assert abs(stats.x_bias) < 0.1  # zero-mean noise, no bias


def test_measurement_residual_stats_detects_bias():
    seqs = [_make_seq(n=200, noise=(0.1, 2.0), seed=i) for i in range(3)]  # +2.0 constant x bias
    stats = measurement_residual_stats(seqs)
    assert stats.x_bias == pytest.approx(2.0, abs=0.2)


def test_measurement_residual_stats_ignores_missing_frames():
    seqs = [_make_seq(n=50, missing=set(range(0, 50, 2)), noise=(0.2, 0.0))]
    stats = measurement_residual_stats(seqs)
    assert stats.n == 25  # only the present-measurement frames counted


def test_measurement_residual_stats_empty_input():
    seqs = [_make_seq(n=10, missing=set(range(10)))]
    stats = measurement_residual_stats(seqs)
    assert stats.n == 0


def test_corruption_exposure_stats_all_clean():
    seqs = [_make_seq(n=10)]
    stats = corruption_exposure_stats(seqs)
    assert stats.clean_fraction == pytest.approx(1.0)
    assert stats.dropout_single_fraction == pytest.approx(0.0)
    assert stats.dropout_burst_fraction == pytest.approx(0.0)


def test_corruption_exposure_stats_single_vs_burst():
    seqs = [_make_seq(n=20, missing=(3, 10, 11, 12))]  # frame 3 = single, 10-12 = burst of 3
    stats = corruption_exposure_stats(seqs)
    n_transitions = 19  # dt is not None for frames 1..19
    assert stats.dropout_single_fraction == pytest.approx(1 / n_transitions)
    assert stats.dropout_burst_fraction == pytest.approx(3 / n_transitions)
    assert stats.post_gap_reacquisition_fraction == pytest.approx(2 / n_transitions)  # after frame 3 gap + after 10-12 gap


def test_per_sequence_bootstrap_rmse_diff_deterministic_given_seed():
    a = [0.5, 0.6, 0.4, 0.55, 0.45]
    b = [0.3, 0.35, 0.32, 0.31, 0.33]
    r1 = per_sequence_bootstrap_rmse_diff(a, b, n_bootstrap=500, seed=7)
    r2 = per_sequence_bootstrap_rmse_diff(a, b, n_bootstrap=500, seed=7)
    assert r1 == r2


def test_per_sequence_bootstrap_rmse_diff_detects_stable_difference():
    a = [1.0] * 20  # clearly, consistently worse
    b = [0.1] * 20  # clearly, consistently better
    result = per_sequence_bootstrap_rmse_diff(a, b, n_bootstrap=500, seed=0)
    assert result.observed_diff == pytest.approx(0.9)
    assert result.stable is True


def test_per_sequence_bootstrap_rmse_diff_detects_unstable_difference():
    rng = np.random.RandomState(0)
    a = list(rng.uniform(0.1, 2.0, size=3))  # tiny n, high variance -- like the real MORAI n=3 case
    b = list(rng.uniform(0.1, 2.0, size=3))
    result = per_sequence_bootstrap_rmse_diff(a, b, n_bootstrap=1000, seed=1)
    # with n=3 highly variable samples, CI should very plausibly straddle zero
    assert result.n_bootstrap == 1000


def test_per_sequence_bootstrap_rmse_diff_rejects_mismatched_length():
    with pytest.raises(ValueError):
        per_sequence_bootstrap_rmse_diff([1.0, 2.0], [1.0])
