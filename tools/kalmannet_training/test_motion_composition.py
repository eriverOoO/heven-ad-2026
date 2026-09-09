"""Tests for motion_composition.py (AV2 Motion-Composition Ablation v1,
docs/perception/kalmannet_av2_motion_composition_v1.md, section 21)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from motion_composition import (  # noqa: E402
    BALANCED_MOTION,
    MOVING,
    MOVING_FOCUSED,
    NATURAL,
    NEAR_STATIONARY,
    MotionSamplerConfig,
    classify_motion,
    compute_sample_weights,
    compute_segment_motion_stats,
    definition_agreement,
    deterministic_weighted_draw_order,
    segment_key,
    speed_bin,
    weight_manifest_fingerprint,
)


def _seq(vx, vy, dt=0.1, x0=0.0, y0=0.0):
    t = len(vx)
    x_true = np.zeros((t, 4))
    x, y = x0, y0
    for i in range(t):
        x_true[i] = [x, y, vx[i], vy[i]]
        x += vx[i] * dt
        y += vy[i] * dt
    return {"x_true": x_true, "dt": [None] + [dt] * (t - 1), "scenario_id": "s0", "track_id": "t0"}


# --------------------------------------------------------------------------
# compute_segment_motion_stats / classify_motion
# --------------------------------------------------------------------------


def test_stationary_sequence_classified_near_stationary():
    seq = _seq(vx=[0.0] * 10, vy=[0.0] * 10)
    stats = compute_segment_motion_stats(seq)
    assert stats["median_speed_mps"] == pytest.approx(0.0)
    assert stats["max_speed_mps"] == pytest.approx(0.0)
    assert stats["displacement_m"] == pytest.approx(0.0)
    assert stats["stationary_frame_fraction"] == pytest.approx(1.0)
    assert classify_motion(stats) == NEAR_STATIONARY


def test_fast_sequence_classified_moving():
    seq = _seq(vx=[10.0] * 10, vy=[0.0] * 10)
    stats = compute_segment_motion_stats(seq)
    assert stats["median_speed_mps"] == pytest.approx(10.0)
    assert stats["displacement_m"] > 0
    assert stats["stationary_frame_fraction"] == pytest.approx(0.0)
    assert classify_motion(stats) == MOVING


def test_classification_is_deterministic_given_same_stats():
    seq = _seq(vx=[3.0] * 5, vy=[4.0] * 5)  # speed = 5.0 exactly
    stats = compute_segment_motion_stats(seq)
    assert classify_motion(stats) == classify_motion(stats) == MOVING


def test_classification_boundary_uses_greater_equal():
    stats = {"median_speed_mps": 0.5, "mean_speed_mps": 0.5, "p90_speed_mps": 0.5}
    assert classify_motion(stats, threshold_mps=0.5) == MOVING


def test_acceleration_proxy_nonzero_for_accelerating_segment():
    seq = _seq(vx=list(np.linspace(0, 10, 10)), vy=[0.0] * 10)
    stats = compute_segment_motion_stats(seq)
    assert stats["acceleration_proxy_mps2"] > 0


def test_compute_segment_motion_stats_rejects_too_short_sequence():
    with pytest.raises(ValueError):
        compute_segment_motion_stats({"x_true": np.zeros((1, 4)), "dt": [None]})


# --------------------------------------------------------------------------
# Definition sensitivity audit (median vs mean vs p90) -- section 2
# --------------------------------------------------------------------------


def test_definition_agreement_reports_all_three_keys_and_bounds_in_0_1():
    stats_list = [
        compute_segment_motion_stats(_seq(vx=[0.0] * 8, vy=[0.0] * 8)),
        compute_segment_motion_stats(_seq(vx=[5.0] * 8, vy=[0.0] * 8)),
        compute_segment_motion_stats(_seq(vx=[0.0] * 4 + [3.0] * 4, vy=[0.0] * 8)),  # mixed: transitions mid-way
    ]
    report = definition_agreement(stats_list)
    assert report["n_segments"] == 3
    assert set(report["moving_fraction_by_key"]) == {"median_speed_mps", "mean_speed_mps", "p90_speed_mps"}
    for frac in report["moving_fraction_by_key"].values():
        assert 0.0 <= frac <= 1.0
    for agree in report["pairwise_agreement_fraction"].values():
        assert 0.0 <= agree <= 1.0


def test_definition_agreement_all_identical_for_uniformly_moving_segments():
    stats_list = [compute_segment_motion_stats(_seq(vx=[7.0] * 6, vy=[0.0] * 6)) for _ in range(4)]
    report = definition_agreement(stats_list)
    assert all(f == 1.0 for f in report["moving_fraction_by_key"].values())
    assert all(a == 1.0 for a in report["pairwise_agreement_fraction"].values())


# --------------------------------------------------------------------------
# speed_bin
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "speed,expected",
    [(0.0, "0-0.5"), (0.49, "0-0.5"), (0.5, "0.5-2"), (1.9, "0.5-2"), (2.0, "2-5"), (4.9, "2-5"),
     (5.0, "5-10"), (9.9, "5-10"), (10.0, "10+"), (100.0, "10+")],
)
def test_speed_bin_boundaries(speed, expected):
    assert speed_bin(speed) == expected


# --------------------------------------------------------------------------
# compute_sample_weights -- exact proportions (section 21)
# --------------------------------------------------------------------------


def test_natural_sampler_gives_uniform_weight_of_one():
    categories = [MOVING, NEAR_STATIONARY, MOVING, NEAR_STATIONARY, MOVING]
    weights = compute_sample_weights(categories, NATURAL)
    assert np.allclose(weights, 1.0)


def test_balanced_motion_achieves_exact_50_50_weight_mass():
    categories = [MOVING] * 20 + [NEAR_STATIONARY] * 80  # natural 20% moving
    weights = compute_sample_weights(categories, BALANCED_MOTION)
    moving_mass = weights[np.array(categories) == MOVING].sum()
    stationary_mass = weights[np.array(categories) == NEAR_STATIONARY].sum()
    total = weights.sum()
    assert moving_mass / total == pytest.approx(0.5, abs=1e-9)
    assert stationary_mass / total == pytest.approx(0.5, abs=1e-9)
    assert total == pytest.approx(len(categories))  # section 11: expected draws per epoch unchanged


def test_moving_focused_achieves_target_fraction():
    categories = [MOVING] * 36 + [NEAR_STATIONARY] * 64  # natural 36% moving
    weights = compute_sample_weights(categories, MOVING_FOCUSED)
    moving_mass = weights[np.array(categories) == MOVING].sum()
    total = weights.sum()
    assert moving_mass / total == pytest.approx(MOVING_FOCUSED.target_moving_fraction, abs=1e-9)
    assert total == pytest.approx(len(categories))


def test_compute_sample_weights_rejects_empty_class():
    categories = [MOVING] * 10  # zero NEAR_STATIONARY
    with pytest.raises(ValueError):
        compute_sample_weights(categories, BALANCED_MOTION)


def test_compute_sample_weights_rejects_empty_input():
    with pytest.raises(ValueError):
        compute_sample_weights([], BALANCED_MOTION)


# --------------------------------------------------------------------------
# deterministic_weighted_draw_order -- determinism + epoch draw-count
# invariance + approximate target proportion achieved by real draws
# --------------------------------------------------------------------------


def test_same_seed_gives_identical_draw_order():
    weights = np.array([1.0, 3.0, 1.0, 5.0])
    a = deterministic_weighted_draw_order(weights, n_draws=200, seed=42)
    b = deterministic_weighted_draw_order(weights, n_draws=200, seed=42)
    assert np.array_equal(a, b)


def test_different_seed_gives_different_draw_order():
    weights = np.array([1.0, 3.0, 1.0, 5.0])
    a = deterministic_weighted_draw_order(weights, n_draws=200, seed=42)
    b = deterministic_weighted_draw_order(weights, n_draws=200, seed=43)
    assert not np.array_equal(a, b)


def test_epoch_draw_count_invariance_regardless_of_weighting():
    for weights in (np.ones(10), np.array([1.0] * 8 + [50.0, 50.0])):
        for n in (1, 10, 500):
            draws = deterministic_weighted_draw_order(weights, n_draws=n, seed=7)
            assert len(draws) == n


def test_deterministic_weighted_draw_order_rejects_nonpositive_n_draws():
    with pytest.raises(ValueError):
        deterministic_weighted_draw_order(np.ones(5), n_draws=0, seed=1)


def test_draw_order_achieves_approximate_target_fraction_at_scale():
    categories = [MOVING] * 20 + [NEAR_STATIONARY] * 80
    weights = compute_sample_weights(categories, BALANCED_MOTION)
    draws = deterministic_weighted_draw_order(weights, n_draws=200_000, seed=123)
    drawn_categories = np.array(categories)[draws]
    empirical_moving_fraction = float(np.mean(drawn_categories == MOVING))
    assert empirical_moving_fraction == pytest.approx(0.5, abs=0.01)


def test_draw_indices_always_within_bounds_no_train_val_leakage():
    """Section 21 'no train/val leakage': draw indices only ever index
    into the TRAIN-sized weight array passed in -- this function has no
    access to a val set at all, so leakage is structurally impossible."""
    weights = np.array([1.0, 2.0, 3.0])
    draws = deterministic_weighted_draw_order(weights, n_draws=1000, seed=5)
    assert draws.min() >= 0
    assert draws.max() < len(weights)


# --------------------------------------------------------------------------
# segment_key / weight_manifest_fingerprint
# --------------------------------------------------------------------------


def test_segment_key_combines_scenario_and_track():
    seq = {"scenario_id": "abc", "track_id": "xyz"}
    assert segment_key(seq) == "abc::xyz"


def test_weight_manifest_fingerprint_deterministic():
    categories = [MOVING, NEAR_STATIONARY, MOVING]
    a = weight_manifest_fingerprint(categories, BALANCED_MOTION)
    b = weight_manifest_fingerprint(categories, BALANCED_MOTION)
    assert a == b


def test_weight_manifest_fingerprint_differs_across_configs():
    categories = [MOVING, NEAR_STATIONARY, MOVING]
    a = weight_manifest_fingerprint(categories, BALANCED_MOTION)
    b = weight_manifest_fingerprint(categories, MOVING_FOCUSED)
    c = weight_manifest_fingerprint(categories, NATURAL)
    assert len({a, b, c}) == 3


def test_weight_manifest_fingerprint_differs_when_composition_differs():
    a = weight_manifest_fingerprint([MOVING, MOVING, NEAR_STATIONARY], BALANCED_MOTION)
    b = weight_manifest_fingerprint([MOVING, NEAR_STATIONARY, NEAR_STATIONARY], BALANCED_MOTION)
    assert a != b


def test_motion_sampler_config_is_frozen_dataclass():
    cfg = MotionSamplerConfig(name="X", target_moving_fraction=0.6)
    with pytest.raises(Exception):
        cfg.name = "Y"  # type: ignore[misc]
