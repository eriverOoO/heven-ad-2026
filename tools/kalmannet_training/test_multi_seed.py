"""Tests for multi_seed.py: deterministic seed generation + summary
statistics."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from multi_seed import deterministic_seed_list, summarize_multi_seed_results  # noqa: E402


def test_deterministic_seed_list_is_deterministic_and_correct_length():
    a = deterministic_seed_list(3, base_seed=0)
    b = deterministic_seed_list(3, base_seed=0)
    assert a == b == [0, 1, 2]


def test_deterministic_seed_list_respects_base_seed():
    assert deterministic_seed_list(3, base_seed=10) == [10, 11, 12]


def test_deterministic_seed_list_rejects_nonpositive():
    with pytest.raises(ValueError):
        deterministic_seed_list(0)


def test_summarize_multi_seed_results_computes_expected_stats():
    results = [
        {"seed": 0, "val_position_rmse": 0.040, "val_velocity_rmse": 0.80, "best_epoch": 9, "wall_s": 100.0, "catastrophic": False},
        {"seed": 1, "val_position_rmse": 0.042, "val_velocity_rmse": 0.82, "best_epoch": 11, "wall_s": 105.0, "catastrophic": False},
        {"seed": 2, "val_position_rmse": 0.038, "val_velocity_rmse": 0.78, "best_epoch": 10, "wall_s": 98.0, "catastrophic": False},
    ]
    s = summarize_multi_seed_results(results)
    assert s.n_seeds == 3
    assert s.seeds == [0, 1, 2]
    assert abs(s.val_position_rmse_mean - 0.04) < 1e-9
    assert s.val_position_rmse_min == pytest.approx(0.038)
    assert s.val_position_rmse_max == pytest.approx(0.042)
    assert s.best_epoch_values == [9, 11, 10]
    assert s.wall_s_total == pytest.approx(303.0)
    assert s.any_catastrophic is False


def test_summarize_multi_seed_results_flags_any_catastrophic():
    results = [
        {"seed": 0, "val_position_rmse": 0.04, "val_velocity_rmse": 0.8, "best_epoch": 5, "wall_s": 1.0, "catastrophic": False},
        {"seed": 1, "val_position_rmse": 200.0, "val_velocity_rmse": 500.0, "best_epoch": 0, "wall_s": 1.0, "catastrophic": True},
    ]
    s = summarize_multi_seed_results(results)
    assert s.any_catastrophic is True


def test_summarize_multi_seed_results_rejects_empty():
    with pytest.raises(ValueError):
        summarize_multi_seed_results([])


# ---------------------------------------------------------------------------
# Numerical-health aggregation (AV2 10k multi-seed task, sections 13/14) --
# norm-overflow/element-nonfinite counts and catastrophic/unstable seed
# identification, sourced with safe defaults so pre-existing minimal
# result dicts (without these keys) are unaffected.
# ---------------------------------------------------------------------------


def test_numerical_health_aggregation_identifies_unstable_and_catastrophic_seeds():
    results = [
        {"seed": 0, "val_position_rmse": 0.30, "val_velocity_rmse": 1.4, "best_epoch": 6, "wall_s": 23237.0,
         "catastrophic": False, "norm_overflow_count": 0, "element_nonfinite_count": 0, "training_unstable": False},
        {"seed": 1, "val_position_rmse": 0.31, "val_velocity_rmse": 1.5, "best_epoch": 8, "wall_s": 24000.0,
         "catastrophic": False, "norm_overflow_count": 15, "element_nonfinite_count": 0, "training_unstable": False},
        {"seed": 2, "val_position_rmse": 100.0, "val_velocity_rmse": 200.0, "best_epoch": 0, "wall_s": 5000.0,
         "catastrophic": True, "norm_overflow_count": 3, "element_nonfinite_count": 1, "training_unstable": True},
    ]
    s = summarize_multi_seed_results(results)
    assert s.norm_overflow_counts == [0, 15, 3]
    assert s.element_nonfinite_counts == [0, 0, 1]
    assert s.catastrophic_seeds == [2]
    assert s.unstable_seeds == [2]
    assert s.any_catastrophic is True


def test_numerical_health_aggregation_defaults_when_keys_absent():
    """Backward compatibility: a result dict without the new health-
    counter keys must not crash, and must default to 0/0/no-instability
    (never fabricate a non-zero count)."""
    results = [
        {"seed": 0, "val_position_rmse": 0.04, "val_velocity_rmse": 0.8, "best_epoch": 9, "wall_s": 100.0,
         "catastrophic": False},
    ]
    s = summarize_multi_seed_results(results)
    assert s.norm_overflow_counts == [0]
    assert s.element_nonfinite_counts == [0]
    assert s.catastrophic_seeds == []
    assert s.unstable_seeds == []


def test_no_seed_silently_dropped_from_catastrophic_or_unstable_lists():
    """Section 16: a catastrophic/unstable seed must never be silently
    discarded from the aggregation -- it stays in ``seeds`` (for the
    RMSE arrays too, whatever value it reported) AND in the dedicated
    catastrophic/unstable lists."""
    results = [
        {"seed": 0, "val_position_rmse": 0.30, "val_velocity_rmse": 1.4, "best_epoch": 6, "wall_s": 100.0,
         "catastrophic": False},
        {"seed": 1, "val_position_rmse": float("nan"), "val_velocity_rmse": float("nan"), "best_epoch": 0,
         "wall_s": 50.0, "catastrophic": True, "training_unstable": True},
    ]
    s = summarize_multi_seed_results(results)
    assert s.n_seeds == 2
    assert 1 in s.seeds
    assert s.catastrophic_seeds == [1]
    assert s.unstable_seeds == [1]
