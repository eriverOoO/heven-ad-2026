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
