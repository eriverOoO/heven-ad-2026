"""Tests for optimizer_step_accounting.py -- pure counting logic, no
model/training-loop dependency."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from optimizer_step_accounting import (  # noqa: E402
    compare_to_historical,
    epochs_needed_to_match_historical_steps,
    steps_per_epoch,
    total_optimizer_steps,
)


def test_steps_per_epoch_batch_size_one_equals_n_sequences():
    assert steps_per_epoch(5769, 1) == 5769


def test_steps_per_epoch_ceils_partial_last_batch():
    assert steps_per_epoch(10, 3) == 4  # 3,3,3,1 -- last partial batch still counts
    assert steps_per_epoch(9, 3) == 3   # exact division


def test_steps_per_epoch_rejects_nonpositive():
    with pytest.raises(ValueError):
        steps_per_epoch(0, 1)
    with pytest.raises(ValueError):
        steps_per_epoch(10, 0)


def test_total_optimizer_steps_matches_real_stage0_counts():
    """Locks the exact real Stage-0 TRAIN counts this task's own
    "OPTIMIZER-STEP DISCREPANCY" section reports (n_train=5769, CLEAN)."""
    n_train = 5769
    assert steps_per_epoch(n_train, 1) == 5769
    assert steps_per_epoch(n_train, 16) == 361
    assert steps_per_epoch(n_train, 32) == 181
    assert steps_per_epoch(n_train, 64) == 91
    assert steps_per_epoch(n_train, 128) == 46
    assert steps_per_epoch(n_train, 256) == 23

    assert total_optimizer_steps(n_train, 1, 10) == 57690
    assert total_optimizer_steps(n_train, 128, 10) == 460


def test_compare_to_historical_deficit_ratio_matches_expected_magnitude():
    cmp = compare_to_historical(5769, 128, 10)
    assert cmp.steps_per_epoch == 46
    assert cmp.historical_steps_per_epoch == 5769
    assert cmp.steps_at_n_epochs == 460
    assert cmp.historical_steps_at_n_epochs == 57690
    assert abs(cmp.step_deficit_ratio - 125.41) < 0.1


def test_compare_to_historical_batch_size_one_is_no_deficit():
    cmp = compare_to_historical(5769, 1, 10)
    assert cmp.step_deficit_ratio == pytest.approx(1.0)


def test_epochs_needed_to_match_historical_steps_is_consistent_with_ratio():
    n_train, bs, hist_epochs = 5769, 128, 10
    needed = epochs_needed_to_match_historical_steps(n_train, bs, hist_epochs)
    # If we ran `needed` epochs at bs=128, total steps should equal the
    # historical total (up to rounding, since needed is a float).
    hist_total = steps_per_epoch(n_train, 1) * hist_epochs
    approx_total = steps_per_epoch(n_train, bs) * needed
    assert math.isclose(approx_total, hist_total, rel_tol=1e-6)


def test_larger_batch_size_always_has_fewer_or_equal_steps_per_epoch():
    n_train = 5769
    prev = steps_per_epoch(n_train, 1)
    for bs in [16, 32, 64, 128, 256]:
        cur = steps_per_epoch(n_train, bs)
        assert cur < prev
        prev = cur
