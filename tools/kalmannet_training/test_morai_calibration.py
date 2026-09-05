"""Tests for morai_calibration.py -- the MORAI TRAIN/VAL-only corruption
calibration used by train_kalmannet.py's 'morai_calibrated_robust'
condition. Uses only synthetic fixture sequences (never the real
machine-local sequences.pkl), matching the established
test_morai_frozen_stream.py convention."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from morai_calibration import (  # noqa: E402
    MORAI_CALIBRATED_BURST_LENGTH_SAMPLES,
    MORAI_CALIBRATED_COVARIANCE_XY,
    MORAI_FROZEN_TEST_ACTOR_IDS,
    LeakedTestDataError,
    build_morai_calibrated_corruption_config,
    calibrate_gap_stats,
    calibrate_residual_stats,
)


def _train_val_sequences():
    return [
        {"actor_id": 1, "role": "train", "frames": [0, 1, 2], "dt": [None, 0.1, 0.1],
         "x_true": [[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 1.0, 1.0], [0.2, 0.2, 1.0, 1.0]],
         "z_meas": [[-0.5, 0.5], None, [-0.3, 0.7]]},
        {"actor_id": 21, "role": "val", "frames": [0, 1], "dt": [None, 0.1],
         "x_true": [[5.0, 5.0, 0.5, 0.5], [5.05, 5.05, 0.5, 0.5]],
         "z_meas": [[4.5, 5.5], [4.55, 5.55]]},
    ]


def _with_test_row(sequences):
    return sequences + [
        {"actor_id": 16, "role": "test", "frames": [0], "dt": [None],
         "x_true": [[1.0, 1.0, 0.0, 0.0]], "z_meas": [[1.0, 1.0]]},
    ]


def test_calibrate_residual_stats_computes_bias_and_covariance():
    stats = calibrate_residual_stats(_train_val_sequences())
    assert stats["n"] == 4  # 4 present measurements across both sequences
    # every fixture measurement is offset by (-0.5, +0.5) from GT -- bias should be negative-x, positive-y
    assert stats["bias_xy"][0] < 0.0
    assert stats["bias_xy"][1] > 0.0


def test_calibrate_gap_stats_finds_the_one_missing_frame():
    stats = calibrate_gap_stats(_train_val_sequences())
    assert stats["n_frames_total"] == 3  # frame 0 excluded (dt is None) from both sequences: 2+1
    assert stats["n_gaps"] == 1
    assert stats["n_single_frame_gaps"] == 1
    assert stats["n_burst_gaps"] == 0


def test_calibrate_residual_stats_rejects_test_role_rows():
    with pytest.raises(LeakedTestDataError):
        calibrate_residual_stats(_with_test_row(_train_val_sequences()))


def test_calibrate_gap_stats_rejects_test_role_rows():
    with pytest.raises(LeakedTestDataError):
        calibrate_gap_stats(_with_test_row(_train_val_sequences()))


def test_leaked_test_data_error_names_the_offending_actor_ids():
    with pytest.raises(LeakedTestDataError, match=r"16"):
        calibrate_residual_stats(_with_test_row(_train_val_sequences()))


def test_frozen_test_actor_ids_match_the_documented_split():
    assert MORAI_FROZEN_TEST_ACTOR_IDS == (16, 20, 30)


def test_build_morai_calibrated_corruption_config_is_valid_and_deterministic():
    config = build_morai_calibrated_corruption_config()
    config.validate()  # PSD covariance, finite bias, valid dropout probs
    config2 = build_morai_calibrated_corruption_config()
    assert config.canonical_json() == config2.canonical_json()


def test_build_morai_calibrated_corruption_config_seed_override():
    config_a = build_morai_calibrated_corruption_config(seed=1)
    config_b = build_morai_calibrated_corruption_config(seed=2)
    assert config_a.seed != config_b.seed
    assert config_a.bias_xy == config_b.bias_xy  # only the seed differs


def test_covariance_constant_is_symmetric_and_psd():
    import numpy as np

    cov = np.array(MORAI_CALIBRATED_COVARIANCE_XY)
    assert np.allclose(cov, cov.T)
    assert np.all(np.linalg.eigvalsh(cov) >= 0.0)


def test_burst_length_pool_is_all_positive_integers_matching_real_gap_data():
    assert len(MORAI_CALIBRATED_BURST_LENGTH_SAMPLES) == 64
    assert all(isinstance(v, int) and v >= 2 for v in MORAI_CALIBRATED_BURST_LENGTH_SAMPLES)
    assert max(MORAI_CALIBRATED_BURST_LENGTH_SAMPLES) == 44
