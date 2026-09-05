"""Tests for morai_frozen_stream.py -- the frozen-stream loader/guard,
using a synthetic fixture pickle (never the real local file, so these
tests run without that machine-local asset present)."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from morai_frozen_stream import (  # noqa: E402
    FROZEN_TEST_ACTOR_IDS,
    load_morai_frozen_test_sequences,
)


def _make_fixture_pickle(tmp_path, actor_ids=(16, 20, 30), include_val_train=True):
    seqs = []
    if include_val_train:
        seqs.append({"actor_id": 1, "role": "train", "run_idx": 0, "frames": [0, 1],
                     "dt": [None, 0.1], "x_true": [[10.0, 20.0, 1.0, 2.0], [10.1, 20.2, 1.0, 2.0]],
                     "z_meas": [[10.0, 20.0], [10.1, 20.2]]})
        seqs.append({"actor_id": 21, "role": "val", "run_idx": 0, "frames": [0, 1],
                     "dt": [None, 0.1], "x_true": [[5.0, 5.0, 0.5, 0.5], [5.05, 5.05, 0.5, 0.5]],
                     "z_meas": [[5.0, 5.0], None]})
    for i, aid in enumerate(actor_ids):
        x0, y0 = 100.0 + i, -50.0 - i
        seqs.append({
            "actor_id": aid, "role": "test", "run_idx": 0, "frames": [0, 1, 2],
            "dt": [None, 0.1, 0.1],
            "x_true": [[x0, y0, 3.0, -1.0], [x0 + 0.3, y0 - 0.1, 3.0, -1.0], [x0 + 0.6, y0 - 0.2, 3.0, -1.0]],
            "z_meas": [[x0, y0], None, [x0 + 0.6, y0 - 0.2]],
        })
    path = tmp_path / "fixture_sequences.pkl"
    with open(path, "wb") as f:
        pickle.dump(seqs, f)
    return path


def test_load_filters_to_test_role_only(tmp_path):
    path = _make_fixture_pickle(tmp_path)
    result = load_morai_frozen_test_sequences(path)
    assert len(result) == 3
    assert sorted(s["actor_id"] for s in result) == sorted(FROZEN_TEST_ACTOR_IDS)


def test_load_applies_first_state_relative_shift(tmp_path):
    path = _make_fixture_pickle(tmp_path)
    result = load_morai_frozen_test_sequences(path)
    for seq in result:
        assert seq["x_true"][0][0] == pytest.approx(0.0)
        assert seq["x_true"][0][1] == pytest.approx(0.0)
        # measurement at frame 0 shifted by the same origin
        assert seq["z_meas"][0][0] == pytest.approx(0.0)
        assert seq["z_meas"][0][1] == pytest.approx(0.0)


def test_load_preserves_missing_measurements_as_none(tmp_path):
    path = _make_fixture_pickle(tmp_path)
    result = load_morai_frozen_test_sequences(path)
    for seq in result:
        assert seq["z_meas"][1] is None  # frame 1 is missing in every fixture sequence


def test_load_preserves_velocity_unshifted(tmp_path):
    path = _make_fixture_pickle(tmp_path)
    result = load_morai_frozen_test_sequences(path)
    for seq in result:
        assert seq["x_true"][0][2] == pytest.approx(3.0)
        assert seq["x_true"][0][3] == pytest.approx(-1.0)


def test_load_raises_if_test_actor_set_ever_changes(tmp_path):
    path = _make_fixture_pickle(tmp_path, actor_ids=(16, 20, 99))  # wrong actor set
    with pytest.raises(ValueError):
        load_morai_frozen_test_sequences(path)


def test_result_shape_matches_run_sequence_contract(tmp_path):
    path = _make_fixture_pickle(tmp_path)
    result = load_morai_frozen_test_sequences(path)
    for seq in result:
        assert set(seq.keys()) >= {"actor_id", "frames", "dt", "x_true", "z_meas"}
        assert len(seq["frames"]) == len(seq["dt"]) == len(seq["x_true"]) == len(seq["z_meas"])
