"""Tests for freeze_manifest.py: serialization round trip and the
freeze-before-test guard."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from freeze_manifest import (  # noqa: E402
    FreezeManifestMissingError,
    build_freeze_manifest,
    load_freeze_manifest,
    require_freeze_manifest_exists,
    save_freeze_manifest,
)


def _sample_manifest():
    return build_freeze_manifest(
        batch_size=32, learning_rate=0.001, max_epochs=60, patience=15, gradient_clip=10.0,
        loss_on_predict_only=True,
        bucket_policy={"use_length_bucketing": True, "n_buckets": 8, "sampler_version": "length_bucketed_v1"},
        shuffle_seed_policy={"order_seed": 0, "init_seed": 0},
        corruption_policy={"mode": "none"},
        model_architecture={"hidden_size": 32, "state_dim": 4, "measurement_dim": 2},
        selection_evidence={"selected_on": "validation_only", "val_position_rmse": 0.04},
        three_seed_validation_summary={"seeds": [0, 1, 2], "val_position_rmse_mean": 0.041},
        git_sha="deadbeef",
        notes="test manifest",
    )


def test_build_freeze_manifest_records_all_required_fields():
    m = _sample_manifest()
    assert m.batch_size == 32
    assert m.learning_rate == 0.001
    assert m.bucket_policy["use_length_bucketing"] is True
    assert m.corruption_policy["mode"] == "none"
    assert m.selection_evidence["selected_on"] == "validation_only"
    assert m.three_seed_validation_summary["seeds"] == [0, 1, 2]


def test_freeze_manifest_round_trips_through_disk(tmp_path):
    m = _sample_manifest()
    path = tmp_path / "freeze.json"
    save_freeze_manifest(m, path)
    loaded = load_freeze_manifest(path)
    assert loaded == m


def test_require_freeze_manifest_exists_raises_when_missing(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FreezeManifestMissingError):
        require_freeze_manifest_exists(missing)


def test_require_freeze_manifest_exists_returns_manifest_when_present(tmp_path):
    m = _sample_manifest()
    path = tmp_path / "freeze.json"
    save_freeze_manifest(m, path)
    loaded = require_freeze_manifest_exists(path)
    assert loaded.batch_size == 32
