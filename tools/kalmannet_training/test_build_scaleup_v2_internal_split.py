from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_scaleup_v2_internal_split import (  # noqa: E402
    InternalSplitError,
    assign_internal_split,
    build_internal_split_manifest,
    check_disjoint_from_official_val,
    main,
)


def _ids(n: int, prefix: str = "s") -> list[str]:
    return [f"{prefix}{i:05d}" for i in range(n)]


def test_assign_internal_split_covers_exactly_and_no_overlap():
    ids = _ids(100)
    assignment = assign_internal_split(ids, n_train=90, n_val=10, seed=1)
    assert set(assignment) == set(ids)
    assert sum(1 for v in assignment.values() if v == "train") == 90
    assert sum(1 for v in assignment.values() if v == "val") == 10
    assert set(assignment.values()) == {"train", "val"}


def test_assign_internal_split_deterministic_regardless_of_input_order():
    ids = _ids(50)
    a = assign_internal_split(ids, n_train=40, n_val=10, seed=7)
    b = assign_internal_split(list(reversed(ids)), n_train=40, n_val=10, seed=7)
    assert a == b


def test_assign_internal_split_rejects_count_mismatch():
    with pytest.raises(InternalSplitError):
        assign_internal_split(_ids(100), n_train=90, n_val=20, seed=1)


def test_assign_internal_split_rejects_duplicates():
    ids = _ids(10) + ["s00000"]
    with pytest.raises(InternalSplitError):
        assign_internal_split(ids, n_train=8, n_val=3, seed=1)


def test_build_manifest_has_empty_test_bucket_and_correct_schema():
    ids = _ids(1000)
    manifest = build_internal_split_manifest(
        ids, scenario_manifest_provenance={"foo": "bar"}, n_train=900, n_val=100, seed=20260930,
    )
    assert manifest["split_counts"] == {"train": 900, "val": 100, "test": 0}
    assert manifest["splits"]["test"] == []
    assert manifest["not_av2_official_test_split"] is True
    assert manifest["config"]["seed"] == 20260930
    assert manifest["schema_version"] == "av2_kalmannet_stage0_split_v1"


def test_check_disjoint_from_official_val_detects_overlap():
    internal_val = ["a", "b", "c"]
    official_val_clean = ["x", "y", "z"]
    official_val_leaked = ["x", "b"]
    assert check_disjoint_from_official_val(internal_val, official_val_clean) == 0
    assert check_disjoint_from_official_val(internal_val, official_val_leaked) == 1


def test_main_refuses_when_official_val_overlaps_internal_split(tmp_path):
    # Construct a pathological case where the "train scenario manifest"
    # accidentally shares an id with the "official val scenario manifest"
    # -- main() must raise rather than silently write a leaky manifest.
    shared = "shared-scenario-id"
    train_manifest = {
        "manifest_schema_version": "test_v1", "selection_seed": 1, "selection_pool_size": 10,
        "split": "train", "scenario_count": 4,
        "scenarios": [{"scenario_id": sid} for sid in ["a", "b", "c", shared]],
    }
    official_val_manifest = {
        "manifest_schema_version": "test_v1", "selection_seed": 2, "selection_pool_size": 10,
        "split": "val", "scenario_count": 1,
        "scenarios": [{"scenario_id": shared}],
    }
    train_path = tmp_path / "train_scenarios.json"
    val_path = tmp_path / "official_val_scenarios.json"
    train_path.write_text(json.dumps(train_manifest), encoding="utf-8")
    val_path.write_text(json.dumps(official_val_manifest), encoding="utf-8")

    output_path = tmp_path / "internal_split.json"
    with pytest.raises(InternalSplitError):
        main([
            "--train-scenario-manifest", str(train_path),
            "--official-val-scenario-manifest", str(val_path),
            "--n-train", "3", "--n-val", "1", "--seed", "1",
            "--output", str(output_path),
        ])
    assert not output_path.exists()


def test_main_writes_manifest_and_records_zero_overlap(tmp_path):
    train_manifest = {
        "manifest_schema_version": "test_v1", "selection_seed": 1, "selection_pool_size": 10,
        "split": "train", "scenario_count": 10,
        "scenarios": [{"scenario_id": sid} for sid in _ids(10, prefix="train-")],
    }
    official_val_manifest = {
        "manifest_schema_version": "test_v1", "selection_seed": 2, "selection_pool_size": 10,
        "split": "val", "scenario_count": 3,
        "scenarios": [{"scenario_id": sid} for sid in _ids(3, prefix="val-")],
    }
    train_path = tmp_path / "train_scenarios.json"
    val_path = tmp_path / "official_val_scenarios.json"
    train_path.write_text(json.dumps(train_manifest), encoding="utf-8")
    val_path.write_text(json.dumps(official_val_manifest), encoding="utf-8")

    output_path = tmp_path / "internal_split.json"
    rc = main([
        "--train-scenario-manifest", str(train_path),
        "--official-val-scenario-manifest", str(val_path),
        "--n-train", "8", "--n-val", "2", "--seed", "20260930",
        "--output", str(output_path),
    ])
    assert rc == 0
    written = json.loads(output_path.read_text("utf-8"))
    assert written["split_counts"] == {"train": 8, "val": 2, "test": 0}
    assert written["verified_overlap_with_official_val"] == {
        "internal_train_vs_official_val": 0, "internal_val_vs_official_val": 0,
    }


def test_repeated_run_is_byte_identical(tmp_path):
    ids = _ids(200)
    m1 = build_internal_split_manifest(ids, scenario_manifest_provenance={}, n_train=180, n_val=20, seed=20260930)
    m2 = build_internal_split_manifest(ids, scenario_manifest_provenance={}, n_train=180, n_val=20, seed=20260930)
    assert m1 == m2
