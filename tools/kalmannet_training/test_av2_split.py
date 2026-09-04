"""Unit tests for the deterministic AV2 Stage-0 scenario-level split."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from av2_split import (  # noqa: E402
    SplitConfig,
    SplitError,
    assign_split,
    build_split_manifest,
    check_no_leakage,
    load_split_manifest,
    write_split_manifest,
)


def _scenario_ids(n=120):
    return [f"scenario-{i:04d}" for i in range(n)]


def test_split_counts_match_120_scenario_stage0_result():
    ids = _scenario_ids(120)
    assignment = assign_split(ids, SplitConfig())
    counts = {"train": 0, "val": 0, "test": 0}
    for split in assignment.values():
        counts[split] += 1
    assert counts == {"train": 96, "val": 12, "test": 12}  # matches the real Stage-0 run


def test_split_is_deterministic_given_same_seed():
    ids = _scenario_ids(50)
    a = assign_split(ids, SplitConfig(seed=42))
    b = assign_split(ids, SplitConfig(seed=42))
    assert a == b


def test_split_independent_of_input_order():
    ids = _scenario_ids(50)
    a = assign_split(ids, SplitConfig(seed=7))
    b = assign_split(list(reversed(ids)), SplitConfig(seed=7))
    assert a == b


def test_different_seed_gives_different_split():
    ids = _scenario_ids(50)
    a = assign_split(ids, SplitConfig(seed=1))
    b = assign_split(ids, SplitConfig(seed=2))
    assert a != b


def test_no_split_leakage_every_scenario_exactly_one_bucket():
    ids = _scenario_ids(120)
    assignment = assign_split(ids, SplitConfig())
    assert check_no_leakage(assignment) == []
    by_split: dict[str, set] = {"train": set(), "val": set(), "test": set()}
    for sid, split in assignment.items():
        by_split[split].add(sid)
    assert by_split["train"] & by_split["val"] == set()
    assert by_split["train"] & by_split["test"] == set()
    assert by_split["val"] & by_split["test"] == set()
    assert by_split["train"] | by_split["val"] | by_split["test"] == set(ids)


def test_build_split_manifest_covers_exactly_the_input_scenarios():
    ids = _scenario_ids(30)
    manifest = build_split_manifest(ids, SplitConfig(seed=5), scenario_manifest_provenance={"dataset_source": "test"})
    all_ids = manifest["splits"]["train"] + manifest["splits"]["val"] + manifest["splits"]["test"]
    assert sorted(all_ids) == sorted(ids)
    assert len(all_ids) == len(set(all_ids))
    assert manifest["not_av2_official_test_split"] is True


def test_split_rejects_duplicate_scenario_ids():
    with pytest.raises(SplitError):
        assign_split(["a", "b", "a"], SplitConfig())


def test_split_rejects_too_few_scenarios():
    with pytest.raises(SplitError):
        assign_split(["a", "b"], SplitConfig())


def test_split_config_validation():
    with pytest.raises(SplitError):
        SplitConfig(train_frac=0.5, val_frac=0.3, test_frac=0.3).validate()  # sums to 1.1
    with pytest.raises(SplitError):
        SplitConfig(train_frac=1.5, val_frac=-0.3, test_frac=-0.2).validate()


def test_write_and_load_split_manifest_round_trip(tmp_path):
    ids = _scenario_ids(20)
    manifest = build_split_manifest(ids, SplitConfig(seed=9), scenario_manifest_provenance={})
    path = tmp_path / "split.json"
    write_split_manifest(manifest, path)
    reloaded = load_split_manifest(path)
    assert reloaded == manifest
