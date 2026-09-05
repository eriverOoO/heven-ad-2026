"""Unit tests for the Stage-1 fetch tool's exclusion/selection logic --
all offline, no network access. Focuses on the ONE new behavior vs.
Stage-0: excluding a fixed set of already-used scenario ids before
sampling."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_av2_stage0 import Av2FetchError  # noqa: E402
from fetch_av2_stage1 import load_excluded_scenario_ids, select_stage1  # noqa: E402


def _write_stage0_manifest(tmp_path: Path, scenario_ids: list[str]) -> Path:
    manifest = {
        "manifest_schema_version": "av2_stage0_scenario_manifest_v1",
        "scenarios": [{"scenario_id": sid, "split": "train"} for sid in scenario_ids],
    }
    path = tmp_path / "stage0_scenarios.json"
    path.write_text(json.dumps(manifest))
    return path


def test_load_excluded_scenario_ids_reads_stage0_manifest(tmp_path):
    path = _write_stage0_manifest(tmp_path, ["a", "b", "c"])
    excluded = load_excluded_scenario_ids(path)
    assert excluded == {"a", "b", "c"}


def test_select_stage1_never_returns_an_excluded_id():
    pool = [f"s{i}" for i in range(100)]
    excluded = {f"s{i}" for i in range(0, 20)}  # first 20 excluded
    selected = select_stage1(pool, count=50, seed=1, excluded=excluded)
    assert len(selected) == 50
    assert not (set(selected) & excluded)


def test_select_stage1_is_deterministic_given_seed():
    pool = [f"s{i}" for i in range(200)]
    excluded = {f"s{i}" for i in range(10)}
    a = select_stage1(pool, count=50, seed=42, excluded=excluded)
    b = select_stage1(pool, count=50, seed=42, excluded=excluded)
    assert a == b


def test_select_stage1_different_seeds_give_different_selections():
    pool = [f"s{i}" for i in range(500)]
    excluded = set()
    a = select_stage1(pool, count=100, seed=1, excluded=excluded)
    b = select_stage1(pool, count=100, seed=2, excluded=excluded)
    assert a != b


def test_select_stage1_raises_when_not_enough_scenarios_remain_after_exclusion():
    pool = [f"s{i}" for i in range(30)]
    excluded = {f"s{i}" for i in range(20)}  # only 10 remain
    with pytest.raises(Av2FetchError):
        select_stage1(pool, count=20, seed=1, excluded=excluded)


def test_select_stage1_empty_exclusion_matches_plain_selection_behavior():
    pool = [f"s{i}" for i in range(50)]
    selected = select_stage1(pool, count=10, seed=7, excluded=set())
    assert len(selected) == 10
    assert set(selected).issubset(set(pool))
