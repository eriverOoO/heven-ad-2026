"""Unit tests for the Scale-Up v2 fetch tool's exclusion/selection logic
-- all offline, no network access. Focuses on the ONE new behavior vs.
Stage-1: excluding the UNION of multiple prior scenario manifests (not
just one) before sampling."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_av2_stage0 import Av2FetchError  # noqa: E402
from fetch_av2_scaleup_v2 import load_excluded_scenario_ids, select_scaleup  # noqa: E402


def _write_manifest(tmp_path: Path, name: str, scenario_ids: list[str]) -> Path:
    manifest = {
        "manifest_schema_version": "av2_stage0_scenario_manifest_v1",
        "scenarios": [{"scenario_id": sid, "split": "train"} for sid in scenario_ids],
    }
    path = tmp_path / name
    path.write_text(json.dumps(manifest))
    return path


def test_load_excluded_scenario_ids_unions_multiple_manifests(tmp_path):
    p0 = _write_manifest(tmp_path, "stage0.json", ["a", "b", "c"])
    p1 = _write_manifest(tmp_path, "stage1.json", ["c", "d", "e"])
    excluded = load_excluded_scenario_ids([p0, p1])
    assert excluded == {"a", "b", "c", "d", "e"}


def test_load_excluded_scenario_ids_empty_list_gives_empty_set():
    assert load_excluded_scenario_ids([]) == set()


def test_load_excluded_scenario_ids_raises_on_missing_manifest(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(Av2FetchError):
        load_excluded_scenario_ids([missing])


def test_select_scaleup_never_returns_an_excluded_id():
    pool = [f"s{i}" for i in range(500)]
    excluded = {f"s{i}" for i in range(0, 100)}
    selected = select_scaleup(pool, count=200, seed=1, excluded=excluded)
    assert len(selected) == 200
    assert not (set(selected) & excluded)


def test_select_scaleup_is_deterministic_given_seed():
    pool = [f"s{i}" for i in range(500)]
    excluded = {f"s{i}" for i in range(10)}
    a = select_scaleup(pool, count=200, seed=42, excluded=excluded)
    b = select_scaleup(pool, count=200, seed=42, excluded=excluded)
    assert a == b


def test_select_scaleup_different_seeds_give_different_selections():
    pool = [f"s{i}" for i in range(500)]
    a = select_scaleup(pool, count=100, seed=1, excluded=set())
    b = select_scaleup(pool, count=100, seed=2, excluded=set())
    assert a != b


def test_select_scaleup_raises_when_not_enough_scenarios_remain_after_exclusion():
    pool = [f"s{i}" for i in range(50)]
    excluded = {f"s{i}" for i in range(40)}  # only 10 remain
    with pytest.raises(Av2FetchError):
        select_scaleup(pool, count=20, seed=1, excluded=excluded)


def test_select_scaleup_union_exclusion_beats_single_manifest_exclusion(tmp_path):
    # The whole point of this tool vs. fetch_av2_stage1.py: two prior
    # experiments' ids must BOTH be excluded, not just one.
    p0 = _write_manifest(tmp_path, "stage0.json", [f"s{i}" for i in range(0, 10)])
    p1 = _write_manifest(tmp_path, "stage1.json", [f"s{i}" for i in range(10, 20)])
    excluded = load_excluded_scenario_ids([p0, p1])
    pool = [f"s{i}" for i in range(30)]
    selected = select_scaleup(pool, count=10, seed=7, excluded=excluded)
    assert not (set(selected) & excluded)
    assert all(int(sid[1:]) >= 20 for sid in selected)
