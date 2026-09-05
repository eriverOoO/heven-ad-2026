"""Unit tests for build_scaleup_v2_split_manifests.py -- single-role
split-manifest construction (all-train or all-val), offline only."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_scaleup_v2_split_manifests import build_single_role_manifest  # noqa: E402


def test_all_train_manifest_puts_every_id_in_train_role():
    manifest = build_single_role_manifest(["a", "b", "c"], "train", {})
    assert manifest["splits"]["train"] == ["a", "b", "c"]
    assert manifest["splits"]["val"] == []
    assert manifest["splits"]["test"] == []
    assert manifest["split_counts"] == {"train": 3, "val": 0, "test": 0}


def test_all_val_manifest_puts_every_id_in_val_role():
    manifest = build_single_role_manifest(["x", "y"], "val", {})
    assert manifest["splits"]["val"] == ["x", "y"]
    assert manifest["splits"]["train"] == []
    assert manifest["splits"]["test"] == []


def test_rejects_unsupported_role():
    with pytest.raises(ValueError):
        build_single_role_manifest(["a"], "not_a_role", {})


def test_rejects_duplicate_scenario_ids():
    with pytest.raises(ValueError):
        build_single_role_manifest(["a", "a", "b"], "train", {})


def test_scenario_ids_are_sorted_regardless_of_input_order():
    manifest = build_single_role_manifest(["z", "a", "m"], "train", {})
    assert manifest["splits"]["train"] == ["a", "m", "z"]


def test_provenance_is_preserved_verbatim():
    prov = {"selection_seed": 42, "split": "train"}
    manifest = build_single_role_manifest(["a"], "train", prov)
    assert manifest["scenario_manifest_provenance"] == prov
