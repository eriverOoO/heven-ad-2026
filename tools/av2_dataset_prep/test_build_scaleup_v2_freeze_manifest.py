"""Unit tests for build_scaleup_v2_freeze_manifest.py -- offline only,
synthetic fixture manifests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_scaleup_v2_freeze_manifest import main  # noqa: E402


def _write(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data))
    return path


def _fixture_scenario_manifest(ids: list[str], seed: int, split: str) -> dict:
    return {
        "scenario_count": len(ids), "selection_seed": seed, "selection_pool_size": 1000,
        "split": split, "scenarios": [{"scenario_id": sid} for sid in ids],
    }


def _fixture_split_manifest(ids: list[str], role: str) -> dict:
    splits = {"train": [], "val": [], "test": []}
    splits[role] = ids
    return {"schema_version": "x", "splits": splits}


def _fixture_export_manifest(shard_count: int) -> dict:
    return {"content_fingerprint": "abc123", "shard_count": shard_count, "counts": {"segments_exported": 42}}


def test_freeze_manifest_writes_ok_when_no_overlap(tmp_path):
    train_scenarios = _write(tmp_path / "train_scen.json", _fixture_scenario_manifest(["a", "b"], 1, "train"))
    train_split = _write(tmp_path / "train_split.json", _fixture_split_manifest(["a", "b"], "train"))
    train_export = _write(tmp_path / "train_export.json", _fixture_export_manifest(1))
    val_scenarios = _write(tmp_path / "val_scen.json", _fixture_scenario_manifest(["x", "y"], 2, "val"))
    val_split = _write(tmp_path / "val_split.json", _fixture_split_manifest(["x", "y"], "val"))
    val_export = _write(tmp_path / "val_export.json", _fixture_export_manifest(1))
    exclusion = _write(tmp_path / "exclusion.json", _fixture_scenario_manifest(["p", "q"], 0, "train"))
    output = tmp_path / "freeze.json"

    rc = main([
        "--train-scenario-manifest", str(train_scenarios), "--train-split-manifest", str(train_split),
        "--train-export-manifest", str(train_export),
        "--val-scenario-manifest", str(val_scenarios), "--val-split-manifest", str(val_split),
        "--val-export-manifest", str(val_export),
        "--exclusion-manifests", str(exclusion),
        "--output", str(output),
    ])
    assert rc == 0
    manifest = json.loads(output.read_text())
    assert manifest["train"]["scenario_count"] == 2
    assert manifest["official_val"]["scenario_count"] == 2
    assert manifest["overlap_train_vs_official_val"] == 0
    assert manifest["overlap_train_vs_exclusion"] == 0


def test_freeze_manifest_refuses_when_train_val_overlap(tmp_path):
    train_scenarios = _write(tmp_path / "train_scen.json", _fixture_scenario_manifest(["a", "b"], 1, "train"))
    train_split = _write(tmp_path / "train_split.json", _fixture_split_manifest(["a", "b"], "train"))
    train_export = _write(tmp_path / "train_export.json", _fixture_export_manifest(1))
    val_scenarios = _write(tmp_path / "val_scen.json", _fixture_scenario_manifest(["a", "y"], 2, "val"))
    val_split = _write(tmp_path / "val_split.json", _fixture_split_manifest(["a", "y"], "val"))
    val_export = _write(tmp_path / "val_export.json", _fixture_export_manifest(1))
    exclusion = _write(tmp_path / "exclusion.json", _fixture_scenario_manifest([], 0, "train"))
    output = tmp_path / "freeze.json"

    rc = main([
        "--train-scenario-manifest", str(train_scenarios), "--train-split-manifest", str(train_split),
        "--train-export-manifest", str(train_export),
        "--val-scenario-manifest", str(val_scenarios), "--val-split-manifest", str(val_split),
        "--val-export-manifest", str(val_export),
        "--exclusion-manifests", str(exclusion),
        "--output", str(output),
    ])
    assert rc == 1
    assert not output.exists()


def test_freeze_manifest_refuses_when_exclusion_overlap(tmp_path):
    train_scenarios = _write(tmp_path / "train_scen.json", _fixture_scenario_manifest(["a", "b"], 1, "train"))
    train_split = _write(tmp_path / "train_split.json", _fixture_split_manifest(["a", "b"], "train"))
    train_export = _write(tmp_path / "train_export.json", _fixture_export_manifest(1))
    val_scenarios = _write(tmp_path / "val_scen.json", _fixture_scenario_manifest(["x", "y"], 2, "val"))
    val_split = _write(tmp_path / "val_split.json", _fixture_split_manifest(["x", "y"], "val"))
    val_export = _write(tmp_path / "val_export.json", _fixture_export_manifest(1))
    exclusion = _write(tmp_path / "exclusion.json", _fixture_scenario_manifest(["a"], 0, "train"))
    output = tmp_path / "freeze.json"

    rc = main([
        "--train-scenario-manifest", str(train_scenarios), "--train-split-manifest", str(train_split),
        "--train-export-manifest", str(train_export),
        "--val-scenario-manifest", str(val_scenarios), "--val-split-manifest", str(val_split),
        "--val-export-manifest", str(val_export),
        "--exclusion-manifests", str(exclusion),
        "--output", str(output),
    ])
    assert rc == 1
    assert not output.exists()


def test_freeze_manifest_hashes_exclusion_manifests_by_basename(tmp_path):
    train_scenarios = _write(tmp_path / "train_scen.json", _fixture_scenario_manifest(["a"], 1, "train"))
    train_split = _write(tmp_path / "train_split.json", _fixture_split_manifest(["a"], "train"))
    train_export = _write(tmp_path / "train_export.json", _fixture_export_manifest(1))
    val_scenarios = _write(tmp_path / "val_scen.json", _fixture_scenario_manifest(["x"], 2, "val"))
    val_split = _write(tmp_path / "val_split.json", _fixture_split_manifest(["x"], "val"))
    val_export = _write(tmp_path / "val_export.json", _fixture_export_manifest(1))
    exclusion = _write(tmp_path / "stage0_scenarios.json", _fixture_scenario_manifest(["p"], 0, "train"))
    output = tmp_path / "freeze.json"

    main([
        "--train-scenario-manifest", str(train_scenarios), "--train-split-manifest", str(train_split),
        "--train-export-manifest", str(train_export),
        "--val-scenario-manifest", str(val_scenarios), "--val-split-manifest", str(val_split),
        "--val-export-manifest", str(val_export),
        "--exclusion-manifests", str(exclusion),
        "--output", str(output),
    ])
    manifest = json.loads(output.read_text())
    assert "stage0_scenarios.json" in manifest["exclusion_manifests"]
    assert len(manifest["exclusion_manifests"]["stage0_scenarios.json"]) == 64  # sha256 hex digest length
