"""End-to-end fixture-based tests for the AV2 Scale-Up v2 10k
GENERIC-ROBUST evaluation pipeline: the freeze-manifest writer, the
freeze-before-official-val guard, and ``evaluate_kalmannet_official_val.py``
itself -- including the requirement that AV2-tuned KF calibration is
fitted ONLY on the internal TRAIN/VAL split and never touches the
physically separate official-VAL scenario pool.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ad_morai_bridge_dev"))

import evaluate_kalmannet_official_val as eval_mod  # noqa: E402
import train_kalmannet_batched  # noqa: E402
import write_freeze_manifest_av2_10k_generic as freeze_writer  # noqa: E402
from av2_split import SplitConfig, build_split_manifest, write_split_manifest  # noqa: E402
from build_scaleup_v2_internal_split import build_internal_split_manifest, write_manifest  # noqa: E402
from freeze_manifest import FreezeManifestMissingError  # noqa: E402

from ad_morai_bridge_dev.dataset.av2_motion_forecasting_adapter import (  # noqa: E402
    Av2KalmanNetExporter,
    Av2ObjectStateRecord,
    Av2ScenarioRecord,
    Av2TrackRecord,
    CoordinateConfig,
    CorruptionConfig,
    SegmentConfig,
    ShardConfig,
)


def _track(track_id, n=16, x0=0.0, y0=0.0, vx=2.0, vy=0.3, dt=0.1):
    states = [
        Av2ObjectStateRecord(True, i, x0 + vx * dt * i, y0 + vy * dt * i, 0.0, vx, vy)
        for i in range(n)
    ]
    return Av2TrackRecord(track_id=track_id, object_type="VEHICLE", category="SCORED_TRACK", states=tuple(states))


def _scenario(sid, tracks, n_timestamps=16):
    ts = tuple(int(i * 1e8) for i in range(n_timestamps))
    return Av2ScenarioRecord(scenario_id=sid, city_name="austin", focal_track_id=tracks[0].track_id,
                              timestamps_ns=ts, tracks=tuple(tracks))


def _build_shard_export(tmp_path, monkeypatch, scenario_ids, tag, tracks_per_scenario=2):
    import ad_morai_bridge_dev.dataset.av2_motion_forecasting_adapter as adapter_mod

    scenarios = {
        sid: _scenario(sid, [_track(f"{sid}-t{j}", x0=float(j) * 30.0 + i, vx=2.0 + 0.1 * i)
                              for j in range(tracks_per_scenario)])
        for i, sid in enumerate(scenario_ids)
    }

    def fake_loader(path: Path):
        return scenarios[path.parent.name]

    monkeypatch.setattr(adapter_mod, "load_av2_scenario_parquet", fake_loader)

    raw_root = tmp_path / f"raw_{tag}"
    for sid in scenario_ids:
        d = raw_root / "train" / sid
        d.mkdir(parents=True)
        (d / f"scenario_{sid}.parquet").write_bytes(b"fixture")

    output_root = tmp_path / f"shards_{tag}"
    exporter = Av2KalmanNetExporter(
        input_root=raw_root, output_root=output_root, split="train", scenario_ids=scenario_ids,
        segment_config=SegmentConfig(), coordinate_config=CoordinateConfig(), corruption_config=CorruptionConfig(),
        adapter_commit="test", scenario_manifest_provenance={"dataset_source": "argoverse2_motion_forecasting"},
        shard_config=ShardConfig(scenarios_per_shard=5),
    )
    exporter.export()
    return output_root


def _train_a_tiny_checkpoint(tmp_path, monkeypatch):
    """Builds a physically separate TRAIN-source pool (12 scenarios) and
    OFFICIAL-VAL pool (4 scenarios) -- disjoint ids, mirroring the real
    Scale-Up v2 layout -- trains a tiny checkpoint on an internal 9/3
    split of the TRAIN pool, and returns everything needed to drive the
    freeze + evaluation scripts."""
    train_ids = [f"train-scn-{i:03d}" for i in range(12)]
    official_val_ids = [f"official-val-scn-{i:03d}" for i in range(4)]

    train_root = _build_shard_export(tmp_path, monkeypatch, train_ids, tag="train")
    official_val_root = _build_shard_export(tmp_path, monkeypatch, official_val_ids, tag="official_val")

    internal_split_manifest = build_internal_split_manifest(
        train_ids, scenario_manifest_provenance={}, n_train=9, n_val=3, seed=20260930,
    )
    internal_split_path = tmp_path / "internal_split.json"
    write_manifest(internal_split_manifest, internal_split_path)

    official_val_split_manifest = {
        "schema_version": internal_split_manifest["schema_version"],
        "split_policy": "single_role_physically_separate_dataset",
        "not_av2_official_test_split": True,
        "config": {"role": "val"},
        "scenario_count": len(official_val_ids),
        "split_counts": {"train": 0, "val": len(official_val_ids), "test": 0},
        "splits": {"train": [], "val": sorted(official_val_ids), "test": []},
        "scenario_manifest_provenance": {},
    }
    official_val_split_path = tmp_path / "official_val_split.json"
    official_val_split_path.write_text(json.dumps(official_val_split_manifest), encoding="utf-8")

    checkpoint_path = tmp_path / "ckpt.pt"
    rc = train_kalmannet_batched.main([
        "--shard-root", str(train_root), "--split-manifest", str(internal_split_path),
        "--condition", "generic_robust", "--device", "cpu", "--batch-size", "2",
        "--hidden-size", "4", "--lr", "0.01", "--max-epochs", "2", "--patience", "10", "--seed", "0",
        "--no-resume",
        "--output-checkpoint", str(checkpoint_path), "--output-history", str(tmp_path / "hist.csv"),
    ])
    assert rc == 0

    return {
        "train_root": train_root, "official_val_root": official_val_root,
        "internal_split_path": internal_split_path, "official_val_split_path": official_val_split_path,
        "checkpoint_path": checkpoint_path, "train_ids": train_ids, "official_val_ids": official_val_ids,
    }


def test_evaluation_refuses_without_a_freeze_manifest(tmp_path, monkeypatch):
    ctx = _train_a_tiny_checkpoint(tmp_path, monkeypatch)
    missing_freeze = tmp_path / "does_not_exist_freeze.json"
    with pytest.raises(FreezeManifestMissingError):
        eval_mod.main([
            "--freeze-manifest", str(missing_freeze),
            "--checkpoint", str(ctx["checkpoint_path"]), "--training-condition", "generic_robust",
            "--calibration-shard-root", str(ctx["train_root"]),
            "--calibration-split-manifest", str(ctx["internal_split_path"]),
            "--official-val-shard-root", str(ctx["official_val_root"]),
            "--official-val-split-manifest", str(ctx["official_val_split_path"]),
            "--skip-dense-v2",
            "--output", str(tmp_path / "report.json"),
        ])


def test_freeze_writer_rejects_mismatched_split_manifest(tmp_path, monkeypatch):
    ctx = _train_a_tiny_checkpoint(tmp_path, monkeypatch)
    wrong_split = build_split_manifest(
        ctx["train_ids"], SplitConfig(train_frac=0.7, val_frac=0.2, test_frac=0.1, seed=999),
        scenario_manifest_provenance={},
    )
    wrong_split_path = tmp_path / "wrong_split.json"
    write_split_manifest(wrong_split, wrong_split_path)
    with pytest.raises(ValueError):
        freeze_writer.main([
            "--checkpoint", str(ctx["checkpoint_path"]),
            "--train-scenario-manifest-sha256", "deadbeef",
            "--internal-split-manifest", str(wrong_split_path),
            "--official-val-scenario-manifest-sha256", "deadbeef2",
            "--output", str(tmp_path / "freeze.json"),
        ])


def test_full_pipeline_freeze_then_official_val_evaluation(tmp_path, monkeypatch):
    ctx = _train_a_tiny_checkpoint(tmp_path, monkeypatch)
    freeze_path = tmp_path / "freeze.json"
    rc = freeze_writer.main([
        "--checkpoint", str(ctx["checkpoint_path"]),
        "--train-scenario-manifest-sha256", "deadbeef",
        "--internal-split-manifest", str(ctx["internal_split_path"]),
        "--official-val-scenario-manifest-sha256", "deadbeef2",
        "--output", str(freeze_path),
    ])
    assert rc == 0
    assert freeze_path.is_file()

    report_path = tmp_path / "report.json"
    rc = eval_mod.main([
        "--freeze-manifest", str(freeze_path),
        "--checkpoint", str(ctx["checkpoint_path"]), "--training-condition", "generic_robust",
        "--calibration-shard-root", str(ctx["train_root"]),
        "--calibration-split-manifest", str(ctx["internal_split_path"]),
        "--official-val-shard-root", str(ctx["official_val_root"]),
        "--official-val-split-manifest", str(ctx["official_val_split_path"]),
        "--add-morai-calibrated-diagnostic", "--skip-dense-v2",
        "--output", str(report_path),
    ])
    assert rc == 0
    report = json.loads(report_path.read_text("utf-8"))

    assert set(report["conditions"]) == {
        "A_clean_official_val", "B_same_corruption_family", "C_different_corruption_seed",
        "D_morai_calibrated_diagnostic",
    }
    for cond_name, cond_report in report["conditions"].items():
        assert cond_report["n_official_val_sequences"] == len(ctx["official_val_ids"]) * 2  # 2 tracks/scenario
        assert "overall" in cond_report["knet"]["buckets"]
        assert "linear_kf_av2_tuned" in cond_report
        assert cond_report["knet_motion_stratified"]  # at least one motion bucket populated

    kf_calib = report["linear_kf_av2_tuned"]
    assert "internal" in kf_calib["provenance"] and "TRAIN" in kf_calib["provenance"]
    assert "official AV2 VAL never touched" in kf_calib["provenance"]
    assert kf_calib["params"]["sigma_a"] > 0


def test_kf_calibration_never_reads_official_val_sequences(tmp_path, monkeypatch):
    """The core anti-leakage guarantee: patch load_split_sequences to
    detect any call whose shard_root is the OFFICIAL VAL root being used
    for calibration (it must only ever be called with the TRAIN/internal
    root for calibration purposes)."""
    ctx = _train_a_tiny_checkpoint(tmp_path, monkeypatch)
    freeze_path = tmp_path / "freeze.json"
    freeze_writer.main([
        "--checkpoint", str(ctx["checkpoint_path"]),
        "--train-scenario-manifest-sha256", "deadbeef",
        "--internal-split-manifest", str(ctx["internal_split_path"]),
        "--official-val-scenario-manifest-sha256", "deadbeef2",
        "--output", str(freeze_path),
    ])

    calls = []
    real_load = eval_mod.load_split_sequences

    def spy(shard_root, split_manifest, corruption_config=None, only_valid_for_kalmannet_gt=True):
        calls.append(Path(shard_root))
        return real_load(shard_root, split_manifest, corruption_config=corruption_config,
                          only_valid_for_kalmannet_gt=only_valid_for_kalmannet_gt)

    monkeypatch.setattr(eval_mod, "load_split_sequences", spy)

    eval_mod.main([
        "--freeze-manifest", str(freeze_path),
        "--checkpoint", str(ctx["checkpoint_path"]), "--training-condition", "generic_robust",
        "--calibration-shard-root", str(ctx["train_root"]),
        "--calibration-split-manifest", str(ctx["internal_split_path"]),
        "--official-val-shard-root", str(ctx["official_val_root"]),
        "--official-val-split-manifest", str(ctx["official_val_split_path"]),
        "--skip-dense-v2",
        "--output", str(tmp_path / "report.json"),
    ])

    # The FIRST call (calibration) must be against the TRAIN root; every
    # subsequent call (one per evaluation condition) must be against the
    # official-VAL root. The train root must never receive more than the
    # single calibration call.
    assert calls[0] == Path(ctx["train_root"])
    train_root_calls = [c for c in calls if c == Path(ctx["train_root"])]
    official_val_calls = [c for c in calls if c == Path(ctx["official_val_root"])]
    assert len(train_root_calls) == 1
    assert len(official_val_calls) == len(calls) - 1
    assert len(official_val_calls) >= 1


def test_precomputed_kf_calibration_json_round_trips_and_skips_recalibration(tmp_path, monkeypatch):
    """Regression test for a real bug found while running this pipeline:
    the first report's ``linear_kf_av2_tuned`` dict uses
    ``internal_val_position_rmse``/``internal_val_velocity_rmse`` keys
    (not ``val_position_rmse``/``val_velocity_rmse``) -- reusing that
    report as ``--precomputed-kf-calibration-json`` for a second
    checkpoint must not KeyError, and must not touch the calibration
    (TRAIN) root at all."""
    ctx = _train_a_tiny_checkpoint(tmp_path, monkeypatch)
    freeze_path = tmp_path / "freeze.json"
    freeze_writer.main([
        "--checkpoint", str(ctx["checkpoint_path"]),
        "--train-scenario-manifest-sha256", "deadbeef",
        "--internal-split-manifest", str(ctx["internal_split_path"]),
        "--official-val-scenario-manifest-sha256", "deadbeef2",
        "--output", str(freeze_path),
    ])
    first_report_path = tmp_path / "report_first.json"
    eval_mod.main([
        "--freeze-manifest", str(freeze_path),
        "--checkpoint", str(ctx["checkpoint_path"]), "--training-condition", "generic_robust",
        "--calibration-shard-root", str(ctx["train_root"]),
        "--calibration-split-manifest", str(ctx["internal_split_path"]),
        "--official-val-shard-root", str(ctx["official_val_root"]),
        "--official-val-split-manifest", str(ctx["official_val_split_path"]),
        "--skip-dense-v2",
        "--output", str(first_report_path),
    ])

    calls = []
    real_load = eval_mod.load_split_sequences

    def spy(shard_root, split_manifest, corruption_config=None, only_valid_for_kalmannet_gt=True):
        calls.append(Path(shard_root))
        return real_load(shard_root, split_manifest, corruption_config=corruption_config,
                          only_valid_for_kalmannet_gt=only_valid_for_kalmannet_gt)

    monkeypatch.setattr(eval_mod, "load_split_sequences", spy)

    second_report_path = tmp_path / "report_second.json"
    rc = eval_mod.main([
        "--freeze-manifest", str(freeze_path),
        "--checkpoint", str(ctx["checkpoint_path"]), "--training-condition", "generic_robust",
        "--calibration-shard-root", str(ctx["train_root"]),
        "--calibration-split-manifest", str(ctx["internal_split_path"]),
        "--official-val-shard-root", str(ctx["official_val_root"]),
        "--official-val-split-manifest", str(ctx["official_val_split_path"]),
        "--precomputed-kf-calibration-json", str(first_report_path),
        "--skip-dense-v2",
        "--output", str(second_report_path),
    ])
    assert rc == 0
    # The calibration (TRAIN) root must never be touched when reusing a
    # precomputed calibration.
    assert Path(ctx["train_root"]) not in calls
    assert all(c == Path(ctx["official_val_root"]) for c in calls)

    second_report = json.loads(second_report_path.read_text("utf-8"))
    first_report = json.loads(first_report_path.read_text("utf-8"))
    assert second_report["linear_kf_av2_tuned"]["params"] == first_report["linear_kf_av2_tuned"]["params"]
