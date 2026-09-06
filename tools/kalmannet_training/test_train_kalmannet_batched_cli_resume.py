"""End-to-end CLI-level resume test for ``train_kalmannet_batched.py``,
against a real (fixture-scenario) sharded AV2 export -- not just the
library-level ``train_one_run_batched`` resume test in
``test_batched_trainer.py``. Exercises the exact entry point used to
launch the AV2 Scale-Up v2 10k GENERIC-ROBUST run, including CLI argument
parsing, dataset loading, the resume-checkpoint auto-path derivation, and
the per-epoch log file.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ad_morai_bridge_dev"))

import train_kalmannet_batched  # noqa: E402
from av2_split import SplitConfig, build_split_manifest, write_split_manifest  # noqa: E402

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


def _straight_track(track_id, n=20, x0=0.0, y0=0.0, vx=3.0, vy=0.5, dt=0.1):
    states = [
        Av2ObjectStateRecord(True, i, x0 + vx * dt * i, y0 + vy * dt * i, 0.0, vx, vy)
        for i in range(n)
    ]
    return Av2TrackRecord(track_id=track_id, object_type="VEHICLE", category="SCORED_TRACK", states=tuple(states))


def _scenario(sid, tracks, n_timestamps=20):
    ts = tuple(int(i * 1e8) for i in range(n_timestamps))
    return Av2ScenarioRecord(scenario_id=sid, city_name="austin", focal_track_id=tracks[0].track_id,
                              timestamps_ns=ts, tracks=tuple(tracks))


def _build_tiny_shard_export(tmp_path, monkeypatch, n_scenarios=10):
    import ad_morai_bridge_dev.dataset.av2_motion_forecasting_adapter as adapter_mod

    scenario_ids = [f"cli-resume-smoke-{i:03d}" for i in range(n_scenarios)]
    scenarios = {
        sid: _scenario(sid, [_straight_track(f"{sid}-t{j}", x0=float(j) * 50.0 + i) for j in range(2)])
        for i, sid in enumerate(scenario_ids)
    }

    def fake_loader(path: Path):
        return scenarios[path.parent.name]

    monkeypatch.setattr(adapter_mod, "load_av2_scenario_parquet", fake_loader)

    raw_root = tmp_path / "raw"
    for sid in scenario_ids:
        d = raw_root / "train" / sid
        d.mkdir(parents=True)
        (d / f"scenario_{sid}.parquet").write_bytes(b"fixture")

    output_root = tmp_path / "shards"
    exporter = Av2KalmanNetExporter(
        input_root=raw_root, output_root=output_root, split="train", scenario_ids=scenario_ids,
        segment_config=SegmentConfig(), coordinate_config=CoordinateConfig(), corruption_config=CorruptionConfig(),
        adapter_commit="test", scenario_manifest_provenance={"dataset_source": "argoverse2_motion_forecasting"},
        shard_config=ShardConfig(scenarios_per_shard=5),
    )
    exporter.export()
    return output_root, scenario_ids


def _write_split(tmp_path, scenario_ids):
    manifest = build_split_manifest(
        scenario_ids, SplitConfig(train_frac=0.7, val_frac=0.2, test_frac=0.1, seed=1),
        scenario_manifest_provenance={},
    )
    path = tmp_path / "split.json"
    write_split_manifest(manifest, path)
    return path


def test_cli_resume_reaches_the_same_result_as_an_uninterrupted_cli_run(tmp_path, monkeypatch, capsys):
    shard_root, scenario_ids = _build_tiny_shard_export(tmp_path / "data", monkeypatch)
    split_path = _write_split(tmp_path / "data", scenario_ids)

    common_args = [
        "--shard-root", str(shard_root), "--split-manifest", str(split_path),
        "--condition", "generic_robust", "--device", "cpu", "--batch-size", "2",
        "--hidden-size", "4", "--lr", "0.01", "--patience", "10", "--seed", "0",
    ]

    # Uninterrupted reference run -- SAME max_epochs as the "live" run
    # below; a real crash does not change how many epochs the run was
    # *configured* for, it just stops execution before max_epochs (or
    # early stopping) is reached.
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    rc = train_kalmannet_batched.main(common_args + [
        "--max-epochs", "5", "--no-resume",
        "--output-checkpoint", str(ref_dir / "ckpt.pt"), "--output-history", str(ref_dir / "hist.csv"),
    ])
    assert rc == 0
    ref_manifest = json.loads((ref_dir / "ckpt.pt.manifest.json").read_text("utf-8"))

    # Simulated crash: patch save_resume_state so it raises right after
    # epoch index 1's checkpoint has been durably written (epoch_next=2
    # on disk), mimicking a process kill/crash that happens to land just
    # after a checkpoint write -- the realistic failure mode this
    # capability protects against.
    live_dir = tmp_path / "live"
    live_dir.mkdir()
    output_checkpoint = live_dir / "ckpt.pt"
    log_path = live_dir / "train.log"

    import batched_trainer
    real_save = batched_trainer.save_resume_state
    call_count = {"n": 0}

    def crash_after_two_saves(*args, **kwargs):
        real_save(*args, **kwargs)
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated crash right after a resume checkpoint write")

    monkeypatch.setattr(batched_trainer, "save_resume_state", crash_after_two_saves)
    with pytest.raises(RuntimeError, match="simulated crash"):
        train_kalmannet_batched.main(common_args + [
            "--max-epochs", "5",
            "--output-checkpoint", str(output_checkpoint), "--output-history", str(live_dir / "hist.csv"),
            "--log-file", str(log_path),
        ])
    monkeypatch.setattr(batched_trainer, "save_resume_state", real_save)

    resume_checkpoint_path = output_checkpoint.with_suffix(output_checkpoint.suffix + ".resume.pt")
    assert resume_checkpoint_path.is_file(), "resume checkpoint must survive the simulated crash"

    # "Restart": rerun with the IDENTICAL arguments; it must pick up from
    # the resume checkpoint automatically and reach the same final result
    # as the uninterrupted reference run.
    rc = train_kalmannet_batched.main(common_args + [
        "--max-epochs", "5",
        "--output-checkpoint", str(output_checkpoint), "--output-history", str(live_dir / "hist.csv"),
        "--log-file", str(log_path),
    ])
    assert rc == 0
    resumed_manifest = json.loads((output_checkpoint.with_suffix(".pt.manifest.json")).read_text("utf-8"))
    assert resumed_manifest["checkpoint_sha256"] == ref_manifest["checkpoint_sha256"]  # byte-identical final weights

    assert resumed_manifest["best_epoch"] == ref_manifest["best_epoch"]
    assert resumed_manifest["best_val_loss"] == pytest.approx(ref_manifest["best_val_loss"], abs=1e-6)

    stderr_text = capsys.readouterr().err
    assert "resume checkpoint found" in stderr_text

    log_lines = [ln for ln in log_path.read_text("utf-8").splitlines() if ln.startswith("epoch=")]
    # crashed run logs epochs 0,1 (2 lines); resumed run's loop starts at
    # epoch_next=2 and only logs the NEW epochs 2,3,4 (3 lines) -- the
    # loaded epochs 0,1 are restored into result.history but are not
    # re-logged via progress_callback. 2 + 3 = 5 total log lines, one per
    # distinct epoch, never a duplicate.
    assert len(log_lines) == 5
    assert [ln.split()[0] for ln in log_lines] == [f"epoch={i}" for i in range(5)]

    with open(live_dir / "hist.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5
    assert [int(r["epoch"]) for r in rows] == [0, 1, 2, 3, 4]


def test_cli_resume_refuses_when_hyperparameters_change(tmp_path, monkeypatch):
    shard_root, scenario_ids = _build_tiny_shard_export(tmp_path / "data", monkeypatch)
    split_path = _write_split(tmp_path / "data", scenario_ids)

    live_dir = tmp_path / "live"
    live_dir.mkdir()
    output_checkpoint = live_dir / "ckpt.pt"
    base_args = [
        "--shard-root", str(shard_root), "--split-manifest", str(split_path),
        "--condition", "generic_robust", "--device", "cpu", "--batch-size", "2",
        "--hidden-size", "4", "--seed", "0", "--patience", "10",
        "--output-checkpoint", str(output_checkpoint), "--output-history", str(live_dir / "hist.csv"),
    ]
    train_kalmannet_batched.main(base_args + ["--lr", "0.01", "--max-epochs", "2"])

    from resume_state import ResumeValidationError
    with pytest.raises(ResumeValidationError):
        train_kalmannet_batched.main(base_args + ["--lr", "0.02", "--max-epochs", "5"])
