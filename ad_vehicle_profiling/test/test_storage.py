import json
import os

import pytest

from ad_vehicle_profiling.experiment import (
    ExperimentCell,
    TrialSample,
    TrialSummary,
)
from ad_vehicle_profiling.storage import RunStore


def _cell(speed=0, kind="accelerator", command=0):
    return ExperimentCell(speed, kind, command)


def _summary(*, valid=True, reason=None):
    return TrialSummary(
        valid=valid,
        rejection_reason=reason,
        sample_count=10,
        mean_acceleration_mps2=1.0 if valid else None,
        median_acceleration_mps2=1.0 if valid else None,
        acceleration_stddev_mps2=0.02 if valid else None,
        acceleration_mad_mps2=0.01 if valid else None,
        minimum_acceleration_mps2=0.9 if valid else None,
        maximum_acceleration_mps2=1.1 if valid else None,
        velocity_derived_acceleration_mps2=1.02 if valid else None,
        cross_check_disagreement_mps2=0.02 if valid else None,
        peak_abs_jerk_mps3=1.5 if valid else None,
        effective_acceleration_mps2=1.02 if valid else None,
        acceleration_source="velocity_derived" if valid else "unavailable",
    )


def _manifest():
    return {
        "run_id": "test-run",
        "minimum_valid_trials": 3,
        "maximum_attempts": 7,
    }


def test_store_resumes_at_first_incomplete_cell(tmp_path):
    cells = (_cell(0), _cell(10))
    store = RunStore.create(tmp_path / "run", _manifest(), cells=cells)
    store.append_trial(cells[0], _summary())

    resumed = RunStore.resume(tmp_path / "run")

    assert resumed.valid_trial_count(cells[0]) == 1
    assert resumed.attempted_trial_count(cells[0]) == 1
    assert resumed.pending_cells()[0] == cells[0]


def test_store_preserves_effective_acceleration_and_source(tmp_path):
    cell = _cell(10, "accelerator", 10)
    store = RunStore.create(tmp_path / "run", _manifest(), cells=(cell,))
    store.append_trial(cell, _summary())

    summary = RunStore.resume(tmp_path / "run").summaries(cell)[0]

    assert summary.effective_acceleration_mps2 == pytest.approx(1.02)
    assert summary.acceleration_source == "velocity_derived"


def test_three_valid_trials_complete_a_cell(tmp_path):
    cells = (_cell(0), _cell(10))
    store = RunStore.create(tmp_path / "run", _manifest(), cells=cells)
    for _ in range(3):
        store.append_trial(cells[0], _summary())

    assert store.pending_cells() == (cells[1],)


def test_unstable_valid_trials_repeat_until_attempt_limit(tmp_path):
    cells = (_cell(0),)
    manifest = {
        **_manifest(),
        "mad_limit_mps2": 0.15,
        "cross_check_limit_mps2": 0.2,
    }
    store = RunStore.create(tmp_path / "run", manifest, cells=cells)
    unstable = TrialSummary(
        **{
            **_summary().__dict__,
            "acceleration_mad_mps2": 0.4,
        }
    )
    for _ in range(3):
        store.append_trial(cells[0], unstable)

    assert store.pending_cells() == cells

    for _ in range(4):
        store.append_trial(cells[0], unstable)
    assert store.pending_cells() == ()
    assert store.cell_status(cells[0]) == "attempt_limit"


def test_rejected_trials_count_toward_attempt_limit(tmp_path):
    cells = (_cell(0),)
    store = RunStore.create(tmp_path / "run", _manifest(), cells=cells)
    for _ in range(7):
        store.append_trial(cells[0], _summary(valid=False, reason="stale"))

    assert store.attempted_trial_count(cells[0]) == 7
    assert store.valid_trial_count(cells[0]) == 0
    assert store.pending_cells() == ()
    assert store.cell_status(cells[0]) == "attempt_limit"


def test_limiter_bound_cell_is_not_scheduled_again(tmp_path):
    cells = (_cell(185, "accelerator", 100),)
    store = RunStore.create(tmp_path / "run", _manifest(), cells=cells)

    store.classify_cell(cells[0], "limiter_bound")

    assert store.pending_cells() == ()
    assert RunStore.resume(tmp_path / "run").cell_status(cells[0]) == (
        "limiter_bound"
    )


def test_state_update_uses_atomic_replace(tmp_path, monkeypatch):
    store = RunStore.create(tmp_path / "run", _manifest(), cells=(_cell(),))
    replacements = []
    real_replace = os.replace

    def record_replace(source, target):
        replacements.append((source, target))
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", record_replace)

    store.write_state({"phase": "SETTLE"})

    assert replacements
    assert replacements[-1][1].name == "run_state.json"
    assert json.loads((tmp_path / "run" / "run_state.json").read_text()) == {
        "phase": "SETTLE"
    }


def test_interrupted_temporary_state_does_not_replace_good_state(tmp_path):
    store = RunStore.create(tmp_path / "run", _manifest(), cells=(_cell(),))
    store.write_state({"phase": "SETTLE"})
    (tmp_path / "run" / "run_state.json.tmp").write_text("{broken")

    resumed = RunStore.resume(tmp_path / "run")

    assert resumed.read_state() == {"phase": "SETTLE"}


def test_resume_ignores_only_incomplete_trailing_csv_row(tmp_path):
    store = RunStore.create(tmp_path / "run", _manifest(), cells=(_cell(),))
    store.append_trial(_cell(), _summary())
    with (tmp_path / "run" / "trials.csv").open("a", encoding="utf-8") as stream:
        stream.write("0,accelerator")

    resumed = RunStore.resume(tmp_path / "run")

    assert resumed.attempted_trial_count(_cell()) == 1


def test_create_rejects_existing_nonempty_directory(tmp_path):
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    (run_directory / "foreign.txt").write_text("do not overwrite")

    with pytest.raises(FileExistsError):
        RunStore.create(run_directory, _manifest(), cells=(_cell(),))


def test_raw_samples_are_flushed_with_trial_identity(tmp_path):
    cell = _cell(20, "brake", 15)
    store = RunStore.create(tmp_path / "run", _manifest(), cells=(cell,))

    store.append_samples(
        cell,
        trial_index=2,
        samples=(
            TrialSample(
                elapsed_sec=0.25,
                velocity_mps=5.0,
                acceleration_mps2=-1.2,
                requested_brake=0.15,
                echoed_brake=0.15,
                position_x_m=10.0,
                position_y_m=2.0,
                yaw_rad=0.1,
                gear=4,
                link_id="highway-link",
            ),
        ),
    )

    lines = (tmp_path / "run" / "raw_samples.csv").read_text().splitlines()
    assert len(lines) == 2
    assert "trial_index" in lines[0]
    assert "highway-link" in lines[1]
