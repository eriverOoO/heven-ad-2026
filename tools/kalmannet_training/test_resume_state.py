from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from resume_state import (  # noqa: E402
    ResumeValidationError,
    load_resume_state,
    save_resume_state,
)


def _dummy_payload_kwargs(seed: int = 0):
    net_state = {"w": torch.zeros(3)}
    opt_state = {"state": {}, "param_groups": []}
    rng = np.random.RandomState(seed)
    return dict(
        model_state_dict=net_state,
        optimizer_state_dict=opt_state,
        rng_state=rng.get_state(),
        epoch_next=5,
        best_val=1.23,
        best_epoch=4,
        best_state_dict=net_state,
        epochs_since_improve=2,
        history=[{"epoch": 0, "train_loss": 1.0, "val_loss": 1.0, "grad_norm_mean": 0.1,
                  "grad_norm_max": 0.2, "any_nan_train": False, "any_nan_val": False}],
        nonfinite_step_count=0,
        cumulative_train_time_s=123.4,
        validation_key={"seed": 0, "lr": 0.004, "batch_size": 64},
    )


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "resume.pt"
    kwargs = _dummy_payload_kwargs()
    save_resume_state(path, **kwargs)
    loaded = load_resume_state(path, expected_validation_key=kwargs["validation_key"])
    assert loaded["epoch_next"] == 5
    assert loaded["best_val"] == pytest.approx(1.23)
    assert loaded["best_epoch"] == 4
    assert loaded["epochs_since_improve"] == 2
    assert loaded["nonfinite_step_count"] == 0
    assert loaded["cumulative_train_time_s"] == pytest.approx(123.4)
    assert loaded["history"][0]["epoch"] == 0
    assert torch.equal(loaded["model_state_dict"]["w"], torch.zeros(3))


def test_load_rejects_mismatched_validation_key(tmp_path):
    path = tmp_path / "resume.pt"
    kwargs = _dummy_payload_kwargs()
    save_resume_state(path, **kwargs)
    with pytest.raises(ResumeValidationError):
        load_resume_state(path, expected_validation_key={"seed": 0, "lr": 0.999, "batch_size": 64})


def test_load_rejects_missing_key_present_in_expected(tmp_path):
    path = tmp_path / "resume.pt"
    kwargs = _dummy_payload_kwargs()
    save_resume_state(path, **kwargs)
    with pytest.raises(ResumeValidationError):
        load_resume_state(path, expected_validation_key={**kwargs["validation_key"], "new_key_not_present": True})


def test_save_is_atomic_no_tmp_file_left(tmp_path):
    path = tmp_path / "resume.pt"
    save_resume_state(path, **_dummy_payload_kwargs())
    assert path.is_file()
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_overwriting_resume_checkpoint_keeps_latest_state(tmp_path):
    path = tmp_path / "resume.pt"
    kwargs = _dummy_payload_kwargs()
    save_resume_state(path, **kwargs)
    kwargs2 = {**kwargs, "epoch_next": 6, "best_val": 0.5}
    save_resume_state(path, **kwargs2)
    loaded = load_resume_state(path, expected_validation_key=kwargs["validation_key"])
    assert loaded["epoch_next"] == 6
    assert loaded["best_val"] == pytest.approx(0.5)
