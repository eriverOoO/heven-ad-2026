"""Minimal, correct mid-run resume capability for
``batched_trainer.train_one_run_batched``.

Neither ``batched_trainer.py`` nor ``trainer_core.py`` had any resume
support before this task (verified by reading both source files): a
crash or interruption at any point meant restarting the whole run from
epoch 0. At the AV2 Scale-Up v2 10k scale (estimated ~29 hours for one
GENERIC-ROBUST seed), that is not acceptable -- this module adds exactly
the state a per-epoch checkpoint/resume cycle needs and nothing else
(no trainer redesign, no change to the per-epoch training math).

Resume checkpoint contents (one ``torch.save`` payload):

- ``model_state_dict`` / ``optimizer_state_dict`` -- exact optimizer
  state (Adam moment estimates), not just model weights, so resuming
  does not reset momentum.
- ``rng_state`` -- the ``numpy.random.RandomState`` used for per-epoch
  shuffle order, so the post-resume epoch order is not a decision made
  from scratch (it is *not* required to reproduce the pre-crash epoch's
  own already-consumed order, only to continue deterministically from
  here).
- ``epoch_next`` -- the next epoch index to run.
- ``best_val`` / ``best_epoch`` / ``best_state_dict`` / ``epochs_since_improve``
  -- exact early-stopping/checkpoint-selection state.
- ``history`` -- every completed epoch's ``EpochRecord`` (as plain
  dicts), so the final history/report is not truncated at the last
  resume point.
- ``nonfinite_step_count`` / ``cumulative_train_time_s`` -- running
  counters that must not reset to zero on resume.
- ``validation_key`` -- a caller-supplied dict of everything that must
  be IDENTICAL across a resume (seed, lr, batch_size, hidden_size,
  grad_clip, loss_on_predict_only, use_length_bucketing, n_buckets,
  dataset/split content hashes, corruption config). ``load_resume_state``
  raises ``ResumeValidationError`` on any mismatch -- a resume must never
  silently continue training under a different configuration than the
  one that produced the checkpoint so far.

Saved atomically (write to ``.tmp``, then ``os.replace``) so a crash
mid-write can never leave a corrupt/partial checkpoint that a later
resume would load.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

RESUME_SCHEMA_VERSION = "av2_kalmannet_resume_v1"


class ResumeValidationError(RuntimeError):
    """Raised when a resume checkpoint's ``validation_key`` does not
    match the configuration of the run attempting to resume from it."""


def save_resume_state(
    path: Path,
    *,
    model_state_dict: dict[str, Any],
    optimizer_state_dict: dict[str, Any],
    rng_state: tuple,
    epoch_next: int,
    best_val: float,
    best_epoch: int | None,
    best_state_dict: dict[str, Any] | None,
    epochs_since_improve: int,
    history: list[dict[str, Any]],
    nonfinite_step_count: int,
    cumulative_train_time_s: float,
    validation_key: dict[str, Any],
    extra_counters: dict[str, Any] | None = None,
) -> None:
    """``extra_counters``, optional (default ``None`` -> stored as
    ``{}``, fully backward compatible): a forward-compatible bag for
    cumulative ``TrainResult`` counters added after this module's own
    initial design (e.g. the PR #58 gradient-state health counters --
    ``grad_skip_count``/``norm_overflow_count``/``training_unstable``/
    etc.), added as one generic dict rather than a growing list of named
    parameters so a future new counter needs no signature change here.
    Restoring these onto a resumed ``TrainResult`` is the caller's
    responsibility (see ``batched_trainer.train_one_run_batched``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": RESUME_SCHEMA_VERSION,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "rng_state": rng_state,
        "epoch_next": epoch_next,
        "best_val": best_val,
        "best_epoch": best_epoch,
        "best_state_dict": best_state_dict,
        "epochs_since_improve": epochs_since_improve,
        "history": history,
        "nonfinite_step_count": nonfinite_step_count,
        "cumulative_train_time_s": cumulative_train_time_s,
        "validation_key": validation_key,
        "extra_counters": extra_counters or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_resume_state(path: Path, *, expected_validation_key: dict[str, Any]) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu")
    if payload.get("schema_version") != RESUME_SCHEMA_VERSION:
        raise ResumeValidationError(
            f"resume checkpoint schema_version {payload.get('schema_version')!r} != {RESUME_SCHEMA_VERSION!r}"
        )
    stored_key = payload.get("validation_key", {})
    mismatches = {
        k: (stored_key.get(k), expected_validation_key[k])
        for k in expected_validation_key
        if stored_key.get(k) != expected_validation_key[k]
    }
    if mismatches:
        raise ResumeValidationError(
            f"resume checkpoint validation_key mismatch, refusing to resume under a different "
            f"configuration than the one that produced this checkpoint: {mismatches}"
        )
    return payload


def rng_state_from(rng: np.random.RandomState) -> tuple:
    return rng.get_state()


def restore_rng(rng: np.random.RandomState, state: tuple) -> None:
    rng.set_state(state)
