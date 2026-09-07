"""End-to-end tests for the training-instability escalation policy
(tier 1: isolated skip-and-continue; tier 2: repeated skips -> abort as
unstable; tier 3: parameter/optimizer-state collapse -> immediate
catastrophic abort) in BOTH trainers (`batched_trainer.train_one_run_batched`
and `trainer_core.train_one_run`).

Uses controlled monkeypatching of ``safe_clip_and_step`` (rather than
trying to naturally trigger a non-finite gradient through real KalmanNet
data, which was found during this task's own forensics to require the
GRU's specific recurrent dynamics over many real timesteps, not easily
reproduced with a synthetic scale sweep -- see
``docs/perception/kalmannet_training_instability_v1.md`` "First
Non-Finite Transition"). This directly exercises the escalation-policy
CODE PATHS with deterministic, controlled inputs -- the *mechanism*
itself (clip_grad_norm_ + Adam permanent poisoning) is separately proven
against real gradients in ``test_nonfinite_guard.py``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import batched_trainer  # noqa: E402
import trainer_core  # noqa: E402
from nonfinite_guard import SafeStepOutcome  # noqa: E402


def _make_sequence(n=8, dt=0.1, x0=0.0, y0=0.0, vx=2.0, vy=-1.0, seed=0):
    rng = np.random.RandomState(seed)
    frames = list(range(n))
    dts = [None] + [dt] * (n - 1)
    x_true, z_meas = [], []
    for i in range(n):
        x, y = x0 + vx * dt * i, y0 + vy * dt * i
        x_true.append([x, y, vx, vy])
        z_meas.append([x + rng.normal(0, 0.01), y + rng.normal(0, 0.01)])
    return {"actor_id": f"a{x0}", "frames": frames, "dt": dts, "x_true": x_true, "z_meas": z_meas}


def _skip_outcome():
    return SafeStepOutcome(
        applied=False, grad_norm_pre_clip=float("inf"),
        grad_nonfinite_pre_clip=True, grad_nonfinite_post_clip=False,
    )


def _collapsed_outcome():
    return SafeStepOutcome(
        applied=True, grad_norm_pre_clip=1.0,
        grad_nonfinite_pre_clip=False, grad_nonfinite_post_clip=False,
        params_nonfinite_after_step=True, optimizer_state_nonfinite_after_step=False,
    )


def _norm_overflow_outcome():
    return SafeStepOutcome(
        applied=False, grad_norm_pre_clip=float("inf"),
        grad_nonfinite_pre_clip=False, grad_nonfinite_post_clip=False,
        gradient_state="norm_overflow", norm_overflow=True,
        grad_norm_robust=1.2e20,
    )


def _applied_outcome(real_outcome):
    return real_outcome


# ---------------------------------------------------------------------------
# batched_trainer.train_one_run_batched
# ---------------------------------------------------------------------------

def test_batched_isolated_skip_does_not_abort_training(monkeypatch):
    train_seqs = [_make_sequence(n=8, x0=float(i), seed=i) for i in range(6)]
    val_seqs = [_make_sequence(n=6, x0=float(i) + 0.5, seed=100 + i) for i in range(2)]

    real_fn = batched_trainer.safe_clip_and_step
    call_count = {"n": 0}

    def flaky(net, opt, grad_clip):
        call_count["n"] += 1
        if call_count["n"] == 1:  # exactly ONE isolated skip across the whole run
            return _skip_outcome()
        return real_fn(net, opt, grad_clip)

    monkeypatch.setattr(batched_trainer, "safe_clip_and_step", flaky)
    result = batched_trainer.train_one_run_batched(
        train_seqs, val_seqs, seed=0, device="cpu", batch_size=2, use_length_bucketing=False,
        lr=0.01, max_epochs=4, patience=10, hidden_size=4,
    )
    assert result.grad_skip_count == 1
    assert result.training_unstable is False
    assert result.training_collapsed is False
    assert result.catastrophic is False
    assert result.n_epochs_run == 4  # ran the full budget, not aborted


def test_batched_repeated_skips_in_one_epoch_marks_unstable_and_aborts(monkeypatch):
    train_seqs = [_make_sequence(n=8, x0=float(i), seed=i) for i in range(20)]
    val_seqs = [_make_sequence(n=6, x0=float(i) + 0.5, seed=100 + i) for i in range(2)]

    def always_skip(net, opt, grad_clip):
        return _skip_outcome()

    monkeypatch.setattr(batched_trainer, "safe_clip_and_step", always_skip)
    result = batched_trainer.train_one_run_batched(
        train_seqs, val_seqs, seed=0, device="cpu", batch_size=2, use_length_bucketing=False,
        lr=0.01, max_epochs=10, patience=10, hidden_size=4,
    )
    assert result.training_unstable is True
    assert result.abort_reason is not None
    assert "repeated_nonfinite_gradients" in result.abort_reason
    assert result.n_epochs_run == 1  # aborted after the FIRST epoch, never reached epoch budget
    # Every gradient was skipped -- best_state should still be whatever
    # val loop produced (network never updated, so val_loss is whatever
    # the randomly-initialized network gives -- may or may not be
    # "best", but must not be catastrophic purely from this).
    assert result.training_collapsed is False


def test_batched_isolated_skip_below_threshold_does_not_trigger_unstable(monkeypatch):
    """Exactly at the threshold (5) must NOT trigger -- only STRICTLY
    more than the threshold does (see MAX_NONFINITE_GRAD_SKIPS_PER_EPOCH
    docstring: 'More than this many ... signals instability')."""
    train_seqs = [_make_sequence(n=8, x0=float(i), seed=i) for i in range(12)]
    val_seqs = [_make_sequence(n=6, x0=float(i) + 0.5, seed=100 + i) for i in range(2)]

    real_fn = batched_trainer.safe_clip_and_step
    call_count = {"n": 0}

    def skip_five_then_normal(net, opt, grad_clip):
        call_count["n"] += 1
        if call_count["n"] <= 5:
            return _skip_outcome()
        return real_fn(net, opt, grad_clip)

    monkeypatch.setattr(batched_trainer, "safe_clip_and_step", skip_five_then_normal)
    result = batched_trainer.train_one_run_batched(
        train_seqs, val_seqs, seed=0, device="cpu", batch_size=2, use_length_bucketing=False,
        lr=0.01, max_epochs=3, patience=10, hidden_size=4,
    )
    assert result.grad_skip_count == 5
    assert result.training_unstable is False
    assert result.n_epochs_run == 3


def test_batched_collapse_triggers_immediate_catastrophic_abort(monkeypatch):
    train_seqs = [_make_sequence(n=8, x0=float(i), seed=i) for i in range(6)]
    val_seqs = [_make_sequence(n=6, x0=float(i) + 0.5, seed=100 + i) for i in range(2)]

    call_count = {"n": 0}

    def collapse_on_second_call(net, opt, grad_clip):
        call_count["n"] += 1
        if call_count["n"] == 2:
            return _collapsed_outcome()
        return _skip_outcome() if call_count["n"] == 1 else _collapsed_outcome()

    monkeypatch.setattr(batched_trainer, "safe_clip_and_step", collapse_on_second_call)
    result = batched_trainer.train_one_run_batched(
        train_seqs, val_seqs, seed=0, device="cpu", batch_size=2, use_length_bucketing=False,
        lr=0.01, max_epochs=10, patience=10, hidden_size=4,
    )
    assert result.training_collapsed is True
    assert result.abort_reason == "params_or_optimizer_state_nonfinite_after_step"
    assert result.catastrophic is True  # THE core catastrophic-flag fix: collapse always forces this True
    assert result.n_epochs_run == 1  # aborted mid-run, not another 9 epochs of NaN


def test_batched_collapse_forces_catastrophic_true_even_with_a_good_earlier_checkpoint(monkeypatch):
    """The EXACT real-world scenario this task's own bug report
    describes: a good checkpoint was already secured (best_epoch found),
    and the model collapses LATER. catastrophic must be True regardless."""
    train_seqs = [_make_sequence(n=8, x0=float(i), seed=i) for i in range(6)]
    val_seqs = [_make_sequence(n=6, x0=float(i) + 0.5, seed=100 + i) for i in range(2)]

    real_fn = batched_trainer.safe_clip_and_step
    call_count = {"n": 0}

    def normal_then_collapse(net, opt, grad_clip):
        call_count["n"] += 1
        # Run several epochs' worth of normal steps first (letting a
        # real best_epoch/best_state get recorded), then collapse.
        if call_count["n"] <= 9:  # 3 batches/epoch x 3 epochs
            return real_fn(net, opt, grad_clip)
        return _collapsed_outcome()

    monkeypatch.setattr(batched_trainer, "safe_clip_and_step", normal_then_collapse)
    result = batched_trainer.train_one_run_batched(
        train_seqs, val_seqs, seed=0, device="cpu", batch_size=2, use_length_bucketing=False,
        lr=0.01, max_epochs=10, patience=10, hidden_size=4,
    )
    assert result.best_epoch is not None  # a real checkpoint WAS secured
    assert result.best_state_dict is not None
    assert math.isfinite(result.best_val)
    assert result.training_collapsed is True
    assert result.catastrophic is True  # <-- the bug: this used to be False in this exact scenario


# ---------------------------------------------------------------------------
# STATE B (norm overflow) accounting -- must NOT feed the STATE-C-specific
# tier-2 escalation counter (see docs/perception/
# kalmannet_gradient_norm_overflow_v1.md section 4/5).
# ---------------------------------------------------------------------------

def test_batched_repeated_norm_overflow_events_do_not_trigger_state_c_escalation(monkeypatch):
    """Many STATE-B (norm-overflow) skips in one epoch -- well beyond the
    STATE-C tier-2 threshold of 5 -- must NOT mark training_unstable or
    abort, since STATE B is self-neutralizing and has no proven Adam-
    poisoning risk. Only STATE C escalates."""
    train_seqs = [_make_sequence(n=8, x0=float(i), seed=i) for i in range(20)]
    val_seqs = [_make_sequence(n=6, x0=float(i) + 0.5, seed=100 + i) for i in range(2)]

    def always_norm_overflow(net, opt, grad_clip):
        return _norm_overflow_outcome()

    monkeypatch.setattr(batched_trainer, "safe_clip_and_step", always_norm_overflow)
    result = batched_trainer.train_one_run_batched(
        train_seqs, val_seqs, seed=0, device="cpu", batch_size=2, use_length_bucketing=False,
        lr=0.01, max_epochs=4, patience=10, hidden_size=4,
    )
    assert result.training_unstable is False
    assert result.training_collapsed is False
    assert result.n_epochs_run == 4  # ran the full budget -- STATE B never aborts
    assert result.norm_overflow_count == 10 * 4  # 20 seqs / batch_size=2 = 10 batches/epoch, 4 epochs
    assert result.norm_overflow_skip_count == result.norm_overflow_count
    assert result.per_element_nonfinite_gradient_count == 0
    assert result.nonfinite_gradient_skip_count == 0
    assert result.grad_skip_count == result.norm_overflow_count  # combined counter includes STATE B


def test_batched_norm_overflow_and_element_nonfinite_are_tracked_in_separate_counters(monkeypatch):
    """A mix of STATE-B and STATE-C events must never let one hide inside
    the other's dedicated counter."""
    train_seqs = [_make_sequence(n=8, x0=float(i), seed=i) for i in range(20)]
    val_seqs = [_make_sequence(n=6, x0=float(i) + 0.5, seed=100 + i) for i in range(2)]

    call_count = {"n": 0}

    def alternating(net, opt, grad_clip):
        call_count["n"] += 1
        return _norm_overflow_outcome() if call_count["n"] % 2 == 0 else _skip_outcome()

    monkeypatch.setattr(batched_trainer, "safe_clip_and_step", alternating)
    result = batched_trainer.train_one_run_batched(
        train_seqs, val_seqs, seed=0, device="cpu", batch_size=2, use_length_bucketing=False,
        lr=0.01, max_epochs=1, patience=10, hidden_size=4,
    )
    # 10 batches in this one epoch: 5 STATE C (odd calls), 5 STATE B (even calls).
    # STATE C alone (5) does not exceed the >5 threshold, so no abort.
    assert result.per_element_nonfinite_gradient_count == 5
    assert result.norm_overflow_count == 5
    assert result.nonfinite_gradient_skip_count == 5
    assert result.norm_overflow_skip_count == 5
    assert result.grad_skip_count == 10
    assert result.training_unstable is False


def test_batched_large_finite_gradient_count_tracks_actual_clipping(monkeypatch):
    """STATE A steps whose grad_norm_pre_clip exceeds grad_clip (i.e.
    clip_grad_norm_ actually rescaled something) are counted distinctly
    from trivially-small-gradient healthy steps. Fully deterministic --
    every call is mocked, so the count depends only on the fixed
    grad_norm_pre_clip values chosen here, never on real gradient
    magnitudes from synthetic data."""
    train_seqs = [_make_sequence(n=8, x0=float(i), seed=i) for i in range(6)]
    val_seqs = [_make_sequence(n=6, x0=float(i) + 0.5, seed=100 + i) for i in range(2)]

    # 6 seqs / batch_size=2 = 3 batches/epoch, 1 epoch -> exactly 3 calls.
    reported_norms = [3.0, 10.0, 25.0]  # below, exactly at, above grad_clip=10.0
    call_count = {"n": 0}

    def fixed_norms(net, opt, grad_clip):
        norm = reported_norms[call_count["n"]]
        call_count["n"] += 1
        return SafeStepOutcome(
            applied=True, grad_norm_pre_clip=norm,
            grad_nonfinite_pre_clip=False, grad_nonfinite_post_clip=False,
            gradient_state="healthy", grad_norm_robust=norm,
        )

    monkeypatch.setattr(batched_trainer, "safe_clip_and_step", fixed_norms)
    result = batched_trainer.train_one_run_batched(
        train_seqs, val_seqs, seed=0, device="cpu", batch_size=2, use_length_bucketing=False,
        lr=0.01, max_epochs=1, patience=10, hidden_size=4, grad_clip=10.0,
    )
    # Strictly greater than grad_clip only -- 25.0 counts, 3.0 and the
    # exactly-at-threshold 10.0 do not.
    assert result.large_finite_gradient_count == 1


def test_batched_parameter_and_optimizer_state_collapse_counted_separately(monkeypatch):
    """Section 5: parameter_collapse_count and optimizer_state_collapse_count
    must be tracked independently -- a collapse event may set either or
    both flags, and a caller diagnosing WHICH failed should not have to
    guess from a single combined boolean."""
    train_seqs = [_make_sequence(n=8, x0=float(i), seed=i) for i in range(6)]
    val_seqs = [_make_sequence(n=6, x0=float(i) + 0.5, seed=100 + i) for i in range(2)]

    def optimizer_only_collapse(net, opt, grad_clip):
        return SafeStepOutcome(
            applied=True, grad_norm_pre_clip=1.0,
            grad_nonfinite_pre_clip=False, grad_nonfinite_post_clip=False,
            params_nonfinite_after_step=False, optimizer_state_nonfinite_after_step=True,
        )

    monkeypatch.setattr(batched_trainer, "safe_clip_and_step", optimizer_only_collapse)
    result = batched_trainer.train_one_run_batched(
        train_seqs, val_seqs, seed=0, device="cpu", batch_size=2, use_length_bucketing=False,
        lr=0.01, max_epochs=10, patience=10, hidden_size=4,
    )
    assert result.training_collapsed is True
    assert result.optimizer_state_collapse_count == 1
    assert result.parameter_collapse_count == 0


# ---------------------------------------------------------------------------
# trainer_core.train_one_run (historical per-sequence trainer -- identical fix)
# ---------------------------------------------------------------------------

def test_historical_trainer_repeated_skips_marks_unstable_and_aborts(monkeypatch):
    train_seqs = [_make_sequence(n=8, x0=float(i), seed=i) for i in range(20)]
    val_seqs = [_make_sequence(n=6, x0=float(i) + 0.5, seed=100 + i) for i in range(2)]

    def always_skip(net, opt, grad_clip):
        return _skip_outcome()

    monkeypatch.setattr(trainer_core, "safe_clip_and_step", always_skip)
    result = trainer_core.train_one_run(
        train_seqs, val_seqs, seed=0, device="cpu", lr=0.01, max_epochs=10, patience=10, hidden_size=4,
    )
    assert result.training_unstable is True
    assert result.n_epochs_run == 1


def test_historical_trainer_collapse_forces_catastrophic_true(monkeypatch):
    train_seqs = [_make_sequence(n=8, x0=float(i), seed=i) for i in range(6)]
    val_seqs = [_make_sequence(n=6, x0=float(i) + 0.5, seed=100 + i) for i in range(2)]

    def always_collapse(net, opt, grad_clip):
        return _collapsed_outcome()

    monkeypatch.setattr(trainer_core, "safe_clip_and_step", always_collapse)
    result = trainer_core.train_one_run(
        train_seqs, val_seqs, seed=0, device="cpu", lr=0.01, max_epochs=10, patience=10, hidden_size=4,
    )
    assert result.training_collapsed is True
    assert result.catastrophic is True
    assert result.n_epochs_run == 1
