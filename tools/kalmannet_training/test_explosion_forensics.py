"""Tests for the bounded 10k-explosion forensic search
(``explosion_forensics.py``): ring-buffer retention, first-event capture,
parameter-level non-finite localization, per-sequence loss attribution,
stop-after-first-event, epoch-bounded stop, and deterministic replay
metadata."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import explosion_forensics as ef  # noqa: E402
from nonfinite_guard import SafeStepOutcome  # noqa: E402


def _make_sequence(n=8, dt=0.1, x0=0.0, vx=2.0, vy=1.0, seed=0, actor_id=None):
    rng = np.random.RandomState(seed)
    frames = list(range(n))
    dts = [None] + [dt] * (n - 1)
    x_true, z_meas = [], []
    for i in range(n):
        x, y = x0 + vx * dt * i, x0 * 0.5 + vy * dt * i
        x_true.append([x, y, vx, vy])
        z_meas.append([x + rng.normal(0, 0.01), y + rng.normal(0, 0.01)])
    return {
        "actor_id": actor_id or f"a{x0}_{seed}", "scenario_id": "scn1", "track_id": "trk1",
        "frames": frames, "dt": dts, "x_true": x_true, "z_meas": z_meas,
        "av2_object_type": "VEHICLE", "coarse_object_type": "VEHICLE",
    }


def _train_seqs(n_seqs=20, seq_len=8):
    return [_make_sequence(n=seq_len, x0=float(i), seed=i) for i in range(n_seqs)]


# ---------------------------------------------------------------------------
# Ring-buffer retention / bounded stop, no event
# ---------------------------------------------------------------------------


def test_no_event_runs_full_bounded_epochs_and_retains_ring_buffer():
    result = ef.run_bounded_forensic_search(
        _train_seqs(), seed=0, device="cpu", batch_size=4, use_length_bucketing=True, n_buckets=4,
        lr=0.01, grad_clip=10.0, loss_on_predict_only=True, hidden_size=4, max_epochs=3, ring_buffer_size=5,
    )
    assert result.reproduced is False
    assert "no genuine non-finite gradient observed" in result.stop_reason
    assert len(result.per_epoch_trajectory) == 3  # ran the full bounded window, no early exit
    assert len(result.ring_buffer) == 5  # capped at ring_buffer_size
    assert result.failure_entry is None
    assert result.failure_sequence_details == []
    assert result.param_grad_reports == []


def test_progress_callback_fires_once_per_epoch_end_and_is_optional():
    calls = []

    def cb(epoch, batch_index, n_batches, loss_mean, grad_max):
        calls.append((epoch, batch_index))

    result = ef.run_bounded_forensic_search(
        _train_seqs(), seed=0, device="cpu", batch_size=4, use_length_bucketing=True, n_buckets=4,
        lr=0.01, grad_clip=10.0, loss_on_predict_only=True, hidden_size=4, max_epochs=2, ring_buffer_size=5,
        progress_callback=cb, progress_every_n_batches=1,
    )
    epoch_end_calls = [c for c in calls if c[1] == -1]
    assert len(epoch_end_calls) == 2  # once per completed epoch
    assert len(result.per_epoch_trajectory) == 2  # result still built normally

    # Default (progress_callback=None) must not raise and must behave identically.
    result2 = ef.run_bounded_forensic_search(
        _train_seqs(), seed=0, device="cpu", batch_size=4, use_length_bucketing=True, n_buckets=4,
        lr=0.01, grad_clip=10.0, loss_on_predict_only=True, hidden_size=4, max_epochs=2, ring_buffer_size=5,
    )
    assert result2.per_epoch_trajectory[-1].train_loss_mean == result.per_epoch_trajectory[-1].train_loss_mean


def test_ring_buffer_never_exceeds_configured_size():
    result = ef.run_bounded_forensic_search(
        _train_seqs(n_seqs=40), seed=1, device="cpu", batch_size=4, use_length_bucketing=False, n_buckets=4,
        lr=0.01, grad_clip=10.0, loss_on_predict_only=True, hidden_size=4, max_epochs=2, ring_buffer_size=3,
    )
    assert len(result.ring_buffer) <= 3


def test_ring_buffer_entry_has_real_per_sequence_loss_stats():
    result = ef.run_bounded_forensic_search(
        _train_seqs(), seed=0, device="cpu", batch_size=4, use_length_bucketing=False, n_buckets=4,
        lr=0.01, grad_clip=10.0, loss_on_predict_only=True, hidden_size=4, max_epochs=1, ring_buffer_size=5,
    )
    entry = result.ring_buffer[0]
    assert entry.per_seq_loss_min <= entry.per_seq_loss_median <= entry.per_seq_loss_max
    assert entry.per_seq_loss_min <= entry.per_seq_loss_p90 <= entry.per_seq_loss_max
    assert len(entry.sequence_ids) == 4
    assert entry.max_target_speed > 0


# ---------------------------------------------------------------------------
# Deterministic given the same seed
# ---------------------------------------------------------------------------


def test_no_event_run_is_deterministic_given_same_seed():
    r1 = ef.run_bounded_forensic_search(
        _train_seqs(), seed=0, device="cpu", batch_size=4, use_length_bucketing=True, n_buckets=4,
        lr=0.01, grad_clip=10.0, loss_on_predict_only=True, hidden_size=4, max_epochs=2, ring_buffer_size=5,
    )
    r2 = ef.run_bounded_forensic_search(
        _train_seqs(), seed=0, device="cpu", batch_size=4, use_length_bucketing=True, n_buckets=4,
        lr=0.01, grad_clip=10.0, loss_on_predict_only=True, hidden_size=4, max_epochs=2, ring_buffer_size=5,
    )
    assert [p.train_loss_mean for p in r1.per_epoch_trajectory] == [p.train_loss_mean for p in r2.per_epoch_trajectory]
    assert [e.batch_loss for e in r1.ring_buffer] == [e.batch_loss for e in r2.ring_buffer]


# ---------------------------------------------------------------------------
# First-event capture / stop-after-first-event / parameter localization
# ---------------------------------------------------------------------------


def _inject_nonfinite_gradient_on_call(net, target_param_name: str):
    for name, p in net.named_parameters():
        if name == target_param_name:
            p.grad = torch.full_like(p, float("nan"))
        elif p.grad is None:
            p.grad = torch.zeros_like(p)


def test_first_event_capture_stops_immediately_and_localizes_parameter(monkeypatch):
    """Injects the non-finite gradient at the PRE-GUARD snapshot point
    (mirroring exactly where a real backward() would have produced it),
    then lets the REAL (unmocked) safe_clip_and_step naturally detect and
    handle it -- this is also the regression test for a real bug found
    during this task's own 20-epoch run: localizing the offending
    parameter AFTER calling safe_clip_and_step reads back all-zero
    gradients, since the guard's STATE-C path zeroes them before
    returning. See the "CRITICAL ORDERING" comment in
    run_bounded_forensic_search."""
    train_seqs = _train_seqs(n_seqs=20)
    real_localize = ef.localize_first_nonfinite_parameter
    calls = {"n": 0}

    def flaky_localize(net):
        calls["n"] += 1
        if calls["n"] == 3:  # trigger on the 3rd batch, not the first
            _inject_nonfinite_gradient_on_call(net, "output_fc.weight")
        return real_localize(net)

    monkeypatch.setattr(ef, "localize_first_nonfinite_parameter", flaky_localize)
    result = ef.run_bounded_forensic_search(
        train_seqs, seed=0, device="cpu", batch_size=4, use_length_bucketing=False, n_buckets=4,
        lr=0.01, grad_clip=10.0, loss_on_predict_only=True, hidden_size=4, max_epochs=5, ring_buffer_size=5,
    )
    assert result.reproduced is True
    assert result.first_event_epoch == 0
    assert result.first_event_batch_index == 2  # 0-indexed: 3rd batch = batch_index 2
    assert result.first_nonfinite_param_name == "output_fc.weight"
    assert len(result.per_epoch_trajectory) == 1  # stopped mid-epoch-0, never reached epoch 1
    assert len(result.failure_sequence_details) == 4  # batch_size=4
    # top-3 replay probes attempted (default replay_top_n_sequences=3)
    replayed = [d for d in result.failure_sequence_details if d.single_sequence_loss_finite is not None]
    assert len(replayed) == 3


def test_state_b_diagnostics_reflect_pre_guard_gradient_not_post_zero(monkeypatch):
    """Regression for the exact real bug found during the 20-epoch run:
    a STATE-B event's `largest_norm_parameter_name`/
    `max_individual_abs_gradient` must reflect the TRUE pre-zero gradient
    magnitude, never the trivial first-parameter/0.0 that reading `net`
    AFTER safe_clip_and_step's zero_grad would produce.

    Injects at the ``largest_gradient_parameter`` call site -- the exact
    pre-guard snapshot point in ``run_bounded_forensic_search`` -- so the
    REAL (unmocked) ``safe_clip_and_step`` naturally detects the
    resulting STATE-B condition and zeroes it afterward, exactly as in
    production."""
    train_seqs = _train_seqs(n_seqs=20)
    real_largest = ef.largest_gradient_parameter
    calls = {"n": 0}

    def flaky_largest(net):
        calls["n"] += 1
        if calls["n"] == 2:
            # Simulate a real STATE-B batch: inject a huge-but-finite
            # gradient on a SPECIFIC, identifiable parameter at the exact
            # point a real backward() would have already produced it.
            for p in net.parameters():
                p.grad = torch.zeros_like(p)
            net.output_fc.weight.grad = torch.full_like(net.output_fc.weight, 1e19)
        return real_largest(net)

    monkeypatch.setattr(ef, "largest_gradient_parameter", flaky_largest)
    result = ef.run_bounded_forensic_search(
        train_seqs, seed=0, device="cpu", batch_size=4, use_length_bucketing=False, n_buckets=4,
        lr=0.01, grad_clip=10.0, loss_on_predict_only=True, hidden_size=4, max_epochs=1, ring_buffer_size=5,
    )
    assert len(result.norm_overflow_events) == 1
    event = result.norm_overflow_events[0]
    # Must NOT be the trivial post-zero result (first param / 0.0).
    assert event.largest_norm_parameter_name == "output_fc.weight"
    assert event.max_individual_abs_gradient == pytest.approx(1e19, rel=1e-6)
    assert event.max_individual_abs_gradient > 0.0


def test_parameter_localization_reports_every_parameter_group():
    net = ef.KalmanNetGRU(hidden_size=4)
    for p in net.parameters():
        p.grad = torch.zeros_like(p)
    # Poison exactly one named parameter.
    poisoned_name = None
    for name, p in net.named_parameters():
        p.grad[...] = float("nan")
        poisoned_name = name
        break
    reports, first_name = ef.localize_first_nonfinite_parameter(net)
    assert first_name == poisoned_name
    assert len(reports) == sum(1 for _ in net.parameters())
    poisoned_report = next(r for r in reports if r.name == poisoned_name)
    assert poisoned_report.n_nonfinite == poisoned_report.n_total


def test_parameter_localization_finds_nothing_when_all_finite():
    net = ef.KalmanNetGRU(hidden_size=4)
    for p in net.parameters():
        p.grad = torch.ones_like(p)
    reports, first_name = ef.localize_first_nonfinite_parameter(net)
    assert first_name is None
    assert all(r.n_nonfinite == 0 for r in reports)


# ---------------------------------------------------------------------------
# Individual-sequence replay
# ---------------------------------------------------------------------------


def test_replay_single_sequence_never_calls_optimizer_step():
    net = ef.KalmanNetGRU(hidden_size=4)
    before = {n: p.clone() for n, p in net.named_parameters()}
    seq = _make_sequence(n=8, seed=1)
    loss_finite, grad_finite = ef.replay_single_sequence(seq, net, "cpu", True)
    assert loss_finite is True
    assert grad_finite is True
    for n, p in net.named_parameters():
        assert torch.equal(p, before[n]), "replay_single_sequence must never mutate parameters"
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in net.parameters())


def test_replay_single_sequence_isolated_from_previous_probe():
    net = ef.KalmanNetGRU(hidden_size=4)
    seq_a = _make_sequence(n=8, seed=1)
    seq_b = _make_sequence(n=8, seed=2)
    ef.replay_single_sequence(seq_a, net, "cpu", True)
    # A second, independent probe must not accumulate gradients from the first.
    loss_finite, grad_finite = ef.replay_single_sequence(seq_b, net, "cpu", True)
    assert loss_finite is True
    assert grad_finite is True


# ---------------------------------------------------------------------------
# Gap/motion stats helpers
# ---------------------------------------------------------------------------


def test_speed_and_accel_stats_from_gt():
    seq = _make_sequence(n=10, vx=3.0, vy=4.0, seed=0)
    speed_p50, speed_p90, speed_max, accel_p90, accel_max = ef._speed_and_accel_stats(seq)
    assert speed_max == 5.0  # hypot(3,4) constant velocity
    assert accel_max == 0.0  # no acceleration in this synthetic trajectory


def test_run_batch_style_diagnostics_present_in_ring_buffer_entry():
    result = ef.run_bounded_forensic_search(
        _train_seqs(), seed=0, device="cpu", batch_size=4, use_length_bucketing=False, n_buckets=4,
        lr=0.01, grad_clip=10.0, loss_on_predict_only=True, hidden_size=4, max_epochs=1, ring_buffer_size=5,
    )
    entry = result.ring_buffer[0]
    assert entry.max_innovation_abs >= 0.0
    assert entry.max_prior_state_abs >= 0.0
    assert entry.max_posterior_state_abs >= 0.0
    assert entry.max_hidden_state_abs >= 0.0
    assert entry.max_gain_abs >= 0.0


# ---------------------------------------------------------------------------
# STATE B (norm overflow) event capture -- does NOT stop the search, and
# every occurrence is recorded with its own compact forensic snapshot.
# ---------------------------------------------------------------------------


def test_norm_overflow_events_are_captured_without_stopping_the_search(monkeypatch):
    train_seqs = _train_seqs(n_seqs=20)
    real_fn = ef.safe_clip_and_step
    calls = {"n": 0}

    def flaky(net, opt, grad_clip):
        calls["n"] += 1
        if calls["n"] in (2, 4):  # two norm-overflow events, not consecutive
            return SafeStepOutcome(
                applied=False, grad_norm_pre_clip=float("inf"),
                grad_nonfinite_pre_clip=False, grad_nonfinite_post_clip=False,
                gradient_state="norm_overflow", norm_overflow=True,
                grad_norm_robust=5.5e19,
            )
        return real_fn(net, opt, grad_clip)

    monkeypatch.setattr(ef, "safe_clip_and_step", flaky)
    result = ef.run_bounded_forensic_search(
        train_seqs, seed=0, device="cpu", batch_size=4, use_length_bucketing=False, n_buckets=4,
        lr=0.01, grad_clip=10.0, loss_on_predict_only=True, hidden_size=4, max_epochs=2, ring_buffer_size=20,
    )
    # STATE B never stops the search -- the full bounded window still ran.
    assert result.reproduced is False
    assert len(result.per_epoch_trajectory) == 2
    assert len(result.norm_overflow_events) == 2
    for event in result.norm_overflow_events:
        assert event.grad_norm_pre_clip == float("inf")
        assert event.grad_norm_robust == 5.5e19
        assert event.robust_norm_also_overflowed is False
        assert len(event.top_loss_sequences) <= 3
        assert event.top_loss_sequences[0].per_sequence_loss >= event.top_loss_sequences[-1].per_sequence_loss
    # Ring buffer entries for these two batches are correctly classified.
    overflow_entries = [e for e in result.ring_buffer if e.gradient_state == "norm_overflow"]
    assert len(overflow_entries) == 2
    for e in overflow_entries:
        assert e.grad_nonfinite_pre_clip is False
        assert not (e.grad_norm_pre_clip == e.grad_norm_pre_clip and e.grad_norm_pre_clip < float("inf"))  # inf, not finite
    # Per-epoch trajectory correctly attributes the overflow count.
    assert sum(p.n_norm_overflow_batches for p in result.per_epoch_trajectory) == 2
    assert sum(p.n_element_nonfinite_batches for p in result.per_epoch_trajectory) == 0


def test_true_event_after_norm_overflow_history_records_preceding_context(monkeypatch):
    """Section 11: when STATE C finally fires, the search must record how
    many STATE-B events preceded it (both within the retained ring buffer
    and cumulatively over the whole run) so a reader can tell whether the
    failure was an isolated outlier or the endpoint of accumulated
    instability."""
    train_seqs = _train_seqs(n_seqs=20)
    real_fn = ef.safe_clip_and_step
    calls = {"n": 0}

    def flaky(net, opt, grad_clip):
        calls["n"] += 1
        if calls["n"] in (1, 2, 3):  # three preceding STATE-B events
            return SafeStepOutcome(
                applied=False, grad_norm_pre_clip=float("inf"),
                grad_nonfinite_pre_clip=False, grad_nonfinite_post_clip=False,
                gradient_state="norm_overflow", norm_overflow=True,
                grad_norm_robust=5.5e19,
            )
        if calls["n"] == 4:  # then the true STATE-C event
            return SafeStepOutcome(
                applied=False, grad_norm_pre_clip=float("nan"),
                grad_nonfinite_pre_clip=True, grad_nonfinite_post_clip=False,
                gradient_state="element_nonfinite",
            )
        return real_fn(net, opt, grad_clip)

    monkeypatch.setattr(ef, "safe_clip_and_step", flaky)
    result = ef.run_bounded_forensic_search(
        train_seqs, seed=0, device="cpu", batch_size=4, use_length_bucketing=False, n_buckets=4,
        lr=0.01, grad_clip=10.0, loss_on_predict_only=True, hidden_size=4, max_epochs=5, ring_buffer_size=20,
    )
    assert result.reproduced is True
    assert len(result.norm_overflow_events) == 3
    assert result.preceding_norm_overflow_count_in_ring_buffer == 3
    assert result.cumulative_norm_overflow_count_before_failure == 3


# ---------------------------------------------------------------------------
# Adam-state / parameter-magnitude trajectories (sections 12/13)
# ---------------------------------------------------------------------------


def test_adam_state_and_parameter_magnitude_trajectories_have_one_entry_per_epoch():
    result = ef.run_bounded_forensic_search(
        _train_seqs(), seed=0, device="cpu", batch_size=4, use_length_bucketing=False, n_buckets=4,
        lr=0.01, grad_clip=10.0, loss_on_predict_only=True, hidden_size=4, max_epochs=3, ring_buffer_size=5,
    )
    assert len(result.adam_state_trajectory) == 3
    assert len(result.parameter_magnitude_trajectory) == 3
    for snap in result.adam_state_trajectory:
        assert snap.max_abs_exp_avg >= 0.0
        assert snap.max_abs_exp_avg_sq >= 0.0
        assert snap.p99_abs_exp_avg >= 0.0
        assert snap.p99_abs_exp_avg_sq >= 0.0
        assert snap.p99_abs_exp_avg <= snap.max_abs_exp_avg + 1e-9
    for snap in result.parameter_magnitude_trajectory:
        assert snap.max_abs_parameter >= 0.0
        assert snap.gru_weight_norm > 0.0   # GRU always has real params
        assert snap.output_head_weight_norm > 0.0


def test_adam_state_snapshot_is_zero_for_fresh_optimizer():
    net = ef.KalmanNetGRU(hidden_size=4)
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    snap = ef.adam_state_snapshot(epoch=0, opt=opt)
    assert snap.max_abs_exp_avg == 0.0
    assert snap.max_abs_exp_avg_sq == 0.0


def test_parameter_magnitude_snapshot_separates_gru_and_output_head():
    net = ef.KalmanNetGRU(hidden_size=4)
    with torch.no_grad():
        net.output_fc.weight.fill_(100.0)
    snap = ef.parameter_magnitude_snapshot(epoch=0, net=net)
    assert snap.output_head_weight_norm > snap.gru_weight_norm
    assert snap.max_abs_parameter >= 100.0


def test_largest_gradient_parameter_ranks_by_robust_norm_not_ordinary_norm():
    net = ef.KalmanNetGRU(hidden_size=4)
    for name, p in net.named_parameters():
        p.grad = torch.zeros_like(p)
    net.output_fc.weight.grad = torch.full_like(net.output_fc.weight, 1e19)
    name, max_abs, robust_norm = ef.largest_gradient_parameter(net)
    assert name == "output_fc.weight"
    assert max_abs == pytest.approx(1e19, rel=1e-6)
    assert robust_norm > 1e19


# ---------------------------------------------------------------------------
# Extended bound (task section 8: 12 -> 20 epochs) -- the library function
# itself is bound-agnostic; this just confirms a larger max_epochs works
# identically to the smaller values already exercised above.
# ---------------------------------------------------------------------------


def test_extended_20_epoch_bound_runs_to_completion_when_no_event_occurs():
    result = ef.run_bounded_forensic_search(
        _train_seqs(n_seqs=8), seed=0, device="cpu", batch_size=4, use_length_bucketing=False, n_buckets=4,
        lr=0.01, grad_clip=10.0, loss_on_predict_only=True, hidden_size=4, max_epochs=20, ring_buffer_size=5,
    )
    assert result.reproduced is False
    assert len(result.per_epoch_trajectory) == 20
    assert len(result.adam_state_trajectory) == 20
    assert len(result.parameter_magnitude_trajectory) == 20
    assert "20-epoch" in result.stop_reason
