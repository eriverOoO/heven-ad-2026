"""Tests for the bounded 10k-explosion forensic search
(``explosion_forensics.py``): ring-buffer retention, first-event capture,
parameter-level non-finite localization, per-sequence loss attribution,
stop-after-first-event, epoch-bounded stop, and deterministic replay
metadata."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
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
    train_seqs = _train_seqs(n_seqs=20)
    real_fn = ef.safe_clip_and_step
    calls = {"n": 0}

    def flaky(net, opt, grad_clip):
        calls["n"] += 1
        if calls["n"] == 3:  # trigger on the 3rd batch, not the first
            _inject_nonfinite_gradient_on_call(net, "output_fc.weight")
            return SafeStepOutcome(applied=False, grad_norm_pre_clip=float("nan"),
                                    grad_nonfinite_pre_clip=True, grad_nonfinite_post_clip=False)
        return real_fn(net, opt, grad_clip)

    monkeypatch.setattr(ef, "safe_clip_and_step", flaky)
    result = ef.run_bounded_forensic_search(
        train_seqs, seed=0, device="cpu", batch_size=4, use_length_bucketing=False, n_buckets=4,
        lr=0.01, grad_clip=10.0, loss_on_predict_only=True, hidden_size=4, max_epochs=5, ring_buffer_size=5,
    )
    assert result.reproduced is True
    assert result.first_event_epoch == 0
    assert result.first_event_batch_index == 2  # 0-indexed: 3rd call = batch_index 2
    assert result.first_nonfinite_param_name == "output_fc.weight"
    assert len(result.per_epoch_trajectory) == 1  # stopped mid-epoch-0, never reached epoch 1
    assert len(result.failure_sequence_details) == 4  # batch_size=4
    # top-3 replay probes attempted (default replay_top_n_sequences=3)
    replayed = [d for d in result.failure_sequence_details if d.single_sequence_loss_finite is not None]
    assert len(replayed) == 3


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
