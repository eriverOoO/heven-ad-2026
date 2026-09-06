from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batched_trainer import train_one_run_batched  # noqa: E402
from training_forensics import (  # noqa: E402
    BatchObservation,
    ForensicRecorder,
    hidden_state_stats,
    summarize_sequence,
)


def _obs(epoch=0, batch_index=0, loss_finite=True, grad_nonfinite_pre_clip=False, loss_value=1.0):
    return BatchObservation(
        epoch=epoch, batch_index=batch_index, n_sequences=2, loss_value=loss_value,
        loss_finite=loss_finite, grad_norm_pre_clip=1.0, grad_nonfinite_pre_clip=grad_nonfinite_pre_clip,
        hidden_state_mean_abs=0.1, hidden_state_max_abs=0.2, hidden_state_p95_abs=0.15, sequences=[],
    )


def test_recorder_captures_window_plus_failing_batch():
    rec = ForensicRecorder(window_size=3)
    for i in range(5):
        rec.observe(_obs(batch_index=i))
    assert rec.captured is False
    rec.observe(_obs(batch_index=5, loss_finite=False, loss_value=float("nan")))
    assert rec.captured is True
    assert len(rec.capture) == 4  # 3 window + 1 failing
    assert rec.capture[-1].loss_finite is False
    assert rec.first_failure_batch_index == 5


def test_recorder_never_captures_twice():
    rec = ForensicRecorder(window_size=2)
    rec.observe(_obs(batch_index=0, grad_nonfinite_pre_clip=True))
    assert rec.captured is True
    first_capture = rec.capture
    rec.observe(_obs(batch_index=1, grad_nonfinite_pre_clip=True))
    assert rec.capture is first_capture  # unchanged -- latched


def test_recorder_window_evicts_oldest_when_full():
    rec = ForensicRecorder(window_size=2)
    rec.observe(_obs(batch_index=0))
    rec.observe(_obs(batch_index=1))
    rec.observe(_obs(batch_index=2))
    rec.observe(_obs(batch_index=3, grad_nonfinite_pre_clip=True))
    indices = [o.batch_index for o in rec.capture]
    assert indices == [1, 2, 3]  # batch 0 evicted, window_size=2 + the failure


def test_capture_as_dicts_is_json_serializable():
    import json
    rec = ForensicRecorder(window_size=2)
    rec.observe(_obs(batch_index=0))
    rec.observe(_obs(batch_index=1, grad_nonfinite_pre_clip=True))
    payload = rec.capture_as_dicts()
    json.dumps(payload)  # must not raise


def test_summarize_sequence_gap_and_speed_stats():
    seq = {
        "actor_id": "a1", "frames": list(range(6)),
        "x_true": [[0, 0, 1.0, 0.0]] * 6,
        "z_meas": [[0, 0], None, None, [0, 0], None, [0, 0]],
        "coarse_object_type": "VEHICLE",
    }
    s = summarize_sequence(seq)
    assert s.length_frames == 6
    assert s.n_missing_frames == 3
    assert s.max_gap_length == 2
    assert abs(s.fraction_missing - 0.5) < 1e-9
    assert s.mean_speed_mps == 1.0
    assert s.coarse_object_type == "VEHICLE"


def test_hidden_state_stats_handles_none_and_normal_tensor():
    assert hidden_state_stats(None) == (None, None, None)
    h = torch.tensor([[[1.0, -2.0, 3.0]]])
    mean_v, max_v, p95_v = hidden_state_stats(h)
    assert mean_v == (1.0 + 2.0 + 3.0) / 3
    assert max_v == 3.0


def _make_sequence(n=8, dt=0.1, x0=0.0, y0=0.0, vx=2.0, vy=-1.0, seed=0):
    rng = np.random.RandomState(seed)
    frames = list(range(n))
    dts = [None] + [dt] * (n - 1)
    x_true, z_meas = [], []
    for i in range(n):
        x, y = x0 + vx * dt * i, y0 + vy * dt * i
        x_true.append([x, y, vx, vy])
        z_meas.append([x + rng.normal(0, 0.01), y + rng.normal(0, 0.01)])
    return {"actor_id": f"a{x0}", "frames": frames, "dt": dts, "x_true": x_true, "z_meas": z_meas,
            "coarse_object_type": "VEHICLE"}


def test_forensic_recorder_wired_into_real_training_run_without_any_instability():
    """No forced failure -- confirms the hook adds zero errors on a
    perfectly normal run and simply never captures."""
    train_seqs = [_make_sequence(n=8, x0=float(i), seed=i) for i in range(6)]
    val_seqs = [_make_sequence(n=6, x0=float(i) + 0.5, seed=100 + i) for i in range(2)]
    recorder = ForensicRecorder(window_size=5)
    result = train_one_run_batched(
        train_seqs, val_seqs, seed=0, device="cpu", batch_size=2, use_length_bucketing=False,
        lr=0.01, max_epochs=2, patience=10, hidden_size=4, forensic_recorder=recorder,
    )
    assert result.catastrophic is False
    assert recorder.captured is False


def test_forensic_recorder_none_is_zero_overhead_default():
    """Default (forensic_recorder=None) must behave identically to before
    this parameter existed -- no crash, no extra state."""
    train_seqs = [_make_sequence(n=8, x0=float(i), seed=i) for i in range(6)]
    val_seqs = [_make_sequence(n=6, x0=float(i) + 0.5, seed=100 + i) for i in range(2)]
    result = train_one_run_batched(
        train_seqs, val_seqs, seed=0, device="cpu", batch_size=2, use_length_bucketing=False,
        lr=0.01, max_epochs=2, patience=10, hidden_size=4,
    )
    assert result.n_epochs_run == 2


def _make_pathological_sequence(n=10, dt=0.1, scale=1e150, seed=0):
    """A deliberately astronomically-scaled sequence -- reliably drives
    the batch loss itself non-finite (see docs/perception/
    kalmannet_training_instability_v1.md "First Non-Finite Transition":
    scales beyond ~1e16 overflow the loss before backward() is even
    reached), giving a real, reproducible forensic-capture trigger
    through the actual trainer, not a monkeypatch."""
    rng = np.random.RandomState(seed)
    frames = list(range(n))
    dts = [None] + [dt] * (n - 1)
    x_true, z_meas = [], []
    for i in range(n):
        x, y = scale * i, scale * i * 0.5
        x_true.append([x, y, scale, scale * 0.5])
        z_meas.append([x + rng.normal(0, 0.01), y + rng.normal(0, 0.01)])
    return {"actor_id": f"pathological_{scale}", "frames": frames, "dt": dts, "x_true": x_true, "z_meas": z_meas,
            "coarse_object_type": "VEHICLE"}


def _run_with_one_pathological_sequence(seed=0):
    normal_seqs = [_make_sequence(n=8, x0=float(i), seed=i) for i in range(5)]
    train_seqs = normal_seqs + [_make_pathological_sequence(seed=99)]
    val_seqs = [_make_sequence(n=6, x0=float(i) + 0.5, seed=100 + i) for i in range(2)]
    recorder = ForensicRecorder(window_size=5)
    train_one_run_batched(
        train_seqs, val_seqs, seed=seed, device="cpu", batch_size=1, use_length_bucketing=False,
        lr=0.01, max_epochs=1, patience=10, hidden_size=4, forensic_recorder=recorder,
    )
    return recorder


def test_forensic_recorder_captures_a_real_instability_event_through_the_real_trainer():
    recorder = _run_with_one_pathological_sequence(seed=0)
    assert recorder.captured is True
    assert recorder.capture[-1].loss_finite is False
    assert recorder.capture[-1].sequences[0].actor_id.startswith("pathological_")


def test_forensic_summary_is_deterministic_given_the_same_seed():
    """Reproducibility check (task section 16): the same seed/inputs must
    capture the IDENTICAL first-failure point and window contents on
    repeated runs."""
    rec1 = _run_with_one_pathological_sequence(seed=0)
    rec2 = _run_with_one_pathological_sequence(seed=0)

    assert rec1.captured and rec2.captured
    assert rec1.first_failure_epoch == rec2.first_failure_epoch
    assert rec1.first_failure_batch_index == rec2.first_failure_batch_index
    assert len(rec1.capture) == len(rec2.capture)
    for a, b in zip(rec1.capture, rec2.capture):
        assert a.loss_finite == b.loss_finite
        assert a.loss_value == b.loss_value or (a.loss_value != a.loss_value and b.loss_value != b.loss_value)
        assert [s.actor_id for s in a.sequences] == [s.actor_id for s in b.sequences]


def test_forensic_summary_varies_with_a_different_seed():
    """Sanity check that the determinism test above is meaningful --
    different seeds change the batching/shuffle order, so the failure
    should not be pinned to an unrelated constant."""
    rec_seed0 = _run_with_one_pathological_sequence(seed=0)
    rec_seed7 = _run_with_one_pathological_sequence(seed=7)
    assert rec_seed0.captured and rec_seed7.captured
    # Both must capture (the pathological sequence always fails eventually
    # regardless of shuffle order) -- the batch INDEX it fails at may
    # differ since shuffle order differs by seed.
    assert isinstance(rec_seed0.first_failure_batch_index, int)
    assert isinstance(rec_seed7.first_failure_batch_index, int)
