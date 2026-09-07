"""Bounded (<=20 epoch) forensic reproduction of the real AV2 Scale-Up
v2 10k GENERIC-ROBUST seed-0 run's gradient explosion, using the frozen
dataset/split/config and the PR #56/PR #58 safe-step guard.

**Analysis run, not model training.** Stops the moment a genuine
(element-level, STATE C) non-finite gradient is captured, or after
``max_epochs`` completes with none observed -- never trains to
completion, never starts a new seed, never touches model architecture/
hyperparameters.

Maintains a lightweight ring buffer of COMPACT per-batch summaries (never
full tensors) for the ``ring_buffer_size`` batches preceding a failure,
performs a detailed one-shot capture (per-sequence statistics, first
non-finite parameter, individual-sequence replay) only for the actual
failing batch, AND -- new in this task -- separately captures every
STATE B (aggregate gradient-norm overflow, every element still finite)
event with its own compact forensic snapshot, plus per-epoch Adam-state
and parameter-magnitude trajectories, per
``docs/perception/kalmannet_gradient_norm_overflow_v1.md``.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

from batched_kalmannet import (
    bucketed_epoch_batch_order,
    build_padded_batch,
    run_batch,
)
from nonfinite_guard import SafeStepOutcome, safe_clip_and_step
from trainer_core import set_all_seeds
from training_forensics import _gap_stats

_AD_LIDAR_PKG_DIR = None
try:
    from ad_lidar_perception.kalmannet_core import KalmanNetGRU
except ImportError:  # pragma: no cover - path wiring handled by callers/tests
    import sys
    from pathlib import Path
    _AD_LIDAR_PKG_DIR = Path(__file__).resolve().parents[2] / "ad_lidar_perception"
    sys.path.insert(0, str(_AD_LIDAR_PKG_DIR))
    from ad_lidar_perception.kalmannet_core import KalmanNetGRU  # noqa: E402


# ---------------------------------------------------------------------------
# Compact dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RingBufferEntry:
    epoch: int
    batch_index: int
    sequence_ids: list[str]
    scenario_ids: list[str | None]
    sequence_lengths: list[int]
    total_valid_timesteps: int
    measurement_valid_timesteps: int
    missing_fraction: float
    max_missing_gap: int
    batch_loss: float
    per_seq_loss_min: float
    per_seq_loss_median: float
    per_seq_loss_p90: float
    per_seq_loss_max: float
    grad_norm_pre_clip: float
    grad_nonfinite_pre_clip: bool
    max_target_speed: float
    max_target_accel_proxy: float
    max_innovation_abs: float
    max_prior_state_abs: float
    max_posterior_state_abs: float
    max_hidden_state_abs: float
    max_gain_abs: float
    # New in this task (see docs/perception/kalmannet_gradient_norm_overflow_v1.md):
    # every entry now also records which of the three nonfinite_guard.py
    # gradient states this batch was in, plus the overflow-resistant
    # float64 diagnostic norm and the single largest individual gradient
    # element magnitude across all parameters.
    gradient_state: str = "healthy"      # "healthy" | "norm_overflow" | "element_nonfinite" | "loss_nonfinite_no_gradient"
    grad_norm_robust: float = math.nan
    max_individual_abs_gradient: float = math.nan


@dataclass
class SequenceDetail:
    actor_id: str | None
    scenario_id: str | None
    track_id: str | None
    av2_object_type: str | None
    coarse_object_type: str | None
    length_frames: int
    per_sequence_loss: float
    n_valid: int
    n_missing: int
    longest_gap: int
    speed_p50: float
    speed_p90: float
    speed_max: float
    accel_p90: float
    accel_max: float
    single_sequence_loss_finite: bool | None = None
    single_sequence_grad_finite: bool | None = None


@dataclass
class ParamGradReport:
    name: str
    n_nonfinite: int
    n_total: int
    finite_max_abs: float | None
    grad_norm: float


@dataclass
class EpochTrajectoryPoint:
    epoch: int
    train_loss_mean: float
    grad_norm_mean: float
    grad_norm_max: float
    n_batches: int
    n_nonfinite_loss_batches: int
    # New: per-epoch counts broken out by gradient state (see
    # nonfinite_guard.py's three-state classification) -- so a norm-
    # overflow-heavy epoch is never indistinguishable from a genuinely
    # healthy one just by looking at `n_nonfinite_loss_batches` (which
    # only ever counts the loss-itself-nonfinite case, a fourth,
    # even-earlier failure point).
    n_norm_overflow_batches: int = 0
    n_element_nonfinite_batches: int = 0


@dataclass
class NormOverflowEvent:
    """One STATE B occurrence: every gradient element finite, but the
    aggregate (ordinary, float32-native) L2 norm reduction overflowed to
    ``inf``. Captured for EVERY occurrence (not just the first), since
    section 10 of this task asks whether the recurring epoch 6-11
    instability episodes share a common data pattern -- answering that
    needs the full population, not one example."""

    epoch: int
    batch_index: int
    grad_norm_pre_clip: float          # ordinary norm -- inf by definition of this event
    grad_norm_robust: float            # float64 diagnostic norm; the true large-but-finite magnitude
    robust_norm_also_overflowed: bool  # honest flag: even float64 overflowed (should be exceedingly rare)
    max_individual_abs_gradient: float
    largest_norm_parameter_name: str | None
    batch_loss: float
    missing_fraction: float
    max_missing_gap: int
    max_target_speed: float
    max_target_accel_proxy: float
    max_hidden_state_abs: float
    max_gain_abs: float
    top_loss_sequences: list[SequenceDetail] = field(default_factory=list)


@dataclass
class AdamStateSnapshot:
    """Compact per-epoch Adam moment-buffer summary (never full tensors) --
    section 12: "we need to know whether optimizer moments grow
    progressively before the true per-element failure.\""""

    epoch: int
    max_abs_exp_avg: float
    max_abs_exp_avg_sq: float
    p99_abs_exp_avg: float
    p99_abs_exp_avg_sq: float


@dataclass
class ParameterMagnitudeSnapshot:
    """Compact per-epoch parameter-magnitude summary -- section 13:
    "determine whether weights themselves grow before the instability
    episodes.\""""

    epoch: int
    max_abs_parameter: float
    gru_weight_norm: float
    output_head_weight_norm: float


@dataclass
class ForensicResult:
    reproduced: bool
    stop_reason: str
    first_event_epoch: int | None
    first_event_batch_index: int | None
    ring_buffer: list[RingBufferEntry]
    failure_entry: RingBufferEntry | None
    failure_sequence_details: list[SequenceDetail]
    param_grad_reports: list[ParamGradReport]
    first_nonfinite_param_name: str | None
    per_epoch_trajectory: list[EpochTrajectoryPoint]
    # New in this task:
    norm_overflow_events: list[NormOverflowEvent] = field(default_factory=list)
    adam_state_trajectory: list[AdamStateSnapshot] = field(default_factory=list)
    parameter_magnitude_trajectory: list[ParameterMagnitudeSnapshot] = field(default_factory=list)
    # Section 11: was the true STATE-C event isolated, or the endpoint of
    # accumulated STATE-B instability? None when not reproduced.
    preceding_norm_overflow_count_in_ring_buffer: int | None = None
    cumulative_norm_overflow_count_before_failure: int | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "reproduced": self.reproduced,
            "stop_reason": self.stop_reason,
            "first_event_epoch": self.first_event_epoch,
            "first_event_batch_index": self.first_event_batch_index,
            "ring_buffer": [asdict(e) for e in self.ring_buffer],
            "failure_entry": asdict(self.failure_entry) if self.failure_entry else None,
            "failure_sequence_details": [asdict(d) for d in self.failure_sequence_details],
            "param_grad_reports": [asdict(p) for p in self.param_grad_reports],
            "first_nonfinite_param_name": self.first_nonfinite_param_name,
            "per_epoch_trajectory": [asdict(p) for p in self.per_epoch_trajectory],
            "norm_overflow_events": [asdict(e) for e in self.norm_overflow_events],
            "adam_state_trajectory": [asdict(a) for a in self.adam_state_trajectory],
            "parameter_magnitude_trajectory": [asdict(p) for p in self.parameter_magnitude_trajectory],
            "preceding_norm_overflow_count_in_ring_buffer": self.preceding_norm_overflow_count_in_ring_buffer,
            "cumulative_norm_overflow_count_before_failure": self.cumulative_norm_overflow_count_before_failure,
        }


# ---------------------------------------------------------------------------
# Per-sequence GT-derived motion stats
# ---------------------------------------------------------------------------


def _speed_and_accel_stats(seq: dict[str, Any]) -> tuple[float, float, float, float, float]:
    """(speed_p50, speed_p90, speed_max, accel_p90, accel_max) from GT
    [vx, vy] and the sequence's own dt -- never from an estimate."""
    x_true = np.asarray(seq["x_true"], dtype=np.float64)
    speed = np.hypot(x_true[:, 2], x_true[:, 3])
    speed_p50, speed_p90, speed_max = (
        float(np.percentile(speed, 50)), float(np.percentile(speed, 90)), float(speed.max())
    )
    accel = np.zeros(0)
    if len(speed) > 1:
        dvx = np.diff(x_true[:, 2])
        dvy = np.diff(x_true[:, 3])
        dt = np.array([d if d is not None else np.nan for d in seq["dt"][1:]], dtype=np.float64)
        dt = np.where((dt is None) | (dt <= 0) | np.isnan(dt), np.nan, dt)
        accel = np.hypot(dvx, dvy) / dt
        accel = accel[np.isfinite(accel)]
    accel_p90 = float(np.percentile(accel, 90)) if accel.size else 0.0
    accel_max = float(accel.max()) if accel.size else 0.0
    return speed_p50, speed_p90, speed_max, accel_p90, accel_max


def _per_sequence_loss(seq: dict[str, Any], net: KalmanNetGRU, device: str, loss_on_predict_only: bool) -> float:
    padded = build_padded_batch([seq], device=device)
    with torch.no_grad():
        result = run_batch(net, padded, device, loss_on_predict_only)
    return float(result.loss.item())


# ---------------------------------------------------------------------------
# Parameter-level non-finite localization (STATE C)
# ---------------------------------------------------------------------------


def localize_first_nonfinite_parameter(net: torch.nn.Module) -> tuple[list[ParamGradReport], str | None]:
    """Iterates ``net.named_parameters()`` in definition order (the GRU's
    own weight/bias groups first, then the gain output head -- see
    ``KalmanNetGRU.__init__``), reporting per-parameter gradient
    finiteness. The FIRST name with any non-finite element is returned
    separately for convenience; every parameter's report is still
    included so the caller can see the full picture (e.g. whether only
    one group or several are affected)."""
    reports: list[ParamGradReport] = []
    first_name: str | None = None
    for name, param in net.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        finite_mask = torch.isfinite(grad)
        n_nonfinite = int((~finite_mask).sum().item())
        n_total = int(grad.numel())
        finite_vals = grad[finite_mask]
        finite_max_abs = float(finite_vals.abs().max().item()) if finite_vals.numel() else None
        grad_norm = float(grad.norm(2).item())
        reports.append(ParamGradReport(
            name=name, n_nonfinite=n_nonfinite, n_total=n_total,
            finite_max_abs=finite_max_abs, grad_norm=grad_norm,
        ))
        if n_nonfinite > 0 and first_name is None:
            first_name = name
    return reports, first_name


# ---------------------------------------------------------------------------
# Parameter-level magnitude ranking (STATE B -- every element finite)
# ---------------------------------------------------------------------------


def largest_gradient_parameter(net: torch.nn.Module) -> tuple[str | None, float, float]:
    """Returns ``(name_of_largest_robust_norm_param,
    max_individual_abs_gradient_overall, largest_param_robust_norm)``.

    Uses the SAME float64-upcast-before-square technique as
    ``nonfinite_guard._grad_norm_robust_float64`` -- computed
    PER-PARAMETER this time, so ranking parameters by gradient magnitude
    on a STATE-B batch doesn't ALSO silently overflow in float32. Only
    meaningful when every gradient element is already known finite
    (callers must check this first)."""
    best_name: str | None = None
    best_norm = -1.0
    max_abs_overall = 0.0
    for name, p in net.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        max_abs_overall = max(max_abs_overall, float(g.abs().max().item()))
        robust_norm = float(torch.sqrt(torch.sum(g.to(torch.float64) ** 2)).item())
        if robust_norm > best_norm:
            best_norm = robust_norm
            best_name = name
    return best_name, max_abs_overall, best_norm


def max_individual_abs_gradient(net: torch.nn.Module) -> float:
    """Single largest |gradient element| across every parameter -- cheap,
    used to populate every ring-buffer entry (not just STATE B ones)."""
    max_abs = 0.0
    for p in net.parameters():
        if p.grad is not None:
            max_abs = max(max_abs, float(p.grad.detach().abs().max().item()))
    return max_abs


# ---------------------------------------------------------------------------
# Adam-state / parameter-magnitude per-epoch snapshots (sections 12/13)
# ---------------------------------------------------------------------------


def adam_state_snapshot(epoch: int, opt: torch.optim.Optimizer) -> AdamStateSnapshot:
    """Compact per-epoch summary of Adam's moment buffers -- max and p99
    absolute value of ``exp_avg``/``exp_avg_sq`` across every tracked
    parameter, never the full tensors. Non-finite entries (should be
    unreachable given the guard) are excluded from both stats rather than
    silently propagating a NaN into a percentile computation; an entirely
    non-finite buffer reports 0.0 for both (a fresh/uninitialized
    optimizer -- ``opt.state`` empty -- reports the same)."""
    all_ea: list[torch.Tensor] = []
    all_eas: list[torch.Tensor] = []
    for state in opt.state.values():
        ea = state.get("exp_avg")
        eas = state.get("exp_avg_sq")
        if ea is not None:
            flat = ea.detach().flatten().abs().double()
            all_ea.append(flat[torch.isfinite(flat)])
        if eas is not None:
            flat = eas.detach().flatten().abs().double()
            all_eas.append(flat[torch.isfinite(flat)])

    def _max_and_p99(chunks: list[torch.Tensor]) -> tuple[float, float]:
        nonempty = [c for c in chunks if c.numel() > 0]
        if not nonempty:
            return 0.0, 0.0
        cat = torch.cat(nonempty)
        return float(cat.max().item()), float(torch.quantile(cat, 0.99).item())

    max_ea, p99_ea = _max_and_p99(all_ea)
    max_eas, p99_eas = _max_and_p99(all_eas)
    return AdamStateSnapshot(
        epoch=epoch, max_abs_exp_avg=max_ea, max_abs_exp_avg_sq=max_eas,
        p99_abs_exp_avg=p99_ea, p99_abs_exp_avg_sq=p99_eas,
    )


def parameter_magnitude_snapshot(epoch: int, net: torch.nn.Module) -> ParameterMagnitudeSnapshot:
    """Compact per-epoch summary of the model's own parameter magnitudes
    -- max |parameter| overall, plus the combined (float64-robust) norm
    of the GRU's own parameters (``gru.*``) and the gain output head's
    parameters (``output_fc.*``) separately, per ``KalmanNetGRU``'s own
    submodule names (``input_fc``/``gru``/``output_fc``)."""
    max_abs = 0.0
    gru_sq = 0.0
    out_sq = 0.0
    for name, p in net.named_parameters():
        v = p.detach()
        max_abs = max(max_abs, float(v.abs().max().item()))
        sq = float(torch.sum(v.to(torch.float64) ** 2).item())
        if name.startswith("gru."):
            gru_sq += sq
        elif name.startswith("output_fc."):
            out_sq += sq
    return ParameterMagnitudeSnapshot(
        epoch=epoch, max_abs_parameter=max_abs,
        gru_weight_norm=math.sqrt(gru_sq), output_head_weight_norm=math.sqrt(out_sq),
    )


# ---------------------------------------------------------------------------
# Ring-buffer / norm-overflow-event entry construction
# ---------------------------------------------------------------------------


def build_ring_buffer_entry(
    epoch: int, batch_index: int, batch_seqs: list[dict[str, Any]],
    batch_loss: float, outcome: SafeStepOutcome | None, net: torch.nn.Module | None,
    diagnostics: dict[str, Any],
    max_individual_abs_gradient_override: float | None = None,
) -> RingBufferEntry:
    """``outcome=None`` represents the "loss itself was already
    non-finite, backward() was never called" case (no gradient exists at
    all) -- distinct from all three ``nonfinite_guard`` states, which all
    presuppose a real gradient was computed.

    ``max_individual_abs_gradient_override``, when given, is used
    verbatim instead of re-inspecting ``net``'s current gradients --
    REQUIRED for a correct value on any STATE-B/STATE-C batch, since
    ``safe_clip_and_step`` already zeroes gradients before returning
    (callers must snapshot this immediately after ``backward()``, before
    calling ``safe_clip_and_step``; see ``run_bounded_forensic_search``).
    Falls back to a live (net) inspection only when no override is given
    (e.g. direct unit-test calls against a still-populated ``net``)."""
    total_valid = 0
    total_meas_valid = 0
    max_gap = 0
    total_frames = 0
    max_speed = 0.0
    max_accel = 0.0
    for seq in batch_seqs:
        n = len(seq["frames"])
        n_missing, gap, _ = _gap_stats(seq["z_meas"])
        total_valid += n
        total_meas_valid += (n - n_missing)
        max_gap = max(max_gap, gap)
        total_frames += n
        _, _, speed_max, _, accel_max = _speed_and_accel_stats(seq)
        max_speed = max(max_speed, speed_max)
        max_accel = max(max_accel, accel_max)

    missing_fraction = 1.0 - (total_meas_valid / total_valid) if total_valid else 0.0
    per_row_loss = diagnostics.get("per_row_loss")
    if per_row_loss is not None and len(per_row_loss) == len(batch_seqs):
        per_seq_losses_arr = np.asarray(per_row_loss, dtype=np.float64)
    else:
        per_seq_losses_arr = np.array([batch_loss])

    if outcome is None:
        gradient_state = "loss_nonfinite_no_gradient"
        grad_norm_pre_clip = float("nan")
        grad_nonfinite_pre_clip = False
        grad_norm_robust = float("nan")
        max_abs_grad = float("nan")
    else:
        gradient_state = outcome.gradient_state
        grad_norm_pre_clip = outcome.grad_norm_pre_clip
        grad_nonfinite_pre_clip = outcome.grad_nonfinite_pre_clip
        grad_norm_robust = outcome.grad_norm_robust
        if max_individual_abs_gradient_override is not None:
            max_abs_grad = max_individual_abs_gradient_override
        else:
            max_abs_grad = max_individual_abs_gradient(net) if net is not None else float("nan")

    return RingBufferEntry(
        epoch=epoch, batch_index=batch_index,
        sequence_ids=[s.get("actor_id") for s in batch_seqs],
        scenario_ids=[s.get("scenario_id") for s in batch_seqs],
        sequence_lengths=[len(s["frames"]) for s in batch_seqs],
        total_valid_timesteps=total_valid, measurement_valid_timesteps=total_meas_valid,
        missing_fraction=missing_fraction, max_missing_gap=max_gap,
        batch_loss=batch_loss,
        per_seq_loss_min=float(per_seq_losses_arr.min()), per_seq_loss_median=float(np.median(per_seq_losses_arr)),
        per_seq_loss_p90=float(np.percentile(per_seq_losses_arr, 90)), per_seq_loss_max=float(per_seq_losses_arr.max()),
        grad_norm_pre_clip=grad_norm_pre_clip, grad_nonfinite_pre_clip=grad_nonfinite_pre_clip,
        max_target_speed=max_speed, max_target_accel_proxy=max_accel,
        max_innovation_abs=diagnostics.get("max_innovation_abs", 0.0),
        max_prior_state_abs=diagnostics.get("max_prior_state_abs", 0.0),
        max_posterior_state_abs=diagnostics.get("max_posterior_state_abs", 0.0),
        max_hidden_state_abs=diagnostics.get("max_hidden_state_abs", 0.0),
        max_gain_abs=diagnostics.get("max_gain_abs", 0.0),
        gradient_state=gradient_state, grad_norm_robust=grad_norm_robust,
        max_individual_abs_gradient=max_abs_grad,
    )


def build_sequence_detail(
    seq: dict[str, Any], net: KalmanNetGRU, device: str, loss_on_predict_only: bool,
) -> SequenceDetail:
    n = len(seq["frames"])
    n_missing, longest_gap, _ = _gap_stats(seq["z_meas"])
    speed_p50, speed_p90, speed_max, accel_p90, accel_max = _speed_and_accel_stats(seq)
    per_seq_loss = _per_sequence_loss(seq, net, device, loss_on_predict_only)
    return SequenceDetail(
        actor_id=seq.get("actor_id"), scenario_id=seq.get("scenario_id"), track_id=seq.get("track_id"),
        av2_object_type=seq.get("av2_object_type"), coarse_object_type=seq.get("coarse_object_type"),
        length_frames=n, per_sequence_loss=per_seq_loss, n_valid=n, n_missing=n_missing,
        longest_gap=longest_gap, speed_p50=speed_p50, speed_p90=speed_p90, speed_max=speed_max,
        accel_p90=accel_p90, accel_max=accel_max,
    )


def build_norm_overflow_event(
    epoch: int, batch_index: int, batch_seqs: list[dict[str, Any]], batch_loss: float,
    outcome: SafeStepOutcome, net: KalmanNetGRU, device: str, loss_on_predict_only: bool,
    top_n_sequences: int = 3,
    largest_norm_parameter_name_override: str | None = None,
    max_individual_abs_gradient_override: float | None = None,
) -> NormOverflowEvent:
    """Full compact forensic snapshot for ONE STATE-B (norm overflow)
    occurrence -- section 10. Per-sequence loss requires one extra
    forward pass per sequence in the batch (no backward, no
    optimizer.step); called for every STATE-B event, not just the first,
    since section 10 asks whether the recurring episodes share a common
    data pattern across the whole population.

    ``*_override``, when given, are used verbatim instead of
    re-inspecting ``net``'s current gradients -- REQUIRED for a correct
    value, since ``safe_clip_and_step`` already zeroes gradients before
    returning (callers must snapshot this immediately after
    ``backward()``, before calling ``safe_clip_and_step``; see
    ``run_bounded_forensic_search``). Falls back to a live (net)
    inspection only when no override is given (e.g. direct unit-test
    calls against a still-populated ``net``)."""
    if largest_norm_parameter_name_override is not None or max_individual_abs_gradient_override is not None:
        largest_name = largest_norm_parameter_name_override
        max_abs_grad = max_individual_abs_gradient_override if max_individual_abs_gradient_override is not None else 0.0
    else:
        largest_name, max_abs_grad, _largest_norm = largest_gradient_parameter(net)

    total_valid = 0
    total_meas_valid = 0
    max_gap = 0
    max_speed = 0.0
    max_accel = 0.0
    for seq in batch_seqs:
        n = len(seq["frames"])
        n_missing, gap, _ = _gap_stats(seq["z_meas"])
        total_valid += n
        total_meas_valid += (n - n_missing)
        max_gap = max(max_gap, gap)
        _, _, speed_max, _, accel_max = _speed_and_accel_stats(seq)
        max_speed = max(max_speed, speed_max)
        max_accel = max(max_accel, accel_max)
    missing_fraction = 1.0 - (total_meas_valid / total_valid) if total_valid else 0.0

    details = [build_sequence_detail(seq, net, device, loss_on_predict_only) for seq in batch_seqs]
    details.sort(key=lambda d: d.per_sequence_loss, reverse=True)

    return NormOverflowEvent(
        epoch=epoch, batch_index=batch_index,
        grad_norm_pre_clip=outcome.grad_norm_pre_clip, grad_norm_robust=outcome.grad_norm_robust,
        robust_norm_also_overflowed=outcome.robust_norm_also_overflowed,
        max_individual_abs_gradient=max_abs_grad, largest_norm_parameter_name=largest_name,
        batch_loss=batch_loss, missing_fraction=missing_fraction, max_missing_gap=max_gap,
        max_target_speed=max_speed, max_target_accel_proxy=max_accel,
        max_hidden_state_abs=0.0, max_gain_abs=0.0,  # populated by caller from the batch's own diagnostics
        top_loss_sequences=details[:top_n_sequences],
    )


def replay_single_sequence(
    seq: dict[str, Any], net: KalmanNetGRU, device: str, loss_on_predict_only: bool,
) -> tuple[bool, bool]:
    """Forward + backward on ONE sequence alone, from the CURRENT model
    state (never calls optimizer.step) -- used to test whether a single
    suspected sequence alone reproduces the batch's non-finite gradient.
    Returns (loss_finite, grad_finite_pre_clip). Gradients are zeroed
    afterward so this probe never contaminates a subsequent probe or the
    caller's own state."""
    net.zero_grad()
    padded = build_padded_batch([seq], device=device)
    result = run_batch(net, padded, device, loss_on_predict_only)
    loss_finite = bool(torch.isfinite(result.loss))
    grad_finite = True
    if loss_finite and result.n_loss_terms > 0:
        result.loss.backward()
        for p in net.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                grad_finite = False
                break
    net.zero_grad()
    return loss_finite, grad_finite


# ---------------------------------------------------------------------------
# Main bounded search
# ---------------------------------------------------------------------------


def run_bounded_forensic_search(
    train_seqs: list[dict[str, Any]],
    seed: int,
    device: str,
    batch_size: int,
    use_length_bucketing: bool,
    n_buckets: int,
    lr: float,
    grad_clip: float | None,
    loss_on_predict_only: bool,
    hidden_size: int,
    max_epochs: int = 12,
    ring_buffer_size: int = 20,
    replay_top_n_sequences: int = 3,
    progress_callback=None,
    progress_every_n_batches: int = 500,
) -> ForensicResult:
    """Bounded, analysis-only reproduction. Stops the moment a genuine
    STATE-C (element-level non-finite PRE-CLIP) gradient is detected (the
    exact ``safe_clip_and_step`` signal PR #56 uses to skip a step), or
    after ``max_epochs`` complete with none observed. Never calls
    ``optimizer.step()`` on a STATE-B (norm-overflow) or STATE-C
    (element-nonfinite) gradient -- ``safe_clip_and_step`` (PR #58)
    already guarantees both are skipped -- and never continues training
    after a STATE-C event is captured. STATE-B events do NOT stop the
    search; every occurrence is captured into ``norm_overflow_events``
    and the search continues.

    ``progress_callback(epoch, batch_index, n_batches_this_epoch,
    running_train_loss_mean, running_grad_norm_max)``, optional, default
    ``None`` (zero behavior change) -- called every
    ``progress_every_n_batches`` batches and once at the end of every
    epoch, purely for long-run visibility; never affects the search
    itself."""
    set_all_seeds(seed)
    net = KalmanNetGRU(hidden_size=hidden_size).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = np.random.RandomState(seed)

    ring_buffer: deque[RingBufferEntry] = deque(maxlen=ring_buffer_size)
    per_epoch_trajectory: list[EpochTrajectoryPoint] = []
    norm_overflow_events: list[NormOverflowEvent] = []
    adam_state_trajectory: list[AdamStateSnapshot] = []
    parameter_magnitude_trajectory: list[ParameterMagnitudeSnapshot] = []
    cumulative_norm_overflow_count = 0

    failure_entry: RingBufferEntry | None = None
    failure_batch_seqs: list[dict[str, Any]] | None = None
    first_event_epoch: int | None = None
    first_event_batch_index: int | None = None
    param_grad_reports: list[ParamGradReport] = []
    first_nonfinite_param_name: str | None = None
    preceding_norm_overflow_count_in_ring_buffer: int | None = None
    cumulative_norm_overflow_count_before_failure: int | None = None

    for epoch in range(max_epochs):
        if use_length_bucketing:
            batch_index_groups = bucketed_epoch_batch_order(train_seqs, rng, batch_size, n_buckets)
        else:
            order = rng.permutation(len(train_seqs))
            batch_index_groups = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]

        epoch_losses: list[float] = []
        grad_norms: list[float] = []
        n_nonfinite_loss = 0
        n_norm_overflow = 0
        n_element_nonfinite = 0

        for batch_index, idx_group in enumerate(batch_index_groups):
            if len(idx_group) == 0:
                continue
            batch_seqs = [train_seqs[i] for i in idx_group]
            padded = build_padded_batch(batch_seqs, device=device)
            opt.zero_grad()
            diag: dict[str, float] = {}
            batch_result = run_batch(net, padded, device, loss_on_predict_only, diagnostics_out=diag)
            if batch_result.n_loss_terms == 0:
                continue

            loss_finite = bool(torch.isfinite(batch_result.loss))
            if not loss_finite:
                n_nonfinite_loss += 1
                epoch_losses.append(float("nan"))
                entry = build_ring_buffer_entry(
                    epoch, batch_index, batch_seqs, float(batch_result.loss.item()),
                    outcome=None, net=None, diagnostics=diag,
                )
                ring_buffer.append(entry)
                continue

            batch_result.loss.backward()

            # CRITICAL ORDERING: snapshot gradient diagnostics BEFORE
            # calling safe_clip_and_step -- its STATE-B and STATE-C skip
            # paths both call opt.zero_grad() before returning, so any
            # per-parameter inspection done AFTER that call sees only
            # zeros (a real bug found and fixed during this task's own
            # 20-epoch run: pre-fix, every captured norm_overflow_event's
            # `largest_norm_parameter_name`/`max_individual_abs_gradient`
            # trivially showed the first-declared parameter / 0.0, since
            # the true gradient had already been zeroed by the time it
            # was inspected). Both snapshots are cheap (this model has
            # only ~7,016 scalar parameters across ~10 tensors) and are
            # computed unconditionally so they are available regardless
            # of which state this batch turns out to be.
            pre_guard_param_reports, pre_guard_first_nonfinite_name = localize_first_nonfinite_parameter(net)
            pre_guard_largest_name, pre_guard_max_abs_grad, _pre_guard_largest_norm = largest_gradient_parameter(net)

            outcome = safe_clip_and_step(net, opt, grad_clip)
            grad_norms.append(outcome.grad_norm_pre_clip)
            epoch_losses.append(float(batch_result.loss.item()))

            entry = build_ring_buffer_entry(
                epoch, batch_index, batch_seqs, float(batch_result.loss.item()),
                outcome=outcome, net=net, diagnostics=diag,
                max_individual_abs_gradient_override=pre_guard_max_abs_grad,
            )
            ring_buffer.append(entry)

            if progress_callback is not None and (batch_index + 1) % progress_every_n_batches == 0:
                progress_callback(
                    epoch, batch_index, len(batch_index_groups),
                    float(np.nanmean(epoch_losses)) if epoch_losses else float("nan"),
                    float(np.max(grad_norms)) if grad_norms else 0.0,
                )

            if outcome.grad_nonfinite_pre_clip:
                # THE EVENT -- genuine STATE-C (element-level) non-finite
                # gradient, caught BEFORE clip_grad_norm_/optimizer.step
                # by the safe-step guard (no step was applied; parameters/
                # optimizer state remain exactly as they were before this
                # batch). Capture and stop immediately. Uses the PRE-guard
                # snapshot (see comment above) -- the guard has already
                # zeroed the real gradient by this point.
                n_element_nonfinite += 1
                failure_entry = entry
                failure_batch_seqs = batch_seqs
                first_event_epoch = epoch
                first_event_batch_index = batch_index
                param_grad_reports = pre_guard_param_reports
                first_nonfinite_param_name = pre_guard_first_nonfinite_name
                # Section 11: was this isolated, or the endpoint of
                # accumulated STATE-B instability? Look at the ring
                # buffer's own recent history (up to ring_buffer_size
                # entries preceding this one) and the running total.
                preceding_norm_overflow_count_in_ring_buffer = sum(
                    1 for e in ring_buffer if e.gradient_state == "norm_overflow"
                )
                cumulative_norm_overflow_count_before_failure = cumulative_norm_overflow_count
                break

            if outcome.norm_overflow:
                # STATE B -- capture the full forensic snapshot (section
                # 10) for EVERY occurrence, not just the first, and
                # continue (never stops the search). Uses the PRE-guard
                # largest-parameter snapshot (see comment above).
                n_norm_overflow += 1
                cumulative_norm_overflow_count += 1
                event = build_norm_overflow_event(
                    epoch, batch_index, batch_seqs, float(batch_result.loss.item()),
                    outcome, net, device, loss_on_predict_only, replay_top_n_sequences,
                    largest_norm_parameter_name_override=pre_guard_largest_name,
                    max_individual_abs_gradient_override=pre_guard_max_abs_grad,
                )
                event.max_hidden_state_abs = diag.get("max_hidden_state_abs", 0.0)
                event.max_gain_abs = diag.get("max_gain_abs", 0.0)
                norm_overflow_events.append(event)

        epoch_point = EpochTrajectoryPoint(
            epoch=epoch,
            train_loss_mean=float(np.nanmean(epoch_losses)) if epoch_losses else float("nan"),
            grad_norm_mean=float(np.mean(grad_norms)) if grad_norms else 0.0,
            grad_norm_max=float(np.max(grad_norms)) if grad_norms else 0.0,
            n_batches=len(epoch_losses), n_nonfinite_loss_batches=n_nonfinite_loss,
            n_norm_overflow_batches=n_norm_overflow, n_element_nonfinite_batches=n_element_nonfinite,
        )
        per_epoch_trajectory.append(epoch_point)
        adam_state_trajectory.append(adam_state_snapshot(epoch, opt))
        parameter_magnitude_trajectory.append(parameter_magnitude_snapshot(epoch, net))
        if progress_callback is not None:
            progress_callback(epoch, -1, len(batch_index_groups), epoch_point.train_loss_mean, epoch_point.grad_norm_max)

        if failure_entry is not None:
            break

    reproduced = failure_entry is not None
    failure_sequence_details: list[SequenceDetail] = []
    if reproduced and failure_batch_seqs is not None:
        details = [
            build_sequence_detail(seq, net, device, loss_on_predict_only)
            for seq in failure_batch_seqs
        ]
        details.sort(key=lambda d: d.per_sequence_loss, reverse=True)
        failure_sequence_details = details

        # Individual-sequence replay probes (never optimizer.step) --
        # top-N highest-loss sequences, from the SAME (unmodified,
        # pre-update) model state the real failing batch used.
        for detail in failure_sequence_details[:replay_top_n_sequences]:
            seq = next(s for s in failure_batch_seqs if s.get("actor_id") == detail.actor_id)
            loss_finite, grad_finite = replay_single_sequence(seq, net, device, loss_on_predict_only)
            detail.single_sequence_loss_finite = loss_finite
            detail.single_sequence_grad_finite = grad_finite

    stop_reason = (
        f"genuine non-finite gradient captured at epoch {first_event_epoch}, batch {first_event_batch_index}"
        if reproduced else f"no genuine non-finite gradient observed within the bounded {max_epochs}-epoch window"
    )

    return ForensicResult(
        reproduced=reproduced, stop_reason=stop_reason,
        first_event_epoch=first_event_epoch, first_event_batch_index=first_event_batch_index,
        ring_buffer=list(ring_buffer), failure_entry=failure_entry,
        failure_sequence_details=failure_sequence_details,
        param_grad_reports=param_grad_reports, first_nonfinite_param_name=first_nonfinite_param_name,
        per_epoch_trajectory=per_epoch_trajectory,
        norm_overflow_events=norm_overflow_events,
        adam_state_trajectory=adam_state_trajectory,
        parameter_magnitude_trajectory=parameter_magnitude_trajectory,
        preceding_norm_overflow_count_in_ring_buffer=preceding_norm_overflow_count_in_ring_buffer,
        cumulative_norm_overflow_count_before_failure=cumulative_norm_overflow_count_before_failure,
    )
