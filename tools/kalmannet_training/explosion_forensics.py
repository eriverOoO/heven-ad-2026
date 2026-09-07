"""Bounded (<=12 epoch) forensic reproduction of the real AV2 Scale-Up
v2 10k GENERIC-ROBUST seed-0 run's gradient explosion, using the frozen
dataset/split/config and the PR #56 safe-step guard.

**Analysis run, not model training.** Stops the moment a genuine
(element-level) non-finite gradient is captured, or after epoch 12
completes with none observed -- never trains to completion, never
starts a new seed, never touches model architecture/hyperparameters.

Maintains a lightweight ring buffer of COMPACT per-batch summaries (never
full tensors) for the ~20 batches preceding a failure, and performs a
detailed one-shot capture (per-sequence statistics, first non-finite
parameter, individual-sequence replay) only for the actual failing
batch.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from batched_kalmannet import (
    bucketed_epoch_batch_order,
    build_padded_batch,
    run_batch,
)
from nonfinite_guard import safe_clip_and_step
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
# Parameter-level non-finite localization
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
# Ring-buffer entry construction
# ---------------------------------------------------------------------------


def build_ring_buffer_entry(
    epoch: int, batch_index: int, batch_seqs: list[dict[str, Any]],
    batch_loss: float, grad_norm_pre_clip: float, grad_nonfinite_pre_clip: bool,
    diagnostics: dict[str, Any],
) -> RingBufferEntry:
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
    (element-level) non-finite PRE-CLIP gradient is detected (the exact
    ``safe_clip_and_step`` signal PR #56 uses to skip a step), or after
    ``max_epochs`` complete with none observed. Never calls
    ``optimizer.step()`` on a non-finite gradient (the guard already
    guarantees this) and never continues training after the event is
    captured.

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

    failure_entry: RingBufferEntry | None = None
    failure_batch_seqs: list[dict[str, Any]] | None = None
    first_event_epoch: int | None = None
    first_event_batch_index: int | None = None
    param_grad_reports: list[ParamGradReport] = []
    first_nonfinite_param_name: str | None = None

    for epoch in range(max_epochs):
        if use_length_bucketing:
            batch_index_groups = bucketed_epoch_batch_order(train_seqs, rng, batch_size, n_buckets)
        else:
            order = rng.permutation(len(train_seqs))
            batch_index_groups = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]

        epoch_losses: list[float] = []
        grad_norms: list[float] = []
        n_nonfinite_loss = 0

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
                    grad_norm_pre_clip=float("nan"), grad_nonfinite_pre_clip=False, diagnostics=diag,
                )
                ring_buffer.append(entry)
                continue

            batch_result.loss.backward()
            outcome = safe_clip_and_step(net, opt, grad_clip)
            grad_norms.append(outcome.grad_norm_pre_clip)
            epoch_losses.append(float(batch_result.loss.item()))

            entry = build_ring_buffer_entry(
                epoch, batch_index, batch_seqs, float(batch_result.loss.item()),
                grad_norm_pre_clip=outcome.grad_norm_pre_clip,
                grad_nonfinite_pre_clip=outcome.grad_nonfinite_pre_clip, diagnostics=diag,
            )
            ring_buffer.append(entry)

            if progress_callback is not None and (batch_index + 1) % progress_every_n_batches == 0:
                progress_callback(
                    epoch, batch_index, len(batch_index_groups),
                    float(np.nanmean(epoch_losses)) if epoch_losses else float("nan"),
                    float(np.max(grad_norms)) if grad_norms else 0.0,
                )

            if outcome.grad_nonfinite_pre_clip:
                # THE EVENT -- genuine (element-level) non-finite
                # gradient, caught BEFORE clip_grad_norm_/optimizer.step
                # by the PR #56 guard (no step was applied; parameters/
                # optimizer state remain exactly as they were before this
                # batch). Capture and stop immediately.
                failure_entry = entry
                failure_batch_seqs = batch_seqs
                first_event_epoch = epoch
                first_event_batch_index = batch_index
                param_grad_reports, first_nonfinite_param_name = localize_first_nonfinite_parameter(net)
                break

        epoch_point = EpochTrajectoryPoint(
            epoch=epoch,
            train_loss_mean=float(np.nanmean(epoch_losses)) if epoch_losses else float("nan"),
            grad_norm_mean=float(np.mean(grad_norms)) if grad_norms else 0.0,
            grad_norm_max=float(np.max(grad_norms)) if grad_norms else 0.0,
            n_batches=len(epoch_losses), n_nonfinite_loss_batches=n_nonfinite_loss,
        )
        per_epoch_trajectory.append(epoch_point)
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
    )
