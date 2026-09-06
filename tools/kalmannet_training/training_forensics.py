"""Bounded forensic instrumentation for the first training-instability
event, per this task's own explicit instruction: "Capture a bounded
forensic record only around the first instability event... Do not dump
every tensor for the whole run."

``ForensicRecorder`` keeps a small rolling window of recent, ordinary
batch summaries (sequence lengths, gap/missing-frame structure, hidden-
state magnitude, loss/gradient finiteness) and, the FIRST time a
non-finite loss or gradient is observed, freezes that window plus the
triggering batch into ``.capture`` -- a bounded, JSON-serializable record
comparing the offending batch against the immediately preceding ordinary
ones. It never captures a second time (``.captured`` latches True), so a
long, otherwise-unstable run does not spam an unbounded forensic log.

No estimator/architecture change -- this module only READS values already
computed by the existing training loop / ``BatchedKalmanNetFilter``
(``net.h`` after a batch's forward pass, the batch's own sequence dicts)
and never mutates any of them.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class BatchSequenceSummary:
    """Plain-data summary of one sequence's role in a batch -- computed
    directly from the sequence dict already built by ``kalmannet_sequences.py``,
    no new data collection."""

    actor_id: str | None
    length_frames: int
    n_missing_frames: int
    max_gap_length: int
    fraction_missing: float
    coarse_object_type: str | None
    mean_speed_mps: float | None   # from x_true's own [vx, vy], GT-derived, never the estimate


@dataclass
class BatchObservation:
    epoch: int
    batch_index: int
    n_sequences: int
    loss_value: float
    loss_finite: bool
    grad_norm_pre_clip: float
    grad_nonfinite_pre_clip: bool
    hidden_state_mean_abs: float | None
    hidden_state_max_abs: float | None
    hidden_state_p95_abs: float | None
    sequences: list[BatchSequenceSummary] = field(default_factory=list)


def _gap_stats(z_meas: list) -> tuple[int, int, float]:
    """(n_missing, max_gap_length, fraction_missing) for one sequence's
    measurement stream."""
    n = len(z_meas)
    if n == 0:
        return 0, 0, 0.0
    n_missing = sum(1 for z in z_meas if z is None)
    max_gap = 0
    cur = 0
    for z in z_meas:
        if z is None:
            cur += 1
            max_gap = max(max_gap, cur)
        else:
            cur = 0
    return n_missing, max_gap, n_missing / n


def summarize_sequence(seq: dict[str, Any]) -> BatchSequenceSummary:
    n_missing, max_gap, frac_missing = _gap_stats(seq["z_meas"])
    x_true = np.asarray(seq["x_true"], dtype=np.float64)
    mean_speed = float(np.mean(np.hypot(x_true[:, 2], x_true[:, 3]))) if len(x_true) else None
    return BatchSequenceSummary(
        actor_id=seq.get("actor_id"), length_frames=len(seq["frames"]),
        n_missing_frames=n_missing, max_gap_length=max_gap, fraction_missing=frac_missing,
        coarse_object_type=seq.get("coarse_object_type"), mean_speed_mps=mean_speed,
    )


def hidden_state_stats(h: torch.Tensor | None) -> tuple[float | None, float | None, float | None]:
    if h is None:
        return None, None, None
    abs_h = h.detach().abs()
    if not torch.isfinite(abs_h).any():
        return float("nan"), float("nan"), float("nan")
    finite_vals = abs_h[torch.isfinite(abs_h)]
    if finite_vals.numel() == 0:
        return float("nan"), float("nan"), float("nan")
    mean_v = float(finite_vals.mean().item())
    max_v = float(abs_h.max().item()) if torch.isfinite(abs_h).all() else float("inf")
    p95_v = float(torch.quantile(finite_vals, 0.95).item())
    return mean_v, max_v, p95_v


class ForensicRecorder:
    """``window_size`` ordinary batches are retained before any failure;
    on the FIRST non-finite loss/gradient, ``.capture`` is set to the
    retained window PLUS the triggering batch (so a caller can compare
    the failing batch directly against the immediately preceding healthy
    ones) and no further capture is ever taken."""

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self._window: deque[BatchObservation] = deque(maxlen=window_size)
        self.captured: bool = False
        self.capture: list[BatchObservation] = []
        self.first_failure_epoch: int | None = None
        self.first_failure_batch_index: int | None = None

    def observe(self, observation: BatchObservation) -> None:
        if self.captured:
            return
        is_failure = observation.loss_finite is False or observation.grad_nonfinite_pre_clip
        if is_failure:
            self.capture = list(self._window) + [observation]
            self.captured = True
            self.first_failure_epoch = observation.epoch
            self.first_failure_batch_index = observation.batch_index
        else:
            self._window.append(observation)

    def capture_as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(obs) for obs in self.capture]
