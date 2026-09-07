"""Training-only batched recursion for the EXISTING, UNMODIFIED
``KalmanNetGRU`` (``ad_lidar_perception/ad_lidar_perception/kalmannet_core.py``).

Does **not** modify the runtime ``KalmanNetFilter`` API, the network
architecture, the analytical ``F_matrix``/``Q_matrix``/``H_matrix``, or
``STATE_DIM``/``MEAS_DIM``. This module reuses the network's own
``forward()`` (already batch-shaped: ``[batch, dim]`` in, ``[batch, m, n]``
gain out) and only adds a batched version of the *recursion* that drives
it -- ``KalmanNetFilter.step()``'s per-sequence Python loop, replaced here
by one loop over TIME across a batch of B independent sequences.

================================================================
Why a second recursion, not a change to ``KalmanNetFilter``
================================================================

``KalmanNetFilter.step(z, dt)`` (runtime + the single-sequence trainer
path in ``trainer_core.py``) already accepts a batch dimension in
principle (``x_post`` is ``[batch, m]``), but two things make it unusable
as-is for a batch of independent, variably-corrupted, variable-length AV2
sequences:

1. ``dt`` is a **scalar** Python float passed straight into
   ``F_matrix(dt)`` (a plain NumPy 4x4 built once per call) -- one ``dt``
   for the whole batch. Real AV2 sequences have their own independent
   ``dt`` per frame; batching them under one shared ``dt`` would be a
   silent CV-model approximation, explicitly forbidden by this task.
2. ``step()`` unconditionally calls the network for the whole batch, so it
   has no way to express "this row has a missing measurement this frame,
   do the analytical predict-only branch instead" or "this row is padding
   past its own sequence's end, do nothing" for a *subset* of the batch
   at a given time index -- exactly the situation with variable-length,
   variable-corruption AV2 sequences batched together.

Rather than complicate the runtime filter (used by AB3DMOT) with training-
only masking logic, this module owns a parallel, training-only
``BatchedKalmanNetFilter`` that:

- reuses the exact same ``KalmanNetGRU`` instance/weights (never forks the
  architecture),
- reuses ``F_matrix``/``H_matrix`` (via vectorised per-row versions,
  mathematically identical to calling the scalar versions once per row --
  see ``f_batched``/``h_batched`` and their equivalence test),
- reproduces ``KalmanNetFilter.step()``'s exact update-vs-predict-only
  branch semantics per row, per timestep, via masked subset indexing, and
- reproduces the exact frame-0 initialization semantics
  (``KalmanNetFilter.init_sequence``: ``x_post = x_post_prev =
  x_prior_prev = x0``, ``y_prev = h(x0)``) per row.

``run_sequence`` (``trainer_core.py``, the historical single-sequence
path, byte-for-byte unchanged) remains CORRECTNESS MODE and the
canonical reference; every number this module produces at ``batch_size=1``
is tested against it directly (see ``test_batched_kalmannet.py``).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

_AD_LIDAR_PKG_DIR = Path(__file__).resolve().parents[2] / "ad_lidar_perception"
if str(_AD_LIDAR_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_AD_LIDAR_PKG_DIR))
from ad_lidar_perception.kalmannet_core import (  # noqa: E402
    H_matrix,
    KalmanNetGRU,
    MEAS_DIM,
    STATE_DIM,
)

_H_NP = H_matrix  # [n, m], no dt dependence -- shared across every row/timestep


def f_batched(x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """Per-row constant-velocity prediction, mathematically identical to
    calling ``KalmanNetFilter._f(x[i], dt[i].item())`` once per row --
    each row gets its OWN ``F(dt_i)``, never a shared/averaged ``dt``.

    ``x``: ``[B, STATE_DIM]``. ``dt``: ``[B]`` (real, finite, per-row).
    """
    b = x.shape[0]
    zeros = x.new_zeros(b)
    ones = x.new_ones(b)
    row0 = torch.stack([ones, zeros, dt, zeros], dim=-1)
    row1 = torch.stack([zeros, ones, zeros, dt], dim=-1)
    row2 = torch.stack([zeros, zeros, ones, zeros], dim=-1)
    row3 = torch.stack([zeros, zeros, zeros, ones], dim=-1)
    fm = torch.stack([row0, row1, row2, row3], dim=1)  # [B, 4, 4]
    return torch.bmm(fm, x.unsqueeze(-1)).squeeze(-1)


def h_batched(x: torch.Tensor) -> torch.Tensor:
    """``H`` has no ``dt`` dependence -- one shared matrix applied to every
    row identically. ``x``: ``[B, STATE_DIM]`` -> ``[B, MEAS_DIM]``."""
    hm = torch.tensor(_H_NP, dtype=x.dtype, device=x.device)
    return x @ hm.T


class BatchedKalmanNetFilter:
    """Training-only batched drop-in for ``KalmanNetFilter``'s recursion.
    One instance = one batch of ``B`` independent sequences sharing the
    SAME ``KalmanNetGRU`` weights but each owning its own state row and
    hidden-state row -- see the module docstring's "no cross-track
    leakage" discussion, and ``test_batched_kalmannet.py``'s explicit
    isolation tests.
    """

    def __init__(self, net: KalmanNetGRU, device: str = "cpu"):
        self.net = net
        self.device = device
        self.x_post: torch.Tensor | None = None
        self.x_post_prev: torch.Tensor | None = None
        self.x_prior_prev: torch.Tensor | None = None
        self.y_prev: torch.Tensor | None = None

    def init_batch(self, x0: torch.Tensor) -> None:
        """``x0``: ``[B, STATE_DIM]`` -- each row's own frame-0 state
        (``[z0_x, z0_y, 0, 0]``), mirroring ``KalmanNetFilter.init_sequence``
        exactly, per row."""
        b = x0.shape[0]
        self.net.reset_hidden(b, device=self.device)
        self.x_post = x0.clone()
        self.x_post_prev = x0.clone()
        self.x_prior_prev = x0.clone()
        self.y_prev = h_batched(x0)

    def step_masked(
        self,
        z: torch.Tensor,
        dt: torch.Tensor,
        measurement_valid: torch.Tensor,
        sequence_valid: torch.Tensor,
        diagnostics_out: dict | None = None,
    ) -> torch.Tensor:
        """One global timestep across the whole batch.

        ``z``: ``[B, MEAS_DIM]`` -- garbage/zero permitted wherever
        ``measurement_valid`` is False; never read there (never treated as
        a real ``[0,0]`` measurement -- see module docstring / task
        section 5).
        ``dt``: ``[B]`` -- each row's own real ``dt`` at this timestep;
        garbage permitted wherever ``sequence_valid`` is False (never used
        there).
        ``measurement_valid``: ``[B]`` bool -- this timestep's detector
        measurement is present for this row (CASE C) vs. missing (CASE B).
        Only meaningful where ``sequence_valid`` is True.
        ``sequence_valid``: ``[B]`` bool -- this row actually HAS a real
        timestep here (True) vs. is padding past its own sequence's end
        (CASE A, False).
        ``diagnostics_out``: optional, default ``None`` (zero behavior
        change when omitted -- purely additive). If given a dict, this
        call populates it in place with this timestep's intermediate
        quantities (``meas_idx``, ``predict_only_idx``, ``x_prior``,
        ``innovation``, ``gain`` -- all detached, on the ``meas``/
        ``predict_only`` subset's own rows) for forensic instrumentation
        (see ``explosion_forensics.py``) -- these are read directly from
        THIS execution, never recomputed by a second, potentially
        divergent replay.

        Returns the POST state at this timestep for every row
        (``[B, STATE_DIM]``) -- rows where ``sequence_valid`` is False are
        simply the previous ``x_post`` carried forward unchanged (never
        used downstream since the caller must additionally gate on
        ``sequence_valid`` before using this for loss).
        """
        active = sequence_valid
        meas = active & measurement_valid
        predict_only = active & (~measurement_valid)

        new_x_post = self.x_post.clone()
        new_x_post_prev = self.x_post_prev.clone()
        new_x_prior_prev = self.x_prior_prev.clone()
        new_y_prev = self.y_prev.clone()

        # CASE B: real timestep, missing measurement -> analytical
        # predict-only. Mirrors trainer_core.run_sequence's own
        # `else: kf.x_post = KalmanNetFilter._f(kf.x_post, dt)` branch
        # EXACTLY -- only x_post is overwritten; x_post_prev/x_prior_prev/
        # y_prev/hidden state are left untouched for these rows (no
        # network call at all).
        if bool(predict_only.any()):
            idx = predict_only.nonzero(as_tuple=True)[0]
            x_prior_predict_only = f_batched(self.x_post[idx], dt[idx])
            new_x_post[idx] = x_prior_predict_only
            if diagnostics_out is not None:
                diagnostics_out["predict_only_idx"] = idx.detach().clone()
                diagnostics_out["predict_only_x_prior"] = x_prior_predict_only.detach().clone()

        # CASE C: real timestep, measurement present -> full predict +
        # learned-gain update, exactly mirroring
        # trainer_core.run_sequence's `if z is not None and dt is not
        # None:` branch, but only for this subset of rows.
        if bool(meas.any()):
            idx = meas.nonzero(as_tuple=True)[0]
            x_post_sub = self.x_post[idx]
            x_post_prev_sub = self.x_post_prev[idx]
            x_prior_prev_sub = self.x_prior_prev[idx]
            y_prev_sub = self.y_prev[idx]
            z_sub = z[idx]
            dt_sub = dt[idx]

            x_prior_sub = f_batched(x_post_sub, dt_sub)
            y_pred_sub = h_batched(x_prior_sub)

            obs_diff = z_sub - y_prev_sub
            obs_innov_diff = z_sub - y_pred_sub
            fw_evol_diff = x_post_sub - x_post_prev_sub
            fw_update_diff = x_post_sub - x_prior_prev_sub

            # Subset-indexed hidden state: only THIS subset's rows of the
            # shared [1, B, hidden] tensor are read and written; every
            # other row (predict-only or padded) keeps its own untouched
            # tensor slice -- no cross-track leakage (see
            # test_batched_kalmannet.py::HiddenStateIsolation*).
            full_h = self.net.h
            self.net.h = full_h[:, idx, :]
            gain = self.net(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff)
            updated_h_sub = self.net.h
            new_full_h = full_h.clone()
            new_full_h[:, idx, :] = updated_h_sub
            self.net.h = new_full_h

            innovation = z_sub - y_pred_sub
            correction = torch.bmm(gain, innovation.unsqueeze(-1)).squeeze(-1)
            x_post_result = x_prior_sub + correction

            if diagnostics_out is not None:
                diagnostics_out["meas_idx"] = idx.detach().clone()
                diagnostics_out["meas_x_prior"] = x_prior_sub.detach().clone()
                diagnostics_out["innovation"] = innovation.detach().clone()
                diagnostics_out["gain"] = gain.detach().clone()
                diagnostics_out["meas_x_post"] = x_post_result.detach().clone()

            new_x_post[idx] = x_post_result
            new_x_post_prev[idx] = x_post_sub
            new_x_prior_prev[idx] = x_prior_sub
            new_y_prev[idx] = z_sub

        self.x_post = new_x_post
        self.x_post_prev = new_x_post_prev
        self.x_prior_prev = new_x_prior_prev
        self.y_prev = new_y_prev
        return self.x_post


@dataclass
class PaddedBatch:
    """Padded temporal tensors for a batch of ``B`` independent sequences,
    ``T_max`` = the longest sequence in the batch. Layout ``[B, T_max, D]``
    (batch-major -- chosen because ``BatchedKalmanNetFilter`` state is
    naturally ``[B, D]`` per timestep and this avoids a transpose in the
    hot loop)."""

    x_true: torch.Tensor          # [B, T_max, STATE_DIM]
    z_meas: torch.Tensor          # [B, T_max, MEAS_DIM] (garbage where measurement_valid is False)
    dt: torch.Tensor              # [B, T_max]            (garbage at t=0 and where sequence_valid is False)
    measurement_valid: torch.Tensor  # [B, T_max] bool
    sequence_valid: torch.Tensor  # [B, T_max] bool -- CASE A (False) vs. a real timestep (True)
    t_max: int
    batch_size: int


def build_padded_batch(sequences: list[dict[str, Any]], device: str = "cpu") -> PaddedBatch:
    """Pads a list of training-ready sequence dicts (the
    ``{"frames","dt","x_true","z_meas"}`` contract from
    ``kalmannet_sequences.py``, unchanged) into one ``PaddedBatch``.
    Every sequence's own frame 0 (``dt[0] is None``, its own first valid
    measurement by construction -- see ``_truncate_to_first_measurement``)
    is preserved as the batch's own ``t=0``; this is used only for
    initialization (never included in ``measurement_valid``/loss at
    ``t=0`` -- exactly matching the single-sequence path where ``dt[0]``
    is never in the loss)."""
    b = len(sequences)
    t_max = max(len(s["frames"]) for s in sequences)

    x_true = torch.zeros(b, t_max, STATE_DIM, dtype=torch.float32)
    z_meas = torch.zeros(b, t_max, MEAS_DIM, dtype=torch.float32)
    dt = torch.zeros(b, t_max, dtype=torch.float32)
    measurement_valid = torch.zeros(b, t_max, dtype=torch.bool)
    sequence_valid = torch.zeros(b, t_max, dtype=torch.bool)

    for i, seq in enumerate(sequences):
        n = len(seq["frames"])
        sequence_valid[i, :n] = True
        for t in range(n):
            x_true[i, t] = torch.tensor(seq["x_true"][t], dtype=torch.float32)
            z = seq["z_meas"][t]
            if z is not None:
                z_meas[i, t] = torch.tensor(z, dtype=torch.float32)
                measurement_valid[i, t] = True
            d = seq["dt"][t]
            if d is not None:
                dt[i, t] = float(d)

    return PaddedBatch(
        x_true=x_true.to(device), z_meas=z_meas.to(device), dt=dt.to(device),
        measurement_valid=measurement_valid.to(device), sequence_valid=sequence_valid.to(device),
        t_max=t_max, batch_size=b,
    )


@dataclass
class BatchLossResult:
    loss: torch.Tensor          # scalar mean over every counted (row, t) loss term across the WHOLE batch
    n_loss_terms: int
    nan_inf: bool


def run_batch(
    net: KalmanNetGRU, batch: PaddedBatch, device: str, loss_on_predict_only: bool,
    diagnostics_out: dict | None = None,
) -> BatchLossResult:
    """Batched equivalent of ``trainer_core.run_sequence``, run once per
    ``PaddedBatch`` instead of once per sequence. At ``batch.batch_size ==
    1`` this is numerically equivalent (within float32 tolerance) to
    calling ``run_sequence`` on the single underlying sequence -- see
    ``test_batched_kalmannet.py::test_batch_size_one_loss_equivalence``.

    Loss reduction: every individual (row, t) loss term across the WHOLE
    batch is collected into one flat list and averaged once -- i.e. for
    ``batch_size=1`` this is IDENTICAL to ``run_sequence``'s own
    ``torch.stack(losses).mean()`` (same terms, same reduction). For
    ``batch_size>1`` this differs from "sequential per-sequence
    optimizer.step" in the SAME way any mini-batch SGD differs from
    single-example SGD -- documented explicitly in
    ``docs/perception/kalmannet_batched_training_v1.md`` as
    THROUGHPUT MODE, never claimed numerically identical to the
    historical per-sequence optimizer path at ``batch_size>1``.

    ``diagnostics_out``: optional, default ``None`` (zero behavior change
    when omitted). If given a dict, populated in place with SCALAR
    running-max diagnostics across the whole batch/sequence
    (``max_innovation_abs``, ``max_prior_state_abs``,
    ``max_posterior_state_abs``, ``max_hidden_state_abs``,
    ``max_gain_abs``) -- read directly from THIS execution's own
    ``step_masked(..., diagnostics_out=...)`` calls (see
    ``explosion_forensics.py``), never a second, potentially divergent
    replay. Kept scalar-only (not per-timestep/per-row) so this stays
    cheap enough to enable on every batch of a bounded forensic run.
    """
    # Mirrors trainer_core.run_sequence's own x0 construction exactly:
    # x0 = [first_meas.x, first_meas.y, 0.0, 0.0] -- t=0 is guaranteed to
    # carry a real measurement by build_padded_batch's own precondition
    # (every input sequence is already truncated to its first measurement).
    b = batch.batch_size
    x0 = batch.x_true.new_zeros(b, STATE_DIM)
    x0[:, 0] = batch.z_meas[:, 0, 0]
    x0[:, 1] = batch.z_meas[:, 0, 1]

    bkf = BatchedKalmanNetFilter(net, device=device)
    bkf.init_batch(x0)

    all_losses: list[torch.Tensor] = []
    nan_inf = False

    def _running_max(key: str, tensor: torch.Tensor) -> None:
        if diagnostics_out is None or tensor.numel() == 0:
            return
        val = float(tensor.detach().abs().max().item())
        prev = diagnostics_out.get(key, 0.0)
        # NaN comparisons are always False, so max(nan, x) would silently
        # drop a NaN -- propagate it explicitly instead (a NaN magnitude
        # IS the signal a forensic run needs to see, not something to
        # discard via an unlucky comparison order).
        diagnostics_out[key] = val if (val != val or val > prev) else prev

    for t in range(1, batch.t_max):
        z_t = batch.z_meas[:, t, :]
        dt_t = batch.dt[:, t]
        meas_valid_t = batch.measurement_valid[:, t]
        seq_valid_t = batch.sequence_valid[:, t]
        gt_t = batch.x_true[:, t, :]

        step_diag: dict | None = {} if diagnostics_out is not None else None
        x_post_t = bkf.step_masked(z_t, dt_t, meas_valid_t, seq_valid_t, diagnostics_out=step_diag)
        if step_diag:
            if "innovation" in step_diag:
                _running_max("max_innovation_abs", step_diag["innovation"])
                _running_max("max_gain_abs", step_diag["gain"])
                _running_max("max_prior_state_abs", step_diag["meas_x_prior"])
                _running_max("max_posterior_state_abs", step_diag["meas_x_post"])
            if "predict_only_x_prior" in step_diag:
                _running_max("max_prior_state_abs", step_diag["predict_only_x_prior"])
            if bkf.net.h is not None:
                _running_max("max_hidden_state_abs", bkf.net.h)

        if loss_on_predict_only:
            counted = seq_valid_t
        else:
            counted = seq_valid_t & meas_valid_t

        if bool(counted.any()):
            idx = counted.nonzero(as_tuple=True)[0]
            per_row_sq_err = torch.mean((x_post_t[idx] - gt_t[idx]) ** 2, dim=-1)  # [n_counted]
            if not torch.isfinite(per_row_sq_err).all():
                nan_inf = True
            all_losses.append(per_row_sq_err)
            if diagnostics_out is not None:
                # Cheap per-ROW loss accumulation (sum + count), reusing
                # the SAME per_row_sq_err just computed above -- avoids a
                # second, separate per-sequence forward pass just to get
                # a per-sequence loss breakdown (see explosion_forensics.py
                # "RingBufferEntry" -- per-sequence loss min/median/p90/max
                # for a bounded forensic run's per-batch summary).
                row_loss_sum = diagnostics_out.setdefault(
                    "_per_row_loss_sum", torch.zeros(batch.batch_size, device=device)
                )
                row_loss_count = diagnostics_out.setdefault(
                    "_per_row_loss_count", torch.zeros(batch.batch_size, device=device)
                )
                row_loss_sum.index_add_(0, idx, per_row_sq_err.detach())
                row_loss_count.index_add_(0, idx, torch.ones_like(per_row_sq_err))
        if bkf.net.h is not None and not torch.isfinite(bkf.net.h).all():
            nan_inf = True

    if diagnostics_out is not None and "_per_row_loss_sum" in diagnostics_out:
        row_sum = diagnostics_out.pop("_per_row_loss_sum")
        row_count = diagnostics_out.pop("_per_row_loss_count")
        safe_count = torch.where(row_count > 0, row_count, torch.ones_like(row_count))
        diagnostics_out["per_row_loss"] = (row_sum / safe_count).cpu().numpy()

    if not all_losses:
        return BatchLossResult(torch.tensor(0.0, device=device), 0, False)
    flat = torch.cat(all_losses)
    return BatchLossResult(flat.mean(), int(flat.numel()), nan_inf)


def group_into_batches(
    sequences: list[dict[str, Any]], order: np.ndarray, batch_size: int
) -> list[list[dict[str, Any]]]:
    """Simple contiguous chunking of a (caller-supplied, already shuffled
    or length-bucketed) index order into fixed-size groups -- the last
    group may be smaller. No hidden reordering beyond what ``order``
    already encodes, so callers control determinism entirely."""
    ordered = [sequences[i] for i in order]
    return [ordered[i : i + batch_size] for i in range(0, len(ordered), batch_size)]


def make_length_buckets(sequences: list[dict[str, Any]], n_buckets: int = 8) -> list[np.ndarray]:
    """Deterministic, seed-independent (pure function of each sequence's
    own ``len(frames)``) assignment of sequence indices into
    ``n_buckets`` length buckets, ascending by length. No class/actor-type
    information is used -- purely temporal length, per this task's own
    "no class-based batching" constraint."""
    lengths = np.array([len(s["frames"]) for s in sequences])
    order_by_len = np.argsort(lengths, kind="stable")
    n_buckets = max(1, min(n_buckets, len(sequences)))
    return list(np.array_split(order_by_len, n_buckets))


def bucketed_epoch_batch_order(
    sequences: list[dict[str, Any]], rng: np.random.RandomState, batch_size: int, n_buckets: int = 8
) -> list[np.ndarray]:
    """Deterministic-given-``rng`` per-epoch batch construction that keeps
    each batch's sequences close in length (reduces padding waste vs. a
    fully random shuffle): shuffle WITHIN each length bucket (using
    ``rng``, buckets visited in a fixed ascending-length order so the same
    ``rng`` state always advances identically), concatenate bucket-by-
    bucket, chunk into fixed-size groups, then shuffle the resulting
    BATCH order (never batch *contents*) with the same ``rng``. Returns a
    list of index arrays (one per batch), ready for
    ``group_into_batches``-style materialization (here inlined: each
    returned array of original sequence indices IS the batch)."""
    buckets = make_length_buckets(sequences, n_buckets)
    ordered_indices: list[int] = []
    for bucket in buckets:
        shuffled = bucket[rng.permutation(len(bucket))]
        ordered_indices.extend(int(i) for i in shuffled)
    batches = [
        np.array(ordered_indices[i : i + batch_size]) for i in range(0, len(ordered_indices), batch_size)
    ]
    batch_order = rng.permutation(len(batches))
    return [batches[i] for i in batch_order]


def padding_waste_fraction(sequences: list[dict[str, Any]], batches_of_indices: list[np.ndarray]) -> float:
    """Diagnostic only: fraction of (row, timestep) slots in the padded
    tensors that are padding (``sequence_valid == False``) under a given
    batch grouping -- used to compare naive contiguous batching vs.
    length-bucketed batching on real data (see the batched-training doc's
    own measured figures), never used inside the training loop itself."""
    total_slots = 0
    real_slots = 0
    for idx in batches_of_indices:
        batch_seqs = [sequences[i] for i in idx]
        t_max = max(len(s["frames"]) for s in batch_seqs)
        total_slots += len(batch_seqs) * t_max
        real_slots += sum(len(s["frames"]) for s in batch_seqs)
    if total_slots == 0:
        return 0.0
    return 1.0 - (real_slots / total_slots)
