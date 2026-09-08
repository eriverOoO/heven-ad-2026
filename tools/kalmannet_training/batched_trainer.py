"""THROUGHPUT MODE training loop: the batched epoch/optimizer driver
built on top of ``batched_kalmannet.py``'s ``run_batch``/
``BatchedKalmanNetFilter``.

**Distinct optimization regime, documented explicitly, never claimed
numerically identical to the historical per-sequence optimizer path at
``batch_size>1``** (see ``docs/perception/kalmannet_batched_training_v1.md``
"Mini-batch optimization semantics"). At ``batch_size=1`` this reduces to
CORRECTNESS MODE and IS numerically equivalent to
``trainer_core.train_one_run`` -- see
``test_batched_kalmannet.py``/``test_batched_trainer.py`` for the exact
equivalence tests.

Reuses ``trainer_core.py``'s ``EpochRecord``/``TrainResult``/
``set_all_seeds``/``_grad_norm``/``FAILURE_VAL_LOSS_THRESHOLD`` verbatim
-- no duplicated bookkeeping types, no change to that module.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from batched_kalmannet import (
    BatchLossResult,
    bucketed_epoch_batch_order,
    build_padded_batch,
    make_length_buckets,
    run_batch,
)
from nonfinite_guard import MAX_NONFINITE_GRAD_SKIPS_PER_EPOCH, safe_clip_and_step
from resume_state import load_resume_state, save_resume_state
from trainer_core import (
    FAILURE_VAL_LOSS_THRESHOLD,
    EpochRecord,
    TrainResult,
    set_all_seeds,
)
from training_forensics import BatchObservation, ForensicRecorder, hidden_state_stats, summarize_sequence

from ad_lidar_perception.kalmannet_core import KalmanNetGRU  # noqa: E402  (path added by batched_kalmannet's own import)


@dataclass
class BatchedTrainConfig:
    batch_size: int = 32
    use_length_bucketing: bool = True
    n_buckets: int = 8


def _val_batches(val_seqs: list[dict[str, Any]], batch_size: int, n_buckets: int) -> list[np.ndarray]:
    """Deterministic (no RNG needed) validation batching: length-sorted
    contiguous chunks, minimizing padding without needing a seed --
    evaluation order never affects the reported metric (mean over all
    terms), so no shuffling is needed for reproducibility here."""
    buckets = make_length_buckets(val_seqs, n_buckets)
    ordered = np.concatenate(buckets) if buckets else np.array([], dtype=int)
    return [ordered[i : i + batch_size] for i in range(0, len(ordered), batch_size)]


def train_one_run_batched(
    train_seqs: list[dict[str, Any]],
    val_seqs: list[dict[str, Any]],
    seed: int,
    device: str,
    batch_size: int = 32,
    use_length_bucketing: bool = True,
    n_buckets: int = 8,
    lr: float = 0.001,
    max_epochs: int = 60,
    patience: int = 10,
    hidden_size: int = 32,
    grad_clip: float | None = 10.0,
    loss_on_predict_only: bool = True,
    init_seed: int | None = None,
    order_seed: int | None = None,
    deterministic_cuda: bool = False,
    progress_callback=None,
    resume_checkpoint_path: Path | None = None,
    resume_validation_key: dict[str, Any] | None = None,
    checkpoint_every_epochs: int = 1,
    forensic_recorder: ForensicRecorder | None = None,
) -> TrainResult:
    """THROUGHPUT MODE training loop. Same seeds/hyperparameter contract
    and ``TrainResult`` shape as ``trainer_core.train_one_run`` (so
    downstream checkpoint/reporting code needs no change) -- the only
    behavioral difference is the batched forward/backward path and,
    ONLY when ``batch_size>1``, mini-batch (not per-sequence) gradient
    averaging.

    ``resume_checkpoint_path`` (optional, default ``None`` -- fully
    backward compatible, identical behavior to before this parameter
    existed): when given, a resume checkpoint is written after every
    ``checkpoint_every_epochs`` completed epoch(s). If a checkpoint
    already exists at that path when this function is called, training
    resumes from it instead of starting fresh (``resume_validation_key``
    -- e.g. seed/lr/batch_size/hidden_size/dataset+split content hashes
    -- must match exactly what produced that checkpoint, or
    ``resume_state.ResumeValidationError`` is raised rather than silently
    continuing under a different configuration)."""
    init_seed = seed if init_seed is None else init_seed
    order_seed = seed if order_seed is None else order_seed

    set_all_seeds(init_seed, deterministic_cuda=deterministic_cuda)
    net = KalmanNetGRU(hidden_size=hidden_size).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    rng = np.random.RandomState(order_seed)
    result = TrainResult(
        seed=seed, init_seed=init_seed, order_seed=order_seed, device=device, lr=lr,
        grad_clip=grad_clip if grad_clip is not None else -1.0, hidden_size=hidden_size,
        loss_on_predict_only=loss_on_predict_only,
    )
    best_val = math.inf
    best_state = None
    best_epoch = None
    epochs_since_improve = 0
    start_epoch = 0
    cumulative_train_time_s = 0.0

    if resume_checkpoint_path is not None and Path(resume_checkpoint_path).is_file():
        payload = load_resume_state(resume_checkpoint_path, expected_validation_key=resume_validation_key or {})
        net.load_state_dict(payload["model_state_dict"])
        opt.load_state_dict(payload["optimizer_state_dict"])
        rng.set_state(payload["rng_state"])
        start_epoch = payload["epoch_next"]
        best_val = payload["best_val"]
        best_epoch = payload["best_epoch"]
        best_state = payload["best_state_dict"]
        epochs_since_improve = payload["epochs_since_improve"]
        result.history = [EpochRecord(**rec) for rec in payload["history"]]
        result.nonfinite_step_count = payload["nonfinite_step_count"]
        cumulative_train_time_s = payload["cumulative_train_time_s"]
        # PR #58 gradient-state health counters (see docs/perception/
        # kalmannet_gradient_norm_overflow_v1.md) -- restored from the
        # generic extra_counters bag so a resumed run's final cumulative
        # counts include the pre-crash portion too, not just what
        # accumulates after resuming. Missing keys (an OLDER resume
        # checkpoint saved before this field existed) default to the
        # TrainResult dataclass's own "nothing happened yet" values.
        extra = payload.get("extra_counters", {})
        result.grad_skip_count = extra.get("grad_skip_count", result.grad_skip_count)
        result.training_unstable = extra.get("training_unstable", result.training_unstable)
        result.training_collapsed = extra.get("training_collapsed", result.training_collapsed)
        result.abort_reason = extra.get("abort_reason", result.abort_reason)
        result.large_finite_gradient_count = extra.get(
            "large_finite_gradient_count", result.large_finite_gradient_count)
        result.norm_overflow_count = extra.get("norm_overflow_count", result.norm_overflow_count)
        result.norm_overflow_skip_count = extra.get("norm_overflow_skip_count", result.norm_overflow_skip_count)
        result.per_element_nonfinite_gradient_count = extra.get(
            "per_element_nonfinite_gradient_count", result.per_element_nonfinite_gradient_count)
        result.nonfinite_gradient_skip_count = extra.get(
            "nonfinite_gradient_skip_count", result.nonfinite_gradient_skip_count)
        result.parameter_collapse_count = extra.get("parameter_collapse_count", result.parameter_collapse_count)
        result.optimizer_state_collapse_count = extra.get(
            "optimizer_state_collapse_count", result.optimizer_state_collapse_count)

    t_start = time.time()

    val_batch_idx = _val_batches(val_seqs, batch_size, n_buckets) if val_seqs else []

    for epoch in range(start_epoch, max_epochs):
        if use_length_bucketing:
            batch_index_groups = bucketed_epoch_batch_order(train_seqs, rng, batch_size, n_buckets)
        else:
            order = rng.permutation(len(train_seqs))
            batch_index_groups = [
                order[i : i + batch_size] for i in range(0, len(order), batch_size)
            ]
            batch_order = rng.permutation(len(batch_index_groups))
            batch_index_groups = [batch_index_groups[i] for i in batch_order]

        net.train()
        epoch_train_losses: list[float] = []
        grad_norms: list[float] = []
        any_nan_train = False
        nonfinite_grad_skips_this_epoch = 0
        epoch_norm_overflow_count = 0
        epoch_element_nonfinite_count = 0

        for batch_index, idx_group in enumerate(batch_index_groups):
            if len(idx_group) == 0:
                continue
            batch_seqs = [train_seqs[i] for i in idx_group]
            padded = build_padded_batch(batch_seqs, device=device)
            opt.zero_grad()
            batch_result: BatchLossResult = run_batch(net, padded, device, loss_on_predict_only)
            if batch_result.n_loss_terms == 0:
                continue

            loss_finite = bool(torch.isfinite(batch_result.loss))

            if not loss_finite:
                any_nan_train = True
                result.nonfinite_step_count += 1
                epoch_train_losses.append(float("nan"))
                if forensic_recorder is not None:
                    h_mean, h_max, h_p95 = hidden_state_stats(net.h)
                    forensic_recorder.observe(BatchObservation(
                        epoch=epoch, batch_index=batch_index, n_sequences=len(batch_seqs),
                        loss_value=float(batch_result.loss.item()),
                        loss_finite=False, grad_norm_pre_clip=float("nan"), grad_nonfinite_pre_clip=False,
                        hidden_state_mean_abs=h_mean, hidden_state_max_abs=h_max, hidden_state_p95_abs=h_p95,
                        sequences=[summarize_sequence(s) for s in batch_seqs],
                    ))
                continue
            batch_result.loss.backward()

            # SAFE STEP GUARD (see nonfinite_guard.py / docs/perception/
            # kalmannet_training_instability_v1.md): a FINITE loss can
            # still backward() into a non-finite gradient -- this is the
            # exact, root-caused mechanism that permanently corrupted the
            # AV2 10k GENERIC-ROBUST run's weights from epoch 16 onward.
            # clip_grad_norm_ does NOT neutralize a non-finite gradient
            # into a safe update; calling optimizer.step() with one
            # permanently poisons Adam's moment buffers. This guard
            # inspects the gradient BEFORE clipping and refuses to step
            # at all if it is non-finite.
            outcome = safe_clip_and_step(net, opt, grad_clip)
            grad_norms.append(outcome.grad_norm_pre_clip)

            if forensic_recorder is not None:
                h_mean, h_max, h_p95 = hidden_state_stats(net.h)
                forensic_recorder.observe(BatchObservation(
                    epoch=epoch, batch_index=batch_index, n_sequences=len(batch_seqs),
                    loss_value=float(batch_result.loss.item()), loss_finite=True,
                    grad_norm_pre_clip=outcome.grad_norm_pre_clip,
                    grad_nonfinite_pre_clip=outcome.grad_nonfinite_pre_clip,
                    hidden_state_mean_abs=h_mean, hidden_state_max_abs=h_max, hidden_state_p95_abs=h_p95,
                    sequences=[summarize_sequence(s) for s in batch_seqs],
                ))

            if outcome.gradient_state == "norm_overflow":
                # STATE B: every gradient element finite, but the
                # aggregate norm reduction overflowed. Self-neutralizing
                # (never poisons Adam), so this deliberately does NOT
                # feed the STATE-C-specific tier-2 escalation counter
                # (`nonfinite_grad_skips_this_epoch`) -- it is tracked in
                # its own dedicated counters instead (section 5 of
                # docs/perception/kalmannet_gradient_norm_overflow_v1.md).
                any_nan_train = True
                result.nonfinite_step_count += 1
                result.grad_skip_count += 1
                result.norm_overflow_count += 1
                result.norm_overflow_skip_count += 1
                epoch_norm_overflow_count += 1
                epoch_train_losses.append(float("nan"))
                continue

            if not outcome.applied:
                # Remaining not-applied case: STATE C (element-nonfinite),
                # possibly the defensive post-clip path (unreachable in
                # practice now that STATE B is routed away above).
                any_nan_train = True
                result.nonfinite_step_count += 1
                result.grad_skip_count += 1
                result.per_element_nonfinite_gradient_count += 1
                result.nonfinite_gradient_skip_count += 1
                nonfinite_grad_skips_this_epoch += 1
                epoch_element_nonfinite_count += 1
                epoch_train_losses.append(float("nan"))
                continue

            if outcome.collapsed:
                # Escalation policy tier 3 (immediate): parameters or
                # optimizer state became non-finite despite the pre-step
                # guard -- an invariant this design should make
                # unreachable, so treat it as an unrecoverable, immediate
                # failure rather than continuing to train on a corrupted
                # model. FAIL FAST: mark collapsed and stop training now,
                # mid-epoch, rather than producing further NaN epochs.
                if outcome.params_nonfinite_after_step:
                    result.parameter_collapse_count += 1
                if outcome.optimizer_state_nonfinite_after_step:
                    result.optimizer_state_collapse_count += 1
                any_nan_train = True
                result.training_collapsed = True
                result.abort_reason = "params_or_optimizer_state_nonfinite_after_step"
                epoch_train_losses.append(float("nan"))
                break

            if grad_clip is not None and outcome.grad_norm_pre_clip > grad_clip:
                result.large_finite_gradient_count += 1

            epoch_train_losses.append(batch_result.loss.item())
            if batch_result.nan_inf:
                any_nan_train = True
                result.nonfinite_step_count += 1

        if result.training_collapsed:
            record = EpochRecord(
                epoch=epoch,
                train_loss=(float(np.nanmean(epoch_train_losses)) if epoch_train_losses else float("nan")),
                val_loss=float("nan"), grad_norm_mean=(float(np.mean(grad_norms)) if grad_norms else 0.0),
                grad_norm_max=(float(np.max(grad_norms)) if grad_norms else 0.0),
                any_nan_train=True, any_nan_val=False,
                n_norm_overflow_batches=epoch_norm_overflow_count,
                n_element_nonfinite_batches=epoch_element_nonfinite_count,
            )
            result.history.append(record)
            break

        # Escalation policy tier 2 (per-epoch): repeated non-finite
        # gradients within one epoch, well beyond an isolated outlier
        # batch, signal systematic instability rather than one unlucky
        # sequence -- abort after this epoch (its own EpochRecord is
        # still recorded truthfully below) rather than continuing to burn
        # further epochs of compute on a run that keeps re-triggering the
        # guard.
        epoch_unstable = nonfinite_grad_skips_this_epoch > MAX_NONFINITE_GRAD_SKIPS_PER_EPOCH

        net.eval()
        val_losses: list[float] = []
        any_nan_val = False
        with torch.no_grad():
            for idx_group in val_batch_idx:
                if len(idx_group) == 0:
                    continue
                batch_seqs = [val_seqs[i] for i in idx_group]
                padded = build_padded_batch(batch_seqs, device=device)
                batch_result = run_batch(net, padded, device, loss_on_predict_only)
                if batch_result.n_loss_terms == 0:
                    continue
                v = batch_result.loss.item() if torch.isfinite(batch_result.loss) else float("nan")
                val_losses.append(v)
                any_nan_val = any_nan_val or not math.isfinite(v)

        train_loss = float(np.nanmean(epoch_train_losses)) if epoch_train_losses else float("nan")
        val_loss = float(np.nanmean(val_losses)) if val_losses else float("nan")
        record = EpochRecord(
            epoch=epoch, train_loss=train_loss, val_loss=val_loss,
            grad_norm_mean=(float(np.mean(grad_norms)) if grad_norms else 0.0),
            grad_norm_max=(float(np.max(grad_norms)) if grad_norms else 0.0),
            any_nan_train=any_nan_train, any_nan_val=any_nan_val,
            n_norm_overflow_batches=epoch_norm_overflow_count,
            n_element_nonfinite_batches=epoch_element_nonfinite_count,
        )
        result.history.append(record)
        if progress_callback is not None:
            progress_callback(record, cumulative_train_time_s + (time.time() - t_start))

        if math.isfinite(val_loss) and val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if epoch_unstable:
            result.training_unstable = True
            result.abort_reason = (
                f"repeated_nonfinite_gradients: {nonfinite_grad_skips_this_epoch} skips in epoch {epoch} "
                f"(threshold {MAX_NONFINITE_GRAD_SKIPS_PER_EPOCH})"
            )

        stop_early = epochs_since_improve >= patience or epoch_unstable

        if resume_checkpoint_path is not None and (
            (epoch + 1) % checkpoint_every_epochs == 0 or epoch == max_epochs - 1 or stop_early
        ):
            save_resume_state(
                resume_checkpoint_path,
                model_state_dict=net.state_dict(),
                optimizer_state_dict=opt.state_dict(),
                rng_state=rng.get_state(),
                epoch_next=epoch + 1,
                best_val=best_val,
                best_epoch=best_epoch,
                best_state_dict=best_state,
                epochs_since_improve=epochs_since_improve,
                history=[asdict(r) for r in result.history],
                nonfinite_step_count=result.nonfinite_step_count,
                cumulative_train_time_s=cumulative_train_time_s + (time.time() - t_start),
                validation_key=resume_validation_key or {},
                extra_counters={
                    "grad_skip_count": result.grad_skip_count,
                    "training_unstable": result.training_unstable,
                    "training_collapsed": result.training_collapsed,
                    "abort_reason": result.abort_reason,
                    "large_finite_gradient_count": result.large_finite_gradient_count,
                    "norm_overflow_count": result.norm_overflow_count,
                    "norm_overflow_skip_count": result.norm_overflow_skip_count,
                    "per_element_nonfinite_gradient_count": result.per_element_nonfinite_gradient_count,
                    "nonfinite_gradient_skip_count": result.nonfinite_gradient_skip_count,
                    "parameter_collapse_count": result.parameter_collapse_count,
                    "optimizer_state_collapse_count": result.optimizer_state_collapse_count,
                },
            )

        if stop_early:
            break

    result.train_time_s = cumulative_train_time_s + (time.time() - t_start)
    result.n_epochs_run = len(result.history)
    result.best_val = best_val
    result.best_epoch = best_epoch
    result.best_state_dict = best_state
    # Fixed catastrophic-flag semantics (see docs/perception/
    # kalmannet_training_instability_v1.md "Catastrophic-Flag Bug"): the
    # ORIGINAL definition below only reflected whether the SELECTED
    # (best-epoch) checkpoint was itself degenerate -- a run whose weights
    # permanently went non-finite mid-run but which had ALREADY locked in
    # a good best_epoch before that point (exactly what happened in the
    # real AV2 10k GENERIC-ROBUST run) reported catastrophic=False despite
    # the model having numerically collapsed. `training_collapsed` now
    # captures that failure mode directly and unconditionally forces
    # catastrophic=True, regardless of whether an earlier checkpoint was
    # already secured -- "the model numerically collapsed" is no longer
    # allowed to silently coexist with catastrophic=False.
    result.catastrophic = (
        not math.isfinite(best_val)
        or best_val > FAILURE_VAL_LOSS_THRESHOLD
        or best_state is None
        or result.training_collapsed
    )
    return result
