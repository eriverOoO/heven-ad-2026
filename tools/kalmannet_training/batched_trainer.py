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
from dataclasses import dataclass
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
from trainer_core import (
    FAILURE_VAL_LOSS_THRESHOLD,
    EpochRecord,
    TrainResult,
    _grad_norm,
    set_all_seeds,
)

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
) -> TrainResult:
    """THROUGHPUT MODE training loop. Same seeds/hyperparameter contract
    and ``TrainResult`` shape as ``trainer_core.train_one_run`` (so
    downstream checkpoint/reporting code needs no change) -- the only
    behavioral difference is the batched forward/backward path and,
    ONLY when ``batch_size>1``, mini-batch (not per-sequence) gradient
    averaging."""
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
    t_start = time.time()

    val_batch_idx = _val_batches(val_seqs, batch_size, n_buckets) if val_seqs else []

    for epoch in range(max_epochs):
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

        for idx_group in batch_index_groups:
            if len(idx_group) == 0:
                continue
            batch_seqs = [train_seqs[i] for i in idx_group]
            padded = build_padded_batch(batch_seqs, device=device)
            opt.zero_grad()
            batch_result: BatchLossResult = run_batch(net, padded, device, loss_on_predict_only)
            if batch_result.n_loss_terms == 0:
                continue
            if not torch.isfinite(batch_result.loss):
                any_nan_train = True
                result.nonfinite_step_count += 1
                epoch_train_losses.append(float("nan"))
                continue
            batch_result.loss.backward()
            gnorm = _grad_norm(net.parameters())
            grad_norms.append(gnorm)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            opt.step()
            epoch_train_losses.append(batch_result.loss.item())
            if batch_result.nan_inf:
                any_nan_train = True
                result.nonfinite_step_count += 1

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
        )
        result.history.append(record)
        if progress_callback is not None:
            progress_callback(record, time.time() - t_start)

        if math.isfinite(val_loss) and val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                break

    result.train_time_s = time.time() - t_start
    result.n_epochs_run = len(result.history)
    result.best_val = best_val
    result.best_epoch = best_epoch
    result.best_state_dict = best_state
    result.catastrophic = (
        not math.isfinite(best_val)
        or best_val > FAILURE_VAL_LOSS_THRESHOLD
        or best_state is None
    )
    return result
