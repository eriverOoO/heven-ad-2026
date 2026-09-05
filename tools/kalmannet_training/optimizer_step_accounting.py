"""Pure-function optimizer-step accounting: makes explicit why "same
epoch count" is NOT "same training budget" once ``batch_size>1`` --
one epoch at ``batch_size=B`` produces ``ceil(N/B)`` optimizer steps
(one gradient-averaged Adam update per batch), vs. ``N`` steps at
``batch_size=1`` (one Adam update per sequence). No model/training-loop
code -- these functions only count, they never run anything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def steps_per_epoch(n_sequences: int, batch_size: int) -> int:
    """Number of optimizer.step() calls one epoch produces -- the last,
    possibly-smaller batch still counts as one step, exactly matching
    ``batched_trainer.py``'s own batch-grouping behavior."""
    if n_sequences <= 0 or batch_size <= 0:
        raise ValueError("n_sequences and batch_size must be positive")
    return math.ceil(n_sequences / batch_size)


def total_optimizer_steps(n_sequences: int, batch_size: int, n_epochs: int) -> int:
    return steps_per_epoch(n_sequences, batch_size) * n_epochs


@dataclass(frozen=True)
class StepBudgetComparison:
    batch_size: int
    steps_per_epoch: int
    steps_at_n_epochs: int
    n_epochs_compared: int
    historical_steps_per_epoch: int
    historical_steps_at_n_epochs: int
    step_deficit_ratio: float  # historical_steps / this_config_steps, at the SAME n_epochs


def compare_to_historical(n_sequences: int, batch_size: int, n_epochs: int) -> StepBudgetComparison:
    """``step_deficit_ratio`` is how many FEWER optimizer updates this
    ``batch_size`` receives than the historical ``batch_size=1`` trainer,
    at the SAME epoch count -- the number this task's own premise (10
    epochs is 10 epochs regardless of batch size) gets wrong."""
    hist_spe = steps_per_epoch(n_sequences, 1)
    this_spe = steps_per_epoch(n_sequences, batch_size)
    hist_total = hist_spe * n_epochs
    this_total = this_spe * n_epochs
    return StepBudgetComparison(
        batch_size=batch_size, steps_per_epoch=this_spe, steps_at_n_epochs=this_total,
        n_epochs_compared=n_epochs, historical_steps_per_epoch=hist_spe,
        historical_steps_at_n_epochs=hist_total,
        step_deficit_ratio=(hist_total / this_total if this_total > 0 else math.inf),
    )


def epochs_needed_to_match_historical_steps(
    n_sequences: int, batch_size: int, historical_n_epochs: int
) -> float:
    """How many epochs at this ``batch_size`` would be needed to reach the
    SAME total optimizer-step count the historical trainer accumulated
    over ``historical_n_epochs`` -- a reference number only (this task
    does NOT require matching it exactly, see ``docs/perception/
    kalmannet_batch_calibration_v1.md``'s "Do not seek exact historical
    equivalence" section); useful for sizing a generous ``max_epochs``
    upper bound for the calibration search."""
    hist_total = steps_per_epoch(n_sequences, 1) * historical_n_epochs
    this_spe = steps_per_epoch(n_sequences, batch_size)
    return hist_total / this_spe
