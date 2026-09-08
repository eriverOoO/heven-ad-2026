"""Deterministic multi-seed config generation + summary statistics for
the final 3-seed stability check (this task's own section 11) -- pure
functions, no training-loop dependency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def deterministic_seed_list(n_seeds: int, base_seed: int = 0) -> list[int]:
    """Simple, fully deterministic, human-legible seed list --
    ``[base_seed, base_seed+1, ..., base_seed+n_seeds-1]``. Not derived
    from any hash/RNG (there is nothing to hide or make unpredictable
    about which seeds were run -- the manifest records them verbatim)."""
    if n_seeds <= 0:
        raise ValueError("n_seeds must be positive")
    return [base_seed + i for i in range(n_seeds)]


@dataclass(frozen=True)
class MultiSeedSummary:
    n_seeds: int
    seeds: list[int]
    val_position_rmse_mean: float
    val_position_rmse_std: float
    val_position_rmse_min: float
    val_position_rmse_max: float
    val_velocity_rmse_mean: float
    val_velocity_rmse_std: float
    best_epoch_values: list[int]
    wall_s_total: float
    any_catastrophic: bool
    # Added for the AV2 10k multi-seed task (docs/perception/
    # kalmannet_av2_10k_multiseed_v1.md) -- per-seed numerical-health
    # aggregation, sections 13/14 (norm-overflow/element-nonfinite counts,
    # catastrophic/unstable seed identification). Sourced via ``.get()``
    # with a 0/False default, so any EXISTING caller's minimal result
    # dict (without these keys, e.g. the pre-existing calibration-sweep
    # callers this function was originally built for) is unaffected.
    norm_overflow_counts: list[int]
    element_nonfinite_counts: list[int]
    catastrophic_seeds: list[int]
    unstable_seeds: list[int]


def summarize_multi_seed_results(results: list[dict[str, Any]]) -> MultiSeedSummary:
    """``results``: one dict per seed, each with at minimum
    ``seed``, ``val_position_rmse``, ``val_velocity_rmse``, ``best_epoch``,
    ``wall_s``, ``catastrophic`` keys (the shape this task's own
    calibration/confirmation scripts produce). Optionally also
    ``norm_overflow_count``, ``element_nonfinite_count``,
    ``training_unstable`` -- when present, folded into the numerical-
    health aggregation fields below; when absent, those default to
    0/0/False per seed (never fabricated)."""
    if not results:
        raise ValueError("no results to summarize")
    pos = np.array([r["val_position_rmse"] for r in results], dtype=float)
    vel = np.array([r["val_velocity_rmse"] for r in results], dtype=float)
    return MultiSeedSummary(
        n_seeds=len(results),
        seeds=[r["seed"] for r in results],
        val_position_rmse_mean=float(pos.mean()),
        val_position_rmse_std=float(pos.std()),
        val_position_rmse_min=float(pos.min()),
        val_position_rmse_max=float(pos.max()),
        val_velocity_rmse_mean=float(vel.mean()),
        val_velocity_rmse_std=float(vel.std()),
        best_epoch_values=[r["best_epoch"] for r in results],
        wall_s_total=float(sum(r["wall_s"] for r in results)),
        any_catastrophic=any(r["catastrophic"] for r in results),
        norm_overflow_counts=[int(r.get("norm_overflow_count", 0)) for r in results],
        element_nonfinite_counts=[int(r.get("element_nonfinite_count", 0)) for r in results],
        catastrophic_seeds=[r["seed"] for r in results if r["catastrophic"]],
        unstable_seeds=[r["seed"] for r in results if r.get("training_unstable", False)],
    )
