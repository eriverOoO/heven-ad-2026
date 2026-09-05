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


def summarize_multi_seed_results(results: list[dict[str, Any]]) -> MultiSeedSummary:
    """``results``: one dict per seed, each with at minimum
    ``seed``, ``val_position_rmse``, ``val_velocity_rmse``, ``best_epoch``,
    ``wall_s``, ``catastrophic`` keys (the shape this task's own
    calibration/confirmation scripts produce)."""
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
    )
