from __future__ import annotations

import math


def perception_epoch_is_ready(
    *,
    reset_wall_s: float,
    now_wall_s: float,
    prediction_received_wall_s: float,
    occupancy_received_wall_s: float,
    baseline_prediction_sequence: int,
    prediction_sequence: int,
    minimum_prediction_samples: int,
    first_post_reset_sim_s: float | None,
    current_sim_s: float | None,
    settle_sim_s: float,
    stale_timeout_s: float,
    require_occupancy_grid: bool,
) -> bool:
    """Check that perception belongs to a settled post-reset simulator epoch."""
    if minimum_prediction_samples <= 0:
        raise ValueError("minimum_prediction_samples must be positive")
    if not math.isfinite(settle_sim_s) or settle_sim_s < 0.0:
        raise ValueError("settle_sim_s must be finite and nonnegative")
    if not math.isfinite(stale_timeout_s) or stale_timeout_s <= 0.0:
        raise ValueError("stale_timeout_s must be finite and positive")

    timestamps = (
        reset_wall_s,
        now_wall_s,
        prediction_received_wall_s,
    )
    if not all(math.isfinite(value) for value in timestamps):
        return False
    if (
        first_post_reset_sim_s is None
        or current_sim_s is None
        or not math.isfinite(first_post_reset_sim_s)
        or not math.isfinite(current_sim_s)
        or current_sim_s < first_post_reset_sim_s
    ):
        return False
    if prediction_sequence - baseline_prediction_sequence < (
        minimum_prediction_samples
    ):
        return False
    if (
        prediction_received_wall_s <= reset_wall_s
        or now_wall_s - prediction_received_wall_s > stale_timeout_s
    ):
        return False
    if require_occupancy_grid and (
        not math.isfinite(occupancy_received_wall_s)
        or occupancy_received_wall_s <= reset_wall_s
        or now_wall_s - occupancy_received_wall_s > stale_timeout_s
    ):
        return False
    return current_sim_s - first_post_reset_sim_s >= settle_sim_s
