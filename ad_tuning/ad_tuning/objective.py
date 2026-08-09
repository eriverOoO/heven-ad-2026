from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ObjectiveConfig:
    maximum_cte_m: float = 0.8
    elapsed_time_weight: float = 1.0
    front_cte_squared_weight: float = 30.0
    rear_cte_squared_weight: float = 30.0
    competition_overspeed_penalty_s: float = 15.0
    competition_overspeed_interval_s: float = 3.0
    target_overspeed_squared_weight: float = 1.0
    unnecessary_brake_squared_weight: float = 5.0
    brake_saturation_time_weight: float = 1.0
    maximum_collision_count: int = 0
    minimum_local_planner_active_s: float = 0.0
    collision_penalty: float = 5000.0
    local_planner_failure_time_weight: float = 0.0
    incomplete_penalty: float = 2000.0
    incomplete_cte_weight: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.maximum_cte_m,
            self.elapsed_time_weight,
            self.front_cte_squared_weight,
            self.rear_cte_squared_weight,
            self.competition_overspeed_penalty_s,
            self.competition_overspeed_interval_s,
            self.target_overspeed_squared_weight,
            self.unnecessary_brake_squared_weight,
            self.brake_saturation_time_weight,
            self.minimum_local_planner_active_s,
            self.collision_penalty,
            self.local_planner_failure_time_weight,
            self.incomplete_penalty,
            self.incomplete_cte_weight,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("objective configuration must be finite")
        if self.maximum_cte_m <= 0.0:
            raise ValueError("maximum_cte_m must be positive")
        if self.maximum_collision_count < 0:
            raise ValueError("maximum_collision_count must be nonnegative")
        if self.competition_overspeed_interval_s <= 0.0:
            raise ValueError(
                "competition_overspeed_interval_s must be positive"
            )
        if any(
            value < 0.0
            for value in (
                self.elapsed_time_weight,
                self.front_cte_squared_weight,
                self.rear_cte_squared_weight,
                self.competition_overspeed_penalty_s,
                self.target_overspeed_squared_weight,
                self.unnecessary_brake_squared_weight,
                self.brake_saturation_time_weight,
                self.minimum_local_planner_active_s,
                self.collision_penalty,
                self.local_planner_failure_time_weight,
                self.incomplete_cte_weight,
            )
        ):
            raise ValueError("objective weights must be nonnegative")
        if self.incomplete_penalty <= 0.0:
            raise ValueError("incomplete_penalty must be positive")


@dataclass(frozen=True)
class RunMetrics:
    completed: bool
    elapsed_s: float
    progress_m: float
    mean_cte_sq_m2: float
    max_cte_m: float
    rear_mean_cte_sq_m2: float
    rear_max_cte_m: float
    overspeed_s: float
    distance_cte_mse_m2: float | None = None
    time_cte_mse_m2: float | None = None
    rear_distance_cte_mse_m2: float | None = None
    rear_time_cte_mse_m2: float | None = None
    wall_elapsed_s: float = 0.0
    real_time_factor: float = 0.0
    target_overspeed_sq_integral: float = 0.0
    unnecessary_brake_sq_integral: float = 0.0
    brake_saturation_s: float = 0.0
    throttle_saturation_s: float = 0.0
    collision_count: int = 0
    local_planner_active_s: float = 0.0
    local_planner_failure_s: float = 0.0
    stopped_s: float = 0.0
    reset_failed: bool = False
    disconnected: bool = False
    aborted: bool = False
    reason: str = ""


@dataclass(frozen=True)
class TrialEvaluation:
    feasible: bool
    cost: float

    @property
    def selection_key(self) -> tuple[int, float]:
        return (0 if self.feasible else 1, self.cost)


def _validate_metrics(metrics: RunMetrics) -> None:
    values = (
        metrics.elapsed_s,
        metrics.progress_m,
        metrics.mean_cte_sq_m2,
        metrics.max_cte_m,
        metrics.rear_mean_cte_sq_m2,
        metrics.rear_max_cte_m,
        metrics.overspeed_s,
        metrics.wall_elapsed_s,
        metrics.real_time_factor,
        metrics.target_overspeed_sq_integral,
        metrics.unnecessary_brake_sq_integral,
        metrics.brake_saturation_s,
        metrics.throttle_saturation_s,
        metrics.local_planner_active_s,
        metrics.local_planner_failure_s,
        metrics.stopped_s,
    )
    optional_values = tuple(
        value
        for value in (
            metrics.distance_cte_mse_m2,
            metrics.time_cte_mse_m2,
            metrics.rear_distance_cte_mse_m2,
            metrics.rear_time_cte_mse_m2,
        )
        if value is not None
    )
    if not all(math.isfinite(value) for value in values + optional_values):
        raise ValueError("run metrics must be finite")
    if any(value < 0.0 for value in values + optional_values):
        raise ValueError("run metrics must be nonnegative")
    if metrics.collision_count < 0:
        raise ValueError("collision_count must be nonnegative")


def is_feasible(metrics: RunMetrics, config: ObjectiveConfig) -> bool:
    try:
        _validate_metrics(metrics)
    except ValueError:
        return False
    return (
        metrics.completed
        and not metrics.reset_failed
        and not metrics.disconnected
        and not metrics.aborted
        and metrics.max_cte_m <= config.maximum_cte_m
        and metrics.rear_max_cte_m <= config.maximum_cte_m
        and metrics.collision_count <= config.maximum_collision_count
        and metrics.local_planner_active_s
        >= config.minimum_local_planner_active_s
    )


def evaluate_trial(
    metrics: RunMetrics,
    course_length_m: float,
    config: ObjectiveConfig,
) -> TrialEvaluation:
    _validate_metrics(metrics)
    if not math.isfinite(course_length_m) or course_length_m <= 0.0:
        raise ValueError("course_length_m must be finite and positive")
    overspeed_penalty = 0.0
    if metrics.overspeed_s > 0.0:
        intervals = math.floor(
            metrics.overspeed_s / config.competition_overspeed_interval_s
        )
        overspeed_penalty = (
            config.competition_overspeed_penalty_s * (1 + intervals)
        )
    control_cost = (
        overspeed_penalty
        + config.target_overspeed_squared_weight
        * metrics.target_overspeed_sq_integral
        + config.unnecessary_brake_squared_weight
        * metrics.unnecessary_brake_sq_integral
        + config.brake_saturation_time_weight
        * metrics.brake_saturation_s
        + config.collision_penalty * metrics.collision_count
        + config.local_planner_failure_time_weight
        * metrics.local_planner_failure_s
    )
    if metrics.completed:
        front_cte_mse = (
            metrics.mean_cte_sq_m2
            if metrics.distance_cte_mse_m2 is None
            else metrics.distance_cte_mse_m2
        )
        rear_cte_mse = (
            metrics.rear_mean_cte_sq_m2
            if metrics.rear_distance_cte_mse_m2 is None
            else metrics.rear_distance_cte_mse_m2
        )
        cost = (
            config.elapsed_time_weight * metrics.elapsed_s
            + config.front_cte_squared_weight * front_cte_mse
            + config.rear_cte_squared_weight * rear_cte_mse
            + control_cost
        )
    else:
        cost = (
            config.incomplete_penalty
            + max(0.0, course_length_m - metrics.progress_m)
            + config.incomplete_cte_weight
            * max(metrics.max_cte_m, metrics.rear_max_cte_m)
            + control_cost
        )
    return TrialEvaluation(is_feasible(metrics, config), float(cost))
