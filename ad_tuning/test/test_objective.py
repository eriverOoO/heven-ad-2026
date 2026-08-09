import pytest

from ad_tuning.objective import (
    ObjectiveConfig,
    RunMetrics,
    evaluate_trial,
)


def metrics(**overrides):
    values = {
        "completed": True,
        "elapsed_s": 40.0,
        "progress_m": 400.0,
        "mean_cte_sq_m2": 0.01,
        "distance_cte_mse_m2": 0.01,
        "time_cte_mse_m2": 0.01,
        "max_cte_m": 0.2,
        "rear_mean_cte_sq_m2": 0.01,
        "rear_distance_cte_mse_m2": 0.01,
        "rear_time_cte_mse_m2": 0.01,
        "rear_max_cte_m": 0.2,
        "overspeed_s": 0.0,
        "target_overspeed_sq_integral": 0.0,
        "unnecessary_brake_sq_integral": 0.0,
        "brake_saturation_s": 0.0,
    }
    values.update(overrides)
    return RunMetrics(**values)


def test_completed_trial_minimizes_time_cte_and_overspeed():
    config = ObjectiveConfig()
    evaluation = evaluate_trial(metrics(), 400.0, config)
    assert evaluation.feasible
    assert evaluation.cost == pytest.approx(40.6)
    off_center_but_inside = evaluate_trial(
        metrics(
            distance_cte_mse_m2=0.49,
            rear_distance_cte_mse_m2=0.49,
            max_cte_m=0.7,
            rear_max_cte_m=0.7,
        ),
        400.0,
        config,
    )
    assert off_center_but_inside.feasible
    assert off_center_but_inside.cost == pytest.approx(69.4)
    assert (
        evaluate_trial(metrics(overspeed_s=1.0), 400.0, config).cost
        > evaluation.cost
    )


def test_objective_uses_distance_weighted_cte_not_sample_jitter():
    config = ObjectiveConfig(
        front_cte_squared_weight=50.0,
        rear_cte_squared_weight=30.0,
    )
    baseline = evaluate_trial(metrics(), 400.0, config)
    changed_sample_mean = evaluate_trial(
        metrics(mean_cte_sq_m2=10.0, rear_mean_cte_sq_m2=10.0),
        400.0,
        config,
    )
    changed_front_distance_mean = evaluate_trial(
        metrics(distance_cte_mse_m2=0.02), 400.0, config
    )
    changed_rear_distance_mean = evaluate_trial(
        metrics(rear_distance_cte_mse_m2=0.02), 400.0, config
    )

    assert changed_sample_mean.cost == pytest.approx(baseline.cost)
    assert changed_front_distance_mean.cost == pytest.approx(
        baseline.cost + 0.5
    )
    assert changed_rear_distance_mean.cost == pytest.approx(
        baseline.cost + 0.3
    )


def test_legacy_runner_without_distance_metric_falls_back_to_sample_mean():
    config = ObjectiveConfig(
        front_cte_squared_weight=50.0,
        rear_cte_squared_weight=30.0,
    )

    evaluation = evaluate_trial(
        metrics(
            distance_cte_mse_m2=None,
            mean_cte_sq_m2=0.02,
            rear_distance_cte_mse_m2=None,
            rear_mean_cte_sq_m2=0.03,
        ),
        400.0,
        config,
    )

    assert evaluation.cost == pytest.approx(41.9)


def test_overspeed_penalty_matches_immediate_and_three_second_rule():
    config = ObjectiveConfig(
        elapsed_time_weight=0.0,
        front_cte_squared_weight=0.0,
        rear_cte_squared_weight=0.0,
        target_overspeed_squared_weight=0.0,
        unnecessary_brake_squared_weight=0.0,
        brake_saturation_time_weight=0.0,
    )
    below_interval = evaluate_trial(
        metrics(elapsed_s=40.0, overspeed_s=2.999), 400.0, config
    )
    at_interval = evaluate_trial(
        metrics(elapsed_s=40.0, overspeed_s=3.0), 400.0, config
    )
    assert below_interval.cost == pytest.approx(15.0)
    assert at_interval.cost == pytest.approx(30.0)


def test_brake_objective_penalizes_underbraking_and_unnecessary_braking():
    config = ObjectiveConfig()
    baseline = evaluate_trial(metrics(), 400.0, config)
    underbraking = evaluate_trial(
        metrics(target_overspeed_sq_integral=2.0), 400.0, config
    )
    overbraking = evaluate_trial(
        metrics(
            unnecessary_brake_sq_integral=1.0,
            brake_saturation_s=0.5,
        ),
        400.0,
        config,
    )
    assert underbraking.cost == pytest.approx(baseline.cost + 2.0)
    assert overbraking.cost == pytest.approx(baseline.cost + 5.5)


def test_incomplete_trial_is_never_feasible_and_rewards_progress():
    config = ObjectiveConfig()
    near = evaluate_trial(
        metrics(completed=False, progress_m=390.0), 400.0, config
    )
    far = evaluate_trial(
        metrics(completed=False, progress_m=100.0), 400.0, config
    )
    assert not near.feasible
    assert near.cost < far.cost


def test_cte_constraint_dominates_raw_cost_selection():
    config = ObjectiveConfig(maximum_cte_m=1.0)
    feasible = evaluate_trial(metrics(elapsed_s=100.0), 400.0, config)
    front_infeasible = evaluate_trial(
        metrics(elapsed_s=1.0, max_cte_m=2.0), 400.0, config
    )
    rear_infeasible = evaluate_trial(
        metrics(elapsed_s=1.0, rear_max_cte_m=2.0), 400.0, config
    )
    assert min(
        (front_infeasible, feasible), key=lambda value: value.selection_key
    ) is feasible
    assert min(
        (rear_infeasible, feasible), key=lambda value: value.selection_key
    ) is feasible


def test_collision_is_an_absolute_constraint_and_large_cost():
    config = ObjectiveConfig(collision_penalty=5000.0)
    baseline = evaluate_trial(metrics(), 400.0, config)
    collision = evaluate_trial(
        metrics(collision_count=1), 400.0, config
    )

    assert baseline.feasible
    assert not collision.feasible
    assert collision.cost == pytest.approx(baseline.cost + 5000.0)


def test_dwa_study_requires_real_local_planner_activation():
    config = ObjectiveConfig(minimum_local_planner_active_s=1.0)

    inactive = evaluate_trial(
        metrics(local_planner_active_s=0.0), 400.0, config
    )
    active = evaluate_trial(
        metrics(local_planner_active_s=1.0), 400.0, config
    )

    assert not inactive.feasible
    assert active.feasible
