import json

import optuna
import pytest

from ad_tuning.objective import ObjectiveConfig, RunMetrics
from ad_tuning.study import run_study, select_best_trial


OBJECTIVE = ObjectiveConfig(
    maximum_cte_m=1.0,
    elapsed_time_weight=0.3,
    front_cte_squared_weight=5.0,
    rear_cte_squared_weight=5.0,
    competition_overspeed_penalty_s=5.0,
    competition_overspeed_interval_s=3.0,
    incomplete_penalty=2000.0,
    incomplete_cte_weight=1.0,
)


def metrics(**overrides):
    values = {
        "completed": True,
        "elapsed_s": 50.0,
        "progress_m": 100.0,
        "mean_cte_sq_m2": 0.1,
        "distance_cte_mse_m2": 0.1,
        "time_cte_mse_m2": 0.1,
        "max_cte_m": 0.5,
        "rear_mean_cte_sq_m2": 0.1,
        "rear_distance_cte_mse_m2": 0.1,
        "rear_time_cte_mse_m2": 0.1,
        "rear_max_cte_m": 0.5,
        "overspeed_s": 0.0,
    }
    values.update(overrides)
    return RunMetrics(**values)


class RecordingRunner:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.parameters = []

    def run_trial(self, parameters):
        self.parameters.append(dict(parameters))
        return next(self.outcomes)


def test_run_study_passes_only_selected_target_parameters():
    runner = RecordingRunner([metrics()])

    run_study(
        target="control/profile_stanley",
        runner=runner,
        study_name="profile-parameters",
        storage=None,
        n_trials=1,
        objective_config=OBJECTIVE,
        course_length_m=100.0,
        sampler=optuna.samplers.RandomSampler(seed=3),
    )

    assert len(runner.parameters) == 1
    assert runner.parameters[0]
    assert all(name.startswith("profile_stanley.") for name in runner.parameters[0])


def test_resume_rejects_a_different_target_for_the_same_study(tmp_path):
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    run_study(
        target="control/stanley",
        runner=RecordingRunner([metrics()]),
        study_name="shared-study",
        storage=storage,
        n_trials=1,
        objective_config=OBJECTIVE,
        course_length_m=100.0,
    )

    with pytest.raises(ValueError, match="stored target"):
        run_study(
            target="planning/dwa",
            runner=RecordingRunner([metrics()]),
            study_name="shared-study",
            storage=storage,
            n_trials=1,
            objective_config=OBJECTIVE,
            course_length_m=100.0,
        )


def test_resume_rejects_an_existing_untagged_study():
    storage = optuna.storages.InMemoryStorage()
    legacy = optuna.create_study(
        study_name="legacy-study",
        storage=storage,
        direction="minimize",
    )
    _add_completed_trial(legacy, feasible=True, cost=0.1)
    runner = RecordingRunner([metrics()])

    with pytest.raises(ValueError, match="untagged existing study"):
        run_study(
            target="control/stanley",
            runner=runner,
            study_name="legacy-study",
            storage=storage,
            n_trials=1,
            objective_config=OBJECTIVE,
            course_length_m=100.0,
        )

    assert runner.parameters == []


def test_trial_attributes_record_metrics_feasibility_and_target():
    result = run_study(
        target="control/stanley",
        runner=RecordingRunner([metrics()]),
        study_name="attributes",
        storage=None,
        n_trials=1,
        objective_config=OBJECTIVE,
        course_length_m=100.0,
    )
    trial = result.study.trials[0]

    assert trial.user_attrs["target_id"] == "control/stanley"
    assert trial.user_attrs["feasible"] is True
    assert trial.user_attrs["cost"] == pytest.approx(16.0)
    assert json.loads(json.dumps(trial.user_attrs["metrics"])) == {
        "completed": True,
        "elapsed_s": 50.0,
        "progress_m": 100.0,
        "mean_cte_sq_m2": 0.1,
        "distance_cte_mse_m2": 0.1,
        "time_cte_mse_m2": 0.1,
        "max_cte_m": 0.5,
        "rear_mean_cte_sq_m2": 0.1,
        "rear_distance_cte_mse_m2": 0.1,
        "rear_time_cte_mse_m2": 0.1,
        "rear_max_cte_m": 0.5,
        "overspeed_s": 0.0,
        "wall_elapsed_s": 0.0,
        "real_time_factor": 0.0,
        "target_overspeed_sq_integral": 0.0,
        "unnecessary_brake_sq_integral": 0.0,
            "brake_saturation_s": 0.0,
            "throttle_saturation_s": 0.0,
            "collision_count": 0,
            "local_planner_active_s": 0.0,
            "local_planner_failure_s": 0.0,
            "stopped_s": 0.0,
            "reset_failed": False,
            "disconnected": False,
            "aborted": False,
        "reason": "",
    }


def _add_completed_trial(study, *, feasible, cost, target_id=None):
    trial = study.ask()
    trial.set_user_attr("feasible", feasible)
    trial.set_user_attr("cost", cost)
    if target_id is not None:
        trial.set_user_attr("target_id", target_id)
    study.tell(trial, cost)
    return trial.number


def test_selection_prefers_feasible_trial_with_higher_raw_cost():
    study = optuna.create_study(direction="minimize")
    infeasible_number = _add_completed_trial(study, feasible=False, cost=1.0)
    feasible_number = _add_completed_trial(study, feasible=True, cost=100.0)

    assert study.best_trial.number == infeasible_number
    assert select_best_trial(study).number == feasible_number


def test_selection_uses_cost_between_trials_with_equal_feasibility():
    study = optuna.create_study(direction="minimize")
    slower_number = _add_completed_trial(study, feasible=True, cost=100.0)
    faster_number = _add_completed_trial(study, feasible=True, cost=10.0)

    assert select_best_trial(study).number == faster_number
    assert slower_number != faster_number


@pytest.mark.parametrize("trial_target", [None, "planning/dwa"])
def test_tagged_selection_rejects_a_mixed_completed_trial(trial_target):
    study = optuna.create_study(direction="minimize")
    study.set_user_attr("target_id", "control/stanley")
    _add_completed_trial(
        study,
        feasible=True,
        cost=100.0,
        target_id="control/stanley",
    )
    _add_completed_trial(
        study,
        feasible=True,
        cost=1.0,
        target_id=trial_target,
    )

    with pytest.raises(ValueError, match="trial target"):
        select_best_trial(study)


def test_selection_rejects_a_study_without_completed_evaluations():
    study = optuna.create_study(direction="minimize")

    with pytest.raises(ValueError, match="completed evaluations"):
        select_best_trial(study)
