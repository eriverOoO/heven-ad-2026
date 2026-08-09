from __future__ import annotations

from dataclasses import asdict, dataclass

import optuna

from ad_tuning.objective import ObjectiveConfig, TrialEvaluation, evaluate_trial
from ad_tuning.runner import TrialRunner
from ad_tuning.targets import TuningTarget, get_target


@dataclass(frozen=True)
class StudyResult:
    target: TuningTarget
    study: optuna.Study
    winner: optuna.trial.FrozenTrial


def _trial_evaluation(trial: optuna.trial.FrozenTrial) -> TrialEvaluation:
    try:
        feasible = trial.user_attrs["feasible"]
        cost = trial.user_attrs["cost"]
    except KeyError as error:
        raise ValueError("completed trial has no recorded evaluation") from error
    if not isinstance(feasible, bool) or not isinstance(cost, (int, float)):
        raise ValueError("completed trial has an invalid recorded evaluation")
    return TrialEvaluation(feasible=feasible, cost=float(cost))


def _validate_trial_targets(
    trials: list[optuna.trial.FrozenTrial],
    target_id: str,
) -> None:
    for trial in trials:
        trial_target = trial.user_attrs.get("target_id")
        if trial_target != target_id:
            raise ValueError(
                "trial target mismatch for completed trial "
                f"{trial.number}: {trial_target!r} != {target_id!r}"
            )


def select_best_trial(study: optuna.Study) -> optuna.trial.FrozenTrial:
    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed:
        raise ValueError("study has no completed evaluations")
    stored_target = study.user_attrs.get("target_id")
    if stored_target is not None:
        _validate_trial_targets(completed, stored_target)
    return min(completed, key=lambda trial: _trial_evaluation(trial).selection_key)


def run_study(
    *,
    target: str | TuningTarget,
    runner: TrialRunner,
    study_name: str,
    storage: str | None,
    n_trials: int,
    objective_config: ObjectiveConfig,
    course_length_m: float,
    sampler: optuna.samplers.BaseSampler | None = None,
) -> StudyResult:
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    selected_target = get_target(target) if isinstance(target, str) else target
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=sampler,
    )
    stored_target = study.user_attrs.get("target_id")
    if stored_target is not None and stored_target != selected_target.identifier:
        raise ValueError(
            "stored target differs from requested target: "
            f"{stored_target} != {selected_target.identifier}"
        )
    if stored_target is None:
        if study.trials:
            raise ValueError("cannot claim an untagged existing study with trials")
        study.set_user_attr("target_id", selected_target.identifier)
    else:
        completed = [
            trial
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        _validate_trial_targets(completed, selected_target.identifier)

    def objective(trial: optuna.Trial) -> float:
        parameters = selected_target.suggest(trial)
        metrics = runner.run_trial(parameters)
        evaluation = evaluate_trial(metrics, course_length_m, objective_config)
        trial.set_user_attr("target_id", selected_target.identifier)
        trial.set_user_attr("metrics", asdict(metrics))
        trial.set_user_attr("feasible", evaluation.feasible)
        trial.set_user_attr("cost", evaluation.cost)
        return evaluation.cost

    study.optimize(objective, n_trials=n_trials)
    return StudyResult(
        target=selected_target,
        study=study,
        winner=select_best_trial(study),
    )
