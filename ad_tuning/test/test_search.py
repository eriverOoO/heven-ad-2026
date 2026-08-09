from contextlib import contextmanager

import optuna
import pytest

import ad_tuning.search as search_module
from ad_tuning.search import (
    DWA_PARAMETER_NAMES,
    DWA_WARM_STARTS,
    PROFILE_STANLEY_PARAMETER_NAMES,
    PROFILE_STANLEY_WARM_STARTS,
    build_experiment_fingerprint,
    create_dwa_study,
    create_profile_stanley_study,
    derive_worker_seed,
    dwa_constraints,
    initialize_profile_stanley_study,
    normalize_storage_url,
    profile_stanley_constraints,
    redact_storage_url,
    resolve_worker_id,
    runtime_artifact_manifest,
    suggest_dwa_parameters,
    suggest_profile_stanley_parameters,
)


def test_search_tunes_only_profile_stanley_response_parameters(tmp_path):
    study = create_profile_stanley_study(tmp_path, "route-a", seed=7)
    trial = study.ask()
    parameters = suggest_profile_stanley_parameters(trial)

    assert tuple(parameters) == PROFILE_STANLEY_PARAMETER_NAMES
    assert parameters == PROFILE_STANLEY_WARM_STARTS[0]
    assert all(name.startswith("profile_stanley.") for name in parameters)
    assert "profile_stanley.target_speed_mps" not in parameters
    assert "profile_stanley.maximum_speed_mps" not in parameters
    assert "profile_stanley.lateral_acceleration_mps2" not in parameters
    assert "profile_stanley.brake_pid.ki" not in parameters


def test_search_tunes_only_dwa_score_ratios_and_local_speed_response(
    tmp_path,
):
    study = create_dwa_study(tmp_path, "obstacle-route-a", seed=7)
    trial = study.ask()
    parameters = suggest_dwa_parameters(trial)

    assert tuple(parameters) == DWA_PARAMETER_NAMES
    assert parameters == DWA_WARM_STARTS[0]
    assert all(name.startswith("dwa.") for name in parameters)
    assert "dwa.maximum_deceleration_mps2" not in parameters
    assert "dwa.emergency_deceleration_mps2" not in parameters
    assert "dwa.maximum_lateral_acceleration_mps2" not in parameters
    assert "dwa.maximum_path_distance_m" not in parameters
    assert "dwa.footprint.occupied_threshold" not in parameters


def test_dwa_study_uses_an_independent_sqlite_database(tmp_path):
    study = create_dwa_study(tmp_path, "obstacle-route-a", seed=11)

    assert study.study_name.startswith("dwa_ros_morai_v4_")
    assert (tmp_path / "dwa_optuna.sqlite3").is_file()
    assert not (tmp_path / "profile_stanley_optuna.sqlite3").exists()


def test_optuna_study_persists_and_resumes_from_sqlite(tmp_path):
    first = create_profile_stanley_study(tmp_path, "route-a", seed=11)
    trial = first.ask()
    suggest_profile_stanley_parameters(trial)
    first.tell(trial, 42.0)

    resumed = create_profile_stanley_study(tmp_path, "route-a", seed=11)
    complete = [
        item
        for item in resumed.trials
        if item.state is optuna.trial.TrialState.COMPLETE
    ]
    assert len(complete) == 1
    assert complete[0].value == 42.0


def test_route_digest_isolates_studies_in_the_same_database(tmp_path):
    first = create_profile_stanley_study(tmp_path, "route-a", seed=13)
    trial = first.ask()
    suggest_profile_stanley_parameters(trial)
    first.tell(trial, 1.0)

    second = create_profile_stanley_study(tmp_path, "route-b", seed=13)
    assert second.study_name != first.study_name
    assert not [
        item
        for item in second.trials
        if item.state is optuna.trial.TrialState.COMPLETE
    ]


def test_storage_url_defaults_to_sqlite_and_normalizes_postgres(tmp_path):
    sqlite = normalize_storage_url(tmp_path)
    assert sqlite.startswith("sqlite:///")
    assert sqlite.endswith("profile_stanley_optuna.sqlite3")
    postgres = normalize_storage_url(
        tmp_path, "postgresql://user:secret@db:5432/tuning"
    )
    assert postgres == (
        "postgresql+psycopg://user:secret@db:5432/tuning"
    )
    assert "secret" not in redact_storage_url(postgres)


def test_worker_identity_changes_sampler_seed_and_is_file_safe(monkeypatch):
    monkeypatch.setenv("AD_TUNING_WORKER_ID", "lab pc/2")
    worker = resolve_worker_id()

    assert worker == "lab-pc-2"
    assert derive_worker_seed(7, "worker-a") != derive_worker_seed(
        7, "worker-b"
    )


def test_experiment_fingerprint_is_stable_and_sensitive():
    first, canonical = build_experiment_fingerprint(
        {"route": "abc", "objective": {"cte": 50.0}}
    )
    reordered, _ = build_experiment_fingerprint(
        {"objective": {"cte": 50.0}, "route": "abc"}
    )
    changed, _ = build_experiment_fingerprint(
        {"route": "abc", "objective": {"cte": 51.0}}
    )

    assert first == reordered
    assert first != changed
    assert canonical == {
        "objective": {"cte": 50.0},
        "route": "abc",
    }


def test_study_tags_experiment_and_avoids_duplicate_warm_starts(tmp_path):
    metadata = {"route": "abc", "weather": "clear"}
    first = create_profile_stanley_study(
        tmp_path,
        "fingerprint",
        seed=1,
        worker_id="worker-a",
        experiment_metadata=metadata,
    )
    second = create_profile_stanley_study(
        tmp_path,
        "fingerprint",
        seed=1,
        worker_id="worker-b",
        experiment_metadata=metadata,
    )

    assert first.user_attrs["experiment_fingerprint"] == "fingerprint"
    assert first.user_attrs["experiment_metadata"] == metadata
    assert first.user_attrs["warm_start_initialization_complete"] is True
    assert first.user_attrs["warm_start_count"] == 3
    assert len(second.trials) == len(PROFILE_STANLEY_WARM_STARTS)


def compatible_metadata(objective_weight, route_digest="route-digest"):
    return {
        "search_space_version": "profile_stanley_ros_morai_v4",
        "route_digest": route_digest,
        "route_length_m": 800.0,
        "course_length_m": 795.0,
        "fixed_parameters": {"profile_stanley.target_speed_mps": 16.25},
        "stack_conditions": {
            "path_tracking.backend": "profile_stanley",
            "perception.enabled": False,
        },
        "objective": {"front_cte_squared_weight": objective_weight},
        "context": {
            "scenario": "competition",
            "weather": "clear",
            "morai_version": "S4.251001",
            "vehicle_profile_id": "ioniq-profile",
            "timing_source": "device_stamp",
            "code_revision": f"metric-{objective_weight}",
            "runtime_source_digest": f"digest-{objective_weight}",
        },
    }


def complete_feasible_trial(study, cost):
    trial = study.ask()
    parameters = suggest_profile_stanley_parameters(trial)
    trial.set_user_attr("feasible", True)
    trial.set_user_attr("cost", cost)
    study.tell(trial, cost)
    return trial.number, parameters


def test_new_metric_inherits_best_seed_from_mature_compatible_study(
    tmp_path,
):
    previous = create_profile_stanley_study(
        tmp_path,
        "previous-fingerprint",
        seed=1,
        experiment_metadata=compatible_metadata(150.0),
    )
    complete_feasible_trial(previous, 20.0)
    best_number, best_parameters = complete_feasible_trial(previous, 10.0)

    current = create_profile_stanley_study(
        tmp_path,
        "current-fingerprint",
        seed=1,
        experiment_metadata=compatible_metadata(30.0),
        inherit_minimum_complete_trials=2,
        inherit_top_k=1,
    )
    inherited_trial = current.ask()
    inherited_parameters = suggest_profile_stanley_parameters(inherited_trial)

    assert inherited_parameters == best_parameters
    assert current.user_attrs["inherited_warm_start_count"] == 1
    assert current.user_attrs["inherited_from_study"] == previous.study_name
    assert current.user_attrs["inherited_source_complete_trials"] == 2
    assert current.user_attrs["inherited_source_trial_numbers"] == [
        best_number
    ]


def test_small_or_unrelated_study_does_not_seed_new_metric(tmp_path):
    previous = create_profile_stanley_study(
        tmp_path,
        "previous-fingerprint",
        seed=1,
        experiment_metadata=compatible_metadata(
            150.0, route_digest="old-route"
        ),
    )
    complete_feasible_trial(previous, 1.0)

    current = create_profile_stanley_study(
        tmp_path,
        "current-fingerprint",
        seed=1,
        experiment_metadata=compatible_metadata(
            30.0, route_digest="new-route"
        ),
        inherit_minimum_complete_trials=1,
        inherit_top_k=5,
    )
    trial = current.ask()
    parameters = suggest_profile_stanley_parameters(trial)

    assert parameters == PROFILE_STANLEY_WARM_STARTS[0]
    assert current.user_attrs["inherited_warm_start_count"] == 0
    assert current.user_attrs["inherited_from_study"] == ""


def test_compatible_study_must_reach_minimum_complete_trial_count(tmp_path):
    previous = create_profile_stanley_study(
        tmp_path,
        "previous-fingerprint",
        seed=1,
        experiment_metadata=compatible_metadata(150.0),
    )
    complete_feasible_trial(previous, 1.0)

    current = create_profile_stanley_study(
        tmp_path,
        "current-fingerprint",
        seed=1,
        experiment_metadata=compatible_metadata(30.0),
        inherit_minimum_complete_trials=2,
        inherit_top_k=5,
    )
    trial = current.ask()
    parameters = suggest_profile_stanley_parameters(trial)

    assert parameters == PROFILE_STANLEY_WARM_STARTS[0]
    assert current.user_attrs["inherited_warm_start_count"] == 0


def test_any_postgres_worker_can_atomically_initialize_warm_starts(
    monkeypatch,
):
    study = optuna.create_study()

    @contextmanager
    def fake_lock(storage_url, study_name):
        assert storage_url.startswith("postgresql+psycopg://")
        assert study_name == study.study_name
        yield

    monkeypatch.setattr(search_module, "_warm_start_lock", fake_lock)
    initialize_profile_stanley_study(
        study,
        storage_url="postgresql+psycopg://db/tuning",
        worker_id="heven-left",
        single_worker_storage=False,
    )
    initialize_profile_stanley_study(
        study,
        storage_url="postgresql+psycopg://db/tuning",
        worker_id="heven-laptop",
        single_worker_storage=False,
    )

    assert study.user_attrs["warm_start_coordinator"] == "heven-left"
    assert len(study.trials) == len(PROFILE_STANLEY_WARM_STARTS)


def test_multiworker_sqlite_warm_start_is_rejected(tmp_path):
    study = optuna.create_study()

    with pytest.raises(ValueError, match="require PostgreSQL"):
        initialize_profile_stanley_study(
            study,
            storage_url=f"sqlite:///{tmp_path / 'study.db'}",
            worker_id="worker-a",
            single_worker_storage=False,
        )


def test_constraints_are_read_from_real_trial_attributes():
    trial = optuna.trial.create_trial(
        value=1.0,
        user_attrs={"constraint_values": [0.0, -0.2]},
    )

    assert profile_stanley_constraints(trial) == (0.0, -0.2)
    assert dwa_constraints(trial) == (0.0, -0.2)


def test_runtime_artifact_manifest_hashes_exact_file_contents(tmp_path):
    artifact = tmp_path / "planner.bin"
    artifact.write_bytes(b"planner-v3")

    manifest = runtime_artifact_manifest({"planner": artifact})

    assert manifest["planner"]["size_bytes"] == len(b"planner-v3")
    assert len(manifest["planner"]["sha256"]) == 64
