import csv
import json

import yaml

from ad_tuning.storage import TrialStorage, is_reusable_trial


def test_best_yaml_is_directly_usable_as_a_ros_parameter_file(tmp_path):
    storage = TrialStorage(tmp_path, "profile_stanley")
    row = {
        "trial": 3,
        "feasible": True,
        "cost": 42.5,
        "route_digest": "abc",
        "metrics": {"completed": True},
        "parameters": {"profile_stanley.lookahead_time_s": 0.16},
    }
    storage.write_best(
        row, {"profile_stanley.lateral_acceleration_mps2": 6.0}
    )
    document = yaml.safe_load(storage.best_path.read_text(encoding="utf-8"))
    assert document["ad_planner"]["ros__parameters"] == {
        "profile_stanley.lookahead_time_s": 0.16,
        "profile_stanley.lateral_acceleration_mps2": 6.0,
    }
    assert storage.best_path.name == "best_profile_stanley_optuna.yaml"
    metadata = json.loads(
        storage.best_metadata_path.read_text(encoding="utf-8")
    )
    assert metadata["trial"] == 3


def test_only_clean_same_route_trials_are_reusable():
    row = {
        "status": "completed",
        "route_digest": "route-a",
        "search_space_version": "v1",
        "metrics": {
            "aborted": False,
            "reset_failed": False,
            "disconnected": False,
        },
    }
    assert is_reusable_trial(row, "route-a", "v1")
    row["metrics"]["reset_failed"] = True
    assert not is_reusable_trial(row, "route-a", "v1")


def test_worker_artifacts_are_isolated_and_pending_is_acknowledged(tmp_path):
    first = TrialStorage(tmp_path, "profile_stanley", "worker-a")
    second = TrialStorage(tmp_path, "profile_stanley", "worker-b")
    row = {"trial": 9, "cost": 1.5}

    pending = first.stage_pending(row)
    assert pending.parent == first.pending_dir
    assert first.pending_results() == [row]
    assert first.output_dir != second.output_dir
    assert not second.pending_results()

    first.acknowledge_pending(9)
    assert not first.pending_results()


def test_trajectory_csv_records_simulator_and_wall_time_diagnostics(
    tmp_path,
):
    storage = TrialStorage(tmp_path, "profile_stanley")
    sample = tuple(range(21)) + ("failure,with detail", 0)

    path = storage.write_trajectory(1, [sample])
    with open(path, encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))

    assert rows[0][:4] == [
        "sim_elapsed_s",
        "wall_elapsed_s",
        "sim_time_s",
        "header_time_s",
    ]
    assert "steering_normalized" in rows[0]
    assert "lateral_acceleration_mps2" in rows[0]
    assert rows[0][15:17] == ["front_cte_m", "rear_cte_m"]
    assert rows[0][-2:] == ["dwa_failure_reason", "collision_count"]
    assert rows[1][-2] == "failure,with detail"
    assert len(rows[1]) == 23
