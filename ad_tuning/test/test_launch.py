from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_reset_profile_enables_only_multi_ego_command():
    config = yaml.safe_load(
        (ROOT / "config" / "morai_reset.yaml").read_text(encoding="utf-8")
    )["/**"]["ros__parameters"]
    enabled = {
        name
        for name, value in config.items()
        if isinstance(value, dict) and value.get("enabled")
    }
    assert config["grpc"]["enabled"] is False
    assert enabled == {"multi_ego"}


def test_profile_stanley_tuning_has_bounded_defaults():
    config = yaml.safe_load(
        (ROOT / "config" / "tuning.yaml").read_text(encoding="utf-8")
    )["ad_global_path_tuner"]["ros__parameters"]

    assert config["maximum_trials"] == 30
    assert config["startup_timeout_sec"] == 60.0
    assert config["course_length_m"] == 0.0
    assert config["overspeed_kph"] == 60.0
    assert config["objective.maximum_cte_m"] == 0.7
    assert config["objective.front_cte_squared_weight"] == 300.0
    assert config["objective.rear_cte_squared_weight"] == 300.0
    assert config["storage.url_env"] == "OPTUNA_STORAGE_URL"
    assert config["storage.heartbeat_interval_sec"] == 15
    assert config["storage.connection_retry_attempts"] == 6
    assert config["storage.connect_timeout_sec"] == 3
    assert config["warm_start.inherit_minimum_complete_trials"] == 36
    assert config["warm_start.inherit_top_k"] == 5
    assert "warm_start.coordinator_worker_id" not in config
    assert config["timing.require_device_stamp"] is True


def test_tuning_stack_excludes_unneeded_perception_and_visualization():
    components = yaml.safe_load(
        (
            ROOT / "config" / "profile_stanley_components.yaml"
        ).read_text(encoding="utf-8")
    )["components"]

    assert components == {
        "description": True,
        "localization": True,
        "planner": True,
        "lidar_perception": False,
        "camera_perception": False,
        "bridge": True,
        "visualization": False,
    }

    bridge = yaml.safe_load(
        (ROOT / "config" / "morai_tuning_bridge.yaml").read_text(
            encoding="utf-8"
        )
    )["/**"]["ros__parameters"]
    enabled_inputs = {
        name
        for name, value in bridge.items()
        if isinstance(value, dict)
        and value.get("enabled")
        and name != "control"
    }
    assert enabled_inputs == {
        "competition_status",
        "collisions",
        "gps",
        "imu",
    }
    assert bridge["collisions"]["topic"] == "/ad/safety/collisions"
    assert bridge["timestamp_mode"] == "source_preferred"
    assert bridge["source_stamp_tolerance_sec"] == 1.0
    assert bridge["control"]["enabled"] is True


def test_dwa_tuning_has_bounded_search():
    config = yaml.safe_load(
        (ROOT / "config" / "dwa_tuning.yaml").read_text(encoding="utf-8")
    )["ad_dwa_tuner"]["ros__parameters"]

    assert config["maximum_trials"] == 120
    assert config["startup_timeout_sec"] == 60.0
    assert config["maximum_total_trials"] == 150
    assert config["minimum_feasible_trials"] == 30
    assert config["objective.maximum_collision_count"] == 0
    assert config["objective.maximum_cte_m"] == 4.5
    assert config["objective.minimum_local_planner_active_sec"] == 1.0
    assert config["objective.local_planner_failure_time_weight"] == 0.0
    assert config["timing.require_occupancy_grid"] is True
    assert config["perception_reset.required"] is True
    assert config["perception_reset.settle_sim_sec"] > 1.0
    assert config["perception_reset.minimum_prediction_samples"] >= 5
    assert config["warm_start.inherit_minimum_complete_trials"] == 40
    assert config["warm_start.inherit_top_k"] == 5
    assert "warm_start.coordinator_worker_id" not in config


def test_dwa_tuning_stack_enables_only_required_lidar_inputs():
    components = yaml.safe_load(
        (ROOT / "config" / "dwa_components.yaml").read_text(encoding="utf-8")
    )["components"]
    assert components == {
        "description": True,
        "localization": True,
        "planner": True,
        "lidar_perception": True,
        "camera_perception": False,
        "bridge": True,
        "visualization": False,
    }

    bridge = yaml.safe_load(
        (ROOT / "config" / "morai_dwa_tuning_bridge.yaml").read_text(
            encoding="utf-8"
        )
    )["/**"]["ros__parameters"]
    enabled_inputs = {
        name
        for name, value in bridge.items()
        if isinstance(value, dict)
        and value.get("enabled")
        and name != "control"
    }
    assert enabled_inputs == {
        "competition_status",
        "collisions",
        "gps",
        "imu",
        "velodyne",
    }
    assert bridge["velodyne"]["topic"] == "/ad/sensors/lidar/raw"
    assert bridge["collisions"]["topic"] == "/ad/safety/collisions"
    assert bridge["timestamp_mode"] == "source_preferred"
    assert bridge["source_stamp_tolerance_sec"] == 1.0
    assert bridge["control"]["enabled"] is True
