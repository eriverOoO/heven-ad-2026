import pytest

from ad_tuning.perception_epoch import perception_epoch_is_ready


BASE = {
    "reset_wall_s": 10.0,
    "now_wall_s": 11.3,
    "prediction_received_wall_s": 11.2,
    "occupancy_received_wall_s": 11.2,
    "baseline_prediction_sequence": 20,
    "prediction_sequence": 25,
    "minimum_prediction_samples": 5,
    "first_post_reset_sim_s": 100.0,
    "current_sim_s": 101.2,
    "settle_sim_s": 1.05,
    "stale_timeout_s": 2.0,
    "require_occupancy_grid": True,
}


def test_perception_epoch_requires_fresh_samples_and_simulator_time():
    assert perception_epoch_is_ready(**BASE)

    for override in (
        {"prediction_sequence": 24},
        {"prediction_received_wall_s": 9.9},
        {"occupancy_received_wall_s": 9.9},
        {"current_sim_s": 100.9},
        {"prediction_received_wall_s": 8.0},
    ):
        arguments = dict(BASE)
        arguments.update(override)
        assert not perception_epoch_is_ready(**arguments)


def test_perception_epoch_allows_missing_occupancy_only_when_not_required():
    arguments = dict(BASE)
    arguments.update(
        require_occupancy_grid=False,
        occupancy_received_wall_s=0.0,
    )
    assert perception_epoch_is_ready(**arguments)


def test_perception_epoch_fails_closed_on_clock_rollback_or_missing_clock():
    for current_sim_s in (None, 99.0):
        arguments = dict(BASE)
        arguments["current_sim_s"] = current_sim_s
        assert not perception_epoch_is_ready(**arguments)


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("minimum_prediction_samples", 0),
        ("settle_sim_s", -1.0),
        ("stale_timeout_s", 0.0),
    ),
)
def test_perception_epoch_rejects_invalid_contract(name, value):
    arguments = dict(BASE)
    arguments[name] = value
    with pytest.raises(ValueError):
        perception_epoch_is_ready(**arguments)
