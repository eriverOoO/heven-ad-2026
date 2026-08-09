import pytest

from ad_tuning.parameters import FloatParameter
from ad_tuning.targets import TuningTarget, get_target, list_targets


def test_targets_are_listed_in_sorted_identifier_order():
    assert list_targets() == (
        "control/profile_stanley",
        "control/stanley",
        "planning/dwa",
    )


def test_target_identifier_requires_exact_domain_algorithm_shape():
    target = TuningTarget(
        domain="control",
        algorithm="stanley",
        parameters=(FloatParameter("stanley.lookahead_m", 2.0, 8.0),),
    )

    assert target.identifier == "control/stanley"
    assert get_target("control/stanley") is not None
    with pytest.raises(ValueError, match="domain/algorithm"):
        get_target("stanley")
    with pytest.raises(ValueError, match="unknown tuning target"):
        get_target("localization/eskf")


def test_target_rejects_duplicate_parameter_names():
    parameter = FloatParameter("stanley.lookahead_m", 2.0, 8.0)

    with pytest.raises(ValueError, match="duplicate"):
        TuningTarget("control", "stanley", (parameter, parameter))


def test_builtin_targets_only_expose_their_algorithm_namespace():
    for identifier in list_targets():
        target = get_target(identifier)
        namespace = target.algorithm + "."

        assert target.parameters
        assert all(parameter.name.startswith(namespace) for parameter in target.parameters)


def test_builtin_target_names_match_factory_tunable_parameter_contract():
    expected = {
        "control/stanley": {
            "stanley.cross_track_gain",
            "stanley.speed_softening_mps",
            "stanley.heading_error_gain",
            "stanley.lookahead_time_s",
            "stanley.curvature_lookahead_m",
            "stanley.speed_pid.kp",
            "stanley.speed_pid.kd",
            "stanley.brake_pid.kp",
            "stanley.brake_pid.kd",
        },
        "control/profile_stanley": {
            "profile_stanley.cross_track_gain",
            "profile_stanley.speed_softening_mps",
            "profile_stanley.heading_error_gain",
            "profile_stanley.lookahead_time_s",
            "profile_stanley.curvature_lookahead_m",
            "profile_stanley.speed_pid.kp",
            "profile_stanley.speed_pid.kd",
            "profile_stanley.brake_pid.kp",
            "profile_stanley.brake_pid.kd",
        },
        "planning/dwa": {
            "dwa.goal_weight",
            "dwa.heading_weight",
            "dwa.clearance_weight",
            "dwa.smoothness_weight",
            "dwa.path_distance_weight",
            "dwa.speed_weight",
            "dwa.speed_pid.kp",
            "dwa.speed_pid.kd",
            "dwa.brake_pid.kp",
            "dwa.brake_pid.kd",
        },
    }

    assert {
        identifier: {
            parameter.name for parameter in get_target(identifier).parameters
        }
        for identifier in list_targets()
    } == expected


@pytest.mark.parametrize(
    "identifier", ["control/stanley", "control/profile_stanley"]
)
def test_stanley_ranges_keep_ordered_lookahead_and_fixed_safety_limits(identifier):
    parameters = {
        parameter.name: parameter for parameter in get_target(identifier).parameters
    }
    prefix = get_target(identifier).algorithm

    assert f"{prefix}.lookahead_min_m" not in parameters
    assert f"{prefix}.lookahead_max_m" not in parameters
    assert f"{prefix}.target_speed_mps" not in parameters
    assert f"{prefix}.lateral_acceleration_mps2" not in parameters
    assert f"{prefix}.deceleration_mps2" not in parameters
    throttle_kp = parameters[f"{prefix}.speed_pid.kp"]
    assert throttle_kp.low == 0.05
    assert throttle_kp.high == 1.2
    assert throttle_kp.log is True
