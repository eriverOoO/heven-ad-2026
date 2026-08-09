import pytest

from ad_vehicle_profiling.controller import (
    ControllerConfig,
    ProfilerController,
    ProfilerPhase,
    VehicleObservation,
)
from ad_vehicle_profiling.experiment import ExperimentCell


def _observation(**overrides):
    values = {
        "speed_mps": 0.0,
        "acceleration_mps2": 0.0,
        "jerk_mps3": 0.0,
        "status_age_sec": 0.01,
        "command_publishers": 1,
        "collision": False,
        "ctrl_mode": 2,
        "gear": 4,
        "yaw_deviation_rad": 0.0,
        "lateral_displacement_m": 0.0,
    }
    values.update(overrides)
    return VehicleObservation(**values)


def _controller(speed_kph=20, kind="accelerator", command=25):
    return ProfilerController(
        ExperimentCell(speed_kph, kind, command),
        ControllerConfig(
            speed_tolerance_mps=0.15,
            settle_duration_sec=0.5,
            test_duration_sec=0.8,
            safe_stop_hold_sec=0.5,
            maximum_reach_duration_sec=2.0,
        ),
    )


def test_preflight_rejects_second_command_publisher():
    controller = _controller()

    decision = controller.update(
        _observation(command_publishers=2),
        now_sec=1.0,
    )

    assert decision.phase is ProfilerPhase.SAFE_STOP
    assert decision.fault == "control_topic_has_other_publisher"
    assert decision.accelerator == 0.0
    assert 0.0 < decision.brake <= controller.config.safe_stop_brake


@pytest.mark.parametrize(
    ("overrides", "fault"),
    [
        ({"status_age_sec": 1.5}, "stale_vehicle_status"),
        ({"collision": True}, "collision"),
        ({"yaw_deviation_rad": 0.3}, "yaw_deviation"),
        ({"lateral_displacement_m": 2.0}, "lateral_deviation"),
        ({"speed_mps": 53.0}, "overspeed"),
        ({"acceleration_mps2": 13.0}, "acceleration_limit"),
        ({"jerk_mps3": 55.0}, "jerk_limit"),
    ],
)
def test_every_safety_violation_enters_safe_stop(overrides, fault):
    controller = _controller()

    decision = controller.update(_observation(**overrides), now_sec=1.0)

    assert decision.phase is ProfilerPhase.SAFE_STOP
    assert decision.fault == fault


def test_reach_settle_apply_and_checkpoint_sequence():
    controller = _controller(speed_kph=20, kind="accelerator", command=25)
    target_mps = 20.0 / 3.6

    reaching = controller.update(_observation(), now_sec=0.0)
    settling = controller.update(
        _observation(speed_mps=target_mps),
        now_sec=2.0,
    )
    still_settling = controller.update(
        _observation(speed_mps=target_mps),
        now_sec=2.4,
    )
    applying = controller.update(
        _observation(speed_mps=target_mps),
        now_sec=2.6,
    )
    checkpoint = controller.update(
        _observation(speed_mps=target_mps + 0.5),
        now_sec=3.5,
    )

    assert reaching.phase is ProfilerPhase.REACH_TARGET_SPEED
    assert reaching.accelerator > 0.0
    assert settling.phase is ProfilerPhase.SETTLE
    assert still_settling.phase is ProfilerPhase.SETTLE
    assert applying.phase is ProfilerPhase.APPLY_TEST_COMMAND
    assert applying.accelerator == pytest.approx(0.25)
    assert applying.brake == 0.0
    assert checkpoint.phase is ProfilerPhase.SAVE_CHECKPOINT
    assert checkpoint.trial_complete


def test_settling_keeps_small_speed_control_instead_of_coasting():
    controller = _controller(speed_kph=20)
    target_mps = 20.0 / 3.6
    controller.update(_observation(), now_sec=0.0)
    controller.update(
        _observation(speed_mps=target_mps - 0.1),
        now_sec=1.0,
    )

    decision = controller.update(
        _observation(speed_mps=target_mps - 0.1),
        now_sec=1.2,
    )

    assert decision.phase is ProfilerPhase.SETTLE
    assert 0.0 < decision.accelerator < 0.1
    assert decision.brake == 0.0


def test_speed_integral_overcomes_constant_drag_near_target():
    controller = ProfilerController(
        ExperimentCell(40, "brake", 10),
        ControllerConfig(maximum_reach_duration_sec=120.0),
    )
    target_mps = 40.0 / 3.6
    speed_with_steady_state_error = target_mps - 0.16

    first = controller.update(
        _observation(speed_mps=speed_with_steady_state_error),
        now_sec=0.0,
    )
    decision = first
    for index in range(1, 51):
        decision = controller.update(
            _observation(speed_mps=speed_with_steady_state_error),
            now_sec=index * 0.1,
        )

    assert first.phase is ProfilerPhase.REACH_TARGET_SPEED
    assert decision.phase is ProfilerPhase.REACH_TARGET_SPEED
    assert decision.accelerator > first.accelerator + 0.05


def test_checkpoint_acknowledgement_resets_speed_integral():
    controller = _controller(speed_kph=40)
    target_mps = 40.0 / 3.6
    for index in range(20):
        controller.update(
            _observation(speed_mps=target_mps - 0.16),
            now_sec=index * 0.1,
        )
    assert controller._speed_error_integral > 0.0

    controller.phase = ProfilerPhase.SAVE_CHECKPOINT
    controller.acknowledge_checkpoint(
        ExperimentCell(45, "brake", 10)
    )

    assert controller._speed_error_integral == 0.0
    assert controller._previous_speed_error is None
    assert controller._last_speed_control_sec is None


def test_preflight_arms_auto_drive_without_applying_pedals():
    controller = _controller()

    decision = controller.update(
        _observation(ctrl_mode=1, gear=1),
        now_sec=0.0,
    )

    assert decision.phase is ProfilerPhase.PREFLIGHT
    assert decision.gear == 4
    assert decision.accelerator == 0.0
    assert decision.brake == 0.0
    assert decision.fault is None


@pytest.mark.parametrize(
    ("overrides", "fault"),
    [
        ({"ctrl_mode": 1}, "automatic_control_lost"),
        ({"gear": 2}, "drive_gear_lost"),
    ],
)
def test_auto_or_drive_loss_after_arming_enters_safe_stop(overrides, fault):
    controller = _controller()
    controller.update(_observation(), now_sec=0.0)

    decision = controller.update(_observation(**overrides), now_sec=0.1)

    assert decision.phase is ProfilerPhase.SAFE_STOP
    assert decision.fault == fault


def test_brake_test_never_sends_accelerator_at_the_same_time():
    controller = _controller(speed_kph=20, kind="brake", command=35)
    target_mps = 20.0 / 3.6
    controller.update(_observation(), now_sec=0.0)
    controller.update(_observation(speed_mps=target_mps), now_sec=1.0)

    decision = controller.update(
        _observation(speed_mps=target_mps),
        now_sec=1.6,
    )

    assert decision.phase is ProfilerPhase.APPLY_TEST_COMMAND
    assert decision.accelerator == 0.0
    assert decision.brake == pytest.approx(0.35)


def test_brake_test_finishes_early_after_vehicle_stops():
    controller = _controller(speed_kph=5, kind="brake", command=100)
    target_mps = 5.0 / 3.6
    controller.update(_observation(), now_sec=0.0)
    controller.update(_observation(speed_mps=target_mps), now_sec=1.0)
    controller.update(_observation(speed_mps=target_mps), now_sec=1.6)

    decision = controller.update(
        _observation(speed_mps=0.2),
        now_sec=2.0,
    )

    assert decision.phase is ProfilerPhase.SAVE_CHECKPOINT
    assert decision.trial_complete


def test_brake_test_duration_scales_inversely_with_command():
    low = _controller(speed_kph=20, kind="brake", command=10)
    high = _controller(speed_kph=20, kind="brake", command=100)

    assert low._test_duration_sec() == pytest.approx(0.8)
    assert high._test_duration_sec() == pytest.approx(0.35)


def test_checkpoint_acknowledgement_starts_next_cell_recovery():
    controller = _controller()
    controller.phase = ProfilerPhase.SAVE_CHECKPOINT
    next_cell = ExperimentCell(30, "brake", 10)

    controller.acknowledge_checkpoint(next_cell)
    decision = controller.update(_observation(speed_mps=2.0), now_sec=5.0)

    assert controller.cell == next_cell
    assert decision.phase is ProfilerPhase.REACH_TARGET_SPEED


def test_normal_completion_stops_before_complete():
    controller = _controller()
    controller.phase = ProfilerPhase.SAVE_CHECKPOINT
    controller.acknowledge_checkpoint(None)

    moving = controller.update(_observation(speed_mps=5.0), now_sec=1.0)
    first_stopped = controller.update(
        _observation(speed_mps=0.05),
        now_sec=2.0,
    )
    complete = controller.update(
        _observation(speed_mps=0.04),
        now_sec=2.6,
    )

    assert moving.phase is ProfilerPhase.SAFE_STOP
    assert moving.brake > 0.0
    assert first_stopped.phase is ProfilerPhase.SAFE_STOP
    assert complete.phase is ProfilerPhase.COMPLETE
    assert complete.brake == 0.0


def test_unreachable_speed_cell_checkpoints_instead_of_accelerating_forever():
    controller = _controller(speed_kph=185, command=100)

    controller.update(_observation(), now_sec=0.0)
    decision = controller.update(_observation(speed_mps=20.0), now_sec=2.1)

    assert decision.phase is ProfilerPhase.SAVE_CHECKPOINT
    assert decision.cell_unreachable
    assert decision.accelerator == 0.0


def test_external_stop_request_uses_safe_stop_path():
    controller = _controller()

    controller.request_stop("signal")
    decision = controller.update(_observation(speed_mps=3.0), now_sec=1.0)

    assert decision.phase is ProfilerPhase.SAFE_STOP
    assert decision.fault == "signal"
    assert decision.brake == controller.config.safe_stop_brake
