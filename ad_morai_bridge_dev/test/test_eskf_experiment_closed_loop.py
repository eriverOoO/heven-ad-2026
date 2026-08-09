from dataclasses import replace
import math

import pytest

from ad_morai_bridge_dev.eskf_experiment.closed_loop import (
    ClosedLoopPulseConfig,
    ClosedLoopPulseExecutor,
    load_closed_loop_pulse,
)
from ad_morai_bridge_dev.eskf_experiment.types import (
    ActorIdentity,
    SafetyLimits,
    TruthSample,
    VehicleCommand,
)


class FakeClock:
    def __init__(self):
        self.nanoseconds = 0

    def monotonic_ns(self):
        return self.nanoseconds

    def sleep(self, duration_sec):
        self.nanoseconds += int(duration_sec * 1.0e9)


def _truth(clock, *, x=0.0, speed=0.0, collisions=()):
    return TruthSample(
        receipt_monotonic_ns=clock.monotonic_ns(),
        rpc_start_monotonic_ns=clock.monotonic_ns(),
        simulator_timestamp=None,
        position_xyz=(x, 0.0, 0.0),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        world_velocity_xyz=(speed, 0.0, 0.0),
        world_acceleration_xyz=(0.0, 0.0, 0.0),
        gear_mode="GEAR_MODE_D",
        throttle=0.0,
        brake=0.0,
        steer=0.0,
        collision_object_ids=tuple(collisions),
    )


class FakeClient:
    def __init__(self, clock, states):
        self.clock = clock
        self.states = list(states)
        self.last_state = self.states[-1]
        self.commands = []
        self.actor_identity = ActorIdentity("Ego", "OBJECT_TYPE_VEHICLE", "")

    def get_truth(self):
        if self.states:
            self.last_state = self.states.pop(0)
        return replace(
            self.last_state,
            rpc_start_monotonic_ns=self.clock.monotonic_ns(),
            receipt_monotonic_ns=self.clock.monotonic_ns(),
        )

    def send_command(self, command):
        self.commands.append(command)

    def full_brake(self):
        self.commands.append(VehicleCommand(0.0, 1.0, 0.0))


class DelayedTruthClient(FakeClient):
    def __init__(self, clock, states, *, delay_sec, delay_after_calls=0):
        super().__init__(clock, states)
        self.delay_sec = delay_sec
        self.delay_after_calls = delay_after_calls
        self.truth_calls = 0

    def get_truth(self):
        if self.states:
            self.last_state = self.states.pop(0)
        rpc_start_ns = self.clock.monotonic_ns()
        if self.truth_calls >= self.delay_after_calls:
            self.clock.sleep(self.delay_sec)
        self.truth_calls += 1
        return replace(
            self.last_state,
            rpc_start_monotonic_ns=rpc_start_ns,
            receipt_monotonic_ns=self.clock.monotonic_ns(),
        )


@pytest.fixture
def limits():
    return SafetyLimits(
        maximum_start_speed_mps=0.05,
        maximum_speed_mps=0.50,
        maximum_travel_m=0.25,
        truth_stale_timeout_sec=0.25,
        estimator_stale_timeout_sec=0.25,
        command_rate_hz=20.0,
        maximum_command_delta_per_sec=0.50,
        stopped_speed_mps=0.02,
        stopped_stable_duration_sec=0.10,
    )


@pytest.fixture
def pulse():
    return ClosedLoopPulseConfig(
        target_speed_mps=0.15,
        soft_speed_limit_mps=0.25,
        soft_travel_limit_m=0.15,
        release_duration_sec=2.0,
        maximum_tracking_duration_sec=0.50,
        maximum_throttle=0.10,
        speed_deadband_mps=0.01,
        throttle_kp=0.40,
        throttle_ki=0.30,
    )


def test_closed_loop_config_is_explicit_and_bounded_by_hard_safety(limits):
    config = load_closed_loop_pulse(
        {
            "closed_loop_pulse": {
                "target_speed_mps": 0.15,
                "soft_speed_limit_mps": 0.25,
                "soft_travel_limit_m": 0.15,
                "release_duration_sec": 2.0,
                "maximum_tracking_duration_sec": 2.0,
                "maximum_throttle": 0.10,
                "speed_deadband_mps": 0.01,
                "throttle_kp": 0.40,
                "throttle_ki": 0.30,
            }
        },
        limits,
    )

    assert config.target_speed_mps < config.soft_speed_limit_mps
    assert config.soft_speed_limit_mps < limits.maximum_speed_mps
    assert config.soft_travel_limit_m < limits.maximum_travel_m
    assert config.maximum_throttle == pytest.approx(0.10)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("soft_speed_limit_mps", 0.50),
        ("soft_travel_limit_m", 0.25),
        ("maximum_throttle", 1.01),
        ("target_speed_mps", float("nan")),
    ),
)
def test_closed_loop_config_rejects_values_at_or_beyond_hard_bounds(
    limits, pulse, field, value
):
    payload = {**pulse.__dict__, field: value}
    with pytest.raises(ValueError):
        load_closed_loop_pulse({"closed_loop_pulse": payload}, limits)


def test_feedback_commands_are_finite_bounded_and_never_overlap(limits, pulse):
    clock = FakeClock()
    # preflight, release, then a low-speed ramp, followed by stopped cleanup.
    speeds = [0.0] * 41 + [0.0, 0.02, 0.06, 0.11, 0.15] + [0.0] * 20
    positions = [min(index * 0.01, 0.10) for index in range(len(speeds))]
    client = FakeClient(
        clock,
        [_truth(clock, x=x, speed=speed) for x, speed in zip(positions, speeds)],
    )
    truth_events = []
    command_events = []
    executor = ClosedLoopPulseExecutor(
        client,
        client.actor_identity,
        limits,
        monotonic_ns=clock.monotonic_ns,
        sleep=clock.sleep,
        on_truth=lambda phase, truth: truth_events.append((phase, truth)),
        on_command=lambda phase, command, stamp: command_events.append(
            (phase, command, stamp)
        ),
    )

    samples = executor.run(pulse)

    assert samples
    assert any(phase == "closed_loop_track" for phase, _truth in truth_events)
    assert any(
        phase == "closed_loop_track" and command.throttle > 0.0
        for phase, command, _stamp in command_events
    )
    assert all(command.steer == 0.0 for command in client.commands)
    assert all(
        math.isfinite(command.throttle) and math.isfinite(command.brake)
        for command in client.commands
    )
    assert max(command.throttle for command in client.commands) <= 0.10
    assert all(
        command.throttle == 0.0 or command.brake == 0.0
        for command in client.commands
    )
    assert any(phase == "closed_loop_stop" for phase, _truth in truth_events)


def test_soft_speed_limit_transitions_to_braking_below_hard_abort(limits, pulse):
    clock = FakeClock()
    states = [
        _truth(clock, speed=0.0),
        *[_truth(clock, speed=0.0) for _ in range(40)],
        _truth(clock, speed=0.26, x=0.01),
        *[_truth(clock, speed=0.0, x=0.02) for _ in range(20)],
    ]
    client = FakeClient(clock, states)
    phases = []
    executor = ClosedLoopPulseExecutor(
        client,
        client.actor_identity,
        limits,
        monotonic_ns=clock.monotonic_ns,
        sleep=clock.sleep,
        on_truth=lambda phase, _truth: phases.append(phase),
    )

    executor.run(replace(pulse, maximum_tracking_duration_sec=0.50))

    assert "closed_loop_soft_limit" in phases
    first_soft = phases.index("closed_loop_soft_limit")
    assert "closed_loop_track" not in phases[(first_soft + 1):]
    assert client.commands[-1] == VehicleCommand(0.0, 1.0, 0.0)


def test_nonself_collision_aborts_but_still_commands_full_brake(limits, pulse):
    clock = FakeClock()
    client = FakeClient(
        clock,
        [
            _truth(clock),
            _truth(clock),
            _truth(clock, collisions=("NPC-7",)),
            *[_truth(clock) for _ in range(20)],
        ],
    )
    executor = ClosedLoopPulseExecutor(
        client,
        client.actor_identity,
        limits,
        monotonic_ns=clock.monotonic_ns,
        sleep=clock.sleep,
    )

    with pytest.raises(RuntimeError, match="collision"):
        executor.run(pulse)

    assert sum(command.brake == 1.0 for command in client.commands) >= 5


def test_truth_rpc_duration_beyond_freshness_limit_aborts_and_brakes(
    limits, pulse
):
    clock = FakeClock()
    client = DelayedTruthClient(
        clock,
        [_truth(clock)],
        delay_sec=limits.truth_stale_timeout_sec + 0.01,
    )
    executor = ClosedLoopPulseExecutor(
        client,
        client.actor_identity,
        limits,
        monotonic_ns=clock.monotonic_ns,
        sleep=clock.sleep,
    )

    with pytest.raises(RuntimeError, match="stale"):
        executor.run(pulse)

    assert not [command for command in client.commands if command.throttle > 0.0]
    assert sum(command.brake == 1.0 for command in client.commands) >= 6


def test_tracking_uses_an_absolute_deadline_when_truth_calls_are_slow(
    limits, pulse
):
    clock = FakeClock()
    # Preflight plus 40 release samples remain immediate. Each tracking truth
    # call then consumes 0.20 s, below the 0.25 s freshness guard.
    client = DelayedTruthClient(
        clock,
        [_truth(clock)],
        delay_sec=0.20,
        delay_after_calls=41,
    )
    command_events = []
    executor = ClosedLoopPulseExecutor(
        client,
        client.actor_identity,
        limits,
        monotonic_ns=clock.monotonic_ns,
        sleep=clock.sleep,
        on_command=lambda phase, command, stamp: command_events.append(
            (phase, command, stamp)
        ),
    )

    executor.run(replace(pulse, maximum_tracking_duration_sec=0.50))

    tracking_commands = [
        event for event in command_events if event[0] == "closed_loop_track"
    ]
    assert len(tracking_commands) == 1
    first_stop = next(
        event for event in command_events if event[0] == "closed_loop_stop"
    )
    assert first_stop[2] - tracking_commands[0][2] <= 250_000_000
