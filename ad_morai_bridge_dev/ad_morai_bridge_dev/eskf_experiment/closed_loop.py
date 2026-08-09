"""Portable bounded longitudinal feedback for the ESKF A/B drive probe.

The estimator candidates never consume these commands or MORAI truth.  This
controller exists only to make the evaluation trajectory less dependent on an
open-loop pedal-to-acceleration mapping.  Its hard guards remain independent
of the feedback target and actuator gains.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Mapping

from ad_morai_bridge_dev.eskf_experiment.types import (
    ActorIdentity,
    SafetyLimits,
    TruthSample,
    VehicleCommand,
    truth_freshness_error,
)


@dataclass(frozen=True)
class ClosedLoopPulseConfig:
    """Reviewed low-speed pulse parameters, separate from ESKF parameters."""

    target_speed_mps: float
    soft_speed_limit_mps: float
    soft_travel_limit_m: float
    release_duration_sec: float
    maximum_tracking_duration_sec: float
    maximum_throttle: float
    speed_deadband_mps: float
    throttle_kp: float
    throttle_ki: float


_CLOSED_LOOP_FIELDS = tuple(ClosedLoopPulseConfig.__dataclass_fields__)


def load_closed_loop_pulse(
    document: Mapping[str, object], limits: SafetyLimits
) -> ClosedLoopPulseConfig:
    """Load the fixed schema and require margin inside every hard guard."""
    raw = document.get("closed_loop_pulse")
    if not isinstance(raw, Mapping):
        raise ValueError("closed_loop_pulse must be a mapping")
    unknown = set(raw) - set(_CLOSED_LOOP_FIELDS)
    missing = set(_CLOSED_LOOP_FIELDS) - set(raw)
    if unknown or missing:
        raise ValueError(
            "closed_loop_pulse fields mismatch: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    values = {field: float(raw[field]) for field in _CLOSED_LOOP_FIELDS}
    if not all(math.isfinite(value) and value > 0.0 for value in values.values()):
        raise ValueError("closed_loop_pulse values must be positive and finite")
    config = ClosedLoopPulseConfig(**values)
    if not (
        limits.stopped_speed_mps
        < config.target_speed_mps
        < config.soft_speed_limit_mps
        < limits.maximum_speed_mps
    ):
        raise ValueError("closed-loop speed targets must remain inside hard limits")
    if not 0.0 < config.soft_travel_limit_m < limits.maximum_travel_m:
        raise ValueError("soft_travel_limit_m must remain inside maximum_travel_m")
    if not 0.0 < config.maximum_throttle <= 1.0:
        raise ValueError("maximum_throttle must be in (0, 1]")
    if config.speed_deadband_mps >= config.target_speed_mps:
        raise ValueError("speed_deadband_mps must be below target_speed_mps")
    minimum_release_sec = 1.0 / limits.maximum_command_delta_per_sec
    if config.release_duration_sec < minimum_release_sec:
        raise ValueError(
            "release_duration_sec must fully release the brake at the slew limit"
        )
    return config


def _speed_mps(truth: TruthSample) -> float:
    return math.sqrt(sum(value * value for value in truth.world_velocity_xyz))


def _distance_m(lhs: TruthSample, rhs: TruthSample) -> float:
    return math.sqrt(
        sum(
            (left - right) * (left - right)
            for left, right in zip(lhs.position_xyz, rhs.position_xyz)
        )
    )


def _toward(current: float, target: float, maximum_step: float) -> float:
    if target > current:
        return min(target, current + maximum_step)
    return max(target, current - maximum_step)


def _slew_command(
    current: VehicleCommand, target: VehicleCommand, maximum_step: float
) -> VehicleCommand:
    throttle_target = target.throttle
    brake_target = target.brake
    if current.brake > 0.0 and throttle_target > 0.0:
        throttle_target = 0.0
    if current.throttle > 0.0 and brake_target > 0.0:
        brake_target = 0.0
    throttle = _toward(current.throttle, throttle_target, maximum_step)
    brake = _toward(current.brake, brake_target, maximum_step)
    if throttle > 0.0 and brake > 0.0:
        if current.brake > 0.0:
            throttle = 0.0
        else:
            brake = 0.0
    return VehicleCommand(throttle, brake, 0.0)


class ClosedLoopPulseExecutor:
    """Run one straight low-speed feedback pulse with fail-closed braking."""

    def __init__(
        self,
        client,
        actor: ActorIdentity,
        limits: SafetyLimits,
        *,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], None] = time.sleep,
        on_truth: Callable[[str, TruthSample], None] | None = None,
        on_command: Callable[[str, VehicleCommand, int], None] | None = None,
        health_check: Callable[[int], None] | None = None,
        abort_requested: Callable[[], bool] | None = None,
    ) -> None:
        self._client = client
        self._actor = actor
        self._limits = limits
        self._monotonic_ns = monotonic_ns
        self._sleep = sleep
        self._on_truth = on_truth or (lambda _phase, _truth: None)
        self._on_command = on_command or (
            lambda _phase, _command, _receipt_ns: None
        )
        self._health_check = health_check or (lambda _now_ns: None)
        self._abort_requested = abort_requested or (lambda: False)

    def run(self, config: ClosedLoopPulseConfig) -> tuple[TruthSample, ...]:
        # Re-run schema validation before the first vehicle write, even when a
        # caller constructed the frozen dataclass directly.
        config = load_closed_loop_pulse(
            {"closed_loop_pulse": config.__dict__}, self._limits
        )
        period_sec = 1.0 / self._limits.command_rate_hz
        maximum_step = (
            self._limits.maximum_command_delta_per_sec * period_sec
        )
        samples: list[TruthSample] = []
        previous: TruthSample | None = None
        start: TruthSample | None = None
        cumulative_travel_m = 0.0
        current = VehicleCommand(0.0, 1.0, 0.0)
        primary_error: BaseException | None = None

        def validate(truth: TruthSample, phase: str) -> TruthSample:
            nonlocal previous, start, cumulative_travel_m
            values = (
                *truth.position_xyz,
                *truth.orientation_xyzw,
                *truth.world_velocity_xyz,
                *truth.world_acceleration_xyz,
            )
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError("MORAI truth contains a non-finite value")
            # Finite safety-triggering samples are written before the guard.
            self._on_truth(phase, truth)
            samples.append(truth)
            now_ns = self._monotonic_ns()
            freshness_error = truth_freshness_error(
                truth, now_ns, self._limits.truth_stale_timeout_sec
            )
            if freshness_error is not None:
                raise RuntimeError(freshness_error)
            if truth.gear_mode != "GEAR_MODE_D":
                raise RuntimeError(f"Ego is not in drive gear: {truth.gear_mode}")
            collision_ids = tuple(
                value
                for value in truth.collision_object_ids
                if value and value != self._actor.id_value
            )
            if collision_ids:
                raise RuntimeError(f"MORAI reports a non-self collision: {collision_ids}")
            if _speed_mps(truth) > self._limits.maximum_speed_mps:
                raise RuntimeError("Ego exceeded maximum_speed_mps")
            start = start or truth
            if previous is not None:
                cumulative_travel_m += _distance_m(truth, previous)
            previous = truth
            if _distance_m(truth, start) > self._limits.maximum_travel_m:
                raise RuntimeError("Ego exceeded maximum_travel_m")
            if cumulative_travel_m > self._limits.maximum_travel_m:
                raise RuntimeError("Ego exceeded maximum cumulative travel")
            self._health_check(now_ns)
            return truth

        def sample(phase: str) -> TruthSample:
            return validate(self._client.get_truth(), phase)

        def verify_stopped(phase: str) -> None:
            stable_since_ns: int | None = None
            deadline_ns = self._monotonic_ns() + int(
                max(2.0, self._limits.stopped_stable_duration_sec * 4.0)
                * 1.0e9
            )
            while self._monotonic_ns() <= deadline_ns:
                self._check_abort()
                self._send_full_brake(phase)
                self._sleep(period_sec)
                truth = sample(phase)
                if _speed_mps(truth) <= self._limits.stopped_speed_mps:
                    stable_since_ns = stable_since_ns or self._monotonic_ns()
                    if self._monotonic_ns() - stable_since_ns >= int(
                        self._limits.stopped_stable_duration_sec * 1.0e9
                    ):
                        return
                else:
                    stable_since_ns = None
            raise RuntimeError("closed-loop pulse could not verify stable stop")

        try:
            self._send_full_brake("closed_loop_preflight")
            preflight = sample("closed_loop_preflight")
            if _speed_mps(preflight) > self._limits.maximum_start_speed_mps:
                raise RuntimeError("Ego start speed exceeds maximum_start_speed_mps")

            release_steps = max(
                1, math.ceil(config.release_duration_sec / period_sec)
            )
            for _ in range(release_steps):
                self._check_abort()
                release_truth = sample("closed_loop_release")
                if _speed_mps(release_truth) > self._limits.maximum_start_speed_mps:
                    raise RuntimeError("Ego moved while releasing the brake")
                current = _slew_command(
                    current, VehicleCommand(0.0, 0.0, 0.0), maximum_step
                )
                self._send_command("closed_loop_release", current)
                self._sleep(period_sec)

            integral_error = 0.0
            tracking_deadline_ns = self._monotonic_ns() + round(
                config.maximum_tracking_duration_sec * 1.0e9
            )
            truth_deadline_guard_ns = round(
                self._limits.truth_stale_timeout_sec * 1.0e9
            )
            while self._monotonic_ns() < tracking_deadline_ns:
                self._check_abort()
                # A previously issued throttle may remain active while this
                # synchronous read blocks. The gRPC wrapper caps that block at
                # truth_stale_timeout_sec, so begin braking instead of starting
                # a read whose full deadline no longer fits in this phase.
                if (
                    tracking_deadline_ns - self._monotonic_ns()
                    <= truth_deadline_guard_ns
                ):
                    break
                truth = self._client.get_truth()
                speed = _speed_mps(truth)
                projected_travel = cumulative_travel_m
                if previous is not None:
                    projected_travel += _distance_m(truth, previous)
                soft_limited = (
                    speed >= config.soft_speed_limit_mps
                    or projected_travel >= config.soft_travel_limit_m
                )
                # Record why feedback transitioned to braking, then apply all
                # hard checks to this same single RPC sample.
                truth = validate(
                    truth,
                    (
                        "closed_loop_soft_limit"
                        if soft_limited
                        else "closed_loop_track"
                    ),
                )
                if soft_limited:
                    break
                # A successful but slow read must not extend throttle exposure
                # beyond the reviewed wall-clock phase duration.
                if self._monotonic_ns() >= tracking_deadline_ns:
                    break
                speed = _speed_mps(truth)
                error = config.target_speed_mps - speed
                if error > config.speed_deadband_mps:
                    integral_error += error * period_sec
                    target_throttle = min(
                        config.maximum_throttle,
                        config.throttle_kp * error
                        + config.throttle_ki * integral_error,
                    )
                else:
                    integral_error = 0.0
                    target_throttle = 0.0
                current = _slew_command(
                    current,
                    VehicleCommand(target_throttle, 0.0, 0.0),
                    maximum_step,
                )
                self._send_command("closed_loop_track", current)
                self._sleep(period_sec)

            verify_stopped("closed_loop_stop")
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                self._repeat_full_brake("closed_loop_cleanup", count=5)
            except BaseException:
                if primary_error is None:
                    raise
        return tuple(samples)

    def _check_abort(self) -> None:
        if self._abort_requested():
            raise RuntimeError("experiment abort requested")

    def _send_command(self, phase: str, command: VehicleCommand) -> None:
        self._on_command(phase, command, self._monotonic_ns())
        self._client.send_command(command)

    def _send_full_brake(self, phase: str) -> None:
        command = VehicleCommand(0.0, 1.0, 0.0)
        self._on_command(phase, command, self._monotonic_ns())
        self._client.full_brake()

    def _repeat_full_brake(self, phase: str, *, count: int) -> None:
        period_sec = 1.0 / self._limits.command_rate_hz
        for _ in range(count):
            self._send_full_brake(phase)
            self._sleep(period_sec)
