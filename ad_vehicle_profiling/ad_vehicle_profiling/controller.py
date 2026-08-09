from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from .experiment import ExperimentCell


class ProfilerPhase(Enum):
    PREFLIGHT = auto()
    REACH_TARGET_SPEED = auto()
    SETTLE = auto()
    APPLY_TEST_COMMAND = auto()
    RECOVER = auto()
    SAVE_CHECKPOINT = auto()
    SAFE_STOP = auto()
    COMPLETE = auto()


@dataclass(frozen=True)
class ControllerConfig:
    speed_tolerance_mps: float = 0.15
    settle_duration_sec: float = 1.0
    settle_acceleration_mps2: float = 0.2
    test_duration_sec: float = 1.5
    minimum_test_duration_sec: float = 0.35
    test_end_speed_mps: float = 0.3
    scale_brake_duration_by_command: bool = True
    brake_duration_reference_command: float = 0.1
    speed_control_kp: float = 0.25
    speed_control_ki: float = 0.08
    speed_control_integral_limit: float = 12.5
    speed_control_maximum_dt_sec: float = 0.1
    maximum_reach_accelerator: float = 1.0
    maximum_recovery_brake: float = 0.35
    maximum_reach_duration_sec: float = 120.0
    stale_status_sec: float = 1.0
    maximum_yaw_deviation_rad: float = 0.2
    maximum_lateral_displacement_m: float = 1.5
    maximum_speed_mps: float = 190.0 / 3.6
    maximum_abs_acceleration_mps2: float = 12.0
    maximum_abs_jerk_mps3: float = 50.0
    safe_stop_brake: float = 0.4
    stopped_speed_mps: float = 0.1
    safe_stop_hold_sec: float = 1.0


@dataclass(frozen=True)
class VehicleObservation:
    speed_mps: float
    acceleration_mps2: float
    jerk_mps3: float
    status_age_sec: float
    command_publishers: int
    collision: bool
    ctrl_mode: int
    gear: int
    yaw_deviation_rad: float
    lateral_displacement_m: float


@dataclass(frozen=True)
class ControlDecision:
    phase: ProfilerPhase
    accelerator: float
    brake: float
    steering: float = 0.0
    gear: int = 4
    fault: Optional[str] = None
    trial_complete: bool = False
    cell_unreachable: bool = False


class ProfilerController:
    def __init__(
        self,
        cell: ExperimentCell,
        config: ControllerConfig | None = None,
    ) -> None:
        self.cell = cell
        self.config = config or ControllerConfig()
        self.phase = ProfilerPhase.PREFLIGHT
        self.fault: Optional[str] = None
        self._settle_started_sec: Optional[float] = None
        self._trial_started_sec: Optional[float] = None
        self._stopped_started_sec: Optional[float] = None
        self._reach_started_sec: Optional[float] = None
        self._speed_error_integral = 0.0
        self._previous_speed_error: Optional[float] = None
        self._last_speed_control_sec: Optional[float] = None

    def update(
        self,
        observation: VehicleObservation,
        now_sec: float,
    ) -> ControlDecision:
        if self.phase is ProfilerPhase.COMPLETE:
            return self._decision()
        if self.phase is not ProfilerPhase.SAFE_STOP:
            fault = self._safety_fault(observation)
            if fault is not None:
                self._enter_safe_stop(fault)
        if self.phase is ProfilerPhase.SAFE_STOP:
            return self._safe_stop_decision(observation, now_sec)

        if self.phase is ProfilerPhase.PREFLIGHT:
            if observation.ctrl_mode != 2 or observation.gear != 4:
                return self._decision()
            self.phase = ProfilerPhase.REACH_TARGET_SPEED
            self._reach_started_sec = now_sec

        if self.phase in (
            ProfilerPhase.REACH_TARGET_SPEED,
            ProfilerPhase.RECOVER,
        ):
            if self._at_target_speed(observation):
                self.phase = ProfilerPhase.SETTLE
                self._settle_started_sec = now_sec
                self._reach_started_sec = None
                return self._speed_control_decision(observation, now_sec)
            if self._reach_started_sec is None:
                self._reach_started_sec = now_sec
            if (
                now_sec - self._reach_started_sec
                >= self.config.maximum_reach_duration_sec
            ):
                self.phase = ProfilerPhase.SAVE_CHECKPOINT
                return self._decision(cell_unreachable=True)
            self.phase = ProfilerPhase.REACH_TARGET_SPEED
            return self._speed_control_decision(observation, now_sec)

        if self.phase is ProfilerPhase.SETTLE:
            if not self._at_target_speed(observation):
                self.phase = ProfilerPhase.REACH_TARGET_SPEED
                self._settle_started_sec = None
                self._reach_started_sec = now_sec
                return self._speed_control_decision(observation, now_sec)
            if (
                abs(observation.acceleration_mps2)
                > self.config.settle_acceleration_mps2
            ):
                self._settle_started_sec = now_sec
                return self._speed_control_decision(observation, now_sec)
            if self._settle_started_sec is None:
                self._settle_started_sec = now_sec
            if (
                now_sec - self._settle_started_sec
                >= self.config.settle_duration_sec
            ):
                self.phase = ProfilerPhase.APPLY_TEST_COMMAND
                self._trial_started_sec = now_sec
                return self._test_command_decision()
            return self._speed_control_decision(observation, now_sec)

        if self.phase is ProfilerPhase.APPLY_TEST_COMMAND:
            assert self._trial_started_sec is not None
            trial_elapsed = now_sec - self._trial_started_sec
            stopped_early = (
                self.cell.command_kind == "brake"
                and self.cell.command_percent > 0
                and trial_elapsed >= self.config.minimum_test_duration_sec
                and observation.speed_mps
                <= self.config.test_end_speed_mps
            )
            if (
                trial_elapsed >= self._test_duration_sec()
                or stopped_early
            ):
                self.phase = ProfilerPhase.SAVE_CHECKPOINT
                return self._decision(trial_complete=True)
        return self._test_command_decision()

    def _test_duration_sec(self) -> float:
        if (
            not self.config.scale_brake_duration_by_command
            or self.cell.command_kind != "brake"
            or self.cell.command_percent <= 0
        ):
            return self.config.test_duration_sec
        command = self.cell.command_percent / 100.0
        scaled = (
            self.config.test_duration_sec
            * self.config.brake_duration_reference_command
            / command
        )
        return max(
            self.config.minimum_test_duration_sec,
            min(self.config.test_duration_sec, scaled),
        )

    def acknowledge_checkpoint(
        self, next_cell: ExperimentCell | None
    ) -> None:
        if self.phase is not ProfilerPhase.SAVE_CHECKPOINT:
            raise RuntimeError("checkpoint acknowledgement in invalid phase")
        self._settle_started_sec = None
        self._trial_started_sec = None
        self._reach_started_sec = None
        self._speed_error_integral = 0.0
        self._previous_speed_error = None
        self._last_speed_control_sec = None
        if next_cell is None:
            self.cell = self.cell
            self._enter_safe_stop(None)
            return
        self.cell = next_cell
        self.phase = ProfilerPhase.RECOVER

    def request_stop(self, reason: str = "requested_stop") -> None:
        if self.phase is not ProfilerPhase.COMPLETE:
            self._enter_safe_stop(reason)

    def _safety_fault(
        self, observation: VehicleObservation
    ) -> Optional[str]:
        if observation.command_publishers > 1:
            return "control_topic_has_other_publisher"
        if observation.status_age_sec > self.config.stale_status_sec:
            return "stale_vehicle_status"
        if observation.collision:
            return "collision"
        if self.phase is not ProfilerPhase.PREFLIGHT:
            if observation.ctrl_mode != 2:
                return "automatic_control_lost"
            if observation.gear != 4:
                return "drive_gear_lost"
        if (
            abs(observation.yaw_deviation_rad)
            > self.config.maximum_yaw_deviation_rad
        ):
            return "yaw_deviation"
        if (
            abs(observation.lateral_displacement_m)
            > self.config.maximum_lateral_displacement_m
        ):
            return "lateral_deviation"
        if observation.speed_mps > self.config.maximum_speed_mps:
            return "overspeed"
        if (
            abs(observation.acceleration_mps2)
            > self.config.maximum_abs_acceleration_mps2
        ):
            return "acceleration_limit"
        if (
            abs(observation.jerk_mps3)
            > self.config.maximum_abs_jerk_mps3
        ):
            return "jerk_limit"
        return None

    def _at_target_speed(self, observation: VehicleObservation) -> bool:
        target_mps = self.cell.speed_kph / 3.6
        return (
            abs(target_mps - observation.speed_mps)
            <= self.config.speed_tolerance_mps
        )

    def _speed_control_decision(
        self,
        observation: VehicleObservation,
        now_sec: float,
    ) -> ControlDecision:
        target_mps = self.cell.speed_kph / 3.6
        error = target_mps - observation.speed_mps
        if (
            self._previous_speed_error is not None
            and error * self._previous_speed_error < 0.0
        ):
            self._speed_error_integral = 0.0
        elapsed = (
            0.0
            if self._last_speed_control_sec is None
            else min(
                max(0.0, now_sec - self._last_speed_control_sec),
                self.config.speed_control_maximum_dt_sec,
            )
        )
        self._speed_error_integral = max(
            -self.config.speed_control_integral_limit,
            min(
                self.config.speed_control_integral_limit,
                self._speed_error_integral + error * elapsed,
            ),
        )
        self._previous_speed_error = error
        self._last_speed_control_sec = now_sec
        output = (
            self.config.speed_control_kp * error
            + self.config.speed_control_ki * self._speed_error_integral
        )
        if output >= 0.0:
            return self._decision(
                accelerator=min(
                    output,
                    self.config.maximum_reach_accelerator,
                )
            )
        return self._decision(
            brake=min(-output, self.config.maximum_recovery_brake)
        )

    def _test_command_decision(self) -> ControlDecision:
        command = self.cell.command_percent / 100.0
        if self.cell.command_kind == "accelerator":
            return self._decision(accelerator=command)
        return self._decision(brake=command)

    def _enter_safe_stop(self, fault: Optional[str]) -> None:
        self.phase = ProfilerPhase.SAFE_STOP
        self.fault = fault
        self._stopped_started_sec = None

    def _safe_stop_decision(
        self,
        observation: VehicleObservation,
        now_sec: float,
    ) -> ControlDecision:
        if observation.speed_mps <= self.config.stopped_speed_mps:
            if self._stopped_started_sec is None:
                self._stopped_started_sec = now_sec
            if (
                now_sec - self._stopped_started_sec
                >= self.config.safe_stop_hold_sec
            ):
                self.phase = ProfilerPhase.COMPLETE
                return self._decision()
        else:
            self._stopped_started_sec = None
        return self._decision(brake=self.config.safe_stop_brake)

    def _decision(
        self,
        *,
        accelerator: float = 0.0,
        brake: float = 0.0,
        trial_complete: bool = False,
        cell_unreachable: bool = False,
    ) -> ControlDecision:
        return ControlDecision(
            phase=self.phase,
            accelerator=accelerator,
            brake=brake,
            fault=self.fault,
            trial_complete=trial_complete,
            cell_unreachable=cell_unreachable,
        )
