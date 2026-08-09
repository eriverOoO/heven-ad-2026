from __future__ import annotations

from collections import deque
from dataclasses import replace
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import platform
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data

from ad_morai_interfaces.msg import CollisionArray, CtrlCmd, EgoVehicleStatus

from .controller import (
    ControlDecision,
    ControllerConfig,
    ProfilerController,
    ProfilerPhase,
    VehicleObservation,
)
from .experiment import (
    DEFAULT_SPEEDS_KPH,
    ExperimentCell,
    TrialSample,
    summarize_trial,
)
from .report import write_report
from .storage import RunStore


def resolve_output_root(explicit: str) -> Path:
    value = explicit.strip() or os.environ.get("AD_DATA_DIR", "").strip()
    if not value:
        raise ValueError(
            "output_root is empty and AD_DATA_DIR is not configured"
        )
    return Path(value).expanduser().resolve()


def resolve_run_id(explicit: str, *, now: float | None = None) -> str:
    value = explicit.strip()
    if value:
        return value
    timestamp = time.time() if now is None else now
    return time.strftime(
        "%Y%m%d-%H%M%S-ioniq5-longitudinal",
        time.gmtime(timestamp),
    )


def make_ctrl_cmd(decision: ControlDecision) -> CtrlCmd:
    if decision.accelerator > 0.0 and decision.brake > 0.0:
        raise ValueError("accelerator and brake cannot be active together")
    message = CtrlCmd()
    message.ctrl_mode = CtrlCmd.CTRL_MODE_AUTO
    message.gear = decision.gear
    message.long_cmd_type = CtrlCmd.LONG_CMD_THROTTLE
    message.velocity = 0.0
    message.acceleration = 0.0
    message.accel = float(decision.accelerator)
    message.brake = float(decision.brake)
    message.steering = float(decision.steering)
    return message


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


class VehicleProfilerNode(Node):
    def __init__(self) -> None:
        super().__init__("ad_vehicle_profiler")
        self._declare_parameters()
        self._command_topic = str(
            self.get_parameter("topics.command").value
        )
        status_topic = str(
            self.get_parameter("topics.vehicle_status").value
        )
        collision_topic = str(
            self.get_parameter("topics.collisions").value
        )

        cells = self._build_configured_cells()
        output_root = resolve_output_root(
            str(self.get_parameter("output_root").value)
        )
        run_id = resolve_run_id(str(self.get_parameter("run_id").value))
        manifest = self._manifest(cells, run_id)
        run_directory = output_root / "vehicle_dynamics" / run_id
        if (
            bool(self.get_parameter("resume").value)
            and (run_directory / "manifest.json").exists()
        ):
            self._store = RunStore.resume(run_directory)
        else:
            self._store = RunStore.create(
                run_directory,
                manifest,
                cells=cells,
            )
        self.run_directory = run_directory

        pending = self._store.pending_cells()
        if not pending:
            raise RuntimeError("profiling matrix has no pending cells")
        self._controller = ProfilerController(
            pending[0], self._controller_config()
        )
        self._discard_sec = float(
            self.get_parameter("discard_sec").value
        )
        self._window_end_sec = float(
            self.get_parameter("measurement_window_end_sec").value
        )
        self._maximum_sample_gap_sec = float(
            self.get_parameter("maximum_sample_gap_sec").value
        )
        self._baseline_duration_sec = float(
            self.get_parameter("baseline_duration_sec").value
        )
        self._minimum_samples = int(
            self.get_parameter("minimum_measurement_samples").value
        )
        self._minimum_braking_speed_mps = float(
            self.get_parameter("minimum_braking_speed_mps").value
        )
        self._echo_fraction = float(
            self.get_parameter("command_echo_fraction").value
        )
        self._onset_minimum_deceleration_mps2 = float(
            self.get_parameter(
                "onset_minimum_deceleration_mps2"
            ).value
        )
        self._onset_sigma_multiplier = float(
            self.get_parameter("onset_sigma_multiplier").value
        )
        self._onset_sustain_samples = int(
            self.get_parameter("onset_sustain_samples").value
        )

        command_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._command_publisher = self.create_publisher(
            CtrlCmd, self._command_topic, command_qos
        )
        self.create_subscription(
            EgoVehicleStatus,
            status_topic,
            self._on_status,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CollisionArray,
            collision_topic,
            self._on_collisions,
            qos_profile_sensor_data,
        )
        self._timer = self.create_timer(0.05, self._on_timer)

        self._latest_status: Optional[EgoVehicleStatus] = None
        self._latest_status_received_sec: Optional[float] = None
        self._collision = False
        self._reference_position: Optional[tuple[float, float]] = None
        self._reference_yaw: Optional[float] = None
        self._previous_acceleration: Optional[float] = None
        self._previous_acceleration_time: Optional[float] = None
        self._jerk_mps3 = 0.0
        self._trial_started_sec: Optional[float] = None
        self._trial_samples: list[TrialSample] = []
        self._baseline_buffer: deque[TrialSample] = deque()
        self._last_decision = ControlDecision(
            ProfilerPhase.PREFLIGHT, 0.0, 0.0
        )
        self._last_phase = ProfilerPhase.PREFLIGHT
        self._last_state_write_sec = 0.0
        self.finished = False
        self._store.append_event(
            unix_time_sec=time.time(),
            monotonic_time_sec=time.monotonic(),
            event="run_started_or_resumed",
            phase=self._controller.phase.name,
            cell=self._controller.cell,
            trial_index=(
                self._store.attempted_trial_count(self._controller.cell) + 1
            ),
            detail=("resume" if (run_directory / "run_state.json").exists() else "new"),
        )
        self.get_logger().info(f"profiling run: {run_directory}")

    def _declare_parameters(self) -> None:
        defaults = {
            "run_id": "",
            "output_root": "",
            "resume": True,
            "topics.command": "/ad/control/command",
            "topics.vehicle_status": "/ad/vehicle/status",
            "topics.collisions": "/ad/safety/collisions",
            "vehicle_model": "2023_Hyundai_Ioniq5",
            "simulator_build": "S4.251001.MolitComp03",
            "command_kinds": ["brake"],
            "speed_bins_kph": list(DEFAULT_SPEEDS_KPH),
            "command_percentages": list(range(0, 101, 10)),
            "minimum_valid_trials": 3,
            "maximum_attempts": 7,
            "mad_limit_mps2": 0.15,
            "cross_check_limit_mps2": 0.2,
            "repeatability_mad_limit_mps2": 0.5,
            "baseline_duration_sec": 0.75,
            "discard_sec": 0.08,
            "measurement_window_end_sec": 1.45,
            "minimum_measurement_samples": 2,
            "minimum_braking_speed_mps": 0.5,
            "maximum_sample_gap_sec": 0.1,
            "command_echo_fraction": 0.8,
            "onset_minimum_deceleration_mps2": 0.15,
            "onset_sigma_multiplier": 3.0,
            "onset_sustain_samples": 2,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        for name, value in ControllerConfig().__dict__.items():
            self.declare_parameter(f"controller.{name}", value)

    def _build_configured_cells(self) -> tuple[ExperimentCell, ...]:
        speeds = tuple(
            int(value)
            for value in self.get_parameter("speed_bins_kph").value
        )
        commands = tuple(
            int(value)
            for value in self.get_parameter("command_percentages").value
        )
        kinds = tuple(
            str(value)
            for value in self.get_parameter("command_kinds").value
        )
        return tuple(
            ExperimentCell(speed, kind, command)
            for speed in speeds
            for command in commands
            for kind in kinds
        )

    def _manifest(
        self,
        cells: tuple[ExperimentCell, ...],
        run_id: str,
    ) -> dict[str, object]:
        return {
            "format_version": 2,
            "run_id": run_id,
            "created_unix_sec": time.time(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": platform.node(),
            "platform": platform.platform(),
            "ros_domain_id": os.environ.get("ROS_DOMAIN_ID", "0"),
            "vehicle_model": str(
                self.get_parameter("vehicle_model").value
            ),
            "simulator_build": str(
                self.get_parameter("simulator_build").value
            ),
            "command_kinds": list(
                self.get_parameter("command_kinds").value
            ),
            "speed_bins_kph": list(
                self.get_parameter("speed_bins_kph").value
            ),
            "command_percentages": list(
                self.get_parameter("command_percentages").value
            ),
            "minimum_valid_trials": int(
                self.get_parameter("minimum_valid_trials").value
            ),
            "maximum_attempts": int(
                self.get_parameter("maximum_attempts").value
            ),
            "mad_limit_mps2": float(
                self.get_parameter("mad_limit_mps2").value
            ),
            "cross_check_limit_mps2": float(
                self.get_parameter("cross_check_limit_mps2").value
            ),
            "repeatability_mad_limit_mps2": float(
                self.get_parameter(
                    "repeatability_mad_limit_mps2"
                ).value
            ),
            "measurement": {
                name: self.get_parameter(name).value
                for name in (
                    "baseline_duration_sec",
                    "discard_sec",
                    "measurement_window_end_sec",
                    "minimum_measurement_samples",
                    "minimum_braking_speed_mps",
                    "maximum_sample_gap_sec",
                    "command_echo_fraction",
                    "onset_minimum_deceleration_mps2",
                    "onset_sigma_multiplier",
                    "onset_sustain_samples",
                )
            },
            "controller": self._controller_config().__dict__,
            "cell_count": len(cells),
        }

    def _controller_config(self) -> ControllerConfig:
        values = {
            name: self.get_parameter(f"controller.{name}").value
            for name in ControllerConfig().__dict__
        }
        return ControllerConfig(**values)

    def _on_status(self, message: EgoVehicleStatus) -> None:
        now_sec = time.monotonic()
        if self._reference_position is None:
            self._reference_position = (
                float(message.position.x),
                float(message.position.y),
            )
            self._reference_yaw = float(message.rpy.z)
        if (
            self._previous_acceleration is not None
            and self._previous_acceleration_time is not None
        ):
            elapsed = now_sec - self._previous_acceleration_time
            if elapsed > 0.0:
                self._jerk_mps3 = (
                    float(message.acceleration.x)
                    - self._previous_acceleration
                ) / elapsed
        self._previous_acceleration = float(message.acceleration.x)
        self._previous_acceleration_time = now_sec
        self._latest_status = message
        self._latest_status_received_sec = now_sec

        sample = self._sample_from_status(message, now_sec)
        if (
            self._controller.phase is ProfilerPhase.APPLY_TEST_COMMAND
            and self._trial_started_sec is not None
        ):
            self._trial_samples.append(
                replace(
                    sample,
                    elapsed_sec=now_sec - self._trial_started_sec,
                    sample_phase="command",
                    requested_accelerator=self._last_decision.accelerator,
                    requested_brake=self._last_decision.brake,
                )
            )
        else:
            self._baseline_buffer.append(sample)
            cutoff = now_sec - max(2.0, self._baseline_duration_sec * 2.0)
            while (
                self._baseline_buffer
                and self._baseline_buffer[0].monotonic_time_sec < cutoff
            ):
                self._baseline_buffer.popleft()

    def _sample_from_status(
        self, message: EgoVehicleStatus, now_sec: float
    ) -> TrialSample:
        stamp = message.header.stamp
        device_stamp = message.device_stamp
        return TrialSample(
            elapsed_sec=0.0,
            monotonic_time_sec=now_sec,
            ros_time_sec=float(stamp.sec)
            + float(stamp.nanosec) * 1e-9,
            device_time_sec=(
                float(device_stamp.sec)
                + float(device_stamp.nanosec) * 1e-9
                if message.has_device_stamp
                else 0.0
            ),
            sample_phase="baseline",
            velocity_mps=abs(float(message.signed_velocity)),
            acceleration_mps2=float(message.acceleration.x),
            requested_accelerator=self._last_decision.accelerator,
            requested_brake=self._last_decision.brake,
            echoed_accelerator=float(message.accel),
            echoed_brake=float(message.brake),
            position_x_m=float(message.position.x),
            position_y_m=float(message.position.y),
            yaw_rad=float(message.rpy.z),
            gear=int(message.gear),
            link_id=str(message.link_id),
            ctrl_mode=int(message.ctrl_mode),
            map_data_id=int(message.map_data_id),
            steering_rad=float(message.steering),
            velocity_x_mps=float(message.velocity.x),
            velocity_y_mps=float(message.velocity.y),
            velocity_z_mps=float(message.velocity.z),
            acceleration_y_mps2=float(message.acceleration.y),
            acceleration_z_mps2=float(message.acceleration.z),
            angular_velocity_x_radps=float(message.angular_velocity.x),
            angular_velocity_y_radps=float(message.angular_velocity.y),
            angular_velocity_z_radps=float(message.angular_velocity.z),
            collision=self._collision,
        )

    def _on_collisions(self, message: CollisionArray) -> None:
        self._collision = bool(message.collisions)

    def _observation(self, now_sec: float) -> VehicleObservation:
        assert self._latest_status is not None
        assert self._latest_status_received_sec is not None
        assert self._reference_position is not None
        assert self._reference_yaw is not None
        status = self._latest_status
        dx = float(status.position.x) - self._reference_position[0]
        dy = float(status.position.y) - self._reference_position[1]
        lateral = (
            -math.sin(self._reference_yaw) * dx
            + math.cos(self._reference_yaw) * dy
        )
        return VehicleObservation(
            speed_mps=abs(float(status.signed_velocity)),
            acceleration_mps2=float(status.acceleration.x),
            jerk_mps3=self._jerk_mps3,
            status_age_sec=now_sec - self._latest_status_received_sec,
            command_publishers=len(
                self.get_publishers_info_by_topic(self._command_topic)
            ),
            collision=self._collision,
            ctrl_mode=int(status.ctrl_mode),
            gear=int(status.gear),
            yaw_deviation_rad=_wrap_angle(
                float(status.rpy.z) - self._reference_yaw
            ),
            lateral_displacement_m=lateral,
        )

    def _on_timer(self) -> None:
        if self.finished or self._latest_status is None:
            return
        now_sec = time.monotonic()
        decision = self._controller.update(
            self._observation(now_sec), now_sec
        )
        if (
            decision.phase is ProfilerPhase.APPLY_TEST_COMMAND
            and self._trial_started_sec is None
        ):
            self._trial_started_sec = now_sec
            cutoff = now_sec - self._baseline_duration_sec
            self._trial_samples = [
                replace(
                    sample,
                    elapsed_sec=sample.monotonic_time_sec - now_sec,
                    sample_phase="baseline",
                )
                for sample in self._baseline_buffer
                if sample.monotonic_time_sec >= cutoff
            ]
        self._last_decision = decision
        self._command_publisher.publish(make_ctrl_cmd(decision))

        if decision.cell_unreachable:
            self._classify_unreachable_and_advance()
        elif decision.trial_complete:
            self._save_trial_and_advance()
        self._write_progress(now_sec, decision)
        if decision.phase is ProfilerPhase.COMPLETE:
            outputs = write_report(self._store.run_directory)
            self.get_logger().info(
                f"profile report: {outputs['json']}"
            )
            self.finished = True
            self._timer.cancel()

    def _save_trial_and_advance(self) -> None:
        cell = self._controller.cell
        trial_index = self._store.attempted_trial_count(cell) + 1
        samples = tuple(self._trial_samples)
        summary = summarize_trial(
            samples,
            discard_sec=self._discard_sec,
            window_end_sec=self._window_end_sec,
            maximum_gap_sec=self._maximum_sample_gap_sec,
            command_kind=cell.command_kind,
            command_value=cell.command_percent / 100.0,
            target_speed_kph=float(cell.speed_kph),
            minimum_samples=self._minimum_samples,
            minimum_braking_speed_mps=self._minimum_braking_speed_mps,
            echo_fraction=self._echo_fraction,
            onset_minimum_deceleration_mps2=(
                self._onset_minimum_deceleration_mps2
            ),
            onset_sigma_multiplier=self._onset_sigma_multiplier,
            onset_sustain_samples=self._onset_sustain_samples,
        )
        self._store.append_samples(
            cell, trial_index=trial_index, samples=samples
        )
        self._store.append_trial(cell, summary)
        self._store.append_event(
            unix_time_sec=time.time(),
            monotonic_time_sec=time.monotonic(),
            event="trial_saved",
            phase=ProfilerPhase.SAVE_CHECKPOINT.name,
            cell=cell,
            trial_index=trial_index,
            detail=(
                f"valid={summary.valid};reason={summary.rejection_reason or ''};"
                f"flags={summary.quality_flags}"
            ),
        )
        pending = self._store.pending_cells()
        self._controller.acknowledge_checkpoint(
            pending[0] if pending else None
        )
        self._trial_started_sec = None
        self._trial_samples = []

    def _classify_unreachable_and_advance(self) -> None:
        cell = self._controller.cell
        classification = (
            "limiter_bound" if cell.speed_kph >= 185 else "unreachable"
        )
        self._store.classify_cell(cell, classification)
        self._store.append_event(
            unix_time_sec=time.time(),
            monotonic_time_sec=time.monotonic(),
            event="cell_classified",
            phase=ProfilerPhase.SAVE_CHECKPOINT.name,
            cell=cell,
            trial_index=self._store.attempted_trial_count(cell) + 1,
            detail=classification,
        )
        pending = self._store.pending_cells()
        self._controller.acknowledge_checkpoint(
            pending[0] if pending else None
        )
        self._trial_started_sec = None
        self._trial_samples = []

    def _write_progress(
        self, now_sec: float, decision: ControlDecision
    ) -> None:
        phase_changed = decision.phase is not self._last_phase
        if not phase_changed and now_sec - self._last_state_write_sec < 1.0:
            return
        pending_count = len(self._store.pending_cells())
        completed_count = len(self._store.cells) - pending_count
        self._store.write_state(
            {
                "phase": decision.phase.name,
                "speed_kph": self._controller.cell.speed_kph,
                "command_kind": self._controller.cell.command_kind,
                "command_percent": self._controller.cell.command_percent,
                "completed_cell_count": completed_count,
                "total_cell_count": len(self._store.cells),
                "pending_cell_count": pending_count,
                "fault": decision.fault,
                "current_trial_index": (
                    self._store.attempted_trial_count(
                        self._controller.cell
                    )
                    + 1
                ),
                "current_speed_mps": (
                    abs(float(self._latest_status.signed_velocity))
                    if self._latest_status is not None
                    else None
                ),
                "updated_unix_sec": time.time(),
            }
        )
        if phase_changed:
            self._store.append_event(
                unix_time_sec=time.time(),
                monotonic_time_sec=now_sec,
                event="phase_changed",
                phase=decision.phase.name,
                cell=self._controller.cell,
                trial_index=(
                    self._store.attempted_trial_count(
                        self._controller.cell
                    )
                    + 1
                ),
                detail=decision.fault or "",
            )
        self._last_phase = decision.phase
        self._last_state_write_sec = now_sec

    def request_stop(self, reason: str) -> None:
        self._controller.request_stop(reason)


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[VehicleProfilerNode] = None
    try:
        node = VehicleProfilerNode()
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        if node is not None:
            node.request_stop("signal")
            deadline = time.monotonic() + 15.0
            while (
                rclpy.ok()
                and not node.finished
                and time.monotonic() < deadline
            ):
                rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
