"""ROS-free value types shared by the ESKF experiment harness and analysis."""

from __future__ import annotations

from dataclasses import dataclass
import math


Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


@dataclass(frozen=True)
class ActorIdentity:
    id_value: str
    object_type: str
    client_key: str


@dataclass(frozen=True)
class VehicleCommand:
    throttle: float
    brake: float
    steer: float = 0.0


@dataclass(frozen=True)
class CommandPhase:
    name: str
    duration_sec: float
    throttle: float
    brake: float


@dataclass(frozen=True)
class SafetyLimits:
    maximum_start_speed_mps: float
    maximum_speed_mps: float
    maximum_travel_m: float
    truth_stale_timeout_sec: float
    estimator_stale_timeout_sec: float
    command_rate_hz: float
    maximum_command_delta_per_sec: float
    stopped_speed_mps: float
    stopped_stable_duration_sec: float


@dataclass(frozen=True)
class TruthSample:
    receipt_monotonic_ns: int
    rpc_start_monotonic_ns: int
    simulator_timestamp: int | None
    position_xyz: Vector3
    orientation_xyzw: Quaternion
    world_velocity_xyz: Vector3
    world_acceleration_xyz: Vector3
    gear_mode: str
    throttle: float
    brake: float
    steer: float
    collision_object_ids: tuple[str, ...]


def truth_freshness_error(
    truth: TruthSample, now_monotonic_ns: int, timeout_sec: float
) -> str | None:
    """Return a timing-contract error, including time spent inside the RPC."""
    if (
        not math.isfinite(timeout_sec)
        or timeout_sec <= 0.0
        or isinstance(now_monotonic_ns, bool)
        or not isinstance(now_monotonic_ns, int)
    ):
        return "invalid MORAI truth freshness limit"
    start_ns = truth.rpc_start_monotonic_ns
    receipt_ns = truth.receipt_monotonic_ns
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (
        start_ns,
        receipt_ns,
    )):
        return "MORAI truth timestamps are invalid"
    timeout_ns = round(timeout_sec * 1.0e9)
    if receipt_ns > now_monotonic_ns:
        return "MORAI truth receipt time is in the future"
    if now_monotonic_ns - receipt_ns > timeout_ns:
        return "MORAI truth sample is stale"
    if start_ns > receipt_ns:
        return "MORAI truth RPC timestamps are out of order"
    if now_monotonic_ns - start_ns > timeout_ns:
        return "MORAI truth sample is stale after a slow RPC"
    return None


@dataclass(frozen=True)
class CandidateSample:
    candidate: str
    phase: str
    receipt_monotonic_ns: int
    header_stamp_ns: int
    position_xyz: Vector3
    orientation_xyzw: Quaternion
    body_velocity_xyz: Vector3
    accel_bias_xyz: Vector3
    gyro_bias_xyz: Vector3
    accel_bias_covariance_xyz: Vector3
    gyro_bias_covariance_xyz: Vector3


@dataclass(frozen=True)
class AlignedSample:
    candidate: CandidateSample
    truth: TruthSample
    pairing_delta_ns: int
    pairing_age_ns: int


@dataclass(frozen=True)
class FrameOffsets:
    translation_xyz: Vector3
    rotation_xyzw: Quaternion
    settle_phase: str
    settle_sample_count: int


@dataclass(frozen=True)
class ErrorSample:
    candidate: str
    phase: str
    receipt_monotonic_ns: int
    header_stamp_ns: int
    simulator_timestamp: int | None
    pairing_age_ns: int
    position_error_xyz: Vector3
    candidate_body_velocity_xyz: Vector3
    truth_body_velocity_xyz: Vector3
    body_velocity_error_xyz: Vector3
    world_velocity_error_xyz: Vector3
    attitude_error_rpy_rad: Vector3
    accel_bias_xyz: Vector3
    gyro_bias_xyz: Vector3
    accel_bias_covariance_xyz: Vector3
    gyro_bias_covariance_xyz: Vector3
    truth_world_acceleration_xyz: Vector3
    truth_pitch_rad: float
