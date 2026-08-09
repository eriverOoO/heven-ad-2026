"""Deterministic, ROS-free alignment and metrics for parallel ESKF runs."""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
import math
from typing import Iterable, Mapping, Sequence

from ad_morai_bridge_dev.eskf_experiment.types import (
    AlignedSample,
    CandidateSample,
    ErrorSample,
    FrameOffsets,
    Quaternion,
    TruthSample,
    Vector3,
)


def shortest_angle_rad(value: float) -> float:
    """Wrap an angle to (-pi, pi], retaining +pi at the closed boundary."""
    if not math.isfinite(value):
        raise ValueError("angle must be finite")
    wrapped = (value + math.pi) % (2.0 * math.pi) - math.pi
    if math.isclose(wrapped, -math.pi, abs_tol=1.0e-15) and value > 0.0:
        return math.pi
    return wrapped


def quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> Quaternion:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return _normalize_quaternion(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )
    )


def rotate_vector(quaternion: Quaternion, vector: Vector3) -> Vector3:
    q = _normalize_quaternion(quaternion)
    vector_quaternion = (vector[0], vector[1], vector[2], 0.0)
    rotated = _quaternion_multiply(
        _quaternion_multiply(q, vector_quaternion), _quaternion_conjugate(q)
    )
    return (rotated[0], rotated[1], rotated[2])


def align_candidate_samples(
    candidates: Sequence[CandidateSample],
    truth: Sequence[TruthSample],
    *,
    maximum_pair_age_ns: int,
) -> list[AlignedSample]:
    """Pair estimates to nearest truth by local receipt time only."""
    if maximum_pair_age_ns < 0:
        raise ValueError("maximum_pair_age_ns must be nonnegative")
    ordered_truth = sorted(truth, key=lambda sample: sample.receipt_monotonic_ns)
    truth_times = [sample.receipt_monotonic_ns for sample in ordered_truth]
    aligned: list[AlignedSample] = []
    for candidate in candidates:
        if not ordered_truth:
            break
        insertion = bisect_left(truth_times, candidate.receipt_monotonic_ns)
        indices = []
        if insertion > 0:
            indices.append(insertion - 1)
        if insertion < len(ordered_truth):
            indices.append(insertion)
        nearest_index = min(
            indices,
            key=lambda index: (
                abs(candidate.receipt_monotonic_ns - truth_times[index]),
                truth_times[index],
            ),
        )
        nearest = ordered_truth[nearest_index]
        delta = candidate.receipt_monotonic_ns - nearest.receipt_monotonic_ns
        age = abs(delta)
        if age <= maximum_pair_age_ns:
            aligned.append(
                AlignedSample(
                    candidate=candidate,
                    truth=nearest,
                    pairing_delta_ns=delta,
                    pairing_age_ns=age,
                )
            )
    return aligned


def align_parallel_candidate_samples(
    candidates_by_name: Mapping[str, Sequence[CandidateSample]],
    truth: Sequence[TruthSample],
    *,
    maximum_pair_age_ns: int,
) -> dict[str, list[AlignedSample]]:
    """Pair common output headers from every candidate to one shared truth."""
    if maximum_pair_age_ns < 0:
        raise ValueError("maximum_pair_age_ns must be nonnegative")
    indexed: dict[str, dict[int, CandidateSample]] = {}
    for name, samples in candidates_by_name.items():
        by_header: dict[int, CandidateSample] = {}
        for sample in samples:
            if sample.candidate != name:
                raise ValueError("candidate mapping name does not match sample")
            if sample.header_stamp_ns in by_header:
                raise ValueError("duplicate candidate output header stamp")
            by_header[sample.header_stamp_ns] = sample
        indexed[name] = by_header
    result = {name: [] for name in indexed}
    if not indexed or not truth:
        return result

    common_headers = set.intersection(
        *(set(samples) for samples in indexed.values())
    )
    ordered_truth = sorted(truth, key=lambda sample: sample.receipt_monotonic_ns)
    truth_times = [sample.receipt_monotonic_ns for sample in ordered_truth]
    names = tuple(indexed)
    for header_stamp_ns in sorted(common_headers):
        group = tuple(indexed[name][header_stamp_ns] for name in names)
        if len({sample.phase for sample in group}) != 1:
            continue
        reference_receipt_ns = sum(
            sample.receipt_monotonic_ns for sample in group
        ) // len(group)
        insertion = bisect_left(truth_times, reference_receipt_ns)
        indices = []
        if insertion > 0:
            indices.append(insertion - 1)
        if insertion < len(ordered_truth):
            indices.append(insertion)
        nearest_index = min(
            indices,
            key=lambda index: (
                abs(reference_receipt_ns - truth_times[index]),
                truth_times[index],
            ),
        )
        nearest = ordered_truth[nearest_index]
        deltas = tuple(
            sample.receipt_monotonic_ns - nearest.receipt_monotonic_ns
            for sample in group
        )
        if any(abs(delta) > maximum_pair_age_ns for delta in deltas):
            continue
        for name, sample, delta in zip(names, group, deltas):
            result[name].append(
                AlignedSample(
                    candidate=sample,
                    truth=nearest,
                    pairing_delta_ns=delta,
                    pairing_age_ns=abs(delta),
                )
            )
    return result


def estimate_settle_offsets(
    samples: Sequence[AlignedSample],
    *,
    settle_phase: str = "settle",
) -> FrameOffsets:
    """Estimate the rigid truth-world to estimator-world transform from settle."""
    settle = [sample for sample in samples if sample.candidate.phase == settle_phase]
    if not settle:
        raise ValueError(f"no aligned samples for settle phase {settle_phase!r}")
    rotation_offsets = [
        _quaternion_multiply(
            _normalize_quaternion(sample.candidate.orientation_xyzw),
            _quaternion_conjugate(
                _normalize_quaternion(sample.truth.orientation_xyzw)
            ),
        )
        for sample in settle
    ]
    rotation = _mean_quaternion(rotation_offsets)
    translations = [
        _subtract(
            sample.candidate.position_xyz,
            rotate_vector(rotation, sample.truth.position_xyz),
        )
        for sample in settle
    ]
    return FrameOffsets(
        translation_xyz=_mean_vector(translations),
        rotation_xyzw=rotation,
        settle_phase=settle_phase,
        settle_sample_count=len(settle),
    )


def apply_fixed_offsets(sample: AlignedSample, offsets: FrameOffsets) -> ErrorSample:
    truth_world_rotation = _normalize_quaternion(offsets.rotation_xyzw)
    corrected_truth_position = _add(
        rotate_vector(truth_world_rotation, sample.truth.position_xyz),
        offsets.translation_xyz,
    )
    corrected_truth_orientation = _normalize_quaternion(
        _quaternion_multiply(
            truth_world_rotation, sample.truth.orientation_xyzw
        )
    )
    corrected_truth_world_velocity = rotate_vector(
        truth_world_rotation, sample.truth.world_velocity_xyz
    )
    corrected_truth_world_acceleration = rotate_vector(
        truth_world_rotation, sample.truth.world_acceleration_xyz
    )
    estimate_orientation = _normalize_quaternion(
        sample.candidate.orientation_xyzw
    )
    estimate_world_velocity = rotate_vector(
        estimate_orientation, sample.candidate.body_velocity_xyz
    )
    truth_body_velocity = rotate_vector(
        _quaternion_conjugate(corrected_truth_orientation),
        corrected_truth_world_velocity,
    )
    attitude_error = _quaternion_multiply(
        _quaternion_conjugate(corrected_truth_orientation), estimate_orientation
    )
    return ErrorSample(
        candidate=sample.candidate.candidate,
        phase=sample.candidate.phase,
        receipt_monotonic_ns=sample.candidate.receipt_monotonic_ns,
        header_stamp_ns=sample.candidate.header_stamp_ns,
        simulator_timestamp=sample.truth.simulator_timestamp,
        pairing_age_ns=sample.pairing_age_ns,
        position_error_xyz=_subtract(
            sample.candidate.position_xyz, corrected_truth_position
        ),
        candidate_body_velocity_xyz=sample.candidate.body_velocity_xyz,
        truth_body_velocity_xyz=truth_body_velocity,
        body_velocity_error_xyz=_subtract(
            sample.candidate.body_velocity_xyz, truth_body_velocity
        ),
        world_velocity_error_xyz=_subtract(
            estimate_world_velocity, corrected_truth_world_velocity
        ),
        attitude_error_rpy_rad=_quaternion_to_rpy(attitude_error),
        accel_bias_xyz=sample.candidate.accel_bias_xyz,
        gyro_bias_xyz=sample.candidate.gyro_bias_xyz,
        accel_bias_covariance_xyz=sample.candidate.accel_bias_covariance_xyz,
        gyro_bias_covariance_xyz=sample.candidate.gyro_bias_covariance_xyz,
        truth_world_acceleration_xyz=corrected_truth_world_acceleration,
        truth_pitch_rad=_quaternion_to_rpy(corrected_truth_orientation)[1],
    )


def summarize_candidate(
    samples: Sequence[ErrorSample],
    *,
    stopped_phase: str = "stopped",
    stopped_velocity_threshold_mps: float = 0.02,
) -> dict[str, object]:
    if not samples:
        raise ValueError("cannot summarize an empty candidate sample set")
    if not math.isfinite(stopped_velocity_threshold_mps) or (
        stopped_velocity_threshold_mps < 0.0
    ):
        raise ValueError("stopped velocity threshold must be finite and nonnegative")
    ordered = sorted(samples, key=lambda sample: sample.receipt_monotonic_ns)
    start_ns = ordered[0].receipt_monotonic_ns
    times_sec = [
        (sample.receipt_monotonic_ns - start_ns) * 1.0e-9 for sample in ordered
    ]
    result: dict[str, object] = {
        "sample_count": len(ordered),
        "maximum_pair_age_ns": max(sample.pairing_age_ns for sample in ordered),
        "position": _vector_summary(
            ordered, times_sec, lambda sample: sample.position_error_xyz
        ),
        "body_velocity": _vector_summary(
            ordered, times_sec, lambda sample: sample.body_velocity_error_xyz
        ),
        "world_velocity": _vector_summary(
            ordered, times_sec, lambda sample: sample.world_velocity_error_xyz
        ),
        "attitude": _vector_summary(
            ordered,
            times_sec,
            lambda sample: sample.attitude_error_rpy_rad,
            axis_names=("roll", "pitch", "yaw"),
        ),
        "accelerometer_bias": _vector_history_summary(
            ordered, times_sec, lambda sample: sample.accel_bias_xyz
        ),
        "gyroscope_bias": _vector_history_summary(
            ordered, times_sec, lambda sample: sample.gyro_bias_xyz
        ),
        "accelerometer_bias_covariance": _vector_history_summary(
            ordered, times_sec, lambda sample: sample.accel_bias_covariance_xyz
        ),
        "gyroscope_bias_covariance": _vector_history_summary(
            ordered, times_sec, lambda sample: sample.gyro_bias_covariance_xyz
        ),
        "phase_correlations": phase_correlations(ordered),
    }
    stopped = [sample for sample in ordered if sample.phase == stopped_phase]
    result["stopped"] = _stopped_summary(
        stopped, threshold_mps=stopped_velocity_threshold_mps
    )
    return result


def phase_correlations(
    samples: Sequence[ErrorSample],
) -> dict[str, dict[str, float | None]]:
    grouped: dict[str, list[ErrorSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.phase].append(sample)
    result: dict[str, dict[str, float | None]] = {}
    for phase, phase_samples in sorted(grouped.items()):
        accel_z = [sample.truth_world_acceleration_xyz[2] for sample in phase_samples]
        pitch = [sample.truth_pitch_rad for sample in phase_samples]
        position_z = [sample.position_error_xyz[2] for sample in phase_samples]
        body_velocity_z = [
            sample.body_velocity_error_xyz[2] for sample in phase_samples
        ]
        result[phase] = {
            "truth_accel_z_vs_position_error_z": _correlation(
                accel_z, position_z
            ),
            "truth_pitch_vs_position_error_z": _correlation(pitch, position_z),
            "truth_accel_z_vs_body_velocity_error_z": _correlation(
                accel_z, body_velocity_z
            ),
            "truth_pitch_vs_body_velocity_error_z": _correlation(
                pitch, body_velocity_z
            ),
        }
    return result


def _vector_summary(
    samples: Sequence[ErrorSample],
    times_sec: Sequence[float],
    getter,
    *,
    axis_names: tuple[str, str, str] = ("x", "y", "z"),
) -> dict[str, dict[str, float]]:
    vectors = [getter(sample) for sample in samples]
    return {
        name: _scalar_summary([vector[index] for vector in vectors], times_sec)
        for index, name in enumerate(axis_names)
    }


def _scalar_summary(
    values: Sequence[float], times_sec: Sequence[float]
) -> dict[str, float]:
    _require_finite(values, "metric values")
    absolute = [abs(value) for value in values]
    return {
        "rmse": math.sqrt(sum(value * value for value in values) / len(values)),
        "p95_abs": _percentile_nearest_rank(absolute, 95.0),
        "max_abs": max(absolute),
        "drift_slope_per_sec": _least_squares_slope(times_sec, values),
    }


def _vector_history_summary(
    samples: Sequence[ErrorSample], times_sec: Sequence[float], getter
) -> dict[str, dict[str, float]]:
    vectors = [getter(sample) for sample in samples]
    return {
        axis_name: _history_summary(
            [vector[index] for vector in vectors], times_sec
        )
        for index, axis_name in enumerate(("x", "y", "z"))
    }


def _history_summary(
    values: Sequence[float], times_sec: Sequence[float]
) -> dict[str, float]:
    _require_finite(values, "history values")
    return {
        "initial": values[0],
        "final": values[-1],
        "delta": values[-1] - values[0],
        "slope_per_sec": _least_squares_slope(times_sec, values),
    }


def _stopped_summary(
    samples: Sequence[ErrorSample], *, threshold_mps: float
) -> dict[str, float | int | None]:
    if not samples:
        return {
            "sample_count": 0,
            "settle_time_sec": None,
            "candidate_abs_body_z_velocity_after_3s_mps": None,
            "truth_abs_body_z_velocity_after_3s_mps": None,
            "error_abs_body_z_velocity_after_3s_mps": None,
        }
    ordered = sorted(samples, key=lambda sample: sample.receipt_monotonic_ns)
    start_ns = ordered[0].receipt_monotonic_ns
    candidate_absolute_vz = [
        abs(sample.candidate_body_velocity_xyz[2]) for sample in ordered
    ]
    truth_absolute_vz = [
        abs(sample.truth_body_velocity_xyz[2]) for sample in ordered
    ]
    error_absolute_vz = [
        abs(sample.body_velocity_error_xyz[2]) for sample in ordered
    ]
    settle_time = None
    for index, sample in enumerate(ordered):
        if all(value <= threshold_mps for value in candidate_absolute_vz[index:]):
            settle_time = (sample.receipt_monotonic_ns - start_ns) * 1.0e-9
            break
    residual_indices = [
        index
        for index, sample in enumerate(ordered)
        if sample.receipt_monotonic_ns - start_ns >= 3_000_000_000
    ]

    def residual_mean(values: Sequence[float]) -> float | None:
        residual = [values[index] for index in residual_indices]
        return sum(residual) / len(residual) if residual else None

    return {
        "sample_count": len(ordered),
        "settle_time_sec": settle_time,
        "candidate_abs_body_z_velocity_after_3s_mps": residual_mean(
            candidate_absolute_vz
        ),
        "truth_abs_body_z_velocity_after_3s_mps": residual_mean(
            truth_absolute_vz
        ),
        "error_abs_body_z_velocity_after_3s_mps": residual_mean(
            error_absolute_vz
        ),
    }


def _percentile_nearest_rank(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered) / 100.0))
    return ordered[rank - 1]


def _least_squares_slope(times: Sequence[float], values: Sequence[float]) -> float:
    if len(times) != len(values) or not values:
        raise ValueError("times and values must have equal nonzero length")
    _require_finite(times, "metric times")
    _require_finite(values, "metric values")
    mean_time = sum(times) / len(times)
    mean_value = sum(values) / len(values)
    denominator = sum((value - mean_time) ** 2 for value in times)
    if denominator == 0.0:
        return 0.0
    return sum(
        (time_value - mean_time) * (value - mean_value)
        for time_value, value in zip(times, values)
    ) / denominator


def _correlation(lhs: Sequence[float], rhs: Sequence[float]) -> float | None:
    if len(lhs) != len(rhs) or len(lhs) < 2:
        return None
    _require_finite(lhs, "correlation values")
    _require_finite(rhs, "correlation values")
    lhs_mean = sum(lhs) / len(lhs)
    rhs_mean = sum(rhs) / len(rhs)
    lhs_delta = [value - lhs_mean for value in lhs]
    rhs_delta = [value - rhs_mean for value in rhs]
    denominator = math.sqrt(
        sum(value * value for value in lhs_delta)
        * sum(value * value for value in rhs_delta)
    )
    if denominator == 0.0:
        return None
    return sum(
        left * right for left, right in zip(lhs_delta, rhs_delta)
    ) / denominator


def _normalize_quaternion(quaternion: Quaternion) -> Quaternion:
    _require_finite(quaternion, "quaternion")
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm <= 0.0:
        raise ValueError("quaternion norm must be positive")
    return tuple(value / norm for value in quaternion)  # type: ignore[return-value]


def _quaternion_conjugate(quaternion: Quaternion) -> Quaternion:
    return (-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3])


def _quaternion_multiply(lhs: Quaternion, rhs: Quaternion) -> Quaternion:
    lx, ly, lz, lw = lhs
    rx, ry, rz, rw = rhs
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _mean_quaternion(quaternions: Sequence[Quaternion]) -> Quaternion:
    reference = _normalize_quaternion(quaternions[0])
    accumulated = [0.0, 0.0, 0.0, 0.0]
    for quaternion in quaternions:
        normalized = _normalize_quaternion(quaternion)
        if sum(left * right for left, right in zip(reference, normalized)) < 0.0:
            normalized = tuple(-value for value in normalized)  # type: ignore[assignment]
        for index, value in enumerate(normalized):
            accumulated[index] += value
    return _normalize_quaternion(tuple(accumulated))  # type: ignore[arg-type]


def _quaternion_to_rpy(quaternion: Quaternion) -> Vector3:
    x, y, z, w = _normalize_quaternion(quaternion)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_input = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(pitch_input)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return (
        shortest_angle_rad(roll),
        shortest_angle_rad(pitch),
        shortest_angle_rad(yaw),
    )


def _add(lhs: Vector3, rhs: Vector3) -> Vector3:
    return tuple(left + right for left, right in zip(lhs, rhs))  # type: ignore[return-value]


def _subtract(lhs: Vector3, rhs: Vector3) -> Vector3:
    return tuple(left - right for left, right in zip(lhs, rhs))  # type: ignore[return-value]


def _mean_vector(vectors: Iterable[Vector3]) -> Vector3:
    values = list(vectors)
    if not values:
        raise ValueError("mean vector requires at least one value")
    return tuple(
        sum(vector[index] for vector in values) / len(values) for index in range(3)
    )  # type: ignore[return-value]


def _require_finite(values: Iterable[float], label: str) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{label} must be finite")
