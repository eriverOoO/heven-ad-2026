from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Iterable, Optional, Sequence


COMMAND_KINDS = ("accelerator", "brake")
DEFAULT_SPEEDS_KPH = tuple(sorted((*range(0, 186, 5), 57)))


@dataclass(frozen=True, order=True)
class ExperimentCell:
    speed_kph: int
    command_kind: str
    command_percent: int

    def __post_init__(self) -> None:
        if self.command_kind not in COMMAND_KINDS:
            raise ValueError(f"unsupported command kind: {self.command_kind}")
        if self.speed_kph < 0:
            raise ValueError("speed must be non-negative")
        if not 0 <= self.command_percent <= 100:
            raise ValueError("command percent must be in [0, 100]")


@dataclass(frozen=True)
class TrialSample:
    elapsed_sec: float
    velocity_mps: float
    acceleration_mps2: float
    monotonic_time_sec: float = 0.0
    device_time_sec: float = 0.0
    sample_phase: str = "command"
    collision: bool = False
    ros_time_sec: float = 0.0
    requested_accelerator: float = 0.0
    requested_brake: float = 0.0
    echoed_accelerator: float = 0.0
    echoed_brake: float = 0.0
    position_x_m: float = 0.0
    position_y_m: float = 0.0
    yaw_rad: float = 0.0
    gear: int = 0
    link_id: str = ""
    ctrl_mode: int = 0
    map_data_id: int = 0
    steering_rad: float = 0.0
    velocity_x_mps: float = 0.0
    velocity_y_mps: float = 0.0
    velocity_z_mps: float = 0.0
    acceleration_y_mps2: float = 0.0
    acceleration_z_mps2: float = 0.0
    angular_velocity_x_radps: float = 0.0
    angular_velocity_y_radps: float = 0.0
    angular_velocity_z_radps: float = 0.0


@dataclass(frozen=True)
class TrialSummary:
    valid: bool
    rejection_reason: Optional[str]
    sample_count: int
    mean_acceleration_mps2: Optional[float]
    median_acceleration_mps2: Optional[float]
    acceleration_stddev_mps2: Optional[float]
    acceleration_mad_mps2: Optional[float]
    minimum_acceleration_mps2: Optional[float]
    maximum_acceleration_mps2: Optional[float]
    velocity_derived_acceleration_mps2: Optional[float]
    cross_check_disagreement_mps2: Optional[float]
    peak_abs_jerk_mps3: Optional[float]
    effective_acceleration_mps2: Optional[float] = None
    acceleration_source: str = "unavailable"
    baseline_sample_count: int = 0
    measurement_duration_sec: Optional[float] = None
    sample_rate_hz: Optional[float] = None
    maximum_sample_gap_sec: Optional[float] = None
    baseline_mean_acceleration_mps2: Optional[float] = None
    baseline_acceleration_stddev_mps2: Optional[float] = None
    initial_speed_mps: Optional[float] = None
    final_speed_mps: Optional[float] = None
    minimum_speed_mps: Optional[float] = None
    speed_drop_mps: Optional[float] = None
    distance_travelled_m: Optional[float] = None
    mean_deceleration_mps2: Optional[float] = None
    median_deceleration_mps2: Optional[float] = None
    peak_deceleration_mps2: Optional[float] = None
    p95_deceleration_mps2: Optional[float] = None
    velocity_regression_acceleration_mps2: Optional[float] = None
    command_echo_delay_sec: Optional[float] = None
    deceleration_onset_delay_sec: Optional[float] = None
    peak_deceleration_time_sec: Optional[float] = None
    mean_echoed_brake: Optional[float] = None
    maximum_echoed_brake: Optional[float] = None
    mean_brake_echo_error: Optional[float] = None
    quality_flags: str = ""


def build_cells(
    speeds: Sequence[int] = DEFAULT_SPEEDS_KPH,
    commands: Sequence[int] = tuple(range(0, 101, 10)),
    command_kinds: Sequence[str] = ("brake",),
) -> tuple[ExperimentCell, ...]:
    return tuple(
        ExperimentCell(speed, command_kind, command)
        for speed in speeds
        for command in commands
        for command_kind in command_kinds
    )


def _invalid(reason: str, sample_count: int = 0) -> TrialSummary:
    return TrialSummary(
        valid=False,
        rejection_reason=reason,
        sample_count=sample_count,
        mean_acceleration_mps2=None,
        median_acceleration_mps2=None,
        acceleration_stddev_mps2=None,
        acceleration_mad_mps2=None,
        minimum_acceleration_mps2=None,
        maximum_acceleration_mps2=None,
        velocity_derived_acceleration_mps2=None,
        cross_check_disagreement_mps2=None,
        peak_abs_jerk_mps3=None,
        effective_acceleration_mps2=None,
        acceleration_source="unavailable",
        quality_flags=reason,
    )


def _linear_regression_slope(
    samples: Sequence[TrialSample],
) -> Optional[float]:
    if len(samples) < 2:
        return None
    mean_time = statistics.fmean(sample.elapsed_sec for sample in samples)
    mean_velocity = statistics.fmean(
        sample.velocity_mps for sample in samples
    )
    denominator = sum(
        (sample.elapsed_sec - mean_time) ** 2 for sample in samples
    )
    if denominator <= 1e-12:
        return None
    return sum(
        (sample.elapsed_sec - mean_time)
        * (sample.velocity_mps - mean_velocity)
        for sample in samples
    ) / denominator


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = percentile * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _first_sustained_index(
    values: Sequence[bool], required_count: int
) -> Optional[int]:
    consecutive = 0
    for index, value in enumerate(values):
        consecutive = consecutive + 1 if value else 0
        if consecutive >= required_count:
            return index - required_count + 1
    return None


def _distance(samples: Sequence[TrialSample]) -> float:
    return sum(
        math.hypot(
            current.position_x_m - previous.position_x_m,
            current.position_y_m - previous.position_y_m,
        )
        for previous, current in zip(samples, samples[1:])
    )


def summarize_trial(
    samples: Sequence[TrialSample],
    *,
    discard_sec: float,
    window_end_sec: float,
    maximum_gap_sec: float = 0.1,
    command_kind: str = "accelerator",
    command_value: float = 0.0,
    target_speed_kph: float | None = None,
    minimum_samples: int = 5,
    minimum_braking_speed_mps: float = 0.5,
    echo_fraction: float = 0.8,
    onset_minimum_deceleration_mps2: float = 0.15,
    onset_sigma_multiplier: float = 3.0,
    onset_sustain_samples: int = 2,
) -> TrialSummary:
    if command_kind not in COMMAND_KINDS:
        raise ValueError(f"unsupported command kind: {command_kind}")
    if window_end_sec <= discard_sec:
        raise ValueError("window end must be after discard time")
    if not 0.0 <= command_value <= 1.0:
        raise ValueError("command value must be in [0, 1]")

    command_samples = tuple(
        sample
        for sample in samples
        if 0.0 <= sample.elapsed_sec <= window_end_sec
    )
    baseline = tuple(sample for sample in samples if sample.elapsed_sec < 0.0)
    if len(command_samples) < minimum_samples:
        return _invalid("insufficient_command_samples", len(command_samples))

    for sample in (*baseline, *command_samples):
        if not all(
            math.isfinite(value)
            for value in (
                sample.elapsed_sec,
                sample.velocity_mps,
                sample.acceleration_mps2,
            )
        ):
            return _invalid("non_finite_sample", len(command_samples))
        if sample.collision:
            return _invalid("collision", len(command_samples))

    command_time_steps = [
        current.elapsed_sec - previous.elapsed_sec
        for previous, current in zip(command_samples, command_samples[1:])
    ]
    if any(step <= 0.0 for step in command_time_steps):
        return _invalid(
            "non_monotonic_timestamp", len(command_samples)
        )
    if any(step > maximum_gap_sec + 1e-9 for step in command_time_steps):
        return _invalid("stale_sample_gap", len(command_samples))

    baseline_accelerations = [
        sample.acceleration_mps2 for sample in baseline
    ]
    baseline_mean = (
        statistics.fmean(baseline_accelerations)
        if baseline_accelerations
        else 0.0
    )
    baseline_stddev = (
        statistics.pstdev(baseline_accelerations)
        if len(baseline_accelerations) >= 2
        else 0.0
    )
    onset_threshold = max(
        onset_minimum_deceleration_mps2,
        onset_sigma_multiplier * baseline_stddev,
    )
    velocity_step_accelerations: list[Optional[float]] = [None]
    for previous, current, step in zip(
        command_samples,
        command_samples[1:],
        command_time_steps,
    ):
        velocity_step_accelerations.append(
            (current.velocity_mps - previous.velocity_mps) / step
        )
    onset_index = (
        _first_sustained_index(
            [
                (
                    sample.acceleration_mps2
                    <= baseline_mean - onset_threshold
                )
                or (
                    derived_step is not None
                    and derived_step
                    <= baseline_mean - onset_threshold
                )
                for sample, derived_step in zip(
                    command_samples,
                    velocity_step_accelerations,
                    strict=True,
                )
            ],
            onset_sustain_samples,
        )
        if (
            command_kind == "brake"
            and command_value > 0.0
            and target_speed_kph != 0.0
        )
        else None
    )
    onset_delay = (
        command_samples[onset_index].elapsed_sec
        if onset_index is not None
        else None
    )
    measurement_start_sec = (
        onset_delay
        if onset_delay is not None
        else discard_sec
    )
    selected = tuple(
        sample
        for sample in command_samples
        if measurement_start_sec <= sample.elapsed_sec
        and (
            command_kind != "brake"
            or target_speed_kph == 0.0
            or sample.velocity_mps >= minimum_braking_speed_mps
        )
    )
    if len(selected) < minimum_samples:
        return _invalid("insufficient_samples", len(selected))

    time_steps = [
        current.elapsed_sec - previous.elapsed_sec
        for previous, current in zip(selected, selected[1:])
    ]
    elapsed = selected[-1].elapsed_sec - selected[0].elapsed_sec
    if elapsed <= 0.0:
        return _invalid("insufficient_measurement_duration", len(selected))

    accelerations = [sample.acceleration_mps2 for sample in selected]
    median_acceleration = statistics.median(accelerations)
    absolute_deviations = [
        abs(value - median_acceleration) for value in accelerations
    ]
    endpoint_acceleration = (
        selected[-1].velocity_mps - selected[0].velocity_mps
    ) / elapsed
    regression_acceleration = _linear_regression_slope(selected)
    derived_acceleration = (
        regression_acceleration
        if regression_acceleration is not None
        else endpoint_acceleration
    )
    simulator_field_is_stuck = (
        max(abs(value) for value in accelerations) <= 0.02
        and abs(derived_acceleration) >= 0.05
    )
    if simulator_field_is_stuck:
        effective_acceleration = derived_acceleration
        acceleration_source = "velocity_derived"
    else:
        effective_acceleration = median_acceleration
        acceleration_source = "simulator_field"

    jerks = [
        abs(
            (current.acceleration_mps2 - previous.acceleration_mps2)
            / time_step
        )
        for previous, current, time_step in zip(
            selected, selected[1:], time_steps
        )
    ]
    simulator_decelerations = [
        max(0.0, -value) for value in accelerations
    ]
    velocity_decelerations = [
        (
            current.elapsed_sec,
            max(
                0.0,
                -(
                    (current.velocity_mps - previous.velocity_mps)
                    / (current.elapsed_sec - previous.elapsed_sec)
                ),
            ),
        )
        for previous, current in zip(selected, selected[1:])
    ]
    if simulator_field_is_stuck and velocity_decelerations:
        timed_decelerations = velocity_decelerations
    else:
        timed_decelerations = [
            (sample.elapsed_sec, value)
            for sample, value in zip(
                selected, simulator_decelerations, strict=True
            )
        ]
    decelerations = [value for _, value in timed_decelerations]
    peak_deceleration_time, peak_deceleration = max(
        timed_decelerations, key=lambda item: item[1]
    )

    echoed_brakes = [sample.echoed_brake for sample in command_samples]
    echo_threshold = max(0.02, command_value * echo_fraction)
    echo_index = (
        _first_sustained_index(
            [value >= echo_threshold for value in echoed_brakes],
            onset_sustain_samples,
        )
        if command_kind == "brake" and command_value > 0.0
        else None
    )
    echo_delay = (
        command_samples[echo_index].elapsed_sec
        if echo_index is not None
        else None
    )
    mean_echoed_brake = statistics.fmean(
        sample.echoed_brake for sample in selected
    )
    quality_flags = []
    if not baseline:
        quality_flags.append("no_baseline_samples")
    if command_kind == "brake" and command_value > 0.0:
        if echo_delay is None:
            quality_flags.append("command_echo_not_detected")
        if target_speed_kph != 0.0 and onset_delay is None:
            quality_flags.append("deceleration_onset_not_detected")
    if target_speed_kph == 0.0:
        quality_flags.append("stationary_command_check")
    if simulator_field_is_stuck:
        quality_flags.append("simulator_acceleration_field_stuck")
    if len(selected) < 5:
        quality_flags.append("short_measurement_window")
    mean_effective_acceleration = (
        derived_acceleration
        if simulator_field_is_stuck
        else statistics.fmean(accelerations)
    )

    return TrialSummary(
        valid=True,
        rejection_reason=None,
        sample_count=len(selected),
        mean_acceleration_mps2=statistics.fmean(accelerations),
        median_acceleration_mps2=median_acceleration,
        acceleration_stddev_mps2=statistics.pstdev(accelerations),
        acceleration_mad_mps2=statistics.median(absolute_deviations),
        minimum_acceleration_mps2=min(accelerations),
        maximum_acceleration_mps2=max(accelerations),
        velocity_derived_acceleration_mps2=derived_acceleration,
        cross_check_disagreement_mps2=abs(
            median_acceleration - derived_acceleration
        ),
        peak_abs_jerk_mps3=max(jerks, default=0.0),
        effective_acceleration_mps2=effective_acceleration,
        acceleration_source=acceleration_source,
        baseline_sample_count=len(baseline),
        measurement_duration_sec=elapsed,
        sample_rate_hz=(len(selected) - 1) / elapsed,
        maximum_sample_gap_sec=max(time_steps, default=0.0),
        baseline_mean_acceleration_mps2=baseline_mean,
        baseline_acceleration_stddev_mps2=baseline_stddev,
        initial_speed_mps=command_samples[0].velocity_mps,
        final_speed_mps=command_samples[-1].velocity_mps,
        minimum_speed_mps=min(
            sample.velocity_mps for sample in command_samples
        ),
        speed_drop_mps=(
            command_samples[0].velocity_mps
            - command_samples[-1].velocity_mps
        ),
        distance_travelled_m=_distance(command_samples),
        mean_deceleration_mps2=max(0.0, -mean_effective_acceleration),
        median_deceleration_mps2=max(0.0, -median_acceleration),
        peak_deceleration_mps2=peak_deceleration,
        p95_deceleration_mps2=_percentile(decelerations, 0.95),
        velocity_regression_acceleration_mps2=regression_acceleration,
        command_echo_delay_sec=echo_delay,
        deceleration_onset_delay_sec=onset_delay,
        peak_deceleration_time_sec=peak_deceleration_time,
        mean_echoed_brake=mean_echoed_brake,
        maximum_echoed_brake=max(echoed_brakes, default=None),
        mean_brake_echo_error=(
            statistics.fmean(
                abs(sample.echoed_brake - command_value)
                for sample in selected
            )
            if command_kind == "brake"
            else None
        ),
        quality_flags=";".join(quality_flags),
    )


def needs_more_trials(
    summaries: Iterable[TrialSummary],
    attempted_count: int,
    minimum: int,
    maximum: int,
    mad_limit: float,
    disagreement_limit: float,
    repeatability_mad_limit: float = 0.5,
) -> bool:
    if attempted_count >= maximum:
        return False
    valid = tuple(summary for summary in summaries if summary.valid)
    if len(valid) < minimum:
        return True
    if any(
        (summary.acceleration_mad_mps2 or 0.0) > mad_limit
        or (
            summary.acceleration_source != "velocity_derived"
            and (summary.cross_check_disagreement_mps2 or 0.0)
            > disagreement_limit
        )
        for summary in valid
    ):
        return True
    effective_values = [
        summary.effective_acceleration_mps2
        for summary in valid
        if summary.effective_acceleration_mps2 is not None
    ]
    if len(effective_values) < minimum:
        return True
    median_effective = statistics.median(effective_values)
    repeatability_mad = statistics.median(
        abs(value - median_effective) for value in effective_values
    )
    return repeatability_mad > repeatability_mad_limit
