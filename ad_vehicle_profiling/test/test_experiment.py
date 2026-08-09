import math

import pytest

from ad_vehicle_profiling.experiment import (
    DEFAULT_SPEEDS_KPH,
    ExperimentCell,
    TrialSample,
    TrialSummary,
    build_cells,
    needs_more_trials,
    summarize_trial,
)


def _samples(times, accelerations, velocities):
    return tuple(
        TrialSample(
            elapsed_sec=time,
            velocity_mps=velocity,
            acceleration_mps2=acceleration,
        )
        for time, acceleration, velocity in zip(
            times, accelerations, velocities, strict=True
        )
    )


def _summary(*, mad=0.01, disagreement=0.01, valid=True):
    return TrialSummary(
        valid=valid,
        rejection_reason=None if valid else "rejected",
        sample_count=10,
        mean_acceleration_mps2=1.0,
        median_acceleration_mps2=1.0,
        acceleration_stddev_mps2=mad,
        acceleration_mad_mps2=mad,
        minimum_acceleration_mps2=0.9,
        maximum_acceleration_mps2=1.1,
        velocity_derived_acceleration_mps2=1.0 + disagreement,
        cross_check_disagreement_mps2=disagreement,
        peak_abs_jerk_mps3=1.0,
    )


def test_matrix_has_exact_speed_and_command_axes():
    cells = build_cells()

    assert sorted({cell.speed_kph for cell in cells}) == list(
        DEFAULT_SPEEDS_KPH
    )
    assert sorted({cell.command_percent for cell in cells}) == list(
        range(0, 101, 10)
    )
    assert {cell.command_kind for cell in cells} == {"brake"}
    assert len(cells) == 39 * 11


def test_matrix_order_starts_with_low_risk_cells():
    cells = build_cells()

    assert cells[0] == ExperimentCell(0, "brake", 0)
    assert cells[10] == ExperimentCell(0, "brake", 100)
    assert cells[11] == ExperimentCell(5, "brake", 0)


def test_summary_discards_transport_delay_and_uses_median():
    samples = _samples(
        times=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        accelerations=[0.0, 5.0, 1.0, 1.1, 0.9, 1.0, 1.0],
        velocities=[5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6],
    )

    summary = summarize_trial(
        samples,
        discard_sec=0.2,
        window_end_sec=0.6,
    )

    assert summary.valid
    assert summary.sample_count == 5
    assert summary.median_acceleration_mps2 == pytest.approx(1.0)
    assert summary.velocity_derived_acceleration_mps2 == pytest.approx(1.0)


def test_summary_rejects_stale_sample_gap():
    samples = _samples(
        times=[0.2, 0.25, 0.3, 0.45, 0.5],
        accelerations=[1.0] * 5,
        velocities=[5.0, 5.05, 5.1, 5.25, 5.3],
    )

    summary = summarize_trial(
        samples,
        discard_sec=0.2,
        window_end_sec=0.6,
        maximum_gap_sec=0.1,
    )

    assert not summary.valid
    assert summary.rejection_reason == "stale_sample_gap"


def test_summary_uses_velocity_derivative_when_simulator_field_is_stuck_zero():
    samples = _samples(
        times=[0.2, 0.3, 0.4, 0.5, 0.6],
        accelerations=[0.0] * 5,
        velocities=[5.0, 5.1, 5.2, 5.3, 5.4],
    )

    summary = summarize_trial(
        samples,
        discard_sec=0.2,
        window_end_sec=0.6,
    )

    assert summary.valid
    assert summary.acceleration_source == "velocity_derived"
    assert summary.effective_acceleration_mps2 == pytest.approx(1.0)
    assert summary.median_acceleration_mps2 == 0.0


def test_summary_rejects_non_finite_measurement():
    samples = _samples(
        times=[0.2, 0.25, 0.3, 0.35, 0.4],
        accelerations=[1.0, 1.0, math.nan, 1.0, 1.0],
        velocities=[5.0, 5.05, 5.1, 5.15, 5.2],
    )

    summary = summarize_trial(
        samples,
        discard_sec=0.2,
        window_end_sec=0.6,
    )

    assert not summary.valid
    assert summary.rejection_reason == "non_finite_sample"


def test_braking_trial_uses_only_samples_before_vehicle_nearly_stops():
    samples = _samples(
        times=[0.2, 0.25, 0.3, 0.35, 0.4],
        accelerations=[-2.0] * 5,
        velocities=[0.8, 0.7, 0.55, 0.45, 0.3],
    )

    summary = summarize_trial(
        samples,
        discard_sec=0.2,
        window_end_sec=0.6,
        command_kind="brake",
        command_value=1.0,
        target_speed_kph=5.0,
        minimum_samples=3,
    )

    assert summary.valid
    assert summary.sample_count == 3
    assert summary.mean_deceleration_mps2 == pytest.approx(2.0)


def test_brake_summary_records_echo_and_physical_onset_delays():
    samples = tuple(
        TrialSample(
            elapsed_sec=time,
            velocity_mps=10.0 - max(0.0, time) * 2.0,
            acceleration_mps2=acceleration,
            echoed_brake=echo,
            position_x_m=10.0 * (time + 0.2),
        )
        for time, acceleration, echo in zip(
            [-0.2, -0.1, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
            [0.01, -0.01, 0.0, -0.1, -2.0, -2.1, -2.0, -1.9, -2.0],
            [0.0, 0.0, 0.0, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5],
            strict=True,
        )
    )

    summary = summarize_trial(
        samples,
        discard_sec=0.1,
        window_end_sec=0.3,
        command_kind="brake",
        command_value=0.5,
        target_speed_kph=36.0,
        minimum_samples=5,
        maximum_gap_sec=0.1,
    )

    assert summary.valid
    assert summary.baseline_sample_count == 2
    assert summary.command_echo_delay_sec == pytest.approx(0.1)
    assert summary.deceleration_onset_delay_sec == pytest.approx(0.05)
    assert summary.mean_deceleration_mps2 > 1.6
    assert summary.peak_deceleration_mps2 == pytest.approx(2.1)
    assert summary.speed_drop_mps == pytest.approx(0.6)


def test_zero_speed_brake_is_labeled_as_stationary_command_check():
    samples = tuple(
        TrialSample(
            elapsed_sec=time,
            velocity_mps=0.0,
            acceleration_mps2=0.0,
            echoed_brake=0.4 if time >= 0.05 else 0.0,
        )
        for time in [-0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2]
    )

    summary = summarize_trial(
        samples,
        discard_sec=0.05,
        window_end_sec=0.2,
        command_kind="brake",
        command_value=0.4,
        target_speed_kph=0.0,
        minimum_samples=3,
    )

    assert summary.valid
    assert summary.mean_deceleration_mps2 == 0.0
    assert "stationary_command_check" in summary.quality_flags


def test_unstable_cell_repeats_but_never_exceeds_seven_attempts():
    unstable = [_summary(mad=0.4) for _ in range(6)]

    assert needs_more_trials(
        unstable,
        attempted_count=6,
        minimum=3,
        maximum=7,
        mad_limit=0.15,
        disagreement_limit=0.2,
    )
    assert not needs_more_trials(
        unstable + [_summary(mad=0.4)],
        attempted_count=7,
        minimum=3,
        maximum=7,
        mad_limit=0.15,
        disagreement_limit=0.2,
    )


def test_invalid_trials_do_not_satisfy_minimum_valid_count():
    summaries = [_summary(valid=False), _summary(), _summary()]

    assert needs_more_trials(
        summaries,
        attempted_count=3,
        minimum=3,
        maximum=7,
        mad_limit=0.15,
        disagreement_limit=0.2,
    )


def test_cross_trial_variation_requests_more_repeats():
    summaries = [
        TrialSummary(
            **{
                **_summary().__dict__,
                "effective_acceleration_mps2": value,
            }
        )
        for value in (-4.0, -6.0, -8.0)
    ]

    assert needs_more_trials(
        summaries,
        attempted_count=3,
        minimum=3,
        maximum=7,
        mad_limit=0.15,
        disagreement_limit=0.2,
        repeatability_mad_limit=0.5,
    )
