from dataclasses import FrozenInstanceError, replace
import math

import pytest

from ad_morai_bridge_dev.eskf_experiment.metrics import (
    align_candidate_samples,
    align_parallel_candidate_samples,
    apply_fixed_offsets,
    estimate_settle_offsets,
    phase_correlations,
    quaternion_from_rpy,
    rotate_vector,
    shortest_angle_rad,
    summarize_candidate,
)
from ad_morai_bridge_dev.eskf_experiment.types import (
    CandidateSample,
    TruthSample,
)


NS = 1_000_000_000


def _truth(
    receipt_ns,
    *,
    position=(0.0, 0.0, 0.0),
    orientation=(0.0, 0.0, 0.0, 1.0),
    velocity=(0.0, 0.0, 0.0),
    acceleration=(0.0, 0.0, 0.0),
    simulator_timestamp=0,
):
    return TruthSample(
        receipt_monotonic_ns=receipt_ns,
        rpc_start_monotonic_ns=receipt_ns - 1_000,
        simulator_timestamp=simulator_timestamp,
        position_xyz=position,
        orientation_xyzw=orientation,
        world_velocity_xyz=velocity,
        world_acceleration_xyz=acceleration,
        gear_mode="GEAR_MODE_D",
        throttle=0.0,
        brake=0.0,
        steer=0.0,
        collision_object_ids=(),
    )


def _candidate(
    receipt_ns,
    *,
    phase="settle",
    position=(10.0, -4.0, 2.0),
    orientation=(0.0, 0.0, 0.0, 1.0),
    body_velocity=(0.0, 0.0, 0.0),
    accel_bias=(0.0, 0.0, 0.0),
    accel_bias_covariance=(0.1, 0.1, 0.1),
):
    return CandidateSample(
        candidate="baseline",
        phase=phase,
        receipt_monotonic_ns=receipt_ns,
        header_stamp_ns=receipt_ns - 50,
        position_xyz=position,
        orientation_xyzw=orientation,
        body_velocity_xyz=body_velocity,
        accel_bias_xyz=accel_bias,
        gyro_bias_xyz=(0.0, 0.0, 0.0),
        accel_bias_covariance_xyz=accel_bias_covariance,
        gyro_bias_covariance_xyz=(0.01, 0.01, 0.01),
    )


def test_experiment_samples_are_immutable_value_records():
    sample = _candidate(0)

    with pytest.raises(FrozenInstanceError):
        sample.phase = "accelerate"


def test_alignment_uses_nearest_monotonic_receipt_and_rejects_old_truth():
    truths = [_truth(1 * NS), _truth(2 * NS), _truth(4 * NS)]
    estimates = [_candidate(2 * NS + 40_000_000), _candidate(3 * NS)]

    aligned = align_candidate_samples(
        estimates,
        truths,
        maximum_pair_age_ns=100_000_000,
    )

    assert len(aligned) == 1
    assert aligned[0].truth.receipt_monotonic_ns == 2 * NS
    assert aligned[0].pairing_delta_ns == 40_000_000
    assert aligned[0].pairing_age_ns == 40_000_000


def test_alignment_breaks_equal_distance_ties_toward_earlier_truth():
    aligned = align_candidate_samples(
        [_candidate(2 * NS)],
        [_truth(1 * NS), _truth(3 * NS)],
        maximum_pair_age_ns=NS,
    )

    assert aligned[0].truth.receipt_monotonic_ns == 1 * NS
    assert aligned[0].pairing_delta_ns == NS


def test_parallel_alignment_uses_one_truth_for_each_shared_output_header():
    baseline = replace(
        _candidate(140, phase="track"),
        candidate="baseline",
        header_stamp_ns=100,
    )
    production = replace(
        _candidate(160, phase="track"),
        candidate="production_bias",
        header_stamp_ns=100,
    )

    aligned = align_parallel_candidate_samples(
        {"baseline": [baseline], "production_bias": [production]},
        [_truth(130), _truth(170)],
        maximum_pair_age_ns=100,
    )

    assert aligned["baseline"][0].truth.receipt_monotonic_ns == 130
    assert aligned["production_bias"][0].truth.receipt_monotonic_ns == 130
    assert aligned["baseline"][0].candidate.header_stamp_ns == 100
    assert aligned["production_bias"][0].candidate.header_stamp_ns == 100


def test_parallel_alignment_excludes_headers_missing_from_any_candidate():
    baseline_only = replace(_candidate(100), header_stamp_ns=50)
    shared_baseline = replace(_candidate(200), header_stamp_ns=150)
    shared_production = replace(
        _candidate(210),
        candidate="production_bias",
        header_stamp_ns=150,
    )

    aligned = align_parallel_candidate_samples(
        {
            "baseline": [baseline_only, shared_baseline],
            "production_bias": [shared_production],
        },
        [_truth(100), _truth(200)],
        maximum_pair_age_ns=100,
    )

    assert [item.candidate.header_stamp_ns for item in aligned["baseline"]] == [
        150
    ]
    assert [
        item.candidate.header_stamp_ns for item in aligned["production_bias"]
    ] == [150]


def test_parallel_alignment_excludes_a_header_with_mismatched_phases():
    baseline = replace(_candidate(100, phase="release"), header_stamp_ns=50)
    production = replace(
        _candidate(110, phase="track"),
        candidate="production_bias",
        header_stamp_ns=50,
    )

    aligned = align_parallel_candidate_samples(
        {"baseline": [baseline], "production_bias": [production]},
        [_truth(100)],
        maximum_pair_age_ns=100,
    )

    assert aligned == {"baseline": [], "production_bias": []}


def test_settle_offsets_are_estimated_only_from_named_settle_phase_and_frozen():
    yaw_offset = quaternion_from_rpy(0.0, 0.0, math.pi / 2.0)
    truths = [
        _truth(0, position=(1.0, 2.0, 3.0)),
        _truth(NS, position=(2.0, 3.0, 4.0)),
        _truth(2 * NS, position=(100.0, 200.0, 300.0)),
    ]
    estimates = [
        _candidate(0, position=(8.0, -3.0, 5.0), orientation=yaw_offset),
        _candidate(NS, position=(7.0, -2.0, 6.0), orientation=yaw_offset),
        _candidate(
            2 * NS,
            phase="accelerate",
            position=(-189.0, 98.0, 305.0),
            orientation=quaternion_from_rpy(0.0, 0.0, -1.0),
        ),
    ]
    aligned = align_candidate_samples(estimates, truths, maximum_pair_age_ns=1)

    offsets = estimate_settle_offsets(aligned, settle_phase="settle")

    assert offsets.translation_xyz == pytest.approx((10.0, -4.0, 2.0))
    assert abs(offsets.rotation_xyzw[2]) == pytest.approx(math.sqrt(0.5))
    assert abs(offsets.rotation_xyzw[3]) == pytest.approx(math.sqrt(0.5))
    motion_error = apply_fixed_offsets(aligned[-1], offsets)
    assert motion_error.position_error_xyz == pytest.approx((1.0, 2.0, 3.0))


def test_rigid_truth_world_transform_rotates_motion_vectors_and_position():
    yaw_90 = quaternion_from_rpy(0.0, 0.0, math.pi / 2.0)
    settle = align_candidate_samples(
        [_candidate(0, position=(8.0, -3.0, 5.0), orientation=yaw_90)],
        [_truth(0, position=(1.0, 2.0, 3.0))],
        maximum_pair_age_ns=0,
    )
    offsets = estimate_settle_offsets(settle)
    motion = align_candidate_samples(
        [
            _candidate(
                NS,
                phase="accelerate",
                position=(6.0, -1.0, 7.0),
                orientation=yaw_90,
                body_velocity=(2.0, 0.0, 0.0),
            )
        ],
        [
            _truth(
                NS,
                position=(3.0, 4.0, 5.0),
                velocity=(2.0, 0.0, 0.0),
                acceleration=(1.0, 2.0, 3.0),
            )
        ],
        maximum_pair_age_ns=0,
    )

    error = apply_fixed_offsets(motion[0], offsets)

    assert error.position_error_xyz == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
    assert error.world_velocity_error_xyz == pytest.approx(
        (0.0, 0.0, 0.0), abs=1e-12
    )
    assert error.candidate_body_velocity_xyz == pytest.approx((2.0, 0.0, 0.0))
    assert error.truth_body_velocity_xyz == pytest.approx(
        (2.0, 0.0, 0.0), abs=1e-12
    )
    assert error.truth_world_acceleration_xyz == pytest.approx(
        (-2.0, 1.0, 3.0), abs=1e-12
    )


def test_velocity_errors_keep_body_and_world_frames_distinct():
    yaw_90 = quaternion_from_rpy(0.0, 0.0, math.pi / 2.0)
    aligned = align_candidate_samples(
        [_candidate(0, orientation=yaw_90, body_velocity=(2.0, 0.0, 0.5))],
        [_truth(0, orientation=yaw_90, velocity=(0.0, 1.0, 0.25))],
        maximum_pair_age_ns=0,
    )
    offsets = estimate_settle_offsets(aligned)

    error = apply_fixed_offsets(aligned[0], offsets)

    assert rotate_vector(yaw_90, (2.0, 0.0, 0.5)) == pytest.approx(
        (0.0, 2.0, 0.5), abs=1e-12
    )
    assert error.world_velocity_error_xyz == pytest.approx((0.0, 1.0, 0.25))
    assert error.body_velocity_error_xyz == pytest.approx((1.0, 0.0, 0.25))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (math.pi + 0.2, -math.pi + 0.2),
        (-math.pi - 0.2, math.pi - 0.2),
        (7.0 * math.pi, math.pi),
    ],
)
def test_shortest_angle_wraps_to_closed_pi_interval(value, expected):
    assert shortest_angle_rad(value) == pytest.approx(expected)


def test_summary_reports_centered_rmse_p95_maximum_and_drift_slope():
    truths = [_truth(index * NS) for index in range(4)]
    estimates = [
        _candidate(
            index * NS,
            phase="accelerate",
            position=(10.0, -4.0, 2.0 + index),
        )
        for index in range(4)
    ]
    aligned = align_candidate_samples(estimates, truths, maximum_pair_age_ns=0)
    offsets = estimate_settle_offsets(
        align_candidate_samples(
            [_candidate(-NS), _candidate(0)],
            [_truth(-NS), _truth(0)],
            maximum_pair_age_ns=0,
        )
    )
    errors = [apply_fixed_offsets(sample, offsets) for sample in aligned]

    summary = summarize_candidate(errors)

    assert summary["position"]["z"]["rmse"] == pytest.approx(math.sqrt(3.5))
    assert summary["position"]["z"]["p95_abs"] == pytest.approx(3.0)
    assert summary["position"]["z"]["max_abs"] == pytest.approx(3.0)
    assert summary["position"]["z"]["drift_slope_per_sec"] == pytest.approx(1.0)


def test_summary_reports_stopped_settle_and_three_second_residual_velocity():
    truths = [_truth(index * NS) for index in range(6)]
    estimates = [
        _candidate(
            index * NS,
            phase="stopped",
            body_velocity=(0.0, 0.0, value),
        )
        for index, value in enumerate((0.20, 0.08, 0.03, 0.015, 0.010, 0.005))
    ]
    aligned = align_candidate_samples(estimates, truths, maximum_pair_age_ns=0)
    offsets = estimate_settle_offsets(
        align_candidate_samples(
            [_candidate(-2 * NS), _candidate(-NS)],
            [_truth(-2 * NS), _truth(-NS)],
            maximum_pair_age_ns=0,
        )
    )
    errors = [apply_fixed_offsets(sample, offsets) for sample in aligned]

    summary = summarize_candidate(errors, stopped_velocity_threshold_mps=0.02)

    assert summary["stopped"]["settle_time_sec"] == pytest.approx(3.0)
    assert summary["stopped"][
        "candidate_abs_body_z_velocity_after_3s_mps"
    ] == pytest.approx(0.01)
    assert summary["stopped"][
        "truth_abs_body_z_velocity_after_3s_mps"
    ] == pytest.approx(0.0)
    assert summary["stopped"][
        "error_abs_body_z_velocity_after_3s_mps"
    ] == pytest.approx(0.01)


def test_stopped_acceptance_uses_absolute_candidate_velocity_not_small_error():
    truths = [
        _truth(index * NS, velocity=(0.0, 0.0, 0.03)) for index in range(5)
    ]
    estimates = [
        _candidate(
            index * NS,
            phase="stopped",
            body_velocity=(0.0, 0.0, 0.04),
        )
        for index in range(5)
    ]
    offsets = estimate_settle_offsets(
        align_candidate_samples(
            [_candidate(-NS)], [_truth(-NS)], maximum_pair_age_ns=0
        )
    )
    errors = [
        apply_fixed_offsets(sample, offsets)
        for sample in align_candidate_samples(
            estimates, truths, maximum_pair_age_ns=0
        )
    ]

    summary = summarize_candidate(errors, stopped_velocity_threshold_mps=0.02)

    assert summary["stopped"]["settle_time_sec"] is None
    assert summary["stopped"][
        "candidate_abs_body_z_velocity_after_3s_mps"
    ] == pytest.approx(0.04)
    assert summary["stopped"][
        "truth_abs_body_z_velocity_after_3s_mps"
    ] == pytest.approx(0.03)
    assert summary["stopped"][
        "error_abs_body_z_velocity_after_3s_mps"
    ] == pytest.approx(0.01)


def test_summary_preserves_bias_convergence_and_covariance_history():
    truths = [_truth(index * NS) for index in range(3)]
    estimates = [
        _candidate(
            index * NS,
            phase="accelerate",
            accel_bias=(0.0, 0.0, value),
            accel_bias_covariance=(0.1, 0.1, covariance),
        )
        for index, (value, covariance) in enumerate(
            ((0.0, 0.04), (0.02, 0.02), (0.03, 0.01))
        )
    ]
    aligned = align_candidate_samples(estimates, truths, maximum_pair_age_ns=0)
    offsets = estimate_settle_offsets(
        align_candidate_samples(
            [_candidate(-2 * NS), _candidate(-NS)],
            [_truth(-2 * NS), _truth(-NS)],
            maximum_pair_age_ns=0,
        )
    )
    errors = [apply_fixed_offsets(sample, offsets) for sample in aligned]

    summary = summarize_candidate(errors)

    assert summary["accelerometer_bias"]["z"] == pytest.approx(
        {
            "initial": 0.0,
            "final": 0.03,
            "delta": 0.03,
            "slope_per_sec": 0.015,
        }
    )
    assert summary["accelerometer_bias_covariance"]["z"] == pytest.approx(
        {
            "initial": 0.04,
            "final": 0.01,
            "delta": -0.03,
            "slope_per_sec": -0.015,
        }
    )


def test_phase_correlations_use_truth_acceleration_pitch_and_vertical_errors():
    truths = [
        _truth(
            index * NS,
            orientation=quaternion_from_rpy(0.0, 0.1 * index, 0.0),
            acceleration=(0.0, 0.0, float(index)),
        )
        for index in range(4)
    ]
    estimates = [
        _candidate(
            index * NS,
            phase="brake",
            position=(10.0, -4.0, 2.0 + 2.0 * index),
            body_velocity=(0.0, 0.0, -float(index)),
        )
        for index in range(4)
    ]
    aligned = align_candidate_samples(estimates, truths, maximum_pair_age_ns=0)
    offsets = estimate_settle_offsets(
        align_candidate_samples(
            [_candidate(-2 * NS), _candidate(-NS)],
            [_truth(-2 * NS), _truth(-NS)],
            maximum_pair_age_ns=0,
        )
    )
    errors = [apply_fixed_offsets(sample, offsets) for sample in aligned]

    correlations = phase_correlations(errors)

    assert correlations["brake"]["truth_accel_z_vs_position_error_z"] == pytest.approx(1.0)
    assert correlations["brake"]["truth_pitch_vs_position_error_z"] == pytest.approx(1.0)
    assert correlations["brake"]["truth_accel_z_vs_body_velocity_error_z"] == pytest.approx(-1.0)


def test_invalid_or_missing_settle_data_is_rejected():
    with pytest.raises(ValueError, match="settle"):
        estimate_settle_offsets([])

    aligned = align_candidate_samples(
        [_candidate(0, phase="accelerate")],
        [_truth(0)],
        maximum_pair_age_ns=0,
    )
    with pytest.raises(ValueError, match="settle"):
        estimate_settle_offsets(aligned)
