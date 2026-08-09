import json
import math

import pytest

from ad_morai_bridge_dev.perception.metrics import (
    MetricValue,
    ObjectFrame,
    ObjectSample,
    PredictionPoint,
    RunMetadata,
    associate_frame,
    score_detections,
    score_predictions,
    score_tracking,
    summarize_rate,
    write_reports,
)


def sample(
    object_id,
    x,
    *,
    y=0.0,
    vx=2.0,
    vy=0.0,
    stamp_ns=1_000_000_000,
    frame_id="odom",
):
    return ObjectSample(
        stamp_ns=stamp_ns,
        frame_id=frame_id,
        object_id=object_id,
        x_m=x,
        y_m=y,
        vx_mps=vx,
        vy_mps=vy,
    )


def frame(stamp_ns, *objects, frame_id="odom"):
    return ObjectFrame(
        stamp_ns=stamp_ns,
        frame_id=frame_id,
        objects=tuple(objects),
    )


def test_rate_summary_counts_large_gaps_without_inventing_receive_rate():
    result = summarize_rate(
        (1_000_000_000, 1_100_000_000, 1_400_000_000),
        maximum_gap_s=0.15,
    )

    assert result.count == 3
    assert result.rate_hz == pytest.approx(5.0)
    assert result.maximum_gap_s == pytest.approx(0.3)
    assert result.gap_count == 1


def test_nearest_gated_association_and_detection_recall_false_positives():
    truth = frame(
        1_000_000_000,
        sample("actor-a", 0.0),
        sample("actor-b", 10.0),
    )
    detections = frame(
        1_000_000_000,
        sample("det-a", 0.5),
        sample("false-positive", 30.0),
    )

    association = associate_frame(truth, detections, maximum_distance_m=2.0)
    metrics = score_detections((truth,), (detections,), maximum_distance_m=2.0)

    assert [(item.truth_id, item.observation_id) for item in association.matches] == [
        ("actor-a", "det-a")
    ]
    assert association.unmatched_truth_ids == ("actor-b",)
    assert association.unmatched_observation_ids == ("false-positive",)
    assert metrics.recall.value == pytest.approx(0.5)
    assert metrics.recall.support == 2
    assert metrics.recall.expected == 2
    assert metrics.recall.excluded == 0
    assert metrics.false_positives.value == 1
    assert metrics.false_positives_per_frame.value == pytest.approx(1.0)


def test_association_maximizes_cardinality_before_minimizing_distance():
    truth = frame(
        1_000_000_000,
        sample("actor-a", 0.0),
        sample("actor-b", 2.0),
    )
    observations = frame(
        1_000_000_000,
        sample("track-x", 1.1),
        sample("track-y", 3.0),
    )

    result = associate_frame(truth, observations, maximum_distance_m=1.2)

    assert [
        (item.truth_id, item.observation_id) for item in result.matches
    ] == [("actor-a", "track-x"), ("actor-b", "track-y")]
    assert result.unmatched_truth_ids == ()
    assert result.unmatched_observation_ids == ()


def test_association_equal_cost_tie_breaks_by_stable_identifiers():
    stamp = 1_000_000_000
    truth = frame(
        stamp,
        sample("actor-b", 1.0, y=0.0),
        sample("actor-a", -1.0, y=0.0),
    )
    observations = frame(
        stamp,
        sample("track-y", 0.0, y=1.0),
        sample("track-x", 0.0, y=-1.0),
    )

    result = associate_frame(truth, observations, maximum_distance_m=2.0)

    assert [
        (item.truth_id, item.observation_id) for item in result.matches
    ] == [("actor-a", "track-x"), ("actor-b", "track-y")]


def test_association_leaves_every_object_unmatched_outside_gate():
    stamp = 1_000_000_000
    truth = frame(stamp, sample("actor", 0.0))
    observations = frame(stamp, sample("track", 2.0))

    result = associate_frame(truth, observations, maximum_distance_m=1.0)

    assert result.matches == ()
    assert result.unmatched_truth_ids == ("actor",)
    assert result.unmatched_observation_ids == ("track",)


def test_exact_association_prevents_false_recall_loss_and_id_switch():
    first_stamp = 1_000_000_000
    second_stamp = 2_000_000_000
    truth = (
        frame(
            first_stamp,
            sample("actor-a", 0.0, stamp_ns=first_stamp),
            sample("actor-b", 2.0, stamp_ns=first_stamp),
        ),
        frame(
            second_stamp,
            sample("actor-a", 0.0, stamp_ns=second_stamp),
            sample("actor-b", 2.0, stamp_ns=second_stamp),
        ),
    )
    tracks = (
        frame(
            first_stamp,
            sample("track-x", 0.0, stamp_ns=first_stamp),
            sample("track-y", 2.0, stamp_ns=first_stamp),
        ),
        frame(
            second_stamp,
            sample("track-x", 1.1, stamp_ns=second_stamp),
            sample("track-y", 3.0, stamp_ns=second_stamp),
        ),
    )

    detection = score_detections(
        truth, tracks, maximum_distance_m=1.2
    )
    tracking = score_tracking(truth, tracks, maximum_distance_m=1.2)

    assert detection.recall.value == pytest.approx(1.0)
    assert detection.false_positives.value == 0
    assert tracking.missed_actor_frames.value == 0
    assert tracking.drop_episodes.value == 0
    assert tracking.id_switches.value == 0


def test_equal_position_crossing_keeps_stable_ids_without_false_switches():
    stamps = (1_000_000_000, 2_000_000_000, 3_000_000_000)
    truth_positions = ((-1.0, 1.0), (0.0, 0.0), (1.0, -1.0))
    truth = tuple(
        frame(
            stamp,
            sample("actor-a", positions[0], stamp_ns=stamp),
            sample("actor-b", positions[1], stamp_ns=stamp),
        )
        for stamp, positions in zip(stamps, truth_positions)
    )
    tracks = tuple(
        frame(
            stamp,
            sample("track-x", positions[0], stamp_ns=stamp),
            sample("track-y", positions[1], stamp_ns=stamp),
        )
        for stamp, positions in zip(stamps, truth_positions)
    )

    result = score_tracking(truth, tracks, maximum_distance_m=1.0)

    assert result.completeness.value == pytest.approx(1.0)
    assert result.id_switches.value == 0


def test_tracking_delay_drop_and_position_velocity_rmse():
    stamps = tuple(index * 1_000_000_000 for index in range(4))
    truth = tuple(
        frame(
            stamp,
            sample("actor", 2.0 * index, stamp_ns=stamp),
        )
        for index, stamp in enumerate(stamps)
    )
    tracks = (
        frame(stamps[0]),
        frame(
            stamps[1],
            sample("track-1", 2.1, vx=2.2, stamp_ns=stamps[1]),
        ),
        frame(stamps[2]),
        frame(
            stamps[3],
            sample("track-2", 6.3, vx=1.8, stamp_ns=stamps[3]),
        ),
    )

    result = score_tracking(truth, tracks, maximum_distance_m=1.0)

    assert result.initialization_delay_s["actor"].value == pytest.approx(1.0)
    assert result.missed_actor_frames.value == 2
    assert result.drop_episodes.value == 1
    assert result.completeness.value == pytest.approx(0.5)
    assert result.id_switches == MetricValue(
        value=None,
        unavailable_reason=(
            "ID switches are unavailable for all 3 actor transitions"
        ),
        support=0,
        expected=3,
        excluded=3,
    )
    assert result.position_rmse_m.value == pytest.approx(math.sqrt(0.05))
    assert result.position_rmse_m.unavailable_reason is None
    assert result.velocity_rmse_mps.value == pytest.approx(0.2)


def test_prediction_ade_fde_uses_future_truth_at_exact_horizons():
    source_stamp = 1_000_000_000
    truth = (
        frame(
            2_000_000_000,
            sample("actor", 2.0, stamp_ns=2_000_000_000),
        ),
        frame(
            3_000_000_000,
            sample("actor", 4.0, stamp_ns=3_000_000_000),
        ),
    )
    predictions = (
        PredictionPoint(source_stamp, "odom", "track-1", 1.0, 2.5, 0.0),
        PredictionPoint(source_stamp, "odom", "track-1", 2.0, 4.8, 0.0),
    )
    source_associations = {
        (source_stamp, "track-1"): "actor",
    }

    result = score_predictions(
        predictions,
        truth,
        source_associations=source_associations,
        horizons_s=(1.0, 2.0),
    )

    assert result[1.0].ade_m.value == pytest.approx(0.5)
    assert result[1.0].ade_m.support == 1
    assert result[1.0].ade_m.expected == 1
    assert result[1.0].fde_m.value == pytest.approx(0.5)
    assert result[2.0].ade_m.value == pytest.approx(0.65)
    assert result[2.0].fde_m.value == pytest.approx(0.8)


def test_unavailable_fields_and_transform_gaps_are_explicit():
    stamp = 1_000_000_000
    truth = frame(stamp, sample("actor", 0.0, vx=None, vy=None))
    tracks = frame(stamp, sample("track", 0.1, vx=None, vy=None))
    tracking = score_tracking((truth,), (tracks,), maximum_distance_m=1.0)
    assert tracking.velocity_rmse_mps == MetricValue(
        value=None,
        unavailable_reason="velocity unavailable for all 1 actor frames",
        support=0,
        expected=1,
        excluded=1,
    )

    wrong_frame = frame(
        stamp,
        sample("track", 0.1, frame_id="map"),
        frame_id="map",
    )
    association = associate_frame(truth, wrong_frame, maximum_distance_m=1.0)
    assert association.matches == ()
    assert association.unavailable_reason == (
        "common-frame association unavailable: odom != map"
    )

    unavailable_prediction = score_predictions(
        (
            PredictionPoint(stamp, "odom", "track", 1.0, 1.0, 0.0),
        ),
        (),
        source_associations={(stamp, "track"): "actor"},
        horizons_s=(1.0,),
    )
    assert unavailable_prediction[1.0].fde_m.unavailable_reason == (
        "future truth is unavailable"
    )


@pytest.mark.parametrize("scorer", (score_detections, score_tracking))
@pytest.mark.parametrize("stream_shape", ("missing", "extra"))
def test_aggregate_scorers_reject_missing_or_extra_stream_stamps(
    scorer, stream_shape
):
    stamp_a = 1_000_000_000
    stamp_b = 2_000_000_000
    truth = (
        frame(stamp_a, sample("actor", 0.0, stamp_ns=stamp_a)),
        frame(stamp_b, sample("actor", 1.0, stamp_ns=stamp_b)),
    )
    observations = (
        frame(stamp_a, sample("track", 0.0, stamp_ns=stamp_a)),
    )
    if stream_shape == "extra":
        observations = observations + (frame(3_000_000_000),)

    with pytest.raises(ValueError, match="stamp set"):
        scorer(truth, observations, maximum_distance_m=1.0)


def test_detection_frame_mismatch_is_labeled_partial_with_exact_coverage():
    stamp_a = 1_000_000_000
    stamp_b = 2_000_000_000
    truth = (
        frame(stamp_a, sample("actor-a", 0.0, stamp_ns=stamp_a)),
        frame(stamp_b, sample("actor-b", 1.0, stamp_ns=stamp_b)),
    )
    detections = (
        frame(stamp_a, sample("det-a", 0.0, stamp_ns=stamp_a)),
        frame(
            stamp_b,
            sample(
                "det-b", 1.0, stamp_ns=stamp_b, frame_id="map"
            ),
            frame_id="map",
        ),
    )

    result = score_detections(truth, detections, maximum_distance_m=1.0)

    assert result.recall == MetricValue(
        value=1.0,
        unavailable_reason=None,
        support=1,
        expected=2,
        excluded=1,
        partial_reason="frame mismatch at stamp 2000000000: odom != map",
    )
    assert result.false_positives_per_frame.support == 1
    assert result.false_positives_per_frame.expected == 2
    assert result.excluded_frames[0].stamp_ns == stamp_b


@pytest.mark.parametrize("scorer", (score_detections, score_tracking))
def test_aggregate_scorers_enforce_explicit_expected_stamp_set(scorer):
    stamp_a = 1_000_000_000
    stamp_b = 2_000_000_000
    truth = (frame(stamp_a),)
    observations = (frame(stamp_a),)

    with pytest.raises(ValueError, match="expected stamp set"):
        scorer(
            truth,
            observations,
            maximum_distance_m=1.0,
            expected_stamps_ns=(stamp_a, stamp_b),
            expected_frame_id="odom",
        )


def test_truth_empty_negative_control_scores_false_positives():
    stamp = 1_000_000_000
    truth = (frame(stamp),)
    detections = (frame(stamp, sample("false-positive", 20.0)),)

    result = score_detections(
        truth, detections, maximum_distance_m=1.0
    )

    assert result.recall == MetricValue(
        value=None,
        unavailable_reason="recall denominator is zero",
        support=0,
        expected=0,
        excluded=0,
        partial_reason=None,
    )
    assert result.false_positives.value == 1
    assert result.false_positives.support == 1
    assert result.false_positives.expected == 1


def test_tracking_represents_never_initialized_actor_and_completeness():
    stamp_a = 1_000_000_000
    stamp_b = 2_000_000_000
    truth = (
        frame(stamp_a, sample("actor", 0.0, stamp_ns=stamp_a)),
        frame(stamp_b, sample("actor", 1.0, stamp_ns=stamp_b)),
    )
    tracks = (frame(stamp_a), frame(stamp_b))

    result = score_tracking(truth, tracks, maximum_distance_m=1.0)

    assert result.initialization_delay_s["actor"].value is None
    assert result.initialization_delay_s["actor"].unavailable_reason == (
        "actor was never initialized"
    )
    assert result.missed_actor_frames.value == 2
    assert result.drop_episodes.value == 0
    assert result.completeness.value == pytest.approx(0.0)
    assert result.completeness.support == 2
    assert result.completeness.expected == 2


def test_tracking_frame_exclusion_propagates_actor_opportunity_support():
    stamp_a = 1_000_000_000
    stamp_b = 2_000_000_000
    truth = (
        frame(stamp_a, sample("actor", 0.0, vx=2.0, stamp_ns=stamp_a)),
        frame(stamp_b, sample("actor", 1.0, vx=2.0, stamp_ns=stamp_b)),
    )
    tracks = (
        frame(
            stamp_a,
            sample(
                "track", 0.0, vx=2.0, stamp_ns=stamp_a, frame_id="map"
            ),
            frame_id="map",
        ),
        frame(
            stamp_b,
            sample("track", 1.5, vx=2.5, stamp_ns=stamp_b),
        ),
    )

    result = score_tracking(truth, tracks, maximum_distance_m=1.0)

    assert result.initialization_delay_s["actor"] == MetricValue(
        value=None,
        unavailable_reason=(
            "initialization delay is unavailable because 1 of 2 "
            "pre-initialization actor frames were excluded"
        ),
        support=1,
        expected=2,
        excluded=1,
    )
    assert result.id_switches == MetricValue(
        value=None,
        unavailable_reason=(
            "ID switches are unavailable for all 1 actor transitions; "
            "frame mismatch at stamp 1000000000: odom != map"
        ),
        support=0,
        expected=1,
        excluded=1,
    )
    assert result.position_rmse_m == MetricValue(
        value=0.5,
        unavailable_reason=None,
        support=1,
        expected=2,
        excluded=1,
        partial_reason=(
            "position unavailable for 1 of 2 actor frames; "
            "frame mismatch at stamp 1000000000: odom != map"
        ),
    )
    assert result.velocity_rmse_mps == MetricValue(
        value=0.5,
        unavailable_reason=None,
        support=1,
        expected=2,
        excluded=1,
        partial_reason=(
            "velocity unavailable for 1 of 2 actor frames; "
            "frame mismatch at stamp 1000000000: odom != map"
        ),
    )


def test_tracking_actor_seen_only_in_excluded_frame_is_not_blame_labeled():
    stamp = 1_000_000_000
    truth = (frame(stamp, sample("actor", 0.0, stamp_ns=stamp)),)
    tracks = (
        frame(
            stamp,
            sample("track", 0.0, stamp_ns=stamp, frame_id="map"),
            frame_id="map",
        ),
    )

    result = score_tracking(truth, tracks, maximum_distance_m=1.0)

    delay = result.initialization_delay_s["actor"]
    assert delay == MetricValue(
        value=None,
        unavailable_reason=(
            "initialization delay is unavailable because all 1 actor frames "
            "were excluded"
        ),
        support=0,
        expected=1,
        excluded=1,
    )
    assert "never initialized" not in delay.unavailable_reason
    assert result.position_rmse_m.support == 0
    assert result.position_rmse_m.expected == 1
    assert result.velocity_rmse_mps.support == 0
    assert result.velocity_rmse_mps.expected == 1


def test_tracking_adjacent_association_change_counts_one_supported_switch():
    stamps = (1_000_000_000, 2_000_000_000)
    truth = tuple(
        frame(stamp, sample("actor", index, stamp_ns=stamp))
        for index, stamp in enumerate(stamps)
    )
    tracks = (
        frame(stamps[0], sample("track-a", 0.0, stamp_ns=stamps[0])),
        frame(stamps[1], sample("track-b", 1.0, stamp_ns=stamps[1])),
    )

    result = score_tracking(truth, tracks, maximum_distance_m=1.0)

    assert result.id_switches == MetricValue(
        value=1,
        unavailable_reason=None,
        support=1,
        expected=1,
        excluded=0,
    )


@pytest.mark.parametrize("post_gap_track_id", ("track-a", "track-b"))
def test_tracking_excluded_middle_breaks_id_continuity(post_gap_track_id):
    stamps = (1_000_000_000, 2_000_000_000, 3_000_000_000)
    truth = tuple(
        frame(stamp, sample("actor", index, stamp_ns=stamp))
        for index, stamp in enumerate(stamps)
    )
    tracks = (
        frame(stamps[0], sample("track-a", 0.0, stamp_ns=stamps[0])),
        frame(
            stamps[1],
            sample(
                "track-gap",
                1.0,
                stamp_ns=stamps[1],
                frame_id="map",
            ),
            frame_id="map",
        ),
        frame(
            stamps[2],
            sample(post_gap_track_id, 2.0, stamp_ns=stamps[2]),
        ),
    )

    result = score_tracking(truth, tracks, maximum_distance_m=1.0)

    assert result.id_switches == MetricValue(
        value=None,
        unavailable_reason=(
            "ID switches are unavailable for all 2 actor transitions; "
            "frame mismatch at stamp 2000000000: odom != map"
        ),
        support=0,
        expected=2,
        excluded=2,
    )


def test_tracking_unmatched_middle_breaks_id_continuity():
    stamps = (1_000_000_000, 2_000_000_000, 3_000_000_000)
    truth = tuple(
        frame(stamp, sample("actor", index, stamp_ns=stamp))
        for index, stamp in enumerate(stamps)
    )
    tracks = (
        frame(stamps[0], sample("track-a", 0.0, stamp_ns=stamps[0])),
        frame(stamps[1]),
        frame(stamps[2], sample("track-b", 2.0, stamp_ns=stamps[2])),
    )

    result = score_tracking(truth, tracks, maximum_distance_m=1.0)

    assert result.id_switches == MetricValue(
        value=None,
        unavailable_reason="ID switches are unavailable for all 2 actor transitions",
        support=0,
        expected=2,
        excluded=2,
    )


def test_tracking_counts_contiguous_drop_episodes_separately_from_misses():
    stamps = tuple(index * 1_000_000_000 for index in range(1, 7))
    truth = tuple(
        frame(stamp, sample("actor", index, stamp_ns=stamp))
        for index, stamp in enumerate(stamps)
    )
    tracks = tuple(
        frame(
            stamp,
            *(
                (sample("track", index, stamp_ns=stamp),)
                if index in {0, 3, 5}
                else ()
            ),
        )
        for index, stamp in enumerate(stamps)
    )

    result = score_tracking(truth, tracks, maximum_distance_m=1.0)

    assert result.missed_actor_frames.value == 3
    assert result.drop_episodes.value == 2
    assert result.completeness.value == pytest.approx(0.5)
    assert result.id_switches == MetricValue(
        value=None,
        unavailable_reason=(
            "ID switches are unavailable for all 5 actor transitions"
        ),
        support=0,
        expected=5,
        excluded=5,
    )


def test_velocity_rmse_labels_mixed_field_coverage_as_partial():
    stamp_a = 1_000_000_000
    stamp_b = 2_000_000_000
    truth = (
        frame(stamp_a, sample("a", 0.0, vx=2.0, stamp_ns=stamp_a)),
        frame(
            stamp_b,
            sample("b", 1.0, vx=None, vy=None, stamp_ns=stamp_b),
        ),
    )
    tracks = (
        frame(stamp_a, sample("ta", 0.0, vx=2.5, stamp_ns=stamp_a)),
        frame(
            stamp_b,
            sample("tb", 1.0, vx=None, vy=None, stamp_ns=stamp_b),
        ),
    )

    result = score_tracking(truth, tracks, maximum_distance_m=1.0)

    assert result.velocity_rmse_mps == MetricValue(
        value=0.5,
        unavailable_reason=None,
        support=1,
        expected=2,
        excluded=1,
        partial_reason="velocity unavailable for 1 of 2 actor frames",
    )


def test_prediction_missing_source_association_is_explicitly_unavailable():
    prediction = PredictionPoint(
        1_000_000_000, "odom", "track", 1.0, 1.0, 0.0
    )

    result = score_predictions(
        (prediction,),
        (),
        source_associations={},
        horizons_s=(1.0,),
    )

    assert result[1.0].ade_m.support == 0
    assert result[1.0].ade_m.expected == 1
    assert result[1.0].ade_m.excluded == 1
    assert result[1.0].ade_m.unavailable_reason == (
        "source association is unavailable"
    )


def test_prediction_mixed_future_truth_coverage_is_labeled_partial():
    source_stamp = 1_000_000_000
    predictions = (
        PredictionPoint(source_stamp, "odom", "track-a", 1.0, 2.5, 0.0),
        PredictionPoint(source_stamp, "odom", "track-b", 1.0, 8.0, 0.0),
    )
    truth = (
        frame(
            2_000_000_000,
            sample("actor-a", 2.0, stamp_ns=2_000_000_000),
        ),
    )

    result = score_predictions(
        predictions,
        truth,
        source_associations={
            (source_stamp, "track-a"): "actor-a",
            (source_stamp, "track-b"): "actor-b",
        },
        horizons_s=(1.0,),
    )

    assert result[1.0].ade_m.value == pytest.approx(0.5)
    assert result[1.0].ade_m.support == 1
    assert result[1.0].ade_m.expected == 2
    assert result[1.0].ade_m.excluded == 1
    assert result[1.0].ade_m.partial_reason == (
        "future truth unavailable for 1 of 2 trajectories"
    )


def test_prediction_distinguishes_common_frame_truth_mismatch():
    source_stamp = 1_000_000_000
    target_stamp = 2_000_000_000
    prediction = PredictionPoint(
        source_stamp, "odom", "track", 1.0, 2.0, 0.0
    )
    truth = (
        frame(
            target_stamp,
            sample(
                "actor",
                2.0,
                stamp_ns=target_stamp,
                frame_id="map",
            ),
            frame_id="map",
        ),
    )

    result = score_predictions(
        (prediction,),
        truth,
        source_associations={(source_stamp, "track"): "actor"},
        horizons_s=(1.0,),
    )

    assert result[1.0].ade_m == MetricValue(
        value=None,
        unavailable_reason="common-frame/TF truth is unavailable",
        support=0,
        expected=1,
        excluded=1,
    )
    assert result[1.0].fde_m.unavailable_reason == (
        "common-frame/TF truth is unavailable"
    )


def test_prediction_incomplete_horizon_is_unavailable_only_at_that_horizon():
    source_stamp = 1_000_000_000
    prediction = PredictionPoint(
        source_stamp, "odom", "track", 1.0, 2.0, 0.0
    )
    truth = (
        frame(
            2_000_000_000,
            sample("actor", 2.0, stamp_ns=2_000_000_000),
        ),
    )

    result = score_predictions(
        (prediction,),
        truth,
        source_associations={(source_stamp, "track"): "actor"},
        horizons_s=(1.0, 2.0),
    )

    assert result[1.0].ade_m.value == pytest.approx(0.0)
    assert result[2.0].ade_m.value is None
    assert result[2.0].ade_m.unavailable_reason == (
        "prediction horizon is incomplete"
    )
    assert result[2.0].ade_m.support == 0
    assert result[2.0].ade_m.expected == 1


def test_prediction_rejects_duplicate_keys_and_undeclared_horizons():
    source_stamp = 1_000_000_000
    duplicate = PredictionPoint(
        source_stamp, "odom", "track", 1.0, 2.0, 0.0
    )
    with pytest.raises(ValueError, match="duplicate prediction key"):
        score_predictions(
            (duplicate, duplicate),
            (),
            source_associations={(source_stamp, "track"): "actor"},
            horizons_s=(1.0,),
        )

    undeclared = PredictionPoint(
        source_stamp, "odom", "track", 2.0, 4.0, 0.0
    )
    with pytest.raises(ValueError, match="undeclared horizon"):
        score_predictions(
            (undeclared,),
            (),
            source_associations={(source_stamp, "track"): "actor"},
            horizons_s=(1.0,),
        )


@pytest.mark.parametrize(
    "operation, message",
    (
        (
            lambda: summarize_rate((), maximum_gap_s=0.1),
            "at least two",
        ),
        (
            lambda: summarize_rate((2, 1), maximum_gap_s=0.1),
            "strictly increasing",
        ),
        (
            lambda: sample("bad", float("nan")),
            "finite",
        ),
        (
            lambda: frame(
                1,
                sample("duplicate", 0.0, stamp_ns=1),
                sample("duplicate", 1.0, stamp_ns=1),
            ),
            "duplicate",
        ),
        (
            lambda: associate_frame(
                frame(1, sample("truth", 0.0, stamp_ns=1)),
                frame(2, sample("track", 0.0, stamp_ns=2)),
                maximum_distance_m=1.0,
            ),
            "same stamp",
        ),
    ),
)
def test_malformed_input_is_rejected(operation, message):
    with pytest.raises(ValueError, match=message):
        operation()


def test_versioned_json_and_markdown_reports_preserve_unavailable_metrics(tmp_path):
    metadata = RunMetadata(
        git_commit="0" * 40,
        profile_hashes={"classical": "1" * 64},
        bag_metadata={"path": "/data/run/bag", "message_count": 42},
        actor_preset="roundabout_loop",
    )
    output = write_reports(
        tmp_path,
        metadata=metadata,
        metrics={
            "velocity_rmse_mps": MetricValue(
                value=None,
                unavailable_reason="velocity field absent",
            ),
            "recall": MetricValue(value=0.75, unavailable_reason=None),
        },
    )

    document = json.loads(output.json_path.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    assert document["metadata"]["git_commit"] == "0" * 40
    assert document["metadata"]["profile_hashes"] == {"classical": "1" * 64}
    assert document["metadata"]["bag_metadata"]["message_count"] == 42
    assert document["metadata"]["actor_preset"] == "roundabout_loop"
    assert document["metrics"]["velocity_rmse_mps"] == {
        "excluded": 0,
        "expected": 0,
        "partial_reason": None,
        "support": 0,
        "unavailable_reason": "velocity field absent",
        "value": None,
    }
    markdown = output.markdown_path.read_text(encoding="utf-8")
    assert "velocity field absent" in markdown
    assert "0.75" in markdown
