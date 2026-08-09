import pytest

from ad_morai_bridge.timestamp_policy import TimestampPolicy


def test_source_preferred_accepts_only_a_near_valid_nonregressing_source_stamp():
    policy = TimestampPolicy(
        mode="source_preferred",
        tolerance_sec=0.5,
        suppress_source_duplicates=False,
    )

    selected = policy.decide((100, 250_000_000), (100, 500_000_000))
    malformed = policy.decide((101, 1_000_000_000), (101, 100_000_000))
    distant = policy.decide((103, 0), (101, 200_000_000))

    assert selected.selected_stamp == (100, 250_000_000)
    assert selected.source_selected is True
    assert selected.arrival_fallback is False
    assert selected.source_rejected is False
    assert selected.publish_normalized is True

    assert malformed.selected_stamp == (101, 100_000_000)
    assert malformed.source_selected is False
    assert malformed.arrival_fallback is True
    assert malformed.source_rejected is True
    assert malformed.publish_normalized is True

    assert distant.selected_stamp == (101, 200_000_000)
    assert distant.source_rejected is True
    assert distant.arrival_fallback is True


def test_source_window_boundary_is_inclusive_but_one_nanosecond_beyond_rejects():
    accepted_policy = TimestampPolicy(
        mode="source_preferred", tolerance_sec=0.5
    )
    rejected_policy = TimestampPolicy(
        mode="source_preferred", tolerance_sec=0.5
    )

    accepted = accepted_policy.decide((10, 0), (10, 500_000_000))
    rejected = rejected_policy.decide((10, 0), (10, 500_000_001))

    assert accepted.source_valid is True
    assert accepted.source_selected is True
    assert rejected.source_valid is False
    assert rejected.source_rejected is True


def test_semantically_rejected_source_does_not_advance_the_source_watermark():
    policy = TimestampPolicy(mode="source_preferred", tolerance_sec=1.0)

    rejected = policy.decide(
        (20, 0), (20, 100), reject_source=True
    )
    accepted = policy.decide((20, 0), (20, 200))

    assert rejected.source_rejected is True
    assert rejected.source_selected is False
    assert accepted.source_selected is True
    assert accepted.duplicate is False


def test_no_output_rejection_does_not_advance_the_chosen_watermark():
    policy = TimestampPolicy(mode="source_preferred", tolerance_sec=1.0)

    rejected = policy.decide(
        None,
        (30, 900_000_000),
        reject_source=True,
        publish_requires_valid_source=True,
    )
    accepted = policy.decide(
        (30, 0),
        (30, 950_000_000),
        publish_requires_valid_source=True,
    )

    assert rejected.publish_normalized is False
    assert accepted.source_selected is True
    assert accepted.stamp_regression is False
    assert accepted.publish_normalized is True


def test_source_regression_falls_back_without_resetting_the_source_watermark():
    policy = TimestampPolicy(
        mode="source_preferred",
        tolerance_sec=1.0,
        suppress_source_duplicates=False,
    )

    assert policy.decide((200, 0), (200, 100_000_000)).source_selected

    regressed = policy.decide((199, 900_000_000), (200, 200_000_000))
    still_behind = policy.decide((199, 950_000_000), (200, 300_000_000))
    caught_up = policy.decide((200, 50_000_000), (200, 400_000_000))

    assert regressed.source_rejected is True
    assert regressed.selected_stamp == (200, 200_000_000)
    assert still_behind.source_rejected is True
    assert still_behind.selected_stamp == (200, 300_000_000)
    assert caught_up.source_selected is True
    assert caught_up.selected_stamp == (200, 50_000_000)

    policy.reset()
    reset_epoch = policy.decide((199, 500_000_000), (200, 0))
    assert reset_epoch.source_selected is True


def test_valid_exact_source_duplicate_is_audited_but_not_republished():
    policy = TimestampPolicy(
        mode="source_preferred",
        tolerance_sec=1.0,
        suppress_source_duplicates=True,
    )

    first = policy.decide((300, 10), (300, 20))
    duplicate = policy.decide((300, 10), (300, 30))

    assert first.publish_normalized is True
    assert duplicate.source_selected is True
    assert duplicate.duplicate is True
    assert duplicate.publish_normalized is False
    assert duplicate.stamp_regression is False


def test_collision_style_policy_counts_duplicate_without_suppressing_it():
    policy = TimestampPolicy(
        mode="source_preferred",
        tolerance_sec=1.0,
        suppress_source_duplicates=False,
    )

    assert policy.decide((350, 10), (350, 20)).publish_normalized
    duplicate = policy.decide((350, 10), (350, 30))

    assert duplicate.duplicate is True
    assert duplicate.publish_normalized is True


def test_chosen_stamp_regression_is_dropped_even_when_source_gate_accepts_it():
    policy = TimestampPolicy(
        mode="source_preferred",
        tolerance_sec=1.0,
        suppress_source_duplicates=False,
    )

    assert policy.decide(None, (400, 0)).publish_normalized
    regressed = policy.decide((399, 900_000_000), (400, 100_000_000))

    assert regressed.source_selected is True
    assert regressed.selected_stamp == (399, 900_000_000)
    assert regressed.stamp_regression is True
    assert regressed.publish_normalized is False


def test_arrival_mode_audits_source_stamp_repeats_without_suppressing_samples():
    policy = TimestampPolicy(
        mode="arrival",
        tolerance_sec=1.0,
        suppress_source_duplicates=False,
    )

    first = policy.decide((500, 0), (500, 100_000_000))
    duplicate = policy.decide((500, 0), (500, 200_000_000))

    assert first.selected_stamp == (500, 100_000_000)
    assert first.source_selected is False
    assert first.arrival_fallback is True
    assert duplicate.duplicate is True
    assert duplicate.publish_normalized is True


@pytest.mark.parametrize("mode", ["device_when_available", "", "source"])
def test_retired_or_unknown_timestamp_modes_are_rejected(mode):
    with pytest.raises(ValueError, match="source_preferred.*arrival"):
        TimestampPolicy(mode=mode, tolerance_sec=1.0)
