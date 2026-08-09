"""Tests for mm-2025 signal mapping, target selection, and voting."""

from ad_camera_perception.traffic_light.traffic import (
    MM2025_CLASS_ASPECTS,
    SignalVoteFilter,
    TargetSelector,
    TrafficDetection,
    aspects_for_class,
)


def _detection(
    class_name="1301",
    detection_id="0",
    confidence=0.8,
    center_x=640.0,
    center_y=200.0,
    width=80.0,
    height=120.0,
):
    return TrafficDetection(
        detection_id=detection_id,
        class_name=class_name,
        confidence=confidence,
        center_x=center_x,
        center_y=center_y,
        width=width,
        height=height,
    )


def test_all_mm2025_classes_map_to_independent_aspects():
    """The inherited model's twelve labels all have supported semantics."""
    assert set(MM2025_CLASS_ASPECTS) == {
        "1300", "1301", "1302", "1303", "1305", "1400",
        "1401", "1402", "1403", "1404", "1405", "1406",
    }
    assert aspects_for_class("1303").red
    assert aspects_for_class("1303").left_green
    assert aspects_for_class("1405").straight_green
    assert aspects_for_class("1405").left_green
    assert aspects_for_class("9999") is None


def test_target_selector_ignores_outside_and_unsupported_detections():
    """Only a supported bbox whose center is in the target ROI can win."""
    selector = TargetSelector(1280, 720, [0.25, 0.10, 0.85, 0.95])
    outside = _detection(detection_id="outside", center_x=100.0)
    unsupported = _detection(class_name="person", detection_id="unsupported")
    supported = _detection(detection_id="supported")

    assert selector.select([outside, unsupported, supported]) == supported


def test_target_selector_prefers_combined_confidence_area_and_center_score():
    """A strong central target wins over a weak edge target."""
    selector = TargetSelector(1280, 720, [0.25, 0.10, 0.85, 0.95])
    weak_edge = _detection(
        detection_id="edge", confidence=0.4, center_x=1080.0, center_y=450.0
    )
    strong_center = _detection(detection_id="center", confidence=0.9)

    assert selector.select([weak_edge, strong_center]) == strong_center


def test_vote_filter_exposes_candidate_before_three_of_five_confirmation():
    """The overlay can mark the current candidate before it becomes valid."""
    vote_filter = SignalVoteFilter(window_frames=5, minimum_vote_frames=3)
    first = vote_filter.update(_detection(detection_id="candidate"))

    assert not first.valid
    assert first.aspects.red
    assert first.detection_id == "candidate"
    assert first.source_class == "1301"


def test_vote_filter_confirms_composite_signal_after_three_votes():
    """Three matching observations confirm all aspects of a composite class."""
    vote_filter = SignalVoteFilter(window_frames=5, minimum_vote_frames=3)
    assert not vote_filter.update(_detection(class_name="1405")).valid
    assert not vote_filter.update(None).valid
    assert not vote_filter.update(_detection(class_name="1405")).valid
    result = vote_filter.update(_detection(class_name="1405"))

    assert result.valid
    assert result.aspects.straight_green
    assert result.aspects.left_green
    assert not result.aspects.red


def test_vote_filter_clear_removes_old_confirmation():
    """A stale input clears previously accumulated evidence."""
    vote_filter = SignalVoteFilter(window_frames=5, minimum_vote_frames=3)
    for _ in range(3):
        result = vote_filter.update(_detection())
    assert result.valid

    vote_filter.clear()
    assert not vote_filter.update(None).valid
