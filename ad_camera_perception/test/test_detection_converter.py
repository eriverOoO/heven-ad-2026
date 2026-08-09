"""Tests for crop, class filtering, and ROS detection conversion."""

from builtin_interfaces.msg import Time
from std_msgs.msg import Header

from ad_camera_perception.inference.detection import Detection
from ad_camera_perception.inference.detection_converter import (
    detections_to_message,
    filter_excluded_class_ids,
    normalized_crop_to_pixels,
    restore_detection_to_full_image,
)


def _detection(class_id=0, class_name="person"):
    """Build one representative crop-relative detection."""
    return Detection(
        x1=10.0,
        y1=20.0,
        x2=110.0,
        y2=220.0,
        confidence=0.8,
        class_id=class_id,
        class_name=class_name,
    )


def test_normalized_crop_uses_full_image_dimensions():
    """Normalized crop bounds are converted for a 1280x720 MORAI frame."""
    assert normalized_crop_to_pixels(
        (720, 1280, 3), [0.0, 0.2, 1.0, 1.0]
    ) == (0, 144, 1280, 720)


def test_crop_detection_is_restored_to_full_image_coordinates():
    """Crop offsets are added before detections leave the detector."""
    restored = restore_detection_to_full_image(
        _detection(), (100, 144, 1180, 720), (720, 1280, 3)
    )
    assert (restored.x1, restored.y1, restored.x2, restored.y2) == (
        110.0,
        164.0,
        210.0,
        364.0,
    )


def test_initial_excluded_coco_classes_are_removed():
    """Only configured COCO class IDs are removed."""
    detections = [
        _detection(0, "person"),
        _detection(9, "traffic light"),
        _detection(11, "stop sign"),
        _detection(24, "backpack"),
    ]
    filtered = filter_excluded_class_ids(detections, [9, 10, 11, 12])
    assert [item.class_id for item in filtered] == [0, 24]


def test_detection_message_preserves_header_and_bbox():
    """Detection2DArray and each item retain the source image header."""
    header = Header(
        stamp=Time(sec=123, nanosec=456),
        frame_id="camera_front_optical_frame",
    )
    message = detections_to_message(header, [_detection()])

    assert message.header == header
    assert message.detections[0].header == header
    assert message.detections[0].bbox.center.position.x == 60.0
    assert message.detections[0].bbox.center.position.y == 120.0
    assert message.detections[0].bbox.size_x == 100.0
    assert message.detections[0].bbox.size_y == 200.0
    hypothesis = message.detections[0].results[0].hypothesis
    assert hypothesis.class_id == "person"
    assert hypothesis.score == 0.8
