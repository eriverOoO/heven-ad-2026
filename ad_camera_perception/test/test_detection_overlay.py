"""Tests for bbox debug drawing."""

import numpy as np
from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose

from ad_camera_perception.visualization.detection_overlay import draw_detection_overlay


def test_overlay_draws_bbox_and_crop_without_mutating_source():
    """Overlay changes a copy while retaining the source image."""
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    detection = Detection2D()
    detection.bbox.center.position.x = 100.0
    detection.bbox.center.position.y = 50.0
    detection.bbox.size_x = 40.0
    detection.bbox.size_y = 30.0
    hypothesis = ObjectHypothesisWithPose()
    hypothesis.hypothesis.class_id = "person"
    hypothesis.hypothesis.score = 0.9
    detection.results.append(hypothesis)

    overlay = draw_detection_overlay(
        image, [detection], [0.0, 0.2, 1.0, 1.0]
    )

    assert np.count_nonzero(image) == 0
    assert np.count_nonzero(overlay) > 0
