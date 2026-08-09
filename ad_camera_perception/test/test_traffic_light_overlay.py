"""Tests for the standalone OpenCV traffic-light visualization."""

from ad_interfaces.msg import TrafficLightStatus
import numpy as np
from std_msgs.msg import Header

from ad_camera_perception.inference.detection import Detection
from ad_camera_perception.inference.detection_converter import (
    detections_to_message,
)
from ad_camera_perception.visualization.traffic_light_overlay import (
    draw_traffic_light_overlay,
)


def test_overlay_draws_roi_boxes_selection_and_composite_state():
    """The imshow frame contains all information needed for live inspection."""
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    detections = detections_to_message(
        Header(),
        [
            Detection(
                x1=500.0,
                y1=100.0,
                x2=580.0,
                y2=220.0,
                confidence=0.91,
                class_id=9,
                class_name="1405",
            )
        ],
    )
    status = TrafficLightStatus()
    status.valid = True
    status.confidence = 0.91
    status.straight_green = True
    status.left_green = True
    status.source_class = "1405"
    status.detection_id = "0"

    overlay = draw_traffic_light_overlay(
        image, detections, status, [0.142857, 0.0, 0.857143, 1.0]
    )

    assert overlay.shape == image.shape
    assert np.any(overlay != image)
    assert tuple(overlay[100, 500]) == (0, 255, 0)
