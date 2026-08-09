"""OpenCV drawing helpers for 2D detections."""

from typing import Iterable, Sequence

import cv2
import numpy as np
from vision_msgs.msg import Detection2D

from ad_camera_perception.inference.detection_converter import (
    normalized_crop_to_pixels,
)


def _box_corners(detection: Detection2D) -> tuple:
    """Return integer left, top, right, bottom bbox corners."""
    center = detection.bbox.center.position
    half_width = detection.bbox.size_x / 2.0
    half_height = detection.bbox.size_y / 2.0
    return (
        int(round(center.x - half_width)),
        int(round(center.y - half_height)),
        int(round(center.x + half_width)),
        int(round(center.y + half_height)),
    )


def _class_color(class_name: str) -> tuple:
    """Return a stable bright BGR color for a class name."""
    seed = sum((index + 1) * ord(char) for index, char in enumerate(class_name))
    return (
        64 + seed % 192,
        64 + (seed // 3) % 192,
        64 + (seed // 7) % 192,
    )


def draw_detection_overlay(
    image: np.ndarray,
    detections: Iterable[Detection2D],
    crop_normalized: Sequence[float],
) -> np.ndarray:
    """Draw the configured crop, boxes, labels, confidence, and foot points."""
    output = image.copy()
    left, top, right, bottom = normalized_crop_to_pixels(
        output.shape, crop_normalized
    )
    cv2.rectangle(output, (left, top), (right - 1, bottom - 1), (0, 255, 255), 2)
    cv2.putText(
        output,
        "inference crop",
        (left + 4, max(18, top + 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    detection_list = list(detections)
    for detection in detection_list:
        x1, y1, x2, y2 = _box_corners(detection)
        if detection.results:
            hypothesis = detection.results[0].hypothesis
            class_name = hypothesis.class_id
            confidence = hypothesis.score
        else:
            class_name = "unknown"
            confidence = 0.0
        color = _class_color(class_name)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        foot = (int(round((x1 + x2) / 2.0)), y2)
        cv2.circle(output, foot, 5, color, -1)
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(
            output,
            label,
            (x1, max(18, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        output,
        f"detections: {len(detection_list)}",
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output
