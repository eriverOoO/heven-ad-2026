"""OpenCV overlay helpers for traffic-light detector and evaluator output."""

from typing import Sequence

from ad_interfaces.msg import TrafficLightStatus
import cv2
import numpy as np
from vision_msgs.msg import Detection2DArray

from ad_camera_perception.inference.detection_converter import normalized_crop_to_pixels


def _corners(detection) -> tuple[int, int, int, int]:
    center = detection.bbox.center.position
    half_width = detection.bbox.size_x / 2.0
    half_height = detection.bbox.size_y / 2.0
    return (
        int(round(center.x - half_width)),
        int(round(center.y - half_height)),
        int(round(center.x + half_width)),
        int(round(center.y + half_height)),
    )


def _aspects_text(status: TrafficLightStatus) -> str:
    active = []
    if status.red:
        active.append("RED")
    if status.yellow:
        active.append("YELLOW")
    if status.straight_green:
        active.append("STRAIGHT_GREEN")
    if status.left_green:
        active.append("LEFT_GREEN")
    return "+".join(active) if active else "UNKNOWN"


def draw_traffic_light_overlay(
    image: np.ndarray,
    detections: Detection2DArray,
    status: TrafficLightStatus,
    target_roi_normalized: Sequence[float],
) -> np.ndarray:
    """Draw all detections, selected target, and final aspect state."""
    output = image.copy()
    left, top, right, bottom = normalized_crop_to_pixels(
        output.shape, target_roi_normalized
    )
    dimmed = output.copy()
    dimmed[:top, :] = 0
    dimmed[bottom:, :] = 0
    dimmed[top:bottom, :left] = 0
    dimmed[top:bottom, right:] = 0
    output = cv2.addWeighted(output, 0.35, dimmed, 0.65, 0.0)
    cv2.rectangle(output, (left, top), (right - 1, bottom - 1), (255, 180, 0), 2)
    cv2.putText(
        output,
        "YOLO inference + target ROI",
        (left + 5, max(22, top + 22)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 180, 0),
        2,
        cv2.LINE_AA,
    )

    for detection in detections.detections:
        selected = bool(status.detection_id) and detection.id == status.detection_id
        color = (0, 255, 0) if selected else (180, 180, 180)
        thickness = 4 if selected else 2
        x1, y1, x2, y2 = _corners(detection)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        if detection.results:
            hypothesis = detection.results[0].hypothesis
            label = f"{hypothesis.class_id} {hypothesis.score:.2f}"
        else:
            label = "unknown 0.00"
        if selected:
            label = f"SELECTED {label}"
        cv2.putText(
            output,
            label,
            (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            color,
            2,
            cv2.LINE_AA,
        )

    state_color = (0, 255, 0) if status.valid else (0, 0, 255)
    lines = [
        f"valid={status.valid} confidence={status.confidence:.2f}",
        f"signal={_aspects_text(status)}",
        f"class={status.source_class or '-'} id={status.detection_id or '-'}",
    ]
    for index, text in enumerate(lines):
        cv2.putText(
            output,
            text,
            (20, 34 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            state_color,
            2,
            cv2.LINE_AA,
        )
    return output
