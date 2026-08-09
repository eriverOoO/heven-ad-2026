"""Crop and ROS message conversion helpers for detections."""

from dataclasses import replace
from math import ceil, floor
from typing import Iterable, Sequence, Tuple

from std_msgs.msg import Header
from vision_msgs.msg import Detection2D, Detection2DArray
from vision_msgs.msg import ObjectHypothesisWithPose

from ad_camera_perception.inference.detection import Detection
from ad_camera_perception.utils.parameters import validate_normalized_crop


PixelCrop = Tuple[int, int, int, int]


def normalized_crop_to_pixels(
    image_shape: Sequence[int], normalized_crop: Iterable[float]
) -> PixelCrop:
    """Convert a normalized crop to clipped integer pixel boundaries."""
    if len(image_shape) < 2:
        raise ValueError("image_shape must contain height and width")
    image_height, image_width = int(image_shape[0]), int(image_shape[1])
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image dimensions must be positive")

    x_min, y_min, x_max, y_max = validate_normalized_crop(normalized_crop)
    left = max(0, min(image_width - 1, floor(x_min * image_width)))
    top = max(0, min(image_height - 1, floor(y_min * image_height)))
    right = max(left + 1, min(image_width, ceil(x_max * image_width)))
    bottom = max(top + 1, min(image_height, ceil(y_max * image_height)))
    return left, top, right, bottom


def restore_detection_to_full_image(
    detection: Detection,
    crop: PixelCrop,
    image_shape: Sequence[int],
) -> Detection:
    """Offset a crop-relative detection and clip it to the full image."""
    image_height, image_width = int(image_shape[0]), int(image_shape[1])
    left, top, _, _ = crop
    return replace(
        detection,
        x1=max(0.0, min(float(image_width), detection.x1 + left)),
        y1=max(0.0, min(float(image_height), detection.y1 + top)),
        x2=max(0.0, min(float(image_width), detection.x2 + left)),
        y2=max(0.0, min(float(image_height), detection.y2 + top)),
    )


def filter_excluded_class_ids(
    detections: Iterable[Detection], excluded_class_ids: Iterable[int]
) -> list:
    """Remove detections whose numeric class ID is configured as excluded."""
    excluded = {int(class_id) for class_id in excluded_class_ids}
    return [
        detection
        for detection in detections
        if detection.class_id not in excluded
    ]


def detections_to_message(
    header: Header, detections: Iterable[Detection]
) -> Detection2DArray:
    """Build a Detection2DArray while preserving the source header."""
    message = Detection2DArray()
    message.header = header

    for detection_index, detection in enumerate(detections):
        item = Detection2D()
        item.header = header
        item.id = str(detection_index)
        item.bbox.center.position.x = (detection.x1 + detection.x2) / 2.0
        item.bbox.center.position.y = (detection.y1 + detection.y2) / 2.0
        item.bbox.center.theta = 0.0
        item.bbox.size_x = detection.width
        item.bbox.size_y = detection.height

        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = detection.class_name
        hypothesis.hypothesis.score = detection.confidence
        item.results.append(hypothesis)
        message.detections.append(item)

    return message
