"""Unit tests for the external mm-2025 YOLOv7 adapter."""

import pytest
import torch

from ad_camera_perception.inference.yolov7_backend import (
    _rescale_boxes,
    _resolve_yolov7_modules,
    _stride_aligned_shape,
)


def test_rescale_boxes_restores_camera_4_coordinates():
    """Inference-space bboxes return to the 1280x720 Camera-4 frame."""
    boxes = torch.tensor([[10.0, 20.0, 700.0, 400.0]])

    _rescale_boxes(boxes, (360, 640), (720, 1280))

    assert boxes.tolist() == [[20.0, 40.0, 1280.0, 720.0]]


def test_external_repository_path_is_validated_before_import(tmp_path):
    """A missing YOLOv7 checkout produces an actionable startup error."""
    with pytest.raises(FileNotFoundError, match="models/experimental.py"):
        _resolve_yolov7_modules(str(tmp_path))


def test_camera_4_shape_is_aligned_to_yolov7_stride():
    """A 16:9 frame avoids tensor concat failures in the YOLOv7 neck."""
    assert _stride_aligned_shape(720, 1280, 640, 32) == (352, 640)
