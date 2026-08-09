"""Model backends and detection conversion helpers."""

from .detection import Detection
from .yolo_backend import YoloBackend

__all__ = ["Detection", "YoloBackend"]
