"""Ultralytics YOLO detection backend."""

from pathlib import Path
from typing import Any, List, Mapping, Optional

import numpy as np

from ad_camera_perception.inference.detection import Detection


def _class_name(names: Any, class_id: int) -> str:
    """Resolve a class ID from Ultralytics dict/list name containers."""
    if isinstance(names, Mapping):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def detections_from_result(result: Any) -> List[Detection]:
    """Convert one Ultralytics Results object to immutable detections."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()
    class_ids = boxes.cls.detach().cpu().numpy().astype(int)
    names = result.names

    return [
        Detection(
            x1=float(coordinates[0]),
            y1=float(coordinates[1]),
            x2=float(coordinates[2]),
            y2=float(coordinates[3]),
            confidence=float(confidence),
            class_id=int(class_id),
            class_name=_class_name(names, int(class_id)),
        )
        for coordinates, confidence, class_id in zip(
            xyxy, confidences, class_ids
        )
    ]


class YoloBackend:
    """Run a YOLO Detect model with stable, testable output values."""

    def __init__(
        self,
        model_path: str,
        device: str,
        image_size: int,
        confidence_threshold: float,
        model: Optional[Any] = None,
    ) -> None:
        """Load a model unless an injected test model is supplied."""
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be within [0.0, 1.0]")

        self._device = device
        self._image_size = image_size
        self._confidence_threshold = confidence_threshold

        if model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "Ultralytics is missing. Install ad_camera_perception/requirements.txt"
                ) from exc
            self._model = YOLO(model_path)
        else:
            self._model = model

    def infer(self, image: np.ndarray) -> List[Detection]:
        """Run inference on one BGR image and return its detections."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("YOLO input must be an HxWx3 image")

        results = self._model.predict(
            source=image,
            imgsz=self._image_size,
            conf=self._confidence_threshold,
            device=None if self._device == "auto" else self._device,
            verbose=False,
        )
        if not results:
            return []
        return detections_from_result(results[0])


def describe_model_source(model_path: str) -> str:
    """Describe whether a model is local or an Ultralytics model name."""
    path = Path(model_path).expanduser()
    if path.is_file():
        return str(path.resolve())
    if path.parent == Path("."):
        return f"{model_path} (Ultralytics download/cache)"
    return f"{model_path} (missing local path)"
