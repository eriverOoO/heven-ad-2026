"""YOLOv7 backend compatible with the mm-2025 traffic-light checkpoint."""

from pathlib import Path
import sys
from typing import Any, List, Optional

import cv2
import numpy as np

from ad_camera_perception.inference.detection import Detection


def _resolve_yolov7_modules(repository_path: str):
    root = Path(repository_path).expanduser().resolve()
    if not (root / "models" / "experimental.py").is_file():
        raise FileNotFoundError(
            f"YOLOv7 repository is missing models/experimental.py: {root}"
        )
    if not (root / "utils" / "general.py").is_file():
        raise FileNotFoundError(
            f"YOLOv7 repository is missing utils/general.py: {root}"
        )
    # YOLOv7 checkpoints are pickled with imports such as ``models.yolo``.
    # Keep the external checkout importable after the initial imports so that
    # torch.load and later model execution can resolve those modules.
    root_value = str(root)
    if root_value not in sys.path:
        sys.path.insert(0, root_value)
    from models.experimental import attempt_load
    from utils.general import non_max_suppression
    return attempt_load, non_max_suppression


def _rescale_boxes(boxes, inference_shape, original_shape) -> None:
    inference_height, inference_width = inference_shape
    original_height, original_width = original_shape
    scale_x = float(original_width) / float(inference_width)
    scale_y = float(original_height) / float(inference_height)
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    boxes[:, 0].clamp_(0, original_width)
    boxes[:, 1].clamp_(0, original_height)
    boxes[:, 2].clamp_(0, original_width)
    boxes[:, 3].clamp_(0, original_height)


def _stride_aligned_shape(
    original_height: int,
    original_width: int,
    inference_width: int,
    stride: int,
) -> tuple[int, int]:
    """Preserve aspect ratio while flooring height to a model stride."""
    proportional_height = inference_width * original_height / original_width
    inference_height = max(stride, int(proportional_height) // stride * stride)
    return inference_height, inference_width


class YoloV7Backend:
    """Load and run the original mm-2025 YOLOv7 detector."""

    def __init__(
        self,
        model_path: str,
        repository_path: str,
        device: str,
        image_size: int,
        confidence_threshold: float,
        iou_threshold: float,
        *,
        model: Optional[Any] = None,
        torch_module: Optional[Any] = None,
        non_max_suppression=None,
    ) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be within [0, 1]")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be within [0, 1]")

        if torch_module is None:
            try:
                import torch as torch_module
            except ImportError as exc:
                raise RuntimeError("PyTorch is required for the mm-2025 model") from exc
        self._torch = torch_module
        self._device = str(device)
        self._image_size = int(image_size)
        self._confidence_threshold = float(confidence_threshold)
        self._iou_threshold = float(iou_threshold)

        if model is None:
            weights = Path(model_path).expanduser().resolve()
            if not weights.is_file():
                raise FileNotFoundError(f"mm-2025 model weight not found: {weights}")
            attempt_load, resolved_nms = _resolve_yolov7_modules(repository_path)
            self._model = attempt_load(str(weights), map_location=self._device)
            self._non_max_suppression = resolved_nms
        else:
            if non_max_suppression is None:
                raise ValueError("injected model requires non_max_suppression")
            self._model = model
            self._non_max_suppression = non_max_suppression
        self._names = self._model.names
        model_stride = getattr(self._model, "stride", 32)
        try:
            self._stride = max(1, int(model_stride.max()))
        except AttributeError:
            self._stride = max(1, int(model_stride))

    def _class_name(self, class_id: int) -> str:
        if isinstance(self._names, dict):
            return str(self._names.get(class_id, class_id))
        if isinstance(self._names, (list, tuple)) and 0 <= class_id < len(self._names):
            return str(self._names[class_id])
        return str(class_id)

    def infer(self, image: np.ndarray) -> List[Detection]:
        """Run inference with the resizing and scaling used by mm-2025."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("YOLOv7 input must be an HxWx3 BGR image")
        original_height, original_width = image.shape[:2]
        inference_height, inference_width = _stride_aligned_shape(
            original_height,
            original_width,
            self._image_size,
            self._stride,
        )
        resized = cv2.resize(image, (inference_width, inference_height))
        rgb_chw = resized.transpose((2, 0, 1))[::-1]
        tensor = self._torch.from_numpy(np.ascontiguousarray(rgb_chw)).float()
        tensor = (tensor / 255.0).to(self._device).unsqueeze(0)

        with self._torch.no_grad():
            predictions = self._model(tensor)[0]
            batches = self._non_max_suppression(
                predictions,
                conf_thres=self._confidence_threshold,
                iou_thres=self._iou_threshold,
            )
        if not batches or batches[0] is None or len(batches[0]) == 0:
            return []
        detections = batches[0].clone()
        _rescale_boxes(
            detections[:, :4],
            (inference_height, inference_width),
            (original_height, original_width),
        )
        values = detections.detach().cpu().numpy()
        return [
            Detection(
                x1=float(item[0]),
                y1=float(item[1]),
                x2=float(item[2]),
                y2=float(item[3]),
                confidence=float(item[4]),
                class_id=int(item[5]),
                class_name=self._class_name(int(item[5])),
            )
            for item in values
        ]
