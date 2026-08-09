"""Tests for the mockable Ultralytics adapter."""

import numpy as np

from ad_camera_perception.inference.yolo_backend import YoloBackend


class _FakeTensor:
    """Minimal tensor adapter used by the result converter."""

    def __init__(self, values):
        self._values = np.asarray(values)

    def detach(self):
        """Mirror the PyTorch detach API."""
        return self

    def cpu(self):
        """Mirror the PyTorch cpu API."""
        return self

    def numpy(self):
        """Return the stored NumPy values."""
        return self._values


class _FakeBoxes:
    """Ultralytics-like boxes container."""

    xyxy = _FakeTensor([[1.0, 2.0, 11.0, 22.0]])
    conf = _FakeTensor([0.75])
    cls = _FakeTensor([0])

    def __len__(self):
        """Return the number of fake detections."""
        return 1


class _FakeResult:
    """Ultralytics-like inference result."""

    boxes = _FakeBoxes()
    names = {0: "person"}


class _FakeModel:
    """Record predict arguments and return one fake result."""

    def __init__(self):
        self.arguments = None

    def predict(self, **kwargs):
        """Record keyword arguments from the backend."""
        self.arguments = kwargs
        return [_FakeResult()]


def test_backend_is_mockable_and_converts_results():
    """An injected model receives configured inference arguments."""
    model = _FakeModel()
    backend = YoloBackend(
        model_path="unused.pt",
        device="cuda:0",
        image_size=640,
        confidence_threshold=0.2,
        model=model,
    )
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    detections = backend.infer(image)

    assert len(detections) == 1
    assert detections[0].class_name == "person"
    assert detections[0].confidence == 0.75
    assert model.arguments["source"] is image
    assert model.arguments["imgsz"] == 640
    assert model.arguments["conf"] == 0.2
    assert model.arguments["device"] == "cuda:0"
    assert model.arguments["verbose"] is False
