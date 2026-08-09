"""ROS node-level tests for the traffic-light detector."""

from contextlib import contextmanager

import cv2
import numpy as np
import rclpy
from sensor_msgs.msg import CompressedImage

from ad_camera_perception.inference.detection import Detection
from ad_camera_perception.nodes import traffic_light_detector_node as node_module


class CapturePublisher:
    """Collect published ROS messages without an executor."""

    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class FakeBackend:
    """Avoid loading a model during node contract tests."""

    def __init__(self, **_kwargs):
        pass

    def infer(self, _image):
        return []


@contextmanager
def running_node(monkeypatch):
    monkeypatch.setattr(node_module, "YoloV7Backend", FakeBackend)
    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            "yolov7_repository_path:=/unused/in/fake/backend",
        ]
    )
    node = node_module.TrafficLightDetectorNode()
    try:
        yield node
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _compressed_image() -> CompressedImage:
    encoded_ok, encoded = cv2.imencode(
        ".jpg", np.zeros((8, 8, 3), dtype=np.uint8)
    )
    assert encoded_ok
    message = CompressedImage()
    message.header.frame_id = "camera_4"
    message.format = "jpeg"
    message.data = encoded.tobytes()
    return message


def test_default_subscription_uses_camera_4(monkeypatch):
    """The inherited model consumes MORAI's dedicated traffic-light camera."""
    with running_node(monkeypatch) as node:
        assert node._subscription.topic_name == (
            "/ad/sensors/camera/traffic_light/compressed"
        )


def test_detector_preserves_source_header(monkeypatch):
    """Detection timestamps stay aligned with Camera-4 for visualization."""
    with running_node(monkeypatch) as node:
        capture = CapturePublisher()
        node._publisher = capture
        source = _compressed_image()

        node._on_image(source)
        node._on_timer()

        assert len(capture.messages) == 1
        assert capture.messages[0].header.frame_id == "camera_4"
        assert capture.messages[0].detections == []


def test_detector_infers_target_roi_and_restores_full_frame_coordinates(monkeypatch):
    """YOLO sees the center ROI while published boxes use full-frame pixels."""
    with running_node(monkeypatch) as node:
        inferred_shapes = []

        class RecordingBackend:
            def infer(self, image):
                inferred_shapes.append(image.shape)
                return [
                    Detection(
                        x1=1.0,
                        y1=1.0,
                        x2=5.0,
                        y2=3.0,
                        confidence=0.9,
                        class_id=1,
                        class_name="RED",
                    )
                ]

        node._backend = RecordingBackend()
        capture = CapturePublisher()
        node._publisher = capture
        node._on_image(_compressed_image())
        node._on_timer()

        assert inferred_shapes == [(8, 6, 3)]
        detection = capture.messages[0].detections[0]
        assert detection.bbox.center.position.x == 4.0
        assert detection.bbox.center.position.y == 2.0
