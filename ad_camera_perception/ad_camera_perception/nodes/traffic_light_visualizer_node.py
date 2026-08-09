"""Standalone OpenCV imshow visualizer for the traffic-light pipeline."""

from collections import OrderedDict
import os
from typing import Tuple

from ad_interfaces.msg import TrafficLightStatus
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from vision_msgs.msg import Detection2DArray

from ad_camera_perception.utils.image_messages import (
    ImageMessage,
    image_message_to_bgr,
    image_message_type,
)
from ad_camera_perception.utils.parameters import validate_normalized_crop
from ad_camera_perception.visualization.traffic_light_overlay import (
    draw_traffic_light_overlay,
)


StampKey = Tuple[int, int]


class TrafficLightVisualizerNode(Node):
    """Show Camera-4, detector bboxes, selection, and aspects via cv2.imshow."""

    def __init__(self) -> None:
        super().__init__("traffic_light_visualizer")
        self.declare_parameter(
            "image_topic", "/ad/sensors/camera/traffic_light/compressed"
        )
        self.declare_parameter(
            "detections_topic", "/vision/traffic_light/detections"
        )
        self.declare_parameter("status_topic", "/vision/traffic_light/status")
        self.declare_parameter("image_transport", "compressed")
        self.declare_parameter("image_buffer_size", 20)
        self.declare_parameter(
            "target_roi_normalized", [0.25, 0.10, 0.85, 0.95]
        )
        self.declare_parameter("show_window", True)
        self.declare_parameter("window_name", "mm2025_traffic_light")

        image_topic = str(self.get_parameter("image_topic").value)
        detections_topic = str(self.get_parameter("detections_topic").value)
        status_topic = str(self.get_parameter("status_topic").value)
        self._image_transport = str(self.get_parameter("image_transport").value)
        self._buffer_size = int(self.get_parameter("image_buffer_size").value)
        self._target_roi = validate_normalized_crop(
            self.get_parameter("target_roi_normalized").value
        )
        self._show_window = bool(self.get_parameter("show_window").value)
        self._window_name = str(self.get_parameter("window_name").value)
        self._window_created = False
        if self._buffer_size <= 0:
            raise ValueError("image_buffer_size must be positive")
        if not self._window_name:
            raise ValueError("window_name must not be empty")
        if self._show_window and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        ):
            self.get_logger().error(
                "OpenCV window requested without DISPLAY/WAYLAND_DISPLAY; "
                "visualizer will stay inactive"
            )
            self._show_window = False

        self._images = OrderedDict()
        self._detections = OrderedDict()
        self._statuses = OrderedDict()
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._image_subscription = self.create_subscription(
            image_message_type(self._image_transport),
            image_topic,
            self._on_image,
            qos_profile_sensor_data,
        )
        self._detection_subscription = self.create_subscription(
            Detection2DArray, detections_topic, self._on_detections, qos
        )
        self._status_subscription = self.create_subscription(
            TrafficLightStatus, status_topic, self._on_status, qos
        )
        self.get_logger().info(
            f"OpenCV visualization: {image_topic}, {detections_topic}, {status_topic}"
        )

    @staticmethod
    def _stamp_key(message) -> StampKey:
        return message.header.stamp.sec, message.header.stamp.nanosec

    def _retain(self, buffer, key, value) -> None:
        buffer[key] = value
        buffer.move_to_end(key)
        while len(buffer) > self._buffer_size:
            buffer.popitem(last=False)

    def _on_image(self, message: ImageMessage) -> None:
        try:
            image = image_message_to_bgr(message, self._image_transport)
        except Exception as exc:
            self.get_logger().error(
                f"Traffic visualization image decode failed: {exc}",
                throttle_duration_sec=5.0,
            )
            return
        self._retain(self._images, self._stamp_key(message), image)
        self._try_render(self._stamp_key(message))

    def _on_detections(self, message: Detection2DArray) -> None:
        key = self._stamp_key(message)
        self._retain(self._detections, key, message)
        self._try_render(key)

    def _on_status(self, message: TrafficLightStatus) -> None:
        if not self._show_window:
            return
        key = self._stamp_key(message)
        self._retain(self._statuses, key, message)
        self._try_render(key)

    def _try_render(self, key: StampKey) -> None:
        if not self._show_window:
            return
        if not all(
            key in buffer
            for buffer in (self._images, self._detections, self._statuses)
        ):
            return
        image = self._images.pop(key, None)
        detections = self._detections.pop(key, None)
        status = self._statuses.pop(key, None)
        overlay = draw_traffic_light_overlay(
            image, detections, status, self._target_roi
        )
        self._show(overlay)

    def _show(self, overlay) -> None:
        try:
            if not self._window_created:
                cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
                self._window_created = True
            cv2.imshow(self._window_name, overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                cv2.destroyWindow(self._window_name)
                self._window_created = False
                self._show_window = False
        except cv2.error as exc:
            self.get_logger().error(f"OpenCV imshow failed: {exc}")
            self._show_window = False

    def destroy_node(self):
        if self._window_created:
            try:
                cv2.destroyWindow(self._window_name)
                cv2.waitKey(1)
            except cv2.error:
                pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = TrafficLightVisualizerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
