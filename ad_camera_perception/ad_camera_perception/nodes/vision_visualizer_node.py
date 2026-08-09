"""ROS 2 image publisher for detection debugging overlays."""

from collections import OrderedDict
import os
from typing import Tuple

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

from ad_camera_perception.utils.image_messages import (
    ImageMessage,
    image_message_to_bgr,
    image_message_type,
)
from ad_camera_perception.utils.parameters import validate_normalized_crop
from ad_camera_perception.visualization.detection_overlay import draw_detection_overlay


StampKey = Tuple[int, int]


class VisionVisualizerNode(Node):
    """Match detections to buffered source frames and publish an overlay."""

    def __init__(self) -> None:
        """Declare parameters and create visualization subscriptions."""
        super().__init__("vision_visualizer_node")

        self.declare_parameter(
            "image_topic", "/ad/sensors/camera/front/compressed"
        )
        self.declare_parameter(
            "detections_topic", "/vision/dynamic_obstacle/detections"
        )
        self.declare_parameter(
            "output_image_topic",
            "/ad/viz/perception/camera/dynamic_obstacle",
        )
        self.declare_parameter("image_transport", "compressed")
        self.declare_parameter("image_buffer_size", 10)
        self.declare_parameter("crop_normalized", [0.0, 0.2, 1.0, 1.0])
        self.declare_parameter("show_window", False)
        self.declare_parameter("window_name", "dynamic_obstacle_debug")

        image_topic = str(self.get_parameter("image_topic").value)
        detections_topic = str(self.get_parameter("detections_topic").value)
        output_topic = str(self.get_parameter("output_image_topic").value)
        self._image_transport = str(
            self.get_parameter("image_transport").value
        )
        self._buffer_size = int(self.get_parameter("image_buffer_size").value)
        self._crop_normalized = validate_normalized_crop(
            self.get_parameter("crop_normalized").value
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
                "show_window requested without DISPLAY/WAYLAND_DISPLAY; "
                "continuing with debug topic only"
            )
            self._show_window = False

        self._images = OrderedDict()
        message_type = image_message_type(self._image_transport)
        self._image_subscription = self.create_subscription(
            message_type,
            image_topic,
            self._on_image,
            qos_profile_sensor_data,
        )
        detection_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._detection_subscription = self.create_subscription(
            Detection2DArray,
            detections_topic,
            self._on_detections,
            detection_qos,
        )
        self._publisher = self.create_publisher(Image, output_topic, detection_qos)
        self.get_logger().info(
            f"Visualizing {detections_topic} with {image_topic}; "
            f"publishing {output_topic}"
        )
        if self._show_window:
            self.get_logger().info(
                f"OpenCV window enabled: {self._window_name}"
            )

    @staticmethod
    def _stamp_key(message) -> StampKey:
        """Return an exact source timestamp key."""
        return message.header.stamp.sec, message.header.stamp.nanosec

    def _on_image(self, message: ImageMessage) -> None:
        """Decode and retain a bounded buffer of source frames."""
        try:
            image = image_message_to_bgr(message, self._image_transport)
        except Exception as exc:
            self.get_logger().error(
                f"Debug image decode failed: {exc}",
                throttle_duration_sec=5.0,
            )
            return

        key = self._stamp_key(message)
        self._images[key] = image
        self._images.move_to_end(key)
        while len(self._images) > self._buffer_size:
            self._images.popitem(last=False)

    def _on_detections(self, message: Detection2DArray) -> None:
        """Draw detections only on the frame with the matching timestamp."""
        key = self._stamp_key(message)
        image = self._images.pop(key, None)
        if image is None:
            self.get_logger().warning(
                "No buffered image matches the detection timestamp",
                throttle_duration_sec=5.0,
            )
            return

        overlay = draw_detection_overlay(
            image, message.detections, self._crop_normalized
        )
        output = self._bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
        output.header = message.header
        self._publisher.publish(output)
        self._show_overlay(overlay)

    def _show_overlay(self, overlay) -> None:
        """Display an overlay when the optional local OpenCV GUI is enabled."""
        if not self._show_window:
            return
        try:
            if not self._window_created:
                cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
                self._window_created = True
            cv2.imshow(self._window_name, overlay)
            cv2.waitKey(1)
        except cv2.error as exc:
            self.get_logger().error(
                f"OpenCV window failed; continuing with debug topic only: {exc}"
            )
            self._show_window = False

    def destroy_node(self):
        """Close the optional OpenCV window before destroying the ROS node."""
        if self._window_created:
            try:
                cv2.destroyWindow(self._window_name)
                cv2.waitKey(1)
            except cv2.error:
                pass
        return super().destroy_node()


def main(args=None) -> None:
    """Run the reusable vision visualizer node."""
    rclpy.init(args=args)
    node = None
    try:
        node = VisionVisualizerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
