"""ROS 2 node for COCO-pretrained dynamic-obstacle candidate detection."""

from threading import Lock
from time import perf_counter
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from vision_msgs.msg import Detection2DArray

from ad_camera_perception.inference.detection import Detection
from ad_camera_perception.inference.detection_converter import (
    detections_to_message,
    filter_excluded_class_ids,
    normalized_crop_to_pixels,
    restore_detection_to_full_image,
)
from ad_camera_perception.inference.yolo_backend import describe_model_source, YoloBackend
from ad_camera_perception.utils.image_messages import (
    ImageMessage,
    image_message_to_bgr,
    image_message_type,
)
from ad_camera_perception.utils.parameters import validate_normalized_crop


class DynamicObstacleDetectorNode(Node):
    """Keep only the newest camera frame and run YOLO from a timer."""

    def __init__(self) -> None:
        """Declare parameters and construct the detector pipeline."""
        super().__init__("dynamic_obstacle_detector")

        self.declare_parameter(
            "image_topic", "/ad/sensors/camera/front/compressed"
        )
        self.declare_parameter(
            "detections_topic", "/vision/dynamic_obstacle/detections"
        )
        self.declare_parameter("image_transport", "compressed")
        self.declare_parameter("model_path", "yolo26s.pt")
        self.declare_parameter("device", "auto")
        self.declare_parameter("image_size", 640)
        self.declare_parameter("confidence_threshold", 0.20)
        self.declare_parameter("maximum_inference_rate", 20.0)
        self.declare_parameter("crop_enabled", True)
        self.declare_parameter("crop_normalized", [0.0, 0.2, 1.0, 1.0])
        self.declare_parameter("excluded_class_ids", [9, 10, 11, 12])

        image_topic = str(self.get_parameter("image_topic").value)
        detections_topic = str(self.get_parameter("detections_topic").value)
        self._image_transport = str(
            self.get_parameter("image_transport").value
        )
        model_path = str(self.get_parameter("model_path").value)
        device = str(self.get_parameter("device").value)
        image_size = int(self.get_parameter("image_size").value)
        confidence_threshold = float(
            self.get_parameter("confidence_threshold").value
        )
        maximum_rate = float(
            self.get_parameter("maximum_inference_rate").value
        )
        self._crop_enabled = bool(self.get_parameter("crop_enabled").value)
        self._crop_normalized = validate_normalized_crop(
            self.get_parameter("crop_normalized").value
        )
        self._excluded_class_ids = {
            int(class_id)
            for class_id in self.get_parameter("excluded_class_ids").value
        }
        if maximum_rate <= 0.0:
            raise ValueError("maximum_inference_rate must be positive")

        self.get_logger().info(
            f"Loading dynamic-obstacle model: {describe_model_source(model_path)}"
        )
        self._backend = YoloBackend(
            model_path=model_path,
            device=device,
            image_size=image_size,
            confidence_threshold=confidence_threshold,
        )
        self._message_lock = Lock()
        self._latest_message: Optional[ImageMessage] = None

        output_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._publisher = self.create_publisher(
            Detection2DArray, detections_topic, output_qos
        )
        message_type = image_message_type(self._image_transport)
        self._subscription = self.create_subscription(
            message_type,
            image_topic,
            self._on_image,
            qos_profile_sensor_data,
        )
        self._timer = self.create_timer(1.0 / maximum_rate, self._on_timer)
        self.get_logger().info(
            f"Listening on {image_topic} ({self._image_transport}); "
            f"publishing {detections_topic}"
        )

    def _on_image(self, message: ImageMessage) -> None:
        """Replace any pending camera frame with the newest one."""
        with self._message_lock:
            self._latest_message = message

    def _take_latest_message(self) -> Optional[ImageMessage]:
        """Atomically consume the newest pending camera frame."""
        with self._message_lock:
            message = self._latest_message
            self._latest_message = None
        return message

    def _on_timer(self) -> None:
        """Decode and infer one pending frame without building a backlog."""
        message = self._take_latest_message()
        if message is None:
            return

        started_at = perf_counter()
        try:
            full_image = image_message_to_bgr(message, self._image_transport)
            if self._crop_enabled:
                crop = normalized_crop_to_pixels(
                    full_image.shape, self._crop_normalized
                )
            else:
                crop = (0, 0, full_image.shape[1], full_image.shape[0])
            left, top, right, bottom = crop
            inference_image = full_image[top:bottom, left:right]

            raw_detections = filter_excluded_class_ids(
                self._backend.infer(inference_image),
                self._excluded_class_ids,
            )
            detections: List[Detection] = [
                restore_detection_to_full_image(
                    detection, crop, full_image.shape
                )
                for detection in raw_detections
            ]
            self._publisher.publish(
                detections_to_message(message.header, detections)
            )
        except Exception as exc:
            self.get_logger().error(
                f"Dynamic-obstacle inference failed: {exc}",
                throttle_duration_sec=5.0,
            )
            return

        elapsed_ms = (perf_counter() - started_at) * 1000.0
        self.get_logger().debug(
            f"Published {len(detections)} detections in {elapsed_ms:.1f} ms"
        )


def main(args=None) -> None:
    """Run the dynamic-obstacle detector node."""
    rclpy.init(args=args)
    node = None
    try:
        node = DynamicObstacleDetectorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
