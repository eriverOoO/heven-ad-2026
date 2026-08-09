"""ROS 2 detector node for the mm-2025 YOLOv7 traffic-light model."""

from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import List, Optional

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from vision_msgs.msg import Detection2DArray

from ad_camera_perception.inference.detection import Detection
from ad_camera_perception.inference.detection_converter import (
    detections_to_message,
    normalized_crop_to_pixels,
    restore_detection_to_full_image,
)
from ad_camera_perception.inference.yolov7_backend import YoloV7Backend
from ad_camera_perception.utils.image_messages import (
    ImageMessage,
    image_message_to_bgr,
    image_message_type,
)
from ad_camera_perception.utils.parameters import validate_normalized_crop


def _resolve_model_path(value: str) -> str:
    path = Path(value).expanduser()
    if path.is_file() or path.is_absolute():
        return str(path)
    package_share = Path(get_package_share_directory("ad_camera_perception"))
    candidate = package_share / path
    return str(candidate)


class TrafficLightDetectorNode(Node):
    """Keep the newest Camera-4 frame and publish YOLOv7 detections."""

    def __init__(self) -> None:
        super().__init__("traffic_light_detector")
        self.declare_parameter(
            "image_topic", "/ad/sensors/camera/traffic_light/compressed"
        )
        self.declare_parameter(
            "detections_topic", "/vision/traffic_light/detections"
        )
        self.declare_parameter("image_transport", "compressed")
        self.declare_parameter("model_path", "models/yolov7_best.pt")
        self.declare_parameter("yolov7_repository_path", "")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("image_size", 640)
        self.declare_parameter("confidence_threshold", 0.25)
        self.declare_parameter("iou_threshold", 0.45)
        self.declare_parameter("maximum_inference_rate", 20.0)
        self.declare_parameter(
            "target_roi_normalized", [0.142857, 0.0, 0.857143, 1.0]
        )

        image_topic = str(self.get_parameter("image_topic").value)
        detections_topic = str(self.get_parameter("detections_topic").value)
        self._image_transport = str(self.get_parameter("image_transport").value)
        model_path = _resolve_model_path(
            str(self.get_parameter("model_path").value)
        )
        repository_path = str(
            self.get_parameter("yolov7_repository_path").value
        )
        device = str(self.get_parameter("device").value)
        image_size = int(self.get_parameter("image_size").value)
        confidence_threshold = float(
            self.get_parameter("confidence_threshold").value
        )
        iou_threshold = float(self.get_parameter("iou_threshold").value)
        maximum_rate = float(
            self.get_parameter("maximum_inference_rate").value
        )
        self._target_roi = validate_normalized_crop(
            self.get_parameter("target_roi_normalized").value
        )
        if not repository_path:
            raise ValueError(
                "yolov7_repository_path must point to the external "
                "YOLOv7 source directory"
            )
        if maximum_rate <= 0.0:
            raise ValueError("maximum_inference_rate must be positive")

        self.get_logger().info(f"Loading mm-2025 traffic model: {model_path}")
        self._backend = YoloV7Backend(
            model_path=model_path,
            repository_path=repository_path,
            device=device,
            image_size=image_size,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
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
        self._subscription = self.create_subscription(
            image_message_type(self._image_transport),
            image_topic,
            self._on_image,
            qos_profile_sensor_data,
        )
        self._timer = self.create_timer(1.0 / maximum_rate, self._on_timer)
        self.get_logger().info(
            f"Camera-4 input {image_topic}; YOLO inference ROI "
            f"{self._target_roi}; publishing {detections_topic}"
        )

    def _on_image(self, message: ImageMessage) -> None:
        with self._message_lock:
            self._latest_message = message

    def _take_latest_message(self) -> Optional[ImageMessage]:
        with self._message_lock:
            message = self._latest_message
            self._latest_message = None
        return message

    def _on_timer(self) -> None:
        message = self._take_latest_message()
        if message is None:
            return
        started_at = perf_counter()
        try:
            full_image = image_message_to_bgr(message, self._image_transport)
            crop = normalized_crop_to_pixels(full_image.shape, self._target_roi)
            left, top, right, bottom = crop
            inference_image = full_image[top:bottom, left:right]
            raw_detections = self._backend.infer(inference_image)
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
                f"Traffic-light inference failed: {exc}",
                throttle_duration_sec=5.0,
            )
            return
        elapsed_ms = (perf_counter() - started_at) * 1000.0
        self.get_logger().debug(
            f"Published {len(detections)} traffic detections in {elapsed_ms:.1f} ms"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = TrafficLightDetectorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
