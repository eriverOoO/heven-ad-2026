"""Select, stabilize, and publish mm-2025 traffic-light aspects."""

from time import monotonic

from ad_interfaces.msg import TrafficLightStatus
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from vision_msgs.msg import Detection2DArray

from ad_camera_perception.traffic_light.traffic import (
    EvaluatedSignal,
    SignalVoteFilter,
    TargetSelector,
    TrafficDetection,
)


def _traffic_detection(message) -> TrafficDetection:
    if message.results:
        hypothesis = message.results[0].hypothesis
        class_name = str(hypothesis.class_id)
        confidence = float(hypothesis.score)
    else:
        class_name = ""
        confidence = 0.0
    center = message.bbox.center.position
    return TrafficDetection(
        detection_id=str(message.id),
        class_name=class_name,
        confidence=confidence,
        center_x=float(center.x),
        center_y=float(center.y),
        width=float(message.bbox.size_x),
        height=float(message.bbox.size_y),
    )


class TrafficLightEvaluatorNode(Node):
    """Publish independent aspects without making a stop/go decision."""

    def __init__(self) -> None:
        super().__init__("traffic_light_evaluator")
        self.declare_parameter(
            "detections_topic", "/vision/traffic_light/detections"
        )
        self.declare_parameter(
            "status_topic", "/vision/traffic_light/status"
        )
        self.declare_parameter("expected_image_width", 1280)
        self.declare_parameter("expected_image_height", 720)
        self.declare_parameter(
            "target_roi_normalized", [0.25, 0.10, 0.85, 0.95]
        )
        self.declare_parameter("confidence_weight", 0.60)
        self.declare_parameter("area_weight", 0.25)
        self.declare_parameter("center_weight", 0.15)
        self.declare_parameter("voting_window_frames", 5)
        self.declare_parameter("minimum_vote_frames", 3)
        self.declare_parameter("stale_timeout_sec", 0.5)

        detections_topic = str(self.get_parameter("detections_topic").value)
        status_topic = str(self.get_parameter("status_topic").value)
        self._stale_timeout = float(
            self.get_parameter("stale_timeout_sec").value
        )
        if self._stale_timeout <= 0.0:
            raise ValueError("stale_timeout_sec must be positive")
        self._selector = TargetSelector(
            image_width=int(self.get_parameter("expected_image_width").value),
            image_height=int(self.get_parameter("expected_image_height").value),
            target_roi_normalized=self.get_parameter(
                "target_roi_normalized"
            ).value,
            confidence_weight=float(
                self.get_parameter("confidence_weight").value
            ),
            area_weight=float(self.get_parameter("area_weight").value),
            center_weight=float(self.get_parameter("center_weight").value),
        )
        self._filter = SignalVoteFilter(
            window_frames=int(
                self.get_parameter("voting_window_frames").value
            ),
            minimum_vote_frames=int(
                self.get_parameter("minimum_vote_frames").value
            ),
        )
        self._last_receipt_time = None
        self._last_header = None
        self._stale_published = False

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._publisher = self.create_publisher(
            TrafficLightStatus, status_topic, qos
        )
        self._subscription = self.create_subscription(
            Detection2DArray, detections_topic, self._on_detections, qos
        )
        self._timer = self.create_timer(
            min(0.1, self._stale_timeout / 2.0), self._on_stale_timer
        )
        self.get_logger().info(
            f"Evaluating {detections_topic}; publishing aspects on {status_topic}"
        )

    @staticmethod
    def _status(header, evaluated: EvaluatedSignal) -> TrafficLightStatus:
        message = TrafficLightStatus()
        message.header = header
        message.valid = evaluated.valid
        message.confidence = float(evaluated.confidence)
        message.red = evaluated.aspects.red
        message.yellow = evaluated.aspects.yellow
        message.straight_green = evaluated.aspects.straight_green
        message.left_green = evaluated.aspects.left_green
        message.source_class = evaluated.source_class
        message.detection_id = evaluated.detection_id
        return message

    def _on_detections(self, message: Detection2DArray) -> None:
        self._last_receipt_time = monotonic()
        self._last_header = message.header
        self._stale_published = False
        detections = [_traffic_detection(item) for item in message.detections]
        selected = self._selector.select(detections)
        self._publisher.publish(
            self._status(message.header, self._filter.update(selected))
        )

    def _on_stale_timer(self) -> None:
        if self._last_receipt_time is None or self._last_header is None:
            return
        if self._stale_published:
            return
        if monotonic() - self._last_receipt_time <= self._stale_timeout:
            return
        self._filter.clear()
        self._publisher.publish(
            self._status(self._last_header, EvaluatedSignal(valid=False))
        )
        self._stale_published = True


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = TrafficLightEvaluatorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
