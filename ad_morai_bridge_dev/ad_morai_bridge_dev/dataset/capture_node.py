"""rclpy shell around :class:`CaptureSession`.

Subscribes the canonical topic contract, pumps deserialised messages into
the session, and requests shutdown once the frame / duration limit is hit.
The node performs no scenario reset itself - that is the CLI's job, so the
node stays trivially testable and reusable.
"""

from __future__ import annotations

from typing import Any

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from ad_morai_bridge_dev.dataset.capture import CaptureLimits, CaptureSession
from ad_morai_bridge_dev.dataset.schema import (
    ACTOR_GT_TOPIC,
    EGO_GT_TOPIC,
    LIDAR_TOPIC,
    TF_STATIC_TOPIC,
    TF_TOPIC,
    SkewConfig,
)
from ad_morai_bridge_dev.dataset.transforms import StaticExtrinsics


def _reliable(depth: int) -> QoSProfile:
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
    )


def _tf_static_qos() -> QoSProfile:
    profile = _reliable(1)
    profile.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
    return profile


class DatasetCaptureNode(Node):
    def __init__(
        self,
        writer,
        limits: CaptureLimits,
        skew: SkewConfig,
        scenario_file: str,
        scenario_sha256: str,
        **session_kwargs: Any,
    ) -> None:
        super().__init__("ad_morai_dataset_capture")
        from ad_morai_interfaces_dev.msg import EgoVehicleStatus, ObjectStatusArray
        from sensor_msgs.msg import PointCloud2
        from tf2_msgs.msg import TFMessage

        self._pending_writer = writer
        self._limits = limits
        self._skew = skew
        self._scenario_file = scenario_file
        self._scenario_sha256 = scenario_sha256
        self._session_kwargs = session_kwargs
        self._session: CaptureSession | None = None
        self._static: StaticExtrinsics | None = None
        self.result_status = "aborted"

        self.create_subscription(
            TFMessage, TF_STATIC_TOPIC, self._on_tf_static, _tf_static_qos()
        )
        self.create_subscription(TFMessage, TF_TOPIC, self._on_tf, _reliable(100))
        self.create_subscription(
            EgoVehicleStatus, EGO_GT_TOPIC, self._on_ego, _reliable(50)
        )
        self.create_subscription(
            ObjectStatusArray, ACTOR_GT_TOPIC, self._on_actor, _reliable(50)
        )
        self.create_subscription(
            PointCloud2, LIDAR_TOPIC, self._on_lidar, _reliable(10)
        )

    def _on_tf_static(self, message: Any) -> None:
        if self._static is not None:
            return
        try:
            self._static = StaticExtrinsics.from_tf_static(list(message.transforms))
        except ValueError as exc:
            self.get_logger().warn(f"incomplete /tf_static: {exc}")
            return
        self._session = CaptureSession(
            self._pending_writer,
            self._static,
            self._skew,
            self._limits,
            self._scenario_file,
            self._scenario_sha256,
            logger=lambda m: self.get_logger().info(m),
            **self._session_kwargs,
        )
        self.get_logger().info("static extrinsics received; capture armed")

    def _on_tf(self, message: Any) -> None:
        if self._session is not None:
            self._session.on_tf(message)

    def _on_ego(self, message: Any) -> None:
        if self._session is not None:
            self._session.on_ego(message)

    def _on_actor(self, message: Any) -> None:
        if self._session is not None:
            self._session.on_actor_gt(message)

    def _on_lidar(self, message: Any) -> None:
        if self._session is None:
            return
        try:
            action = self._session.on_lidar(message)
        except Exception as exc:  # noqa: BLE001 - finalize then re-raise
            self.get_logger().error(f"capture failed: {exc}")
            self._session.finalize("exception", error=str(exc))
            self.result_status = "error"
            rclpy.shutdown()
            raise
        if action == "captured_final":
            self.get_logger().info("frame limit reached")
            self._session.finalize("limit_reached")
            self.result_status = "complete"
            rclpy.shutdown()

    def abort(self, reason: str) -> None:
        if self._session is not None and not self._session.finalized:
            self._session.finalize(reason)


def spin_capture(node: DatasetCaptureNode) -> str:
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.abort("keyboard_interrupt")
    return node.result_status
