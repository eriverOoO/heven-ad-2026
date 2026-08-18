#!/usr/bin/env python3
"""Optional AB3DMOT-based ROS2 tracker (T-1B); Autoware remains the default.

Subscribes the same `/ad/perception/objects/detected` contract the
production Autoware `multi_object_tracker` already consumes, transforms
detections into `odom`, drives the T-1A `AB3DMOTTracker` core with the
message's own timestamp, and publishes to a separate experimental topic
(`/experiment/tracked/ab3dmot` by default). Does not touch
`tracking.launch.py`, `autoware.yaml`, or the production tracker path --
both can run simultaneously off the same detections.
"""

from __future__ import annotations

from pathlib import Path

import rclpy
from autoware_perception_msgs.msg import (
    DetectedObjects,
    ObjectClassification,
    Shape,
    TrackedObject,
    TrackedObjectKinematics,
    TrackedObjects,
)
from rclpy.node import Node
from tf2_ros import (
    Buffer,
    ConnectivityException,
    ExtrapolationException,
    LookupException,
    TransformListener,
)

from ad_lidar_perception.ab3dmot_config import AB3DMOTConfig
from ad_lidar_perception.ab3dmot_core import AB3DMOTTracker
from ad_lidar_perception.ab3dmot_ros import (
    DetectedObjectsAdapterError,
    TimestampDecision,
    classify_timestamp,
    detected_objects_to_detections,
    stamp_to_ns,
    tracked_states_to_message,
)

MESSAGE_TYPES = {
    "TrackedObject": TrackedObject,
    "TrackedObjectKinematics": TrackedObjectKinematics,
    "TrackedObjects": TrackedObjects,
    "ObjectClassification": ObjectClassification,
    "Shape": Shape,
}


class Ab3dmotTrackerNode(Node):
    def __init__(self, **node_kwargs) -> None:
        # `**node_kwargs` forwards straight to `rclpy.node.Node` (e.g.
        # `parameter_overrides=[...]`), which is how tests construct this
        # node with `enabled=True` without a live launch/CLI invocation.
        super().__init__("ad_ab3dmot_tracker", **node_kwargs)
        self.declare_parameter("enabled", False)
        self.declare_parameter("input_topic", "/ad/perception/objects/detected")
        self.declare_parameter("output_topic", "/experiment/tracked/ab3dmot")
        self.declare_parameter("target_frame", "odom")
        self.declare_parameter("ab3dmot_root", "")
        self.declare_parameter("association_metric", "giou_3d")
        self.declare_parameter("matcher", "greedy")
        self.declare_parameter("min_hits", 1)
        self.declare_parameter("max_age", 2)
        self.declare_parameter("giou_gate", 0.0)

        self.enabled = bool(self.get_parameter("enabled").value)
        self.target_frame = str(self.get_parameter("target_frame").value)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._tracker: AB3DMOTTracker | None = None
        self._last_stamp_ns: int | None = None
        if self.enabled:
            self._tracker = self._build_tracker()

        self.publisher = self.create_publisher(
            TrackedObjects, str(self.get_parameter("output_topic").value), 1
        )
        self.subscription = self.create_subscription(
            DetectedObjects,
            str(self.get_parameter("input_topic").value),
            self._on_detected_objects,
            10,
        )

    def _build_tracker(self) -> AB3DMOTTracker:
        config = AB3DMOTConfig(
            association_metric=str(self.get_parameter("association_metric").value),
            matcher=str(self.get_parameter("matcher").value),
            min_hits=int(self.get_parameter("min_hits").value),
            max_age=int(self.get_parameter("max_age").value),
            giou_gate=float(self.get_parameter("giou_gate").value),
        )
        root_param = str(self.get_parameter("ab3dmot_root").value)
        ab3dmot_root = Path(root_param) if root_param else None
        return AB3DMOTTracker(config, ab3dmot_root)

    def _on_detected_objects(self, msg: DetectedObjects) -> None:
        if not self.enabled or self._tracker is None:
            return

        try:
            stamp_ns = stamp_to_ns(msg.header.stamp)
        except DetectedObjectsAdapterError as error:
            self.get_logger().warn(f"rejected DetectedObjects: malformed stamp: {error}")
            return

        decision = classify_timestamp(stamp_ns, self._last_stamp_ns)
        if decision is TimestampDecision.SKIP_DUPLICATE:
            self.get_logger().warn("rejected DetectedObjects: duplicate timestamp")
            return
        if decision is TimestampDecision.RESET_ROLLBACK:
            # MORAI resets simulated time together with object tracks --
            # reset AB3DMOT's experimental state cleanly instead of ever
            # propagating a negative dt into the KF, mirroring
            # AutowarePredictionNode's own clock-rollback handling.
            self.get_logger().info(
                "detected clock rollback; resetting the experimental AB3DMOT tracker"
            )
            self._tracker = self._build_tracker()
            self._last_stamp_ns = None

        transform = None
        if msg.objects:
            try:
                transform = self._tf_buffer.lookup_transform(
                    self.target_frame, msg.header.frame_id, msg.header.stamp
                )
            except (LookupException, ConnectivityException, ExtrapolationException) as error:
                self.get_logger().warn(f"rejected DetectedObjects: TF unavailable: {error}")
                return

        try:
            detections = detected_objects_to_detections(msg, transform)
        except DetectedObjectsAdapterError as error:
            self.get_logger().warn(f"rejected DetectedObjects: {error}")
            return

        timestamp_seconds = stamp_ns * 1.0e-9
        try:
            states = self._tracker.step(detections, timestamp_seconds)
        except ValueError as error:
            # Should not happen: the duplicate/rollback gating above already
            # guarantees a strictly-increasing timestamp reaches step().
            # Kept as a defensive backstop, matching this repo's existing
            # reject-rather-than-crash policy for malformed timing.
            self.get_logger().error(f"AB3DMOT tracker rejected step: {error}")
            return

        output = tracked_states_to_message(
            states, msg.header.stamp, self.target_frame, MESSAGE_TYPES
        )
        self.publisher.publish(output)
        self._last_stamp_ns = stamp_ns


def main() -> None:
    rclpy.init()
    node = Ab3dmotTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
