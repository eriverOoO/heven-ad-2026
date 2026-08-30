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

import math
from pathlib import Path
import statistics
import time

import rclpy
from autoware_perception_msgs.msg import (
    DetectedObjects,
    ObjectClassification,
    Shape,
    TrackedObject,
    TrackedObjectKinematics,
    TrackedObjects,
)
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
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
    track_id_to_uuid,
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
        self.declare_parameter("euclidean_gate_m", 3.0)
        self.declare_parameter("mahalanobis_gate", 11.62)
        self.declare_parameter("mahalanobis_max_distance_m", 0.0)
        self.declare_parameter("state_estimator", "linear_kf")
        self.declare_parameter("yaw_measurement_mode", "detector")
        self.declare_parameter("imm_cv_to_cv_probability", 0.95)
        self.declare_parameter("imm_ctrv_to_ctrv_probability", 0.95)
        self.declare_parameter("kalmannet_checkpoint", "")
        self.declare_parameter("kalmannet_device", "cpu")
        self.declare_parameter("runtime_summary_interval_frames", 0)
        self.declare_parameter("velocity_audit_enabled", False)
        self.declare_parameter(
            "velocity_audit_topic",
            "/ad/perception/objects/tracked/ab3dmot_velocity_audit",
        )

        self.enabled = bool(self.get_parameter("enabled").value)
        self.target_frame = str(self.get_parameter("target_frame").value)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._tracker: AB3DMOTTracker | None = None
        self._last_stamp_ns: int | None = None
        self._runtime_summary_interval_frames = int(
            self.get_parameter("runtime_summary_interval_frames").value
        )
        if self._runtime_summary_interval_frames < 0:
            raise ValueError("runtime_summary_interval_frames must be >= 0")
        self._step_latency_ms: list[float] = []
        self._tracks_created = 0
        self._tracks_deleted = 0
        self._previous_live_track_ids: set[int] = set()
        if self.enabled:
            self._tracker = self._build_tracker()

        self.publisher = self.create_publisher(
            TrackedObjects, str(self.get_parameter("output_topic").value), 1
        )
        self._velocity_audit_publisher = None
        if bool(self.get_parameter("velocity_audit_enabled").value):
            self._velocity_audit_publisher = self.create_publisher(
                DiagnosticArray,
                str(self.get_parameter("velocity_audit_topic").value),
                1,
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
            euclidean_gate_m=float(self.get_parameter("euclidean_gate_m").value),
            mahalanobis_gate=float(self.get_parameter("mahalanobis_gate").value),
            mahalanobis_max_distance_m=float(self.get_parameter("mahalanobis_max_distance_m").value),
            state_estimator=str(self.get_parameter("state_estimator").value),
            yaw_measurement_mode=str(self.get_parameter("yaw_measurement_mode").value),
            imm_cv_to_cv_probability=float(self.get_parameter("imm_cv_to_cv_probability").value),
            imm_ctrv_to_ctrv_probability=float(self.get_parameter("imm_ctrv_to_ctrv_probability").value),
            kalmannet_checkpoint=str(self.get_parameter("kalmannet_checkpoint").value),
            kalmannet_device=str(self.get_parameter("kalmannet_device").value),
        )
        root_param = str(self.get_parameter("ab3dmot_root").value)
        ab3dmot_root = Path(root_param) if root_param else None
        tracker = AB3DMOTTracker(config, ab3dmot_root)
        # T-9B Phase 16: log checkpoint provenance exactly once, at build
        # time -- never per-frame/per-track.
        if tracker.kalmannet_provenance is not None:
            self.get_logger().info(
                "KalmanNet checkpoint loaded: "
                f"path={tracker.kalmannet_provenance['checkpoint_path']} "
                f"sha256={tracker.kalmannet_provenance['checkpoint_sha256']} "
                f"hidden_size={tracker.kalmannet_provenance['hidden_size']} "
                f"n_trainable_params={tracker.kalmannet_provenance['n_trainable_params']} "
                f"device={tracker.kalmannet_provenance['device']}"
            )
        return tracker

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
        if decision is TimestampDecision.REJECT_ROLLBACK:
            self.get_logger().warn(
                "rejected DetectedObjects: timestamp moved backwards"
            )
            return

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
            step_started = time.perf_counter()
            states = self._tracker.step(detections, timestamp_seconds)
            step_latency_ms = (time.perf_counter() - step_started) * 1000.0
        except ValueError as error:
            # Should not happen: the non-increasing timestamp gating above already
            # guarantees a strictly-increasing timestamp reaches step().
            # Kept as a defensive backstop, matching this repo's existing
            # reject-rather-than-crash policy for malformed timing.
            self.get_logger().error(f"AB3DMOT tracker rejected step: {error}")
            return

        self._record_runtime_metrics(step_latency_ms)

        output = tracked_states_to_message(
            states,
            msg.header.stamp,
            self.target_frame,
            MESSAGE_TYPES,
            orientation_available=(
                self._tracker.config.yaw_measurement_mode != "unobserved"
            ),
        )
        self.publisher.publish(output)
        self._publish_velocity_audit(states, msg.header.stamp)
        self._last_stamp_ns = stamp_ns

    def _publish_velocity_audit(self, states, stamp) -> None:
        if self._velocity_audit_publisher is None:
            return
        audit = DiagnosticArray()
        audit.header.stamp = stamp
        audit.header.frame_id = self.target_frame
        for state in states:
            status = DiagnosticStatus()
            status.level = DiagnosticStatus.OK
            status.name = bytes(track_id_to_uuid(state.track_id)).hex()
            status.hardware_id = "ab3dmot_world_velocity"
            status.message = "internal velocity before ROS serialization"
            status.values = [
                KeyValue(key="vx_world_mps", value=repr(float(state.vx_mps))),
                KeyValue(key="vy_world_mps", value=repr(float(state.vy_mps))),
            ]
            audit.status.append(status)
        self._velocity_audit_publisher.publish(audit)

    def _record_runtime_metrics(self, step_latency_ms: float) -> None:
        if self._tracker is None:
            return
        current_ids = {track.track_id for track in self._tracker.tracks}
        self._tracks_created += len(current_ids - self._previous_live_track_ids)
        self._tracks_deleted += len(self._previous_live_track_ids - current_ids)
        self._previous_live_track_ids = current_ids
        self._step_latency_ms.append(step_latency_ms)

        interval = self._runtime_summary_interval_frames
        if interval <= 0 or len(self._step_latency_ms) % interval != 0:
            return
        ordered = sorted(self._step_latency_ms)
        p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
        self.get_logger().info(
            "AB3DMOT_RUNTIME_SUMMARY "
            f"frames={len(ordered)} "
            f"tracks_created={self._tracks_created} "
            f"tracks_deleted={self._tracks_deleted} "
            f"live_tracks={len(current_ids)} "
            f"median_step_ms={statistics.median(ordered):.6f} "
            f"p95_step_ms={ordered[p95_index]:.6f} "
            f"max_step_ms={ordered[-1]:.6f}"
        )


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
