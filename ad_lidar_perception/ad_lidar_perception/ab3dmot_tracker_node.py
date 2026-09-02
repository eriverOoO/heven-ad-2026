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
from ad_lidar_perception.ab3dmot_tf_deferred_queue import (
    DeferredDetectionQueue,
    ReleaseEvent,
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
        # Option C published-uncertainty bound (see
        # docs/perception/competition_dynamic_object_pipeline.md). <= 0
        # disables it; the internal KF covariance is never affected either
        # way. COMPETITION_MOT_BASELINE_V1 sets 7.0 m.
        self.declare_parameter("maximum_position_std_m", 0.0)
        self.declare_parameter("velocity_audit_enabled", False)
        self.declare_parameter(
            "velocity_audit_topic",
            "/ad/perception/objects/tracked/ab3dmot_velocity_audit",
        )
        # AB3DMOT Exact-Stamp TF Deferred Processing v1 (see
        # docs/perception/ab3dmot_tf_deferred_processing_v1.md): when the
        # exact-stamp `target_frame <- detection frame_id` transform is not
        # yet available because only its FUTURE side hasn't arrived (a
        # tf2 "extrapolation into the future" rejection), hold the
        # detection in a small bounded FIFO queue instead of discarding it
        # immediately, and process it -- still at its ORIGINAL stamp -- once
        # tf2 can supply the exact transform. Never extrapolates, never
        # blocks, never reorders. Any other TF failure (unknown frame,
        # extrapolation into the past / evicted, disconnected tree) still
        # fails fast exactly as before. Default `true`: this only ever
        # widens the set of frames a strictly-experimental tracker accepts
        # under the same exact-stamp geometry and safety guarantees: no
        # timestamp is changed, no transform is fabricated, memory/wait are
        # bounded. See "Production default decision" in the design doc.
        self.declare_parameter("defer_until_tf_ready", True)
        # Evidence-derived default: see Phase 2 of the design doc
        # (measured detection-arrival -> exact-stamp-transform-ready lag on
        # the morai_cam4_20260813_163222 replay with enable_localization:=true).
        self.declare_parameter("max_tf_wait_ms", 500)
        self.declare_parameter("max_pending_detections", 8)

        self.enabled = bool(self.get_parameter("enabled").value)
        self.target_frame = str(self.get_parameter("target_frame").value)
        self._defer_until_tf_ready = bool(self.get_parameter("defer_until_tf_ready").value)
        max_tf_wait_ms = int(self.get_parameter("max_tf_wait_ms").value)
        max_pending_detections = int(self.get_parameter("max_pending_detections").value)
        if max_tf_wait_ms <= 0:
            raise ValueError("max_tf_wait_ms must be > 0")
        if max_pending_detections <= 0:
            raise ValueError("max_pending_detections must be > 0")

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._deferred_queue: DeferredDetectionQueue | None = None
        if self._defer_until_tf_ready:
            self._deferred_queue = DeferredDetectionQueue(
                max_pending=max_pending_detections,
                max_wait_s=max_tf_wait_ms / 1000.0,
                can_transform=self._can_transform_for,
            )
        self._last_admitted_stamp_ns: int | None = None

        self._tracker: AB3DMOTTracker | None = None
        self._runtime_summary_interval_frames = int(
            self.get_parameter("runtime_summary_interval_frames").value
        )
        if self._runtime_summary_interval_frames < 0:
            raise ValueError("runtime_summary_interval_frames must be >= 0")
        self._maximum_position_std_m = float(
            self.get_parameter("maximum_position_std_m").value
        )
        if self._maximum_position_std_m < 0.0:
            raise ValueError("maximum_position_std_m must be >= 0")
        self._step_latency_ms: list[float] = []
        self._tracks_created = 0
        self._tracks_deleted = 0
        self._position_covariance_bounded = 0
        self._frames_with_position_covariance_bounded = 0
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

        # Release anything already resolvable, oldest-first, before
        # admitting the newly-arrived message -- this is what lets a
        # detection whose exact-stamp transform just became available get
        # processed even without a *new* detection sharing that exact
        # readiness event (Phase 12/21: event-driven, non-blocking drain).
        if self._deferred_queue is not None:
            self._drain_deferred_queue()

        try:
            stamp_ns = stamp_to_ns(msg.header.stamp)
        except DetectedObjectsAdapterError as error:
            self.get_logger().warn(f"rejected DetectedObjects: malformed stamp: {error}")
            return

        decision = classify_timestamp(stamp_ns, self._last_admitted_stamp_ns)
        if decision is TimestampDecision.SKIP_DUPLICATE:
            self.get_logger().warn("rejected DetectedObjects: duplicate timestamp")
            return
        if decision is TimestampDecision.REJECT_ROLLBACK:
            self.get_logger().warn(
                "rejected DetectedObjects: timestamp moved backwards"
            )
            return

        if self._deferred_queue is None:
            # defer_until_tf_ready:=false -- byte-identical to the
            # pre-existing behaviour: any TF failure drops the frame
            # immediately, never deferred.
            self._last_admitted_stamp_ns = stamp_ns
            self._process_detection(msg, stamp_ns)
            return

        # Admitted into the pipeline now (whether processed immediately or
        # deferred): a duplicate/rollback of this exact stamp arriving
        # while this message is still pending must still be rejected, so
        # this must advance *before* the (possibly deferred) outcome is
        # known -- not only once processing eventually completes.
        self._last_admitted_stamp_ns = stamp_ns
        event, payload = self._deferred_queue.offer(stamp_ns, msg)
        if event is ReleaseEvent.READY:
            self._process_detection(payload, stamp_ns)
        elif event is ReleaseEvent.DROPPED_PERMANENT:
            self.get_logger().warn(
                "rejected DetectedObjects: TF permanently unavailable "
                f"stamp_ns={stamp_ns}"
            )
        # event is ReleaseEvent.DEFERRED: nothing further to do here; a
        # later drain() call (triggered by the next detection callback)
        # will release this same message exactly once.

    def _can_transform_for(self, msg: DetectedObjects) -> tuple[bool, str]:
        """Non-blocking, exact-stamp transform-availability check for one
        `DetectedObjects` message. An empty message needs no transform at
        all, so it is trivially "ready" without ever touching the TF
        buffer -- matches the pre-existing skip-TF-for-empty-messages
        contract exactly."""
        if not msg.objects:
            return True, ""
        return self._tf_buffer.can_transform(
            self.target_frame,
            msg.header.frame_id,
            msg.header.stamp,
            return_debug_tuple=True,
        )

    def _drain_deferred_queue(self) -> None:
        assert self._deferred_queue is not None
        for event, msg in self._deferred_queue.drain():
            try:
                stamp_ns = stamp_to_ns(msg.header.stamp)
            except DetectedObjectsAdapterError:
                # Cannot happen: only already-`stamp_to_ns`-validated
                # messages are ever offered to the queue. Kept as a
                # defensive backstop matching this repo's reject-rather-
                # than-crash convention.
                continue
            if event is ReleaseEvent.READY:
                self._process_detection(msg, stamp_ns)
            elif event is ReleaseEvent.DROPPED_TIMEOUT:
                self.get_logger().warn(
                    "dropped deferred DetectedObjects: tf_timeout "
                    f"stamp_ns={stamp_ns}"
                )
            elif event is ReleaseEvent.DROPPED_PERMANENT:
                self.get_logger().warn(
                    "dropped deferred DetectedObjects: TF permanently "
                    f"unavailable stamp_ns={stamp_ns}"
                )

    def _process_detection(self, msg: DetectedObjects, stamp_ns: int) -> None:
        """Runs the actual AB3DMOT tracker step for one already-admitted
        message, using its own original stamp. Called either immediately
        (TF already exact-stamp available) or later, from
        `_drain_deferred_queue`, once it becomes available -- both paths
        converge on this single tracker-update implementation (no
        copy/paste algorithm branch)."""
        transform = None
        if msg.objects:
            try:
                transform = self._tf_buffer.lookup_transform(
                    self.target_frame, msg.header.frame_id, msg.header.stamp
                )
            except (LookupException, ConnectivityException, ExtrapolationException) as error:
                # `_can_transform_for` said ready (or deferral is
                # disabled), but the underlying lookup still failed -- a
                # genuine race is possible between the non-blocking check
                # and this call. Fails exactly like the pre-existing
                # unconditional-drop behaviour; never re-enqueued (would
                # risk unbounded bouncing).
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
            # Should not happen: the non-increasing timestamp gating in
            # `_on_detected_objects` already guarantees a strictly-
            # increasing timestamp reaches step(), and the deferred queue
            # never reorders admitted messages. Kept as a defensive
            # backstop, matching this repo's existing reject-rather-than-
            # crash policy for malformed timing.
            self.get_logger().error(f"AB3DMOT tracker rejected step: {error}")
            return

        self._record_runtime_metrics(step_latency_ms)

        output, position_covariance_bounded = tracked_states_to_message(
            states,
            msg.header.stamp,
            self.target_frame,
            MESSAGE_TYPES,
            orientation_available=(
                self._tracker.config.yaw_measurement_mode != "unobserved"
            ),
            maximum_position_std_m=self._maximum_position_std_m,
        )
        self._position_covariance_bounded += position_covariance_bounded
        if position_covariance_bounded:
            self._frames_with_position_covariance_bounded += 1
        self.publisher.publish(output)
        self._publish_velocity_audit(states, msg.header.stamp)

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
        deferred_summary = ""
        if self._deferred_queue is not None:
            s = self._deferred_queue.stats
            deferred_summary = (
                " deferred_tf_detections_received="
                f"{s.detections_received} "
                f"deferred_tf_processed_immediately={s.processed_immediately} "
                f"deferred_tf_processed_deferred={s.processed_deferred} "
                f"deferred_tf_dropped_timeout={s.dropped_tf_timeout} "
                f"deferred_tf_dropped_permanent={s.dropped_permanent_tf_failure} "
                f"deferred_tf_dropped_overflow={s.dropped_queue_overflow} "
                f"deferred_tf_queue_depth={s.pending_queue_depth} "
                f"deferred_tf_max_queue_depth={s.max_observed_queue_depth}"
            )
        self.get_logger().info(
            "AB3DMOT_RUNTIME_SUMMARY "
            f"frames={len(ordered)} "
            f"tracks_created={self._tracks_created} "
            f"tracks_deleted={self._tracks_deleted} "
            f"live_tracks={len(current_ids)} "
            f"position_cov_bounded={self._position_covariance_bounded} "
            f"position_cov_bounded_frames={self._frames_with_position_covariance_bounded} "
            f"median_step_ms={statistics.median(ordered):.6f} "
            f"p95_step_ms={ordered[p95_index]:.6f} "
            f"max_step_ms={ordered[-1]:.6f}"
            f"{deferred_summary}"
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
