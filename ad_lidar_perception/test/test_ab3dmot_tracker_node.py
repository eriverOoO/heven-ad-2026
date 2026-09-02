import math
import unittest

import rclpy
from autoware_perception_msgs.msg import (
    DetectedObject,
    DetectedObjectKinematics,
    DetectedObjects,
    ObjectClassification,
    Shape,
)
from geometry_msgs.msg import TransformStamped
from rclpy.parameter import Parameter
from tf2_ros import LookupException

from ad_lidar_perception.ab3dmot_core import _load_ab3dmot_kf_class
from ad_lidar_perception.ab3dmot_tracker_node import Ab3dmotTrackerNode


def make_detected_objects(sec, nanosec, boxes, frame_id="lidar_link"):
    msg = DetectedObjects()
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = nanosec
    msg.header.frame_id = frame_id
    for x, y, z, yaw, length, width, height, label, score in boxes:
        obj = DetectedObject()
        obj.existence_probability = float(score)
        classification = ObjectClassification()
        classification.label = int(label)
        classification.probability = float(score)
        obj.classification.append(classification)
        pose = obj.kinematics.pose_with_covariance.pose
        pose.position.x, pose.position.y, pose.position.z = float(x), float(y), float(z)
        pose.orientation.z = float(math.sin(0.5 * yaw))
        pose.orientation.w = float(math.cos(0.5 * yaw))
        obj.kinematics.orientation_availability = DetectedObjectKinematics.AVAILABLE
        obj.shape.type = Shape.BOUNDING_BOX
        obj.shape.dimensions.x = float(length)
        obj.shape.dimensions.y = float(width)
        obj.shape.dimensions.z = float(height)
        msg.objects.append(obj)
    return msg


def identity_transform():
    transform = TransformStamped()
    transform.transform.rotation.w = 1.0
    return transform


class RecordingPublisher:
    def __init__(self):
        self.published = []

    def publish(self, message):
        self.published.append(message)


def build_enabled_node(**parameters):
    overrides = [Parameter("enabled", Parameter.Type.BOOL, True)]
    overrides.extend(
        Parameter(name, value=value) for name, value in parameters.items()
    )
    node = Ab3dmotTrackerNode(
        parameter_overrides=overrides
    )
    node.publisher = RecordingPublisher()
    node._tf_buffer.lookup_transform = lambda *args, **kwargs: identity_transform()
    # Deferred-processing (T-FIX: AB3DMOT Exact-Stamp TF Deferred
    # Processing v1) is on by default and additionally checks
    # `can_transform` (non-blocking) before ever calling `lookup_transform`.
    # These pre-existing tests exercise the surrounding tracker logic, not
    # tf2's own buffer state, so `can_transform` is stubbed "always ready"
    # consistently with the `lookup_transform` stub above -- every message
    # in this file is still processed synchronously/immediately.
    node._tf_buffer.can_transform = lambda *args, **kwargs: (True, "")
    return node


class Ab3dmotTrackerNodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def test_ab3dmot_submodule_imports_from_installed_ros2_environment(self):
        """Not just a source-tree pytest run: this test file itself only
        runs inside a sourced ROS2 environment (it imports
        autoware_perception_msgs/rclpy), so a successful import here is
        exactly the "installed/runtime environment" check this task
        requires -- see also the identical check already exercised at
        module scope by test_ab3dmot_core.py's `KF_CLASS = ...`."""
        kf_class = _load_ab3dmot_kf_class()
        self.assertEqual(kf_class.__name__, "KF")

    def test_disabled_by_default_ignores_messages(self):
        node = Ab3dmotTrackerNode()
        node.publisher = RecordingPublisher()
        node._on_detected_objects(make_detected_objects(1, 0, [(0, 0, 0, 0, 4, 2, 1.5, 1, 0.9)]))
        self.assertEqual(node.publisher.published, [])
        node.destroy_node()

    def test_first_frame_publishes_odom_frame_with_input_stamp(self):
        node = build_enabled_node()
        msg = make_detected_objects(100, 0, [(0, 0, 0, 0, 4, 2, 1.5, 1, 0.9)])
        node._on_detected_objects(msg)
        self.assertEqual(len(node.publisher.published), 1)
        output = node.publisher.published[0]
        self.assertEqual(output.header.frame_id, "odom")
        self.assertEqual((output.header.stamp.sec, output.header.stamp.nanosec), (100, 0))
        node.destroy_node()

    def test_real_dt_propagates_to_mps_velocity(self):
        node = build_enabled_node()
        node._on_detected_objects(make_detected_objects(100, 0, [(0, 0, 0, 0, 4, 2, 1.5, 1, 0.9)]))
        for step in range(1, 11):
            node._on_detected_objects(
                make_detected_objects(100 + step, 0, [(step * 2.0, 0, 0, 0, 4, 2, 1.5, 1, 0.9)])
            )
        last = node.publisher.published[-1]
        self.assertEqual(len(last.objects), 1)
        vx = last.objects[0].kinematics.twist_with_covariance.twist.linear.x
        self.assertAlmostEqual(vx, 2.0, delta=0.3)  # true speed: 2 m/s, dt = 1s
        node.destroy_node()

    def test_stable_track_identity_across_frames(self):
        node = build_enabled_node()
        node._on_detected_objects(make_detected_objects(100, 0, [(0, 0, 0, 0, 4, 2, 1.5, 1, 0.9)]))
        first_uuid = node.publisher.published[-1].objects[0].object_id.uuid.copy()
        node._on_detected_objects(make_detected_objects(101, 0, [(0.1, 0, 0, 0, 4, 2, 1.5, 1, 0.9)]))
        second_uuid = node.publisher.published[-1].objects[0].object_id.uuid.copy()
        self.assertTrue((first_uuid == second_uuid).all())
        node.destroy_node()

    def test_duplicate_timestamp_is_skipped_not_republished(self):
        node = build_enabled_node()
        node._on_detected_objects(make_detected_objects(100, 0, [(0, 0, 0, 0, 4, 2, 1.5, 1, 0.9)]))
        published_before = len(node.publisher.published)
        node._on_detected_objects(make_detected_objects(100, 0, [(0.5, 0, 0, 0, 4, 2, 1.5, 1, 0.9)]))
        self.assertEqual(len(node.publisher.published), published_before)
        node.destroy_node()

    def test_clock_rollback_is_rejected_without_reset_or_publication(self):
        node = build_enabled_node()
        node._on_detected_objects(make_detected_objects(200, 0, [(0, 0, 0, 0, 4, 2, 1.5, 1, 0.9)]))
        first_tracker = node._tracker
        node._on_detected_objects(make_detected_objects(50, 0, [(0, 0, 0, 0, 4, 2, 1.5, 1, 0.9)]))
        self.assertIs(node._tracker, first_tracker)
        self.assertEqual(len(node.publisher.published), 1)
        node.destroy_node()

    def test_empty_detections_does_not_require_tf(self):
        node = build_enabled_node()

        def fail(*args, **kwargs):
            raise AssertionError("TF should not be looked up for an empty message")

        node._tf_buffer.lookup_transform = fail
        node._on_detected_objects(make_detected_objects(100, 0, []))
        self.assertEqual(len(node.publisher.published), 1)
        self.assertEqual(node.publisher.published[0].objects, [])
        node.destroy_node()

    def test_yaw_unobserved_publishes_orientation_unavailable(self):
        node = build_enabled_node(yaw_measurement_mode="unobserved")
        node._on_detected_objects(
            make_detected_objects(
                100, 0, [(0, 0, 0, 0, 4, 2, 1.5, 1, 0.9)]
            )
        )
        tracked = node.publisher.published[0].objects[0]
        self.assertEqual(
            tracked.kinematics.orientation_availability,
            tracked.kinematics.UNAVAILABLE,
        )
        node.destroy_node()

    def test_tf_failure_skips_frame_without_crashing(self):
        node = build_enabled_node()

        def raise_lookup(*args, **kwargs):
            raise LookupException("no transform available")

        node._tf_buffer.lookup_transform = raise_lookup
        node._on_detected_objects(make_detected_objects(100, 0, [(0, 0, 0, 0, 4, 2, 1.5, 1, 0.9)]))
        self.assertEqual(node.publisher.published, [])
        node.destroy_node()

    def test_malformed_object_skips_frame_without_crashing(self):
        node = build_enabled_node()
        msg = make_detected_objects(100, 0, [(0, 0, 0, 0, 4, 2, 1.5, 1, 0.9)])
        msg.objects[0].shape.dimensions.x = -1.0  # invalid
        node._on_detected_objects(msg)
        self.assertEqual(node.publisher.published, [])
        node.destroy_node()

    def test_negative_maximum_position_std_is_rejected(self):
        with self.assertRaises(ValueError):
            Ab3dmotTrackerNode(
                parameter_overrides=[
                    Parameter("enabled", Parameter.Type.BOOL, True),
                    Parameter("maximum_position_std_m", Parameter.Type.DOUBLE, -1.0),
                ]
            )

    def _pos_var(self, tracked_object):
        return tracked_object.kinematics.pose_with_covariance.covariance[0]

    def test_born_then_lost_track_position_covariance_is_bounded_when_enabled(self):
        node = build_enabled_node(
            association_metric="euclidean",
            euclidean_gate_m=3.0,
            matcher="hungarian",
            yaw_measurement_mode="unobserved",
            maximum_position_std_m=7.0,
        )
        node._on_detected_objects(
            make_detected_objects(100, 0, [(0, 0, 0, 0, 4, 2, 1.5, 1, 0.9)])
        )
        # next frame: a different, far detection -- the first track is
        # unmatched and coasts one step (tsu=1) before max_age=2 deletes it.
        node._on_detected_objects(
            make_detected_objects(100, int(2e8), [(50, 50, 0, 0, 4, 2, 1.5, 1, 0.9)])
        )
        coasted = next(
            obj
            for obj in node.publisher.published[-1].objects
            if abs(obj.kinematics.pose_with_covariance.pose.position.x) < 5.0
        )
        self.assertLessEqual(self._pos_var(coasted), 49.0 + 1e-6)
        self.assertGreaterEqual(node._position_covariance_bounded, 1)
        self.assertGreaterEqual(node._frames_with_position_covariance_bounded, 1)
        node.destroy_node()

    def test_born_then_lost_track_covariance_is_unbounded_by_default(self):
        node = build_enabled_node(
            association_metric="euclidean",
            euclidean_gate_m=3.0,
            matcher="hungarian",
            yaw_measurement_mode="unobserved",
        )
        node._on_detected_objects(
            make_detected_objects(100, 0, [(0, 0, 0, 0, 4, 2, 1.5, 1, 0.9)])
        )
        node._on_detected_objects(
            make_detected_objects(100, int(2e8), [(50, 50, 0, 0, 4, 2, 1.5, 1, 0.9)])
        )
        coasted = next(
            obj
            for obj in node.publisher.published[-1].objects
            if abs(obj.kinematics.pose_with_covariance.pose.position.x) < 5.0
        )
        self.assertGreater(self._pos_var(coasted), 100.0)
        self.assertEqual(node._position_covariance_bounded, 0)
        node.destroy_node()

    def test_bounding_one_track_does_not_change_a_healthy_neighbour(self):
        node = build_enabled_node(
            association_metric="euclidean",
            euclidean_gate_m=3.0,
            matcher="hungarian",
            yaw_measurement_mode="unobserved",
            maximum_position_std_m=7.0,
        )
        # track A gets two hits (healthy); track B is seen once then lost.
        node._on_detected_objects(
            make_detected_objects(
                100, 0, [(0, 0, 0, 0, 4, 2, 1.5, 1, 0.9), (20, 0, 0, 0, 4, 2, 1.5, 1, 0.9)]
            )
        )
        node._on_detected_objects(
            make_detected_objects(100, int(2e8), [(0.1, 0, 0, 0, 4, 2, 1.5, 1, 0.9)])
        )
        healthy = next(
            obj
            for obj in node.publisher.published[-1].objects
            if abs(obj.kinematics.pose_with_covariance.pose.position.x) < 5.0
        )
        # A matched detection: its covariance is small and untouched.
        self.assertLess(self._pos_var(healthy), 10.0)
        node.destroy_node()


class _ToggleCanTransform:
    """Stateful `Buffer.can_transform` fake: stamps in `ready` succeed;
    every other stamp reports "extrapolation into the future" until added
    to `ready` (mirroring the real tf2 message text this repo's code
    classifies on)."""

    def __init__(self):
        self.ready: set[tuple[int, int]] = set()
        self.calls: list[tuple[int, int]] = []

    def __call__(self, target_frame, source_frame, stamp, return_debug_tuple=False):
        key = (stamp.sec, stamp.nanosec)
        self.calls.append(key)
        if key in self.ready:
            return True, ""
        return False, "Lookup would require extrapolation into the future."


class DeferredTfProcessingTest(unittest.TestCase):
    """T-FIX: AB3DMOT Exact-Stamp TF Deferred Processing v1 -- node-level
    integration (the pure queue mechanics are covered exhaustively by
    test_ab3dmot_tf_deferred_queue.py; these lock the node wiring: which
    stamp reaches the tracker/publisher, in what order, and when)."""

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def _build(self, **parameters):
        node = build_enabled_node(**parameters)
        toggle = _ToggleCanTransform()
        node._tf_buffer.can_transform = toggle
        return node, toggle

    # Phase 19: immediate TF -> processed once immediately, pending == 0.
    def test_immediate_tf_processes_synchronously_with_empty_queue(self):
        node, toggle = self._build()
        toggle.ready.add((100, 0))
        node._on_detected_objects(make_detected_objects(100, 0, [(0, 0, 0, 0, 4, 2, 1.5, 1, 0.9)]))
        self.assertEqual(len(node.publisher.published), 1)
        self.assertEqual(len(node._deferred_queue), 0)
        self.assertEqual(node._deferred_queue.stats.processed_immediately, 1)
        node.destroy_node()

    # Phase 20: delayed TF -> not processed/dropped while pending; released
    # exactly once, at its ORIGINAL stamp, once TF becomes available.
    def test_delayed_tf_is_held_then_released_with_original_stamp(self):
        node, toggle = self._build()
        # D1's transform is not ready yet.
        node._on_detected_objects(make_detected_objects(100, 0, [(0, 0, 0, 0, 4, 2, 1.5, 1, 0.9)]))
        self.assertEqual(node.publisher.published, [])
        self.assertEqual(len(node._deferred_queue), 1)

        # A later detection arrives while D1 is still not ready: it must be
        # enqueued behind D1, not processed ahead of it.
        node._on_detected_objects(make_detected_objects(100, int(1e8), [(0, 0, 0, 0, 4, 2, 1.5, 1, 0.9)]))
        self.assertEqual(node.publisher.published, [])
        self.assertEqual(len(node._deferred_queue), 2)

        # D1's transform becomes available; the next callback (any new
        # detection, here D3, whose own TF is also already ready) triggers
        # a drain that releases D1 and D2 in order, then processes D3
        # itself.
        toggle.ready.add((100, 0))
        toggle.ready.add((100, int(1e8)))
        toggle.ready.add((100, int(2e8)))
        node._on_detected_objects(make_detected_objects(100, int(2e8), [(0, 0, 0, 0, 4, 2, 1.5, 1, 0.9)]))

        self.assertEqual(len(node.publisher.published), 3)
        stamps = [
            (m.header.stamp.sec, m.header.stamp.nanosec) for m in node.publisher.published
        ]
        self.assertEqual(stamps, [(100, 0), (100, int(1e8)), (100, int(2e8))])
        self.assertEqual(len(node._deferred_queue), 0)
        self.assertEqual(node._deferred_queue.stats.processed_deferred, 2)
        node.destroy_node()

    # Phase 21 (node level): a ready later detection must never overtake an
    # earlier still-pending one.
    def test_later_ready_detection_waits_behind_earlier_pending_one(self):
        node, toggle = self._build()
        toggle.ready.add((100, int(1e8)))  # D2's own TF is ready immediately.
        node._on_detected_objects(make_detected_objects(100, 0, [(0, 0, 0, 0, 4, 2, 1.5, 1, 0.9)]))
        node._on_detected_objects(
            make_detected_objects(100, int(1e8), [(0, 0, 0, 0, 4, 2, 1.5, 1, 0.9)])
        )
        # D2's own TF was ready, but D1 was still pending ahead of it in the
        # queue -- neither has published yet.
        self.assertEqual(node.publisher.published, [])
        self.assertEqual(len(node._deferred_queue), 2)
        node.destroy_node()

    # Phase 22: timeout -> dropped once, no tracker update.
    def test_never_ready_detection_is_dropped_after_max_wait(self):
        node, toggle = self._build(max_tf_wait_ms=100)
        clock = {"t": 0.0}
        node._deferred_queue._monotonic = lambda: clock["t"]
        node._on_detected_objects(make_detected_objects(100, 0, [(0, 0, 0, 0, 4, 2, 1.5, 1, 0.9)]))
        self.assertEqual(len(node._deferred_queue), 1)

        clock["t"] = 0.2  # 200ms > 100ms max_tf_wait_ms
        # A subsequent detection's callback triggers the drain that expires D1.
        node._on_detected_objects(make_detected_objects(100, int(3e8), [(0, 0, 0, 0, 4, 2, 1.5, 1, 0.9)]))
        # D3 itself is still not ready (never added to `toggle.ready`), so
        # only D1's timeout-drop happened; nothing published yet.
        self.assertEqual(node.publisher.published, [])
        self.assertEqual(node._deferred_queue.stats.dropped_tf_timeout, 1)
        self.assertEqual(len(node._deferred_queue), 1)  # D3 itself now pending
        node.destroy_node()

    # Phase 23: a permanent (non-future) TF failure drops immediately,
    # never sits in the queue.
    def test_permanent_tf_failure_drops_without_queueing(self):
        node = build_enabled_node()
        node._tf_buffer.can_transform = lambda *a, **k: (
            False,
            'Invalid frame ID "lidar_link" passed to canTransform argument source_frame - frame does not exist',
        )
        node._on_detected_objects(make_detected_objects(100, 0, [(0, 0, 0, 0, 4, 2, 1.5, 1, 0.9)]))
        self.assertEqual(node.publisher.published, [])
        self.assertEqual(len(node._deferred_queue), 0)
        self.assertEqual(node._deferred_queue.stats.dropped_permanent_tf_failure, 1)
        node.destroy_node()

    # Phase 24: bounded queue, deterministic oldest-drop overflow.
    def test_queue_overflow_drops_oldest_deterministically(self):
        node, toggle = self._build(max_pending_detections=2)
        for i in range(4):
            node._on_detected_objects(
                make_detected_objects(100, int(i * 1e8), [(0, 0, 0, 0, 4, 2, 1.5, 1, 0.9)])
            )
        self.assertEqual(len(node._deferred_queue), 2)
        self.assertEqual(node._deferred_queue.stats.dropped_queue_overflow, 2)
        node.destroy_node()

    # Phase 25: no duplicate tracker update for one admitted detection.
    def test_deferred_detection_is_processed_at_most_once(self):
        node, toggle = self._build()
        node._on_detected_objects(make_detected_objects(100, 0, [(0, 0, 0, 0, 4, 2, 1.5, 1, 0.9)]))
        toggle.ready.add((100, 0))
        # Two callbacks in a row after readiness: the second must find
        # nothing left to release.
        node._on_detected_objects(make_detected_objects(100, int(1e8), [(0, 0, 0, 0, 4, 2, 1.5, 1, 0.9)]))
        published_after_first_drain = len(node.publisher.published)
        node._drain_deferred_queue()
        self.assertEqual(len(node.publisher.published), published_after_first_drain)
        self.assertEqual(node._deferred_queue.stats.processed_deferred, 1)
        node.destroy_node()

    # Phase 27: out-of-order input relative to an in-flight deferral is
    # still rejected by the pre-existing rollback policy -- no new
    # mechanism, and it must not silently jump the queue.
    def test_rollback_while_a_detection_is_deferred_is_still_rejected(self):
        node, toggle = self._build()
        node._on_detected_objects(make_detected_objects(100, 0, [(0, 0, 0, 0, 4, 2, 1.5, 1, 0.9)]))
        self.assertEqual(len(node._deferred_queue), 1)
        node._on_detected_objects(make_detected_objects(50, 0, [(0, 0, 0, 0, 4, 2, 1.5, 1, 0.9)]))
        # Rejected outright: still only the one (still-pending) entry in
        # the queue, no rollback frame admitted anywhere.
        self.assertEqual(len(node._deferred_queue), 1)
        self.assertEqual(node.publisher.published, [])
        node.destroy_node()

    def test_duplicate_of_a_still_pending_stamp_is_rejected(self):
        node, toggle = self._build()
        node._on_detected_objects(make_detected_objects(100, 0, [(0, 0, 0, 0, 4, 2, 1.5, 1, 0.9)]))
        self.assertEqual(len(node._deferred_queue), 1)
        # Same exact stamp arrives again while the first is still pending.
        node._on_detected_objects(make_detected_objects(100, 0, [(1, 1, 0, 0, 4, 2, 1.5, 1, 0.9)]))
        self.assertEqual(len(node._deferred_queue), 1)  # not admitted twice
        node.destroy_node()

    def test_defer_disabled_matches_original_immediate_drop_behavior(self):
        node = build_enabled_node(defer_until_tf_ready=False)
        self.assertIsNone(node._deferred_queue)

        def raise_lookup(*args, **kwargs):
            raise LookupException("no transform available")

        node._tf_buffer.lookup_transform = raise_lookup
        node._on_detected_objects(make_detected_objects(100, 0, [(0, 0, 0, 0, 4, 2, 1.5, 1, 0.9)]))
        self.assertEqual(node.publisher.published, [])
        node.destroy_node()

    def test_max_tf_wait_ms_must_be_positive(self):
        with self.assertRaises(ValueError):
            Ab3dmotTrackerNode(
                parameter_overrides=[
                    Parameter("enabled", Parameter.Type.BOOL, True),
                    Parameter("max_tf_wait_ms", Parameter.Type.INTEGER, 0),
                ]
            )

    def test_max_pending_detections_must_be_positive(self):
        with self.assertRaises(ValueError):
            Ab3dmotTrackerNode(
                parameter_overrides=[
                    Parameter("enabled", Parameter.Type.BOOL, True),
                    Parameter("max_pending_detections", Parameter.Type.INTEGER, 0),
                ]
            )


if __name__ == "__main__":
    unittest.main()
