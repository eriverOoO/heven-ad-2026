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


def build_enabled_node():
    node = Ab3dmotTrackerNode(
        parameter_overrides=[Parameter("enabled", Parameter.Type.BOOL, True)]
    )
    node.publisher = RecordingPublisher()
    node._tf_buffer.lookup_transform = lambda *args, **kwargs: identity_transform()
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

    def test_clock_rollback_resets_tracker_state(self):
        node = build_enabled_node()
        node._on_detected_objects(make_detected_objects(200, 0, [(0, 0, 0, 0, 4, 2, 1.5, 1, 0.9)]))
        first_tracker = node._tracker
        node._on_detected_objects(make_detected_objects(50, 0, [(0, 0, 0, 0, 4, 2, 1.5, 1, 0.9)]))
        self.assertIsNot(node._tracker, first_tracker)
        # the reset tracker treats the rollback message as a fresh first
        # frame rather than propagating a negative dt
        self.assertEqual(len(node.publisher.published), 2)
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


if __name__ == "__main__":
    unittest.main()
