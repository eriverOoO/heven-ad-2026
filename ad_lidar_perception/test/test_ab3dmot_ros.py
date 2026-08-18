import math
import unittest
from types import SimpleNamespace

import numpy as np

from ad_lidar_perception.ab3dmot_core import TrackedState
from ad_lidar_perception.ab3dmot_ros import (
    DetectedObjectsAdapterError,
    TimestampDecision,
    classify_timestamp,
    detected_objects_to_detections,
    quaternion_from_yaw,
    select_classification,
    stamp_to_ns,
    track_id_to_uuid,
    tracked_state_to_message,
    tracked_states_to_message,
    transform_pose_z_up,
    yaw_from_quaternion,
)


def classification(label, probability):
    return SimpleNamespace(label=label, probability=probability)


def quaternion(x, y, z, w):
    return SimpleNamespace(x=x, y=y, z=z, w=w)


def vector3(x, y, z):
    return SimpleNamespace(x=x, y=y, z=z)


def detected_object(*, position, yaw, dims, classifications, existence_probability=1.0, shape_type=0):
    qx, qy, qz, qw = quaternion_from_yaw(yaw)
    return SimpleNamespace(
        existence_probability=existence_probability,
        classification=classifications,
        kinematics=SimpleNamespace(
            pose_with_covariance=SimpleNamespace(
                pose=SimpleNamespace(
                    position=vector3(*position),
                    orientation=quaternion(qx, qy, qz, qw),
                )
            )
        ),
        shape=SimpleNamespace(type=shape_type, dimensions=vector3(*dims)),
    )


def detected_objects(objects, frame_id="lidar_link"):
    return SimpleNamespace(header=SimpleNamespace(frame_id=frame_id), objects=objects)


def identity_transform():
    return SimpleNamespace(
        transform=SimpleNamespace(
            translation=vector3(0.0, 0.0, 0.0),
            rotation=quaternion(0.0, 0.0, 0.0, 1.0),
        )
    )


def transform_with(*, translation, yaw):
    qx, qy, qz, qw = quaternion_from_yaw(yaw)
    return SimpleNamespace(
        transform=SimpleNamespace(
            translation=vector3(*translation),
            rotation=quaternion(qx, qy, qz, qw),
        )
    )


class TrackedObjectClassification:
    def __init__(self):
        self.label = 0
        self.probability = 0.0


class TrackedObjectShape:
    BOUNDING_BOX = 0

    def __init__(self):
        self.type = -1
        self.dimensions = SimpleNamespace(x=0.0, y=0.0, z=0.0)


class TrackedObjectKinematicsFake:
    AVAILABLE = 2

    def __init__(self):
        self.pose_with_covariance = SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0),
            ),
            covariance=[0.0] * 36,
        )
        self.twist_with_covariance = SimpleNamespace(
            twist=SimpleNamespace(linear=SimpleNamespace(x=0.0, y=0.0, z=0.0)),
            covariance=[0.0] * 36,
        )
        self.orientation_availability = 0
        self.acceleration_with_covariance = SimpleNamespace(
            accel=SimpleNamespace(linear=SimpleNamespace(x=0.0, y=0.0, z=0.0)),
            covariance=[0.0] * 36,
        )
        self.is_stationary = False


class TrackedObjectFake:
    def __init__(self):
        self.object_id = SimpleNamespace(uuid=None)
        self.existence_probability = 0.0
        self.classification = []
        self.kinematics = TrackedObjectKinematicsFake()
        self.shape = TrackedObjectShape()


class TrackedObjectsFake:
    def __init__(self):
        self.header = SimpleNamespace(stamp=None, frame_id="")
        self.objects = []


MESSAGE_TYPES = {
    "TrackedObject": TrackedObjectFake,
    "TrackedObjectKinematics": TrackedObjectKinematicsFake,
    "TrackedObjects": TrackedObjectsFake,
    "ObjectClassification": TrackedObjectClassification,
    "Shape": TrackedObjectShape,
}


def make_state(**overrides):
    defaults = dict(
        track_id=7,
        x=1.0,
        y=2.0,
        z=0.5,
        yaw=0.3,
        length=4.0,
        width=2.0,
        height=1.5,
        vx_mps=3.0,
        vy_mps=-1.0,
        vz_mps=0.0,
        position_covariance=np.eye(3) * 0.5,
        yaw_variance=0.02,
        velocity_covariance=np.eye(3) * 1.5,
        hits=3,
        time_since_update=0,
        label=1,
        label_probability=0.9,
        existence_probability=0.8,
    )
    defaults.update(overrides)
    return TrackedState(**defaults)


class StampTest(unittest.TestCase):
    def test_valid_stamp(self):
        self.assertEqual(stamp_to_ns(SimpleNamespace(sec=1, nanosec=500)), 1_000_000_500)

    def test_rejects_negative_sec(self):
        with self.assertRaises(DetectedObjectsAdapterError):
            stamp_to_ns(SimpleNamespace(sec=-1, nanosec=0))

    def test_rejects_overflowed_nanosec(self):
        with self.assertRaises(DetectedObjectsAdapterError):
            stamp_to_ns(SimpleNamespace(sec=0, nanosec=1_000_000_000))

    def test_rejects_zero_stamp(self):
        with self.assertRaises(DetectedObjectsAdapterError):
            stamp_to_ns(SimpleNamespace(sec=0, nanosec=0))


class ClassifyTimestampTest(unittest.TestCase):
    def test_first_frame_is_process(self):
        self.assertEqual(classify_timestamp(100, None), TimestampDecision.PROCESS)

    def test_strictly_increasing_is_process(self):
        self.assertEqual(classify_timestamp(200, 100), TimestampDecision.PROCESS)

    def test_duplicate_is_skip(self):
        self.assertEqual(classify_timestamp(100, 100), TimestampDecision.SKIP_DUPLICATE)

    def test_decreasing_is_rollback(self):
        self.assertEqual(classify_timestamp(50, 100), TimestampDecision.RESET_ROLLBACK)


class SelectClassificationTest(unittest.TestCase):
    def test_picks_highest_probability(self):
        label, prob = select_classification([classification(0, 0.2), classification(1, 0.9)])
        self.assertEqual((label, prob), (1, 0.9))

    def test_ties_prefer_lower_label(self):
        label, _ = select_classification([classification(3, 0.5), classification(1, 0.5)])
        self.assertEqual(label, 1)

    def test_rejects_empty(self):
        with self.assertRaises(DetectedObjectsAdapterError):
            select_classification([])

    def test_rejects_invalid_probability(self):
        with self.assertRaises(DetectedObjectsAdapterError):
            select_classification([classification(0, 1.5)])


class TransformTest(unittest.TestCase):
    def test_identity_transform_leaves_pose_unchanged(self):
        x, y, z, yaw = transform_pose_z_up(1.0, 2.0, 0.5, 0.3, identity_transform())
        self.assertAlmostEqual(x, 1.0, places=9)
        self.assertAlmostEqual(y, 2.0, places=9)
        self.assertAlmostEqual(z, 0.5, places=9)
        self.assertAlmostEqual(yaw, 0.3, places=9)

    def test_translation_only(self):
        t = transform_with(translation=(10.0, -5.0, 1.0), yaw=0.0)
        x, y, z, yaw = transform_pose_z_up(1.0, 1.0, 0.0, 0.0, t)
        self.assertAlmostEqual(x, 11.0, places=9)
        self.assertAlmostEqual(y, -4.0, places=9)
        self.assertAlmostEqual(z, 1.0, places=9)
        self.assertAlmostEqual(yaw, 0.0, places=9)

    def test_90_degree_yaw_rotation(self):
        t = transform_with(translation=(0.0, 0.0, 0.0), yaw=math.pi / 2.0)
        x, y, z, yaw = transform_pose_z_up(1.0, 0.0, 0.0, 0.0, t)
        self.assertAlmostEqual(x, 0.0, places=9)
        self.assertAlmostEqual(y, 1.0, places=9)
        self.assertAlmostEqual(yaw, math.pi / 2.0, places=9)


class DetectedObjectsToDetectionsTest(unittest.TestCase):
    def test_preserves_class_position_dimensions_yaw(self):
        msg = detected_objects(
            [
                detected_object(
                    position=(1.0, 2.0, 0.3),
                    yaw=0.4,
                    dims=(4.0, 2.0, 1.5),
                    classifications=[classification(1, 0.9)],
                    existence_probability=0.75,
                )
            ]
        )
        detections = detected_objects_to_detections(msg, identity_transform())
        self.assertEqual(len(detections), 1)
        d = detections[0]
        self.assertAlmostEqual(d.x, 1.0)
        self.assertAlmostEqual(d.y, 2.0)
        self.assertAlmostEqual(d.z, 0.3)
        self.assertAlmostEqual(d.yaw, 0.4)
        self.assertEqual((d.length, d.width, d.height), (4.0, 2.0, 1.5))
        self.assertEqual(d.label, 1)
        self.assertAlmostEqual(d.label_probability, 0.9)
        self.assertAlmostEqual(d.existence_probability, 0.75)

    def test_empty_objects_allows_none_transform(self):
        self.assertEqual(detected_objects_to_detections(detected_objects([]), None), [])

    def test_nonempty_requires_a_transform(self):
        msg = detected_objects(
            [detected_object(position=(0, 0, 0), yaw=0, dims=(1, 1, 1), classifications=[classification(0, 1.0)])]
        )
        with self.assertRaises(DetectedObjectsAdapterError):
            detected_objects_to_detections(msg, None)

    def test_rejects_non_bounding_box_shape(self):
        msg = detected_objects(
            [
                detected_object(
                    position=(0, 0, 0), yaw=0, dims=(1, 1, 1),
                    classifications=[classification(0, 1.0)], shape_type=1,
                )
            ]
        )
        with self.assertRaises(DetectedObjectsAdapterError):
            detected_objects_to_detections(msg, identity_transform())

    def test_rejects_nonpositive_dimensions(self):
        msg = detected_objects(
            [detected_object(position=(0, 0, 0), yaw=0, dims=(0.0, 1, 1), classifications=[classification(0, 1.0)])]
        )
        with self.assertRaises(DetectedObjectsAdapterError):
            detected_objects_to_detections(msg, identity_transform())

    def test_rejects_empty_classification(self):
        msg = detected_objects(
            [detected_object(position=(0, 0, 0), yaw=0, dims=(1, 1, 1), classifications=[])]
        )
        with self.assertRaises(DetectedObjectsAdapterError):
            detected_objects_to_detections(msg, identity_transform())


class TrackIdUuidTest(unittest.TestCase):
    def test_deterministic(self):
        self.assertTrue(np.array_equal(track_id_to_uuid(5), track_id_to_uuid(5)))

    def test_distinct_ids_distinct_uuids(self):
        self.assertFalse(np.array_equal(track_id_to_uuid(5), track_id_to_uuid(6)))

    def test_rejects_negative(self):
        with self.assertRaises(ValueError):
            track_id_to_uuid(-1)


class TrackedStateToMessageTest(unittest.TestCase):
    def test_velocity_is_copied_unchanged_in_mps(self):
        message = tracked_state_to_message(make_state(vx_mps=3.5, vy_mps=-2.0, vz_mps=0.1), MESSAGE_TYPES)
        twist = message.kinematics.twist_with_covariance.twist.linear
        self.assertAlmostEqual(twist.x, 3.5)
        self.assertAlmostEqual(twist.y, -2.0)
        self.assertAlmostEqual(twist.z, 0.1)

    def test_position_orientation_dimensions_and_classification(self):
        message = tracked_state_to_message(make_state(), MESSAGE_TYPES)
        pose = message.kinematics.pose_with_covariance.pose
        self.assertAlmostEqual(pose.position.x, 1.0)
        self.assertAlmostEqual(pose.position.y, 2.0)
        self.assertAlmostEqual(pose.position.z, 0.5)
        self.assertAlmostEqual(yaw_from_quaternion(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w), 0.3)
        self.assertEqual(
            (message.shape.dimensions.x, message.shape.dimensions.y, message.shape.dimensions.z), (4.0, 2.0, 1.5)
        )
        self.assertEqual(message.classification[0].label, 1)
        self.assertAlmostEqual(message.classification[0].probability, 0.9)
        self.assertAlmostEqual(message.existence_probability, 0.8)

    def test_stable_track_identity_via_uuid(self):
        first = tracked_state_to_message(make_state(track_id=42), MESSAGE_TYPES)
        second = tracked_state_to_message(make_state(track_id=42), MESSAGE_TYPES)
        third = tracked_state_to_message(make_state(track_id=43), MESSAGE_TYPES)
        self.assertTrue(np.array_equal(first.object_id.uuid, second.object_id.uuid))
        self.assertFalse(np.array_equal(first.object_id.uuid, third.object_id.uuid))

    def test_covariance_blocks_come_from_kf_p_only(self):
        state = make_state(
            position_covariance=np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]),
            yaw_variance=0.07,
            velocity_covariance=np.array([[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]]),
        )
        message = tracked_state_to_message(state, MESSAGE_TYPES)
        pose_cov = message.kinematics.pose_with_covariance.covariance
        self.assertAlmostEqual(pose_cov[0 * 6 + 0], 1.0)
        self.assertAlmostEqual(pose_cov[1 * 6 + 1], 2.0)
        self.assertAlmostEqual(pose_cov[2 * 6 + 2], 3.0)
        self.assertAlmostEqual(pose_cov[3 * 6 + 3], 0.07)
        twist_cov = message.kinematics.twist_with_covariance.covariance
        self.assertAlmostEqual(twist_cov[0 * 6 + 0], 4.0)
        self.assertAlmostEqual(twist_cov[1 * 6 + 1], 5.0)
        self.assertAlmostEqual(twist_cov[2 * 6 + 2], 6.0)

    def test_untracked_fields_left_at_message_default(self):
        """acceleration_with_covariance / is_stationary: AB3DMOT's state has
        no such quantity, so these must stay at the message's own default
        rather than being invented."""
        message = tracked_state_to_message(make_state(), MESSAGE_TYPES)
        self.assertEqual(message.kinematics.is_stationary, False)
        accel = message.kinematics.acceleration_with_covariance
        self.assertEqual(accel.accel.linear.x, 0.0)
        self.assertEqual(accel.covariance, [0.0] * 36)


class TrackedStatesToMessageTest(unittest.TestCase):
    def test_sets_odom_frame_and_preserves_input_stamp(self):
        stamp = SimpleNamespace(sec=12, nanosec=345)
        message = tracked_states_to_message([make_state()], stamp, "odom", MESSAGE_TYPES)
        self.assertEqual(message.header.frame_id, "odom")
        self.assertIs(message.header.stamp, stamp)
        self.assertEqual(len(message.objects), 1)

    def test_empty_states_produce_empty_objects(self):
        message = tracked_states_to_message([], SimpleNamespace(sec=1, nanosec=0), "odom", MESSAGE_TYPES)
        self.assertEqual(message.objects, [])


if __name__ == "__main__":
    unittest.main()
