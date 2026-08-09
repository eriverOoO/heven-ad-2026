import copy
import threading
import time
import unittest

from ad_interfaces.msg import (
    PredictedObject,
    PredictedObjectArray,
    PredictedState,
)
from launch import LaunchDescription
import launch_testing
import launch_testing.actions
import launch_testing.asserts
from launch_ros.actions import Node
import pytest
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node as RclpyNode
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rosgraph_msgs.msg import Clock
from visualization_msgs.msg import Marker, MarkerArray


INPUT_TOPIC = "/ad/perception/objects/predicted"
OUTPUT_TOPIC = "/ad/viz/perception/objects"
TEST_STALE_TIMEOUT_SEC = 2.0


@pytest.mark.launch_test
def generate_test_description():
    marker_process = Node(
        package="ad_viz",
        executable="ad_perception_marker_node",
        name="ad_perception_marker_runtime_fixture",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "marker_lifetime_sec": 0.20,
                "stale_timeout_sec": TEST_STALE_TIMEOUT_SEC,
                "stale_check_period_sec": 0.05,
                "clock_rollback_reset_sec": 2.0,
            }
        ],
    )
    return (
        LaunchDescription(
            [
                marker_process,
                launch_testing.actions.ReadyToTest(),
            ]
        ),
        {"marker_process": marker_process},
    )


def _pose_with_covariance(x, y, z, yaw_z=0.0, yaw_w=1.0):
    result = PredictedObject().initial_pose
    result.pose.position.x = x
    result.pose.position.y = y
    result.pose.position.z = z
    result.pose.orientation.z = yaw_z
    result.pose.orientation.w = yaw_w
    for index in range(6):
        result.covariance[index * 6 + index] = 0.1 + 0.01 * index
    return result


def _state(sec, nanosec, x, y):
    result = PredictedState()
    result.time_from_start.sec = sec
    result.time_from_start.nanosec = nanosec
    result.pose = _pose_with_covariance(x, y, 0.5)
    return result


def _object(seed=1):
    result = PredictedObject()
    result.object_id.uuid = [
        (seed + index) % 256 for index in range(16)
    ]
    result.existence_probability = 0.9
    result.classification = PredictedObject.CAR
    result.classification_probability = 0.8
    result.initial_pose = _pose_with_covariance(1.0, 2.0, 0.5)
    result.initial_twist.twist.linear.x = 2.0
    result.initial_twist.twist.linear.y = -1.0
    for index in range(6):
        result.initial_twist.covariance[index * 6 + index] = (
            0.2 + 0.01 * index
        )
    result.dimensions.x = 4.6
    result.dimensions.y = 1.9
    result.dimensions.z = 1.6
    result.states = [
        _state(0, 500_000_000, 2.0, 1.5),
        _state(1, 0, 3.0, 1.0),
    ]
    result.states.extend(
        _state(
            step // 2,
            0 if step % 2 == 0 else 500_000_000,
            1.0 + step,
            2.0 - 0.5 * step,
        )
        for step in range(3, 13)
    )
    return result


def _array(stamp_sec):
    result = PredictedObjectArray()
    result.header.frame_id = "odom"
    result.header.stamp.sec = stamp_sec
    result.objects = [_object()]
    return result


def _wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class MarkerRuntimeDriver(RclpyNode):
    def __init__(self, context):
        super().__init__("perception_marker_runtime_driver", context=context)
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.messages = []
        self.message_lock = threading.Lock()
        self.publisher = self.create_publisher(
            PredictedObjectArray, INPUT_TOPIC, qos
        )
        clock_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.clock_publisher = self.create_publisher(
            Clock, "/clock", clock_qos
        )
        self.subscription = self.create_subscription(
            MarkerArray, OUTPUT_TOPIC, self._on_markers, qos
        )

    def _on_markers(self, message):
        with self.message_lock:
            self.messages.append(message)

    def snapshot(self):
        with self.message_lock:
            return list(self.messages)

    def publish_clock(self, stamp_sec):
        message = Clock()
        message.clock.sec = stamp_sec
        self.clock_publisher.publish(message)


class TestPerceptionMarkerRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = Context()
        rclpy.init(context=cls.context)
        cls.driver = MarkerRuntimeDriver(cls.context)
        cls.executor = SingleThreadedExecutor(context=cls.context)
        cls.executor.add_node(cls.driver)
        cls.spin_thread = threading.Thread(
            target=cls.executor.spin, daemon=True
        )
        cls.spin_thread.start()
        assert _wait_until(
            lambda: cls.driver.count_subscribers(INPUT_TOPIC) == 1
            and cls.driver.count_publishers(OUTPUT_TOPIC) == 1
            and cls.driver.count_subscribers("/clock") >= 1
        ), "marker node endpoints did not become discoverable"

    @classmethod
    def tearDownClass(cls):
        cls.executor.shutdown(timeout_sec=3.0)
        cls.spin_thread.join(timeout=3.0)
        cls.executor.remove_node(cls.driver)
        cls.driver.destroy_node()
        rclpy.shutdown(context=cls.context)
        assert not cls.spin_thread.is_alive(), "runtime driver did not stop cleanly"

    def test_actual_qos_atomic_retry_and_single_stale_clear(self):
        subscriptions = self.driver.get_subscriptions_info_by_topic(INPUT_TOPIC)
        publishers = self.driver.get_publishers_info_by_topic(OUTPUT_TOPIC)
        self.assertEqual(len(subscriptions), 1)
        self.assertEqual(len(publishers), 1)
        for endpoint in (subscriptions[0], publishers[0]):
            # Humble/Fast DDS reports endpoint history as UNKNOWN even when
            # the rmw profile was constructed with KEEP_LAST. The exact
            # history policy is covered by the C++ QoS contract test.
            self.assertIn(
                endpoint.qos_profile.history,
                (HistoryPolicy.KEEP_LAST, HistoryPolicy.UNKNOWN),
            )
            self.assertIn(
                endpoint.qos_profile.depth,
                (0, 1),
                "Humble/Fast DDS uses depth 0 when endpoint history is unknown",
            )
            self.assertEqual(
                endpoint.qos_profile.reliability, ReliabilityPolicy.RELIABLE
            )
            self.assertEqual(
                endpoint.qos_profile.durability, DurabilityPolicy.VOLATILE
            )

        self.driver.publisher.publish(_array(10))
        self.assertTrue(
            _wait_until(lambda: len(self.driver.snapshot()) >= 1),
            "valid prediction did not produce markers",
        )
        first = self.driver.snapshot()[-1]
        self.assertEqual(len(first.markers), 7)
        self.assertEqual(first.markers[0].action, Marker.DELETEALL)
        self.assertEqual(first.markers[1].type, Marker.LINE_LIST)
        self.assertEqual(first.markers[4].type, Marker.LINE_STRIP)
        self.assertEqual(len(first.markers[4].points), 13)
        self.assertEqual(first.markers[6].type, Marker.TEXT_VIEW_FACING)

        invalid = _array(20)
        invalid.objects.append(copy.deepcopy(_object(seed=50)))
        invalid.objects[1].dimensions.y = -1.0
        before_invalid = len(self.driver.snapshot())
        self.driver.publisher.publish(invalid)
        time.sleep(0.15)
        self.assertEqual(
            len(self.driver.snapshot()),
            before_invalid,
            "malformed multi-object input leaked a partial marker array",
        )

        self.driver.publisher.publish(_array(20))
        self.assertTrue(
            _wait_until(
                lambda: len(self.driver.snapshot()) == before_invalid + 1
            ),
            "same-stamp correction was not accepted after invalid input",
        )
        corrected = self.driver.snapshot()[-1]
        self.assertEqual(len(corrected.markers), 7)
        self.assertEqual(corrected.markers[0].action, Marker.DELETEALL)

        before_duplicate = len(self.driver.snapshot())
        self.driver.publisher.publish(_array(20))
        time.sleep(0.12)
        self.assertEqual(
            len(self.driver.snapshot()),
            before_duplicate,
            "successful same-stamp input was admitted twice",
        )

        self.driver.publisher.publish(_array(19))
        time.sleep(0.12)
        self.assertEqual(
            len(self.driver.snapshot()),
            before_duplicate,
            "small out-of-order input incorrectly reset the watermark",
        )

        self.driver.publisher.publish(_array(10))
        time.sleep(0.12)
        self.assertEqual(
            len(self.driver.snapshot()),
            before_duplicate,
            "old header without a ROS clock jump reset the watermark",
        )

        self.driver.publish_clock(100)
        time.sleep(0.05)
        self.driver.publish_clock(90)
        time.sleep(0.05)
        self.driver.publisher.publish(_array(10))
        self.assertTrue(
            _wait_until(
                lambda: len(self.driver.snapshot()) == before_duplicate + 1
            ),
            "large rosbag clock rollback was not recovered",
        )
        rollback = self.driver.snapshot()[-1]
        self.assertEqual(len(rollback.markers), 7)
        self.assertEqual(rollback.markers[0].action, Marker.DELETEALL)

        self.assertTrue(
            _wait_until(
                lambda: len(self.driver.snapshot()) == before_duplicate + 2,
                timeout=TEST_STALE_TIMEOUT_SEC + 1.0,
            ),
            "stale marker state was not cleared",
        )
        stale_clear = self.driver.snapshot()[-1]
        self.assertEqual(len(stale_clear.markers), 1)
        self.assertEqual(stale_clear.markers[0].action, Marker.DELETEALL)

        after_clear = len(self.driver.snapshot())
        time.sleep(0.30)
        self.assertEqual(
            len(self.driver.snapshot()),
            after_clear,
            "stale DELETEALL was published more than once",
        )


@launch_testing.post_shutdown_test()
class TestPerceptionMarkerShutdown(unittest.TestCase):
    def test_clean_shutdown(self, proc_info, marker_process):
        launch_testing.asserts.assertExitCodes(
            proc_info, process=marker_process
        )
