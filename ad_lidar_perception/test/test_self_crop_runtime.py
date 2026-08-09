import math
import struct
import threading
import time
import unittest

from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
import launch_testing
import launch_testing.actions
import launch_testing.asserts
from launch_ros.actions import Node
import pytest
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node as RclpyNode
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField


INPUT_TOPIC = "/test/self_crop/input"
OUTPUT_TOPIC = "/test/self_crop/output"
DOMAIN_ID = "96"


@pytest.mark.launch_test
def generate_test_description():
    crop_process = Node(
        package="ad_lidar_perception",
        executable="ad_self_crop_filter_node",
        name="ad_self_crop_runtime_fixture",
        output="screen",
        parameters=[
            {
                "topics.input": INPUT_TOPIC,
                "topics.output": OUTPUT_TOPIC,
                "base_frame": "base_link",
                "transform_timeout_sec": 0.0,
            }
        ],
    )
    return (
        LaunchDescription(
            [
                SetEnvironmentVariable("ROS_DOMAIN_ID", DOMAIN_ID),
                crop_process,
                launch_testing.actions.ReadyToTest(),
            ]
        ),
        {"crop_process": crop_process},
    )


def _strict_cloud(stamp_ns, frame_id, points):
    message = PointCloud2()
    message.header.stamp.sec = stamp_ns // 1_000_000_000
    message.header.stamp.nanosec = stamp_ns % 1_000_000_000
    message.header.frame_id = frame_id
    message.height = 1
    message.width = len(points)
    message.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(
            name="intensity", offset=12, datatype=PointField.FLOAT32, count=1
        ),
        PointField(name="ring", offset=16, datatype=PointField.UINT16, count=1),
        PointField(name="time", offset=18, datatype=PointField.FLOAT32, count=1),
    ]
    message.is_bigendian = False
    message.point_step = 22
    message.row_step = message.point_step * message.width
    message.data = b"".join(
        struct.pack("<ffffHf", x, y, z, intensity, ring, point_time)
        for x, y, z, intensity, ring, point_time in points
    )
    message.is_dense = True
    return message


def _stamp_ns(message):
    return message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec


class SelfCropRuntimeDriver(RclpyNode):
    def __init__(self, context):
        super().__init__("self_crop_runtime_driver", context=context)
        self.outputs = []
        self.lock = threading.Lock()
        self.publisher = self.create_publisher(
            PointCloud2, INPUT_TOPIC, qos_profile_sensor_data
        )
        self.subscription = self.create_subscription(
            PointCloud2,
            OUTPUT_TOPIC,
            self._on_output,
            qos_profile_sensor_data,
        )

    def _on_output(self, message):
        with self.lock:
            self.outputs.append(message)

    def stamps(self):
        with self.lock:
            return {_stamp_ns(message) for message in self.outputs}

    def output_at(self, stamp_ns):
        with self.lock:
            return next(
                (
                    message
                    for message in self.outputs
                    if _stamp_ns(message) == stamp_ns
                ),
                None,
            )


class TestSelfCropRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = Context()
        rclpy.init(context=cls.context)
        cls.driver = SelfCropRuntimeDriver(cls.context)
        cls.executor = SingleThreadedExecutor(context=cls.context)
        cls.executor.add_node(cls.driver)
        cls.spin_thread = threading.Thread(target=cls.executor.spin, daemon=True)
        cls.spin_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.executor.shutdown(timeout_sec=3.0)
        cls.spin_thread.join(timeout=3.0)
        cls.executor.remove_node(cls.driver)
        cls.driver.destroy_node()
        rclpy.shutdown(context=cls.context)

    def _publish_for(self, message, duration_sec=0.35):
        deadline = time.monotonic() + duration_sec
        while time.monotonic() < deadline:
            self.driver.publisher.publish(message)
            time.sleep(0.03)

    def test_missing_tf_drops_while_invalid_returns_are_compacted_before_valid(
        self,
    ):
        discovery_deadline = time.monotonic() + 5.0
        while time.monotonic() < discovery_deadline:
            if (
                self.driver.count_subscribers(INPUT_TOPIC) > 0
                and self.driver.count_publishers(OUTPUT_TOPIC) > 0
            ):
                break
            time.sleep(0.05)
        self.assertGreater(
            self.driver.count_subscribers(INPUT_TOPIC),
            0,
            "self-crop input subscription was not discovered",
        )
        self.assertGreater(
            self.driver.count_publishers(OUTPUT_TOPIC),
            0,
            "self-crop output publisher was not discovered",
        )
        time.sleep(0.10)

        unresolved_stamp = 101_000_000_001
        unresolved = _strict_cloud(
            unresolved_stamp,
            "unresolved_lidar",
            [(10.0, 0.0, 0.0, 1.0, 1, 0.0)],
        )
        self._publish_for(unresolved)
        self.assertNotIn(unresolved_stamp, self.driver.stamps())

        malformed_stamp = 102_000_000_002
        late_invalid = _strict_cloud(
            malformed_stamp,
            "base_link",
            [
                (10.0, 0.0, 0.0, 1.0, 2, 0.0),
                (math.nan, 0.0, 0.0, 2.0, 3, 0.01),
            ],
        )
        self._publish_for(late_invalid)
        compacted = self.driver.output_at(malformed_stamp)
        self.assertIsNotNone(compacted)
        self.assertEqual(compacted.height, 1)
        self.assertEqual(compacted.width, 1)
        self.assertEqual(compacted.row_step, compacted.point_step)
        self.assertTrue(compacted.is_dense)
        self.assertEqual(
            compacted.data,
            late_invalid.data[: late_invalid.point_step],
        )

        valid_stamp = 103_000_000_003
        valid = _strict_cloud(
            valid_stamp,
            "base_link",
            [(10.0, 0.0, 0.0, 3.0, 4, 0.0)],
        )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            self.driver.publisher.publish(valid)
            if valid_stamp in self.driver.stamps():
                break
            time.sleep(0.03)
        self.assertIn(
            valid_stamp,
            self.driver.stamps(),
            "valid strict base_link cloud did not reach the output publisher",
        )
        self.assertNotIn(unresolved_stamp, self.driver.stamps())
        self.assertIn(malformed_stamp, self.driver.stamps())


@launch_testing.post_shutdown_test()
class TestSelfCropShutdown(unittest.TestCase):
    def test_clean_shutdown(self, proc_info, crop_process):
        launch_testing.asserts.assertExitCodes(
            proc_info, process=crop_process
        )
