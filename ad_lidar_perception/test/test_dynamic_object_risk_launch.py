"""Runtime contract for the Dynamic Object Risk Interface node.

Feeds a synthetic canonical PredictedObjectArray plus canonical odometry and
checks that /ad/planning/dynamic_object_risks carries physically sensible,
finite, base_link risk metrics with exactly one publisher.
"""

import math
from pathlib import Path
import time
import unittest

from ad_interfaces.msg import (
    DynamicObjectRiskArray,
    PredictedObject,
    PredictedObjectArray,
    PredictedState,
)
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
import launch_testing
import launch_testing.actions
import launch_testing.asserts
from launch_ros.actions import Node as LaunchNode
from nav_msgs.msg import Odometry
import pytest
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)


DOMAIN_ID = 94
PREDICTION_TOPIC = "/ad/perception/objects/predicted"
ODOMETRY_TOPIC = "/ad/localization/odometry"
RISK_TOPIC = "/ad/planning/dynamic_object_risks"

_VOLATILE_RELIABLE = QoSProfile(
    depth=1,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)
_ODOM_QOS = QoSProfile(
    depth=10,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


@pytest.mark.launch_test
def generate_test_description():
    config = str(
        Path(get_package_share_directory("ad_lidar_perception"))
        / "config"
        / "planning"
        / "dynamic_object_risk.yaml"
    )
    return LaunchDescription(
        [
            SetEnvironmentVariable("ROS_DOMAIN_ID", str(DOMAIN_ID)),
            SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
            LaunchNode(
                package="ad_lidar_perception",
                executable="ad_dynamic_object_risk_node",
                name="ad_dynamic_object_risk",
                output="screen",
                parameters=[config, {"runtime_summary_interval_frames": 0}],
            ),
            launch_testing.actions.ReadyToTest(),
        ]
    )


def _stamp(node, seconds_offset=0.0):
    now = node.get_clock().now().to_msg()
    total_ns = now.sec * 1_000_000_000 + now.nanosec + int(seconds_offset * 1e9)
    now.sec = total_ns // 1_000_000_000
    now.nanosec = total_ns % 1_000_000_000
    return now


def _prediction(node, *, lead_x, lead_vx):
    message = PredictedObjectArray()
    message.header.stamp = _stamp(node)
    message.header.frame_id = "odom"
    obj = PredictedObject()
    obj.object_id.uuid = [7] + [0] * 15
    obj.classification = PredictedObject.CAR
    obj.initial_pose.pose.position.x = float(lead_x)
    obj.initial_pose.pose.position.y = 0.0
    obj.initial_pose.pose.orientation.w = 1.0
    obj.dimensions.x = 4.5
    obj.dimensions.y = 1.9
    obj.dimensions.z = 1.5
    obj.initial_twist.twist.linear.x = float(lead_vx)
    for k in range(1, 5):
        state = PredictedState()
        state.time_from_start.sec = k
        state.pose.pose.position.x = float(lead_x + lead_vx * k)
        state.pose.pose.position.y = 0.0
        obj.states.append(state)
    message.objects.append(obj)
    return message


def _odometry(node, *, speed):
    message = Odometry()
    message.header.stamp = _stamp(node)
    message.header.frame_id = "odom"
    message.child_frame_id = "base_link"
    message.pose.pose.orientation.w = 1.0
    message.twist.twist.linear.x = float(speed)
    return message


class TestDynamicObjectRisk(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = Node("dynamic_object_risk_probe")
        cls.received = []
        cls.node.create_subscription(
            DynamicObjectRiskArray, RISK_TOPIC,
            lambda m: cls.received.append(m), _VOLATILE_RELIABLE)
        cls.prediction_pub = cls.node.create_publisher(
            PredictedObjectArray, PREDICTION_TOPIC, _VOLATILE_RELIABLE)
        cls.odometry_pub = cls.node.create_publisher(
            Odometry, ODOMETRY_TOPIC, _ODOM_QOS)

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def _spin(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def test_publishes_finite_base_link_risk_metrics(self):
        # Let discovery settle.
        self._spin(2.0)
        self.assertEqual(
            self.node.count_publishers(RISK_TOPIC), 1,
            "exactly one publisher expected on the risk topic")

        for _ in range(25):
            self.odometry_pub.publish(_odometry(self.node, speed=10.0))
            self._spin(0.05)
            self.prediction_pub.publish(
                _prediction(self.node, lead_x=20.0, lead_vx=0.0))
            self._spin(0.15)

        self.assertTrue(self.received, "no DynamicObjectRiskArray received")
        latest = self.received[-1]
        self.assertEqual(latest.header.frame_id, "base_link")
        self.assertEqual(len(latest.objects), 1)
        risk = latest.objects[0]

        for name in (
            "x_rel_m", "y_rel_m", "distance_m", "vx_rel_mps", "vy_rel_mps",
            "relative_speed_mps", "range_rate_mps", "longitudinal_closing_mps",
            "ttc_s", "cpa_time_s", "cpa_distance_m",
            "predicted_min_separation_m", "position_uncertainty_m",
        ):
            value = getattr(risk, name)
            self.assertTrue(math.isfinite(value), f"{name} is not finite: {value}")

        # Stationary lead 20 m ahead, ego at 10 m/s: closing, TTC valid < 2 s.
        self.assertAlmostEqual(risk.x_rel_m, 20.0, delta=0.5)
        self.assertLess(risk.vx_rel_mps, 0.0)
        self.assertGreater(risk.range_rate_mps, 0.0)
        self.assertTrue(risk.ttc_valid)
        self.assertGreater(risk.ttc_s, 0.0)
        self.assertLess(risk.ttc_s, 2.0)
        self.assertTrue(risk.cpa_valid)
        self.assertTrue(risk.predicted_min_separation_valid)

    def test_node_stays_single_publisher_and_alive(self):
        # Stale/missing-ego rejection is covered exhaustively by the gtest
        # build_risk_frame cases; here just confirm the node did not crash or
        # spawn a duplicate publisher over the run.
        self._spin(0.5)
        self.assertEqual(self.node.count_publishers(RISK_TOPIC), 1)


@launch_testing.post_shutdown_test()
class TestDynamicObjectRiskShutdown(unittest.TestCase):
    def test_clean_shutdown(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info)
