"""Live ad_cut_in_response policy replay driven by synthetic CutInRiskArray.

This isolates the longitudinal policy (no dynamic-object-risk / cut-in-risk
node), so the three response states, left/right symmetry, multi-object
arbitration, and non-candidate handling are all exercised deterministically in
the real ROS node.
"""

import json
import math
from pathlib import Path
import time
import unittest

from ad_interfaces.msg import CutInResponse, CutInRisk, CutInRiskArray
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
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


DOMAIN_ID = 98
RISK_TOPIC = "/ad/planning/cut_in_risks"
ODOMETRY_TOPIC = "/ad/localization/odometry"
RESPONSE_TOPIC = "/ad/planning/cut_in_response"

EGO_SPEED_MPS = 8.0

_QOS_1 = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)
_QOS_10 = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


@pytest.mark.launch_test
def generate_test_description():
    return LaunchDescription(
        [
            SetEnvironmentVariable("ROS_DOMAIN_ID", str(DOMAIN_ID)),
            SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
            SetEnvironmentVariable(
                "ROS_LOG_DIR", "/tmp/heven_cut_in_response_policy_ros_log"
            ),
            LaunchNode(
                package="ad_planner",
                executable="ad_cut_in_response_node",
                name="ad_cut_in_response",
                output="screen",
                parameters=[
                    {
                        "maximum_input_age_s": 1.0,
                        "maximum_odometry_skew_s": 1.0,
                        "runtime_summary_interval_frames": 0,
                    }
                ],
            ),
            launch_testing.actions.ReadyToTest(),
        ]
    )


def _stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def _odometry(stamp, speed_mps):
    message = Odometry()
    message.header.stamp = stamp
    message.header.frame_id = "map"
    message.child_frame_id = "base_link"
    message.pose.pose.orientation.w = 1.0
    message.twist.twist.linear.x = speed_mps
    return message


def _candidate(first_byte, route_s_rel_m, side):
    risk = CutInRisk()
    risk.object_id.uuid = [first_byte] + [0] * 15
    risk.side = side
    risk.route_s_rel_m = float(route_s_rel_m)
    risk.lateral_velocity_toward_corridor_mps = 1.0
    risk.approaching_corridor = True
    risk.adjacent_region = True
    risk.predicted_entry_valid = True
    risk.predicted_entry_time_s = 1.5
    risk.predicted_entry_route_s_rel_m = float(route_s_rel_m)
    risk.predicted_entry_sustained = True
    risk.cut_in_candidate = True
    return risk


def _risk_array(stamp, risks):
    array = CutInRiskArray()
    array.header.stamp = stamp
    array.header.frame_id = "map"
    array.risks = risks
    return array


class TestCutInResponsePolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = Node("cut_in_response_policy_probe")
        cls.responses = []
        cls.node.create_subscription(
            CutInResponse,
            RESPONSE_TOPIC,
            lambda message: cls.responses.append(message),
            _QOS_1,
        )
        cls.risk_publisher = cls.node.create_publisher(
            CutInRiskArray, RISK_TOPIC, _QOS_1
        )
        cls.odometry_publisher = cls.node.create_publisher(
            Odometry, ODOMETRY_TOPIC, _QOS_10
        )

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def _spin(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.02)

    def _spin_until(self, predicate, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.02)
            if predicate():
                return True
        return False

    def _publish(self, risks, speed_mps=EGO_SPEED_MPS):
        stamp = self.node.get_clock().now().to_msg()
        self.odometry_publisher.publish(_odometry(stamp, speed_mps))
        self._spin(0.05)
        self.risk_publisher.publish(_risk_array(stamp, risks))
        expected = len(self.responses) + 1
        self.assertTrue(
            self._spin_until(lambda: len(self.responses) >= expected, 2.0),
            "response node did not publish",
        )
        return self.responses[-1]

    def test_policy(self):
        self.assertTrue(
            self._spin_until(
                lambda: self.node.count_publishers(RESPONSE_TOPIC) == 1
                and self.node.count_subscribers(RISK_TOPIC) == 1
                and self.node.count_subscribers(ODOMETRY_TOPIC) == 1,
                8.0,
            ),
            "response node not discovered",
        )
        for _ in range(5):
            stamp = self.node.get_clock().now().to_msg()
            self.odometry_publisher.publish(_odometry(stamp, EGO_SPEED_MPS))
            self._spin(0.05)

        # 1. Sweep one right-side candidate from far to near.
        actions = []
        ranks = []
        for station in (45.0, 34.0, 24.0, 18.0, 14.0, 10.0, 7.0):
            message = self._publish(
                [_candidate(1, station, CutInRisk.SIDE_RIGHT)]
            )
            actions.append(int(message.action))
            ranks.append(int(message.action))
            self.assertEqual(message.header.frame_id, "map")
            for name in (
                "requested_max_speed_mps",
                "required_deceleration_mps2",
                "route_s_rel_m",
            ):
                self.assertTrue(math.isfinite(getattr(message, name)))
            if message.action == CutInResponse.ACTION_SLOWDOWN:
                self.assertGreater(message.requested_max_speed_mps, 0.0)
                self.assertLess(
                    message.requested_max_speed_mps, EGO_SPEED_MPS + 1e-6
                )
            elif message.action == CutInResponse.ACTION_HOLD:
                self.assertAlmostEqual(message.requested_max_speed_mps, 0.0)
        # Non-decreasing urgency as the station shrinks.
        self.assertEqual(ranks, sorted(ranks))
        self.assertIn(CutInResponse.ACTION_NONE, actions)
        self.assertIn(CutInResponse.ACTION_SLOWDOWN, actions)
        self.assertIn(CutInResponse.ACTION_HOLD, actions)

        # 2. Non-candidate risk -> NONE.
        idle = _candidate(9, 12.0, CutInRisk.SIDE_RIGHT)
        idle.cut_in_candidate = False
        message = self._publish([idle])
        self.assertEqual(message.action, CutInResponse.ACTION_NONE)
        self.assertFalse(message.active)
        self.assertEqual(message.candidate_count, 0)

        # 3. Left/right symmetry: mirrored candidates give the same response.
        left = self._publish([_candidate(2, 18.0, CutInRisk.SIDE_LEFT)])
        right = self._publish([_candidate(3, 18.0, CutInRisk.SIDE_RIGHT)])
        self.assertEqual(left.action, right.action)
        self.assertAlmostEqual(
            left.requested_max_speed_mps, right.requested_max_speed_mps, places=5
        )
        self.assertAlmostEqual(
            left.required_deceleration_mps2,
            right.required_deceleration_mps2,
            places=5,
        )

        # 4. Multi-object arbitration: the most restrictive candidate wins.
        message = self._publish(
            [
                _candidate(4, 40.0, CutInRisk.SIDE_LEFT),
                _candidate(5, 9.0, CutInRisk.SIDE_RIGHT),
            ]
        )
        self.assertEqual(message.action, CutInResponse.ACTION_HOLD)
        self.assertEqual(message.source_object_id.uuid[0], 5)
        self.assertEqual(message.candidate_count, 2)

        # 5. Backward timestamp is not published (count stays put).
        before = len(self.responses)
        old_stamp = self.node.get_clock().now().to_msg()
        old_stamp.sec -= 5
        self.odometry_publisher.publish(_odometry(old_stamp, EGO_SPEED_MPS))
        self._spin(0.05)
        self.risk_publisher.publish(
            _risk_array(old_stamp, [_candidate(6, 10.0, CutInRisk.SIDE_RIGHT)])
        )
        self._spin(0.5)
        self.assertEqual(len(self.responses), before)

        result = {
            "sweep_actions": actions,
            "arbitration_source": 5,
            "left_right_symmetric": True,
        }
        Path("/tmp/heven_cut_in_response_policy_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print("CUT_IN_RESPONSE_POLICY_RESULT=" + json.dumps(result, sort_keys=True))


@launch_testing.post_shutdown_test()
class TestCutInResponsePolicyShutdown(unittest.TestCase):
    def test_clean_shutdown(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info)
