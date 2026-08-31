"""Deterministic pipeline replay for the Highway Merge Ego-on-Ramp Scenario v1.

synthetic DynamicObjectRiskArray + odometry
  -> ad_highway_merge_gap_risk    (real, real corridor + merge geometry)
  -> ad_highway_merge_gap_response (real)

with the ego swept along the REAL `route:0:left:1` acceleration-lane samples
(from ad_data/map/route_corridor.json). CASE B (PR #24) left this chain -- which
previously only ever saw the ego on `route:0` -- unexercised with the ego
physically on the ramp.

Asserts:
  * ego_route_s_m is monotonic across the ramp sweep and matches the source-lane
    station (the gap-risk node projects the ego onto route:0);
  * ego_merge_timing_valid holds for a ramp ego;
  * a response is produced for every accepted frame;
  * traffic cases: no traffic -> MERGE_READY; clear front/rear gap -> MERGE_READY;
    fast-closing rear -> WAIT/HOLD; prediction-uncovered relevant object ->
    not MERGE_READY.

Not a MORAI run (no simulator / grpc here): deterministic ROS replay.
"""

import hashlib
import json
import math
from pathlib import Path
import time
import unittest

from ad_interfaces.msg import (
    DynamicObjectRisk,
    DynamicObjectRiskArray,
    DynamicObjectRiskState,
    HighwayMergeGapResponse,
    HighwayMergeGapRiskArray,
)
from builtin_interfaces.msg import Duration
from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable, Shutdown, TimerAction
import launch_testing
import launch_testing.actions
import launch_testing.asserts
from launch_ros.actions import Node as LaunchNode
from nav_msgs.msg import Odometry
import pytest
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

DOMAIN_ID = 95
REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "ad_data"
GLOBAL_PATH = DATA_DIR / "path" / "2026_molit_comp_global_path.txt"
ROUTE_CORRIDOR = DATA_DIR / "map" / "route_corridor.json"
RISK_CONFIG = REPO / "ad_planner" / "config" / "highway_merge_gap_risk.yaml"
RESPONSE_CONFIG = REPO / "ad_planner" / "config" / "highway_merge_gap_response.yaml"

RISK_TOPIC = "/ad/planning/dynamic_object_risks"
ODOM_TOPIC = "/ad/localization/odometry"
GAP_TOPIC = "/ad/planning/highway_merge_gap_risks"
RESPONSE_TOPIC = "/ad/planning/highway_merge_gap_response"

S_ZONE_ENTRY = 1118.7417511691015
S_MERGE_COMPLETE = 1286.1546454384027
EGO_SPEED = 8.0
HORIZON_S = 24.0
DT_S = 1.0

_QOS_1 = QoSProfile(
    depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE
)
_QOS_10 = QoSProfile(
    depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE
)


def _route0_points():
    doc = json.loads(ROUTE_CORRIDOR.read_text())
    for lane in doc["lanes"]:
        if lane["lane_sequence_id"] == "route:0":
            return lane["points"]
    raise AssertionError("route:0 missing")


def _ramp_points():
    doc = json.loads(ROUTE_CORRIDOR.read_text())
    for lane in doc["lanes"]:
        if lane["lane_sequence_id"] == "route:0:left:1":
            return lane["points"]
    raise AssertionError("route:0:left:1 missing")


_RAMP = _ramp_points()
_R0 = _route0_points()


def _ramp_pose_at(source_route_s):
    """Real route:0:left:1 pose at (or just past) the given source station."""
    for p in _RAMP:
        if p["route_s_m"] >= source_route_s - 1e-9:
            return p["x_m"], p["y_m"], p["yaw_rad"]
    p = _RAMP[-1]
    return p["x_m"], p["y_m"], p["yaw_rad"]


def _route0_pose_at(route_s):
    for p in _R0:
        if p["route_s_m"] >= route_s:
            return p["x_m"], p["y_m"], p["yaw_rad"]
    p = _R0[-1]
    return p["x_m"], p["y_m"], p["yaw_rad"]


# Ego swept along the real acceleration lane, before and inside the merge zone.
EGO_RAMP_STATIONS = [1120.0, 1140.0, 1170.0, 1210.0, 1250.0]


def _duration(seconds):
    message = Duration()
    message.sec = int(seconds)
    message.nanosec = int(round((seconds - int(seconds)) * 1e9))
    return message


def _to_body(px, py, ex, ey, eyaw):
    dx, dy = px - ex, py - ey
    c, s = math.cos(eyaw), math.sin(eyaw)
    return c * dx + s * dy, -s * dx + c * dy


def _mainline_object(uuid_byte, ego_pose, route_s0, speed_mps, horizon_s=HORIZON_S):
    """A vehicle travelling down route:0, expressed relative to the ramp ego."""
    ex, ey, eyaw = ego_pose
    risk = DynamicObjectRisk()
    risk.object_id.uuid = [uuid_byte] + [0] * 15
    risk.classification = 1
    risk.classification_probability = 0.9
    risk.existence_probability = 0.9
    cx, cy, cyaw = _route0_pose_at(route_s0)
    bx, by = _to_body(cx, cy, ex, ey, eyaw)
    risk.x_rel_m = bx
    risk.y_rel_m = by
    vx_map = speed_mps * math.cos(cyaw)
    vy_map = speed_mps * math.sin(cyaw)
    c, s = math.cos(eyaw), math.sin(eyaw)
    risk.vx_rel_mps = (c * vx_map + s * vy_map) - EGO_SPEED
    risk.vy_rel_mps = -s * vx_map + c * vy_map
    t = DT_S
    while t <= horizon_s + 1e-6:
        px, py, _ = _route0_pose_at(route_s0 + speed_mps * t)
        rx, ry = _to_body(px, py, ex, ey, eyaw)
        state = DynamicObjectRiskState()
        state.time_from_start = _duration(t)
        state.x_rel_m = rx - EGO_SPEED * t
        state.y_rel_m = ry
        risk.predicted_states.append(state)
        t += DT_S
    return risk


class TestHighwayMergeEgoRampPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = Node("highway_merge_ego_ramp_pipeline_probe")
        cls.frames = []
        cls.responses = []
        cls.node.create_subscription(
            HighwayMergeGapRiskArray, GAP_TOPIC, lambda m: cls.frames.append(m), _QOS_1
        )
        cls.node.create_subscription(
            HighwayMergeGapResponse, RESPONSE_TOPIC, lambda m: cls.responses.append(m), _QOS_1
        )
        cls.risk_pub = cls.node.create_publisher(DynamicObjectRiskArray, RISK_TOPIC, _QOS_1)
        cls.odom_pub = cls.node.create_publisher(Odometry, ODOM_TOPIC, _QOS_10)

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

    def _odom(self, stamp, pose):
        message = Odometry()
        message.header.stamp = stamp
        message.header.frame_id = "map"
        message.child_frame_id = "base_link"
        message.pose.pose.position.x = pose[0]
        message.pose.pose.position.y = pose[1]
        message.pose.pose.orientation.z = math.sin(pose[2] / 2.0)
        message.pose.pose.orientation.w = math.cos(pose[2] / 2.0)
        message.twist.twist.linear.x = EGO_SPEED
        return message

    def _publish(self, pose, objects):
        stamp = self.node.get_clock().now().to_msg()
        self.odom_pub.publish(self._odom(stamp, pose))
        self._spin(0.1)
        array = DynamicObjectRiskArray()
        array.header.stamp = stamp
        array.header.frame_id = "base_link"
        array.objects = objects
        want = len(self.frames) + 1
        self.risk_pub.publish(array)
        self.assertTrue(
            self._spin_until(lambda: len(self.frames) >= want, 3.0),
            "gap-risk pipeline did not publish",
        )
        want_response = len(self.responses) + 1
        self._spin_until(lambda: len(self.responses) >= want_response, 2.0)
        return self.frames[-1], self.responses[-1] if self.responses else None

    def test_ramp_ego_pipeline(self):
        self.assertTrue(
            self._spin_until(
                lambda: self.node.count_publishers(GAP_TOPIC) >= 1
                and self.node.count_publishers(RESPONSE_TOPIC) >= 1,
                12.0,
            ),
            "pipeline nodes not discovered",
        )
        first = _ramp_pose_at(EGO_RAMP_STATIONS[0])
        for _ in range(5):
            self.odom_pub.publish(self._odom(self.node.get_clock().now().to_msg(), first))
            self._spin(0.1)

        results = {"ego_route_s_m": [], "actions": {}}
        previous_s = None

        # --- ego swept along the real acceleration lane, clear traffic --------
        for station in EGO_RAMP_STATIONS:
            pose = _ramp_pose_at(station)
            objects = [
                _mainline_object(1, pose, station + 80.0, 26.0),   # front, well ahead
                _mainline_object(2, pose, station - 90.0, 24.0),   # rear, well behind
            ]
            frame, response = self._publish(pose, objects)
            self.assertEqual(frame.header.frame_id, "map")
            self.assertEqual(frame.source_lane_sequence_id, "route:0:left:1")
            self.assertEqual(frame.target_lane_sequence_id, "route:0")
            self.assertTrue(math.isfinite(frame.ego_route_s_m))
            # the gap-risk node projects the ramp ego onto route:0; the station
            # matches the source-lane route_s closely and never goes backwards
            self.assertAlmostEqual(frame.ego_route_s_m, station, delta=3.0)
            if previous_s is not None:
                self.assertGreaterEqual(
                    frame.ego_route_s_m, previous_s - 1e-3,
                    "ego route station regressed for a ramp ego",
                )
            previous_s = frame.ego_route_s_m
            self.assertTrue(frame.ego_merge_timing_valid, "ramp ego lost merge timing")
            results["ego_route_s_m"].append(round(frame.ego_route_s_m, 2))
            self.assertIsNotNone(response)

        clear_action = self.responses[-1]
        results["actions"]["clear_gap"] = _action_name(clear_action)

        pose = _ramp_pose_at(1170.0)

        # --- CASE D: no traffic --------------------------------------------
        _, no_traffic = self._publish(pose, [])
        results["actions"]["no_traffic"] = _action_name(no_traffic)
        self.assertTrue(no_traffic.active)
        self.assertEqual(no_traffic.action, HighwayMergeGapResponse.ACTION_MERGE_READY)

        # --- CASE B: fast-closing rear -----------------------------------------
        _, fast_rear = self._publish(
            pose,
            [
                _mainline_object(1, pose, 1170.0 + 70.0, 26.0),
                _mainline_object(3, pose, 1170.0 - 12.0, 33.0),  # right behind, closing hard
            ],
        )
        results["actions"]["fast_rear"] = _action_name(fast_rear)
        self.assertNotEqual(
            fast_rear.action, HighwayMergeGapResponse.ACTION_MERGE_READY,
            "a hard-closing rear vehicle must not yield MERGE_READY",
        )

        # --- CASE C: prediction does not cover the merge time -----------------
        _, uncovered = self._publish(
            pose,
            [
                _mainline_object(1, pose, 1170.0 + 60.0, 24.0),
                _mainline_object(4, pose, 1170.0 - 25.0, 20.0, horizon_s=2.0),
            ],
        )
        results["actions"]["insufficient_prediction"] = _action_name(uncovered)
        self.assertNotEqual(
            uncovered.action, HighwayMergeGapResponse.ACTION_MERGE_READY,
            "an object without merge-time prediction coverage must not yield MERGE_READY",
        )

        self.assertGreaterEqual(len(self.responses), len(EGO_RAMP_STATIONS) + 3)
        for message in self.responses:
            for name in (
                "available_distance_m", "comfortable_stop_distance_m",
                "front_gap_m", "rear_gap_m", "ego_merge_time_s",
            ):
                self.assertTrue(math.isfinite(getattr(message, name)))

        Path("/tmp/heven_highway_merge_ego_ramp_pipeline_result.json").write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def _action_name(message):
    if message is None:
        return "NONE"
    if not message.active:
        return "INACTIVE"
    return {
        HighwayMergeGapResponse.ACTION_MERGE_READY: "MERGE_READY",
        HighwayMergeGapResponse.ACTION_WAIT: "WAIT",
        HighwayMergeGapResponse.ACTION_HOLD: "HOLD",
    }.get(message.action, "UNKNOWN")


@pytest.mark.launch_test
def generate_test_description():
    digest = hashlib.sha256(GLOBAL_PATH.read_bytes()).hexdigest()
    return LaunchDescription(
        [
            SetEnvironmentVariable("ROS_DOMAIN_ID", str(DOMAIN_ID)),
            SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
            SetEnvironmentVariable(
                "ROS_LOG_DIR", "/tmp/heven_highway_merge_ego_ramp_pipeline_ros_log"
            ),
            LaunchNode(
                package="ad_planner",
                executable="ad_highway_merge_gap_risk_node",
                name="ad_highway_merge_gap_risk",
                output="screen",
                parameters=[
                    str(RISK_CONFIG),
                    {
                        "data_dir": str(DATA_DIR),
                        "route_corridor_file": "map/route_corridor.json",
                        "route_corridor.expected_global_path_sha256": digest,
                        "maximum_input_age_s": 5.0,
                        "maximum_odometry_skew_s": 5.0,
                    },
                ],
            ),
            LaunchNode(
                package="ad_planner",
                executable="ad_highway_merge_gap_response_node",
                name="ad_highway_merge_gap_response",
                output="screen",
                parameters=[str(RESPONSE_CONFIG), {"maximum_input_age_s": 5.0}],
            ),
            launch_testing.actions.ReadyToTest(),
            TimerAction(
                period=90.0,
                actions=[Shutdown(reason="ego-on-ramp pipeline validation complete")],
            ),
        ]
    )


@launch_testing.post_shutdown_test()
class TestHighwayMergeEgoRampPipelineShutdown(unittest.TestCase):
    def test_clean_shutdown(self, proc_info):
        launch_testing.asserts.assertExitCodes(
            proc_info, allowable_exit_codes=[0, -15, 130]
        )
