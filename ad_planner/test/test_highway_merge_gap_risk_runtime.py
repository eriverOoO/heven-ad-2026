"""Deterministic canonical replay: synthetic DynamicObjectRisk -> live
ad_highway_merge_gap_risk against the real checksum-verified route corridor and
the shipped source-grounded K-City highway on-ramp merge geometry
(route:0:left:1 tapering into primary route route:0).

No MORAI highway-merge bag with provenance-verified circulating-actor
trajectories exists in the repo (the 'kcity-highway' preset is a pinned link
set with no recorded actor state), so the object trajectories here are
synthetic but the merge zone, the target corridor, the ego route, and the node
are real. The ego is swept along the highway approach; mainline-ahead,
near-ahead, closing-from-behind, and merging-in objects are exercised, and
per-frame counts / distributions / latency are recorded.
"""

import hashlib
import json
import math
import statistics
from pathlib import Path
import time
import unittest

from ad_interfaces.msg import (
    DynamicObjectRisk,
    DynamicObjectRiskArray,
    DynamicObjectRiskState,
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


DOMAIN_ID = 96
REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "ad_data"
GLOBAL_PATH = DATA_DIR / "path" / "2026_molit_comp_global_path.txt"
CONFIG = REPO / "ad_planner" / "config" / "highway_merge_gap_risk.yaml"

RISK_TOPIC = "/ad/planning/dynamic_object_risks"
ODOM_TOPIC = "/ad/localization/odometry"
OUTPUT_TOPIC = "/ad/planning/highway_merge_gap_risks"

# Primary-route centreline samples on the highway approach (station -> pose),
# taken verbatim from ad_data/map/route_corridor.json (route:0).
ROUTE_GRID = {
    400.0: (-59.95, -71.531, 1.5798),
    1000.0: (4.158, 345.621, -0.0452),
    1020.0: (23.529, 341.183, -0.424),
    1040.0: (39.818, 329.764, -0.7918),
    1060.0: (51.272, 313.457, -1.0961),
    1080.0: (58.094, 294.714, -1.3366),
    1100.0: (61.339, 275.004, -1.4817),
    1120.0: (62.097, 254.995, -1.5556),
    1140.0: (62.33, 234.997, -1.5602),
    1160.0: (62.434, 214.998, -1.57),
    1180.0: (62.533, 194.998, -1.5666),
    1200.0: (62.657, 174.998, -1.5541),
    1220.0: (63.223, 155.008, -1.5227),
    1240.0: (64.36, 135.04, -1.5175),
    1260.0: (65.514, 115.074, -1.5085),
    1280.0: (66.639, 95.106, -1.5292),
    1300.0: (66.999, 75.198, -1.5676),
    1320.0: (67.125, 55.199, -1.567),
    1340.0: (67.227, 35.199, -1.5675),
    1360.0: (67.335, 15.2, -1.5648),
}
S_ZONE_ENTRY = 1118.7417511691015
S_MERGE_COMPLETE = 1286.1546454384027
EGO_SPEED = 8.0
EGO_STATIONS = [1040.0, 1075.0, 1105.0, 1130.0, 1160.0]
HORIZON_S = 24.0
DT_S = 1.0

_QOS_1 = QoSProfile(
    depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE
)
_QOS_10 = QoSProfile(
    depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE
)


@pytest.mark.launch_test
def generate_test_description():
    digest = hashlib.sha256(GLOBAL_PATH.read_bytes()).hexdigest()
    return LaunchDescription(
        [
            SetEnvironmentVariable("ROS_DOMAIN_ID", str(DOMAIN_ID)),
            SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
            SetEnvironmentVariable(
                "ROS_LOG_DIR", "/tmp/heven_highway_merge_gap_risk_ros_log"
            ),
            LaunchNode(
                package="ad_planner",
                executable="ad_highway_merge_gap_risk_node",
                name="ad_highway_merge_gap_risk",
                output="screen",
                parameters=[
                    str(CONFIG),
                    {
                        "data_dir": str(DATA_DIR),
                        "route_corridor_file": "map/route_corridor.json",
                        "route_corridor.expected_global_path_sha256": digest,
                        "maximum_input_age_s": 5.0,
                        "runtime_summary_interval_frames": 5,
                    },
                ],
            ),
            launch_testing.actions.ReadyToTest(),
            TimerAction(
                period=30.0,
                actions=[Shutdown(reason="highway merge runtime validation complete")],
            ),
        ]
    )


def _interp_pose(station):
    keys = sorted(ROUTE_GRID)
    station = max(keys[0], min(keys[-1], station))
    lo = max(k for k in keys if k <= station)
    hi = min(k for k in keys if k >= station)
    if hi == lo:
        return ROUTE_GRID[lo]
    ratio = (station - lo) / (hi - lo)
    a, b = ROUTE_GRID[lo], ROUTE_GRID[hi]
    yaw = a[2] + ratio * math.atan2(
        math.sin(b[2] - a[2]), math.cos(b[2] - a[2])
    )
    return (
        a[0] + ratio * (b[0] - a[0]),
        a[1] + ratio * (b[1] - a[1]),
        yaw,
    )


def _lane_point(station, lateral_left=0.0):
    x, y, yaw = _interp_pose(station)
    return (x - math.sin(yaw) * lateral_left, y + math.cos(yaw) * lateral_left)


def _duration(seconds):
    message = Duration()
    message.sec = int(seconds)
    message.nanosec = int(round((seconds - int(seconds)) * 1e9))
    return message


def _to_body(px, py, ego_x, ego_y, ego_yaw):
    dx, dy = px - ego_x, py - ego_y
    c, s = math.cos(ego_yaw), math.sin(ego_yaw)
    return c * dx + s * dy, -s * dx + c * dy


def _object(uuid_byte, ego_pose, station0, speed_mps, lateral0=0.0, lateral_taper=False):
    ex, ey, eyaw = ego_pose
    risk = DynamicObjectRisk()
    risk.object_id.uuid = [uuid_byte] + [0] * 15
    risk.classification = 1
    risk.classification_probability = 0.9
    risk.existence_probability = 0.9
    cx, cy = _lane_point(station0, lateral0)
    bx, by = _to_body(cx, cy, ex, ey, eyaw)
    risk.x_rel_m = bx
    risk.y_rel_m = by
    # Relative velocity along the route tangent at station0.
    _, _, yaw0 = _interp_pose(station0)
    vx_map = speed_mps * math.cos(yaw0)
    vy_map = speed_mps * math.sin(yaw0)
    c, s = math.cos(eyaw), math.sin(eyaw)
    risk.vx_rel_mps = (c * vx_map + s * vy_map) - EGO_SPEED
    risk.vy_rel_mps = -s * vx_map + c * vy_map
    t = DT_S
    while t <= HORIZON_S + 1e-6:
        frac = min(1.0, t / HORIZON_S)
        lateral = lateral0 * (1.0 - frac) if lateral_taper else lateral0
        mx, my = _lane_point(station0 + speed_mps * t, lateral)
        rel_x, rel_y = _to_body(mx, my, ex, ey, eyaw)
        state = DynamicObjectRiskState()
        state.time_from_start = _duration(t)
        state.x_rel_m = rel_x - EGO_SPEED * t
        state.y_rel_m = rel_y
        risk.predicted_states.append(state)
        t += DT_S
    return risk


def _pctile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(q * (len(ordered) - 1))))
    return ordered[idx]


class TestHighwayMergeGapRiskRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = Node("highway_merge_gap_risk_probe")
        cls.frames = []
        cls.latency_ms = []
        cls.node.create_subscription(
            HighwayMergeGapRiskArray,
            OUTPUT_TOPIC,
            lambda m: cls.frames.append(m),
            _QOS_1,
        )
        cls.risk_pub = cls.node.create_publisher(
            DynamicObjectRiskArray, RISK_TOPIC, _QOS_1
        )
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
        started = time.monotonic()
        self.risk_pub.publish(array)
        self.assertTrue(
            self._spin_until(lambda: len(self.frames) >= want, 3.0),
            "highway merge risk pipeline did not publish",
        )
        self.latency_ms.append((time.monotonic() - started) * 1e3)
        return self.frames[-1]

    def test_highway_merge_gap_facts(self):
        self.assertTrue(
            self._spin_until(
                lambda: self.node.count_publishers(OUTPUT_TOPIC) >= 1
                and self.node.count_subscribers(RISK_TOPIC) == 1,
                10.0,
            ),
            "highway merge node not discovered",
        )
        first_pose = _interp_pose(EGO_STATIONS[0])
        for _ in range(5):
            self.odom_pub.publish(
                self._odom(self.node.get_clock().now().to_msg(), first_pose)
            )
            self._spin(0.1)

        frame_count = 0
        input_objects = 0
        relevant_object_frames = 0
        relevant_uuids = set()
        ego_timing_valid_frames = 0
        ahead_frames = 0
        behind_frames = 0
        merging_predicted_enter_frames = 0
        covered_frames = 0
        uncovered_frames = 0
        merge_gap_valid_frames = 0
        merge_gap_values = []
        delta_s_at_merge_values = []
        representative = None

        for station in EGO_STATIONS:
            pose = _interp_pose(station)
            objects = [
                _object(1, pose, station + 70.0, 8.0),   # mainline, well ahead
                _object(2, pose, station + 16.0, 8.0),    # mainline, near ahead
                _object(3, pose, station - 28.0, 11.0),    # closing from behind
                _object(4, pose, station + 45.0, 8.0, lateral0=3.5, lateral_taper=True),
                _object(5, pose, station + 18.0, 3.0),    # slow -> ego overtakes
            ]
            frame = self._publish(pose, objects)
            frame_count += 1
            input_objects += len(objects)
            self.assertEqual(frame.header.frame_id, "map")
            self.assertEqual(frame.merge_zone_id, "kcity_highway_onramp")
            self.assertEqual(frame.target_lane_sequence_id, "route:0")
            self.assertEqual(frame.source_lane_sequence_id, "route:0:left:1")
            self.assertAlmostEqual(
                frame.merge_reference_route_s_m, S_MERGE_COMPLETE, delta=1.0
            )
            for name in (
                "ego_route_s_m", "ego_merge_time_s", "ego_route_distance_to_merge_m",
                "merge_gap_m", "nearest_leading_delta_s_at_merge_m",
                "nearest_trailing_delta_s_at_merge_m",
            ):
                self.assertTrue(math.isfinite(getattr(frame, name)))
            self.assertAlmostEqual(frame.ego_route_s_m, station, delta=3.0)
            if frame.ego_merge_timing_valid:
                ego_timing_valid_frames += 1
            if frame.merge_gap_valid:
                merge_gap_valid_frames += 1
                merge_gap_values.append(frame.merge_gap_m)

            by_uuid = {o.object_id.uuid[0]: o for o in frame.objects}
            for obj in frame.objects:
                for name in (
                    "object_route_s_m", "delta_s_now_m", "delta_s_at_merge_m",
                    "object_longitudinal_speed_mps", "relative_longitudinal_speed_mps",
                    "longitudinal_closing_speed_mps", "time_to_route_coincidence_s",
                    "predicted_min_route_gap_m", "prediction_horizon_s",
                ):
                    self.assertTrue(math.isfinite(getattr(obj, name)))
                self.assertGreaterEqual(obj.predicted_min_route_gap_m, 0.0)
                if obj.relevant_to_merge:
                    relevant_object_frames += 1
                    relevant_uuids.add(obj.object_id.uuid[0])
                if obj.is_ahead_at_merge:
                    ahead_frames += 1
                if obj.is_behind_at_merge:
                    behind_frames += 1
                if obj.delta_s_at_merge_valid:
                    delta_s_at_merge_values.append(obj.delta_s_at_merge_m)
                if obj.prediction_covers_merge_time:
                    covered_frames += 1
                elif obj.relevant_to_merge:
                    uncovered_frames += 1

            merging = by_uuid[4]
            if merging.predicted_to_enter_target_corridor:
                merging_predicted_enter_frames += 1
            self.assertFalse(merging.object_in_target_corridor_now)

            if abs(station - 1130.0) < 1e-6:
                representative = frame

        self.assertEqual(frame_count, len(EGO_STATIONS))
        self.assertEqual(ego_timing_valid_frames, len(EGO_STATIONS))
        self.assertGreaterEqual(len(relevant_uuids), 3)
        self.assertGreater(ahead_frames, 0)
        self.assertGreater(behind_frames, 0)
        self.assertGreater(merging_predicted_enter_frames, 0)
        # Both prediction-coverage branches are exercised across the sweep.
        self.assertGreater(covered_frames, 0)
        self.assertGreater(uncovered_frames, 0)
        self.assertGreater(merge_gap_valid_frames, 0)

        # Ego far from the merge zone (route:0 s~400): the frame still publishes
        # as an inactive result -- it is not dropped.
        far_pose = _interp_pose(400.0)
        far_frame = self._publish(far_pose, [_object(1, far_pose, 415.0, 8.0)])
        self.assertEqual(far_frame.header.frame_id, "map")
        self.assertFalse(far_frame.ego_merge_timing_valid)
        self.assertEqual(far_frame.relevant_object_count, 0)
        self.assertTrue(math.isfinite(far_frame.ego_route_distance_to_merge_m))

        self.assertIsNotNone(representative)
        rep = {o.object_id.uuid[0]: o for o in representative.objects}
        self.assertTrue(rep[1].relevant_to_merge)
        self.assertGreater(rep[1].delta_s_now_m, 0.0)
        self.assertLess(rep[3].delta_s_now_m, 0.0)

        result = {
            "input_risk_messages": frame_count,
            "gap_messages": len(self.frames),
            "input_objects": input_objects,
            "relevant_object_frames": relevant_object_frames,
            "unique_relevant_uuids": len(relevant_uuids),
            "ego_timing_valid_frames": ego_timing_valid_frames,
            "ahead_object_frames": ahead_frames,
            "behind_object_frames": behind_frames,
            "merging_predicted_enter_frames": merging_predicted_enter_frames,
            "prediction_covers_merge_time_object_frames": covered_frames,
            "prediction_uncovered_relevant_object_frames": uncovered_frames,
            "merge_gap_valid_frames": merge_gap_valid_frames,
            "merge_gap_m": {
                "median": round(statistics.median(merge_gap_values), 3),
                "min": round(min(merge_gap_values), 3),
                "max": round(max(merge_gap_values), 3),
            },
            "delta_s_at_merge_m": {
                "median": round(statistics.median(delta_s_at_merge_values), 3),
                "min": round(min(delta_s_at_merge_values), 3),
                "max": round(max(delta_s_at_merge_values), 3),
            },
            "publish_to_receive_latency_ms": {
                "median": round(statistics.median(self.latency_ms), 3),
                "p95": round(_pctile(self.latency_ms, 0.95), 3),
                "max": round(max(self.latency_ms), 3),
                "samples": len(self.latency_ms),
            },
            "representative_1130m": {
                "ego_merge_time_s": round(representative.ego_merge_time_s, 3),
                "merge_gap_valid": representative.merge_gap_valid,
                "merge_gap_m": round(representative.merge_gap_m, 3),
                "ahead_delta_s_at_merge_m": round(rep[1].delta_s_at_merge_m, 3),
                "behind_delta_s_now_m": round(rep[3].delta_s_now_m, 3),
                "merging_predicted_enter": rep[4].predicted_to_enter_target_corridor,
            },
        }
        Path("/tmp/heven_highway_merge_gap_risk_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print("HIGHWAY_MERGE_GAP_RISK_RESULT=" + json.dumps(result, sort_keys=True))


@launch_testing.post_shutdown_test()
class TestHighwayMergeGapRiskRuntimeShutdown(unittest.TestCase):
    def test_clean_shutdown(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info)
