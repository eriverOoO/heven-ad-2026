"""Deterministic canonical replay: synthetic DynamicObjectRisk -> live
ad_roundabout_gap_risk against the real checksum-verified route corridor and the
shipped K-City roundabout conflict geometry, then the live roundabout response
node.

No MORAI roundabout bag with provenance-verified circulating-actor trajectories
exists in the repo, so the object trajectories here are synthetic but the
conflict region, the ego route, and both nodes are real. The ego is swept along
the roundabout approach over a bounded set of frames; five temporal cases
(object clears first / occupancy overlap / object arrives after ego / nearby
non-conflicting object / leave then re-enter) are exercised. A separate valid
zero-object response frame proves RELEASE, and per-frame counts and
distributions are recorded.
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
    RoundaboutGapRiskArray,
    RoundaboutGapResponse,
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


DOMAIN_ID = 94
REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "ad_data"
GLOBAL_PATH = DATA_DIR / "path" / "2026_molit_comp_global_path.txt"
CONFIG = REPO / "ad_planner" / "config" / "roundabout_gap_risk.yaml"
RESPONSE_CONFIG = REPO / "ad_planner" / "config" / "roundabout_gap_response.yaml"

RISK_TOPIC = "/ad/planning/dynamic_object_risks"
ODOM_TOPIC = "/ad/localization/odometry"
OUTPUT_TOPIC = "/ad/planning/roundabout_gap_risks"
RESPONSE_TOPIC = "/ad/planning/roundabout_gap_response"

# Primary-route centreline samples on the roundabout approach (station -> pose).
# Taken from ad_data/map/route_corridor.json.
ROUTE_SAMPLES = {
    850.0: (-99.4150, 282.8157, 1.5779),
    860.0: (-99.4479, 292.7058, 1.5680),
    870.0: (-99.4142, 302.7051, 1.6004),
    880.0: (-99.4847, 312.7037, 1.5620),
    886.0: (-99.0833, 318.7723, 1.3235),
}
EGO_SPEED = 8.0
S_ENTER, S_EXIT = 890.1006, 914.0039
HORIZON_S = 8.0
DT_S = 0.5

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
                "ROS_LOG_DIR", "/tmp/heven_roundabout_gap_risk_ros_log"
            ),
            LaunchNode(
                package="ad_planner",
                executable="ad_roundabout_gap_risk_node",
                name="ad_roundabout_gap_risk",
                output="screen",
                parameters=[
                    str(CONFIG),
                    {
                        "data_dir": str(DATA_DIR),
                        "route_corridor_file": "map/route_corridor.json",
                        "route_corridor.expected_global_path_sha256": digest,
                        "maximum_input_age_s": 5.0,
                        "runtime_summary_interval_frames": 0,
                    },
                ],
            ),
            LaunchNode(
                package="ad_planner",
                executable="ad_roundabout_gap_response_node",
                name="ad_roundabout_gap_response",
                output="screen",
                parameters=[
                    str(RESPONSE_CONFIG),
                    {
                        "maximum_input_age_s": 5.0,
                        "runtime_summary_interval_frames": 6,
                    },
                ],
            ),
            launch_testing.actions.ReadyToTest(),
            TimerAction(
                period=30.0,
                actions=[Shutdown(reason="roundabout runtime validation complete")],
            ),
        ]
    )


def _duration(seconds):
    message = Duration()
    message.sec = int(seconds)
    message.nanosec = int(round((seconds - int(seconds)) * 1e9))
    return message


def _to_body(px, py, ego_x, ego_y, ego_yaw):
    dx, dy = px - ego_x, py - ego_y
    c, s = math.cos(ego_yaw), math.sin(ego_yaw)
    return c * dx + s * dy, -s * dx + c * dy


def _object(uuid_byte, current_xy, sample_positions, ego_pose):
    ex, ey, eyaw = ego_pose
    risk = DynamicObjectRisk()
    risk.object_id.uuid = [uuid_byte] + [0] * 15
    risk.classification = 1
    risk.classification_probability = 0.9
    risk.existence_probability = 0.9
    bx, by = _to_body(current_xy[0], current_xy[1], ex, ey, eyaw)
    risk.x_rel_m = bx
    risk.y_rel_m = by
    for t, mx, my in sample_positions:
        state = DynamicObjectRiskState()
        state.time_from_start = _duration(t)
        rel_x, rel_y = _to_body(mx, my, ex, ey, eyaw)
        state.x_rel_m = rel_x - EGO_SPEED * t
        state.y_rel_m = rel_y
        risk.predicted_states.append(state)
    return risk


def _window_track(inside_xy, outside_xy, window):
    return _multi_window_track(inside_xy, outside_xy, [window])


def _multi_window_track(inside_xy, outside_xy, windows):
    def _inside(t):
        return any(lo - 1e-6 <= t <= hi + 1e-6 for lo, hi in windows)

    out = []
    t = DT_S
    while t <= HORIZON_S + 1e-6:
        out.append((t, *(inside_xy if _inside(t) else outside_xy)))
        t += DT_S
    return out


def _pctile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(q * (len(ordered) - 1))))
    return ordered[idx]


class TestRoundaboutGapRiskRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = Node("roundabout_gap_risk_probe")
        cls.frames = []
        cls.responses = []
        cls.latency_ms = []
        cls.response_latency_ms = []
        cls.node.create_subscription(
            RoundaboutGapRiskArray, OUTPUT_TOPIC, lambda m: cls.frames.append(m), _QOS_1
        )
        cls.node.create_subscription(
            RoundaboutGapResponse,
            RESPONSE_TOPIC,
            lambda m: cls.responses.append(m),
            _QOS_1,
        )
        cls.risk_pub = cls.node.create_publisher(DynamicObjectRiskArray, RISK_TOPIC, _QOS_1)
        cls.odom_pub = cls.node.create_publisher(Odometry, ODOM_TOPIC, _QOS_10)
        cls.response_input_pub = cls.node.create_publisher(
            RoundaboutGapRiskArray, OUTPUT_TOPIC, _QOS_1
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
        want_response = len(self.responses) + 1
        started = time.monotonic()
        self.risk_pub.publish(array)
        self.assertTrue(
            self._spin_until(
                lambda: len(self.frames) >= want
                and len(self.responses) >= want_response,
                3.0,
            ),
            "roundabout risk/response pipeline did not publish",
        )
        elapsed_ms = (time.monotonic() - started) * 1e3
        self.latency_ms.append(elapsed_ms)
        self.response_latency_ms.append(elapsed_ms)
        return self.frames[-1]

    def test_roundabout_conflict_timing_facts(self):
        self.assertTrue(
            self._spin_until(
                lambda: self.node.count_publishers(OUTPUT_TOPIC) >= 1
                and self.node.count_subscribers(RISK_TOPIC) == 1,
                10.0,
            ),
            "roundabout node not discovered",
        )
        first_pose = ROUTE_SAMPLES[870.0]
        for _ in range(5):
            self.odom_pub.publish(
                self._odom(self.node.get_clock().now().to_msg(), first_pose)
            )
            self._spin(0.1)

        zone_a = (-90.5, 330.0)   # inside the conflict polygon
        zone_b = (-88.0, 334.0)   # inside
        zone_c = (-85.0, 337.0)   # inside
        far = (-101.5, 343.6)     # roundabout centre island, outside the annulus
        # Object 4 carries a deliberately short prediction (reaches only 4.0 s),
        # so prediction_covers_ego_exit is false whenever the ego exit ETA
        # exceeds ~4.0 s -- i.e. for the earlier approach stations.
        SHORT_HORIZON_S = 4.0
        never_samples = [
            (DT_S * k, *far)
            for k in range(1, int(SHORT_HORIZON_S / DT_S) + 1)
        ]

        input_frames = 0
        input_objects = 0
        relevant_object_frames = 0
        relevant_uuids = set()
        ego_entry_valid = 0
        object_entry_valid = 0
        temporal_gap_valid = 0
        overlap_frames = 0
        multi_interval_object_frames = 0
        later_reentry_frames = 0
        any_overlap_frames = 0
        minimum_gap_valid_frames = 0
        prediction_covers_ego_exit_frames = 0
        ego_entry_eta = []
        object_entry_eta = []
        arrival_delta = []
        non_overlap_gap = []
        representative = None

        for station in sorted(ROUTE_SAMPLES):
            pose = ROUTE_SAMPLES[station]
            objects = [
                _object(1, far, _window_track(zone_a, far, (0.5, 1.0)), pose),
                _object(2, far, _window_track(zone_b, far, (2.5, 4.5)), pose),
                _object(3, far, _window_track(zone_c, far, (7.0, HORIZON_S)), pose),
                _object(4, far, never_samples, pose),
                # Leaves the conflict region and re-enters it during the ego
                # window: the first-interval facts alone look "clear".
                _object(
                    5, far,
                    _multi_window_track(zone_b, far, [(0.5, 1.0), (3.0, 4.5)]),
                    pose,
                ),
            ]
            frame = self._publish(pose, objects)
            input_frames += 1
            input_objects += len(objects)
            self.assertEqual(frame.conflict_zone_id, "kcity_roundabout")
            self.assertEqual(frame.header.frame_id, "map")
            if frame.ego_entry_valid:
                ego_entry_valid += 1
                ego_entry_eta.append(frame.ego_entry_time_s)
            for obj in frame.objects:
                for name in (
                    "object_entry_time_s", "object_exit_time_s", "arrival_delta_s",
                    "temporal_gap_s", "object_map_distance_to_conflict_m",
                    "minimum_temporal_gap_s", "prediction_horizon_s",
                ):
                    self.assertTrue(math.isfinite(getattr(obj, name)))
                self.assertGreaterEqual(obj.minimum_temporal_gap_s, 0.0)
                if obj.predicted_conflict_interval_count > 1:
                    multi_interval_object_frames += 1
                if obj.later_reentry_detected:
                    later_reentry_frames += 1
                if obj.any_occupancy_overlap:
                    any_overlap_frames += 1
                if obj.minimum_temporal_gap_valid:
                    minimum_gap_valid_frames += 1
                if obj.prediction_covers_ego_exit:
                    prediction_covers_ego_exit_frames += 1
                if obj.relevant_to_conflict:
                    relevant_object_frames += 1
                    relevant_uuids.add(obj.object_id.uuid[0])
                if obj.object_entry_valid:
                    object_entry_valid += 1
                    object_entry_eta.append(obj.object_entry_time_s)
                if obj.arrival_delta_valid:
                    arrival_delta.append(obj.arrival_delta_s)
                if obj.temporal_gap_valid:
                    temporal_gap_valid += 1
                    if obj.occupancy_overlap:
                        overlap_frames += 1
                    else:
                        non_overlap_gap.append(obj.temporal_gap_s)
            if abs(station - 870.0) < 1e-6:
                representative = frame

        # Representative frame (ego ~20 m before the conflict entry).
        self.assertIsNotNone(representative)
        by_uuid = {o.object_id.uuid[0]: o for o in representative.objects}
        self.assertEqual(len(by_uuid), 5)
        self.assertTrue(representative.ego_entry_valid)
        self.assertAlmostEqual(
            representative.ego_entry_time_s, (S_ENTER - 870.0) / EGO_SPEED, delta=0.7
        )
        self.assertGreater(representative.ego_exit_time_s, representative.ego_entry_time_s)

        a = by_uuid[1]
        self.assertTrue(a.relevant_to_conflict)
        self.assertFalse(a.occupancy_overlap)
        self.assertTrue(a.temporal_gap_valid)
        self.assertGreater(a.temporal_gap_s, 0.0)
        self.assertLess(a.arrival_delta_s, 0.0)
        # Single interval: the all-interval summary matches the first-interval.
        self.assertEqual(a.predicted_conflict_interval_count, 1)
        self.assertFalse(a.later_reentry_detected)
        self.assertFalse(a.any_occupancy_overlap)
        self.assertAlmostEqual(a.minimum_temporal_gap_s, a.temporal_gap_s, places=4)
        self.assertTrue(a.prediction_covers_ego_exit)

        b = by_uuid[2]
        self.assertTrue(b.relevant_to_conflict)
        self.assertTrue(b.occupancy_overlap)
        self.assertAlmostEqual(b.temporal_gap_s, 0.0, places=5)
        self.assertTrue(b.any_occupancy_overlap)
        self.assertAlmostEqual(b.minimum_temporal_gap_s, 0.0, places=5)

        c = by_uuid[3]
        self.assertTrue(c.relevant_to_conflict)
        self.assertFalse(c.occupancy_overlap)
        self.assertGreater(c.temporal_gap_s, 0.0)
        self.assertGreater(c.arrival_delta_s, 0.0)

        d = by_uuid[4]
        self.assertFalse(d.relevant_to_conflict)
        self.assertFalse(d.object_entry_valid)
        self.assertFalse(d.temporal_gap_valid)
        self.assertEqual(d.predicted_conflict_interval_count, 0)
        self.assertFalse(d.minimum_temporal_gap_valid)
        # Short prediction: at s=870 the ego exit ETA (5.5 s) is past object 4's
        # 4.0 s horizon, so coverage is (correctly) not asserted.
        self.assertAlmostEqual(d.prediction_horizon_s, 4.0, places=3)
        self.assertFalse(d.prediction_covers_ego_exit)

        # Both branches of prediction_covers_ego_exit are exercised in the sweep:
        # objects 1/2/3/5 (8 s horizon) always cover; object 4 (4 s) does not for
        # the earlier stations.
        self.assertGreater(prediction_covers_ego_exit_frames, 0)
        self.assertLess(prediction_covers_ego_exit_frames, input_objects)

        # Central case: first interval clears before the ego, second re-enters
        # the ego window. First-interval facts alone would say "clear".
        e = by_uuid[5]
        self.assertTrue(e.relevant_to_conflict)
        self.assertEqual(e.predicted_conflict_interval_count, 2)
        self.assertTrue(e.later_reentry_detected)
        self.assertFalse(e.occupancy_overlap)            # first interval only
        self.assertGreater(e.temporal_gap_s, 0.0)        # first interval only
        self.assertLess(e.arrival_delta_s, 0.0)          # first entry before ego
        self.assertTrue(e.any_occupancy_overlap)         # re-entry overlaps ego
        self.assertAlmostEqual(e.minimum_temporal_gap_s, 0.0, places=5)
        self.assertTrue(e.prediction_covers_ego_exit)

        self.assertEqual(representative.relevant_object_count, 4)

        # The canonical five-object scenario always contains an overlap/re-entry
        # object, so exercise RELEASE with a separate valid zero-relevant-object
        # frame rather than weakening that scenario.
        gap_message_count = len(self.frames)
        release_input = RoundaboutGapRiskArray()
        release_input.header.stamp = self.node.get_clock().now().to_msg()
        release_input.header.frame_id = "map"
        release_input.conflict_zone_id = "kcity_roundabout"
        release_input.ego_entry_valid = True
        release_input.ego_entry_time_s = 4.0
        release_input.ego_exit_valid = True
        release_input.ego_exit_time_s = 7.0
        release_input.ego_route_distance_to_entry_m = 32.0
        release_input.ego_route_distance_to_exit_m = 56.0
        release_input.ego_speed_mps = EGO_SPEED
        want_response = len(self.responses) + 1
        self.response_input_pub.publish(release_input)
        self.assertTrue(
            self._spin_until(lambda: len(self.responses) >= want_response, 3.0),
            "roundabout response node did not publish zero-object RELEASE",
        )
        self.assertEqual(
            self.responses[-1].action, RoundaboutGapResponse.ACTION_RELEASE
        )
        self.assertTrue(self.responses[-1].active)

        actions = [response.action for response in self.responses]
        reasons = [response.reason for response in self.responses]
        self.assertEqual(actions.count(RoundaboutGapResponse.ACTION_RELEASE), 1)
        self.assertEqual(actions.count(RoundaboutGapResponse.ACTION_YIELD), 2)
        self.assertEqual(actions.count(RoundaboutGapResponse.ACTION_HOLD), 3)
        self.assertEqual(
            reasons.count(RoundaboutGapResponse.REASON_OVERLAP), 5
        )
        for response in self.responses:
            for name in (
                "ego_speed_mps", "ego_route_distance_to_entry_m",
                "available_distance_m", "comfortable_stop_distance_m",
                "limiting_gap_s",
            ):
                self.assertTrue(math.isfinite(getattr(response, name)))

        result = {
            "input_risk_messages": input_frames,
            "gap_messages": gap_message_count,
            "response_messages": len(self.responses),
            "response_release_frames": actions.count(
                RoundaboutGapResponse.ACTION_RELEASE
            ),
            "response_yield_frames": actions.count(
                RoundaboutGapResponse.ACTION_YIELD
            ),
            "response_hold_frames": actions.count(
                RoundaboutGapResponse.ACTION_HOLD
            ),
            "response_rejected_frames": 0,
            "response_reason_counts": {
                "clear_gap": reasons.count(RoundaboutGapResponse.REASON_CLEAR_GAP),
                "overlap": reasons.count(RoundaboutGapResponse.REASON_OVERLAP),
                "gap_too_small": reasons.count(
                    RoundaboutGapResponse.REASON_GAP_TOO_SMALL
                ),
                "insufficient_prediction": reasons.count(
                    RoundaboutGapResponse.REASON_INSUFFICIENT_PREDICTION
                ),
                "invalid_ego_state": reasons.count(
                    RoundaboutGapResponse.REASON_INVALID_EGO_STATE
                ),
            },
            "input_objects": input_objects,
            "relevant_object_frames": relevant_object_frames,
            "unique_relevant_uuids": len(relevant_uuids),
            "ego_entry_valid_frames": ego_entry_valid,
            "object_entry_valid": object_entry_valid,
            "temporal_gap_valid": temporal_gap_valid,
            "overlap_object_frames": overlap_frames,
            "multi_interval_object_frames": multi_interval_object_frames,
            "later_reentry_object_frames": later_reentry_frames,
            "any_occupancy_overlap_object_frames": any_overlap_frames,
            "minimum_temporal_gap_valid_object_frames": minimum_gap_valid_frames,
            "prediction_covers_ego_exit_object_frames": (
                prediction_covers_ego_exit_frames
            ),
            "ego_entry_eta": {
                "median": round(statistics.median(ego_entry_eta), 3),
                "p10": round(_pctile(ego_entry_eta, 0.10), 3),
                "min": round(min(ego_entry_eta), 3),
                "max": round(max(ego_entry_eta), 3),
            },
            "object_entry_eta": {
                "median": round(statistics.median(object_entry_eta), 3),
                "p10": round(_pctile(object_entry_eta, 0.10), 3),
                "min": round(min(object_entry_eta), 3),
                "max": round(max(object_entry_eta), 3),
            },
            "arrival_delta": {
                "median": round(statistics.median(arrival_delta), 3),
                "p10": round(_pctile(arrival_delta, 0.10), 3),
                "min": round(min(arrival_delta), 3),
                "max": round(max(arrival_delta), 3),
            },
            "non_overlap_gap": {
                "median": round(statistics.median(non_overlap_gap), 3),
                "p10": round(_pctile(non_overlap_gap, 0.10), 3),
                "min": round(min(non_overlap_gap), 3),
            },
            "publish_to_receive_latency_ms": {
                "median": round(statistics.median(self.latency_ms), 3),
                "p95": round(_pctile(self.latency_ms, 0.95), 3),
                "max": round(max(self.latency_ms), 3),
                "samples": len(self.latency_ms),
            },
            "risk_input_to_response_receive_latency_ms": {
                "median": round(statistics.median(self.response_latency_ms), 3),
                "p95": round(_pctile(self.response_latency_ms, 0.95), 3),
                "max": round(max(self.response_latency_ms), 3),
                "samples": len(self.response_latency_ms),
            },
            "response_examples": {
                "release": {
                    "ego_distance_m": round(
                        self.responses[-1].ego_route_distance_to_entry_m, 3
                    ),
                    "relevant_objects": self.responses[-1].relevant_object_count,
                    "complete_coverage": (
                        self.responses[-1].complete_prediction_coverage
                    ),
                },
                "yield": {
                    "ego_distance_m": round(
                        next(
                            r for r in self.responses
                            if r.action == RoundaboutGapResponse.ACTION_YIELD
                        )
                        .ego_route_distance_to_entry_m,
                        3,
                    ),
                    "reason": next(
                        r for r in self.responses
                        if r.action == RoundaboutGapResponse.ACTION_YIELD
                    ).reason,
                },
                "hold": {
                    "ego_distance_m": round(
                        next(
                            r for r in self.responses
                            if r.action == RoundaboutGapResponse.ACTION_HOLD
                        )
                        .ego_route_distance_to_entry_m,
                        3,
                    ),
                    "reason": next(
                        r for r in self.responses
                        if r.action == RoundaboutGapResponse.ACTION_HOLD
                    ).reason,
                },
            },
            "representative_870m": {
                "ego_entry_time_s": round(representative.ego_entry_time_s, 3),
                "ego_exit_time_s": round(representative.ego_exit_time_s, 3),
                "clears_first_object_interval_s": [
                    round(a.object_entry_time_s, 3),
                    round(a.object_exit_time_s, 3),
                ],
                "clears_first_object_exit_valid": a.object_exit_valid,
                "clears_first_gap_s": round(a.temporal_gap_s, 3),
                "clears_first_arrival_delta_s": round(a.arrival_delta_s, 3),
                "overlap_object_interval_s": [
                    round(b.object_entry_time_s, 3),
                    round(b.object_exit_time_s, 3),
                ],
                "overlap_object_exit_valid": b.object_exit_valid,
                "overlap_gap_s": round(b.temporal_gap_s, 3),
                "overlap_arrival_delta_s": round(b.arrival_delta_s, 3),
                "after_ego_object_entry_s": round(c.object_entry_time_s, 3),
                "after_ego_object_exit_valid": c.object_exit_valid,
                "after_ego_gap_s": round(c.temporal_gap_s, 3),
                "after_ego_arrival_delta_s": round(c.arrival_delta_s, 3),
                "reentry_first_interval_gap_s": round(e.temporal_gap_s, 3),
                "reentry_first_interval_overlap": e.occupancy_overlap,
                "reentry_interval_count": e.predicted_conflict_interval_count,
                "reentry_later_reentry_detected": e.later_reentry_detected,
                "reentry_any_occupancy_overlap": e.any_occupancy_overlap,
                "reentry_minimum_temporal_gap_s": round(
                    e.minimum_temporal_gap_s, 3
                ),
                "reentry_prediction_horizon_s": round(e.prediction_horizon_s, 3),
                "reentry_prediction_covers_ego_exit": e.prediction_covers_ego_exit,
            },
        }
        Path("/tmp/heven_roundabout_gap_risk_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print("ROUNDABOUT_GAP_RISK_RESULT=" + json.dumps(result, sort_keys=True))


@launch_testing.post_shutdown_test()
class TestRoundaboutGapRiskRuntimeShutdown(unittest.TestCase):
    def test_clean_shutdown(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info)
