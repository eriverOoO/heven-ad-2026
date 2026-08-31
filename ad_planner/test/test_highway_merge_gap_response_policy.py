"""Live ad_highway_merge_gap_response policy replay driven by synthetic
HighwayMergeGapRiskArray frames.

This isolates the merge-decision policy (no dynamic-object-risk / highway-merge
gap-risk geometry node), so MERGE_READY / WAIT / HOLD, front and rear gap
policy, rear closing, predicted route clearance, multi-object arbitration, the
prediction-coverage requirement, and the WAIT->HOLD approach sequence are all
exercised deterministically in the real ROS node.
"""

import json
from pathlib import Path
import time
import unittest

from ad_interfaces.msg import (
    HighwayMergeGapRisk,
    HighwayMergeGapRiskArray,
    HighwayMergeGapResponse,
)
from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
import launch_testing
import launch_testing.actions
import launch_testing.asserts
from launch_ros.actions import Node as LaunchNode
import pytest
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


DOMAIN_ID = 97
RISK_TOPIC = "/ad/planning/highway_merge_gap_risks"
RESPONSE_TOPIC = "/ad/planning/highway_merge_gap_response"

MERGE_ZONE_ENTRY = 1118.7417511691015
MERGE_REFERENCE = 1286.1546454384027
EGO_SPEED = 8.0

_QOS_1 = QoSProfile(
    depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE
)


@pytest.mark.launch_test
def generate_test_description():
    return LaunchDescription(
        [
            SetEnvironmentVariable("ROS_DOMAIN_ID", str(DOMAIN_ID)),
            SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
            SetEnvironmentVariable(
                "ROS_LOG_DIR", "/tmp/heven_highway_merge_gap_response_policy_ros_log"
            ),
            LaunchNode(
                package="ad_planner",
                executable="ad_highway_merge_gap_response_node",
                name="ad_highway_merge_gap_response",
                output="screen",
                parameters=[
                    {
                        "maximum_input_age_s": 5.0,
                        "runtime_summary_interval_frames": 0,
                    }
                ],
            ),
            launch_testing.actions.ReadyToTest(),
        ]
    )


def _risk(
    first_byte,
    delta_s_at_merge,
    *,
    relation="ahead",
    delta_s_now=None,
    object_speed=8.0,
    covers=True,
    delta_valid=True,
    predicted_route_gap=25.0,
    closing=False,
    closing_speed=0.0,
    coincidence_valid=False,
    coincidence_s=0.0,
):
    risk = HighwayMergeGapRisk()
    risk.object_id.uuid = [first_byte] + [0] * 15
    risk.classification = 1
    risk.classification_probability = 0.9
    risk.existence_probability = 0.9
    risk.relevant_to_merge = True
    risk.object_route_s_m = MERGE_REFERENCE + delta_s_at_merge
    risk.object_lateral_offset_m = 0.0
    risk.delta_s_now_m = (
        delta_s_at_merge - 5.0 if delta_s_now is None else float(delta_s_now)
    )
    risk.object_longitudinal_speed_mps = float(object_speed)
    risk.relative_longitudinal_speed_mps = float(object_speed) - EGO_SPEED
    risk.delta_s_at_merge_valid = delta_valid
    risk.delta_s_at_merge_m = float(delta_s_at_merge)
    risk.is_ahead_at_merge = delta_valid and relation == "ahead"
    risk.is_behind_at_merge = delta_valid and relation == "behind"
    risk.is_alongside_at_merge = delta_valid and relation == "alongside"
    risk.longitudinal_gap_closing = closing
    risk.longitudinal_closing_speed_mps = float(closing_speed)
    risk.time_to_route_coincidence_valid = coincidence_valid
    risk.time_to_route_coincidence_s = float(coincidence_s)
    risk.predicted_min_route_gap_valid = True
    risk.predicted_min_route_gap_m = float(predicted_route_gap)
    risk.predicted_min_route_gap_time_s = 5.0
    # Large enough that "covers" stays self-consistent with ego_merge_time_s
    # across the distance sweep (the response node rejects a covered object
    # whose horizon is shorter than the ego merge time).
    risk.prediction_horizon_s = 40.0
    risk.prediction_covers_merge_time = covers
    return risk


def _array(stamp, risks, *, distance_to_merge=150.0, timing_valid=True, speed=EGO_SPEED):
    array = HighwayMergeGapRiskArray()
    array.header.stamp = stamp
    array.header.frame_id = "map"
    array.merge_zone_id = "kcity_highway_onramp"
    array.target_lane_sequence_id = "route:0"
    array.source_lane_sequence_id = "route:0:left:1"
    array.merge_zone_entry_route_s_m = MERGE_ZONE_ENTRY
    array.merge_reference_route_s_m = MERGE_REFERENCE
    array.ego_route_s_m = MERGE_REFERENCE - distance_to_merge
    array.ego_longitudinal_speed_mps = float(speed)
    array.ego_route_distance_to_zone_entry_m = (
        MERGE_ZONE_ENTRY - array.ego_route_s_m
    )
    array.ego_route_distance_to_merge_m = float(distance_to_merge)
    array.ego_in_merge_zone_now = 0.0 <= array.ego_route_distance_to_zone_entry_m
    array.ego_merge_timing_valid = timing_valid and speed >= 0.5
    array.ego_merge_time_s = (
        float(distance_to_merge) / max(speed, 1e-3)
        if array.ego_merge_timing_valid
        else 0.0
    )
    array.relevant_object_count = sum(1 for r in risks if r.relevant_to_merge)
    array.objects = risks
    return array


class TestHighwayMergeGapResponsePolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = Node("highway_merge_gap_response_policy_probe")
        cls.responses = []
        cls.inputs_published = 0
        cls.node.create_subscription(
            HighwayMergeGapResponse,
            RESPONSE_TOPIC,
            lambda m: cls.responses.append(m),
            _QOS_1,
        )
        cls.pub = cls.node.create_publisher(
            HighwayMergeGapRiskArray, RISK_TOPIC, _QOS_1
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

    def _publish(self, risks, **array_kwargs):
        stamp = self.node.get_clock().now().to_msg()
        array = _array(stamp, risks, **array_kwargs)
        want = len(self.responses) + 1
        type(self).inputs_published += 1
        self.pub.publish(array)
        self.assertTrue(
            self._spin_until(lambda: len(self.responses) >= want, 3.0),
            "response node did not publish",
        )
        return self.responses[-1]

    def test_policy(self):
        self.assertTrue(
            self._spin_until(
                lambda: self.node.count_publishers(RESPONSE_TOPIC) >= 1
                and self.node.count_subscribers(RISK_TOPIC) == 1,
                10.0,
            ),
            "response node not discovered",
        )

        counts = {"MERGE_READY": 0, "WAIT": 0, "HOLD": 0, "INACTIVE": 0}
        reasons = {}
        examples = {}

        def record(message):
            if not message.active:
                counts["INACTIVE"] += 1
            elif message.action == HighwayMergeGapResponse.ACTION_MERGE_READY:
                counts["MERGE_READY"] += 1
            elif message.action == HighwayMergeGapResponse.ACTION_WAIT:
                counts["WAIT"] += 1
            else:
                counts["HOLD"] += 1
            reasons[message.reason] = reasons.get(message.reason, 0) + 1

        # 1. Zero relevant objects with valid ego timing -> MERGE_READY.
        message = self._publish([])
        record(message)
        self.assertEqual(message.action, HighwayMergeGapResponse.ACTION_MERGE_READY)
        self.assertEqual(message.reason, HighwayMergeGapResponse.REASON_CLEAR_GAP)
        self.assertTrue(message.complete_prediction_coverage)

        # 2. Clear front + rear -> MERGE_READY (Phase 40).
        message = self._publish(
            [
                _risk(1, 45.0, relation="ahead", object_speed=8.0),
                _risk(2, -70.0, relation="behind", object_speed=10.0),
            ]
        )
        record(message)
        self.assertEqual(message.action, HighwayMergeGapResponse.ACTION_MERGE_READY)
        self.assertTrue(message.front_object_valid)
        self.assertTrue(message.rear_object_valid)
        self.assertTrue(message.predicted_route_clearance_valid)
        examples["merge_ready"] = {
            "front_gap_m": round(message.front_gap_m, 2),
            "front_time_headway_s": round(message.front_time_headway_s, 3),
            "rear_gap_m": round(message.rear_gap_m, 2),
            "rear_time_headway_s": round(message.rear_time_headway_s, 3),
            "min_predicted_route_gap_m": round(
                message.minimum_predicted_route_gap_m, 2
            ),
        }

        # 3. Front too close -> not ready, REASON_FRONT_GAP (Phase 19B / 5).
        message = self._publish([_risk(1, 6.0, relation="ahead")])
        record(message)
        self.assertNotEqual(
            message.action, HighwayMergeGapResponse.ACTION_MERGE_READY
        )
        self.assertEqual(message.reason, HighwayMergeGapResponse.REASON_FRONT_GAP)

        # 4. Rear too close -> not ready, REASON_REAR_GAP (Phase 6).
        message = self._publish(
            [_risk(2, -10.0, relation="behind", object_speed=20.0)]
        )
        record(message)
        self.assertEqual(message.reason, HighwayMergeGapResponse.REASON_REAR_GAP)

        # 5. Fast-closing rear with an otherwise-fine gap -> REASON_REAR_CLOSING
        #    (Phase 18 / 41). Rear gap 60 m / 12 m/s = 5 s headway (ok), but
        #    coincidence time 2 s < 3 s.
        message = self._publish(
            [
                _risk(
                    3,
                    -60.0,
                    relation="behind",
                    object_speed=12.0,
                    delta_s_now=-30.0,
                    closing=True,
                    closing_speed=4.0,
                    coincidence_valid=True,
                    coincidence_s=2.0,
                )
            ]
        )
        record(message)
        self.assertEqual(
            message.reason, HighwayMergeGapResponse.REASON_REAR_CLOSING
        )
        self.assertTrue(message.rear_closing_object_valid)
        examples["rear_closing"] = {
            "rear_gap_m": round(message.rear_gap_m, 2),
            "rear_closing_time_s": round(message.rear_closing_time_s, 2),
        }

        # 6. Predicted route gap too small even though the merge-instant gap is
        #    large -> REASON_PREDICTED_ROUTE_CONFLICT (Phase 13).
        message = self._publish(
            [_risk(4, 40.0, relation="ahead", predicted_route_gap=3.0)]
        )
        record(message)
        self.assertEqual(
            message.reason,
            HighwayMergeGapResponse.REASON_PREDICTED_ROUTE_CONFLICT,
        )

        # 7. Alongside object -> REASON_ALONGSIDE (Phase 9).
        message = self._publish([_risk(5, 2.0, relation="alongside")])
        record(message)
        self.assertEqual(message.reason, HighwayMergeGapResponse.REASON_ALONGSIDE)

        # 8. Fallback extrapolation looks safe but coverage is false -> not
        #    ready, REASON_INSUFFICIENT_PREDICTION (Phase 26, mandatory).
        message = self._publish(
            [_risk(6, 200.0, relation="ahead", covers=False)]
        )
        record(message)
        self.assertNotEqual(
            message.action, HighwayMergeGapResponse.ACTION_MERGE_READY
        )
        self.assertEqual(
            message.reason,
            HighwayMergeGapResponse.REASON_INSUFFICIENT_PREDICTION,
        )
        self.assertFalse(message.complete_prediction_coverage)
        examples["insufficient_prediction"] = {
            "fallback_delta_s_at_merge_m": 200.0,
            "complete_prediction_coverage": message.complete_prediction_coverage,
        }

        # 9. Large total merge span but unsafe rear -> not ready (Phase 12 / 17).
        message = self._publish(
            [
                _risk(1, 90.0, relation="ahead"),
                _risk(2, -2.0, relation="behind", object_speed=20.0),
            ]
        )
        record(message)
        self.assertNotEqual(
            message.action, HighwayMergeGapResponse.ACTION_MERGE_READY
        )
        self.assertEqual(message.reason, HighwayMergeGapResponse.REASON_REAR_GAP)
        self.assertEqual(message.source_object_id.uuid[0], 2)

        # 10. Multi-object arbitration: one unsafe among many, deterministic
        #     limiting UUID (Phase 14 / 15 / 25).
        message = self._publish(
            [
                _risk(1, 45.0, relation="ahead"),
                _risk(7, 6.0, relation="ahead"),
                _risk(3, 6.0, relation="ahead"),
                _risk(2, -70.0, relation="behind", object_speed=10.0),
            ]
        )
        record(message)
        self.assertEqual(message.reason, HighwayMergeGapResponse.REASON_FRONT_GAP)
        self.assertEqual(message.source_object_id.uuid[0], 3)

        # 11. WAIT -> HOLD as the ego approaches the decision boundary with a
        #     constant unsafe rear (Phase 21 / 42).
        approach = []
        unsafe_rear = _risk(2, -10.0, relation="behind", object_speed=20.0)
        for distance in (200.0, 120.0, 40.0, 20.0, 8.0):
            message = self._publish(
                [unsafe_rear], distance_to_merge=distance
            )
            record(message)
            approach.append(
                (
                    round(distance, 1),
                    int(message.action),
                    round(message.available_distance_m, 2),
                    round(message.comfortable_stop_distance_m, 2),
                )
            )
        approach_actions = [a[1] for a in approach]
        self.assertEqual(approach_actions, sorted(approach_actions))
        self.assertIn(HighwayMergeGapResponse.ACTION_WAIT, approach_actions)
        self.assertIn(HighwayMergeGapResponse.ACTION_HOLD, approach_actions)
        examples["wait"] = {
            "distance_to_merge_m": approach[0][0],
            "available_distance_m": approach[0][2],
            "comfortable_stop_distance_m": approach[0][3],
        }
        examples["hold"] = {
            "distance_to_merge_m": approach[-1][0],
            "available_distance_m": approach[-1][2],
            "comfortable_stop_distance_m": approach[-1][3],
        }

        # 12. Unsafe rear closing -> rear clears -> full safe gap -> MERGE_READY
        #     (Phase 43).
        seq = []
        seq.append(
            self._publish(
                [
                    _risk(
                        3,
                        -60.0,
                        relation="behind",
                        object_speed=12.0,
                        delta_s_now=-30.0,
                        closing=True,
                        closing_speed=6.0,
                        coincidence_valid=True,
                        coincidence_s=1.5,
                    )
                ]
            )
        )
        seq.append(
            self._publish(
                [_risk(3, -70.0, relation="behind", object_speed=9.0)]
            )
        )
        seq.append(
            self._publish(
                [
                    _risk(1, 50.0, relation="ahead", object_speed=8.0),
                    _risk(3, -80.0, relation="behind", object_speed=9.0),
                ]
            )
        )
        for message in seq:
            record(message)
        self.assertEqual(
            seq[0].reason, HighwayMergeGapResponse.REASON_REAR_CLOSING
        )
        self.assertEqual(
            seq[-1].action, HighwayMergeGapResponse.ACTION_MERGE_READY
        )

        # 13. Stopped ego -> HOLD, REASON_INVALID_EGO_STATE (Phase 23).
        message = self._publish([], distance_to_merge=40.0, speed=0.0)
        record(message)
        self.assertEqual(message.action, HighwayMergeGapResponse.ACTION_HOLD)
        self.assertEqual(
            message.reason, HighwayMergeGapResponse.REASON_INVALID_EGO_STATE
        )

        # 14. Ego past merge -> inactive (Phase 24).
        message = self._publish([], distance_to_merge=-15.0, timing_valid=False)
        record(message)
        self.assertFalse(message.active)

        # 15. Stale / backward stamp is rejected, not published.
        before = len(self.responses)
        old = self.node.get_clock().now().to_msg()
        old.sec -= 5
        type(self).inputs_published += 1
        self.pub.publish(_array(old, []))
        self._spin(0.5)
        self.assertEqual(len(self.responses), before)

        # 16. Front threshold boundary: headway exactly 1.5 s (delta 12 m at
        #     8 m/s) is inclusive-ready; just below is not (Phase 16 / 35).
        boundary = self._publish([_risk(1, 12.0, relation="ahead")])
        record(boundary)
        self.assertEqual(
            boundary.action, HighwayMergeGapResponse.ACTION_MERGE_READY
        )
        below = self._publish([_risk(1, 11.9, relation="ahead")])
        record(below)
        self.assertEqual(below.reason, HighwayMergeGapResponse.REASON_FRONT_GAP)

        result = {
            "risk_messages_processed": self.inputs_published,
            "response_messages_published": len(self.responses),
            "rejected_frames": self.inputs_published - len(self.responses),
            "action_counts": counts,
            "reason_counts": {str(k): v for k, v in sorted(reasons.items())},
            "examples": examples,
        }
        Path("/tmp/heven_highway_merge_gap_response_policy_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            "HIGHWAY_MERGE_GAP_RESPONSE_POLICY_RESULT="
            + json.dumps(result, sort_keys=True)
        )
        # No NaN/Inf anywhere in the recorded responses.
        for message in self.responses:
            for name in (
                "ego_speed_mps",
                "ego_route_distance_to_merge_m",
                "ego_merge_time_s",
                "available_distance_m",
                "comfortable_stop_distance_m",
                "front_gap_m",
                "front_time_headway_s",
                "rear_gap_m",
                "rear_time_headway_s",
                "rear_closing_time_s",
                "minimum_predicted_route_gap_m",
            ):
                value = getattr(message, name)
                self.assertEqual(value, value)  # NaN check
                self.assertLess(abs(value), 1e12)


@launch_testing.post_shutdown_test()
class TestHighwayMergeGapResponsePolicyShutdown(unittest.TestCase):
    def test_clean_shutdown(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info)
