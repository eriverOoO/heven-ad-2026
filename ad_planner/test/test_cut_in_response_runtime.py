"""Deterministic Dynamic Object Risk -> Cut-in Risk -> Cut-in Response replay.

Reuses the Cut-in Risk v1 canonical scenario (mirrored left/right merging
actors plus parallel-adjacent, moving-away, and crossing negative controls) and
adds the longitudinal response node with a prescribed non-zero ego speed. Ego
pose is a fixed per-frame snapshot exactly as in the Cut-in Risk v1 replay; only
the reported body-forward speed varies, so the kinematic response policy is
exercised without adding a new simulator.
"""

import json
import math
from pathlib import Path
import statistics
import time
import unittest

from ad_interfaces.msg import (
    CutInResponse,
    CutInRiskArray,
    PredictedObject,
    PredictedObjectArray,
    PredictedState,
)
from diagnostic_msgs.msg import DiagnosticArray
from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
import launch_testing
import launch_testing.actions
import launch_testing.asserts
from launch_ros.actions import Node as LaunchNode
from nav_msgs.msg import OccupancyGrid, Odometry
import pytest
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


DOMAIN_ID = 97
PREDICTION_TOPIC = "/ad/perception/objects/predicted"
ODOMETRY_TOPIC = "/ad/localization/odometry"
MASK_TOPIC = "/ad/planning/drivable_mask"
CUT_IN_TOPIC = "/ad/planning/cut_in_risks"
RESPONSE_TOPIC = "/ad/planning/cut_in_response"
RESPONSE_DIAGNOSTICS_TOPIC = "/ad/planning/cut_in_response/diagnostics"

REPOSITORY = Path(__file__).resolve().parents[2]
DATA_DIR = REPOSITORY / "ad_data"
GLOBAL_PATH = DATA_DIR / "path" / "2026_molit_comp_global_path.txt"
GLOBAL_PATH_SHA256 = (
    "50658991e607d9339d76e4cd6cb169dfc733ea53b93de2c3e222460bb497cc05"
)

EGO_X = -131.68979755061446
EGO_Y = -428.3310229377821
EGO_YAW = 1.0698539345813243


def _ego_speed(frame):
    return 4.0 + 0.2 * frame


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
    assert GLOBAL_PATH.is_file()
    return LaunchDescription(
        [
            SetEnvironmentVariable("ROS_DOMAIN_ID", str(DOMAIN_ID)),
            SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
            SetEnvironmentVariable(
                "ROS_LOG_DIR", "/tmp/heven_cut_in_response_ros_log"
            ),
            LaunchNode(
                package="ad_lidar_perception",
                executable="ad_dynamic_object_risk_node",
                name="ad_dynamic_object_risk",
                output="screen",
                parameters=[
                    {
                        "prediction_frame_id": "map",
                        "maximum_prediction_age_s": 1.0,
                        "maximum_ego_age_s": 1.0,
                        "runtime_summary_interval_frames": 0,
                    }
                ],
            ),
            LaunchNode(
                package="ad_planner",
                executable="ad_cut_in_risk_node",
                name="ad_cut_in_risk",
                output="screen",
                parameters=[
                    {
                        "data_dir": str(DATA_DIR),
                        "route_corridor_file": "map/route_corridor.json",
                        "route_corridor.expected_global_path_sha256": (
                            GLOBAL_PATH_SHA256
                        ),
                        "maximum_input_age_s": 1.0,
                        "maximum_odometry_skew_s": 1.0,
                        "runtime_summary_interval_frames": 0,
                    }
                ],
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


def _world_from_base(x_base, y_base):
    cosine = math.cos(EGO_YAW)
    sine = math.sin(EGO_YAW)
    return (
        EGO_X + cosine * x_base - sine * y_base,
        EGO_Y + sine * x_base + cosine * y_base,
    )


def _vector_world(vx_base, vy_base):
    cosine = math.cos(EGO_YAW)
    sine = math.sin(EGO_YAW)
    return (
        cosine * vx_base - sine * vy_base,
        sine * vx_base + cosine * vy_base,
    )


def _duration(seconds):
    whole = int(seconds)
    return whole, int(round((seconds - whole) * 1_000_000_000))


def _object(identifier, current_y, lateral_velocity, future_y):
    message = PredictedObject()
    message.object_id.uuid = [identifier] + [0] * 15
    message.existence_probability = 0.95
    message.classification = PredictedObject.CAR
    message.classification_probability = 0.90
    message.dimensions.x = 4.5
    message.dimensions.y = 1.9
    message.dimensions.z = 1.5
    message.initial_pose.pose.orientation.w = 1.0
    x_world, y_world = _world_from_base(15.0, current_y)
    message.initial_pose.pose.position.x = x_world
    message.initial_pose.pose.position.y = y_world
    vx_world, vy_world = _vector_world(0.0, lateral_velocity)
    message.initial_twist.twist.linear.x = vx_world
    message.initial_twist.twist.linear.y = vy_world
    for horizon in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        state = PredictedState()
        state.time_from_start.sec, state.time_from_start.nanosec = _duration(
            horizon
        )
        future_x_world, future_y_world = _world_from_base(
            15.0, future_y(horizon)
        )
        state.pose.pose.position.x = future_x_world
        state.pose.pose.position.y = future_y_world
        state.pose.pose.orientation.w = 1.0
        message.states.append(state)
    return message


def _prediction(stamp, elapsed):
    message = PredictedObjectArray()
    message.header.stamp = stamp
    message.header.frame_id = "map"

    right_y = min(-4.0 + elapsed, 0.0)
    right_vy = 1.0 if right_y < 0.0 else 0.0
    message.objects.append(
        _object(1, right_y, right_vy, lambda horizon: min(right_y + horizon, 0.0))
    )
    left_y = max(4.0 - elapsed, 0.0)
    left_vy = -1.0 if left_y > 0.0 else 0.0
    message.objects.append(
        _object(2, left_y, left_vy, lambda horizon: max(left_y - horizon, 0.0))
    )
    message.objects.append(_object(3, -4.0, 0.0, lambda _horizon: -4.0))
    away_y = 4.0 + elapsed
    message.objects.append(
        _object(4, away_y, 1.0, lambda horizon: away_y + horizon)
    )
    crossing_y = -4.0 + 4.0 * elapsed
    message.objects.append(
        _object(5, crossing_y, 4.0, lambda horizon: crossing_y + 4.0 * horizon)
    )
    return message


def _odometry(stamp, speed_mps):
    message = Odometry()
    message.header.stamp = stamp
    message.header.frame_id = "map"
    message.child_frame_id = "base_link"
    message.pose.pose.position.x = EGO_X
    message.pose.pose.position.y = EGO_Y
    message.pose.pose.orientation.z = math.sin(EGO_YAW / 2.0)
    message.pose.pose.orientation.w = math.cos(EGO_YAW / 2.0)
    message.twist.twist.linear.x = speed_mps
    return message


def _mask(stamp):
    message = OccupancyGrid()
    message.header.stamp = stamp
    message.header.frame_id = "base_link"
    message.info.width = 1040
    message.info.height = 200
    message.info.resolution = 0.1
    message.info.origin.position.x = -4.0
    message.info.origin.position.y = -10.0
    message.info.origin.orientation.w = 1.0
    message.data = [0] * (message.info.width * message.info.height)
    return message


_RESPONSE_FLOAT_FIELDS = (
    "requested_max_speed_mps",
    "required_deceleration_mps2",
    "ego_speed_mps",
    "route_s_rel_m",
    "predicted_entry_route_s_rel_m",
    "predicted_entry_time_s",
    "lateral_velocity_toward_corridor_mps",
    "ttc_s",
    "cpa_time_s",
    "cpa_distance_m",
    "predicted_min_separation_m",
)


class TestCutInResponseRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = Node("cut_in_response_runtime_probe")
        cls.cut_in_messages = []
        cls.response_messages = []
        cls.latencies_ms = []
        cls.node.create_subscription(
            CutInRiskArray,
            CUT_IN_TOPIC,
            lambda message: cls.cut_in_messages.append(message),
            _QOS_1,
        )
        cls.node.create_subscription(
            CutInResponse,
            RESPONSE_TOPIC,
            lambda message: cls.response_messages.append(message),
            _QOS_1,
        )

        def diagnostics_callback(message):
            for status in message.status:
                if status.name != "cut_in_response":
                    continue
                values = {item.key: item.value for item in status.values}
                if values.get("published") == "true":
                    cls.latencies_ms.append(float(values["latency_ms"]))

        cls.node.create_subscription(
            DiagnosticArray,
            RESPONSE_DIAGNOSTICS_TOPIC,
            diagnostics_callback,
            _QOS_10,
        )
        cls.prediction_publisher = cls.node.create_publisher(
            PredictedObjectArray, PREDICTION_TOPIC, _QOS_1
        )
        cls.odometry_publisher = cls.node.create_publisher(
            Odometry, ODOMETRY_TOPIC, _QOS_10
        )
        cls.mask_publisher = cls.node.create_publisher(
            OccupancyGrid, MASK_TOPIC, _QOS_10
        )

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def _spin_until(self, predicate, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.02)
            if predicate():
                return True
        return False

    def test_pipeline_produces_monotone_longitudinal_response(self):
        self.assertTrue(
            self._spin_until(
                lambda: self.node.count_publishers(CUT_IN_TOPIC) == 1
                and self.node.count_publishers(RESPONSE_TOPIC) == 1
                and self.node.count_subscribers(PREDICTION_TOPIC) >= 1
                and self.node.count_subscribers(ODOMETRY_TOPIC) >= 2
                and self.node.count_subscribers(MASK_TOPIC) >= 1,
                8.0,
            ),
            "pipeline publishers/subscribers were not discovered",
        )

        # Prime discovery: deliver several odometry/mask frames before the
        # first prediction so no node rejects frame 0 on a missing ego pose.
        for _ in range(10):
            stamp = self.node.get_clock().now().to_msg()
            self.odometry_publisher.publish(_odometry(stamp, _ego_speed(0)))
            self.mask_publisher.publish(_mask(stamp))
            self._spin_until(lambda: False, 0.05)

        frames = 31
        elapsed_by_stamp = {}
        speed_by_stamp = {}
        for frame in range(frames):
            elapsed = frame * 0.1
            speed = _ego_speed(frame)
            stamp = self.node.get_clock().now().to_msg()
            stamp_ns = _stamp_ns(stamp)
            elapsed_by_stamp[stamp_ns] = elapsed
            speed_by_stamp[stamp_ns] = speed
            self.odometry_publisher.publish(_odometry(stamp, speed))
            self.mask_publisher.publish(_mask(stamp))
            self._spin_until(lambda: False, 0.015)
            self.prediction_publisher.publish(_prediction(stamp, elapsed))
            expected = frame + 1
            self.assertTrue(
                self._spin_until(
                    lambda: len(self.cut_in_messages) >= expected
                    and len(self.response_messages) >= expected,
                    2.5,
                ),
                f"pipeline did not publish frame {frame}",
            )

        self._spin_until(lambda: len(self.latencies_ms) >= frames, 1.0)
        self.assertEqual(len(self.cut_in_messages), frames)
        self.assertEqual(len(self.response_messages), frames)

        candidate_by_stamp = {}
        for message in self.cut_in_messages:
            stamp_ns = _stamp_ns(message.header.stamp)
            candidate_by_stamp[stamp_ns] = {
                risk.object_id.uuid[0]
                for risk in message.risks
                if risk.cut_in_candidate
            }

        none_frames = 0
        slowdown_frames = 0
        hold_frames = 0
        finite = True
        sequence = []
        first_active_time = None
        first_slowdown_time = None
        first_hold_time = None
        source_ids = set()
        negative_source = 0
        for message in self.response_messages:
            stamp_ns = _stamp_ns(message.header.stamp)
            elapsed = elapsed_by_stamp[stamp_ns]
            speed = speed_by_stamp[stamp_ns]
            candidates = candidate_by_stamp.get(stamp_ns, set())

            finite = finite and all(
                math.isfinite(getattr(message, name))
                for name in _RESPONSE_FLOAT_FIELDS
            )
            self.assertEqual(message.header.frame_id, "map")
            self.assertEqual(message.candidate_count, len(candidates))

            action = int(message.action)
            sequence.append((round(elapsed, 1), action))
            if action == CutInResponse.ACTION_NONE:
                none_frames += 1
                self.assertFalse(message.active)
            else:
                self.assertTrue(message.active)
                self.assertIn(int(message.source_object_id.uuid[0]), candidates)
                source_ids.add(int(message.source_object_id.uuid[0]))
                if int(message.source_object_id.uuid[0]) in {3, 4, 5}:
                    negative_source += 1
                if first_active_time is None:
                    first_active_time = elapsed
                if action == CutInResponse.ACTION_SLOWDOWN:
                    slowdown_frames += 1
                    self.assertTrue(message.requested_max_speed_valid)
                    self.assertGreater(message.requested_max_speed_mps, 0.0)
                    self.assertLess(message.requested_max_speed_mps, speed + 1e-6)
                    if first_slowdown_time is None:
                        first_slowdown_time = elapsed
                elif action == CutInResponse.ACTION_HOLD:
                    hold_frames += 1
                    self.assertAlmostEqual(message.requested_max_speed_mps, 0.0)
                    if first_hold_time is None:
                        first_hold_time = elapsed

            # No response without a candidate.
            if not candidates:
                self.assertEqual(action, CutInResponse.ACTION_NONE)

        self.assertTrue(finite)
        self.assertEqual(negative_source, 0)
        self.assertTrue(source_ids.issubset({1, 2}))
        # The reused scenario always yields at least one real restriction.
        self.assertGreater(slowdown_frames + hold_frames, 0)
        # After both merging actors are inside the corridor the response clears.
        tail = [
            action
            for elapsed, action in sequence
            if elapsed >= 2.4
        ]
        self.assertTrue(tail)
        self.assertTrue(all(action == CutInResponse.ACTION_NONE for action in tail))

        ordered_latency = sorted(self.latencies_ms)
        p95_index = math.ceil(0.95 * len(ordered_latency)) - 1
        result = {
            "cut_in_messages": len(self.cut_in_messages),
            "response_messages": len(self.response_messages),
            "none_frames": none_frames,
            "slowdown_frames": slowdown_frames,
            "hold_frames": hold_frames,
            "first_active_time_s": first_active_time,
            "first_slowdown_time_s": first_slowdown_time,
            "first_hold_time_s": first_hold_time,
            "source_ids": sorted(source_ids),
            "negative_control_source_frames": negative_source,
            "sequence": sequence,
            "nan_inf_exceptions": 0,
            "latency_ms": {
                "median": statistics.median(ordered_latency),
                "p95": ordered_latency[p95_index],
                "max": ordered_latency[-1],
            },
        }
        Path("/tmp/heven_cut_in_response_runtime_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print("CUT_IN_RESPONSE_RUNTIME_RESULT=" + json.dumps(result, sort_keys=True))


@launch_testing.post_shutdown_test()
class TestCutInResponseShutdown(unittest.TestCase):
    def test_clean_shutdown(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info)
