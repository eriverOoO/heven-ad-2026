"""Live ad_planner replay proving the opt-in roundabout response longitudinal constraint.

The planner is driven into the FollowGlobalPath branch (perception disabled) on a
straight fixture route with an ego reporting 8 m/s. Only the RoundaboutGapResponse
input varies:

  * nothing / inactive       -> no constraint, /ad/planner/roundabout_speed_limit == -1,
                                 governed target speed == baseline
  * RELEASE (active)          -> no constraint, limit == -1, governed target == baseline
  * YIELD (active)            -> cap == ego_speed * sqrt(available / comfortable_stop)
                                 (the response's own comfortable-stop envelope);
                                 governed target drops to that cap, still < baseline
  * HOLD (active)             -> cap 0.0, governed target 0.0, command flips to brake
  * stale (> max age)         -> constraint drops, baseline target returns

The constraint may only ever lower the target, there must remain exactly one
/ad/control/command publisher, and steering must not change between states.
"""

import hashlib
import json
import math
import tempfile
import time
import unittest
from pathlib import Path

from ad_interfaces.msg import PlannerStatus, RoundaboutGapResponse
from ad_morai_interfaces.msg import CollisionArray, CtrlCmd, EgoVehicleStatus
from geometry_msgs.msg import TransformStamped
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
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from std_msgs.msg import Float32
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import yaml
from ament_index_python.packages import get_package_share_directory


DOMAIN_ID = 96
COMMAND_TOPIC = "/ad/control/command"
STATUS_TOPIC = "/ad/planner/status"
LIMIT_TOPIC = "/ad/planner/roundabout_speed_limit"
TARGET_SPEED_TOPIC = "/ad/planner/target_speed"
RESPONSE_TOPIC = "/ad/planning/roundabout_gap_response"
EGO_SPEED_MPS = 8.0
MAX_AGE_S = 0.4
COMFORTABLE_DECEL_MPS2 = 1.8

# YIELD facts: ~40 m to entry -> 34 m available after the 6 m standoff. A genuine
# YIELD (available > comfortable stop distance) whose comfortable-stop cap
# (~11.06 m/s) is well below the 16.25 m/s cruise target, so the governed
# longitudinal target drops without ever commanding a stop.
YIELD_AVAILABLE_M = 34.0
YIELD_COMFORTABLE_STOP_M = EGO_SPEED_MPS * EGO_SPEED_MPS / (2.0 * COMFORTABLE_DECEL_MPS2)
YIELD_EXPECTED_CAP = EGO_SPEED_MPS * math.sqrt(
    YIELD_AVAILABLE_M / YIELD_COMFORTABLE_STOP_M
)
# HOLD facts: comfortable stopping margin already consumed.
HOLD_AVAILABLE_M = 8.0

_QOS_1_RELIABLE = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


def _build_fixture():
    tmp = tempfile.TemporaryDirectory(prefix="ad_planner_roundabout_constraint_")
    root = Path(tmp.name)
    data_dir = root / "data"
    (data_dir / "path").mkdir(parents=True)
    (data_dir / "map").mkdir()

    active_path = data_dir / "path" / "active.txt"
    active_path.write_text(
        "".join(f"{float(x):.1f},0.0,0.0\n" for x in range(-20, 121, 2)),
        encoding="utf-8",
    )
    sha = hashlib.sha256(active_path.read_bytes()).hexdigest()

    corridor = {
        "schema_version": 1,
        "frame_id": "map",
        "primary_lane_sequence_id": "route:0",
        "source_sha256": {
            "global_info.json": "1" * 64,
            "global_path": sha,
            "link_set.json": "2" * 64,
            "node_set.json": "3" * 64,
        },
        "lanes": [
            {
                "lane_sequence_id": "route:0",
                "source_link_ids": ["fixture-link"],
                "adjacent_lane_sequence_ids": {},
                "points": [
                    {
                        "x_m": float(x),
                        "y_m": 0.0,
                        "z_m": 0.0,
                        "yaw_rad": 0.0,
                        "route_s_m": float(x + 20),
                        "curvature_inv_m": 0.0,
                        "left_width_m": 4.0,
                        "right_width_m": 4.0,
                        "speed_limit_mps": 16.0,
                    }
                    for x in range(-20, 121, 2)
                ],
            }
        ],
    }
    (data_dir / "map" / "route_corridor.json").write_text(
        json.dumps(corridor, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    template = yaml.safe_load(
        (Path(get_package_share_directory("ad_planner")) / "config" / "planner.yaml")
        .read_text(encoding="utf-8")
    )
    common_yaml = root / "planner.yaml"
    common_yaml.write_text(yaml.safe_dump(template, sort_keys=False), encoding="utf-8")
    return tmp, data_dir, common_yaml, sha


_FIXTURE_HANDLE, _DATA_DIR, _COMMON_YAML, _PATH_SHA = _build_fixture()


@pytest.mark.launch_test
def generate_test_description():
    return LaunchDescription(
        [
            SetEnvironmentVariable("ROS_DOMAIN_ID", str(DOMAIN_ID)),
            SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
            SetEnvironmentVariable(
                "ROS_LOG_DIR", "/tmp/heven_planner_roundabout_constraint_ros_log"
            ),
            LaunchNode(
                package="ad_planner",
                executable="ad_planner_node",
                name="ad_planner",
                output="screen",
                parameters=[
                    str(_COMMON_YAML),
                    {
                        "data_dir": str(_DATA_DIR),
                        "path_file": "path/active.txt",
                        "route_corridor_file": "map/route_corridor.json",
                        "route_corridor.expected_global_path_sha256": _PATH_SHA,
                        "perception.enabled": False,
                        "local_motion.prediction.mode": "disabled",
                        "enable_roundabout_response_constraint": True,
                        "roundabout_response_max_age_s": MAX_AGE_S,
                    },
                ],
            ),
            launch_testing.actions.ReadyToTest(),
        ]
    )


def _response(action, active, available_m=YIELD_AVAILABLE_M):
    message = RoundaboutGapResponse()
    message.action = action
    message.active = active
    message.reason = RoundaboutGapResponse.REASON_OVERLAP
    message.conflict_zone_id = "kcity_roundabout"
    message.ego_speed_mps = EGO_SPEED_MPS
    message.ego_route_distance_to_entry_m = float(available_m + 6.0)
    message.available_distance_m = float(available_m)
    message.comfortable_stop_distance_m = float(YIELD_COMFORTABLE_STOP_M)
    return message


def _release():
    return _response(RoundaboutGapResponse.ACTION_RELEASE, True)


def _yield():
    return _response(RoundaboutGapResponse.ACTION_YIELD, True)


def _hold():
    return _response(RoundaboutGapResponse.ACTION_HOLD, True, available_m=HOLD_AVAILABLE_M)


def _inactive():
    return _response(RoundaboutGapResponse.ACTION_RELEASE, False)


class TestPlannerRoundaboutConstraint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = Node("planner_roundabout_constraint_probe")
        cls.commands = []
        cls.statuses = []
        cls.limits = []
        cls.targets = []
        cls.node.create_subscription(
            CtrlCmd, COMMAND_TOPIC, lambda m: cls.commands.append(m), 10
        )
        cls.node.create_subscription(
            PlannerStatus, STATUS_TOPIC, lambda m: cls.statuses.append(m), 10
        )
        cls.node.create_subscription(
            Float32, LIMIT_TOPIC, lambda m: cls.limits.append(m), _QOS_1_RELIABLE
        )
        cls.node.create_subscription(
            Float32, TARGET_SPEED_TOPIC, lambda m: cls.targets.append(m), 10
        )
        cls.status_pub = cls.node.create_publisher(
            EgoVehicleStatus, "/ad/vehicle/status", qos_profile_sensor_data
        )
        cls.odom_pub = cls.node.create_publisher(
            Odometry, "/ad/localization/odometry", qos_profile_sensor_data
        )
        cls.collision_pub = cls.node.create_publisher(
            CollisionArray, "/ad/safety/collisions", qos_profile_sensor_data
        )
        cls.grid_pub = cls.node.create_publisher(
            OccupancyGrid, "/ad/perception/occupancy_grid", qos_profile_sensor_data
        )
        cls.ungated_pub = cls.node.create_publisher(
            OccupancyGrid,
            "/ad/viz/perception/occupancy/static_ungated",
            qos_profile_sensor_data,
        )
        cls.response_pub = cls.node.create_publisher(
            RoundaboutGapResponse, RESPONSE_TOPIC, _QOS_1_RELIABLE
        )
        cls.tf = StaticTransformBroadcaster(cls.node)
        transforms = []
        for child in ("map", "base_link"):
            t = TransformStamped()
            t.header.frame_id = "odom"
            t.child_frame_id = child
            t.transform.rotation.w = 1.0
            transforms.append(t)
        cls.tf.sendTransform(transforms)

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def _grid(self, stamp):
        grid = OccupancyGrid()
        grid.header.stamp = stamp
        grid.header.frame_id = "base_link"
        grid.info.resolution = 0.1
        grid.info.width = 200
        grid.info.height = 200
        grid.info.origin.position.x = -4.0
        grid.info.origin.position.y = -10.0
        grid.info.origin.orientation.w = 1.0
        grid.data = [0] * (grid.info.width * grid.info.height)
        return grid

    def _publish_inputs(self):
        stamp = self.node.get_clock().now().to_msg()
        status = EgoVehicleStatus()
        status.header.stamp = stamp
        status.header.frame_id = "base_link"
        status.ctrl_mode = CtrlCmd.CTRL_MODE_AUTO
        status.gear = CtrlCmd.GEAR_DRIVE
        status.velocity.x = EGO_SPEED_MPS
        self.status_pub.publish(status)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.orientation.w = 1.0
        odom.twist.twist.linear.x = EGO_SPEED_MPS
        self.odom_pub.publish(odom)

        collisions = CollisionArray()
        collisions.header.stamp = stamp
        collisions.header.frame_id = "base_link"
        self.collision_pub.publish(collisions)

        self.grid_pub.publish(self._grid(stamp))
        self.ungated_pub.publish(self._grid(stamp))

    def _spin(self, seconds, response=None):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self._publish_inputs()
            if response is not None:
                self.response_pub.publish(response)
            for _ in range(5):
                rclpy.spin_once(self.node, timeout_sec=0.02)

    def _recent_commands(self, count=8):
        return list(self.commands)[-count:]

    def _recent_limit(self):
        self.assertTrue(self.limits, "no roundabout_speed_limit sample seen")
        return self.limits[-1].data

    def _recent_target(self):
        self.assertTrue(self.targets, "no target_speed sample seen")
        return sum(m.data for m in self.targets[-5:]) / len(self.targets[-5:])

    def _mean_throttle(self):
        recent = self._recent_commands()
        self.assertTrue(recent, "no CtrlCmd seen")
        return sum(c.accel - c.brake for c in recent) / len(recent)

    def _mean_abs_steering(self):
        recent = self._recent_commands()
        self.assertTrue(recent, "no CtrlCmd seen")
        return sum(abs(c.steering) for c in recent) / len(recent)

    def test_roundabout_constraint_maps_release_yield_hold(self):
        # Discovery + reach FollowGlobalPath.
        self._spin(6.0)
        self.assertEqual(
            self.node.count_publishers(COMMAND_TOPIC),
            1,
            "there must remain exactly one /ad/control/command publisher",
        )
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            self._spin(0.5)
            if self.statuses and self.statuses[-1].active_behavior == "follow_global_path":
                break
        self.assertEqual(
            self.statuses[-1].active_behavior,
            "follow_global_path",
            "planner never entered FollowGlobalPath",
        )

        # 1. No response yet: no constraint, governed target at the cruise value.
        self._spin(3.0)
        baseline_target = self._recent_target()
        baseline_throttle = self._mean_throttle()
        baseline_steering = self._mean_abs_steering()
        self.assertEqual(self._recent_limit(), -1.0)
        self.assertGreater(
            baseline_target, EGO_SPEED_MPS + 2.0, "expected an unconstrained cruise target"
        )

        # 2. Inactive response is a true no-op.
        self._spin(2.0, response=_inactive())
        self.assertEqual(self._recent_limit(), -1.0)
        self.assertAlmostEqual(self._recent_target(), baseline_target, delta=0.3)

        # 3. RELEASE removes only the roundabout constraint: identical to baseline.
        self._spin(2.0, response=_release())
        self.assertEqual(self._recent_limit(), -1.0)
        release_target = self._recent_target()
        release_steering = self._mean_abs_steering()
        self.assertAlmostEqual(release_target, baseline_target, delta=0.3)

        # 4. YIELD caps the governed target at the response's comfortable-stop
        #    envelope: ego_speed * sqrt(available / comfortable_stop) ~= 11.06 m/s,
        #    below the cruise target and above the 8 m/s ego speed (no stop).
        self._spin(3.0, response=_yield())
        self.assertAlmostEqual(self._recent_limit(), YIELD_EXPECTED_CAP, places=2)
        self.assertGreater(YIELD_EXPECTED_CAP, EGO_SPEED_MPS)
        yield_target = self._recent_target()
        yield_steering = self._mean_abs_steering()
        self.assertAlmostEqual(yield_target, YIELD_EXPECTED_CAP, delta=0.5)
        self.assertLess(yield_target, baseline_target - 1.0)

        # 5. HOLD caps the governed target at 0: the command flips to a brake.
        self._spin(3.0, response=_hold())
        self.assertEqual(self._recent_limit(), 0.0)
        hold_target = self._recent_target()
        hold_throttle = self._mean_throttle()
        hold_steering = self._mean_abs_steering()
        self.assertAlmostEqual(hold_target, 0.0, delta=1e-6)
        self.assertLess(hold_throttle, 0.0)
        self.assertLess(hold_throttle, baseline_throttle)

        # 6. Stale response expires: baseline target returns.
        self._spin(MAX_AGE_S + 3.0)
        self.assertEqual(self._recent_limit(), -1.0)
        self.assertAlmostEqual(self._recent_target(), baseline_target, delta=0.5)

        # No roundabout state ever raised the target above the nominal cruise.
        for label, value in (
            ("release", release_target),
            ("yield", yield_target),
            ("hold", hold_target),
        ):
            self.assertLessEqual(
                value, baseline_target + 1e-3, msg=f"{label} raised the target"
            )

        # Steering must not move between advisory states (straight route).
        for label, value in (
            ("release", release_steering),
            ("yield", yield_steering),
            ("hold", hold_steering),
        ):
            self.assertAlmostEqual(
                value, baseline_steering, delta=1e-6, msg=f"steering moved in {label}"
            )

        Path("/tmp/heven_planner_roundabout_constraint_result.json").write_text(
            json.dumps(
                {
                    "baseline_target": baseline_target,
                    "release_target": release_target,
                    "yield_target": yield_target,
                    "yield_cap_mps": YIELD_EXPECTED_CAP,
                    "hold_target": hold_target,
                    "hold_throttle": hold_throttle,
                    "baseline_abs_steering": baseline_steering,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


@launch_testing.post_shutdown_test()
class TestPlannerRoundaboutConstraintShutdown(unittest.TestCase):
    def test_clean_shutdown(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info)
