"""Live ad_planner replay proving the opt-in cut-in longitudinal speed constraint.

The planner is driven into the FollowGlobalPath branch (perception disabled) on a
straight fixture route with a stationary-ish ego. Only the CutInResponse input
varies:

  * nothing / NONE        -> no constraint, /ad/planner/cut_in_speed_limit == -1
  * SLOWDOWN 3.0 m/s       -> cap 3.0, throttle strictly below the NONE baseline
  * HOLD                   -> cap 0.0, throttle collapses to a brake
  * stale (> max age)      -> constraint drops, baseline throttle returns

This is the first task that changes planner longitudinal behaviour, so the test
also asserts there is still exactly one /ad/control/command publisher and that
the constraint only ever lowers the target.
"""

import hashlib
import json
import math
import tempfile
import time
import unittest
from pathlib import Path

from ad_interfaces.msg import CutInResponse, PlannerStatus
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


DOMAIN_ID = 95
COMMAND_TOPIC = "/ad/control/command"
STATUS_TOPIC = "/ad/planner/status"
LIMIT_TOPIC = "/ad/planner/cut_in_speed_limit"
RESPONSE_TOPIC = "/ad/planning/cut_in_response"
# Ego reports well above both the SLOWDOWN cap (3.0) and the HOLD cap (0.0), so
# the capped target flips the longitudinal PID from accelerate to brake and the
# effect is unambiguous in the command stream.
EGO_SPEED_MPS = 8.0
MAX_AGE_S = 0.4

_QOS_1_RELIABLE = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


def _build_fixture():
    tmp = tempfile.TemporaryDirectory(prefix="ad_planner_cut_in_constraint_")
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
                "ROS_LOG_DIR", "/tmp/heven_planner_cut_in_constraint_ros_log"
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
                        "enable_cut_in_response_constraint": True,
                        "cut_in_response_max_age_s": MAX_AGE_S,
                    },
                ],
            ),
            launch_testing.actions.ReadyToTest(),
        ]
    )


def _slowdown(speed_mps):
    message = CutInResponse()
    message.action = CutInResponse.ACTION_SLOWDOWN
    message.active = True
    message.reason = CutInResponse.REASON_APPROACHING_ENTRY
    message.requested_max_speed_valid = True
    message.requested_max_speed_mps = float(speed_mps)
    return message


def _hold():
    message = CutInResponse()
    message.action = CutInResponse.ACTION_HOLD
    message.active = True
    message.reason = CutInResponse.REASON_COLLISION_CONFLICT
    message.requested_max_speed_valid = True
    message.requested_max_speed_mps = 0.0
    return message


def _none():
    message = CutInResponse()
    message.action = CutInResponse.ACTION_NONE
    message.active = False
    message.requested_max_speed_valid = False
    return message


class TestPlannerCutInConstraint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = Node("planner_cut_in_constraint_probe")
        cls.commands = []
        cls.statuses = []
        cls.limits = []
        cls.node.create_subscription(
            CtrlCmd, COMMAND_TOPIC, lambda m: cls.commands.append(m), 10
        )
        cls.node.create_subscription(
            PlannerStatus, STATUS_TOPIC, lambda m: cls.statuses.append(m), 10
        )
        cls.node.create_subscription(
            Float32, LIMIT_TOPIC, lambda m: cls.limits.append(m), _QOS_1_RELIABLE
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
            CutInResponse, RESPONSE_TOPIC, _QOS_1_RELIABLE
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
        self.assertTrue(self.limits, "no cut_in_speed_limit sample seen")
        return self.limits[-1].data

    def _mean_throttle(self):
        recent = self._recent_commands()
        self.assertTrue(recent, "no CtrlCmd seen")
        return sum(c.accel - c.brake for c in recent) / len(recent)

    def test_constraint_only_ever_lowers_the_target(self):
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

        # 1. No CutInResponse yet: constraint absent, ego below the cruise
        #    target, so the longitudinal PID accelerates.
        self._spin(2.0)
        baseline = self._mean_throttle()
        self.assertEqual(self._recent_limit(), -1.0)
        self.assertGreater(baseline, 0.3, "expected forward throttle at baseline")

        # 2. NONE is a true no-op.
        self._spin(2.0, response=_none())
        self.assertEqual(self._recent_limit(), -1.0)
        self.assertAlmostEqual(self._mean_throttle(), baseline, delta=0.2)

        # 3. SLOWDOWN caps the target below the ego speed: the PID now brakes.
        self._spin(2.5, response=_slowdown(3.0))
        self.assertAlmostEqual(self._recent_limit(), 3.0, places=3)
        slowdown_throttle = self._mean_throttle()
        self.assertLess(slowdown_throttle, baseline - 0.3)

        # 4. HOLD caps the target at 0: braking is at least as strong.
        self._spin(2.5, response=_hold())
        self.assertEqual(self._recent_limit(), 0.0)
        hold_throttle = self._mean_throttle()
        self.assertLessEqual(hold_throttle, slowdown_throttle + 1e-6)
        self.assertLess(hold_throttle, 0.0)

        # 5. Stale response expires: baseline returns.
        self._spin(MAX_AGE_S + 3.0)
        self.assertEqual(self._recent_limit(), -1.0)
        self.assertAlmostEqual(self._mean_throttle(), baseline, delta=0.25)

        Path("/tmp/heven_planner_cut_in_constraint_result.json").write_text(
            json.dumps(
                {
                    "baseline_throttle": baseline,
                    "slowdown_throttle": slowdown_throttle,
                    "hold_throttle": hold_throttle,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


@launch_testing.post_shutdown_test()
class TestPlannerCutInConstraintShutdown(unittest.TestCase):
    def test_clean_shutdown(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info)
