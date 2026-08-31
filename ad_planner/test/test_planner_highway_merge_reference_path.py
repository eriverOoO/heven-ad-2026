"""Live ad_planner replay of the online highway-merge lateral reference path.

MANDATORY test-mode distinction (Phase 36): the planner's base / default global
path here is the REAL PRODUCTION route `2026_molit_comp_global_path.txt`. The
new ONLINE highway-merge reference override is what supplies the
route:0:left:1 -> route:0 ramp reference. Reusing the ego-ramp fixture path as
the base would prove only that the fixture works.

The ego is teleported through the real `route:0:left:1` acceleration-lane
samples (~3.9 m off production route:0 near the zone entry) while
HighwayMergeGapResponse is toggled. Asserts:

  * reference-active sequence: INACTIVE/APPROACH -> production path;
    WAITING/AUTHORIZED/COMMITTED (incl. post-commit auth loss) -> merge
    reference; COMPLETE -> back to production path (seamless);
  * with the feature ON at a ramp pose near the zone entry the lateral command
    is SMALL (the merge controller tracks the ramp centerline the ego is on),
    whereas PR #25's `test_planner_highway_merge_ego_ramp.py` recorded ~0.32-
    0.45 rad at the same poses with the feature OFF (Stanley pulling the ego
    the ~3.7 m toward production route:0);
  * WAITING -> AUTHORIZED at a fixed ramp pose does not change the lateral
    command (authorization is longitudinal, not lateral);
  * pre-commit AUTHORIZED -> WAITING keeps the merge reference (no swap to
    route:0); COMMITTED + auth loss keeps it;
  * exactly one /ad/control/command publisher; clean shutdown.

Not a MORAI run (no simulator / grpc here): deterministic ROS replay.
"""

import hashlib
import json
import math
import time
import unittest
from pathlib import Path

from ad_interfaces.msg import HighwayMergeGapResponse, PlannerStatus
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
from std_msgs.msg import Bool, UInt8
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

DOMAIN_ID = 94
REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "ad_data"
GLOBAL_PATH = DATA_DIR / "path" / "2026_molit_comp_global_path.txt"
ROUTE_CORRIDOR = DATA_DIR / "map" / "route_corridor.json"
CONFIG = REPO / "ad_planner" / "config" / "planner.yaml"

COMMAND_TOPIC = "/ad/control/command"
STATUS_TOPIC = "/ad/planner/status"
STATE_TOPIC = "/ad/planner/highway_merge_mission_state"
REFERENCE_TOPIC = "/ad/planner/highway_merge_reference_active"
RESPONSE_TOPIC = "/ad/planning/highway_merge_gap_response"

EGO_SPEED = 8.0
EXPECTED_ZONE = "kcity_highway_onramp"
INACTIVE, APPROACH, WAITING, AUTHORIZED, COMMITTED, COMPLETE = range(6)

MAINLINE = {
    400.0: (-59.95, -71.531, 1.5798),
    1000.0: (4.158, 345.621, -0.0452),
    1295.0: (66.80, 85.15, -1.548),
    1330.0: (67.16, 45.20, -1.567),
}
RAMP = {
    1120.24: (66.0151, 255.0336, -1.568021),
    1150.26: (66.1282, 225.0340, -1.563692),
    1160.26: (66.1883, 215.0342, -1.566422),
    1180.27: (66.2921, 195.0347, -1.563862),
    1220.29: (66.4560, 155.0351, -1.568666),
    1250.30: (66.6083, 125.0355, -1.564761),
}

_QOS_1_RELIABLE = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


def _route0_points():
    doc = json.loads(ROUTE_CORRIDOR.read_text())
    for lane in doc["lanes"]:
        if lane["lane_sequence_id"] == "route:0":
            return lane["points"]
    raise AssertionError


def _lateral_to_route0(x, y):
    pts = _route0_points()
    p = min(pts, key=lambda q: math.hypot(q["x_m"] - x, q["y_m"] - y))
    return -math.sin(p["yaw_rad"]) * (x - p["x_m"]) + math.cos(p["yaw_rad"]) * (y - p["y_m"])


@pytest.mark.launch_test
def generate_test_description():
    digest = hashlib.sha256(GLOBAL_PATH.read_bytes()).hexdigest()
    return LaunchDescription(
        [
            SetEnvironmentVariable("ROS_DOMAIN_ID", str(DOMAIN_ID)),
            SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
            SetEnvironmentVariable(
                "ROS_LOG_DIR", "/tmp/heven_planner_highway_merge_reference_ros_log"
            ),
            LaunchNode(
                package="ad_planner",
                executable="ad_planner_node",
                name="ad_planner",
                output="screen",
                parameters=[
                    str(CONFIG),
                    {
                        "data_dir": str(DATA_DIR),
                        # MANDATORY: production route as the base / default path.
                        "path_file": "path/2026_molit_comp_global_path.txt",
                        "route_corridor_file": "map/route_corridor.json",
                        "route_corridor.expected_global_path_sha256": digest,
                        "perception.enabled": False,
                        "local_motion.prediction.mode": "disabled",
                        "enable_highway_merge_mission": True,
                        "enable_highway_merge_response_integration": True,
                        "enable_highway_merge_reference_path": True,
                        "highway_merge_response_max_age_s": 0.5,
                    },
                ],
            ),
            launch_testing.actions.ReadyToTest(),
        ]
    )


def _response(action, active):
    m = HighwayMergeGapResponse()
    m.action = action
    m.active = active
    m.reason = HighwayMergeGapResponse.REASON_NONE
    m.merge_zone_id = EXPECTED_ZONE
    m.ego_speed_mps = EGO_SPEED
    m.ego_route_distance_to_merge_m = 40.0
    m.available_distance_m = 34.0
    m.comfortable_stop_distance_m = EGO_SPEED * EGO_SPEED / (2.0 * 1.8)
    m.ego_merge_timing_valid = True
    return m


class TestPlannerHighwayMergeReferencePath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = Node("planner_highway_merge_reference_probe")
        cls.commands = []
        cls.statuses = []
        cls.states = []
        cls.reference_active = []
        cls.node.create_subscription(CtrlCmd, COMMAND_TOPIC, lambda m: cls.commands.append(m), 10)
        cls.node.create_subscription(PlannerStatus, STATUS_TOPIC, lambda m: cls.statuses.append(m), 10)
        cls.node.create_subscription(
            UInt8, STATE_TOPIC, lambda m: cls.states.append(m.data), _QOS_1_RELIABLE
        )
        cls.node.create_subscription(
            Bool, REFERENCE_TOPIC, lambda m: cls.reference_active.append(m.data), _QOS_1_RELIABLE
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
            OccupancyGrid, "/ad/viz/perception/occupancy/static_ungated", qos_profile_sensor_data
        )
        cls.response_pub = cls.node.create_publisher(
            HighwayMergeGapResponse, RESPONSE_TOPIC, _QOS_1_RELIABLE
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
        cls.pose = MAINLINE[400.0]

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def _grid(self, stamp):
        g = OccupancyGrid()
        g.header.stamp = stamp
        g.header.frame_id = "base_link"
        g.info.resolution = 0.2
        g.info.width = 200
        g.info.height = 200
        g.info.origin.position.x = -10.0
        g.info.origin.position.y = -20.0
        g.info.origin.orientation.w = 1.0
        g.data = [0] * (g.info.width * g.info.height)
        return g

    def _publish_inputs(self, response=None):
        x, y, yaw = self.pose
        stamp = self.node.get_clock().now().to_msg()
        s = EgoVehicleStatus()
        s.header.stamp = stamp
        s.header.frame_id = "base_link"
        s.ctrl_mode = CtrlCmd.CTRL_MODE_AUTO
        s.gear = CtrlCmd.GEAR_DRIVE
        s.velocity.x = EGO_SPEED
        self.status_pub.publish(s)
        o = Odometry()
        o.header.stamp = stamp
        o.header.frame_id = "odom"
        o.child_frame_id = "base_link"
        o.pose.pose.position.x = x
        o.pose.pose.position.y = y
        o.pose.pose.orientation.z = math.sin(yaw / 2.0)
        o.pose.pose.orientation.w = math.cos(yaw / 2.0)
        o.twist.twist.linear.x = EGO_SPEED
        self.odom_pub.publish(o)
        c = CollisionArray()
        c.header.stamp = stamp
        c.header.frame_id = "base_link"
        self.collision_pub.publish(c)
        self.grid_pub.publish(self._grid(stamp))
        self.ungated_pub.publish(self._grid(stamp))
        if response is not None:
            self.response_pub.publish(response)

    def _spin(self, seconds, response=None):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self._publish_inputs(response)
            for _ in range(5):
                rclpy.spin_once(self.node, timeout_sec=0.02)

    def _state(self):
        self.assertTrue(self.states)
        return self.states[-1]

    def _ref_active(self):
        self.assertTrue(self.reference_active, "no reference-active sample")
        return self.reference_active[-1]

    def _mean_abs_steering(self):
        recent = list(self.commands)[-8:]
        self.assertTrue(recent)
        return sum(abs(c.steering) for c in recent) / len(recent)

    def _goto(self, pose, response=None, settle=3.0):
        self.pose = pose
        self._spin(settle, response=response)

    def test_online_merge_reference_sequence(self):
        self._spin(6.0)
        self.assertEqual(self.node.count_publishers(COMMAND_TOPIC), 1)
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            self._spin(0.5)
            if self.statuses and self.statuses[-1].active_behavior == "follow_global_path":
                break
        self.assertEqual(self.statuses[-1].active_behavior, "follow_global_path")

        results = {"reference_active": {}, "steering": {}, "mission": {}}

        # 1. far upstream on production route:0 -> production path
        self._goto(MAINLINE[400.0])
        self.assertEqual(self._state(), INACTIVE)
        self.assertFalse(self._ref_active())
        results["reference_active"]["inactive_far"] = self._ref_active()

        # 2. approach window on production route:0 -> production path (Phase 24:
        #    do NOT activate the ramp reference in APPROACH)
        self._goto(MAINLINE[1000.0], response=_response(HighwayMergeGapResponse.ACTION_WAIT, True))
        self.assertEqual(self._state(), APPROACH)
        self.assertFalse(self._ref_active())
        results["reference_active"]["approach"] = self._ref_active()

        # 3. genuine ramp pose, WAITING -> merge reference (lateral follows the
        #    ramp; longitudinal WAIT owns merge permission)
        self._goto(RAMP[1120.24])
        self.assertEqual(self._state(), WAITING)
        self.assertTrue(self._ref_active(), "ramp ego WAITING must use the merge reference")
        results["reference_active"]["waiting"] = self._ref_active()
        steer_waiting = self._mean_abs_steering()
        # ego is ~3.9 m off production route:0 here; with the merge reference the
        # controller tracks the ramp centerline it is ON -> small command.
        results["ego_lateral_to_route0_at_zone_entry_m"] = round(
            abs(_lateral_to_route0(*RAMP[1120.24][:2])), 3
        )
        self.assertLess(
            steer_waiting, 0.15,
            "merge reference should keep the ramp ego near its own centerline",
        )
        results["steering"]["waiting"] = round(steer_waiting, 4)

        # 4. WAITING -> AUTHORIZED at the SAME ramp pose: the lateral reference
        #    must not change (authorization is longitudinal). (Phase 32)
        self._goto(RAMP[1120.24], response=_response(HighwayMergeGapResponse.ACTION_MERGE_READY, True))
        self.assertEqual(self._state(), AUTHORIZED)
        self.assertTrue(self._ref_active())
        steer_authorized = self._mean_abs_steering()
        self.assertLess(
            abs(steer_authorized - steer_waiting), 0.05,
            "authorization changed the lateral command",
        )
        results["steering"]["authorized_same_pose"] = round(steer_authorized, 4)

        # 5. progress + AUTHORIZED
        self._goto(RAMP[1150.26], response=_response(HighwayMergeGapResponse.ACTION_MERGE_READY, True))
        self.assertEqual(self._state(), AUTHORIZED)
        self.assertTrue(self._ref_active())

        # 6. pre-commit revocation AUTHORIZED -> WAITING: keep the merge
        #    reference (Phase 33)
        self._goto(RAMP[1160.26], response=_response(HighwayMergeGapResponse.ACTION_WAIT, True))
        self.assertEqual(self._state(), WAITING)
        self.assertTrue(self._ref_active(), "pre-commit revocation must not drop the merge reference")
        results["reference_active"]["revoked_before_commit"] = self._ref_active()

        # 7. AUTHORIZED again, cross the commit boundary -> COMMITTED
        self._goto(RAMP[1180.27], response=_response(HighwayMergeGapResponse.ACTION_MERGE_READY, True))
        self.assertEqual(self._state(), AUTHORIZED)
        self._goto(RAMP[1220.29], response=_response(HighwayMergeGapResponse.ACTION_MERGE_READY, True))
        self.assertEqual(self._state(), COMMITTED)
        self.assertTrue(self._ref_active())
        results["reference_active"]["committed"] = self._ref_active()

        # 8. COMMITTED + upstream WAIT -> merge reference retained (Phase 34)
        self._goto(RAMP[1250.30], response=_response(HighwayMergeGapResponse.ACTION_WAIT, True))
        self.assertEqual(self._state(), COMMITTED)
        self.assertTrue(self._ref_active(), "post-commit auth loss must not reverse the merge path")
        results["reference_active"]["revoked_after_commit"] = self._ref_active()
        steer_committed = self._mean_abs_steering()
        results["steering"]["committed"] = round(steer_committed, 4)

        # 9. past merge complete -> COMPLETE, seamless handoff back to production
        self._goto(MAINLINE[1295.0], response=_response(HighwayMergeGapResponse.ACTION_WAIT, True))
        self.assertEqual(self._state(), COMPLETE)
        self.assertFalse(self._ref_active(), "COMPLETE returns to the production path")
        results["reference_active"]["complete"] = self._ref_active()
        steer_complete = self._mean_abs_steering()
        results["steering"]["complete"] = round(steer_complete, 4)
        # ego is on route:0 here; the handoff is seamless -> small command
        self.assertLess(steer_complete, 0.2)

        # 10. past the exit release -> INACTIVE, production path
        self._goto(MAINLINE[1330.0])
        self.assertEqual(self._state(), INACTIVE)
        self.assertFalse(self._ref_active())
        results["reference_active"]["inactive_after"] = self._ref_active()

        self.assertEqual(self.node.count_publishers(COMMAND_TOPIC), 1)
        for c in self.commands:
            self.assertTrue(math.isfinite(c.steering))

        Path("/tmp/heven_planner_highway_merge_reference_result.json").write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


@launch_testing.post_shutdown_test()
class TestPlannerHighwayMergeReferencePathShutdown(unittest.TestCase):
    def test_clean_shutdown(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info)
