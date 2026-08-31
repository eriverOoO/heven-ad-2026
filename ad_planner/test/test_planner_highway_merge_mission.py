"""Live ad_planner replay proving the opt-in Highway Merge Mission Primitive.

The planner runs against the real checksum-verified route corridor and the
shipped source-grounded K-City on-ramp merge geometry, in the FollowGlobalPath
branch (perception disabled). The ego is teleported to successive stations on
`route:0` while the HighwayMergeGapResponse authorization is toggled, and the
mission-state topic (/ad/planner/highway_merge_mission_state) is asserted:

  station  authorization        -> mission state
  400      -                    -> INACTIVE   (before the approach window)
  1000     none                 -> APPROACH
  1150     none                 -> WAITING
  1150     MERGE_READY           -> AUTHORIZED
  1160     WAIT                  -> WAITING     (revoked before commit, no latch)
  1175     MERGE_READY           -> AUTHORIZED
  1235     MERGE_READY           -> COMMITTED   (crossed the ~1213 m commit boundary)
  1255     WAIT                  -> COMMITTED   (post-commit revocation is NOT honored)
  1295     WAIT                  -> COMPLETE    (past merge-complete 1286.15 m)
  1330     -                    -> INACTIVE    (past the exit release)

No steering / lane-change / CtrlCmd is produced by the mission; there must
remain exactly one /ad/control/command publisher, and the mission never injects
a lateral swerve (steering stays small on the near-straight merge stretch).
"""

import hashlib
import json
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

DOMAIN_ID = 98
REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "ad_data"
GLOBAL_PATH = DATA_DIR / "path" / "2026_molit_comp_global_path.txt"
CONFIG = REPO / "ad_planner" / "config" / "planner.yaml"

COMMAND_TOPIC = "/ad/control/command"
STATUS_TOPIC = "/ad/planner/status"
STATE_TOPIC = "/ad/planner/highway_merge_mission_state"
AUTHORIZED_TOPIC = "/ad/planner/highway_merge_authorized"
RESPONSE_TOPIC = "/ad/planning/highway_merge_gap_response"

EGO_SPEED = 8.0
EXPECTED_ZONE = "kcity_highway_onramp"

INACTIVE, APPROACH, WAITING, AUTHORIZED, COMMITTED, COMPLETE = range(6)

# route:0 centreline poses (station -> x, y, yaw), verbatim from
# ad_data/map/route_corridor.json.
ROUTE_GRID = {
    400.0: (-59.95, -71.531, 1.5798),
    1000.0: (4.158, 345.621, -0.0452),
    1150.0: (62.20, 224.99, -1.560),
    1160.0: (62.434, 214.998, -1.57),
    1175.0: (62.49, 199.99, -1.566),
    1235.0: (63.83, 145.02, -1.520),
    1255.0: (65.06, 125.05, -1.512),
    1295.0: (66.80, 85.15, -1.548),
    1330.0: (67.16, 45.20, -1.567),
}

_QOS_1_RELIABLE = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


@pytest.mark.launch_test
def generate_test_description():
    digest = hashlib.sha256(GLOBAL_PATH.read_bytes()).hexdigest()
    return LaunchDescription(
        [
            SetEnvironmentVariable("ROS_DOMAIN_ID", str(DOMAIN_ID)),
            SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
            SetEnvironmentVariable(
                "ROS_LOG_DIR", "/tmp/heven_planner_highway_merge_mission_ros_log"
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
                        "path_file": "path/2026_molit_comp_global_path.txt",
                        "route_corridor_file": "map/route_corridor.json",
                        "route_corridor.expected_global_path_sha256": digest,
                        "perception.enabled": False,
                        "local_motion.prediction.mode": "disabled",
                        "enable_highway_merge_mission": True,
                        "enable_highway_merge_response_integration": True,
                        "highway_merge_response_max_age_s": 0.5,
                    },
                ],
            ),
            launch_testing.actions.ReadyToTest(),
        ]
    )


def _response(action, active):
    message = HighwayMergeGapResponse()
    message.action = action
    message.active = active
    message.reason = HighwayMergeGapResponse.REASON_NONE
    message.merge_zone_id = EXPECTED_ZONE
    message.ego_speed_mps = EGO_SPEED
    message.ego_route_distance_to_merge_m = 40.0
    message.available_distance_m = 34.0
    message.comfortable_stop_distance_m = EGO_SPEED * EGO_SPEED / (2.0 * 1.8)
    message.ego_merge_timing_valid = True
    return message


class TestPlannerHighwayMergeMission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = Node("planner_highway_merge_mission_probe")
        cls.commands = []
        cls.statuses = []
        cls.states = []
        cls.authorized = []
        cls.node.create_subscription(
            CtrlCmd, COMMAND_TOPIC, lambda m: cls.commands.append(m), 10
        )
        cls.node.create_subscription(
            PlannerStatus, STATUS_TOPIC, lambda m: cls.statuses.append(m), 10
        )
        cls.node.create_subscription(
            UInt8, STATE_TOPIC, lambda m: cls.states.append(m.data), _QOS_1_RELIABLE
        )
        cls.node.create_subscription(
            Bool, AUTHORIZED_TOPIC, lambda m: cls.authorized.append(m.data),
            _QOS_1_RELIABLE,
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
        cls.pose = ROUTE_GRID[400.0]

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def _grid(self, stamp):
        grid = OccupancyGrid()
        grid.header.stamp = stamp
        grid.header.frame_id = "base_link"
        grid.info.resolution = 0.2
        grid.info.width = 200
        grid.info.height = 200
        grid.info.origin.position.x = -10.0
        grid.info.origin.position.y = -20.0
        grid.info.origin.orientation.w = 1.0
        grid.data = [0] * (grid.info.width * grid.info.height)
        return grid

    def _publish_inputs(self, response=None):
        import math

        x, y, yaw = self.pose
        stamp = self.node.get_clock().now().to_msg()
        status = EgoVehicleStatus()
        status.header.stamp = stamp
        status.header.frame_id = "base_link"
        status.ctrl_mode = CtrlCmd.CTRL_MODE_AUTO
        status.gear = CtrlCmd.GEAR_DRIVE
        status.velocity.x = EGO_SPEED
        self.status_pub.publish(status)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.orientation.z = math.sin(yaw / 2.0)
        odom.pose.pose.orientation.w = math.cos(yaw / 2.0)
        odom.twist.twist.linear.x = EGO_SPEED
        self.odom_pub.publish(odom)

        collisions = CollisionArray()
        collisions.header.stamp = stamp
        collisions.header.frame_id = "base_link"
        self.collision_pub.publish(collisions)

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
        self.assertTrue(self.states, "no mission-state sample seen")
        return self.states[-1]

    def _mean_abs_steering(self):
        recent = list(self.commands)[-8:]
        self.assertTrue(recent, "no CtrlCmd seen")
        return sum(abs(c.steering) for c in recent) / len(recent)

    def _goto(self, station, response=None, settle=2.0):
        self.pose = ROUTE_GRID[station]
        self._spin(settle, response=response)

    def test_highway_merge_mission_state_machine(self):
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

        results = {}
        steering_samples = []

        # 1. far before the approach window
        self._goto(400.0, settle=3.0)
        results["inactive_far"] = self._state()
        self.assertEqual(results["inactive_far"], INACTIVE)

        # 2. inside the approach window, before the merge zone
        self._goto(1000.0, settle=3.0)
        results["approach"] = self._state()
        self.assertEqual(results["approach"], APPROACH)

        # 3. in the merge decision region, unauthorized
        self._goto(1150.0, settle=3.0)
        results["waiting"] = self._state()
        self.assertEqual(results["waiting"], WAITING)

        # 4. authorized before commit
        self._goto(1150.0, response=_response(HighwayMergeGapResponse.ACTION_MERGE_READY, True), settle=3.0)
        results["authorized"] = self._state()
        self.assertEqual(results["authorized"], AUTHORIZED)

        # 5. authorization revoked before commit -> back to WAITING, no latch
        self._goto(1160.0, response=_response(HighwayMergeGapResponse.ACTION_WAIT, True), settle=3.0)
        results["revoked_before_commit"] = self._state()
        self.assertEqual(results["revoked_before_commit"], WAITING)

        # 6. authorization regained
        self._goto(1175.0, response=_response(HighwayMergeGapResponse.ACTION_MERGE_READY, True), settle=3.0)
        self.assertEqual(self._state(), AUTHORIZED)

        # 7. crosses the commit boundary while authorized
        self._goto(1235.0, response=_response(HighwayMergeGapResponse.ACTION_MERGE_READY, True), settle=3.0)
        results["committed"] = self._state()
        self.assertEqual(results["committed"], COMMITTED)
        steering_samples.append(self._mean_abs_steering())

        # 8. post-commit authorization loss -> STAYS committed
        self._goto(1255.0, response=_response(HighwayMergeGapResponse.ACTION_WAIT, True), settle=3.0)
        results["revoked_after_commit"] = self._state()
        self.assertEqual(results["revoked_after_commit"], COMMITTED)
        self.assertFalse(self.authorized[-1], "upstream authorization should be gone")
        steering_samples.append(self._mean_abs_steering())

        # 9. past merge-complete -> COMPLETE
        self._goto(1295.0, response=_response(HighwayMergeGapResponse.ACTION_WAIT, True), settle=3.0)
        results["complete"] = self._state()
        self.assertEqual(results["complete"], COMPLETE)
        steering_samples.append(self._mean_abs_steering())

        # 10. past the exit release -> INACTIVE
        self._goto(1330.0, settle=3.0)
        results["inactive_after"] = self._state()
        self.assertEqual(results["inactive_after"], INACTIVE)

        # The mission never injects a lateral swerve: steering stays small on
        # the near-straight merge stretch through COMMITTED / COMPLETE.
        for value in steering_samples:
            self.assertLess(value, 0.2, f"mission injected steering: {value}")
        self.assertEqual(self.node.count_publishers(COMMAND_TOPIC), 1)

        Path("/tmp/heven_planner_highway_merge_mission_result.json").write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


@launch_testing.post_shutdown_test()
class TestPlannerHighwayMergeMissionShutdown(unittest.TestCase):
    def test_clean_shutdown(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info)
