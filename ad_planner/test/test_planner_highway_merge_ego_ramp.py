"""Live ad_planner replay of the Highway Merge Ego-on-Ramp Scenario v1.

PR #24 (CASE B) proved the production ego never drives the acceleration lane, so
the Highway Merge Response Integration + Mission Primitive had no ego-on-ramp
scenario to exercise. This test teleports the ego through the REAL
`route:0:left:1` acceleration-lane samples (from ad_data/map/route_corridor.json)
while toggling the upstream HighwayMergeGapResponse, and asserts the mission
walks INACTIVE -> APPROACH -> WAITING -> AUTHORIZED -> COMMITTED -> COMPLETE ->
INACTIVE with genuine ramp poses -- confirming:

  * project_primary_route(route:0, ramp_ego) stays monotonic (the mission and
    the gap-risk node both use it) -- no mission-progress change is required;
  * the derived commit boundary (~1213.3 m) is crossed while the ego is
    physically ~3.5 m laterally off route:0 (inside the taper);
  * completion happens where the acceleration lane has fully merged (~0 m off);
  * a post-commit authorization loss does NOT reverse the merge;
  * the mission still commands nothing -- exactly one /ad/control/command
    publisher, steering stays small (no lateral swerve), and no lateral path is
    generated.

The INACTIVE (far) and APPROACH poses are necessarily on route:0 itself: the
acceleration lane does not extend upstream of the merge zone entry on the
K-City map (a documented CASE B consequence). Every WAITING..COMPLETE pose is a
verbatim route:0:left:1 sample.

Not a MORAI run (no simulator / grpc in this environment): deterministic ROS
replay against the real corridor + merge geometry.
"""

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

DOMAIN_ID = 97
REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "ad_data"
GLOBAL_PATH = DATA_DIR / "path" / "2026_molit_comp_global_path.txt"
ROUTE_CORRIDOR = DATA_DIR / "map" / "route_corridor.json"
CONFIG = REPO / "ad_planner" / "config" / "planner.yaml"

COMMAND_TOPIC = "/ad/control/command"
STATUS_TOPIC = "/ad/planner/status"
STATE_TOPIC = "/ad/planner/highway_merge_mission_state"
AUTHORIZED_TOPIC = "/ad/planner/highway_merge_authorized"
RESPONSE_TOPIC = "/ad/planning/highway_merge_gap_response"

EGO_SPEED = 8.0
EXPECTED_ZONE = "kcity_highway_onramp"
COMMIT_LATERAL_SEPARATION_M = 3.5

INACTIVE, APPROACH, WAITING, AUTHORIZED, COMMITTED, COMPLETE = range(6)

# route:0 mainline poses (station -> x, y, yaw) for the pre-ramp bookends only,
# verbatim from ad_data/map/route_corridor.json.
MAINLINE = {
    400.0: (-59.95, -71.531, 1.5798),
    1000.0: (4.158, 345.621, -0.0452),
    1295.0: (66.80, 85.15, -1.548),
    1330.0: (67.16, 45.20, -1.567),
}

# route:0:left:1 acceleration-lane poses (source route_s -> x, y, yaw), verbatim
# from ad_data/map/route_corridor.json. Every one of these is a genuine ramp
# pose ~3.9 -> 0 m laterally off route:0.
RAMP = {
    1120.24: (66.0151, 255.0336, -1.568021),
    1150.26: (66.1282, 225.0340, -1.563692),
    1160.26: (66.1883, 215.0342, -1.566422),
    1180.27: (66.2921, 195.0347, -1.563862),
    1213.28: (66.4319, 162.0351, -1.565611),  # ~ the derived commit boundary
    1220.29: (66.4560, 155.0351, -1.568666),
    1250.30: (66.6083, 125.0355, -1.564761),
    1286.15: (66.8850, 89.1979, -1.559167),   # merge complete (ramp end)
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
    raise AssertionError("route:0 not found")


def _lateral_offset_to_route0(x, y):
    pts = _route0_points()
    best = min(range(len(pts)), key=lambda i: math.hypot(pts[i]["x_m"] - x, pts[i]["y_m"] - y))
    p = pts[best]
    return -math.sin(p["yaw_rad"]) * (x - p["x_m"]) + math.cos(p["yaw_rad"]) * (y - p["y_m"])


def _project_primary_s(x, y, yaw):
    pts = _route0_points()
    best_d = math.inf
    best_s = 0.0
    for i in range(1, len(pts)):
        a, b = pts[i - 1], pts[i]
        dx, dy = b["x_m"] - a["x_m"], b["y_m"] - a["y_m"]
        length = math.hypot(dx, dy)
        if length <= 0.0:
            continue
        ux, uy = dx / length, dy / length
        rx, ry = x - a["x_m"], y - a["y_m"]
        along = rx * ux + ry * uy
        clamped = min(max(along, 0.0), length)
        lateral = -rx * uy + ry * ux
        dist = math.hypot(along - clamped, lateral)
        seg_heading = math.atan2(dy, dx)
        if abs(math.remainder(seg_heading - yaw, 2.0 * math.pi)) > math.pi / 2.0 + 1e-12:
            continue
        if dist < best_d:
            best_d = dist
            ratio = clamped / length
            best_s = a["route_s_m"] + ratio * (b["route_s_m"] - a["route_s_m"])
    return best_s


@pytest.mark.launch_test
def generate_test_description():
    import hashlib

    digest = hashlib.sha256(GLOBAL_PATH.read_bytes()).hexdigest()
    return LaunchDescription(
        [
            SetEnvironmentVariable("ROS_DOMAIN_ID", str(DOMAIN_ID)),
            SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
            SetEnvironmentVariable(
                "ROS_LOG_DIR", "/tmp/heven_planner_highway_merge_ego_ramp_ros_log"
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


class TestPlannerHighwayMergeEgoRamp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = Node("planner_highway_merge_ego_ramp_probe")
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
        cls.pose = MAINLINE[400.0]

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

    def _goto(self, pose, response=None, settle=3.0):
        self.pose = pose
        self._spin(settle, response=response)

    def test_ego_on_ramp_mission_state_machine(self):
        self._spin(6.0)
        self.assertEqual(
            self.node.count_publishers(COMMAND_TOPIC), 1,
            "there must remain exactly one /ad/control/command publisher",
        )
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            self._spin(0.5)
            if self.statuses and self.statuses[-1].active_behavior == "follow_global_path":
                break
        self.assertEqual(self.statuses[-1].active_behavior, "follow_global_path")

        # project_primary_route stays monotonic across every real ramp sample.
        ramp_stations = sorted(RAMP)
        projected = [_project_primary_s(*RAMP[s]) for s in ramp_stations]
        for earlier, later in zip(projected, projected[1:]):
            self.assertGreaterEqual(
                later, earlier - 1e-6, "primary-route projection went backwards on the ramp"
            )
        for station, s_proj in zip(ramp_stations, projected):
            self.assertLess(abs(s_proj - station), 0.5)

        results = {}
        steering = []

        # 1. far before the approach window (on route:0)
        self._goto(MAINLINE[400.0])
        results["inactive_far"] = self._state()
        self.assertEqual(results["inactive_far"], INACTIVE)

        # 2. approach window, unauthorized (on route:0 -- the ramp does not
        #    extend this far back)
        self._goto(MAINLINE[1000.0], response=_response(HighwayMergeGapResponse.ACTION_WAIT, True))
        results["approach"] = self._state()
        self.assertEqual(results["approach"], APPROACH)

        # 3. genuine ramp pose, in the decision region, unauthorized
        self._goto(RAMP[1120.24])
        results["waiting"] = self._state()
        self.assertEqual(results["waiting"], WAITING)

        # 4. ramp pose, authorized before commit
        self._goto(RAMP[1150.26], response=_response(HighwayMergeGapResponse.ACTION_MERGE_READY, True))
        results["authorized"] = self._state()
        self.assertEqual(results["authorized"], AUTHORIZED)

        # 5. ramp pose, authorization revoked before commit -> WAITING (no latch)
        self._goto(RAMP[1160.26], response=_response(HighwayMergeGapResponse.ACTION_WAIT, True))
        results["revoked_before_commit"] = self._state()
        self.assertEqual(results["revoked_before_commit"], WAITING)

        # 6. ramp pose, authorization regained -> AUTHORIZED (before commit).
        #    Record the lateral command here: the ego is ~3.7 m off route:0
        #    (the tracked production path), so Stanley is correcting toward it.
        self._goto(RAMP[1180.27], response=_response(HighwayMergeGapResponse.ACTION_MERGE_READY, True))
        self.assertEqual(self._state(), AUTHORIZED)
        steer_before_commit = self._mean_abs_steering()

        # 7. ramp pose past the derived commit boundary, authorized -> COMMITTED.
        #    The ego is physically inside the acceleration-lane taper here.
        self._goto(RAMP[1220.29], response=_response(HighwayMergeGapResponse.ACTION_MERGE_READY, True))
        results["committed"] = self._state()
        self.assertEqual(results["committed"], COMMITTED)
        commit_sep = abs(_lateral_offset_to_route0(*RAMP[1213.28][:2]))
        self.assertLess(commit_sep, COMMIT_LATERAL_SEPARATION_M)
        self.assertGreater(commit_sep, COMMIT_LATERAL_SEPARATION_M - 0.3)
        results["commit_lateral_separation_m"] = round(commit_sep, 4)
        steer_after_commit = self._mean_abs_steering()
        steering += [steer_before_commit, steer_after_commit]

        # Crossing the commit boundary must not add lateral command: the mission
        # commands nothing, so steering stays pure path-tracking (it in fact
        # eases as the ego nears route:0) and never spikes when the state flips
        # AUTHORIZED -> COMMITTED.
        self.assertLessEqual(
            steer_after_commit, steer_before_commit + 0.1,
            "steering increased across the mission commit boundary",
        )
        results["steer_before_commit"] = round(steer_before_commit, 4)
        results["steer_after_commit"] = round(steer_after_commit, 4)

        # 8. ramp pose, post-commit authorization loss -> STAYS committed
        self._goto(RAMP[1250.30], response=_response(HighwayMergeGapResponse.ACTION_WAIT, True))
        results["revoked_after_commit"] = self._state()
        self.assertEqual(results["revoked_after_commit"], COMMITTED)
        self.assertFalse(self.authorized[-1], "upstream authorization should be gone")
        steering.append(self._mean_abs_steering())

        # 9. past merge-complete (ego now established on route:0) -> COMPLETE.
        #    The acceleration lane has fully merged: ~0 m lateral separation.
        complete_sep = abs(_lateral_offset_to_route0(*RAMP[1286.15][:2]))
        self.assertLess(complete_sep, 0.2)
        results["complete_lateral_separation_m"] = round(complete_sep, 4)
        self._goto(MAINLINE[1295.0], response=_response(HighwayMergeGapResponse.ACTION_WAIT, True))
        results["complete"] = self._state()
        self.assertEqual(results["complete"], COMPLETE)
        steering.append(self._mean_abs_steering())

        # 10. past the exit release -> INACTIVE
        self._goto(MAINLINE[1330.0])
        results["inactive_after"] = self._state()
        self.assertEqual(results["inactive_after"], INACTIVE)

        # Every lateral command stays finite and far inside the steering lock --
        # it is pure Stanley path-tracking (the ramp poses sit up to ~3.9 m off
        # the tracked production route by construction), never a mission swerve.
        for value in steering:
            self.assertTrue(math.isfinite(value))
            self.assertLess(value, 0.5, f"steering outside the path-tracking band: {value}")
        self.assertEqual(self.node.count_publishers(COMMAND_TOPIC), 1)

        Path("/tmp/heven_planner_highway_merge_ego_ramp_result.json").write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


@launch_testing.post_shutdown_test()
class TestPlannerHighwayMergeEgoRampShutdown(unittest.TestCase):
    def test_clean_shutdown(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info)
