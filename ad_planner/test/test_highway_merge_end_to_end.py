"""Highway Merge End-to-End Execution Validation v1.

Composed PLANNING-SIDE chain, all REAL nodes downstream of the canonical
DynamicObjectRisk boundary:

  synthetic DynamicObjectRiskArray
    -> ad_highway_merge_gap_risk        (real)
    -> ad_highway_merge_gap_response    (real)
    -> ad_planner  (real; highway merge response integration + mission +
                    online reference path all enabled; PRODUCTION base path
                    ad_data/path/2026_molit_comp_global_path.txt)
    -> existing FollowGlobalPath -> existing PathTrackingController(s)
    -> the single /ad/control/command publisher

NOT a full perception-to-control E2E: the boundary is DynamicObjectRiskArray
(tracking / prediction / Dynamic Object Risk producer have their own runtime
validation and the real producer OOMs on this host). NOT a MORAI run (no
simulator / gRPC here).

The test drives a deterministic target-mainline traffic script and the real
`route:0:left:1` acceleration-lane ego poses from PR #25, and asserts the
composed outputs (it never recomputes a policy equation): unsafe rear ->
not MERGE_READY; zero relevant objects -> MERGE_READY; the mission walks
WAITING -> AUTHORIZED -> WAITING -> AUTHORIZED -> COMMITTED -> COMMITTED ->
COMPLETE -> INACTIVE; the online ramp reference is active throughout ramp
participation and hands back to the production path at COMPLETE; pre/post-commit
authorization loss follows the intended semantics; a cut-in cap still wins;
exactly one /ad/control/command publisher.
"""

import json
import math
import statistics
import time
import unittest
from pathlib import Path

from ad_interfaces.msg import (
    CutInResponse,
    DynamicObjectRisk,
    DynamicObjectRiskArray,
    DynamicObjectRiskState,
    HighwayMergeGapResponse,
    HighwayMergeGapRiskArray,
    PlannerStatus,
)
from ad_morai_interfaces.msg import CollisionArray, CtrlCmd, EgoVehicleStatus
from builtin_interfaces.msg import Duration
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
from std_msgs.msg import Bool, Float32, UInt8
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

import hashlib

DOMAIN_ID = 93
REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "ad_data"
GLOBAL_PATH = DATA_DIR / "path" / "2026_molit_comp_global_path.txt"
ROUTE_CORRIDOR = DATA_DIR / "map" / "route_corridor.json"
PLANNER_CONFIG = REPO / "ad_planner" / "config" / "planner.yaml"
RISK_CONFIG = REPO / "ad_planner" / "config" / "highway_merge_gap_risk.yaml"
RESPONSE_CONFIG = REPO / "ad_planner" / "config" / "highway_merge_gap_response.yaml"

RISK_IN_TOPIC = "/ad/planning/dynamic_object_risks"
GAP_TOPIC = "/ad/planning/highway_merge_gap_risks"
RESPONSE_TOPIC = "/ad/planning/highway_merge_gap_response"
COMMAND_TOPIC = "/ad/control/command"
STATUS_TOPIC = "/ad/planner/status"
STATE_TOPIC = "/ad/planner/highway_merge_mission_state"
REFERENCE_TOPIC = "/ad/planner/highway_merge_reference_active"
AUTHORIZED_TOPIC = "/ad/planner/highway_merge_authorized"
MERGE_LIMIT_TOPIC = "/ad/planner/highway_merge_speed_limit"
CUT_IN_TOPIC = "/ad/planning/cut_in_response"

EGO_SPEED = 8.0
EXPECTED_ZONE = "kcity_highway_onramp"
S_MERGE_COMPLETE = 1286.1546454384027
INACTIVE, APPROACH, WAITING, AUTHORIZED, COMMITTED, COMPLETE = range(6)
HORIZON_S = 24.0
DT_S = 1.0

_QOS_1_RELIABLE = QoSProfile(
    depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE
)
_QOS_10_RELIABLE = QoSProfile(
    depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE
)


def _lane(sid):
    doc = json.loads(ROUTE_CORRIDOR.read_text())
    for lane in doc["lanes"]:
        if lane["lane_sequence_id"] == sid:
            return lane["points"]
    raise AssertionError(sid)


_RAMP = _lane("route:0:left:1")
_R0 = _lane("route:0")

# real route:0:left:1 poses (source route_s -> x, y, yaw), verbatim
MAINLINE = {
    400.0: (-59.95, -71.531, 1.5798),
    1000.0: (4.158, 345.621, -0.0452),
    1295.0: (66.80, 85.15, -1.548),
    1330.0: (67.16, 45.20, -1.567),
}


def _ramp_pose(source_route_s):
    for p in _RAMP:
        if p["route_s_m"] >= source_route_s - 1e-9:
            return p["x_m"], p["y_m"], p["yaw_rad"]
    p = _RAMP[-1]
    return p["x_m"], p["y_m"], p["yaw_rad"]


def _route0_pose(route_s):
    for p in _R0:
        if p["route_s_m"] >= route_s:
            return p["x_m"], p["y_m"], p["yaw_rad"]
    p = _R0[-1]
    return p["x_m"], p["y_m"], p["yaw_rad"]


def _lateral_to_route0(x, y):
    p = min(_R0, key=lambda q: math.hypot(q["x_m"] - x, q["y_m"] - y))
    return -math.sin(p["yaw_rad"]) * (x - p["x_m"]) + math.cos(p["yaw_rad"]) * (y - p["y_m"])


def _duration(seconds):
    d = Duration()
    d.sec = int(seconds)
    d.nanosec = int(round((seconds - int(seconds)) * 1e9))
    return d


def _to_body(px, py, ex, ey, eyaw):
    dx, dy = px - ex, py - ey
    c, s = math.cos(eyaw), math.sin(eyaw)
    return c * dx + s * dy, -s * dx + c * dy


def _mainline_object(uuid_byte, ego_pose, route_s0, speed_mps, horizon_s=HORIZON_S):
    ex, ey, eyaw = ego_pose
    risk = DynamicObjectRisk()
    risk.object_id.uuid = [uuid_byte] + [0] * 15
    risk.classification = 1
    risk.classification_probability = 0.9
    risk.existence_probability = 0.9
    cx, cy, cyaw = _route0_pose(route_s0)
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
        px, py, _ = _route0_pose(route_s0 + speed_mps * t)
        rx, ry = _to_body(px, py, ex, ey, eyaw)
        st = DynamicObjectRiskState()
        st.time_from_start = _duration(t)
        st.x_rel_m = rx - EGO_SPEED * t
        st.y_rel_m = ry
        risk.predicted_states.append(st)
        t += DT_S
    return risk


@pytest.mark.launch_test
def generate_test_description():
    digest = hashlib.sha256(GLOBAL_PATH.read_bytes()).hexdigest()
    common = {
        "data_dir": str(DATA_DIR),
        "route_corridor_file": "map/route_corridor.json",
        "route_corridor.expected_global_path_sha256": digest,
    }
    return LaunchDescription(
        [
            SetEnvironmentVariable("ROS_DOMAIN_ID", str(DOMAIN_ID)),
            SetEnvironmentVariable("ROS_LOCALHOST_ONLY", "1"),
            SetEnvironmentVariable(
                "ROS_LOG_DIR", "/tmp/heven_highway_merge_e2e_ros_log"
            ),
            LaunchNode(
                package="ad_planner",
                executable="ad_highway_merge_gap_risk_node",
                name="ad_highway_merge_gap_risk",
                output="screen",
                parameters=[
                    str(RISK_CONFIG),
                    {**common, "maximum_input_age_s": 5.0, "maximum_odometry_skew_s": 5.0},
                ],
            ),
            LaunchNode(
                package="ad_planner",
                executable="ad_highway_merge_gap_response_node",
                name="ad_highway_merge_gap_response",
                output="screen",
                parameters=[str(RESPONSE_CONFIG), {"maximum_input_age_s": 5.0}],
            ),
            LaunchNode(
                package="ad_planner",
                executable="ad_planner_node",
                name="ad_planner",
                output="screen",
                parameters=[
                    str(PLANNER_CONFIG),
                    {
                        **common,
                        "path_file": "path/2026_molit_comp_global_path.txt",
                        "perception.enabled": False,
                        "local_motion.prediction.mode": "disabled",
                        "enable_highway_merge_response_integration": True,
                        "enable_highway_merge_mission": True,
                        "enable_highway_merge_reference_path": True,
                        "enable_cut_in_response_constraint": True,
                        "highway_merge_response_max_age_s": 0.5,
                        "cut_in_response_max_age_s": 0.5,
                    },
                ],
            ),
            launch_testing.actions.ReadyToTest(),
        ]
    )


class TestHighwayMergeEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = Node("highway_merge_e2e_probe")
        cls.commands = []
        cls.statuses = []
        cls.states = []
        cls.reference_active = []
        cls.authorized = []
        cls.merge_limit = []
        cls.gap_frames = []
        cls.responses = []
        cls.node.create_subscription(CtrlCmd, COMMAND_TOPIC, lambda m: cls.commands.append(m), 10)
        cls.node.create_subscription(PlannerStatus, STATUS_TOPIC, lambda m: cls.statuses.append(m), 10)
        cls.node.create_subscription(UInt8, STATE_TOPIC, lambda m: cls.states.append(m.data), _QOS_1_RELIABLE)
        cls.node.create_subscription(Bool, REFERENCE_TOPIC, lambda m: cls.reference_active.append(m.data), _QOS_1_RELIABLE)
        cls.node.create_subscription(Bool, AUTHORIZED_TOPIC, lambda m: cls.authorized.append(m.data), _QOS_1_RELIABLE)
        cls.node.create_subscription(Float32, MERGE_LIMIT_TOPIC, lambda m: cls.merge_limit.append(m.data), _QOS_1_RELIABLE)
        cls.node.create_subscription(HighwayMergeGapRiskArray, GAP_TOPIC, lambda m: cls.gap_frames.append(m), _QOS_1_RELIABLE)
        cls.node.create_subscription(HighwayMergeGapResponse, RESPONSE_TOPIC, lambda m: cls.responses.append(m), _QOS_1_RELIABLE)

        # Reliable publishers satisfy both the planner's sensor-data subs and
        # the gap-risk node's reliable odometry sub.
        cls.status_pub = cls.node.create_publisher(EgoVehicleStatus, "/ad/vehicle/status", _QOS_10_RELIABLE)
        cls.odom_pub = cls.node.create_publisher(Odometry, "/ad/localization/odometry", _QOS_10_RELIABLE)
        cls.collision_pub = cls.node.create_publisher(CollisionArray, "/ad/safety/collisions", _QOS_10_RELIABLE)
        cls.grid_pub = cls.node.create_publisher(OccupancyGrid, "/ad/perception/occupancy_grid", _QOS_10_RELIABLE)
        cls.ungated_pub = cls.node.create_publisher(OccupancyGrid, "/ad/viz/perception/occupancy/static_ungated", _QOS_10_RELIABLE)
        cls.risk_pub = cls.node.create_publisher(DynamicObjectRiskArray, RISK_IN_TOPIC, _QOS_1_RELIABLE)
        cls.cut_in_pub = cls.node.create_publisher(CutInResponse, CUT_IN_TOPIC, _QOS_1_RELIABLE)

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
        cls.objects_fn = staticmethod(lambda pose: [])
        cls.cut_in = None
        cls.publish_risk = True

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

    def _publish(self):
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

        if self.publish_risk:
            arr = DynamicObjectRiskArray()
            arr.header.stamp = stamp
            arr.header.frame_id = "base_link"
            arr.objects = self.objects_fn(self.pose)
            self.risk_pub.publish(arr)
        if self.cut_in is not None:
            self.cut_in.header.stamp = stamp
            self.cut_in_pub.publish(self.cut_in)

    def _spin(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self._publish()
            for _ in range(5):
                rclpy.spin_once(self.node, timeout_sec=0.02)

    def _latest(self, buf, name):
        self.assertTrue(buf, f"no {name} sample")
        return buf[-1]

    def _response_action(self):
        r = self._latest(self.responses, "response")
        if not r.active:
            return "INACTIVE", r
        return {
            HighwayMergeGapResponse.ACTION_MERGE_READY: "MERGE_READY",
            HighwayMergeGapResponse.ACTION_WAIT: "WAIT",
            HighwayMergeGapResponse.ACTION_HOLD: "HOLD",
        }.get(r.action, "UNKNOWN"), r

    def _mean_abs_steering(self):
        recent = list(self.commands)[-8:]
        self.assertTrue(recent)
        return sum(abs(c.steering) for c in recent) / len(recent)

    def _step(self, label, pose, objects_fn, trace, settle=3.5, cut_in=None):
        self.pose = pose
        self.objects_fn = objects_fn
        self.cut_in = cut_in
        self._spin(settle)
        action, resp = self._response_action()
        row = {
            "step": label,
            "ego_xy": [round(pose[0], 3), round(pose[1], 3)],
            "ego_lateral_to_route0_m": round(abs(_lateral_to_route0(pose[0], pose[1])), 3),
            "response_action": action,
            "response_reason": int(resp.reason),
            "merge_authorized": bool(self.authorized and self.authorized[-1]),
            "mission_state": self._state(),
            "reference_active": bool(self.reference_active and self.reference_active[-1]),
            "merge_speed_limit": round(self.merge_limit[-1], 3) if self.merge_limit else None,
            "steering": round(self._mean_abs_steering(), 4),
        }
        trace.append(row)
        return row

    def _state(self):
        return self._latest(self.states, "mission state")

    def test_end_to_end(self):
        # discovery + FollowGlobalPath
        self._spin(6.0)
        self.assertEqual(self.node.count_publishers(COMMAND_TOPIC), 1)
        # the real gap-response node publishes the response, and the real
        # planner (integration on) subscribes it -- proves the chain is wired,
        # not individually mocked.
        deadline0 = time.monotonic() + 10.0
        while time.monotonic() < deadline0 and self.node.count_subscribers(RESPONSE_TOPIC) < 2:
            self._spin(0.3)
        self.assertGreaterEqual(self.node.count_subscribers(RESPONSE_TOPIC), 2)
        self.assertGreaterEqual(self.node.count_publishers(RESPONSE_TOPIC), 1)
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            self._spin(0.5)
            if self.statuses and self.statuses[-1].active_behavior == "follow_global_path":
                break
        self.assertEqual(self.statuses[-1].active_behavior, "follow_global_path")

        trace = []

        def FAST_REAR(pose):
            return [
                _mainline_object(1, pose, 1200.0, 24.0),          # front, mainline
                _mainline_object(3, pose, _proj(pose) - 12.0, 33.0),  # rear, closing hard
            ]

        def CLEAR(pose):
            return []  # zero relevant objects -> MERGE_READY

        # A. UNSAFE REAR APPROACH -- ego on ramp, WAITING
        a = self._step("A_unsafe_rear", _ramp_pose(1120.0), FAST_REAR, trace)
        self.assertNotEqual(a["response_action"], "MERGE_READY")
        self.assertFalse(a["merge_authorized"])
        self.assertEqual(a["mission_state"], WAITING)
        self.assertTrue(a["reference_active"])

        # B. GAP OPENS -- zero relevant objects -> MERGE_READY -> AUTHORIZED
        b = self._step("B_gap_opens", _ramp_pose(1150.0), CLEAR, trace)
        self.assertEqual(b["response_action"], "MERGE_READY")
        self.assertTrue(b["merge_authorized"])
        self.assertEqual(b["mission_state"], AUTHORIZED)
        self.assertTrue(b["reference_active"])

        # B2. AUTHORIZED at the SAME pose as A -- lateral reference unchanged
        steer_wait = a["steering"]
        b2 = self._step("B2_authorized_same_pose_as_A", _ramp_pose(1120.0), CLEAR, trace)
        self.assertEqual(b2["mission_state"], AUTHORIZED)
        self.assertTrue(b2["reference_active"])
        self.assertLess(abs(b2["steering"] - steer_wait), 0.05,
                        "authorization changed the lateral command")

        # C. PRE-COMMIT REVOKE -- unsafe again before commit
        c = self._step("C_pre_commit_revoke", _ramp_pose(1160.0), FAST_REAR, trace)
        self.assertNotEqual(c["response_action"], "MERGE_READY")
        self.assertFalse(c["merge_authorized"])
        self.assertEqual(c["mission_state"], WAITING)
        self.assertTrue(c["reference_active"], "pre-commit revoke must not drop the merge reference")

        # D. REAUTHORIZE
        d = self._step("D_reauthorize", _ramp_pose(1180.0), CLEAR, trace)
        self.assertEqual(d["response_action"], "MERGE_READY")
        self.assertTrue(d["merge_authorized"])
        self.assertEqual(d["mission_state"], AUTHORIZED)

        # E. COMMIT -- past ~1213 m while authorized
        e = self._step("E_commit", _ramp_pose(1220.0), CLEAR, trace)
        self.assertEqual(e["mission_state"], COMMITTED)
        self.assertTrue(e["reference_active"])
        # still following route:0:left:1 at the commit region, not route:0
        self.assertGreater(e["ego_lateral_to_route0_m"], 2.5)

        # F. POST-COMMIT REVOKE -- unsafe again; COMMITTED retained
        f = self._step("F_post_commit_revoke", _ramp_pose(1250.0), FAST_REAR, trace)
        self.assertEqual(f["mission_state"], COMMITTED, "post-commit revoke must not reverse the merge")
        self.assertFalse(f["merge_authorized"])
        self.assertTrue(f["reference_active"])
        # independent longitudinal safety still bites: a merge cap is applied
        self.assertIsNotNone(f["merge_speed_limit"])
        self.assertGreaterEqual(f["merge_speed_limit"], 0.0)

        # F2. POST-COMMIT, ego almost at merge complete + unsafe -> HOLD; the
        #     merge longitudinal cap collapses to 0.0 through the existing
        #     external-constraint path (no direct brake publication); COMMITTED
        #     and the merge reference are retained.
        f2 = self._step("F2_post_commit_hold", _ramp_pose(1283.0), FAST_REAR, trace)
        self.assertEqual(f2["mission_state"], COMMITTED)
        self.assertTrue(f2["reference_active"])
        self.assertIn(f2["response_action"], ("WAIT", "HOLD"))
        if f2["response_action"] == "HOLD":
            self.assertLessEqual(f2["merge_speed_limit"], 0.01)
            recent = list(self.commands)[-8:]
            self.assertTrue(any(c.brake > 0.0 or c.accel <= 0.0 for c in recent),
                            "HOLD did not produce braking")

        # G. COMPLETE -- ego reaches merge complete; handoff to production path
        steer_before = self._mean_abs_steering()
        xtrack_before = e["ego_lateral_to_route0_m"]
        g = self._step("G_complete", MAINLINE[1295.0], CLEAR, trace)
        self.assertEqual(g["mission_state"], COMPLETE)
        self.assertFalse(g["reference_active"], "COMPLETE hands back to the production path")
        self.assertLess(g["steering"], 0.2)
        self.assertLess(g["ego_lateral_to_route0_m"], 0.5)

        # H. RELEASE
        h = self._step("H_release", MAINLINE[1330.0], CLEAR, trace)
        self.assertEqual(h["mission_state"], INACTIVE)
        self.assertFalse(h["reference_active"])

        # I. cut-in composition -- during a ramp WAITING frame, a stricter cut-in
        #    HOLD cap must still win the min().
        cut_in = CutInResponse()
        cut_in.action = CutInResponse.ACTION_HOLD
        cut_in.active = True
        cut_in.reason = CutInResponse.REASON_COLLISION_CONFLICT
        cut_in.requested_max_speed_valid = True
        cut_in.requested_max_speed_mps = 0.0
        i = self._step("I_cut_in_hold_during_merge", _ramp_pose(1120.0), FAST_REAR, trace, cut_in=cut_in)
        self.assertTrue(i["reference_active"])
        recent = list(self.commands)[-8:]
        self.assertTrue(any(cmd.brake > 0.0 or cmd.accel <= 0.0 for cmd in recent),
                        "cut-in HOLD did not produce braking during the merge")

        # J. reset mid-merge -- jump the ego far back; latch clears
        self._step("J_reset_backward", MAINLINE[400.0], CLEAR, trace, settle=4.0)
        self.assertEqual(self._state(), INACTIVE)
        self.assertFalse(self.reference_active[-1])

        # --- loopback timing: risk publish -> gap-risk frame / gap-response,
        #     and planner command interval, reference INACTIVE (mainline pose)
        #     vs ACTIVE (ramp pose). Loopback, not internal compute cost.
        timing = {}
        for label, pose, want_ref in (("reference_inactive", MAINLINE[1000.0], False),
                                      ("reference_active", _ramp_pose(1150.0), True)):
            self.pose = pose
            self.objects_fn = CLEAR
            self.cut_in = None
            self._spin(3.0)
            risk_to_gap = []
            risk_to_resp = []
            cmd_intervals = []
            for _ in range(25):
                n_gap, n_resp = len(self.gap_frames), len(self.responses)
                last_cmds = len(self.commands)
                t0 = time.monotonic()
                self._publish()
                while len(self.gap_frames) <= n_gap and time.monotonic() - t0 < 1.0:
                    rclpy.spin_once(self.node, timeout_sec=0.01)
                risk_to_gap.append((time.monotonic() - t0) * 1e3)
                while len(self.responses) <= n_resp and time.monotonic() - t0 < 1.0:
                    rclpy.spin_once(self.node, timeout_sec=0.01)
                risk_to_resp.append((time.monotonic() - t0) * 1e3)
                t_cmd = time.monotonic()
                while len(self.commands) <= last_cmds and time.monotonic() - t_cmd < 1.0:
                    self._publish()
                    rclpy.spin_once(self.node, timeout_sec=0.01)
                cmd_intervals.append((time.monotonic() - t_cmd) * 1e3)
            self.assertEqual(bool(self.reference_active[-1]), want_ref)
            timing[label] = {
                "risk_to_gap_ms": _stats(risk_to_gap),
                "risk_to_response_ms": _stats(risk_to_resp),
                "publish_to_command_ms": _stats(cmd_intervals),
            }

        # numeric finiteness + one publisher
        for cmd in self.commands:
            self.assertTrue(math.isfinite(cmd.steering))
            self.assertTrue(math.isfinite(cmd.accel))
            self.assertTrue(math.isfinite(cmd.brake))
        self.assertEqual(self.node.count_publishers(COMMAND_TOPIC), 1)

        summary = {
            "trace": trace,
            "timing": timing,
            "reference_active_trace": [r["reference_active"] for r in trace],
            "authorization_trace": [r["merge_authorized"] for r in trace],
            "mission_state_trace": [r["mission_state"] for r in trace],
            "response_action_trace": [r["response_action"] for r in trace],
            "counts": {
                "gap_frames": len(self.gap_frames),
                "responses": len(self.responses),
                "commands": len(self.commands),
                "merge_ready": sum(1 for r in self.responses if r.active and r.action == HighwayMergeGapResponse.ACTION_MERGE_READY),
                "wait": sum(1 for r in self.responses if r.active and r.action == HighwayMergeGapResponse.ACTION_WAIT),
                "hold": sum(1 for r in self.responses if r.active and r.action == HighwayMergeGapResponse.ACTION_HOLD),
                "authorized_true": sum(1 for v in self.authorized if v),
                "reference_active_true": sum(1 for v in self.reference_active if v),
            },
        }
        Path("/tmp/heven_highway_merge_e2e_result.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        # Phase 35 key state trace (WAITING -> AUTHORIZED -> WAITING -> AUTHORIZED
        # -> COMMITTED -> COMMITTED -> COMPLETE -> INACTIVE) over steps A..H.
        core = [
            r for r in trace
            if r["step"] in ("A_unsafe_rear", "B_gap_opens", "C_pre_commit_revoke",
                             "D_reauthorize", "E_commit", "F_post_commit_revoke",
                             "G_complete", "H_release")
        ]
        self.assertEqual(
            [r["mission_state"] for r in core],
            [WAITING, AUTHORIZED, WAITING, AUTHORIZED, COMMITTED, COMMITTED, COMPLETE, INACTIVE],
        )
        self.assertEqual(
            [r["reference_active"] for r in core],
            [True, True, True, True, True, True, False, False],
        )
        self.assertEqual(
            [r["merge_authorized"] for r in core],
            [False, True, False, True, True, False, False, False],
        )


def _proj(pose):
    """Approx route:0 station of a ramp pose (for placing a rear object)."""
    x, y, _ = pose
    best = min(_R0, key=lambda q: math.hypot(q["x_m"] - x, q["y_m"] - y))
    return best["route_s_m"]


def _stats(values):
    if not values:
        return {"median": None, "p95": None, "max": None, "n": 0}
    ordered = sorted(values)
    return {
        "median": round(statistics.median(ordered), 3),
        "p95": round(ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))], 3),
        "max": round(ordered[-1], 3),
        "n": len(ordered),
    }


@launch_testing.post_shutdown_test()
class TestHighwayMergeEndToEndShutdown(unittest.TestCase):
    def test_clean_shutdown(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info)
