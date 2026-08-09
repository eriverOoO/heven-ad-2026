import hashlib
import json
import math
import os
import re
import signal
import subprocess
import copy
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import tempfile
import threading
import time

from ad_interfaces.msg import (
    PlannerStatus,
    PredictedObject,
    PredictedObjectArray,
    PredictedState,
)
from ad_morai_interfaces.msg import CollisionArray, CtrlCmd, EgoVehicleStatus
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path as RosPath
import pytest
import rclpy
from rclpy.context import Context
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray
import yaml


CASE_NAMES = (
    "dwa_default_legacy",
    "dwa_future_prediction_uses_past_snapshot",
    "frenet_valid_nonidentity",
    "frenet_missing_cache",
    "frenet_bad_cache",
    "frenet_bad_digest",
    "frenet_missing_tf",
    "frenet_no_candidate",
    "mppi_runtime_live",
)
CASE_DOMAINS = tuple(range(71, 80))
CASE_PARAMETERS = tuple(zip(CASE_NAMES, CASE_DOMAINS))
if len(CASE_PARAMETERS) != 9 or len(set(CASE_DOMAINS)) != 9:
    raise RuntimeError("runtime cases require nine distinct ROS domains")

FAILURE_REASONS = {
    "frenet_missing_cache": ("route occupancy is stale",),
    "frenet_bad_cache": ("route occupancy is stale",),
    "frenet_bad_digest": ("route occupancy is stale",),
    "frenet_missing_tf": ("stamped local motion transform failed",),
    "frenet_no_candidate": ("no valid candidate",),
}
EXPECTED_LOG_REASONS = {
    "frenet_missing_cache": (
        "local motion activation failed",
        "cannot parse route corridor cache",
    ),
    "frenet_bad_cache": ("unsupported schema_version 2",),
    "frenet_bad_digest": ("source SHA-256 mismatch for 'global_path'",),
}

INPUT_TOPICS = (
    "/ad/vehicle/status",
    "/ad/localization/odometry",
    "/ad/safety/collisions",
    "/ad/perception/occupancy_grid",
    "/ad/viz/perception/occupancy/static_ungated",
)
OUTPUT_TOPICS = (
    "/ad/control/command",
    "/ad/planner/status",
    "/ad/viz/planner/local_path",
    "/ad/viz/planner/candidate_paths",
    "/ad/viz/planner/occupancy_relevance",
    "/ad/viz/planner/relevant_objects",
)
MPPI_COMMAND_TOPIC = "/ad/planner/mppi/cmd_vel"
PREDICTED_OBJECT_TOPIC = "/ad/perception/objects/predicted"
DRIVABLE_MASK_TOPIC = "/ad/planning/drivable_mask"

_SEEN_PLANNER_PIDS = set()
_SEEN_PLANNER_PIDS_LOCK = threading.Lock()
_WORKTREE_ROOT = Path(__file__).resolve().parents[2]
_WORKTREE_SETUP = _WORKTREE_ROOT / "install" / "setup.bash"
_WORKTREE_PLANNER_SHARE = (
    _WORKTREE_ROOT / "install" / "ad_planner" / "share" / "ad_planner"
)


def _installed_mppi_proxy_is_enabled():
    proxy = (
        _WORKTREE_ROOT
        / "install"
        / "ad_planner"
        / "lib"
        / "ad_planner"
        / "ad_planner_mppi_follow_path_proxy"
    )
    try:
        target = proxy.resolve(strict=True)
        target_stat = target.stat()
    except OSError:
        return False
    if not target.is_file() or target_stat.st_size <= 0:
        return False
    if not os.access(target, os.X_OK):
        return False
    if not proxy.is_symlink():
        return True

    cache = target.parent / "CMakeCache.txt"
    try:
        cache_lines = set(cache.read_text(encoding="utf-8").splitlines())
    except (OSError, UnicodeError):
        return False
    return {
        "CMAKE_PROJECT_NAME:STATIC=ad_planner",
        "AD_PLANNER_ENABLE_NAV2_MPPI:BOOL=ON",
    }.issubset(cache_lines)


@dataclass
class FixtureCase:
    case_name: str
    temporary_directory: tempfile.TemporaryDirectory
    data_dir: Path
    common_yaml: Path
    corridor_path: Path
    backend: str
    publish_tf: bool
    occupied_grid: bool
    expected_reason_substrings: tuple


@dataclass
class RunningCase:
    fixture: FixtureCase
    domain_id: int
    driver: "RuntimeDriver"
    planner: "PlannerSubprocess"


def _write_fixture_case(case_name):
    temporary_directory = tempfile.TemporaryDirectory(
        prefix=f"ad_planner_{case_name}_"
    )
    root = Path(temporary_directory.name)
    data_dir = root / "data"
    path_dir = data_dir / "path"
    map_dir = data_dir / "map"
    path_dir.mkdir(parents=True)
    map_dir.mkdir()

    active_path = path_dir / "active.txt"
    active_path.write_text(
        "".join(f"{float(x):.1f},0.0,0.0\n" for x in range(-20, 101, 2)),
        encoding="utf-8",
    )
    path_digest = hashlib.sha256(active_path.read_bytes()).hexdigest()
    corridor = {
        "schema_version": 1,
        "frame_id": "map",
        "primary_lane_sequence_id": "route:0",
        "source_sha256": {
            "global_info.json": "1" * 64,
            "global_path": path_digest,
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
                        "speed_limit_mps": 10.0,
                    }
                    for x in range(-20, 101, 2)
                ],
            }
        ],
    }
    if case_name == "frenet_bad_cache":
        corridor["schema_version"] = 2
    elif case_name == "frenet_bad_digest":
        corridor["source_sha256"]["global_path"] = (
            ("0" if path_digest[0] != "0" else "1") + path_digest[1:]
        )

    corridor_path = map_dir / "route_corridor.json"
    if case_name != "frenet_missing_cache":
        corridor_path.write_text(
            json.dumps(
                corridor,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    package_share = (
        _WORKTREE_PLANNER_SHARE
        if _WORKTREE_PLANNER_SHARE.is_dir()
        else Path(get_package_share_directory("ad_planner"))
    )
    template = yaml.safe_load(
        (package_share / "config" / "planner.yaml").read_text(encoding="utf-8")
    )
    parameters = template["ad_planner"]["ros__parameters"]
    if case_name in {
        "dwa_default_legacy",
        "dwa_future_prediction_uses_past_snapshot",
    }:
        if parameters.get("local_motion.backend") != "dwa":
            raise AssertionError("installed planner.yaml must keep DWA as the default")
        backend = "dwa"
    elif case_name == "mppi_runtime_live":
        backend = "mppi_nav2"
    else:
        backend = "frenet_lattice"
    parameters.update(
        {
            "path_file": "path/active.txt",
            "route_corridor_file": "map/route_corridor.json",
        }
    )
    if backend != "dwa":
        parameters["local_motion.backend"] = backend
    if case_name in {
        "frenet_missing_cache",
        "frenet_bad_cache",
        "frenet_bad_digest",
    }:
        parameters["road_gate.enabled"] = False

    common_yaml = root / "planner.yaml"
    common_yaml.write_text(
        yaml.safe_dump(template, sort_keys=False),
        encoding="utf-8",
    )
    return FixtureCase(
        case_name=case_name,
        temporary_directory=temporary_directory,
        data_dir=data_dir,
        common_yaml=common_yaml,
        corridor_path=corridor_path,
        backend=backend,
        publish_tf=case_name != "frenet_missing_tf",
        occupied_grid=case_name == "frenet_no_candidate",
        expected_reason_substrings=FAILURE_REASONS.get(case_name, ()),
    )


def test_default_grid_timing_covers_measured_live_ogm_latency():
    package_share = (
        _WORKTREE_PLANNER_SHARE
        if _WORKTREE_PLANNER_SHARE.is_dir()
        else Path(get_package_share_directory("ad_planner"))
    )
    document = yaml.safe_load(
        (package_share / "config" / "planner.yaml").read_text(encoding="utf-8")
    )
    parameters = document["ad_planner"]["ros__parameters"]

    assert parameters["local_motion.max_grid_odom_skew_sec"] == pytest.approx(
        0.75
    )
    assert parameters["local_motion.max_grid_age_sec"] == pytest.approx(0.80)


def _yaw_quaternion(message, yaw):
    message.z = math.sin(yaw * 0.5)
    message.w = math.cos(yaw * 0.5)


def _matches_suffix(messages, predicate):
    return len(messages) >= 3 and all(predicate(message) for message in list(messages)[-3:])


def _finite_non_braking(command):
    return (
        command.ctrl_mode == CtrlCmd.CTRL_MODE_AUTO
        and math.isfinite(command.accel)
        and math.isfinite(command.brake)
        and math.isfinite(command.steering)
        and command.brake == 0.0
    )


def _dwa_straight_command(command):
    return (
        _finite_non_braking(command)
        and command.accel > 0.0
        and math.isclose(command.steering, 0.0, rel_tol=0.0, abs_tol=0.02)
    )


def _full_brake(command):
    return (
        command.ctrl_mode == CtrlCmd.CTRL_MODE_AUTO
        and command.accel == 0.0
        and command.brake == 1.0
        and math.isfinite(command.steering)
    )


def _future_risk_status(status):
    return (
        status.inputs_ready
        and status.active_behavior == "perception_mission"
        and status.failsafe_reason == ""
    )


def _successful_path(path):
    if path.header.frame_id != "odom" or not path.poses:
        return False
    first = path.poses[0].pose.position
    return math.hypot(first.x - 10.0, first.y + 2.0) < 0.25


def _empty_odom_path(path):
    return path.header.frame_id == "odom" and not path.poses


def _successful_markers(markers):
    return (
        bool(markers.markers)
        and markers.markers[0].action == Marker.DELETEALL
        and any(
            marker.action == Marker.ADD and marker.id >= 0
            for marker in markers.markers[1:]
        )
    )


def _clear_markers(markers):
    return (
        bool(markers.markers)
        and markers.markers[0].action == Marker.DELETEALL
        and all(marker.action != Marker.ADD for marker in markers.markers)
    )


def _relevance_markers(markers, marker_type, frame_id):
    return (
        len(markers.markers) == 3
        and markers.markers[0].action == Marker.DELETEALL
        and all(
            marker.action == Marker.ADD
            and marker.type == marker_type
            and marker.header.frame_id == frame_id
            for marker in markers.markers[1:]
        )
    )


def _process_group_members(process_group_id):
    members = []
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            stat = stat_path.read_text(encoding="utf-8")
            remainder = stat[stat.rfind(")") + 2 :].split()
            if len(remainder) >= 3 and int(remainder[2]) == process_group_id:
                members.append(int(stat_path.parent.name))
        except (FileNotFoundError, PermissionError, ValueError):
            continue
    return sorted(members)


def _process_command(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            errors="replace"
        )
    except (FileNotFoundError, PermissionError):
        return ""


def _worktree_environment():
    if not _WORKTREE_SETUP.is_file():
        return os.environ.copy()
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1" && env -0',
            "ad-planner-runtime-environment",
            str(_WORKTREE_SETUP),
        ],
        check=True,
        stdout=subprocess.PIPE,
    )
    return {
        key.decode(): value.decode()
        for entry in result.stdout.split(b"\0")
        if entry
        for key, value in (entry.split(b"=", 1),)
    }


class PlannerSubprocess:
    def __init__(self, fixture, domain_id):
        self.fixture = fixture
        self.domain_id = domain_id
        self.process = None
        self.process_group_id = None
        self.planner_pid = None
        self.log_path = Path(fixture.temporary_directory.name) / "planner-launch.log"
        self.log_stream = None

    def start(self):
        environment = _worktree_environment()
        environment.update(
            {
                "RCUTILS_LOGGING_BUFFERED_STREAM": "1",
                "ROS_DOMAIN_ID": str(self.domain_id),
                "ROS_LOCALHOST_ONLY": "1",
            }
        )
        self.log_stream = self.log_path.open("w+", encoding="utf-8")
        self.process = subprocess.Popen(
            [
                "ros2",
                "launch",
                "ad_planner",
                "planner.launch.py",
                f"config_file:={self.fixture.common_yaml}",
                f"data_dir:={self.fixture.data_dir}",
            ],
            env=environment,
            stdout=self.log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        self.process_group_id = os.getpgid(self.process.pid)

    def assert_running_planner(self):
        assert self.process is not None
        assert self.process.poll() is None, (
            f"planner launch exited before graph match:\n{self.read_log()}"
        )
        planner_pids = [
            pid
            for pid in _process_group_members(self.process_group_id)
            if re.search(r"(^|/)ad_planner_node(?:\s|$)", _process_command(pid))
        ]
        assert len(planner_pids) == 1, (
            f"expected one planner in process group {self.process_group_id}, "
            f"found {planner_pids}:\n{self.read_log()}"
        )
        self.planner_pid = planner_pids[0]
        with _SEEN_PLANNER_PIDS_LOCK:
            assert self.planner_pid not in _SEEN_PLANNER_PIDS
            _SEEN_PLANNER_PIDS.add(self.planner_pid)

    def read_log(self):
        if self.log_stream is not None:
            self.log_stream.flush()
        try:
            return self.log_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def _wait_for_empty_group(self, timeout_sec):
        deadline = time.monotonic() + timeout_sec
        waiter = threading.Event()
        while True:
            members = _process_group_members(self.process_group_id)
            if not members:
                return []
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return members
            waiter.wait(min(0.02, remaining))

    def stop(self):
        errors = []
        if self.process is None:
            return ["planner launch process was never started"]
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=8.0)
            except subprocess.TimeoutExpired:
                errors.append("planner launch did not exit within 8 seconds after SIGINT")
                os.killpg(self.process_group_id, signal.SIGTERM)
                try:
                    self.process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process_group_id, signal.SIGKILL)
                    self.process.wait(timeout=2.0)
        if self.process.returncode != 0:
            errors.append(
                f"planner launch exited with code {self.process.returncode}"
            )

        remaining = self._wait_for_empty_group(2.0)
        if remaining:
            errors.append(
                f"planner process group {self.process_group_id} retained {remaining}"
            )
            try:
                os.killpg(self.process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
            final_remaining = self._wait_for_empty_group(1.0)
            if final_remaining:
                errors.append(
                    "planner process group still retained "
                    f"{final_remaining} after SIGKILL"
                )
        if self.planner_pid is None:
            errors.append("planner child PID was not recorded")
        elif Path(f"/proc/{self.planner_pid}").exists():
            errors.append(f"planner child PID {self.planner_pid} remains after teardown")

        if self.log_stream is not None:
            self.log_stream.close()
            self.log_stream = None
        return errors


class RuntimeDriver(Node):
    def __init__(self, fixture, context):
        super().__init__(
            f"frenet_runtime_contract_{fixture.case_name}",
            context=context,
        )
        self.fixture = fixture
        self.commands = deque(maxlen=32)
        self.statuses = deque(maxlen=32)
        self.paths = deque(maxlen=32)
        self.markers = deque(maxlen=32)
        self.occupancy_relevance = deque(maxlen=32)
        self.predicted_relevance = deque(maxlen=32)
        self.safety_violations = []
        self.contract_completed = False
        self.result_event = threading.Event()
        self.graph_event = threading.Event()
        self.lock = threading.Lock()
        self.publication_timer = None
        self.graph_timer = None

        self.status_publisher = self.create_publisher(
            EgoVehicleStatus, INPUT_TOPICS[0], qos_profile_sensor_data
        )
        self.odometry_publisher = self.create_publisher(
            Odometry, INPUT_TOPICS[1], qos_profile_sensor_data
        )
        self.collision_publisher = self.create_publisher(
            CollisionArray, INPUT_TOPICS[2], qos_profile_sensor_data
        )
        self.grid_publisher = self.create_publisher(
            OccupancyGrid, INPUT_TOPICS[3], qos_profile_sensor_data
        )
        self.static_ungated_publisher = self.create_publisher(
            OccupancyGrid, INPUT_TOPICS[4], qos_profile_sensor_data
        )
        self.prediction_publisher = self.create_publisher(
            PredictedObjectArray, PREDICTED_OBJECT_TOPIC, 10
        )
        self.drivable_mask_publisher = self.create_publisher(
            OccupancyGrid, DRIVABLE_MASK_TOPIC, qos_profile_sensor_data
        )
        self.create_subscription(CtrlCmd, OUTPUT_TOPICS[0], self._on_command, 10)
        self.create_subscription(PlannerStatus, OUTPUT_TOPICS[1], self._on_status, 10)
        self.create_subscription(RosPath, OUTPUT_TOPICS[2], self._on_path, 10)
        self.create_subscription(MarkerArray, OUTPUT_TOPICS[3], self._on_markers, 10)
        self.create_subscription(
            MarkerArray,
            OUTPUT_TOPICS[4],
            self._on_occupancy_relevance,
            10,
        )
        self.create_subscription(
            MarkerArray,
            OUTPUT_TOPICS[5],
            self._on_predicted_relevance,
            10,
        )

    def _on_command(self, message):
        self._store_and_check("command", self.commands, message)

    def _on_status(self, message):
        self._store_and_check("status", self.statuses, message)

    def _on_path(self, message):
        self._store_and_check("path", self.paths, message)

    def _on_markers(self, message):
        self._store_and_check("markers", self.markers, message)

    def _on_occupancy_relevance(self, message):
        self._store_and_check(
            "occupancy_relevance", self.occupancy_relevance, message
        )

    def _on_predicted_relevance(self, message):
        self._store_and_check(
            "predicted_relevance", self.predicted_relevance, message
        )

    def _store_and_check(self, kind, messages, message):
        with self.lock:
            messages.append(message)
            self._record_sticky_safety_violation_locked(kind, message)
            if self._predicate_locked():
                self.contract_completed = True
            if self.safety_violations or self.contract_completed:
                self.result_event.set()

    def _record_sticky_safety_violation_locked(self, kind, message):
        if not self.fixture.expected_reason_substrings:
            return
        if kind == "command" and not _full_brake(message):
            self.safety_violations.append(
                "observed non-full-brake command "
                f"(accel={message.accel}, brake={message.brake}, "
                f"steering={message.steering})"
            )
        elif kind == "path" and message.poses:
            self.safety_violations.append(
                f"observed nonempty local path with {len(message.poses)} poses"
            )
        elif kind == "markers" and any(
            marker.action == Marker.ADD for marker in message.markers
        ):
            self.safety_violations.append("observed ADD candidate marker")

    def _predicate_locked(self):
        case_name = self.fixture.case_name
        if case_name in {
            "dwa_default_legacy",
            "dwa_future_prediction_uses_past_snapshot",
        }:
            return _matches_suffix(self.commands, _dwa_straight_command) and (
                _matches_suffix(self.statuses, _future_risk_status)
            )
        if case_name == "frenet_valid_nonidentity":
            return (
                _matches_suffix(self.commands, _finite_non_braking)
                and _matches_suffix(self.statuses, _future_risk_status)
                and _matches_suffix(self.paths, _successful_path)
                and _matches_suffix(self.markers, _successful_markers)
                and _matches_suffix(
                    self.occupancy_relevance,
                    lambda markers: _relevance_markers(
                        markers, Marker.CUBE_LIST, "map"
                    ),
                )
                and _matches_suffix(
                    self.predicted_relevance,
                    lambda markers: _relevance_markers(
                        markers, Marker.LINE_LIST, "odom"
                    ),
                )
            )
        if case_name == "mppi_runtime_live":
            return (
                _matches_suffix(
                    self.commands,
                    lambda command: (
                        _finite_non_braking(command) and command.accel > 0.0
                    ),
                )
                and _matches_suffix(self.statuses, _future_risk_status)
                and _matches_suffix(self.paths, _mppi_motion_path)
            )

        def diagnostic_matches(status):
            return all(
                substring in status.failsafe_reason
                for substring in self.fixture.expected_reason_substrings
            )

        return (
            _matches_suffix(self.commands, _full_brake)
            and _matches_suffix(self.statuses, diagnostic_matches)
            and _matches_suffix(self.paths, _empty_odom_path)
            and _matches_suffix(self.markers, _clear_markers)
        )

    def assert_completed_contract(self):
        with self.lock:
            assert not self.safety_violations, (
                "sticky safety violation(s): " + "; ".join(self.safety_violations)
            )
            assert self.contract_completed, (
                f"steady-state predicate was never satisfied for {self.fixture.case_name}"
            )

    def assert_no_sticky_safety_violation(self):
        with self.lock:
            assert not self.safety_violations, (
                "sticky safety violation(s) after predicate completion: "
                + "; ".join(self.safety_violations)
            )

    def start_graph_match_wait(self):
        def check_graph():
            input_topics = INPUT_TOPICS
            if self.fixture.backend == "mppi_nav2":
                input_topics += (MPPI_COMMAND_TOPIC,)
            input_topics += (PREDICTED_OBJECT_TOPIC,)
            inputs_ready = all(
                self.count_subscribers(topic) > 0 for topic in input_topics
            )
            outputs_ready = all(
                self.count_publishers(topic) > 0 for topic in OUTPUT_TOPICS
            )
            if inputs_ready and outputs_ready:
                self.graph_event.set()

        self.graph_timer = self.create_timer(0.01, check_graph)

    def stop_graph_match_wait(self):
        if self.graph_timer is not None:
            self.graph_timer.cancel()

    def publish_static_transforms(self):
        self.static_broadcaster = StaticTransformBroadcaster(self)
        transforms = []
        for child_frame in ("map", "base_link"):
            transform = TransformStamped()
            transform.header.frame_id = "odom"
            transform.child_frame_id = child_frame
            if self.fixture.backend == "dwa":
                transform.transform.rotation.w = 1.0
            else:
                transform.transform.translation.x = 10.0
                transform.transform.translation.y = -2.0
                _yaw_quaternion(
                    transform.transform.rotation, math.pi / 2.0
                )
            transforms.append(transform)
        self.static_broadcaster.sendTransform(transforms)

    def start_publication(self):
        # Five hertz is sufficient for the 0.8 s freshness contract while
        # keeping three exact-stamp 208,000-cell grids from saturating a
        # two-core CI runner's Python/DDS path.
        self.publication_timer = self.create_timer(0.2, self._publish_inputs)

    def stop_publication(self):
        if self.publication_timer is not None:
            self.publication_timer.cancel()

    def destroy_timers(self):
        if self.publication_timer is not None:
            self.destroy_timer(self.publication_timer)
            self.publication_timer = None
        if self.graph_timer is not None:
            self.destroy_timer(self.graph_timer)
            self.graph_timer = None

    def _publish_inputs(self):
        now_ns = self.get_clock().now().nanoseconds
        current_stamp = Time(nanoseconds=now_ns).to_msg()
        odometry_stamp = current_stamp
        grid_stamp = current_stamp
        if (
            self.fixture.case_name
            == "dwa_future_prediction_uses_past_snapshot"
        ):
            grid_stamp = Time(nanoseconds=now_ns - 100_000_000).to_msg()

        status = EgoVehicleStatus()
        status.header.stamp = current_stamp
        status.header.frame_id = "base_link"
        status.ctrl_mode = CtrlCmd.CTRL_MODE_AUTO
        status.gear = CtrlCmd.GEAR_DRIVE
        status.velocity.x = 0.0

        odometry = Odometry()
        odometry.header.stamp = odometry_stamp
        odometry.header.frame_id = "odom"
        odometry.child_frame_id = "base_link"
        if self.fixture.backend == "dwa":
            odometry.pose.pose.orientation.w = 1.0
        else:
            odometry.pose.pose.position.x = 10.0
            odometry.pose.pose.position.y = -2.0
            _yaw_quaternion(odometry.pose.pose.orientation, math.pi / 2.0)

        collisions = CollisionArray()
        collisions.header.stamp = current_stamp
        collisions.header.frame_id = "base_link"

        grid = OccupancyGrid()
        grid.header.stamp = grid_stamp
        grid.header.frame_id = "base_link"
        grid.info.resolution = 0.1
        grid.info.width = 1040
        grid.info.height = 200
        grid.info.origin.position.x = -4.0
        grid.info.origin.position.y = -10.0
        grid.info.origin.orientation.w = 1.0
        cell = 100 if self.fixture.occupied_grid else 0
        grid.data = [cell] * (grid.info.width * grid.info.height)

        prediction = PredictedObjectArray()
        prediction.header.stamp = grid_stamp
        prediction.header.frame_id = "odom"
        obstacle = PredictedObject()
        obstacle.object_id.uuid[0] = 1
        obstacle.existence_probability = 1.0
        obstacle.classification = PredictedObject.CAR
        obstacle.classification_probability = 1.0
        obstacle.dimensions.x = 4.5
        obstacle.dimensions.y = 1.8
        obstacle.dimensions.z = 1.5
        if self.fixture.backend == "dwa":
            obstacle.initial_pose.pose.position.x = 80.0
            obstacle.initial_pose.pose.orientation.w = 1.0
        else:
            obstacle.initial_pose.pose.position.x = 10.0
            obstacle.initial_pose.pose.position.y = 78.0
            _yaw_quaternion(
                obstacle.initial_pose.pose.orientation, math.pi / 2.0
            )
        # DWA needs 4.6 s at its configured maximum speed for rollout plus
        # emergency braking. The 6.0 s fixture matches the checked-in
        # prediction profile and covers that horizon after the planner's
        # accepted 0.5 s prediction age.
        for horizon_index in range(1, 13):
            horizon_s = 0.5 * horizon_index
            state = PredictedState()
            state.time_from_start.sec = int(horizon_s)
            state.time_from_start.nanosec = int(
                round((horizon_s - int(horizon_s)) * 1_000_000_000)
            )
            state.pose = obstacle.initial_pose
            obstacle.states.append(state)
        prediction.objects.append(obstacle)

        # Required predictions go first. Cross-topic callback order is not
        # guaranteed, but sending the light prediction before the two large
        # grids prevents a fresh odometry callback from repeatedly evaluating
        # the previous prediction at the edge of its braking horizon.
        if (
            self.fixture.case_name
            == "dwa_future_prediction_uses_past_snapshot"
        ):
            prediction.header.stamp = Time(
                nanoseconds=now_ns - 100_000_000
            ).to_msg()
            future_prediction = copy.deepcopy(prediction)
            future_prediction.header.stamp = Time(
                nanoseconds=now_ns + 100_000_000
            ).to_msg()
            self.prediction_publisher.publish(prediction)
            self.prediction_publisher.publish(future_prediction)
        else:
            self.prediction_publisher.publish(prediction)
        self.status_publisher.publish(status)
        self.odometry_publisher.publish(odometry)
        self.collision_publisher.publish(collisions)
        self.grid_publisher.publish(grid)
        if self.fixture.case_name in {
            "dwa_default_legacy",
            "dwa_future_prediction_uses_past_snapshot",
        }:
            drivable_mask = copy.deepcopy(grid)
            drivable_mask.data = [0] * len(grid.data)
            self.drivable_mask_publisher.publish(drivable_mask)
        # Visualization must be fed from the pre-road-gate layer, but it may
        # never be mixed with another planning sample. Publish the identical
        # geometry and stamp; the planner's bounded exact-stamp pairer accepts
        # either callback order.
        self.static_ungated_publisher.publish(grid)

    def dump_recent_messages(self):
        mppi_publishers = [
            (info.node_namespace, info.node_name)
            for info in self.get_publishers_info_by_topic(MPPI_COMMAND_TOPIC)
        ]
        with self.lock:
            commands = [
                (message.ctrl_mode, message.accel, message.brake, message.steering)
                for message in list(self.commands)[-10:]
            ]
            statuses = [
                (
                    message.inputs_ready,
                    message.active_behavior,
                    message.failsafe_reason,
                )
                for message in list(self.statuses)[-10:]
            ]
            paths = [
                (
                    message.header.frame_id,
                    len(message.poses),
                    (
                        message.poses[0].pose.position.x,
                        message.poses[0].pose.position.y,
                    )
                    if message.poses
                    else None,
                )
                for message in list(self.paths)[-10:]
            ]
            markers = [
                [(marker.action, marker.id) for marker in message.markers]
                for message in list(self.markers)[-10:]
            ]
            occupancy_relevance = [
                [(marker.action, marker.type) for marker in message.markers]
                for message in list(self.occupancy_relevance)[-10:]
            ]
            predicted_relevance = [
                [(marker.action, marker.type) for marker in message.markers]
                for message in list(self.predicted_relevance)[-10:]
            ]
        print(f"case={self.fixture.case_name}")
        print(f"yaml={self.fixture.common_yaml}")
        print(f"corridor={self.fixture.corridor_path}")
        print(f"commands={commands}")
        print(f"statuses={statuses}")
        print(f"paths={paths}")
        print(f"markers={markers}")
        print(f"occupancy_relevance={occupancy_relevance}")
        print(f"predicted_relevance={predicted_relevance}")
        print(f"mppi_command_publishers={mppi_publishers}")


def _mppi_motion_path(path):
    if path.header.frame_id != "odom" or len(path.poses) < 2:
        return False
    first = path.poses[0].pose.position
    return any(
        math.hypot(
            pose.pose.position.x - first.x,
            pose.pose.position.y - first.y,
        )
        > 0.05
        for pose in path.poses[1:]
    )


@pytest.fixture
def running_case(case_name, domain_id):
    if (
        case_name == "mppi_runtime_live"
        and not _installed_mppi_proxy_is_enabled()
    ):
        pytest.skip(
            "mppi_runtime_live requires the installed optional MPPI "
            "proxy from an AD_PLANNER_ENABLE_NAV2_MPPI=ON build"
        )

    fixture = _write_fixture_case(case_name)
    context = Context()
    driver = None
    executor = None
    spin_thread = None
    planner = PlannerSubprocess(fixture, domain_id)
    cleanup_errors = []
    previous_localhost_only = os.environ.get("ROS_LOCALHOST_ONLY")
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    try:
        rclpy.init(
            context=context,
            domain_id=domain_id,
            signal_handler_options=SignalHandlerOptions.NO,
        )
        assert context.get_domain_id() == domain_id
        driver = RuntimeDriver(fixture, context)
        executor = MultiThreadedExecutor(num_threads=2, context=context)
        executor.add_node(driver)
        spin_thread = threading.Thread(
            target=executor.spin,
            name=f"runtime-contract-spin-{case_name}",
            daemon=True,
        )
        spin_thread.start()

        planner.start()
        driver.start_graph_match_wait()
        if not driver.graph_event.wait(5.0):
            driver.dump_recent_messages()
            pytest.fail(
                "planner graph did not match all fixture inputs and outputs:\n"
                + planner.read_log()
            )
        driver.stop_graph_match_wait()
        planner.assert_running_planner()
        yield RunningCase(fixture, domain_id, driver, planner)
    finally:
        if driver is not None:
            driver.stop_publication()
            driver.stop_graph_match_wait()
        if executor is not None:
            if not executor.shutdown(timeout_sec=2.0):
                cleanup_errors.append("executor.shutdown(timeout_sec=2.0) returned false")
        if spin_thread is not None:
            spin_thread.join(timeout=2.0)
            if spin_thread.is_alive():
                cleanup_errors.append("executor spin thread remains alive")
        if driver is not None:
            try:
                driver.assert_no_sticky_safety_violation()
            except AssertionError as error:
                cleanup_errors.append(str(error))
            driver.destroy_timers()
            driver.destroy_node()
        if context.ok():
            context.shutdown()
        if context.ok():
            cleanup_errors.append("explicit ROS context remains valid after shutdown")

        cleanup_errors.extend(planner.stop())
        planner_log = planner.read_log()
        fixture.temporary_directory.cleanup()
        if previous_localhost_only is None:
            os.environ.pop("ROS_LOCALHOST_ONLY", None)
        else:
            os.environ["ROS_LOCALHOST_ONLY"] = previous_localhost_only
        if cleanup_errors:
            pytest.fail(
                "runtime cleanup contract failed: "
                + "; ".join(cleanup_errors)
                + f"\nplanner log:\n{planner_log}"
            )


@pytest.mark.parametrize(
    "case_name,domain_id",
    CASE_PARAMETERS,
    ids=CASE_NAMES,
)
def test_runtime_contract_case(case_name, domain_id, running_case):
    driver = running_case.driver
    fixture = running_case.fixture
    if fixture.publish_tf:
        driver.publish_static_transforms()
        time.sleep(0.25)
    driver.start_publication()
    if not driver.result_event.wait(10.0):
        driver.dump_recent_messages()
        pytest.fail(
            f"runtime predicate timed out for {fixture.case_name}:\n"
            + running_case.planner.read_log()
        )
    driver.assert_completed_contract()
    for diagnostic in EXPECTED_LOG_REASONS.get(case_name, ()):
        assert diagnostic in running_case.planner.read_log()
