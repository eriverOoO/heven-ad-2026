import math
import os
from pathlib import Path
import signal
import subprocess
from collections import deque
import threading
import time

from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_prefix,
)
from builtin_interfaces.msg import Time as TimeMessage
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid
import pytest
import rclpy
from rclpy.context import Context
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.signals import SignalHandlerOptions
from tf2_ros.transform_broadcaster import TransformBroadcaster


INPUT_TOPIC = "/ad/perception/occupancy_grid"
OUTPUT_TOPIC = "/ad/planner/mppi/occupancy_grid_odom"
_WORKTREE_ROOT = Path(__file__).resolve().parents[2]


def _planner_prefix():
    try:
        return Path(get_package_prefix("ad_planner"))
    except PackageNotFoundError:
        return _WORKTREE_ROOT / "install" / "ad_planner"


_PLANNER_PREFIX = _planner_prefix()
_WORKTREE_SETUP = next(
    (
        candidate
        for candidate in (
            _WORKTREE_ROOT / "install" / "setup.bash",
            _PLANNER_PREFIX.parent / "setup.bash",
            _PLANNER_PREFIX / "setup.bash",
        )
        if candidate.is_file()
    ),
    _WORKTREE_ROOT / "install" / "setup.bash",
)
_EXECUTABLE = (
    _PLANNER_PREFIX
    / "lib"
    / "ad_planner"
    / "ad_occupancy_grid_reprojector_node"
)


def _stamp(seconds, nanoseconds=0):
    result = TimeMessage()
    result.sec = seconds
    result.nanosec = nanoseconds
    return result


def _same_stamp(lhs, rhs):
    return lhs.sec == rhs.sec and lhs.nanosec == rhs.nanosec


def _worktree_environment():
    if not _WORKTREE_SETUP.is_file():
        return os.environ.copy()
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1" && env -0',
            "occupancy-grid-reprojector-runtime-environment",
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


class ReprojectorProcess:
    def __init__(self, temporary_path, domain_id):
        self.domain_id = domain_id
        self.log_path = temporary_path / "occupancy-grid-reprojector.log"
        self.log_stream = None
        self.process = None
        self.process_group_id = None

    def start(self):
        assert _EXECUTABLE.is_file(), (
            f"installed reprojector executable is missing: {_EXECUTABLE}"
        )
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
                str(_EXECUTABLE),
                "--ros-args",
                "-p",
                "transform_timeout_sec:=0.05",
            ],
            env=environment,
            stdout=self.log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        self.process_group_id = os.getpgid(self.process.pid)

    def read_log(self):
        if self.log_stream is not None:
            self.log_stream.flush()
        try:
            return self.log_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def stop(self):
        errors = []
        if self.process is None:
            return ["reprojector process was never started"]
        pid = self.process.pid
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=8.0)
            except subprocess.TimeoutExpired:
                errors.append("reprojector did not exit within 8 seconds after SIGINT")
                os.killpg(self.process_group_id, signal.SIGKILL)
                self.process.wait(timeout=2.0)
        if self.process.returncode != 0:
            errors.append(
                f"reprojector exited with code {self.process.returncode}"
            )
        deadline = time.monotonic() + 2.0
        members = _process_group_members(self.process_group_id)
        while members and time.monotonic() < deadline:
            threading.Event().wait(0.02)
            members = _process_group_members(self.process_group_id)
        if members:
            errors.append(
                f"reprojector process group {self.process_group_id} retained {members}"
            )
            try:
                os.killpg(self.process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if Path(f"/proc/{pid}").exists():
            errors.append(f"reprojector PID {pid} remains after teardown")
        if self.log_stream is not None:
            self.log_stream.close()
            self.log_stream = None
        return errors


class RuntimeDriver(Node):
    def __init__(self, context):
        super().__init__("occupancy_grid_reprojector_runtime_driver", context=context)
        self.messages = deque(maxlen=16)
        self.message_event = threading.Event()
        self.lock = threading.Lock()
        self.grid_publisher = self.create_publisher(
            OccupancyGrid, INPUT_TOPIC, qos_profile_sensor_data
        )
        reliable_volatile = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.output_subscription = self.create_subscription(
            OccupancyGrid,
            OUTPUT_TOPIC,
            self._on_output,
            reliable_volatile,
        )
        self.transform_broadcaster = TransformBroadcaster(self)

    def _on_output(self, message):
        with self.lock:
            self.messages.append(message)
            self.message_event.set()

    def clear_outputs(self):
        with self.lock:
            self.messages.clear()
            self.message_event.clear()

    def wait_for_output(self, timeout=2.0):
        if not self.message_event.wait(timeout):
            return None
        with self.lock:
            return self.messages[-1] if self.messages else None

    def assert_no_output(self, timeout=0.35):
        assert not self.message_event.wait(timeout), (
            f"unexpected reprojected output: {list(self.messages)}"
        )

    def publish_transforms(
        self,
        stamp,
        sensor_x=0.0,
        sensor_y=0.0,
        base_x=1.0,
        base_y=1.0,
    ):
        transforms = []
        for child, x, y in (
            ("sensor_frame", sensor_x, sensor_y),
            ("base_link", base_x, base_y),
        ):
            transform = TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id = "odom"
            transform.child_frame_id = child
            transform.transform.translation.x = x
            transform.transform.translation.y = y
            transform.transform.rotation.w = 1.0
            transforms.append(transform)
        self.transform_broadcaster.sendTransform(transforms)

    def publish_grid(self, stamp, data=(1, 2, 3, 4), orientation_w=1.0):
        message = OccupancyGrid()
        message.header.stamp = stamp
        message.header.frame_id = "sensor_frame"
        message.info.resolution = 0.1
        message.info.width = 2
        message.info.height = 2
        message.info.origin.orientation.w = orientation_w
        message.data = list(data)
        self.grid_publisher.publish(message)


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    waiter = threading.Event()
    while time.monotonic() < deadline:
        if predicate():
            return True
        waiter.wait(0.02)
    return predicate()


@pytest.fixture
def runtime(tmp_path):
    domain_id = 100 + (os.getpid() % 100)
    context = Context()
    driver = None
    executor = None
    spin_thread = None
    process = ReprojectorProcess(tmp_path, domain_id)
    cleanup_errors = []
    previous_localhost_only = os.environ.get("ROS_LOCALHOST_ONLY")
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    try:
        rclpy.init(
            context=context,
            domain_id=domain_id,
            signal_handler_options=SignalHandlerOptions.NO,
        )
        driver = RuntimeDriver(context)
        executor = MultiThreadedExecutor(num_threads=2, context=context)
        executor.add_node(driver)
        spin_thread = threading.Thread(
            target=executor.spin,
            name="occupancy-grid-reprojector-runtime-spin",
            daemon=True,
        )
        spin_thread.start()
        process.start()
        assert _wait_until(
            lambda: driver.count_subscribers(INPUT_TOPIC) == 1
            and driver.count_publishers(OUTPUT_TOPIC) == 1
        ), "reprojector graph did not appear:\n" + process.read_log()
        assert process.process.poll() is None, (
            "reprojector exited during graph discovery:\n" + process.read_log()
        )
        yield driver, process
    finally:
        if executor is not None:
            if not executor.shutdown(timeout_sec=2.0):
                cleanup_errors.append(
                    "executor.shutdown(timeout_sec=2.0) returned false"
                )
        if spin_thread is not None:
            spin_thread.join(timeout=2.0)
            if spin_thread.is_alive():
                cleanup_errors.append("executor spin thread remains alive")
        if driver is not None:
            driver.destroy_node()
        if context.ok():
            context.shutdown()
        if context.ok():
            cleanup_errors.append("explicit ROS context remains valid after shutdown")
        cleanup_errors.extend(process.stop())
        process_log = process.read_log()
        if previous_localhost_only is None:
            os.environ.pop("ROS_LOCALHOST_ONLY", None)
        else:
            os.environ["ROS_LOCALHOST_ONLY"] = previous_localhost_only
        if cleanup_errors:
            pytest.fail(
                "runtime cleanup contract failed: "
                + "; ".join(cleanup_errors)
                + f"\nreprojector log:\n{process_log}"
            )


def test_runtime_exact_stamp_qos_monotonic_retry_and_teardown(runtime):
    driver, process = runtime

    publisher_info = driver.get_publishers_info_by_topic(OUTPUT_TOPIC)
    assert len(publisher_info) == 1
    output_qos = publisher_info[0].qos_profile
    # Fast DDS may report UNKNOWN/0 for discovered history/depth in Humble.
    if output_qos.history != HistoryPolicy.UNKNOWN:
        assert output_qos.history == HistoryPolicy.KEEP_LAST
    if output_qos.depth != 0:
        assert output_qos.depth == 1
    assert output_qos.reliability == ReliabilityPolicy.RELIABLE
    assert output_qos.durability == DurabilityPolicy.VOLATILE

    subscription_info = driver.get_subscriptions_info_by_topic(INPUT_TOPIC)
    assert len(subscription_info) == 1
    input_qos = subscription_info[0].qos_profile
    assert input_qos.reliability == ReliabilityPolicy.BEST_EFFORT
    assert input_qos.durability == DurabilityPolicy.VOLATILE

    first_stamp = _stamp(123, 456_000_000)
    latest_stamp = _stamp(124, 456_000_000)
    driver.publish_transforms(first_stamp)
    driver.publish_transforms(
        latest_stamp,
        sensor_x=50.0,
        sensor_y=-50.0,
        base_x=100.0,
        base_y=100.0,
    )
    threading.Event().wait(0.2)
    driver.publish_grid(first_stamp)
    output = driver.wait_for_output()
    assert output is not None, process.read_log()
    assert output.header.frame_id == "odom"
    assert _same_stamp(output.header.stamp, first_stamp)
    assert _same_stamp(output.info.map_load_time, first_stamp)
    assert output.info.width == 540
    assert output.info.height == 540
    effective_resolution = output.info.resolution
    assert effective_resolution == pytest.approx(0.1)
    expected_origin = 1.0 - 0.5 * 540.0 * effective_resolution
    assert output.info.origin.position.x == expected_origin
    assert output.info.origin.position.y == expected_origin
    assert output.info.origin.position.z == 0.0
    assert output.info.origin.orientation.x == 0.0
    assert output.info.origin.orientation.y == 0.0
    assert output.info.origin.orientation.z == 0.0
    assert output.info.origin.orientation.w == 1.0
    assert (
        output.info.origin.position.x
        + output.info.width * output.info.resolution
        == 1.0 + 0.5 * 540.0 * effective_resolution
    )
    assert (
        output.info.origin.position.y
        + output.info.height * output.info.resolution
        == 1.0 + 0.5 * 540.0 * effective_resolution
    )
    assert len(output.data) == 540 * 540
    assert output.data[260 * 540 + 260] == 1
    assert output.data[260 * 540 + 261] == 2
    assert output.data[261 * 540 + 260] == 3
    assert output.data[261 * 540 + 261] == 4
    assert output.data[0] == -1
    assert output.data[-1] == -1

    missing_tf_stamp = _stamp(125, 456_000_000)
    driver.clear_outputs()
    driver.publish_grid(missing_tf_stamp)
    driver.assert_no_output()

    driver.publish_transforms(missing_tf_stamp)
    threading.Event().wait(0.1)
    driver.publish_grid(missing_tf_stamp)
    retry_output = driver.wait_for_output()
    assert retry_output is not None, process.read_log()
    assert _same_stamp(retry_output.header.stamp, missing_tf_stamp)

    driver.clear_outputs()
    driver.publish_grid(missing_tf_stamp)
    driver.assert_no_output()

    driver.clear_outputs()
    driver.publish_grid(latest_stamp)
    driver.assert_no_output()

    malformed_stamp = _stamp(126, 456_000_000)
    driver.publish_transforms(malformed_stamp)
    threading.Event().wait(0.1)
    driver.clear_outputs()
    driver.publish_grid(malformed_stamp, data=(1, 2, 3))
    driver.assert_no_output()

    driver.clear_outputs()
    driver.publish_grid(malformed_stamp, orientation_w=0.5)
    driver.assert_no_output()

    driver.clear_outputs()
    driver.publish_grid(malformed_stamp)
    recovered_output = driver.wait_for_output()
    assert recovered_output is not None, process.read_log()
    assert _same_stamp(recovered_output.header.stamp, malformed_stamp)
