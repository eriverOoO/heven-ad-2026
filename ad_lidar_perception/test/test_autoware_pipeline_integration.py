"""Opt-in, fail-closed live gate for the installed Autoware LiDAR graph."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid
import pytest
import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
import yaml


LIVE_ENABLED = os.environ.get("AD_RUN_AUTOWARE_INTEGRATION") == "1"
pytestmark = pytest.mark.skipif(
    not LIVE_ENABLED,
    reason="set AD_RUN_AUTOWARE_INTEGRATION=1 for the live Autoware gate",
)

POINT_XYZIRT_STEP = 22
POINT_XYZIRC_STEP = 16
WAIT_STEP_SEC = 0.02
STARTUP_TIMEOUT_SEC = 30.0
MESSAGE_TIMEOUT_SEC = 20.0


@dataclass(frozen=True)
class Sample:
    received_ns: int
    message: object


class ProcessMonitor:
    """Own and inspect a real ``ros2 launch`` process group."""

    def __init__(self, composition_path):
        ros2 = shutil.which("ros2")
        if ros2 is None:
            pytest.fail("live integration requires the ros2 executable")
        self.command = [
            ros2,
            "launch",
            "ad_lidar_perception",
            "lidar_perception.launch.py",
            f"composition_config:={composition_path}",
            "start_ground_segmentation:=true",
        ]
        self._lock = threading.Lock()
        self._started_by_pid = {}
        self._alive_pids = set()
        self._exited_by_pid = {}
        self._output = []
        try:
            self.process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        except OSError as error:
            pytest.fail(f"could not start installed ROS graph: {error}")
        self._reader_thread = threading.Thread(
            target=self._read_output,
            name="autoware-integration-launch-output",
            daemon=True,
        )
        self._reader_thread.start()
        self.refresh()

    @staticmethod
    def _process_children(pid):
        children_path = Path(
            f"/proc/{pid}/task/{pid}/children"
        )
        try:
            text = children_path.read_text(encoding="utf-8").strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            return []
        return [int(value) for value in text.split()]

    @classmethod
    def _process_tree(cls, root_pid):
        pending = [root_pid]
        observed = set()
        while pending:
            pid = pending.pop()
            if pid in observed:
                continue
            observed.add(pid)
            pending.extend(cls._process_children(pid))
        return observed

    @staticmethod
    def _process_record(pid):
        try:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            return None
        command = tuple(
            token.decode("utf-8", errors="replace")
            for token in raw.split(b"\0")
            if token
        )
        if not command:
            return None
        return {
            "pid": pid,
            "name": Path(command[0]).name,
            "cmd": command,
        }

    @staticmethod
    def _matches(record, executable):
        return any(
            Path(token).name == executable
            for token in record["cmd"]
        )

    def _read_output(self):
        assert self.process.stdout is not None
        for line in self.process.stdout:
            with self._lock:
                self._output.append(line.rstrip())

    def refresh(self):
        current_records = {}
        for pid in self._process_tree(self.process.pid):
            record = self._process_record(pid)
            if record is not None:
                current_records[pid] = record
        with self._lock:
            for pid, record in current_records.items():
                self._started_by_pid.setdefault(pid, record)
            for pid in self._alive_pids - current_records.keys():
                record = dict(self._started_by_pid[pid])
                record["returncode"] = None
                self._exited_by_pid.setdefault(pid, record)
            self._alive_pids = set(current_records)

    def has_started(self, executable):
        self.refresh()
        with self._lock:
            return any(
                self._matches(record, executable)
                for record in self._started_by_pid.values()
            )

    def has_exited(self, executable):
        self.refresh()
        with self._lock:
            return any(
                self._matches(record, executable)
                for record in self._exited_by_pid.values()
            )

    def signal_executable(self, executable, signal_number):
        self.refresh()
        with self._lock:
            records = [
                self._started_by_pid[pid]
                for pid in self._alive_pids
                if self._matches(
                    self._started_by_pid[pid], executable
                )
            ]
        for record in records:
            try:
                os.kill(record["pid"], signal_number)
            except ProcessLookupError:
                pass
        return len(records)

    def is_alive(self):
        return self.process.poll() is None

    def snapshot(self):
        self.refresh()
        with self._lock:
            return {
                "command": list(self.command),
                "returncode": self.process.poll(),
                "started": [
                    dict(record)
                    for record in self._started_by_pid.values()
                ],
                "exited": [
                    dict(record)
                    for record in self._exited_by_pid.values()
                ],
                "output_tail": self._output[-80:],
            }

    def _known_survivors(self):
        with self._lock:
            expected = dict(self._started_by_pid)
        survivors = []
        for pid, expected_record in expected.items():
            actual_record = self._process_record(pid)
            if (
                actual_record is not None
                and actual_record["cmd"] == expected_record["cmd"]
            ):
                survivors.append(pid)
        return survivors

    def _signal_group(self, signal_number):
        try:
            os.killpg(self.process.pid, signal_number)
        except ProcessLookupError:
            pass

    def stop(self):
        self.refresh()
        if self.process.poll() is None:
            self._signal_group(signal.SIGINT)
            try:
                self.process.wait(timeout=8.0)
            except subprocess.TimeoutExpired:
                self._signal_group(signal.SIGTERM)
                try:
                    self.process.wait(timeout=4.0)
                except subprocess.TimeoutExpired:
                    self._signal_group(signal.SIGKILL)
                    self.process.wait(timeout=4.0)
        if self._known_survivors():
            self._signal_group(signal.SIGTERM)
            time.sleep(0.25)
        if self._known_survivors():
            self._signal_group(signal.SIGKILL)
            time.sleep(0.25)
        self._reader_thread.join(timeout=5.0)
        self.refresh()
        if self.process.poll() is None:
            pytest.fail("ros2 launch process survived integration cleanup")
        if self._known_survivors():
            pytest.fail("ROS child process survived integration cleanup")
        if self._reader_thread.is_alive():
            pytest.fail("ros2 launch output thread survived cleanup")


class PipelineProbe(Node):
    """ROS endpoints used to stimulate and observe the installed graph."""

    def __init__(
        self,
        detected_objects_type,
        tracked_objects_type,
        predicted_objects_type,
    ):
        super().__init__("ad_autoware_pipeline_integration_probe")
        reliable = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.messages = {
            "normalized": [],
            "detected": [],
            "tracked": [],
            "predicted": [],
            "static": [],
            "dynamic": [],
            "combined": [],
            "compatibility": [],
        }
        self.send_times_ns = {}
        self.raw_publisher = self.create_publisher(
            PointCloud2,
            "/ad/sensors/lidar/points",
            qos_profile_sensor_data,
        )
        self.detected_publisher = self.create_publisher(
            detected_objects_type,
            "/ad/perception/objects/detected",
            reliable,
        )
        self.predicted_publisher = self.create_publisher(
            predicted_objects_type,
            "/ad/perception/objects/predicted",
            reliable,
        )
        self._subscriptions = [
            self.create_subscription(
                PointCloud2,
                "/ad/perception/lidar/points_xyzirc",
                self._capture("normalized"),
                qos_profile_sensor_data,
            ),
            self.create_subscription(
                detected_objects_type,
                "/ad/perception/objects/detected",
                self._capture("detected"),
                reliable,
            ),
            self.create_subscription(
                tracked_objects_type,
                "/ad/perception/objects/tracked",
                self._capture("tracked"),
                reliable,
            ),
            self.create_subscription(
                predicted_objects_type,
                "/ad/perception/objects/predicted",
                self._capture("predicted"),
                reliable,
            ),
        ]
        for key, topic in (
            ("static", "/ad/perception/occupancy/static"),
            ("dynamic", "/ad/perception/occupancy/dynamic"),
            ("combined", "/ad/perception/occupancy/combined"),
            ("compatibility", "/ad/perception/occupancy_grid"),
        ):
            self._subscriptions.append(
                self.create_subscription(
                    OccupancyGrid,
                    topic,
                    self._capture(key),
                    qos_profile_sensor_data,
                )
            )
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer(cache_time=Duration(seconds=20.0))
        self.tf_listener = TransformListener(
            self.tf_buffer, self, spin_thread=False
        )

    def _capture(self, key):
        def callback(message):
            self.messages[key].append(
                Sample(time.perf_counter_ns(), message)
            )

        return callback

    def publish_transforms(self, stamp):
        odom_to_base = TransformStamped()
        odom_to_base.header.stamp = stamp
        odom_to_base.header.frame_id = "odom"
        odom_to_base.child_frame_id = "base_link"
        odom_to_base.transform.rotation.w = 1.0

        base_to_lidar = TransformStamped()
        base_to_lidar.header.stamp = stamp
        base_to_lidar.header.frame_id = "base_link"
        base_to_lidar.child_frame_id = "lidar_link"
        base_to_lidar.transform.translation.z = 1.70
        base_to_lidar.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform([odom_to_base, base_to_lidar])

    def publish_cloud(self, cloud):
        stamp_ns = _stamp_ns(cloud.header.stamp)
        sent_ns = time.perf_counter_ns()
        self.send_times_ns[stamp_ns] = sent_ns
        self.raw_publisher.publish(cloud)
        return sent_ns

    def publish_detected(self, message):
        stamp_ns = _stamp_ns(message.header.stamp)
        sent_ns = time.perf_counter_ns()
        self.send_times_ns[stamp_ns] = sent_ns
        self.detected_publisher.publish(message)
        return sent_ns

    def publish_predicted(self, message):
        stamp_ns = _stamp_ns(message.header.stamp)
        sent_ns = time.perf_counter_ns()
        self.send_times_ns[stamp_ns] = sent_ns
        self.predicted_publisher.publish(message)
        return sent_ns


def _stamp_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _wait_until(predicate, timeout, failure):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(WAIT_STEP_SEC)
    pytest.fail(failure)


def _strict_input_fields():
    def field(name, offset, datatype):
        value = PointField()
        value.name = name
        value.offset = offset
        value.datatype = datatype
        value.count = 1
        return value

    return [
        field("x", 0, PointField.FLOAT32),
        field("y", 4, PointField.FLOAT32),
        field("z", 8, PointField.FLOAT32),
        field("intensity", 12, PointField.FLOAT32),
        field("ring", 16, PointField.UINT16),
        field("time", 18, PointField.FLOAT32),
    ]


def _dense_cloud(stamp, point_count=2048):
    cloud = PointCloud2()
    cloud.header.stamp = stamp
    cloud.header.frame_id = "lidar_link"
    cloud.height = 1
    cloud.width = point_count
    cloud.fields = _strict_input_fields()
    cloud.is_bigendian = False
    cloud.point_step = POINT_XYZIRT_STEP
    cloud.row_step = cloud.point_step * cloud.width
    cloud.is_dense = True
    payload = bytearray()
    for index in range(point_count):
        x = 7.0 + 0.08 * float(index % 32)
        y = -3.0 + 0.08 * float((index // 32) % 64)
        z = 0.25 + 0.12 * float(index % 8)
        intensity = 20.0 + float(index % 180)
        ring = index % 16
        relative_time = float(index) * 1.0e-5
        payload.extend(
            struct.pack(
                "<ffffHf",
                x,
                y,
                z,
                intensity,
                ring,
                relative_time,
            )
        )
    cloud.data = bytes(payload)
    return cloud


def _empty_cloud(stamp):
    cloud = _dense_cloud(stamp, point_count=0)
    assert cloud.row_step == 0
    assert not cloud.data
    return cloud


def _malformed_cloud(stamp):
    cloud = _dense_cloud(stamp, point_count=1)
    cloud.point_step = POINT_XYZIRT_STEP - 1
    cloud.row_step = cloud.point_step
    cloud.data = cloud.data[: cloud.point_step]
    return cloud


def _assert_point_xyzirc(cloud, source):
    assert [
        (field.name, field.offset, field.datatype, field.count)
        for field in cloud.fields
    ] == [
        ("x", 0, PointField.FLOAT32, 1),
        ("y", 4, PointField.FLOAT32, 1),
        ("z", 8, PointField.FLOAT32, 1),
        ("intensity", 12, PointField.UINT8, 1),
        ("return_type", 13, PointField.UINT8, 1),
        ("channel", 14, PointField.UINT16, 1),
    ]
    assert cloud.point_step == POINT_XYZIRC_STEP
    assert cloud.header.stamp == source.header.stamp
    assert cloud.header.frame_id == source.header.frame_id
    assert cloud.row_step == cloud.width * POINT_XYZIRC_STEP
    assert len(cloud.data) == cloud.row_step * cloud.height


def _sample_for_stamp(samples, stamp_ns, start_index=0):
    for sample in samples[start_index:]:
        if _stamp_ns(sample.message.header.stamp) == stamp_ns:
            return sample
    return None


def _gpu_sample():
    executable = shutil.which("nvidia-smi")
    if executable is None:
        pytest.fail(
            "AD_RUN_AUTOWARE_INTEGRATION=1 requires real nvidia-smi metrics"
        )
    command = [
        executable,
        "--query-gpu=uuid,name,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        pytest.fail(f"could not record real GPU metrics: {error}")
    rows = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 3)]
        if len(fields) != 4:
            pytest.fail(f"malformed nvidia-smi metric row: {line!r}")
        try:
            utilization = int(fields[2])
            memory_mib = int(fields[3])
        except ValueError:
            pytest.fail(f"nonnumeric nvidia-smi metric row: {line!r}")
        rows.append(
            {
                "uuid": fields[0],
                "name": fields[1],
                "utilization_percent": utilization,
                "memory_used_mib": memory_mib,
            }
        )
    if not rows:
        pytest.fail("nvidia-smi returned no GPU metrics")
    return rows


def _write_composition(path):
    document = {
        "schema_version": 1,
        "detector": {
            "backend": "centerpoint_tiny",
            "model_subdir": "models/autoware",
            "build_only": False,
        },
        "tracker": {"backend": "autoware"},
        "occupancy": {
            "static_enabled": True,
            "dynamic_enabled": True,
            "publish_combined": True,
        },
    }
    path.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )


def _require_optional_environment(composition_path):
    if os.environ.get("AD_AUTOWARE_MODEL_LICENSE_REVIEWED") != "1":
        pytest.fail(
            "live integration requires "
            "AD_AUTOWARE_MODEL_LICENSE_REVIEWED=1"
        )
    if not os.environ.get("AD_DATA_DIR"):
        pytest.fail("live integration requires an explicit AD_DATA_DIR")

    try:
        from ad_lidar_perception.autoware_provenance import (
            VerificationError,
            verify_selection,
        )
        from ad_lidar_perception.selection import load_selection
        from ad_interfaces.msg import PredictedObjectArray
        from autoware_perception_msgs.msg import (
            DetectedObject,
            DetectedObjectKinematics,
            DetectedObjects,
            ObjectClassification,
            Shape,
            TrackedObjects,
        )
    except (ImportError, ModuleNotFoundError) as error:
        pytest.fail(
            "live integration was requested but the optional typed overlay "
            f"is unavailable: {error}"
        )

    package_share = Path(
        get_package_share_directory("ad_lidar_perception")
    )
    selection = load_selection(composition_path)
    try:
        verified = verify_selection(
            selection,
            lock_path=(
                package_share
                / "config"
                / "autoware_perception.lock.yaml"
            ),
        )
    except VerificationError as error:
        pytest.fail(
            "live integration preflight rejected the installed graph; "
            f"no inference result may be claimed:\n{error}"
        )
    if verified.detector is None or verified.tracker is None:
        pytest.fail("live integration did not resolve detector and tracker")
    return {
        "package_share": package_share,
        "selection": selection,
        "verified": verified,
        "DetectedObject": DetectedObject,
        "DetectedObjectKinematics": DetectedObjectKinematics,
        "DetectedObjects": DetectedObjects,
        "ObjectClassification": ObjectClassification,
        "PredictedObjectArray": PredictedObjectArray,
        "Shape": Shape,
        "TrackedObjects": TrackedObjects,
    }


def _detected_message(types, stamp):
    message = types["DetectedObjects"]()
    message.header.stamp = stamp
    message.header.frame_id = "lidar_link"

    detected = types["DetectedObject"]()
    detected.existence_probability = 0.95
    classification = types["ObjectClassification"]()
    classification.label = types["ObjectClassification"].CAR
    classification.probability = 0.90
    detected.classification = [classification]
    detected.kinematics.pose_with_covariance.pose.position.x = 10.0
    detected.kinematics.pose_with_covariance.pose.position.y = 1.5
    detected.kinematics.pose_with_covariance.pose.position.z = 0.5
    detected.kinematics.pose_with_covariance.pose.orientation.w = 1.0
    detected.kinematics.pose_with_covariance.covariance[0] = 0.20
    detected.kinematics.pose_with_covariance.covariance[7] = 0.20
    detected.kinematics.pose_with_covariance.covariance[35] = 0.05
    detected.kinematics.has_position_covariance = True
    detected.kinematics.orientation_availability = (
        types["DetectedObjectKinematics"].AVAILABLE
    )
    detected.kinematics.has_twist = False
    detected.kinematics.has_twist_covariance = False
    detected.shape.type = types["Shape"].BOUNDING_BOX
    detected.shape.dimensions.x = 4.5
    detected.shape.dimensions.y = 1.9
    detected.shape.dimensions.z = 1.6
    message.objects = [detected]
    return message


def _restamped_prediction(types, source, stamp):
    message = types["PredictedObjectArray"]()
    message.header.stamp = stamp
    message.header.frame_id = source.header.frame_id
    message.objects = source.objects
    return message


def _assert_launch_alive(monitor):
    if not monitor.is_alive():
        pytest.fail(
            "installed Autoware graph stopped unexpectedly: "
            f"{monitor.snapshot()}"
        )


def _grid_equal(left, right):
    return (
        left.header.frame_id == right.header.frame_id
        and left.header.stamp == right.header.stamp
        and left.info == right.info
        and left.data == right.data
    )


def _latency_ms(sample, sent_ns):
    latency = (sample.received_ns - sent_ns) / 1_000_000.0
    assert latency >= 0.0
    return latency


def test_live_installed_autoware_pipeline_contract(
    tmp_path, record_property
):
    """Exercise inference, tracker failure, and occupancy safety end to end."""
    composition_path = tmp_path / "live-autoware-composition.yaml"
    _write_composition(composition_path)
    types = _require_optional_environment(composition_path)
    verified = types["verified"]
    detector_executable = verified.detector.executable
    tracker_executable = verified.tracker.executable
    expected_processes = (
        "patchworkpp_node",
        "ad_lidar_perception_node",
        "ad_combined_occupancy_grid_node",
        "ad_point_layout_adapter_node",
        detector_executable,
        tracker_executable,
        "ad_autoware_prediction_node",
        "ad_dynamic_occupancy_grid_node",
    )

    gpu_before = _gpu_sample()
    monitor = None
    probe = None
    executor = None
    executor_thread = None
    rclpy.init()
    try:
        probe = PipelineProbe(
            types["DetectedObjects"],
            types["TrackedObjects"],
            types["PredictedObjectArray"],
        )
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(probe)
        executor_thread = threading.Thread(
            target=executor.spin,
            name="autoware-integration-probe",
            daemon=True,
        )
        executor_thread.start()
        monitor = ProcessMonitor(composition_path)
        _wait_until(
            lambda: all(
                monitor.has_started(executable)
                for executable in expected_processes
            ),
            STARTUP_TIMEOUT_SEC,
            "installed graph did not start every required process: "
            f"{expected_processes!r}; observed={monitor.snapshot()}",
        )
        time.sleep(0.5)
        _assert_launch_alive(monitor)
        assert not monitor.snapshot()["exited"]

        _wait_until(
            lambda: probe.raw_publisher.get_subscription_count() >= 2,
            10.0,
            "raw cloud did not connect to static and adapter branches",
        )
        _wait_until(
            lambda: probe.detected_publisher.get_subscription_count() >= 1,
            10.0,
            "synthetic detector publisher did not connect to tracker",
        )

        stamp = probe.get_clock().now().to_msg()
        probe.publish_transforms(stamp)
        _wait_until(
            lambda: probe.tf_buffer.can_transform(
                "odom",
                "base_link",
                Time.from_msg(stamp),
                timeout=Duration(seconds=0.05),
            )
            and probe.tf_buffer.can_transform(
                "base_link",
                "lidar_link",
                Time.from_msg(stamp),
                timeout=Duration(seconds=0.05),
            ),
            3.0,
            "stamped odom/base_link/lidar_link TF chain was unavailable",
        )
        odom_to_base = probe.tf_buffer.lookup_transform(
            "odom",
            "base_link",
            Time.from_msg(stamp),
            timeout=Duration(seconds=1.0),
        )
        base_to_lidar = probe.tf_buffer.lookup_transform(
            "base_link",
            "lidar_link",
            Time.from_msg(stamp),
            timeout=Duration(seconds=1.0),
        )
        assert odom_to_base.header.stamp == stamp
        assert base_to_lidar.header.stamp == stamp
        assert odom_to_base.child_frame_id == "base_link"
        assert base_to_lidar.child_frame_id == "lidar_link"

        dense = _dense_cloud(stamp)
        dense_sent_ns = probe.publish_cloud(dense)
        dense_stamp_ns = _stamp_ns(stamp)
        _wait_until(
            lambda: _sample_for_stamp(
                probe.messages["normalized"], dense_stamp_ns
            )
            is not None,
            5.0,
            "PointXYZIRC adapter produced no output",
        )
        normalized = _sample_for_stamp(
            probe.messages["normalized"], dense_stamp_ns
        )
        _assert_point_xyzirc(normalized.message, dense)

        empty_stamp = probe.get_clock().now().to_msg()
        probe.publish_transforms(empty_stamp)
        empty = _empty_cloud(empty_stamp)
        empty_stamp_ns = _stamp_ns(empty_stamp)
        empty_start = len(probe.messages["normalized"])
        probe.publish_cloud(empty)
        _wait_until(
            lambda: _sample_for_stamp(
                probe.messages["normalized"],
                empty_stamp_ns,
                empty_start,
            )
            is not None,
            5.0,
            "valid empty cloud was not converted safely",
        )
        normalized_empty = _sample_for_stamp(
            probe.messages["normalized"], empty_stamp_ns, empty_start
        )
        _assert_point_xyzirc(normalized_empty.message, empty)
        assert normalized_empty.message.width == 0

        malformed_stamp = probe.get_clock().now().to_msg()
        probe.publish_transforms(malformed_stamp)
        malformed = _malformed_cloud(malformed_stamp)
        malformed_stamp_ns = _stamp_ns(malformed_stamp)
        malformed_start = len(probe.messages["normalized"])
        probe.publish_cloud(malformed)
        time.sleep(0.75)
        assert (
            _sample_for_stamp(
                probe.messages["normalized"],
                malformed_stamp_ns,
                malformed_start,
            )
            is None
        )
        assert not monitor.has_exited(detector_executable)
        assert not monitor.has_exited("ad_point_layout_adapter_node")
        _assert_launch_alive(monitor)

        detection_start = len(probe.messages["detected"])
        deadline = time.monotonic() + MESSAGE_TIMEOUT_SEC
        while (
            time.monotonic() < deadline
            and len(probe.messages["detected"]) == detection_start
        ):
            inference_stamp = probe.get_clock().now().to_msg()
            probe.publish_transforms(inference_stamp)
            time.sleep(0.04)
            inference_cloud = _dense_cloud(inference_stamp)
            probe.publish_cloud(inference_cloud)
            time.sleep(0.12)
            _assert_launch_alive(monitor)
            assert not monitor.has_exited(detector_executable)
        assert len(probe.messages["detected"]) > detection_start, (
            "no real detector output was observed; inference is unavailable"
        )
        detector_sample = probe.messages["detected"][detection_start]
        actual_detector_messages = list(
            probe.messages["detected"][detection_start:]
        )
        detector_count_peak = max(
            len(sample.message.objects)
            for sample in actual_detector_messages
        )

        tracked_start = len(probe.messages["tracked"])
        predicted_start = len(probe.messages["predicted"])
        dynamic_start = len(probe.messages["dynamic"])
        for _ in range(6):
            detected_stamp = probe.get_clock().now().to_msg()
            probe.publish_transforms(detected_stamp)
            time.sleep(0.04)
            probe.publish_detected(
                _detected_message(types, detected_stamp)
            )
            time.sleep(0.12)
        _wait_until(
            lambda: any(
                sample.message.objects
                for sample in probe.messages["tracked"][tracked_start:]
            ),
            MESSAGE_TIMEOUT_SEC,
            "actual Autoware tracker produced no nonempty track",
        )
        _wait_until(
            lambda: any(
                sample.message.objects
                for sample in probe.messages["predicted"][predicted_start:]
            ),
            MESSAGE_TIMEOUT_SEC,
            "HEVEN adapter produced no nonempty prediction",
        )
        _wait_until(
            lambda: any(
                any(cell > 0 for cell in sample.message.data)
                for sample in probe.messages["dynamic"][dynamic_start:]
            ),
            5.0,
            "tracked prediction produced no occupied dynamic layer",
        )
        tracked_sample = next(
            sample
            for sample in probe.messages["tracked"][tracked_start:]
            if sample.message.objects
        )
        predicted_sample = next(
            sample
            for sample in probe.messages["predicted"][predicted_start:]
            if sample.message.objects
        )
        dropout_tracked_start = len(probe.messages["tracked"])
        dropout_predicted_start = len(probe.messages["predicted"])
        dropout_send_times_ns = {}
        for _ in range(15):
            dropout_stamp = probe.get_clock().now().to_msg()
            dropout_stamp_ns = _stamp_ns(dropout_stamp)
            assert not probe.tf_buffer.can_transform(
                "odom",
                "lidar_link",
                Time.from_msg(dropout_stamp),
                timeout=Duration(seconds=0.01),
            )
            dropout_message = _detected_message(types, dropout_stamp)
            assert dropout_message.objects
            dropout_send_times_ns[dropout_stamp_ns] = (
                probe.publish_detected(dropout_message)
            )
            time.sleep(0.12)

        def clearing_pair():
            for tracked in probe.messages["tracked"][
                dropout_tracked_start:
            ]:
                if tracked.message.objects:
                    continue
                stamp_ns = _stamp_ns(tracked.message.header.stamp)
                predicted = _sample_for_stamp(
                    probe.messages["predicted"],
                    stamp_ns,
                    dropout_predicted_start,
                )
                if (
                    stamp_ns in dropout_send_times_ns
                    and predicted is not None
                    and not predicted.message.objects
                ):
                    return tracked, predicted
            return None

        _wait_until(
            lambda: clearing_pair() is not None,
            5.0,
            "nonempty detections without stamped TF did not lead to "
            "causally matched empty tracked and predicted arrays",
        )
        clearing_track, clearing_prediction = clearing_pair()
        assert not clearing_track.message.objects
        assert not clearing_prediction.message.objects
        assert (
            clearing_track.message.header.stamp
            == clearing_prediction.message.header.stamp
        )
        clearing_stamp_ns = _stamp_ns(
            clearing_prediction.message.header.stamp
        )

        assert (
            monitor.signal_executable(
                "ad_autoware_prediction_node", signal.SIGTERM
            )
            == 1
        )
        _wait_until(
            lambda: monitor.has_exited(
                "ad_autoware_prediction_node"
            ),
            5.0,
            "could not stop prediction production for the stale-layer test",
        )
        time.sleep(0.25)
        prediction_quiet_start = len(probe.messages["predicted"])
        time.sleep(0.25)
        assert len(probe.messages["predicted"]) == prediction_quiet_start

        stale_epoch_ns = time.perf_counter_ns()
        stale_stamp = probe.get_clock().now().to_msg()
        stale_stamp_ns = _stamp_ns(stale_stamp)
        probe.publish_transforms(stale_stamp)
        time.sleep(0.04)
        stale_dynamic_start = len(probe.messages["dynamic"])
        stale_prediction_start = len(probe.messages["predicted"])
        probe.publish_predicted(
            _restamped_prediction(
                types, predicted_sample.message, stale_stamp
            )
        )
        _wait_until(
            lambda: (
                (
                    sample := _sample_for_stamp(
                        probe.messages["dynamic"],
                        stale_stamp_ns,
                        stale_dynamic_start,
                    )
                )
                is not None
                and any(cell > 0 for cell in sample.message.data)
            ),
            5.0,
            "controlled final prediction did not repopulate dynamic occupancy",
        )
        stale_occupied = _sample_for_stamp(
            probe.messages["dynamic"],
            stale_stamp_ns,
            stale_dynamic_start,
        )
        stale_index = (
            probe.messages["dynamic"].index(stale_occupied) + 1
        )
        _wait_until(
            lambda: any(
                sample.received_ns - stale_epoch_ns >= 500_000_000
                and all(cell == 0 for cell in sample.message.data)
                for sample in probe.messages["dynamic"][stale_index:]
            ),
            3.0,
            "dynamic stale timer did not clear at least 0.50 s after "
            "the final controlled prediction epoch",
        )
        stale_clear = next(
            sample
            for sample in probe.messages["dynamic"][stale_index:]
            if sample.received_ns - stale_epoch_ns >= 500_000_000
            and all(cell == 0 for cell in sample.message.data)
        )
        replayed_predictions = probe.messages["predicted"][
            stale_prediction_start:
        ]
        assert len(replayed_predictions) == 1
        assert (
            _stamp_ns(replayed_predictions[0].message.header.stamp)
            == stale_stamp_ns
        )

        fallback_static_start = len(probe.messages["static"])
        fallback_combined_start = len(probe.messages["combined"])
        fallback_compatibility_start = len(
            probe.messages["compatibility"]
        )
        fallback_stamp = probe.get_clock().now().to_msg()
        fallback_stamp_ns = _stamp_ns(fallback_stamp)
        probe.publish_transforms(fallback_stamp)
        time.sleep(0.08)
        probe.publish_cloud(_dense_cloud(fallback_stamp))
        _wait_until(
            lambda: (
                _sample_for_stamp(
                    probe.messages["static"],
                    fallback_stamp_ns,
                    fallback_static_start,
                )
                is not None
                and _sample_for_stamp(
                    probe.messages["combined"],
                    fallback_stamp_ns,
                    fallback_combined_start,
                )
                is not None
                and _sample_for_stamp(
                    probe.messages["compatibility"],
                    fallback_stamp_ns,
                    fallback_compatibility_start,
                )
                is not None
            ),
            5.0,
            "static-to-combined fallback did not publish every safety topic",
        )
        fallback_static = _sample_for_stamp(
            probe.messages["static"],
            fallback_stamp_ns,
            fallback_static_start,
        ).message
        fallback_combined = _sample_for_stamp(
            probe.messages["combined"],
            fallback_stamp_ns,
            fallback_combined_start,
        ).message
        fallback_compatibility = _sample_for_stamp(
            probe.messages["compatibility"],
            fallback_stamp_ns,
            fallback_compatibility_start,
        ).message
        assert _grid_equal(fallback_static, fallback_combined)
        assert _grid_equal(fallback_static, fallback_compatibility)

        detector_failure_ns = time.perf_counter_ns()
        assert (
            monitor.signal_executable(
                detector_executable, signal.SIGTERM
            )
            == 1
        )
        _wait_until(
            lambda: monitor.has_exited(detector_executable),
            5.0,
            "could not induce and observe detector process failure",
        )
        static_before_failure_probe = len(probe.messages["static"])
        compatibility_before_failure_probe = len(
            probe.messages["compatibility"]
        )
        post_failure_stamp = probe.get_clock().now().to_msg()
        post_failure_stamp_ns = _stamp_ns(post_failure_stamp)
        probe.publish_transforms(post_failure_stamp)
        time.sleep(0.08)
        probe.publish_cloud(_dense_cloud(post_failure_stamp))
        _wait_until(
            lambda: (
                _sample_for_stamp(
                    probe.messages["static"],
                    post_failure_stamp_ns,
                    static_before_failure_probe,
                )
                is not None
                and _sample_for_stamp(
                    probe.messages["compatibility"],
                    post_failure_stamp_ns,
                    compatibility_before_failure_probe,
                )
                is not None
            ),
            5.0,
            "detector failure suppressed static occupancy or its alias",
        )
        post_failure_static = _sample_for_stamp(
            probe.messages["static"],
            post_failure_stamp_ns,
            static_before_failure_probe,
        )
        post_failure_compatibility = _sample_for_stamp(
            probe.messages["compatibility"],
            post_failure_stamp_ns,
            compatibility_before_failure_probe,
        )
        assert post_failure_static.received_ns > detector_failure_ns
        assert (
            post_failure_compatibility.received_ns
            > detector_failure_ns
        )
        _assert_launch_alive(monitor)

        gpu_after = _gpu_sample()
        detector_stamp_ns = _stamp_ns(
            detector_sample.message.header.stamp
        )
        tracked_stamp_ns = _stamp_ns(tracked_sample.message.header.stamp)
        predicted_stamp_ns = _stamp_ns(
            predicted_sample.message.header.stamp
        )
        missing_metric_stamps = [
            stamp_ns
            for stamp_ns in (
                detector_stamp_ns,
                tracked_stamp_ns,
                predicted_stamp_ns,
            )
            if stamp_ns not in probe.send_times_ns
        ]
        if missing_metric_stamps:
            pytest.fail(
                "output stamps did not map to recorded graph inputs; "
                "latency metrics would be guessed: "
                f"{missing_metric_stamps!r}"
            )
        metrics = {
            "adapter_latency_ms": _latency_ms(
                normalized, dense_sent_ns
            ),
            "detector_latency_ms": _latency_ms(
                detector_sample,
                probe.send_times_ns[detector_stamp_ns],
            ),
            "tracker_latency_ms": _latency_ms(
                tracked_sample,
                probe.send_times_ns[tracked_stamp_ns],
            ),
            "prediction_latency_ms": _latency_ms(
                predicted_sample,
                probe.send_times_ns[predicted_stamp_ns],
            ),
            "tf_dropout_clear_latency_ms": _latency_ms(
                clearing_prediction,
                dropout_send_times_ns[clearing_stamp_ns],
            ),
            "dynamic_stale_clear_latency_ms": (
                stale_clear.received_ns - stale_epoch_ns
            )
            / 1_000_000.0,
            "detector_output_messages": len(actual_detector_messages),
            "detector_object_count_peak": detector_count_peak,
            "tracker_track_count_peak": max(
                len(sample.message.objects)
                for sample in probe.messages["tracked"][tracked_start:]
            ),
            "prediction_object_count_peak": max(
                len(sample.message.objects)
                for sample in probe.messages["predicted"][predicted_start:]
            ),
            "clearing_prediction_objects": len(
                clearing_prediction.message.objects
            ),
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
            "processes": monitor.snapshot(),
        }
        metric_text = json.dumps(metrics, sort_keys=True)
        metric_path_text = os.environ.get(
            "AD_AUTOWARE_INTEGRATION_METRICS_FILE", ""
        )
        metric_path = (
            Path(metric_path_text).expanduser()
            if metric_path_text
            else Path.cwd() / "autoware_pipeline_metrics.json"
        )
        metric_path.parent.mkdir(parents=True, exist_ok=True)
        metric_path.write_text(metric_text + "\n", encoding="utf-8")
        record_property("autoware_pipeline_metrics", metric_text)
        record_property(
            "autoware_pipeline_metrics_file", str(metric_path)
        )
        print(f"AUTOWARE_PIPELINE_METRICS={metric_text}")
    finally:
        body_failed = sys.exc_info()[0] is not None
        cleanup_errors = []

        def attempt(label, action):
            try:
                action()
            except BaseException as error:
                cleanup_errors.append(f"{label}: {error!r}")

        if monitor is not None:
            attempt("launch process group", monitor.stop)
        if executor is not None:
            def stop_executor():
                if not executor.shutdown(timeout_sec=5.0):
                    raise RuntimeError("executor did not stop within 5 s")

            attempt("probe executor", stop_executor)
        if executor_thread is not None:
            def join_executor_thread():
                executor_thread.join(timeout=5.0)
                if executor_thread.is_alive():
                    raise RuntimeError(
                        "probe executor thread survived cleanup"
                    )

            attempt("probe executor thread", join_executor_thread)
        if probe is not None:
            attempt("probe node", probe.destroy_node)
        if rclpy.ok():
            attempt("rclpy context", rclpy.shutdown)

        if cleanup_errors:
            diagnostic = "integration cleanup failures: " + "; ".join(
                cleanup_errors
            )
            if body_failed:
                print(diagnostic)
            else:
                pytest.fail(diagnostic)
