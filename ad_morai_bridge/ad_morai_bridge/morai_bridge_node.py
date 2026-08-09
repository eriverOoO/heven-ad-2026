from dataclasses import dataclass, replace
import math
import queue
import threading
import time
from typing import Any

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Imu, NavSatFix, TimeReference

from ad_morai_interfaces.msg import (
    BridgeStatistics,
    CollisionArray,
    CtrlCmd,
    EgoVehicleStatus,
    GpsGga,
    GpsRmc,
    ImuPacket,
    RawPacket,
    SensorTiming,
)

from .codecs.camera import CameraAssembler
from .codecs.common import PacketFormatError
from .codecs.ego import decode_collisions, decode_ego_status, encode_ctrl_cmd
from .codecs.gps import GpsFixAccumulator, decode_nmea, rmc_epoch_stamp
from .codecs.imu import decode_imu
from .codecs.lidar import validate_velodyne_packet
from .arrival_time import ReceiptClockMapper
from .control_watchdog import ControlSafetyGate
from .message_conversion import (
    collision_array_message,
    ego_status_message,
    gps_fix_message,
    gps_gga_message,
    gps_rmc_message,
    gps_time_reference_message,
    imu_message,
    imu_packet_message,
    sensor_timing_message,
)
from .protocol_records import CtrlCommandRecord, GpggaRecord, GprmcRecord
from .stream_defaults import STREAMS, StreamDefaults
from .timestamp_policy import (
    TimestampDecision,
    TimestampPolicy,
)
from .udp_transport import UdpEndpoint, UdpReceiver, UdpSender


@dataclass
class StreamCounters:
    packets: int = 0
    bytes: int = 0
    malformed: int = 0
    dropped: int = 0
    bind_errors: int = 0
    source_selected: int = 0
    arrival_fallback: int = 0
    source_rejected: int = 0
    duplicates: int = 0
    stamp_regressions: int = 0
    first_at: float | None = None
    last_at: float | None = None


def diagnostic_state(
    enabled: bool,
    counter: StreamCounters,
    age: float,
    receiver_alive: bool,
    stale_after_sec: float,
) -> tuple[int, str]:
    if not enabled:
        return DiagnosticStatus.STALE, "disabled"
    if counter.bind_errors:
        return DiagnosticStatus.ERROR, "UDP bind failed"
    if not receiver_alive:
        return DiagnosticStatus.ERROR, "UDP receiver stopped"
    if counter.packets == 0:
        return DiagnosticStatus.WARN, "waiting for UDP"
    if age > stale_after_sec:
        return DiagnosticStatus.WARN, "UDP stream stale"
    if counter.malformed or counter.dropped:
        return DiagnosticStatus.WARN, "receiving with packet errors"
    return DiagnosticStatus.OK, "receiving"


class AdMoraiBridge(Node):
    def __init__(self, *, parameter_overrides=None):
        super().__init__("ad_morai_bridge", parameter_overrides=parameter_overrides)
        self._receivers: dict[str, UdpReceiver] = {}
        self._senders: list[UdpSender] = []
        self._resources_closed = False
        self._control_enabled = False
        self._control_sender = None
        self._control_gate = None
        self._control_subscription = None
        self._control_target: UdpEndpoint | None = None
        self._control_topic: str | None = None
        try:
            self._initialize()
        except BaseException:
            self._cleanup_resources(send_emergency=False)
            try:
                super().destroy_node()
            except BaseException:
                pass
            raise

    def _initialize(self) -> None:
        if self.has_parameter("use_sim_time") and bool(
            self.get_parameter("use_sim_time").value
        ):
            raise ValueError(
                "use_sim_time is unsupported: UDP receipt timestamps require "
                "one stable system ROS clock"
            )
        queue_capacity = int(self.declare_parameter("queue_capacity", 4096).value)
        if queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")
        self._queue: queue.Queue[tuple[str, bytes, float]] = queue.Queue(
            maxsize=queue_capacity
        )
        self._lock = threading.Lock()
        self._stream_cfg: dict[str, dict[str, Any]] = {}
        self._counters = {name: StreamCounters() for name in STREAMS}
        self._stream_publishers: dict[str, Any] = {}
        self._raw_publishers: dict[str, Any] = {}
        self._camera_assemblers: dict[str, CameraAssembler] = {}
        self._gps = GpsFixAccumulator()
        self._control_send_errors = 0

        self._timestamp_mode = str(
            self.declare_parameter("timestamp_mode", "arrival").value
        )
        self._source_stamp_tolerance_sec = float(
            self.declare_parameter("source_stamp_tolerance_sec", 1.0).value
        )
        self._timestamp_policies = {
            name: TimestampPolicy(
                mode=self._timestamp_mode,
                tolerance_sec=self._source_stamp_tolerance_sec,
                # MORAI does not expose an authoritative retransmission flag or
                # sequence contract.  A repeated source stamp is audit evidence,
                # never sufficient grounds to delete a normalized sample.
                suppress_source_duplicates=False,
            )
            for name, defaults in STREAMS.items()
            if defaults.mode != "velodyne"
        }
        sampled_monotonic = time.monotonic()
        self._receipt_clock = ReceiptClockMapper(
            sampled_monotonic=sampled_monotonic,
            sampled_ros_ns=self.get_clock().now().nanoseconds,
        )
        self._publish_raw_packets = bool(
            self.declare_parameter("publish_raw_packets", False).value
        )
        self._read_stream_configs()
        self._read_control_config()

        reliable_stats = QoSProfile(
            depth=20, reliability=ReliabilityPolicy.RELIABLE
        )
        reliable_diagnostics = QoSProfile(
            depth=10, reliability=ReliabilityPolicy.RELIABLE
        )
        self._stats_pub = self.create_publisher(
            BridgeStatistics, "/ad/udp/statistics", reliable_stats
        )
        self._diagnostics_pub = self.create_publisher(
            DiagnosticArray, "/diagnostics", reliable_diagnostics
        )
        self._timing_pub = self.create_publisher(
            SensorTiming, "/ad/sensors/timing", qos_profile_sensor_data
        )
        for name, config in self._stream_cfg.items():
            self._create_stream_publishers(
                name, config, self._publish_raw_packets
            )
            if bool(config["enabled"]):
                self._start_receiver(name, config)

        self._setup_control()
        self._timers = [
            self.create_timer(0.002, self._drain_queue),
            self.create_timer(0.05, self._watchdog_tick),
            self.create_timer(1.0, self._publish_statistics),
        ]

    def _read_stream_configs(self) -> None:
        for name, defaults in STREAMS.items():
            enabled = bool(
                self.declare_parameter(f"{name}.enabled", defaults.enabled).value
            )
            bind_ip = str(
                self.declare_parameter(f"{name}.bind_ip", "127.0.0.1").value
            )
            port = int(
                self.declare_parameter(f"{name}.port", defaults.port).value
            )
            # Convention: every bridge topic lives under /ad/.
            topic = str(
                self.declare_parameter(f"{name}.topic", defaults.topic).value
            )
            config: dict[str, Any] = {
                "enabled": enabled,
                "bind_ip": bind_ip,
                "port": port,
                "endpoint": UdpEndpoint(bind_ip, port),
                "topic": topic,
                "frame_id": str(
                    self.declare_parameter(
                        f"{name}.frame_id", defaults.frame_id
                    ).value
                ),
                "receive_buffer_bytes": int(
                    self.declare_parameter(
                        f"{name}.receive_buffer_bytes",
                        defaults.receive_buffer_bytes,
                    ).value
                ),
                "mode": defaults.mode,
                "stale_after_sec": float(
                    self.declare_parameter(f"{name}.stale_after_sec", 1.0).value
                ),
            }
            if int(config["receive_buffer_bytes"]) <= 0:
                raise ValueError(f"{name}.receive_buffer_bytes must be positive")
            stale_after_sec = float(config["stale_after_sec"])
            if not math.isfinite(stale_after_sec) or stale_after_sec <= 0.0:
                raise ValueError(f"{name}.stale_after_sec must be finite and positive")
            self._stream_cfg[name] = config

    def _read_control_config(self) -> None:
        self._control_enabled = bool(
            self.declare_parameter("control.enabled", False).value
        )
        self._control_topic = str(
            self.declare_parameter("control.topic", "/ad/control/command").value
        )
        if not self._control_enabled:
            return

        self._control_target = UdpEndpoint(
            str(
                self.declare_parameter(
                    "control.target_ip", "127.0.0.1"
                ).value
            ),
            int(self.declare_parameter("control.target_port", 9093).value),
        )
        timeout_sec = float(
            self.declare_parameter(
                "control.watchdog_timeout_sec", 0.2
            ).value
        )
        fallback_interval_sec = float(
            self.declare_parameter(
                "control.fallback_interval_sec", 0.1
            ).value
        )
        self._control_gate = ControlSafetyGate(
            self._send_control_record,
            timeout_sec=timeout_sec,
            fallback_interval_sec=fallback_interval_sec,
        )

    @staticmethod
    def _message_type(mode: str):
        if mode == "ego":
            return EgoVehicleStatus
        if mode == "collisions":
            return CollisionArray
        if mode == "camera":
            return CompressedImage
        if mode == "gps":
            return NavSatFix
        if mode == "imu":
            return Imu
        if mode == "velodyne":
            return RawPacket
        raise ValueError(f"unsupported stream mode {mode!r}")

    @staticmethod
    def _stream_qos(mode: str) -> QoSProfile:
        # A VLP-16 revolution is assembled from roughly 75 UDP packets at
        # 10 Hz.  ROS' generic sensor profile keeps only five samples, which
        # can discard most of a burst before the adapter builds the scan.
        if mode == "velodyne":
            return QoSProfile(
                depth=100,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            )
        return qos_profile_sensor_data

    def _create_stream_publishers(
        self, name: str, config: dict[str, Any], publish_raw: bool
    ) -> None:
        mode = str(config["mode"])
        self._stream_publishers[name] = self.create_publisher(
            self._message_type(mode),
            str(config["topic"]),
            self._stream_qos(mode),
        )
        if mode == "camera":
            self._camera_assemblers[name] = CameraAssembler(
                max_frame_bytes=int(config["receive_buffer_bytes"])
            )
        elif mode == "gps":
            self._gps_rmc_pub = self.create_publisher(
                GpsRmc, "/ad/sensors/gps/rmc", qos_profile_sensor_data
            )
            self._gps_gga_pub = self.create_publisher(
                GpsGga, "/ad/sensors/gps/gga", qos_profile_sensor_data
            )
            self._gps_time_reference_pub = self.create_publisher(
                TimeReference,
                "/ad/sensors/gps/time_reference",
                qos_profile_sensor_data,
            )
        elif mode == "imu":
            self._imu_full_pub = self.create_publisher(
                ImuPacket, "/ad/sensors/imu/full", qos_profile_sensor_data
            )
        elif mode == "ego":
            self._status_full_pub = self.create_publisher(
                EgoVehicleStatus,
                "/ad/vehicle/status/full",
                qos_profile_sensor_data,
            )
        if publish_raw:
            self._raw_publishers[name] = self.create_publisher(
                RawPacket,
                f"/ad/udp/raw/{name}",
                qos_profile_sensor_data,
            )

    def _start_receiver(self, name: str, config: dict[str, Any]) -> None:
        receiver = None
        try:
            receiver = UdpReceiver(
                config["endpoint"],
                lambda data, address, arrived, stream=name: self._enqueue(
                    stream, data, arrived
                ),
                receive_buffer_bytes=int(config["receive_buffer_bytes"]),
            )
            receiver.start()
        except OSError as exc:
            if receiver is not None:
                self._close_resource(receiver, "UDP receiver")
            self._counters[name].bind_errors += 1
            self.get_logger().error(
                f"{name}: UDP bind failed on "
                f"{config['bind_ip']}:{config['port']}: {exc}"
            )
            return
        except BaseException:
            if receiver is not None:
                self._close_resource(receiver, "UDP receiver")
            raise
        self._receivers[name] = receiver
        self.get_logger().info(
            f"{name}: listening on {config['bind_ip']}:{config['port']}"
        )

    def _enqueue(self, stream: str, data: bytes, arrived: float) -> None:
        with self._lock:
            counter = self._counters[stream]
            counter.packets += 1
            counter.bytes += len(data)
            counter.first_at = arrived if counter.first_at is None else counter.first_at
            counter.last_at = arrived
        try:
            self._queue.put_nowait((stream, data, arrived))
        except queue.Full:
            with self._lock:
                self._counters[stream].dropped += 1

    def _drain_queue(self) -> None:
        for _ in range(128):
            try:
                stream, packet, arrived = self._queue.get_nowait()
            except queue.Empty:
                return
            try:
                if self._publish_raw_packets:
                    self._publish_raw(stream, packet, arrived)
                self._decode_and_publish(stream, packet, arrived)
            except Exception as exc:
                with self._lock:
                    counter = self._counters[stream]
                    counter.malformed += 1
                    count = counter.malformed
                if count <= 3 or count & (count - 1) == 0:
                    self.get_logger().warning(
                        f"{stream}: dropped malformed packet #{count} "
                        f"({type(exc).__name__}): {exc}"
                    )

    def _arrival_stamp(self, arrived: float) -> tuple[int, int]:
        return self._receipt_clock.stamp(arrived)

    def _publish_raw(self, stream: str, packet: bytes, arrived: float) -> None:
        publisher = self._raw_publishers.get(stream)
        if publisher is None:
            return
        message = RawPacket()
        (
            message.header.stamp.sec,
            message.header.stamp.nanosec,
        ) = self._arrival_stamp(arrived)
        message.header.frame_id = str(self._stream_cfg[stream]["frame_id"])
        message.stream = stream
        message.data = packet
        publisher.publish(message)

    def _decode_and_publish(self, stream: str, packet: bytes, arrived: float) -> None:
        config = self._stream_cfg[stream]
        frame_id = str(config["frame_id"])
        mode = str(config["mode"])
        publisher = self._stream_publishers[stream]
        arrival_stamp = self._arrival_stamp(arrived)

        if mode == "ego":
            record = decode_ego_status(packet)
            decision = self._timestamp_decision(
                stream, record.stamp, arrival_stamp
            )
            self._status_full_pub.publish(
                ego_status_message(record, frame_id, arrival_stamp)
            )
            self._publish_timing(
                stream, frame_id, arrival_stamp, record.stamp, decision
            )
            if decision.publish_normalized:
                publisher.publish(
                    ego_status_message(record, frame_id, decision.selected_stamp)
                )
        elif mode == "collisions":
            record = decode_collisions(packet)
            decision = self._timestamp_decision(
                stream, record.stamp, arrival_stamp
            )
            self._publish_timing(
                stream, frame_id, arrival_stamp, record.stamp, decision
            )
            if decision.publish_normalized:
                publisher.publish(
                    collision_array_message(
                        replace(record, stamp=decision.selected_stamp),
                        frame_id,
                    )
                )
        elif mode == "camera":
            image = self._camera_assemblers[stream].push(packet, arrived)
            if image is not None:
                frame_arrival_stamp = self._arrival_stamp(image.first_arrived)
                decision = self._timestamp_decision(
                    stream, image.stamp, frame_arrival_stamp
                )
                self._publish_timing(
                    stream,
                    frame_id,
                    frame_arrival_stamp,
                    image.stamp,
                    decision,
                )
                if decision.publish_normalized:
                    message = CompressedImage()
                    (
                        message.header.stamp.sec,
                        message.header.stamp.nanosec,
                    ) = decision.selected_stamp
                    message.header.frame_id = frame_id
                    message.format = "jpeg"
                    message.data = image.jpeg
                    publisher.publish(message)
        elif mode == "gps":
            sentence = decode_nmea(packet)
            if isinstance(sentence, GprmcRecord):
                self._gps_rmc_pub.publish(
                    gps_rmc_message(sentence, frame_id, arrival_stamp)
                )
                source_stamp = rmc_epoch_stamp(sentence)
                decision = self._timestamp_decision(
                    stream,
                    source_stamp,
                    arrival_stamp,
                    reject_source=source_stamp is None,
                    publish_requires_valid_source=True,
                )
                publish_time_reference = (
                    decision.source_valid
                    and decision.publish_normalized
                )
                self._publish_timing(
                    "gps/rmc",
                    frame_id,
                    arrival_stamp,
                    source_stamp,
                    replace(
                        decision,
                        publish_normalized=publish_time_reference,
                    ),
                )
                if publish_time_reference:
                    assert source_stamp is not None
                    self._gps_time_reference_pub.publish(
                        gps_time_reference_message(
                            frame_id, arrival_stamp, source_stamp
                        )
                    )
            elif isinstance(sentence, GpggaRecord):
                self._gps_gga_pub.publish(
                    gps_gga_message(sentence, frame_id, arrival_stamp)
                )
            fix = self._gps.update(sentence)
            if fix is not None:
                decision = self._timestamp_decision(
                    stream,
                    fix.source_stamp,
                    arrival_stamp,
                    reject_source=fix.source_rejected,
                )
                self._publish_timing(
                    "gps/fix",
                    frame_id,
                    arrival_stamp,
                    fix.source_stamp,
                    decision,
                )
                if decision.publish_normalized:
                    publisher.publish(
                        gps_fix_message(
                            fix, frame_id, decision.selected_stamp
                        )
                    )
        elif mode == "imu":
            record = decode_imu(packet)
            decision = self._timestamp_decision(
                stream, record.stamp, arrival_stamp
            )
            self._imu_full_pub.publish(
                imu_packet_message(record, frame_id, arrival_stamp)
            )
            self._publish_timing(
                stream, frame_id, arrival_stamp, record.stamp, decision
            )
            if decision.publish_normalized:
                publisher.publish(
                    imu_message(
                        replace(record, stamp=decision.selected_stamp),
                        frame_id,
                    )
                )
        elif mode == "velodyne":
            valid_packet = validate_velodyne_packet(packet)
            message = RawPacket()
            message.header.stamp.sec, message.header.stamp.nanosec = arrival_stamp
            message.header.frame_id = frame_id
            message.stream = "velodyne"
            message.data = valid_packet
            publisher.publish(message)

    def _timestamp_decision(
        self,
        stream: str,
        source_stamp: tuple[int, int] | None,
        arrival_stamp: tuple[int, int],
        *,
        reject_source: bool = False,
        publish_requires_valid_source: bool = False,
    ) -> TimestampDecision:
        decision = self._timestamp_policies[stream].decide(
            source_stamp,
            arrival_stamp,
            reject_source=reject_source,
            publish_requires_valid_source=publish_requires_valid_source,
        )
        with self._lock:
            counter = self._counters[stream]
            counter.source_selected += int(decision.source_selected)
            counter.arrival_fallback += int(decision.arrival_fallback)
            counter.source_rejected += int(decision.source_rejected)
            counter.duplicates += int(decision.duplicate)
            counter.stamp_regressions += int(decision.stamp_regression)
            counter.dropped += int(decision.stamp_regression)
        return decision

    def _publish_timing(
        self,
        stream: str,
        frame_id: str,
        arrival_stamp: tuple[int, int],
        source_stamp: tuple[int, int] | None,
        decision: TimestampDecision,
    ) -> None:
        self._timing_pub.publish(
            sensor_timing_message(
                stream,
                frame_id,
                arrival_stamp,
                source_stamp,
                decision,
            )
        )

    def _setup_control(self) -> None:
        if not self._control_enabled:
            return
        assert self._control_target is not None
        assert self._control_topic is not None
        assert self._control_gate is not None

        self._control_sender = UdpSender(self._control_target)
        self._senders.append(self._control_sender)
        command_qos = QoSProfile(
            depth=10, reliability=ReliabilityPolicy.RELIABLE
        )
        self._control_subscription = self.create_subscription(
            CtrlCmd, self._control_topic, self._on_control, command_qos
        )

    def _on_control(self, message: CtrlCmd) -> None:
        if self._control_gate is None:
            return
        command = CtrlCommandRecord(
            ctrl_mode=message.ctrl_mode,
            gear=message.gear,
            long_cmd_type=message.long_cmd_type,
            velocity=message.velocity,
            acceleration=message.acceleration,
            accel=message.accel,
            brake=message.brake,
            steering=message.steering,
        )
        try:
            self._control_gate.accept(command, time.monotonic())
        except PacketFormatError as exc:
            self.get_logger().error(f"control command rejected: {exc}")

    def _send_control_record(self, command: CtrlCommandRecord) -> None:
        if self._control_sender is None:
            return
        try:
            self._control_sender.send(encode_ctrl_cmd(command))
        except OSError as exc:
            self._control_send_errors += 1
            count = self._control_send_errors
            if count <= 3 or count & (count - 1) == 0:
                self.get_logger().error(
                    f"control UDP send failed #{count}: {exc}"
                )

    def _watchdog_tick(self) -> None:
        if self._control_gate is not None:
            self._control_gate.tick(time.monotonic())

    def _publish_statistics(self) -> None:
        now_steady = time.monotonic()
        stamp = self.get_clock().now().to_msg()
        diagnostic = DiagnosticArray()
        diagnostic.header.stamp = stamp
        with self._lock:
            snapshots = {
                name: replace(counter) for name, counter in self._counters.items()
            }

        for name, counter in snapshots.items():
            elapsed = (
                max(1e-9, now_steady - counter.first_at)
                if counter.first_at is not None
                else 0.0
            )
            age = (
                now_steady - counter.last_at
                if counter.last_at is not None
                else -1.0
            )
            statistics = BridgeStatistics()
            statistics.header.stamp = stamp
            statistics.stream = name
            statistics.packets = counter.packets
            statistics.bytes = counter.bytes
            statistics.malformed = counter.malformed
            statistics.dropped = counter.dropped
            statistics.bind_errors = counter.bind_errors
            statistics.source_selected = counter.source_selected
            statistics.arrival_fallback = counter.arrival_fallback
            statistics.source_rejected = counter.source_rejected
            statistics.duplicates = counter.duplicates
            statistics.stamp_regressions = counter.stamp_regressions
            statistics.packet_rate_hz = (
                counter.packets / elapsed if elapsed else 0.0
            )
            statistics.byte_rate_bps = (
                counter.bytes / elapsed if elapsed else 0.0
            )
            statistics.last_packet_age_sec = age
            self._stats_pub.publish(statistics)

            status = DiagnosticStatus()
            status.name = f"ad_morai_bridge/{name}"
            status.hardware_id = "MORAI-24.R2"
            enabled = bool(self._stream_cfg[name]["enabled"])
            receiver = self._receivers.get(name)
            status.level, status.message = diagnostic_state(
                enabled,
                counter,
                age,
                bool(receiver and receiver.is_alive),
                float(self._stream_cfg[name]["stale_after_sec"]),
            )
            status.values = [
                KeyValue(key="packets", value=str(counter.packets)),
                KeyValue(key="malformed", value=str(counter.malformed)),
                KeyValue(key="dropped", value=str(counter.dropped)),
                KeyValue(key="bind_errors", value=str(counter.bind_errors)),
                KeyValue(
                    key="source_selected", value=str(counter.source_selected)
                ),
                KeyValue(
                    key="arrival_fallback", value=str(counter.arrival_fallback)
                ),
                KeyValue(
                    key="source_rejected", value=str(counter.source_rejected)
                ),
                KeyValue(key="duplicates", value=str(counter.duplicates)),
                KeyValue(
                    key="stamp_regressions",
                    value=str(counter.stamp_regressions),
                ),
                KeyValue(key="last_packet_age_sec", value=f"{age:.3f}"),
            ]
            diagnostic.status.append(status)
        self._diagnostics_pub.publish(diagnostic)

    def destroy_node(self):
        self._cleanup_resources(send_emergency=True)
        return super().destroy_node()

    def _cleanup_resources(self, *, send_emergency: bool) -> None:
        if self._resources_closed:
            return
        self._resources_closed = True
        # Safety order: burst the emergency stop over the still-open control
        # link BEFORE any UDP socket is closed.
        if send_emergency and self._control_gate is not None:
            try:
                self._control_gate.emergency_stop()
            except Exception as exc:
                self.get_logger().error(f"control emergency stop failed: {exc}")
        for receiver in self._receivers.values():
            self._close_resource(receiver, "UDP receiver")
        for sender in self._senders:
            self._close_resource(sender, "UDP sender")
        self._receivers.clear()
        self._senders.clear()

    def _close_resource(self, resource: Any, label: str) -> None:
        try:
            resource.close()
        except Exception as exc:
            self.get_logger().error(f"{label} failed: {exc}")


def main(args=None):
    rclpy.init(args=args)
    bridge = None
    try:
        bridge = AdMoraiBridge()
        rclpy.spin(bridge)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RCLError:
        if rclpy.ok():
            raise
    finally:
        if bridge is not None:
            try:
                bridge.destroy_node()
            except KeyboardInterrupt:
                pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
