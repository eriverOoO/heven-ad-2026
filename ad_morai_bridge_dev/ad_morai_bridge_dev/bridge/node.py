from dataclasses import dataclass, replace
import json
import queue
import threading
import time
from typing import Any

import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from ad_morai_interfaces_dev.msg import (
    CameraBoundingBoxArray,
    EgoGhostCmd,
    EgoVehicleStatus as DevEgoVehicleStatus,
    IntersectionControl,
    IntersectionStatus,
    Lamps,
    Lidar2D,
    MultiEgoSetting,
    NpcGhostCmd,
    ObjectStatusArray,
    SaveSensorData,
    ScenarioLoad,
    SensorPosControl,
    SimulatorCommand,
    TrafficLightControl,
    TrafficLightStatus,
    VehicleCollisionArray,
)
from ad_morai_interfaces_dev.srv import MoraiGrpcCall, MoraiStep, MoraiSyncMode
from ad_morai_interfaces.msg import (
    BridgeStatistics,
    EgoVehicleStatus,
    RawPacket,
    SensorTiming,
)
from ad_morai_bridge.arrival_time import ReceiptClockMapper
from ad_morai_bridge.codecs.common import PacketFormatError
from ad_morai_bridge.codecs.ego import decode_ego_status
from ad_morai_bridge.message_conversion import (
    ego_status_message,
    sensor_timing_message,
)
from ad_morai_bridge.timestamp_policy import TimestampDecision, TimestampPolicy
from ad_morai_bridge.udp_transport import UdpEndpoint, UdpReceiver, UdpSender

from ad_morai_bridge_dev.codecs import decode_objects
from ad_morai_bridge_dev.codecs.camera import decode_camera_bboxes
from ad_morai_bridge_dev.codecs.collision import decode_npc_collisions
from ad_morai_bridge_dev.codecs.lidar import decode_lidar2d
from ad_morai_bridge_dev.codecs.simulator import (
    decode_known_simulator_packet,
    encode_simulator_command,
)
from ad_morai_bridge_dev.bridge.message_conversion import (
    camera_bounding_boxes_message,
    dev_ego_status_message,
    laser_scan_message,
    lidar2d_message,
    object_array_message,
    vehicle_collision_array_message,
)
from ad_morai_bridge_dev.simulator_grpc.client import GrpcCallResult, MoraiGrpcClient
from ad_morai_bridge_dev.simulator_grpc.descriptors import MoraiApi
from ad_morai_bridge_dev.bridge.endpoints import OUTPUTS, STREAMS
from ad_morai_bridge_dev.bridge.raw_streams import read_raw_stream_specs


FULL_TOPICS = {
    "ego_status": "/ad/dev/vehicle/ego_status/full",
    "lidar2d": "/ad/dev/lidar2d/full",
    "npc_collisions": "/ad/dev/npc/collisions",
}

CAMERA_RAW_INPUTS = {
    "front": "/ad/udp/raw/camera_front",
    "left": "/ad/udp/raw/camera_left",
    "right": "/ad/udp/raw/camera_right",
    "traffic_light": "/ad/udp/raw/camera_traffic_light",
}

CAMERA_FRAMES = {
    "front": "camera_front_optical_frame",
    "left": "camera_left_optical_frame",
    "right": "camera_right_optical_frame",
    "traffic_light": "camera_traffic_light_optical_frame",
}

CAMERA_COUNTER_STREAMS = {
    role: f"camera_bbox_{role}" for role in CAMERA_RAW_INPUTS
}

SYNC_MODES = {
    "SYNC_MODE_TYPE_UNSPECIFIED": 0,
    "SYNC_MODE_TYPE_TIME_MANAGEMENT": 1,
    "SYNC_MODE_TYPE_SYNCHRONOUS": 2,
}


@dataclass
class StreamCounters:
    packets: int = 0
    bytes: int = 0
    malformed: int = 0
    dropped: int = 0
    source_selected: int = 0
    arrival_fallback: int = 0
    source_rejected: int = 0
    duplicates: int = 0
    stamp_regressions: int = 0
    first_at: float | None = None
    last_at: float | None = None


class AdMoraiDevBridge(Node):
    def __init__(self, *, parameter_overrides=None):
        super().__init__("ad_morai_bridge_dev", parameter_overrides=parameter_overrides)
        self._receivers: dict[str, UdpReceiver] = {}
        self._senders: dict[str, UdpSender] = {}
        self._grpc_client = None
        self._closed = False
        try:
            self._initialize()
        except Exception:
            self._close_transports()
            super().destroy_node()
            raise

    def _initialize(self) -> None:
        if self.has_parameter("use_sim_time") and bool(
            self.get_parameter("use_sim_time").value
        ):
            raise ValueError(
                "use_sim_time is unsupported: UDP receipt timestamps require "
                "one stable system ROS clock"
            )
        capacity = int(self.declare_parameter("queue_capacity", 4096).value)
        if capacity <= 0:
            raise ValueError("queue_capacity must be positive")
        self._queue: queue.Queue[tuple[str, bytes, float]] = queue.Queue(
            maxsize=capacity
        )
        self._publish_raw_packets = bool(
            self.declare_parameter("publish_raw_packets", False).value
        )
        timestamp_mode = str(
            self.declare_parameter(
                "timestamp_mode", "arrival"
            ).value
        )
        source_stamp_tolerance_sec = float(
            self.declare_parameter(
                "source_stamp_tolerance_sec", 1.0
            ).value
        )
        extra_raw_specs = (
            read_raw_stream_specs(self) if self._publish_raw_packets else ()
        )
        self._stream_configs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._counters = {
            name: StreamCounters()
            for name in (
                *STREAMS,
                *CAMERA_COUNTER_STREAMS.values(),
                *(spec.name for spec in extra_raw_specs),
            )
        }
        self._stream_publishers: dict[str, Any] = {}
        self._full_publishers: dict[str, Any] = {}
        self._raw_publishers: dict[str, Any] = {}
        self._started_at = time.monotonic()
        self._receipt_clock = ReceiptClockMapper(
            sampled_monotonic=self._started_at,
            sampled_ros_ns=self.get_clock().now().nanoseconds,
        )
        self._timestamp_policies = {
            name: TimestampPolicy(
                mode=timestamp_mode,
                tolerance_sec=source_stamp_tolerance_sec,
                suppress_source_duplicates=False,
            )
            for name in STREAMS
        }
        self._camera_timestamp_policies = {
            role: TimestampPolicy(
                mode=timestamp_mode,
                tolerance_sec=source_stamp_tolerance_sec,
                suppress_source_duplicates=False,
            )
            for role in CAMERA_RAW_INPUTS
        }

        for name, (port, topic, frame_id, mode) in STREAMS.items():
            config = self._stream_config(name, port, topic, frame_id, mode)
            self._stream_configs[name] = config
            self._stream_publishers[name] = self.create_publisher(
                self._message_type(mode), config["topic"], qos_profile_sensor_data
            )
            if name in ("ego_status", "lidar2d"):
                full_type = DevEgoVehicleStatus if name == "ego_status" else Lidar2D
                self._full_publishers[name] = self.create_publisher(
                    full_type,
                    config["full_topic"],
                    qos_profile_sensor_data,
                )
            if self._publish_raw_packets:
                self._raw_publishers[name] = self.create_publisher(
                    RawPacket,
                    f"/ad/dev/udp/raw/{name}",
                    qos_profile_sensor_data,
                )
            if config["enabled"]:
                self._start_receiver(name, config)

        for spec in extra_raw_specs:
            config = {
                "enabled": True,
                "bind_ip": spec.bind_ip,
                "port": spec.port,
                "topic": f"/ad/dev/udp/raw/{spec.name}",
                "frame_id": spec.frame_id,
                "receive_buffer_bytes": 4 * 1024 * 1024,
                "mode": "raw",
            }
            self._stream_configs[spec.name] = config
            self._raw_publishers[spec.name] = self.create_publisher(
                RawPacket, config["topic"], qos_profile_sensor_data
            )
            self._start_receiver(spec.name, config)

        self._timing_pub = self.create_publisher(
            SensorTiming,
            "/ad/dev/sensors/timing",
            qos_profile_sensor_data,
        )
        self._setup_camera_bbox_topics()

        for name, port in OUTPUTS.items():
            enabled = bool(
                self.declare_parameter(f"{name}.enabled", True).value
            )
            target_ip = str(
                self.declare_parameter(f"{name}.target_ip", "127.0.0.1").value
            )
            target_port = int(
                self.declare_parameter(f"{name}.target_port", port).value
            )
            if enabled:
                self._senders[name] = UdpSender(
                    UdpEndpoint(target_ip, target_port)
                )

        reliable_stats = QoSProfile(
            depth=20, reliability=ReliabilityPolicy.RELIABLE
        )
        reliable_commands = QoSProfile(
            depth=10, reliability=ReliabilityPolicy.RELIABLE
        )
        self._stats_pub = self.create_publisher(
            BridgeStatistics, "/ad/dev/udp/statistics", reliable_stats
        )
        self._command_subscription = self.create_subscription(
            SimulatorCommand,
            "/ad/dev/simulator/command",
            self._on_simulator_command,
            reliable_commands,
        )
        self._command_subscriptions = [
            self._command_subscription,
            self.create_subscription(
                EgoGhostCmd,
                "/ad/dev/command/ego_ghost",
                self._on_ego_ghost,
                reliable_commands,
            ),
            self.create_subscription(
                TrafficLightControl,
                "/ad/dev/command/traffic_light",
                self._on_traffic_light_control,
                reliable_commands,
            ),
            self.create_subscription(
                IntersectionControl,
                "/ad/dev/command/intersection",
                self._on_intersection_control,
                reliable_commands,
            ),
            self.create_subscription(
                SensorPosControl,
                "/ad/dev/command/sensor_control",
                self._on_sensor_pos_control,
                reliable_commands,
            ),
            self.create_subscription(
                Lamps,
                "/ad/dev/command/lamp_control",
                self._on_lamps,
                reliable_commands,
            ),
            self.create_subscription(
                ScenarioLoad,
                "/ad/dev/command/scenario_load",
                self._on_scenario_load,
                reliable_commands,
            ),
            self.create_subscription(
                SaveSensorData,
                "/ad/dev/command/save_sensor_data",
                self._on_save_sensor_data,
                reliable_commands,
            ),
            self.create_subscription(
                MultiEgoSetting,
                "/ad/dev/command/multi_ego",
                self._on_multi_ego,
                reliable_commands,
            ),
            self.create_subscription(
                NpcGhostCmd,
                "/ad/dev/command/npc_ghost",
                self._on_npc_ghost,
                reliable_commands,
            ),
        ]
        self._timers = [
            self.create_timer(0.002, self._drain_queue),
            self.create_timer(1.0, self._publish_statistics),
        ]
        self._initialize_grpc()

    def _initialize_grpc(self) -> None:
        enabled = bool(self.declare_parameter("grpc.enabled", True).value)
        target = str(
            self.declare_parameter("grpc.target", "127.0.0.1:7789").value
        )
        self._grpc_timeout = float(
            self.declare_parameter("grpc.timeout_sec", 5.0).value
        )
        self._grpc_services = []
        if not enabled:
            return

        self._grpc_client = MoraiGrpcClient.connect(
            MoraiApi.load(),
            target,
            default_timeout=self._grpc_timeout,
        )
        self._grpc_services = [
            self.create_service(
                MoraiGrpcCall,
                "/ad/dev/morai/grpc_call",
                self._on_grpc_call,
            ),
            self.create_service(
                MoraiSyncMode,
                "/ad/dev/morai/sync_mode",
                self._on_sync_mode,
            ),
            self.create_service(
                MoraiStep,
                "/ad/dev/morai/step",
                self._on_step,
            ),
        ]

    def _stream_config(
        self, name: str, port: int, topic: str, frame_id: str, mode: str
    ) -> dict[str, Any]:
        config = {
            "enabled": bool(
                self.declare_parameter(f"{name}.enabled", True).value
            ),
            "bind_ip": str(
                self.declare_parameter(f"{name}.bind_ip", "127.0.0.1").value
            ),
            "port": int(self.declare_parameter(f"{name}.port", port).value),
            "topic": str(
                self.declare_parameter(f"{name}.topic", topic).value
            ),
            "frame_id": str(
                self.declare_parameter(f"{name}.frame_id", frame_id).value
            ),
            "receive_buffer_bytes": int(
                self.declare_parameter(
                    f"{name}.receive_buffer_bytes", 4 * 1024 * 1024
                ).value
            ),
            "mode": mode,
        }
        # Convention: development-only topics live under /ad/dev/.
        if name in ("ego_status", "lidar2d"):
            config["full_topic"] = str(
                self.declare_parameter(
                    f"{name}.full_topic", FULL_TOPICS[name]
                ).value
            )
        return config

    @staticmethod
    def _message_type(mode: str):
        return {
            "ego": EgoVehicleStatus,
            "objects": ObjectStatusArray,
            "lidar2d": LaserScan,
            "traffic_light": TrafficLightStatus,
            "intersection": IntersectionStatus,
            "npc_collisions": VehicleCollisionArray,
        }[mode]

    def _setup_camera_bbox_topics(self) -> None:
        self._camera_bbox_publishers = {}
        self._camera_raw_subscriptions = []
        enabled = bool(
            self.declare_parameter("camera_bboxes.enabled", False).value
        )
        if not enabled:
            return
        for role, raw_topic in CAMERA_RAW_INPUTS.items():
            self._camera_bbox_publishers[role] = self.create_publisher(
                CameraBoundingBoxArray,
                f"/ad/dev/camera/{role}/bboxes",
                qos_profile_sensor_data,
            )
            expected_stream = raw_topic.rsplit("/", 1)[-1]
            self._camera_raw_subscriptions.append(
                self.create_subscription(
                    RawPacket,
                    raw_topic,
                    lambda message, camera_role=role, stream=expected_stream: (
                        self._on_camera_raw(camera_role, stream, message)
                    ),
                    qos_profile_sensor_data,
                )
            )

    def _on_camera_raw(
        self, role: str, expected_stream: str, message: RawPacket
    ) -> None:
        if message.stream != expected_stream:
            return
        packet = bytes(message.data)
        if not packet.startswith(b"BOX"):
            return
        counter_stream = CAMERA_COUNTER_STREAMS[role]
        self._record_received(counter_stream, packet, time.monotonic())
        try:
            decoded = decode_camera_bboxes(packet)
            arrival_stamp = (
                int(message.header.stamp.sec),
                int(message.header.stamp.nanosec),
            )
            decision = self._camera_timestamp_policies[role].decide(
                decoded.stamp,
                arrival_stamp,
            )
            self._record_timestamp_decision(counter_stream, decision)
            self._publish_timing(
                f"dev/camera/{role}",
                CAMERA_FRAMES[role],
                arrival_stamp,
                decoded.stamp,
                decision,
            )
            if decision.publish_normalized:
                self._camera_bbox_publishers[role].publish(
                    camera_bounding_boxes_message(
                        decoded,
                        CAMERA_FRAMES[role],
                        decision.selected_stamp,
                    )
                )
        except (PacketFormatError, ValueError, TypeError) as exc:
            self._record_malformed(counter_stream)
            self.get_logger().warning(
                f"camera {role}: malformed bounding-box packet: {exc}"
            )

    def _start_receiver(self, name: str, config: dict[str, Any]) -> None:
        receiver = UdpReceiver(
            UdpEndpoint(config["bind_ip"], config["port"]),
            lambda data, _address, arrived, stream=name: self._enqueue(
                stream, data, arrived
            ),
            receive_buffer_bytes=config["receive_buffer_bytes"],
        )
        receiver.start()
        self._receivers[name] = receiver

    def _enqueue(self, stream: str, packet: bytes, arrived: float) -> None:
        self._record_received(stream, packet, arrived)
        try:
            self._queue.put_nowait((stream, packet, arrived))
        except queue.Full:
            with self._lock:
                self._counters[stream].dropped += 1

    def _record_received(
        self, stream: str, packet: bytes, arrived: float
    ) -> None:
        with self._lock:
            counter = self._counters[stream]
            counter.packets += 1
            counter.bytes += len(packet)
            if counter.first_at is None:
                counter.first_at = arrived
            counter.last_at = arrived

    def _record_malformed(self, stream: str) -> None:
        with self._lock:
            self._counters[stream].malformed += 1

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
            except (PacketFormatError, ValueError, TypeError) as exc:
                self._record_malformed(stream)
                self.get_logger().warning(f"{stream}: malformed UDP packet: {exc}")

    def _arrival_stamp(self, arrived: float | None) -> tuple[int, int]:
        if arrived is None:
            now = self.get_clock().now()
            message = now.to_msg()
            return message.sec, message.nanosec
        return self._receipt_clock.stamp(arrived)

    def _publish_raw(
        self, stream: str, packet: bytes, arrived: float
    ) -> None:
        publisher = self._raw_publishers.get(stream)
        if publisher is None:
            return
        message = self._raw_message(stream, packet, arrived)
        publisher.publish(message)

    def _raw_message(
        self, stream: str, packet: bytes, arrived: float | None = None
    ) -> RawPacket:
        message = RawPacket()
        (
            message.header.stamp.sec,
            message.header.stamp.nanosec,
        ) = self._arrival_stamp(arrived)
        message.header.frame_id = self._stream_configs[stream]["frame_id"]
        message.stream = stream
        message.data = packet
        return message

    def _decode_and_publish(
        self, stream: str, packet: bytes, arrived: float | None = None
    ) -> None:
        config = self._stream_configs[stream]
        frame_id = config["frame_id"]
        mode = config["mode"]
        if mode == "raw":
            return
        publisher = self._stream_publishers[stream]
        arrival_stamp = self._arrival_stamp(arrived)

        if mode == "ego":
            record = decode_ego_status(packet)
            decision = self._timestamp_decision(
                stream, record.stamp, arrival_stamp
            )
            self._full_publishers[stream].publish(
                dev_ego_status_message(record, frame_id, arrival_stamp)
            )
            self._publish_timing(
                f"dev/{stream}",
                frame_id,
                arrival_stamp,
                record.stamp,
                decision,
            )
            if decision.publish_normalized:
                publisher.publish(
                    ego_status_message(
                        record, frame_id, decision.selected_stamp
                    )
                )
        elif mode == "objects":
            record = decode_objects(packet)
            decision = self._timestamp_decision(
                stream, record.stamp, arrival_stamp
            )
            self._publish_timing(
                f"dev/{stream}",
                frame_id,
                arrival_stamp,
                record.stamp,
                decision,
            )
            if decision.publish_normalized:
                publisher.publish(
                    object_array_message(
                        record, frame_id, decision.selected_stamp
                    )
                )
        elif mode == "lidar2d":
            record = decode_lidar2d(packet)
            decision = self._timestamp_decision(
                stream, None, arrival_stamp
            )
            self._full_publishers[stream].publish(
                lidar2d_message(record, frame_id, decision.selected_stamp)
            )
            self._publish_timing(
                f"dev/{stream}",
                frame_id,
                arrival_stamp,
                None,
                decision,
            )
            if decision.publish_normalized:
                publisher.publish(
                    laser_scan_message(
                        record, frame_id, decision.selected_stamp
                    )
                )
        elif mode == "traffic_light":
            values = decode_known_simulator_packet(stream, packet)
            decision = self._timestamp_decision(
                stream, None, arrival_stamp
            )
            self._publish_timing(
                f"dev/{stream}",
                frame_id,
                arrival_stamp,
                None,
                decision,
            )
            if not decision.publish_normalized:
                return
            message = TrafficLightStatus()
            (
                message.header.stamp.sec,
                message.header.stamp.nanosec,
            ) = decision.selected_stamp
            message.header.frame_id = frame_id
            message.traffic_light_index = values["traffic_light_index"]
            message.traffic_light_type = values["traffic_light_type"]
            message.traffic_light_status = values["traffic_light_status"]
            publisher.publish(message)
        elif mode == "intersection":
            values = decode_known_simulator_packet(stream, packet)
            decision = self._timestamp_decision(
                stream, None, arrival_stamp
            )
            self._publish_timing(
                f"dev/{stream}",
                frame_id,
                arrival_stamp,
                None,
                decision,
            )
            if not decision.publish_normalized:
                return
            index = str(values["intersection_index"])
            message = IntersectionStatus()
            (
                message.header.stamp.sec,
                message.header.stamp.nanosec,
            ) = decision.selected_stamp
            message.header.frame_id = frame_id
            message.intersection_index = index
            message.traffic_light_statuses = [values["intersection_status"]]
            message.remaining_times = [values["remaining_time"]]
            publisher.publish(message)
        elif mode == "npc_collisions":
            record = decode_npc_collisions(packet)
            decision = self._timestamp_decision(
                stream, None, arrival_stamp
            )
            self._publish_timing(
                f"dev/{stream}",
                frame_id,
                arrival_stamp,
                None,
                decision,
            )
            if decision.publish_normalized:
                publisher.publish(
                    vehicle_collision_array_message(
                        record,
                        frame_id,
                        decision.selected_stamp,
                    )
                )

    def _timestamp_decision(
        self,
        stream: str,
        source_stamp: tuple[int, int] | None,
        arrival_stamp: tuple[int, int],
    ) -> TimestampDecision:
        decision = self._timestamp_policies[stream].decide(
            source_stamp,
            arrival_stamp,
        )
        self._record_timestamp_decision(stream, decision)
        return decision

    def _record_timestamp_decision(
        self, stream: str, decision: TimestampDecision
    ) -> None:
        with self._lock:
            counter = self._counters[stream]
            counter.source_selected += int(decision.source_selected)
            counter.arrival_fallback += int(decision.arrival_fallback)
            counter.source_rejected += int(decision.source_rejected)
            counter.duplicates += int(decision.duplicate)
            counter.stamp_regressions += int(decision.stamp_regression)
            counter.dropped += int(decision.stamp_regression)

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

    def _on_simulator_command(self, message: SimulatorCommand) -> None:
        try:
            values = json.loads(message.json_payload)
            if not isinstance(values, dict):
                raise ValueError("json_payload must decode to an object")
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            self.get_logger().error(
                f"simulator command rejected ({message.command_type}): {exc}"
            )
            return
        self._send_simulator_command(message.command_type, values)

    def _send_simulator_command(
        self, kind: str, values: dict[str, object]
    ) -> None:
        sender = self._senders.get(kind)
        if sender is None:
            self.get_logger().error(
                f"simulator command has no enabled output: {kind}"
            )
            return
        try:
            sender.send(encode_simulator_command(kind, values))
        except (PacketFormatError, OSError, TypeError, ValueError) as exc:
            self.get_logger().error(
                f"simulator command rejected ({kind}): {exc}"
            )

    @staticmethod
    def _pose_values(message) -> dict[str, object]:
        return {
            "pose_x": message.position.x,
            "pose_y": message.position.y,
            "pose_z": message.position.z,
            "roll": message.rpy.x,
            "pitch": message.rpy.y,
            "yaw": message.rpy.z,
        }

    def _on_ego_ghost(self, message: EgoGhostCmd) -> None:
        values = self._pose_values(message)
        values.update(
            {"velocity": message.velocity, "steering": message.steering}
        )
        self._send_simulator_command("ego_ghost", values)

    def _on_traffic_light_control(
        self, message: TrafficLightControl
    ) -> None:
        self._send_simulator_command(
            "traffic_light",
            {
                "traffic_light_index": message.traffic_light_index,
                "traffic_light_status": message.traffic_light_status,
            },
        )

    def _on_intersection_control(
        self, message: IntersectionControl
    ) -> None:
        self._send_simulator_command(
            "intersection",
            {
                "intersection_index": message.intersection_index,
                "intersection_status": message.intersection_status,
                "intersection_time": message.intersection_time,
            },
        )

    def _on_sensor_pos_control(self, message: SensorPosControl) -> None:
        values = self._pose_values(message)
        values["sensor_index"] = message.sensor_index
        self._send_simulator_command("sensor_control", values)

    def _on_lamps(self, message: Lamps) -> None:
        self._send_simulator_command(
            "lamp_control",
            {
                "turn_signal": message.turn_signal,
                "emergency_signal": message.emergency_signal,
            },
        )

    def _on_scenario_load(self, message: ScenarioLoad) -> None:
        self._send_simulator_command(
            "scenario_load",
            {
                "filename": message.filename,
                "delete_all": message.delete_all,
                "network": message.network,
                "ego": message.ego,
                "npc": message.npc,
                "pedestrian": message.pedestrian,
                "object": message.object,
                "pause": message.pause,
            },
        )

    def _on_save_sensor_data(self, message: SaveSensorData) -> None:
        self._send_simulator_command(
            "save_sensor_data",
            {
                "custom": message.custom,
                "file_name": message.file_name,
                "file_dir": message.file_dir,
            },
        )

    def _on_multi_ego(self, message: MultiEgoSetting) -> None:
        vehicles = []
        for vehicle in message.vehicles:
            vehicles.append(
                {
                    "ego_index": vehicle.ego_index,
                    "position_x": vehicle.position.x,
                    "position_y": vehicle.position.y,
                    "position_z": vehicle.position.z,
                    "roll": vehicle.rpy.x,
                    "pitch": vehicle.rpy.y,
                    "yaw": vehicle.rpy.z,
                    "velocity": vehicle.velocity,
                    "gear": vehicle.gear,
                    "ctrl_mode": vehicle.ctrl_mode,
                }
            )
        self._send_simulator_command(
            "multi_ego",
            {
                "number_of_ego": len(vehicles),
                "camera_index": message.camera_index,
                "vehicles": vehicles,
            },
        )

    def _on_npc_ghost(self, message: NpcGhostCmd) -> None:
        vehicles = []
        for vehicle in message.vehicles:
            values = self._pose_values(vehicle)
            values.update(
                {"unique_id": vehicle.unique_id, "car_name": vehicle.car_name}
            )
            vehicles.append(values)
        self._send_simulator_command("npc_ghost", {"vehicles": vehicles})

    def _on_grpc_call(self, request, response):
        result = self._grpc_client.call_json(
            request.method,
            request.request_json,
            request.timeout_sec,
        )
        self._set_grpc_status(response, result)
        response.response_json = result.response_json
        return response

    def _on_sync_mode(self, request, response):
        if request.set_mode:
            result = self._grpc_client.call_json(
                "morai_sim_api.simulation.Simulation/SetSynchronousMode",
                json.dumps(
                    {
                        "type": request.mode,
                        "tick_period": request.tick_period_ms,
                    }
                ),
                self._grpc_timeout,
            )
            self._set_grpc_status(response, result)
            if not result.success:
                return response

            status = json.loads(result.response_json or "{}")
            if status.get("status") not in (None, "STATUS_CODE_SUCCESS", 1):
                response.success = False
                response.error_message = status.get(
                    "description", "MORAI rejected synchronous mode"
                )
                return response

        result = self._grpc_client.call_json(
            "morai_sim_api.simulation.Simulation/GetSynchronousMode",
            "{}",
            self._grpc_timeout,
        )
        self._set_grpc_status(response, result)
        if result.success:
            active = json.loads(result.response_json or "{}")
            response.active_mode = SYNC_MODES.get(active.get("type"), 0)
            response.active_tick_period_ms = active.get("tick_period", 0)
        return response

    def _on_step(self, request, response):
        if request.frame_count <= 0:
            self._set_grpc_status(
                response,
                GrpcCallResult(
                    False,
                    3,
                    "INVALID_ARGUMENT",
                    error="frame_count must be positive",
                ),
            )
            return response

        result = self._grpc_client.call_json(
            "morai_sim_api.simulation.Simulation/Tick",
            json.dumps({"value": request.frame_count}),
            self._grpc_timeout,
        )
        self._set_grpc_status(response, result)
        if result.success:
            timestamp = json.loads(result.response_json or "{}")
            response.current_frame = int(timestamp.get("frame_count", 0))
            response.elapsed_time = int(timestamp.get("elapsed_time", 0))
        return response

    @staticmethod
    def _set_grpc_status(response, result: GrpcCallResult) -> None:
        response.success = result.success
        response.grpc_code = result.code
        response.grpc_status = result.status
        response.error_message = result.error

    def _publish_statistics(self) -> None:
        now = time.monotonic()
        stamp = self.get_clock().now().to_msg()
        with self._lock:
            snapshots = {
                name: replace(counter)
                for name, counter in self._counters.items()
            }
        for name, counter in snapshots.items():
            elapsed = (
                max(1e-9, now - counter.first_at)
                if counter.first_at is not None
                else 0.0
            )
            message = BridgeStatistics()
            message.header.stamp = stamp
            message.stream = name
            message.packets = counter.packets
            message.bytes = counter.bytes
            message.malformed = counter.malformed
            message.dropped = counter.dropped
            message.bind_errors = 0
            message.source_selected = counter.source_selected
            message.arrival_fallback = counter.arrival_fallback
            message.source_rejected = counter.source_rejected
            message.duplicates = counter.duplicates
            message.stamp_regressions = counter.stamp_regressions
            message.packet_rate_hz = (
                counter.packets / elapsed if elapsed else 0.0
            )
            message.byte_rate_bps = counter.bytes / elapsed if elapsed else 0.0
            message.last_packet_age_sec = (
                now - counter.last_at if counter.last_at is not None else -1.0
            )
            self._stats_pub.publish(message)

    def destroy_node(self):
        self._close_transports()
        return super().destroy_node()

    def _close_transports(self) -> None:
        if self._closed:
            return
        self._closed = True
        for receiver in self._receivers.values():
            try:
                receiver.close()
            except OSError as exc:
                self.get_logger().error(f"UDP receiver close failed: {exc}")
        for sender in self._senders.values():
            try:
                sender.close()
            except OSError as exc:
                self.get_logger().error(f"UDP sender close failed: {exc}")
        grpc_client = getattr(self, "_grpc_client", None)
        if grpc_client is not None:
            grpc_client.close()
            self._grpc_client = None
        self._receivers.clear()
        self._senders.clear()


def main(args=None):
    rclpy.init(args=args)
    bridge = None
    try:
        bridge = AdMoraiDevBridge()
        rclpy.spin(bridge)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RCLError:
        if rclpy.ok():
            raise
    finally:
        if bridge is not None:
            bridge.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
