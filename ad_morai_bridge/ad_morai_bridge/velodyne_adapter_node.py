import math
from numbers import Real
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from velodyne_msgs.msg import VelodynePacket, VelodyneScan

from ad_morai_interfaces.msg import RawPacket

from .codecs.common import PacketFormatError
from .codecs.lidar import first_block_azimuth_hundredths
from .protocol_records import VelodynePacketRecord


INPUT_TOPIC = "/ad/sensors/lidar/raw"
OUTPUT_TOPIC = "/ad/sensors/lidar/packets"
FRAME_ID = "lidar_link"
STREAM = "velodyne"


class PacketOrderingError(ValueError):
    pass


def _validated_stamp(stamp: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(stamp, tuple) or len(stamp) != 2:
        raise PacketFormatError(STREAM, f"invalid stamp {stamp!r}")
    sec, nanosec = stamp
    if (
        type(sec) is not int
        or type(nanosec) is not int
        or not 0 <= sec <= 2**31 - 1
        or not 0 <= nanosec < 1_000_000_000
    ):
        raise PacketFormatError(STREAM, f"invalid stamp {stamp!r}")
    return sec, nanosec


def _validated_monotonic(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} monotonic time must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} monotonic time must be a finite number")
    return result


def _validated_scan_rate(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("scan_rate_hz must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("scan_rate_hz must be a finite positive number")
    return result


def _validated_point_timing_mode(value: object) -> str:
    if not isinstance(value, str) or value not in {"azimuth", "zero"}:
        raise ValueError("point_timing_mode must be 'azimuth' or 'zero'")
    return value


def _validated_boundary_mode(value: object) -> str:
    if not isinstance(value, str) or value not in {
        "azimuth",
        "footer_timestamp",
    }:
        raise ValueError(
            "boundary_mode must be 'azimuth' or 'footer_timestamp'"
        )
    return value


def _stamp_nanoseconds(stamp: tuple[int, int]) -> int:
    sec, nanosec = _validated_stamp(stamp)
    return sec * 1_000_000_000 + nanosec


def _validated_record(record: VelodynePacketRecord) -> VelodynePacketRecord:
    if not isinstance(record, VelodynePacketRecord):
        raise PacketFormatError(STREAM, "invalid packet record")
    _validated_stamp(record.stamp)
    payload_azimuth = first_block_azimuth_hundredths(record.data)
    if (
        type(record.first_azimuth_hundredths) is not int
        or record.first_azimuth_hundredths != payload_azimuth
    ):
        raise PacketFormatError(
            STREAM,
            "first azimuth metadata does not match packet payload",
        )
    return record


class VelodyneScanAssembler:
    def __init__(
        self,
        *,
        max_packets: int = 512,
        idle_timeout_sec: float = 0.2,
        boundary_mode: str = "azimuth",
    ):
        if type(max_packets) is not int or max_packets <= 0:
            raise ValueError("max_packets must be a positive integer")
        if (
            isinstance(idle_timeout_sec, bool)
            or not isinstance(idle_timeout_sec, Real)
            or not math.isfinite(float(idle_timeout_sec))
            or float(idle_timeout_sec) <= 0.0
        ):
            raise ValueError("idle_timeout_sec must be finite and positive")
        self._max_packets = max_packets
        self._idle_timeout_sec = float(idle_timeout_sec)
        self._boundary_mode = _validated_boundary_mode(boundary_mode)
        self._buffer: list[VelodynePacketRecord] = []
        self._last_azimuth: int | None = None
        self._last_footer_timestamp: int | None = None
        self._last_stamp: tuple[int, int] | None = None
        self._last_received_monotonic: float | None = None

    def push(
        self,
        record: VelodynePacketRecord,
        *,
        received_monotonic: float,
    ) -> tuple[VelodynePacketRecord, ...] | None:
        _validated_record(record)
        received = _validated_monotonic(
            received_monotonic, name="received"
        )

        if self._last_stamp is not None and record.stamp < self._last_stamp:
            raise PacketOrderingError(
                f"stamp decreased from {self._last_stamp!r} to {record.stamp!r}"
            )
        if (
            self._last_received_monotonic is not None
            and received < self._last_received_monotonic
        ):
            raise PacketOrderingError(
                "received monotonic time decreased from "
                f"{self._last_received_monotonic} to {received}"
            )

        boundary = False
        completes_snapshot = False
        footer_timestamp = int.from_bytes(record.data[1_200:1_204], "little")
        if self._boundary_mode == "footer_timestamp":
            boundary = (
                self._last_footer_timestamp is not None
                and footer_timestamp != self._last_footer_timestamp
            )
            if not boundary and self._last_azimuth is not None:
                current = record.first_azimuth_hundredths
                previous = self._last_azimuth
                delta = (current - previous) % 36_000
                if delta != 0:
                    if delta <= 18_000:
                        completes_snapshot = current < previous
                    else:
                        raise PacketOrderingError(
                            f"stale azimuth moved from {previous} to {current}"
                        )
        elif self._last_azimuth is not None:
            current = record.first_azimuth_hundredths
            previous = self._last_azimuth
            delta = (current - previous) % 36_000
            if delta != 0:
                if delta <= 18_000:
                    boundary = current < previous
                else:
                    raise PacketOrderingError(
                        f"stale azimuth moved from {previous} to {current}"
                    )

        emitted = None
        if (
            boundary or len(self._buffer) >= self._max_packets
        ) and self._buffer:
            emitted = tuple(self._buffer)
            self._buffer.clear()

        self._buffer.append(record)
        self._last_azimuth = record.first_azimuth_hundredths
        self._last_footer_timestamp = footer_timestamp
        self._last_stamp = record.stamp
        self._last_received_monotonic = received
        if completes_snapshot:
            emitted = tuple(self._buffer)
            self._buffer.clear()
        return emitted

    def flush_if_idle(
        self, *, now_monotonic: float
    ) -> tuple[VelodynePacketRecord, ...] | None:
        now = _validated_monotonic(now_monotonic, name="current")
        if (
            self._last_received_monotonic is not None
            and now < self._last_received_monotonic
        ):
            raise PacketOrderingError(
                "current monotonic time precedes the last received packet"
            )
        if not self._buffer or self._last_received_monotonic is None:
            return None
        if now - self._last_received_monotonic < self._idle_timeout_sec:
            return None
        return self.flush()

    def flush(self) -> tuple[VelodynePacketRecord, ...] | None:
        if not self._buffer:
            return None
        emitted = tuple(self._buffer)
        self._buffer.clear()
        self._last_azimuth = None
        self._last_footer_timestamp = None
        return emitted


def raw_packet_record(message: RawPacket) -> VelodynePacketRecord:
    if message.stream != STREAM:
        raise PacketFormatError(
            STREAM, f"invalid stream {message.stream!r}"
        )
    if message.header.frame_id != FRAME_ID:
        raise PacketFormatError(
            STREAM, f"invalid frame {message.header.frame_id!r}"
        )
    stamp = _validated_stamp(
        (message.header.stamp.sec, message.header.stamp.nanosec)
    )
    data = bytes(message.data)
    return VelodynePacketRecord(
        stamp=stamp,
        data=data,
        first_azimuth_hundredths=first_block_azimuth_hundredths(data),
    )


def velodyne_scan_message(
    records: tuple[VelodynePacketRecord, ...],
    *,
    scan_rate_hz: float = 10.0,
    point_timing_mode: str = "azimuth",
) -> VelodyneScan:
    items = tuple(records)
    if not items:
        raise ValueError("cannot convert an empty Velodyne scan")
    rate = _validated_scan_rate(scan_rate_hz)
    timing_mode = _validated_point_timing_mode(point_timing_mode)
    for record in items:
        _validated_record(record)

    message = VelodyneScan()
    message.header.frame_id = FRAME_ID
    scan_start_ns = _stamp_nanoseconds(items[0].stamp)
    scan_reference_stamp = (
        items[-1].stamp if timing_mode == "zero" else items[0].stamp
    )
    (
        message.header.stamp.sec,
        message.header.stamp.nanosec,
    ) = scan_reference_stamp
    scan_reference_ns = _stamp_nanoseconds(scan_reference_stamp)
    scan_period_ns = round(1_000_000_000 / rate)
    first_azimuth = items[0].first_azimuth_hundredths
    previous_phase = -1
    for record in items:
        phase = (
            record.first_azimuth_hundredths - first_azimuth
        ) % 36_000
        if phase < previous_phase:
            raise PacketOrderingError(
                "packet azimuth phase decreased within an assembled scan"
            )
        previous_phase = phase
        packet_stamp_ns = scan_reference_ns
        if timing_mode == "azimuth":
            packet_stamp_ns = scan_start_ns + round(
                phase * scan_period_ns / 36_000
            )
        packet_message = VelodynePacket()
        packet_message.stamp.sec = packet_stamp_ns // 1_000_000_000
        packet_message.stamp.nanosec = packet_stamp_ns % 1_000_000_000
        packet_message.data = bytearray(record.data)
        message.packets.append(packet_message)
    return message


class AdVelodyneAdapter(Node):
    def __init__(self):
        super().__init__("ad_velodyne_adapter")
        self._scan_rate_hz = _validated_scan_rate(
            self.declare_parameter("scan_rate_hz", 10.0).value
        )
        self._point_timing_mode = _validated_point_timing_mode(
            self.declare_parameter("point_timing_mode", "azimuth").value
        )
        self._lock = threading.Lock()
        boundary_mode = (
            "footer_timestamp"
            if self._point_timing_mode == "zero"
            else "azimuth"
        )
        self._assembler = VelodyneScanAssembler(boundary_mode=boundary_mode)
        self._closed = False
        self._error_count = 0
        self._publisher = None
        self._subscription = None
        self._idle_timer = None
        try:
            output_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            input_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=100,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            )
            self._publisher = self.create_publisher(
                VelodyneScan, OUTPUT_TOPIC, output_qos
            )
            self._subscription = self.create_subscription(
                RawPacket, INPUT_TOPIC, self._on_packet, input_qos
            )
            self._idle_timer = self.create_timer(0.05, self._on_idle_timer)
        except BaseException:
            self._closed = True
            try:
                super().destroy_node()
            except BaseException:
                pass
            raise

    def _on_packet(self, message: RawPacket) -> None:
        try:
            record = raw_packet_record(message)
            with self._lock:
                if self._closed:
                    return
                emitted = self._assembler.push(
                    record, received_monotonic=time.monotonic()
                )
                self._publish_records(emitted)
        except Exception as exc:
            self._warn_dropped(exc)

    def _on_idle_timer(self) -> None:
        try:
            with self._lock:
                if self._closed:
                    return
                emitted = self._assembler.flush_if_idle(
                    now_monotonic=time.monotonic()
                )
                self._publish_records(emitted)
        except Exception as exc:
            self._warn_dropped(exc)

    def _publish_records(
        self, records: tuple[VelodynePacketRecord, ...] | None
    ) -> None:
        if records is not None:
            self._publisher.publish(
                velodyne_scan_message(
                    records,
                    scan_rate_hz=self._scan_rate_hz,
                    point_timing_mode=self._point_timing_mode,
                )
            )

    def _warn_dropped(self, exc: Exception) -> None:
        with self._lock:
            self._error_count += 1
            count = self._error_count
        if count <= 3 or count & (count - 1) == 0:
            self.get_logger().warning(
                f"dropped invalid Velodyne packet #{count} "
                f"({type(exc).__name__}): {exc}"
            )

    def destroy_node(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                if self._idle_timer is not None:
                    self._idle_timer.cancel()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    adapter = None
    try:
        adapter = AdVelodyneAdapter()
        rclpy.spin(adapter)
    except KeyboardInterrupt:
        pass
    finally:
        if adapter is not None:
            try:
                adapter.destroy_node()
            except KeyboardInterrupt:
                pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
