from contextlib import contextmanager
import math
from pathlib import Path

import pytest
import rclpy
from ad_morai_interfaces.msg import RawPacket
from rclpy.qos import DurabilityPolicy, HistoryPolicy, ReliabilityPolicy

from ad_morai_bridge.velodyne_adapter_node import (
    INPUT_TOPIC,
    OUTPUT_TOPIC,
    AdVelodyneAdapter,
    PacketOrderingError,
    VelodyneScanAssembler,
    raw_packet_record,
    velodyne_scan_message,
)
from ad_morai_bridge.codecs.common import PacketFormatError
from ad_morai_bridge.codecs.lidar import (
    first_block_azimuth_hundredths,
    validate_velodyne_packet,
)
from ad_morai_bridge.protocol_records import VelodynePacketRecord


def _packet(packet_id: int, first_azimuth: int) -> bytes:
    packet = bytearray(1_206)
    for offset in range(0, 1_200, 100):
        packet[offset : offset + 2] = b"\xff\xee"
        packet[offset + 2 : offset + 4] = first_azimuth.to_bytes(2, "little")
    packet[1_200:1_204] = packet_id.to_bytes(4, "little")
    return bytes(packet)


def _packet_id(data: bytes) -> int:
    return int.from_bytes(data[1_200:1_204], "little")


def _record(
    packet_id: int,
    azimuth: int,
    *,
    stamp: tuple[int, int] | None = None,
    data: bytes | None = None,
    metadata_azimuth: int | None = None,
) -> VelodynePacketRecord:
    return VelodynePacketRecord(
        stamp=stamp if stamp is not None else (packet_id + 1, packet_id),
        data=data if data is not None else _packet(packet_id, azimuth),
        first_azimuth_hundredths=(
            azimuth if metadata_azimuth is None else metadata_azimuth
        ),
    )


def _ids(records) -> list[int]:
    return [_packet_id(record.data) for record in records]


def _raw_message(record: VelodynePacketRecord) -> RawPacket:
    message = RawPacket()
    message.header.stamp.sec, message.header.stamp.nanosec = record.stamp
    message.header.frame_id = "lidar_link"
    message.stream = "velodyne"
    message.data = record.data
    return message


@contextmanager
def _running_adapter(*, point_timing_mode: str | None = None):
    args = None
    if point_timing_mode is not None:
        args = [
            "--ros-args",
            "-p",
            f"point_timing_mode:={point_timing_mode}",
        ]
    rclpy.init(args=args)
    adapter = None
    try:
        adapter = AdVelodyneAdapter()
        yield adapter
    finally:
        if adapter is not None:
            adapter.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_codec_validates_all_twelve_flags_and_azimuths():
    packet = _packet(7, 12_345)

    assert validate_velodyne_packet(packet) == packet
    assert first_block_azimuth_hundredths(packet) == 12_345

    bad_flag = bytearray(packet)
    bad_flag[1_100:1_102] = b"xx"
    with pytest.raises(PacketFormatError, match="flag"):
        validate_velodyne_packet(bytes(bad_flag))

    bad_azimuth = bytearray(packet)
    bad_azimuth[1_102:1_104] = (36_000).to_bytes(2, "little")
    with pytest.raises(PacketFormatError, match="azimuth"):
        validate_velodyne_packet(bytes(bad_azimuth))

    with pytest.raises(PacketFormatError, match="size"):
        validate_velodyne_packet(packet[:-1])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_packets": 0}, "max_packets"),
        ({"max_packets": -1}, "max_packets"),
        ({"max_packets": 1.5}, "max_packets"),
        ({"max_packets": True}, "max_packets"),
        ({"idle_timeout_sec": 0.0}, "idle_timeout_sec"),
        ({"idle_timeout_sec": -1.0}, "idle_timeout_sec"),
        ({"idle_timeout_sec": True}, "idle_timeout_sec"),
        ({"idle_timeout_sec": math.nan}, "idle_timeout_sec"),
        ({"idle_timeout_sec": math.inf}, "idle_timeout_sec"),
        ({"idle_timeout_sec": -math.inf}, "idle_timeout_sec"),
    ],
)
def test_assembler_requires_finite_positive_capacity_and_timeout(kwargs, match):
    with pytest.raises(ValueError, match=match):
        VelodyneScanAssembler(**kwargs)


def test_increasing_and_duplicate_azimuths_stay_in_input_order():
    assembler = VelodyneScanAssembler()

    assert assembler.push(_record(1, 100), received_monotonic=1.0) is None
    assert assembler.push(_record(2, 200), received_monotonic=1.1) is None
    assert assembler.push(_record(3, 200), received_monotonic=1.2) is None

    assert _ids(assembler.flush()) == [1, 2, 3]


def test_true_wrap_emits_old_scan_before_buffering_wrapped_packet():
    assembler = VelodyneScanAssembler()
    assembler.push(_record(1, 35_800), received_monotonic=1.0)
    assembler.push(_record(2, 35_900), received_monotonic=1.1)

    emitted = assembler.push(_record(3, 100), received_monotonic=1.2)

    assert _ids(emitted) == [1, 2]
    assert _ids(assembler.flush()) == [3]


@pytest.mark.parametrize(
    ("previous", "stale"),
    [(1_000, 900), (100, 35_900)],
)
def test_stale_reverse_packet_is_not_treated_as_wrap(previous, stale):
    assembler = VelodyneScanAssembler()
    assembler.push(_record(1, previous), received_monotonic=0.0)

    with pytest.raises(PacketOrderingError):
        assembler.push(_record(2, stale), received_monotonic=0.199)

    assert _ids(assembler.flush_if_idle(now_monotonic=0.2)) == [1]


def test_rejected_azimuth_does_not_move_the_ordering_watermark():
    assembler = VelodyneScanAssembler()
    assembler.push(_record(1, 1_000), received_monotonic=1.0)

    with pytest.raises(PacketOrderingError):
        assembler.push(_record(2, 900), received_monotonic=1.1)
    with pytest.raises(PacketOrderingError):
        assembler.push(_record(3, 950), received_monotonic=1.2)

    assert _ids(assembler.flush()) == [1]


def test_decreasing_stamp_is_rejected_without_moving_stamp_watermark():
    assembler = VelodyneScanAssembler()
    assembler.push(
        _record(1, 1_000, stamp=(10, 0)), received_monotonic=1.0
    )

    with pytest.raises(PacketOrderingError, match="stamp"):
        assembler.push(
            _record(2, 1_100, stamp=(9, 0)), received_monotonic=1.1
        )
    with pytest.raises(PacketOrderingError, match="stamp"):
        assembler.push(
            _record(3, 1_200, stamp=(9, 500_000_000)),
            received_monotonic=1.2,
        )

    assert _ids(assembler.flush()) == [1]


@pytest.mark.parametrize(
    "stamp",
    [(-1, 0), (0, -1), (0, 1_000_000_000), (1.0, 0), (1, 0.0)],
)
def test_invalid_stamp_does_not_change_buffer_or_idle_deadline(stamp):
    assembler = VelodyneScanAssembler()
    assembler.push(_record(1, 100, stamp=(1, 0)), received_monotonic=0.0)

    with pytest.raises(ValueError, match="stamp"):
        assembler.push(
            _record(2, 200, stamp=stamp), received_monotonic=0.199
        )

    assert _ids(assembler.flush_if_idle(now_monotonic=0.2)) == [1]


@pytest.mark.parametrize("received", [math.nan, math.inf, -math.inf])
def test_nonfinite_monotonic_time_does_not_extend_idle_deadline(received):
    assembler = VelodyneScanAssembler()
    assembler.push(_record(1, 100), received_monotonic=0.0)

    with pytest.raises(ValueError, match="monotonic"):
        assembler.push(_record(2, 200), received_monotonic=received)

    assert _ids(assembler.flush_if_idle(now_monotonic=0.2)) == [1]


def test_decreasing_monotonic_time_does_not_move_time_watermark():
    assembler = VelodyneScanAssembler()
    assembler.push(_record(1, 100), received_monotonic=0.0)

    with pytest.raises(PacketOrderingError, match="monotonic"):
        assembler.push(_record(2, 200), received_monotonic=-0.1)
    with pytest.raises(PacketOrderingError, match="monotonic"):
        assembler.push(_record(3, 300), received_monotonic=-0.05)

    assert _ids(assembler.flush_if_idle(now_monotonic=0.2)) == [1]


def test_malformed_or_inconsistent_record_does_not_change_assembler_state():
    assembler = VelodyneScanAssembler()
    assembler.push(_record(1, 100), received_monotonic=1.0)
    bad = bytearray(_packet(2, 200))
    bad[500:502] = b"xx"

    with pytest.raises(PacketFormatError):
        assembler.push(
            _record(2, 200, data=bytes(bad)), received_monotonic=1.1
        )
    with pytest.raises(PacketFormatError, match="metadata"):
        assembler.push(
            _record(3, 300, metadata_azimuth=301),
            received_monotonic=1.2,
        )

    assert _ids(assembler.flush()) == [1]


def test_packet_cap_emits_512_then_buffers_the_next_without_loss_or_reorder():
    assembler = VelodyneScanAssembler(max_packets=512)
    for packet_id in range(512):
        assert assembler.push(
            _record(packet_id, packet_id),
            received_monotonic=float(packet_id),
        ) is None

    emitted = assembler.push(_record(512, 512), received_monotonic=512.0)
    tail = assembler.flush()

    assert _ids(emitted) == list(range(512))
    assert _ids(tail) == [512]


def test_packet_cap_and_wrap_on_same_packet_emit_old_scan_only_once():
    assembler = VelodyneScanAssembler(max_packets=512)
    for packet_id in range(512):
        assert assembler.push(
            _record(packet_id, 35_000),
            received_monotonic=float(packet_id),
        ) is None

    emitted = assembler.push(_record(512, 100), received_monotonic=512.0)

    assert _ids(emitted) == list(range(512))
    assert _ids(assembler.flush()) == [512]
    assert assembler.flush() is None


def test_idle_and_explicit_flush_emit_each_partial_scan_once():
    assembler = VelodyneScanAssembler(idle_timeout_sec=0.2)
    assembler.push(_record(1, 100), received_monotonic=0.0)

    assert assembler.flush_if_idle(now_monotonic=0.199) is None
    assert _ids(assembler.flush_if_idle(now_monotonic=0.2)) == [1]
    assert assembler.flush_if_idle(now_monotonic=10.0) is None
    assert assembler.flush() is None

    assembler.push(_record(2, 200), received_monotonic=10.1)
    assert _ids(assembler.flush()) == [2]
    assert assembler.flush() is None


@pytest.mark.parametrize("flush_mode", ["explicit", "idle"])
def test_flush_clears_scan_azimuth_but_keeps_global_time_watermarks(flush_mode):
    assembler = VelodyneScanAssembler(idle_timeout_sec=0.2)
    assembler.push(
        _record(1, 100, stamp=(10, 0)), received_monotonic=0.0
    )
    if flush_mode == "explicit":
        emitted = assembler.flush()
    else:
        emitted = assembler.flush_if_idle(now_monotonic=0.2)
    assert _ids(emitted) == [1]

    assert assembler.push(
        _record(2, 35_900, stamp=(11, 0)), received_monotonic=0.3
    ) is None
    with pytest.raises(PacketOrderingError, match="stamp"):
        assembler.push(
            _record(3, 100, stamp=(10, 0)), received_monotonic=0.4
        )
    with pytest.raises(PacketOrderingError, match="monotonic"):
        assembler.push(
            _record(4, 100, stamp=(12, 0)), received_monotonic=0.2
        )

    assert _ids(assembler.flush()) == [2]


def test_raw_packet_conversion_rejects_wrong_boundary_and_preserves_stamp():
    source = _record(7, 1_234, stamp=(10, 20))
    message = _raw_message(source)

    assert raw_packet_record(message) == source

    message.stream = "diagnostic_velodyne"
    with pytest.raises(PacketFormatError, match="stream"):
        raw_packet_record(message)
    message.stream = "velodyne"
    message.header.frame_id = "wrong_lidar"
    with pytest.raises(PacketFormatError, match="frame"):
        raw_packet_record(message)


def test_raw_packet_conversion_rejects_invalid_stamp_and_payload():
    message = _raw_message(_record(1, 100))
    message.header.stamp.sec = -1
    with pytest.raises(PacketFormatError, match="stamp"):
        raw_packet_record(message)

    message = _raw_message(_record(1, 100))
    bad = bytearray(message.data)
    bad[1_100:1_102] = b"xx"
    message.data = bytes(bad)
    with pytest.raises(PacketFormatError, match="flag"):
        raw_packet_record(message)


def test_scan_conversion_uses_azimuth_timing_from_first_packet_at_10_hz():
    records = (
        _record(1, 1_000, stamp=(10, 20)),
        # Deliberately jitter the arrival stamp. Half a rotation at 10 Hz is
        # 50 ms, irrespective of when MORAI's UDP burst reaches this process.
        _record(2, 19_000, stamp=(11, 30)),
    )

    message = velodyne_scan_message(records)

    assert message.header.frame_id == "lidar_link"
    # velodyne_pointcloud computes each point time as
    # packet.stamp - scan.header.stamp + firing_offset.  The scan header must
    # therefore be the first packet time so point offsets start at zero.
    assert (message.header.stamp.sec, message.header.stamp.nanosec) == (10, 20)
    assert [
        (packet.stamp.sec, packet.stamp.nanosec) for packet in message.packets
    ] == [(10, 20), (10, 50_000_020)]
    assert [bytes(packet.data) for packet in message.packets] == [
        record.data for record in records
    ]
    assert [_packet_id(bytes(packet.data)) for packet in message.packets] == [1, 2]


def test_zero_timing_uses_last_packet_as_snapshot_measurement_time():
    records = (
        _record(1, 1_000, stamp=(10, 20)),
        _record(2, 19_000, stamp=(11, 30)),
    )

    message = velodyne_scan_message(records, point_timing_mode="zero")

    # MORAI emits one snapshot-like cloud as a UDP packet burst.  With every
    # point offset forced to zero, the scan and packet stamps must represent
    # the completed snapshot rather than the start of packet delivery.
    assert (message.header.stamp.sec, message.header.stamp.nanosec) == (11, 30)
    assert [
        (packet.stamp.sec, packet.stamp.nanosec) for packet in message.packets
    ] == [(11, 30), (11, 30)]


def test_zero_timing_adapter_groups_one_morai_footer_per_snapshot():
    with _running_adapter(point_timing_mode="zero") as adapter:
        class CapturePublisher:
            def __init__(self):
                self.messages = []

            def publish(self, message):
                self.messages.append(message)

        capture = CapturePublisher()
        adapter._publisher = capture
        first_snapshot = (
            _record(1, 750, stamp=(10, 1), data=_packet(100, 750)),
            _record(2, 10_000, stamp=(10, 2), data=_packet(100, 10_000)),
            _record(3, 20_000, stamp=(10, 3), data=_packet(100, 20_000)),
            _record(4, 30_000, stamp=(10, 4), data=_packet(100, 30_000)),
            _record(5, 35_790, stamp=(10, 5), data=_packet(100, 35_790)),
            # MORAI's last packet crosses the 360 -> 0 degree boundary but
            # retains the same footer timestamp as the rest of the snapshot.
            _record(6, 270, stamp=(10, 6), data=_packet(100, 270)),
        )
        for record in first_snapshot:
            adapter._on_packet(_raw_message(record))

        # A complete MORAI burst must publish at its own wrapped tail; waiting
        # for the next footer would add one full sensor period of latency.
        assert len(capture.messages) == 1
        assert [
            _packet_id(bytes(packet.data))
            for packet in capture.messages[0].packets
        ] == [100, 100, 100, 100, 100, 100]


@pytest.mark.parametrize("mode", ["", "legacy", "arrival", None, True])
def test_scan_conversion_rejects_unknown_point_timing_mode(mode):
    with pytest.raises(ValueError, match="point_timing_mode"):
        velodyne_scan_message((_record(1, 100),), point_timing_mode=mode)


@pytest.mark.parametrize("scan_rate_hz", [0.0, -1.0, math.nan, math.inf, True])
def test_scan_conversion_rejects_invalid_scan_rate(scan_rate_hz):
    with pytest.raises(ValueError, match="scan_rate_hz"):
        velodyne_scan_message((_record(1, 100),), scan_rate_hz=scan_rate_hz)


def test_scan_conversion_rejects_empty_scan():
    with pytest.raises(ValueError, match="empty"):
        velodyne_scan_message(())


def test_adapter_graph_topics_types_and_qos_are_exact():
    with _running_adapter() as adapter:
        assert adapter.get_name() == "ad_velodyne_adapter"
        project_topics = {
            topic: types
            for topic, types in adapter.get_topic_names_and_types()
            if topic.startswith("/ad/")
        }
        # Topic discovery is process-global and may include RViz or another
        # concurrently running development node.  Assert this adapter's exact
        # endpoints without assuming the whole ROS domain is otherwise empty.
        assert project_topics[INPUT_TOPIC] == ["ad_morai_interfaces/msg/RawPacket"]
        assert project_topics[OUTPUT_TOPIC] == ["velodyne_msgs/msg/VelodyneScan"]

        subscriptions = adapter.get_subscriptions_info_by_topic(INPUT_TOPIC)
        assert len(subscriptions) == 1
        assert subscriptions[0].topic_type == "ad_morai_interfaces/msg/RawPacket"
        assert subscriptions[0].qos_profile.reliability == ReliabilityPolicy.BEST_EFFORT
        assert subscriptions[0].qos_profile.durability == DurabilityPolicy.VOLATILE
        assert adapter._subscription.qos_profile.history == HistoryPolicy.KEEP_LAST
        # MORAI emits a VLP-16 rotation as a packet burst (roughly 75 packets
        # at 10 Hz); a depth of five drops most of a scan under normal CPU
        # contention. One bounded rotation fits in depth 100.
        assert adapter._subscription.qos_profile.depth == 100

        publishers = adapter.get_publishers_info_by_topic(OUTPUT_TOPIC)
        assert len(publishers) == 1
        assert publishers[0].topic_type == "velodyne_msgs/msg/VelodyneScan"
        assert publishers[0].qos_profile.reliability == ReliabilityPolicy.RELIABLE
        assert publishers[0].qos_profile.durability == DurabilityPolicy.VOLATILE
        assert adapter._publisher.qos_profile.history == HistoryPolicy.KEEP_LAST
        assert adapter._publisher.qos_profile.depth == 10

        assert not adapter.get_subscriptions_info_by_topic(
            "/ad/udp/raw/velodyne"
        )


def test_adapter_contains_bad_messages_and_rate_limits_logging(monkeypatch):
    from ad_morai_bridge import velodyne_adapter_node as adapter_module

    class CapturePublisher:
        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    class CaptureLogger:
        def __init__(self):
            self.warnings = []

        def warning(self, message):
            self.warnings.append(message)

    now = [0.0]
    logger = CaptureLogger()
    with _running_adapter() as adapter:
        capture = CapturePublisher()
        adapter._publisher = capture
        monkeypatch.setattr(adapter_module.time, "monotonic", lambda: now[0])
        monkeypatch.setattr(AdVelodyneAdapter, "get_logger", lambda _self: logger)
        bad = _raw_message(_record(1, 100))
        bad.stream = "wrong"

        for _ in range(5):
            adapter._on_packet(bad)

        adapter._on_packet(_raw_message(_record(2, 200)))
        # Drive the public idle path instead of a private flush hook.
        now[0] = 1000.0
        adapter._on_idle_timer()

        assert len(logger.warnings) == 4
        assert len(capture.messages) == 1
        assert [
            _packet_id(bytes(packet.data))
            for packet in capture.messages[0].packets
        ] == [2]


def test_publish_failure_releases_lock_before_logging_and_is_contained(
    monkeypatch,
):
    class FailingPublisher:
        def publish(self, _message):
            raise RuntimeError("simulated publish failure")

    class CaptureLogger:
        def __init__(self):
            self.warnings = []

        def warning(self, message):
            self.warnings.append(message)

    logger = CaptureLogger()
    with _running_adapter() as adapter:
        adapter._publisher = FailingPublisher()
        monkeypatch.setattr(AdVelodyneAdapter, "get_logger", lambda _self: logger)

        adapter._on_packet(_raw_message(_record(1, 100)))
        from ad_morai_bridge import velodyne_adapter_node as adapter_module

        monkeypatch.setattr(adapter_module.time, "monotonic", lambda: 1.0e9)
        adapter._on_idle_timer()

        assert len(logger.warnings) == 1
        assert "publish failure" in logger.warnings[0]
        with adapter._lock:
            pass


def test_idle_timer_publishes_partial_once(monkeypatch):
    from ad_morai_bridge import velodyne_adapter_node as adapter_module

    class CapturePublisher:
        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    now = [0.0]
    with _running_adapter() as adapter:
        capture = CapturePublisher()
        adapter._publisher = capture
        monkeypatch.setattr(adapter_module.time, "monotonic", lambda: now[0])
        adapter._on_packet(_raw_message(_record(1, 100)))

        now[0] = 0.199
        adapter._on_idle_timer()
        assert capture.messages == []
        now[0] = 0.2
        adapter._on_idle_timer()
        adapter._on_idle_timer()

        assert len(capture.messages) == 1


def test_shutdown_cancels_timer_without_publishing_and_is_idempotent(monkeypatch):
    from ad_morai_bridge import velodyne_adapter_node as adapter_module

    class CapturePublisher:
        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    class TimerProbe:
        def __init__(self, timer):
            self.timer = timer
            self.cancels = 0

        def cancel(self):
            self.cancels += 1
            self.timer.cancel()

    rclpy.init()
    adapter = AdVelodyneAdapter()
    capture = CapturePublisher()
    timer = TimerProbe(adapter._idle_timer)
    adapter._publisher = capture
    adapter._idle_timer = timer
    monkeypatch.setattr(adapter_module.time, "monotonic", lambda: 1.0)
    adapter._on_packet(_raw_message(_record(1, 100)))

    try:
        adapter.destroy_node()
        adapter.destroy_node()
        adapter = None
    finally:
        if adapter is not None:
            adapter.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    assert timer.cancels == 1
    assert capture.messages == []
