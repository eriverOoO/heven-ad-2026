from contextlib import contextmanager
import struct
import time

import pytest
import rclpy
from ad_morai_interfaces.msg import CtrlCmd
from rclpy.parameter import Parameter
from rclpy.qos import ReliabilityPolicy

from ad_morai_bridge.control_watchdog import ControlSafetyGate
from ad_morai_bridge.morai_bridge_node import (
    STREAMS,
    AdMoraiBridge,
    StreamCounters,
    diagnostic_state,
)
from ad_morai_bridge.protocol_records import CtrlCommandRecord


TEST_DOMAIN_ID = 232


@contextmanager
def _running(node_factory):
    rclpy.init(domain_id=TEST_DOMAIN_ID)
    bridge = None
    try:
        bridge = node_factory()
        yield bridge
    finally:
        try:
            if bridge is not None:
                bridge.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


def _overrides(*, publish_raw=False, control_enabled=False, extra=()):
    values = [Parameter(f"{name}.enabled", value=False) for name in STREAMS]
    values.extend(
        [
            Parameter("control.enabled", value=control_enabled),
            Parameter("publish_raw_packets", value=publish_raw),
        ]
    )
    values.extend(extra)
    return values


def _endpoint(node, topic):
    endpoints = node.get_publishers_info_by_topic(topic)
    assert len(endpoints) == 1, topic
    return endpoints[0]


class _CapturePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def _nmea(body: str) -> bytes:
    checksum = 0
    for char in body:
        checksum ^= ord(char)
    return f"${body}*{checksum:02X}\r\n".encode()


def _imu_packet(timestamped: bool = True, stamp=(12, 34)) -> bytes:
    values = (1.0, 0.1, 0.2, 0.3, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    payload = (struct.pack("<ii", *stamp) if timestamped else b"") + struct.pack(
        "<10d", *values
    )
    return b"#IMUData$" + struct.pack("<i3i", len(payload), 0, 0, 0) + payload + b"\r\n"


def _ego_packet(stamp=(12, 34)) -> bytes:
    payload = struct.pack(
        "<ii2bfi2f3f3f3f3f3f3f3ff38s",
        *stamp,
        2,
        4,
        27.5,
        99,
        0.2,
        0.0,
        4.7,
        1.8,
        1.5,
        0.9,
        2.9,
        0.8,
        10.0,
        20.0,
        0.1,
        0.01,
        0.02,
        1.2,
        27.5,
        0.0,
        0.0,
        0.0,
        0.0,
        0.1,
        0.2,
        0.0,
        0.0,
        0.03,
        b"LINK_EGO",
    )
    return b"#MoraiInfo$" + struct.pack("<i3i", len(payload), 0, 0, 0) + payload + b"\r\n"


def _camera_packet(stamp=(12, 34), jpeg=b"\xff\xd8frame\xff\xd9") -> bytes:
    return _camera_fragment(0, jpeg, True, stamp)


def _camera_fragment(index, data, final, stamp=(12, 34)) -> bytes:
    return (
        b"MOR"
        + struct.pack("<iiii", stamp[0], stamp[1], index, len(data))
        + data.ljust(64_979, b"\0")
        + (b"EI" if final else b"AI")
    )


def test_bridge_node_starts_with_the_topics_downstream_consumes():
    with _running(
        lambda: AdMoraiBridge(parameter_overrides=_overrides())
    ) as bridge:
        assert bridge.get_name() == "ad_morai_bridge"

        for topic, topic_type in (
            ("/ad/vehicle/status", "ad_morai_interfaces/msg/EgoVehicleStatus"),
            ("/ad/vehicle/status/full", "ad_morai_interfaces/msg/EgoVehicleStatus"),
            ("/ad/safety/collisions", "ad_morai_interfaces/msg/CollisionArray"),
            ("/ad/sensors/gps/fix", "sensor_msgs/msg/NavSatFix"),
            ("/ad/sensors/gps/time_reference", "sensor_msgs/msg/TimeReference"),
            ("/ad/sensors/imu/data", "sensor_msgs/msg/Imu"),
            ("/ad/sensors/timing", "ad_morai_interfaces/msg/SensorTiming"),
            ("/ad/sensors/lidar/raw", "ad_morai_interfaces/msg/RawPacket"),
            ("/diagnostics", "diagnostic_msgs/msg/DiagnosticArray"),
        ):
            endpoint = _endpoint(bridge, topic)
            assert endpoint.topic_type == topic_type

        # A 10 Hz VLP-16 revolution arrives as a burst of roughly 75 UDP
        # packets; the raw LiDAR publisher must retain a full revolution
        # for the scan assembler. (Depth is not shared over DDS discovery,
        # so read it from the publisher itself.)
        velodyne = _endpoint(bridge, "/ad/sensors/lidar/raw")
        assert velodyne.qos_profile.reliability == ReliabilityPolicy.BEST_EFFORT
        assert bridge._stream_publishers["velodyne"].qos_profile.depth == 100

        # Control stays off unless explicitly enabled.
        assert not bridge.get_subscriptions_info_by_topic("/ad/control/command")


def test_gps_publishes_each_full_sentence_and_keeps_paired_fix():
    with _running(
        lambda: AdMoraiBridge(parameter_overrides=_overrides())
    ) as bridge:
        rmc = _CapturePublisher()
        gga = _CapturePublisher()
        fix = _CapturePublisher()
        bridge._gps_rmc_pub = rmc
        bridge._gps_gga_pub = gga
        bridge._stream_publishers["gps"] = fix

        bridge._decode_and_publish(
            "gps",
            _nmea("GPRMC,123519,A,4807.038,N,01131.000,E,22.4,84.4,230394,3.1,E,D"),
            1.0,
        )
        assert len(rmc.messages) == 1
        assert gga.messages == []
        assert fix.messages == []

        bridge._decode_and_publish(
            "gps",
            _nmea("GPGGA,123520,4807.038,N,01131.000,E,2,11,0.8,545.4,M,46.9,M,1.5,0042"),
            2.0,
        )
        assert len(rmc.messages) == 1
        assert len(gga.messages) == 1
        assert len(fix.messages) == 1


def test_gps_publishes_whole_second_time_reference_and_promotes_paired_rmc_epoch():
    epoch = 764_426_119
    with _running(
        lambda: AdMoraiBridge(parameter_overrides=_overrides())
    ) as bridge:
        fixes = _CapturePublisher()
        references = _CapturePublisher()
        timing = _CapturePublisher()
        bridge._stream_publishers["gps"] = fixes
        bridge._gps_time_reference_pub = references
        bridge._timing_pub = timing
        arrivals = iter(((epoch, 200_000_000), (epoch, 300_000_000)))
        bridge._arrival_stamp = lambda _arrived: next(arrivals)

        bridge._decode_and_publish(
            "gps",
            _nmea(
                "GPRMC,123519.999999999,A,4807.038,N,01131.000,E,"
                "22.4,84.4,230394,3.1,E,D"
            ),
            1.0,
        )
        bridge._decode_and_publish(
            "gps",
            _nmea(
                "GPGGA,123519.1,4807.038,N,01131.000,E,2,11,0.8,"
                "545.4,M,46.9,M,1.5,0042"
            ),
            2.0,
        )

        assert len(references.messages) == 1
        reference = references.messages[0]
        assert (reference.time_ref.sec, reference.time_ref.nanosec) == (epoch, 0)
        assert len(fixes.messages) == 1
        assert (
            fixes.messages[0].header.stamp.sec,
            fixes.messages[0].header.stamp.nanosec,
        ) == (epoch, 300_000_000)
        assert len(timing.messages) == 2
        assert timing.messages[0].stream == "gps/rmc"
        assert timing.messages[1].stream == "gps/fix"
        assert timing.messages[-1].source_selected is False
        assert timing.messages[-1].duplicate is True
        assert timing.messages[-1].normalized_published is True
        assert bridge._counters["gps"].duplicates == 1


def test_void_rmc_still_publishes_safe_time_reference_but_does_not_seed_fix():
    epoch = 764_426_119
    with _running(
        lambda: AdMoraiBridge(parameter_overrides=_overrides())
    ) as bridge:
        references = _CapturePublisher()
        fixes = _CapturePublisher()
        timing = _CapturePublisher()
        bridge._gps_time_reference_pub = references
        bridge._stream_publishers["gps"] = fixes
        bridge._timing_pub = timing
        arrivals = iter(((epoch, 100), (epoch, 200)))
        bridge._arrival_stamp = lambda _arrived: next(arrivals)

        bridge._decode_and_publish(
            "gps",
            _nmea(
                "GPRMC,123519,V,4807.038,N,01131.000,E,"
                "10.0,20.0,230394,,,A"
            ),
            1.0,
        )
        bridge._decode_and_publish(
            "gps",
            _nmea(
                "GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,"
                "1.0,M,0.0,M,,"
            ),
            2.0,
        )

        assert len(references.messages) == 1
        assert len(fixes.messages) == 1
        assert (
            fixes.messages[0].header.stamp.sec,
            fixes.messages[0].header.stamp.nanosec,
        ) == (epoch, 200)
        assert timing.messages[0].source_rejected is False
        assert timing.messages[0].normalized_published is True
        assert timing.messages[1].has_source_stamp is False


def test_regressing_rmc_time_reference_is_rejected_and_audited():
    newer_epoch = 764_426_120
    with _running(
        lambda: AdMoraiBridge(parameter_overrides=_overrides())
    ) as bridge:
        references = _CapturePublisher()
        timing = _CapturePublisher()
        bridge._gps_time_reference_pub = references
        bridge._timing_pub = timing
        arrivals = iter(
            ((newer_epoch - 1, 900_000_000), (newer_epoch, 0))
        )
        bridge._arrival_stamp = lambda _arrived: next(arrivals)

        bridge._decode_and_publish(
            "gps",
            _nmea(
                "GPRMC,123520,A,4807.038,N,01131.000,E,"
                "0.0,0.0,230394,,,A"
            ),
            1.0,
        )
        bridge._decode_and_publish(
            "gps",
            _nmea(
                "GPRMC,123519,A,4807.038,N,01131.000,E,"
                "0.0,0.0,230394,,,A"
            ),
            2.0,
        )

        assert len(references.messages) == 1
        assert len(timing.messages) == 2
        assert timing.messages[-1].source_rejected is True
        assert timing.messages[-1].normalized_published is False
        assert bridge._counters["gps"].source_rejected == 1


def test_invalid_rmc_epoch_preserves_sentence_and_counts_rejected_candidate():
    packet = _nmea(
        "GPRMC,123519.7,A,4807.038,N,01131.000,E,"
        "0.0,0.0,321399,,,A"
    )
    with _running(
        lambda: AdMoraiBridge(parameter_overrides=_overrides())
    ) as bridge:
        rmc = _CapturePublisher()
        references = _CapturePublisher()
        timing = _CapturePublisher()
        bridge._gps_rmc_pub = rmc
        bridge._gps_time_reference_pub = references
        bridge._timing_pub = timing
        bridge._arrival_stamp = lambda _arrived: (700, 123)

        bridge._decode_and_publish("gps", packet, 1.0)

        assert len(rmc.messages) == 1
        assert rmc.messages[0].sentence == packet.decode().rstrip("\r\n")
        assert references.messages == []
        assert len(timing.messages) == 1
        assert timing.messages[0].has_source_stamp is False
        assert timing.messages[0].source_rejected is True
        assert timing.messages[0].normalized_published is False
        assert bridge._counters["gps"].source_rejected == 1


def test_out_of_window_rmc_source_is_preserved_but_not_time_reference():
    epoch = 764_426_119
    with _running(
        lambda: AdMoraiBridge(parameter_overrides=_overrides())
    ) as bridge:
        references = _CapturePublisher()
        timing = _CapturePublisher()
        bridge._gps_time_reference_pub = references
        bridge._timing_pub = timing
        bridge._arrival_stamp = lambda _arrived: (epoch + 5, 0)

        bridge._decode_and_publish(
            "gps",
            _nmea(
                "GPRMC,123519,A,4807.038,N,01131.000,E,"
                "0.0,0.0,230394,,,A"
            ),
            1.0,
        )

        assert references.messages == []
        assert len(timing.messages) == 1
        assert (timing.messages[0].source_sec, timing.messages[0].source_nanosec) == (
            epoch,
            0,
        )
        assert timing.messages[0].source_rejected is True
        assert timing.messages[0].normalized_published is False


def test_rmc_duplicate_is_counted_and_retained_in_time_audit_stream():
    epoch = 764_426_119
    with _running(
        lambda: AdMoraiBridge(parameter_overrides=_overrides())
    ) as bridge:
        references = _CapturePublisher()
        timing = _CapturePublisher()
        bridge._gps_time_reference_pub = references
        bridge._timing_pub = timing
        arrivals = iter(((epoch, 100), (epoch, 200)))
        bridge._arrival_stamp = lambda _arrived: next(arrivals)
        packet = _nmea(
            "GPRMC,123519,A,4807.038,N,01131.000,E,"
            "0.0,0.0,230394,,,A"
        )

        bridge._decode_and_publish("gps", packet, 1.0)
        bridge._decode_and_publish("gps", packet, 2.0)

        assert len(references.messages) == 2
        assert len(timing.messages) == 2
        assert timing.messages[-1].duplicate is True
        assert timing.messages[-1].normalized_published is True
        assert bridge._counters["gps"].duplicates == 1


def test_gga_without_rmc_has_no_epoch_and_uses_arrival_fallback():
    with _running(
        lambda: AdMoraiBridge(parameter_overrides=_overrides())
    ) as bridge:
        fixes = _CapturePublisher()
        references = _CapturePublisher()
        timing = _CapturePublisher()
        bridge._stream_publishers["gps"] = fixes
        bridge._gps_time_reference_pub = references
        bridge._timing_pub = timing
        bridge._arrival_stamp = lambda _arrived: (700, 123)

        bridge._decode_and_publish(
            "gps",
            _nmea(
                "GPGGA,123519.25,4807.038,N,01131.000,E,2,11,0.8,"
                "545.4,M,46.9,M,1.5,0042"
            ),
            1.0,
        )

        assert references.messages == []
        assert len(fixes.messages) == 1
        assert (
            fixes.messages[0].header.stamp.sec,
            fixes.messages[0].header.stamp.nanosec,
        ) == (700, 123)
        assert timing.messages[0].has_source_stamp is False
        assert timing.messages[0].source_rejected is False


def test_imu_packet_uses_receipt_header_and_preserves_device_stamp():
    with _running(
        lambda: AdMoraiBridge(parameter_overrides=_overrides())
    ) as bridge:
        full = _CapturePublisher()
        normalized = _CapturePublisher()
        bridge._imu_full_pub = full
        bridge._stream_publishers["imu"] = normalized
        bridge._arrival_stamp = lambda _arrived: (12, 100)

        bridge._decode_and_publish("imu", _imu_packet(), 1.0)

        assert len(full.messages) == 1
        assert len(normalized.messages) == 1
        assert full.messages[0].has_device_stamp is True
        assert (full.messages[0].device_stamp.sec, full.messages[0].device_stamp.nanosec) == (
            12,
            34,
        )
        assert (
            normalized.messages[0].header.stamp.sec,
            normalized.messages[0].header.stamp.nanosec,
        ) == (12, 100)


def test_imu_source_stamp_repeat_is_audited_without_suppressing_normalized():
    with _running(
        lambda: AdMoraiBridge(parameter_overrides=_overrides())
    ) as bridge:
        full = _CapturePublisher()
        normalized = _CapturePublisher()
        timing = _CapturePublisher()
        bridge._imu_full_pub = full
        bridge._stream_publishers["imu"] = normalized
        bridge._timing_pub = timing
        arrivals = iter(((50, 100), (50, 200)))
        bridge._arrival_stamp = lambda _arrived: next(arrivals)

        bridge._decode_and_publish("imu", _imu_packet(stamp=(50, 10)), 1.0)
        bridge._decode_and_publish("imu", _imu_packet(stamp=(50, 10)), 2.0)

        assert len(full.messages) == 2
        assert len(timing.messages) == 2
        assert len(normalized.messages) == 2
        assert timing.messages[-1].duplicate is True
        assert timing.messages[-1].normalized_published is True
        counter = bridge._counters["imu"]
        assert counter.source_selected == 0
        assert counter.arrival_fallback == 2
        assert counter.duplicates == 1


@pytest.mark.parametrize("rate_hz", [20, 30, 50])
def test_imu_receipt_contract_is_one_for_one_at_supported_rates(rate_hz):
    with _running(
        lambda: AdMoraiBridge(parameter_overrides=_overrides())
    ) as bridge:
        full = _CapturePublisher()
        normalized = _CapturePublisher()
        timing = _CapturePublisher()
        bridge._imu_full_pub = full
        bridge._stream_publishers["imu"] = normalized
        bridge._timing_pub = timing

        period_ns = 1_000_000_000 // rate_hz
        arrivals = [
            (100, index * period_ns) for index in range(rate_hz)
        ]
        bridge._arrival_stamp = lambda received: arrivals[int(received)]

        for index in range(rate_hz):
            # Repeat every source stamp once to model the observed MORAI high-rate
            # pattern.  The bridge may audit the repeat but must not infer that
            # either valid packet is disposable.
            source_nanosec = (index // 2) * 2 * period_ns
            bridge._decode_and_publish(
                "imu",
                _imu_packet(stamp=(100, source_nanosec)),
                float(index),
            )

        assert len(full.messages) == rate_hz
        assert len(normalized.messages) == rate_hz
        assert len(timing.messages) == rate_hz
        assert [
            (message.header.stamp.sec, message.header.stamp.nanosec)
            for message in normalized.messages
        ] == arrivals
        assert all(message.normalized_published for message in timing.messages)


def test_invalid_imu_source_falls_back_and_preserves_signed_raw_audit_pair():
    with _running(
        lambda: AdMoraiBridge(parameter_overrides=_overrides())
    ) as bridge:
        full = _CapturePublisher()
        normalized = _CapturePublisher()
        timing = _CapturePublisher()
        bridge._imu_full_pub = full
        bridge._stream_publishers["imu"] = normalized
        bridge._timing_pub = timing
        bridge._arrival_stamp = lambda _arrived: (60, 100)

        bridge._decode_and_publish(
            "imu", _imu_packet(stamp=(60, 1_000_000_000)), 1.0
        )

        assert len(full.messages) == len(normalized.messages) == 1
        assert full.messages[0].has_device_stamp is False
        assert (
            normalized.messages[0].header.stamp.sec,
            normalized.messages[0].header.stamp.nanosec,
        ) == (60, 100)
        audit = timing.messages[0]
        assert audit.has_source_stamp is True
        assert (audit.source_sec, audit.source_nanosec) == (60, 1_000_000_000)
        assert audit.source_rejected is True


def test_ego_status_uses_receipt_header_and_preserves_source_in_full_audit():
    with _running(
        lambda: AdMoraiBridge(parameter_overrides=_overrides())
    ) as bridge:
        status = _CapturePublisher()
        full = _CapturePublisher()
        timing = _CapturePublisher()
        bridge._stream_publishers["competition_status"] = status
        bridge._status_full_pub = full
        bridge._timing_pub = timing
        bridge._arrival_stamp = lambda _arrived: (12, 100)

        bridge._decode_and_publish("competition_status", _ego_packet(), 1.0)

        assert len(status.messages) == 1
        assert len(full.messages) == 1
        assert len(timing.messages) == 1
        message = status.messages[0]
        assert (message.header.stamp.sec, message.header.stamp.nanosec) == (12, 100)
        assert (
            full.messages[0].header.stamp.sec,
            full.messages[0].header.stamp.nanosec,
        ) == (12, 100)
        assert message.has_device_stamp is True
        assert (message.device_stamp.sec, message.device_stamp.nanosec) == (12, 34)
        assert timing.messages[0].source_selected is False


def test_status_and_camera_source_stamp_repeats_remain_one_for_one():
    with _running(
        lambda: AdMoraiBridge(parameter_overrides=_overrides())
    ) as bridge:
        timing = _CapturePublisher()
        status = _CapturePublisher()
        status_full = _CapturePublisher()
        camera = _CapturePublisher()
        bridge._timing_pub = timing
        bridge._stream_publishers["competition_status"] = status
        bridge._status_full_pub = status_full
        bridge._stream_publishers["camera_front"] = camera
        arrivals = {
            1.0: (70, 100),
            2.0: (70, 200),
            3.0: (80, 100),
            4.0: (80, 200),
        }
        bridge._arrival_stamp = lambda received: arrivals[received]

        bridge._decode_and_publish(
            "competition_status", _ego_packet(stamp=(70, 10)), 1.0
        )
        bridge._decode_and_publish(
            "competition_status", _ego_packet(stamp=(70, 10)), 2.0
        )
        bridge._decode_and_publish(
            "camera_front", _camera_packet(stamp=(80, 10)), 3.0
        )
        bridge._decode_and_publish(
            "camera_front", _camera_packet(stamp=(80, 10)), 4.0
        )

        assert len(status_full.messages) == 2
        assert len(status.messages) == 2
        assert len(camera.messages) == 2
        assert len(timing.messages) == 4
        assert timing.messages[1].duplicate is True
        assert timing.messages[3].duplicate is True
        assert timing.messages[1].normalized_published is True
        assert timing.messages[3].normalized_published is True
        assert bridge._counters["competition_status"].duplicates == 1
        assert bridge._counters["camera_front"].duplicates == 1


def test_camera_arrival_fallback_uses_first_fragment_receipt_not_completion():
    jpeg = b"\xff\xd8first-receipt\xff\xd9"
    split = 7
    with _running(
        lambda: AdMoraiBridge(parameter_overrides=_overrides())
    ) as bridge:
        camera = _CapturePublisher()
        timing = _CapturePublisher()
        bridge._stream_publishers["camera_front"] = camera
        bridge._timing_pub = timing
        bridge._arrival_stamp = lambda arrived: (
            100 + int(arrived),
            int(round((arrived - int(arrived)) * 1_000_000_000)),
        )

        bridge._decode_and_publish(
            "camera_front",
            _camera_fragment(1, jpeg[split:], True, stamp=(0, 0)),
            10.0,
        )
        bridge._decode_and_publish(
            "camera_front",
            _camera_fragment(0, jpeg[:split], False, stamp=(0, 0)),
            10.2,
        )

        assert len(camera.messages) == 1
        assert (
            camera.messages[0].header.stamp.sec,
            camera.messages[0].header.stamp.nanosec,
        ) == (110, 0)
        assert (
            timing.messages[0].header.stamp.sec,
            timing.messages[0].header.stamp.nanosec,
        ) == (110, 0)
        assert timing.messages[0].source_rejected is True


def test_ego_status_header_preserves_socket_receipt_time_across_queue_backlog():
    with _running(
        lambda: AdMoraiBridge(parameter_overrides=_overrides())
    ) as bridge:
        status = _CapturePublisher()
        bridge._stream_publishers["competition_status"] = status
        received_monotonic = time.monotonic() - 0.25
        before = bridge.get_clock().now().nanoseconds

        bridge._decode_and_publish(
            "competition_status", _ego_packet(), received_monotonic
        )

        after = bridge.get_clock().now().nanoseconds
        message = status.messages[0]
        header_ns = (
            message.header.stamp.sec * 1_000_000_000
            + message.header.stamp.nanosec
        )
        assert before - 300_000_000 <= header_ns
        assert header_ns <= after - 200_000_000


def test_control_enabled_creates_only_legal_reliable_subscription():
    with _running(
        lambda: AdMoraiBridge(
            parameter_overrides=_overrides(control_enabled=True)
        )
    ) as bridge:
        endpoints = bridge.get_subscriptions_info_by_topic("/ad/control/command")
        assert len(endpoints) == 1
        assert endpoints[0].topic_type == "ad_morai_interfaces/msg/CtrlCmd"
        assert endpoints[0].qos_profile.reliability == ReliabilityPolicy.RELIABLE
        assert bridge._control_subscription.qos_profile.depth == 10

        own_ad_subscriptions = {
            topic: [endpoint.topic_type for endpoint in topic_endpoints]
            for topic, _ in bridge.get_topic_names_and_types()
            if topic.startswith("/ad/")
            if (
                topic_endpoints := [
                    endpoint
                    for endpoint in bridge.get_subscriptions_info_by_topic(topic)
                    if endpoint.node_name == bridge.get_name()
                    and endpoint.node_namespace == bridge.get_namespace()
                ]
            )
        }
        assert own_ad_subscriptions == {
            "/ad/control/command": ["ad_morai_interfaces/msg/CtrlCmd"]
        }


def test_bounded_queue_counts_overflow_without_blocking_receiver():
    overrides = _overrides(
        extra=(Parameter("queue_capacity", value=1),)
    )
    with _running(lambda: AdMoraiBridge(parameter_overrides=overrides)) as bridge:
        bridge._enqueue("competition_status", b"first", 1.0)
        bridge._enqueue("competition_status", b"second", 2.0)

        counter = bridge._counters["competition_status"]
        assert counter.packets == 2
        assert counter.bytes == 11
        assert counter.dropped == 1
        assert bridge._queue.qsize() == 1


def test_drain_contains_unexpected_per_packet_exception_and_continues():
    with _running(
        lambda: AdMoraiBridge(parameter_overrides=_overrides())
    ) as bridge:
        handled = []

        def decode(stream, packet, arrived):
            if packet == b"bad":
                raise AssertionError("bad stamp")
            handled.append((stream, packet, arrived))

        bridge._decode_and_publish = decode
        bridge._queue.put_nowait(("competition_status", b"bad", 1.0))
        bridge._queue.put_nowait(("competition_status", b"good", 2.0))

        bridge._drain_queue()

        assert bridge._counters["competition_status"].malformed == 1
        assert handled == [("competition_status", b"good", 2.0)]


def test_diagnostic_state_prioritizes_bind_dead_and_stale_conditions():
    bind_failed = StreamCounters(bind_errors=1, packets=2, malformed=1)
    assert diagnostic_state(True, bind_failed, 5.0, False, 1.0)[1] == "UDP bind failed"

    dead = StreamCounters(packets=2, malformed=1)
    assert diagnostic_state(True, dead, 5.0, False, 1.0)[1] == "UDP receiver stopped"

    stale = StreamCounters(packets=2, malformed=1)
    assert diagnostic_state(True, stale, 5.0, True, 1.0)[1] == "UDP stream stale"

    assert diagnostic_state(True, StreamCounters(), -1.0, True, 1.0)[1] == "waiting for UDP"
    assert diagnostic_state(False, bind_failed, 5.0, False, 1.0)[1] == "disabled"


def test_receiver_bind_error_is_contained_and_counted(monkeypatch):
    from ad_morai_bridge import morai_bridge_node as node_module

    class BindFailure:
        def __init__(self, *_args, **_kwargs):
            raise OSError("address unavailable")

    monkeypatch.setattr(node_module, "UdpReceiver", BindFailure)
    overrides = _overrides(
        extra=(Parameter("competition_status.enabled", value=True),)
    )
    with _running(lambda: AdMoraiBridge(parameter_overrides=overrides)) as bridge:
        assert bridge._counters["competition_status"].bind_errors == 1
        assert "competition_status" not in bridge._receivers


def test_control_send_oserror_is_contained():
    class FailingSender:
        def send(self, _packet):
            raise OSError("simulated send failure")

    with _running(
        lambda: AdMoraiBridge(parameter_overrides=_overrides())
    ) as bridge:
        bridge._control_sender = FailingSender()
        bridge._send_control_record(CtrlCommandRecord(brake=1.0))
        bridge._send_control_record(CtrlCommandRecord(brake=1.0))

        assert bridge._control_send_errors == 2


def test_control_message_preserves_every_type_one_wire_field():
    accepted = []

    class Gate:
        def accept(self, command, now):
            accepted.append((command, now))

        def emergency_stop(self):
            return False

    with _running(
        lambda: AdMoraiBridge(parameter_overrides=_overrides())
    ) as bridge:
        bridge._control_gate = Gate()
        message = CtrlCmd()
        message.ctrl_mode = CtrlCmd.CTRL_MODE_AUTO
        message.gear = CtrlCmd.GEAR_DRIVE
        message.long_cmd_type = CtrlCmd.LONG_CMD_THROTTLE
        message.velocity = 12.5
        message.acceleration = 1.25
        message.accel = 0.25
        message.brake = 0.0
        message.steering = -0.1

        bridge._on_control(message)

        assert accepted[0][0] == CtrlCommandRecord(
            ctrl_mode=2,
            gear=4,
            long_cmd_type=1,
            velocity=12.5,
            acceleration=1.25,
            accel=0.25,
            brake=0.0,
            steering=-0.1,
        )


def test_destroy_sends_exact_armed_stop_burst_before_closing_resources():
    events = []

    class Sender:
        def send(self, _packet):
            events.append("send")

        def close(self):
            events.append("sender_close")

    class Receiver:
        def close(self):
            events.append("receiver_close")

    rclpy.init()
    bridge = AdMoraiBridge(parameter_overrides=_overrides())
    sender = Sender()
    bridge._control_sender = sender
    bridge._senders = [sender]
    bridge._receivers = {"competition_status": Receiver()}
    bridge._control_gate = ControlSafetyGate(bridge._send_control_record)
    bridge._control_gate.accept(CtrlCommandRecord(accel=0.1), now=1.0)
    events.clear()

    try:
        bridge.destroy_node()
        bridge = None
    finally:
        if bridge is not None:
            bridge.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    assert events == ["send", "send", "send", "receiver_close", "sender_close"]


def test_arrival_compatibility_mode_never_promotes_a_valid_source_stamp():
    overrides = _overrides(
        extra=(Parameter("timestamp_mode", value="arrival"),)
    )
    with _running(lambda: AdMoraiBridge(parameter_overrides=overrides)) as bridge:
        decision = bridge._timestamp_decision("imu", (10, 10), (10, 20))

        assert decision.selected_stamp == (10, 20)
        assert decision.source_selected is False
        assert decision.arrival_fallback is True


def test_retired_device_when_available_mode_is_rejected_at_construction():
    overrides = _overrides(
        extra=(Parameter("timestamp_mode", value="device_when_available"),)
    )

    rclpy.init()
    try:
        with pytest.raises(ValueError, match="source_preferred.*arrival"):
            AdMoraiBridge(parameter_overrides=overrides)
    finally:
        if rclpy.ok():
            rclpy.shutdown()


def test_receipt_timestamp_bridge_rejects_use_sim_time():
    overrides = _overrides(
        extra=(Parameter("use_sim_time", value=True),)
    )

    rclpy.init()
    try:
        with pytest.raises(ValueError, match="use_sim_time.*receipt"):
            AdMoraiBridge(parameter_overrides=overrides)
    finally:
        if rclpy.ok():
            rclpy.shutdown()


def test_invalid_stream_config_is_rejected_at_construction():
    overrides = _overrides(
        extra=(Parameter("velodyne.stale_after_sec", value=0.0),)
    )

    rclpy.init()
    try:
        with pytest.raises(ValueError, match="velodyne.stale_after_sec"):
            AdMoraiBridge(parameter_overrides=overrides)
    finally:
        if rclpy.ok():
            rclpy.shutdown()


def test_invalid_control_watchdog_is_rejected_at_construction():
    overrides = _overrides(
        control_enabled=True,
        extra=(Parameter("control.watchdog_timeout_sec", value=0.0),),
    )

    rclpy.init()
    try:
        with pytest.raises(ValueError, match="timeout_sec"):
            AdMoraiBridge(parameter_overrides=overrides)
    finally:
        if rclpy.ok():
            rclpy.shutdown()


def test_destroy_cleanup_is_best_effort_and_idempotent():
    events = []

    class Gate:
        def emergency_stop(self):
            events.append("emergency")
            raise RuntimeError("simulated stop failure")

    class Resource:
        def __init__(self, name, fail=False):
            self.name = name
            self.fail = fail

        def close(self):
            events.append(self.name)
            if self.fail:
                raise RuntimeError("simulated close failure")

    rclpy.init()
    bridge = AdMoraiBridge(parameter_overrides=_overrides())
    bridge._control_gate = Gate()
    bridge._receivers = {
        "competition_status": Resource("receiver_fail", fail=True),
        "collisions": Resource("receiver_ok"),
    }
    bridge._senders = [Resource("sender_ok")]

    try:
        bridge.destroy_node()
        bridge.destroy_node()
        bridge = None
    finally:
        if bridge is not None:
            try:
                bridge.destroy_node()
            except RuntimeError:
                pass
        if rclpy.ok():
            rclpy.shutdown()

    assert events == ["emergency", "receiver_fail", "receiver_ok", "sender_ok"]
