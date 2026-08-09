import queue
import struct
import threading
from pathlib import Path

import pytest
import yaml


def _ego_packet(*, sec: int = 12, nanosec: int = 34) -> bytes:
    payload = struct.pack(
        "<ii2bfi2f3f3f3f3f3f3f3ff38s",
        sec,
        nanosec,
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
    return (
        b"#MoraiInfo$"
        + struct.pack("<i3i", len(payload), 0, 0, 0)
        + payload
        + b"\r\n"
    )


def _object_packet(*, sec: int = 10, nanosec: int = 20) -> bytes:
    item = struct.pack(
        "<hh16f38s",
        42,
        1,
        1.0,
        2.0,
        3.0,
        90.0,
        4.0,
        2.0,
        1.5,
        0.8,
        2.7,
        0.9,
        12.0,
        0.5,
        0.0,
        0.1,
        0.2,
        0.0,
        b"LINK_42",
    )
    empty = struct.pack("<hh16f38s", 0, -1, *(0.0,) * 16, b"")
    payload = struct.pack("<ii", sec, nanosec) + item + empty * 19
    return (
        b"#MoraiObjInfo$"
        + struct.pack("<i3i", len(payload), 0, 0, 0)
        + payload
        + b"\r\n"
    )


def _lidar_packet() -> bytes:
    ranges = b"".join(
        struct.pack("<HB", 1000 + index, index % 100)
        for index in range(360)
    )
    return (
        b"#Lidar2D$"
        + struct.pack("<i3f", len(ranges), 1.0, 2.0, 3.0)
        + ranges
        + b"\r\n"
    )


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class _ReceiptClock:
    def __init__(self, *stamps):
        self._stamps = iter(stamps)

    def stamp(self, _arrived):
        return next(self._stamps)


def test_development_profile_defaults_to_arrival_policy():
    config_path = (
        Path(__file__).resolve().parents[1] / "config" / "development.yaml"
    )
    parameters = yaml.safe_load(config_path.read_text(encoding="utf-8"))["/**"][
        "ros__parameters"
    ]

    assert parameters["timestamp_mode"] == "arrival"
    assert parameters["source_stamp_tolerance_sec"] == 1.0


def _bridge_for_stream(
    stream,
    mode,
    *arrival_stamps,
    timestamp_mode="arrival",
    has_full=False,
):
    from ad_morai_bridge.timestamp_policy import TimestampPolicy
    from ad_morai_bridge_dev.bridge.node import (
        AdMoraiDevBridge,
        StreamCounters,
    )

    bridge = object.__new__(AdMoraiDevBridge)
    bridge._stream_configs = {
        stream: {
            "frame_id": "lidar2d_link" if mode == "lidar2d" else "map",
            "mode": mode,
        }
    }
    bridge._stream_publishers = {stream: _Publisher()}
    bridge._full_publishers = {stream: _Publisher()} if has_full else {}
    bridge._timing_pub = _Publisher()
    bridge._receipt_clock = _ReceiptClock(*arrival_stamps)
    bridge._counters = {stream: StreamCounters()}
    bridge._lock = threading.Lock()
    bridge._timestamp_policies = {
        stream: TimestampPolicy(
            mode=timestamp_mode,
            tolerance_sec=1.0,
            suppress_source_duplicates=False,
        )
    }
    return bridge


def _bridge_for_ego(*arrival_stamps, timestamp_mode="arrival"):
    return _bridge_for_stream(
        "ego_status",
        "ego",
        *arrival_stamps,
        timestamp_mode=timestamp_mode,
        has_full=True,
    )


def test_dev_status_uses_arrival_and_audits_valid_source():
    bridge = _bridge_for_ego((12, 500_000_000))

    bridge._decode_and_publish("ego_status", _ego_packet(), arrived=1.0)

    normalized = bridge._stream_publishers["ego_status"].messages
    assert len(normalized) == 1
    assert (normalized[0].header.stamp.sec, normalized[0].header.stamp.nanosec) == (
        12,
        500_000_000,
    )
    assert (normalized[0].device_stamp.sec, normalized[0].device_stamp.nanosec) == (
        12,
        34,
    )
    assert len(bridge._timing_pub.messages) == 1
    timing = bridge._timing_pub.messages[0]
    assert (timing.header.stamp.sec, timing.header.stamp.nanosec) == (
        12,
        500_000_000,
    )
    assert timing.stream == "dev/ego_status"
    assert timing.source_valid is True
    assert timing.source_selected is False
    assert timing.normalized_published is True


def test_dev_status_duplicate_is_audited_and_republished_with_arrival_stamp():
    bridge = _bridge_for_ego(
        (12, 500_000_000),
        (12, 600_000_000),
    )

    bridge._decode_and_publish("ego_status", _ego_packet(), arrived=1.0)
    bridge._decode_and_publish("ego_status", _ego_packet(), arrived=2.0)

    normalized = bridge._stream_publishers["ego_status"].messages
    assert len(normalized) == 2
    assert (
        normalized[-1].header.stamp.sec,
        normalized[-1].header.stamp.nanosec,
    ) == (12, 600_000_000)
    assert len(bridge._full_publishers["ego_status"].messages) == 2
    assert len(bridge._timing_pub.messages) == 2
    duplicate = bridge._timing_pub.messages[-1]
    assert duplicate.duplicate is True
    assert duplicate.normalized_published is True
    assert bridge._counters["ego_status"].duplicates == 1


def test_dev_status_rejects_cross_domain_source_and_falls_back_to_arrival():
    bridge = _bridge_for_ego((100, 200))

    bridge._decode_and_publish("ego_status", _ego_packet(), arrived=1.0)

    message = bridge._stream_publishers["ego_status"].messages[0]
    assert (message.header.stamp.sec, message.header.stamp.nanosec) == (100, 200)
    timing = bridge._timing_pub.messages[0]
    assert timing.source_rejected is True
    assert timing.source_selected is False
    assert bridge._counters["ego_status"].arrival_fallback == 1


def test_dev_status_invalid_raw_source_is_audited_without_constructing_bad_time():
    bridge = _bridge_for_ego((12, 500_000_000))

    bridge._decode_and_publish(
        "ego_status",
        _ego_packet(nanosec=1_000_000_000),
        arrived=1.0,
    )

    normalized = bridge._stream_publishers["ego_status"].messages[0]
    assert (normalized.header.stamp.sec, normalized.header.stamp.nanosec) == (
        12,
        500_000_000,
    )
    assert normalized.has_device_stamp is False
    full = bridge._full_publishers["ego_status"].messages[0]
    assert (full.header.stamp.sec, full.header.stamp.nanosec) == (
        12,
        500_000_000,
    )
    timing = bridge._timing_pub.messages[0]
    assert timing.has_source_stamp is True
    assert (timing.source_sec, timing.source_nanosec) == (12, 1_000_000_000)
    assert timing.source_rejected is True


def test_dev_objects_publish_valid_source_stamp_and_account_for_the_decision():
    bridge = _bridge_for_stream(
        "objects",
        "objects",
        (10, 100),
        timestamp_mode="source_preferred",
    )

    bridge._decode_and_publish("objects", _object_packet(), arrived=1.0)

    normalized = bridge._stream_publishers["objects"].messages
    assert len(normalized) == 1
    assert (
        normalized[0].header.stamp.sec,
        normalized[0].header.stamp.nanosec,
    ) == (10, 20)
    timing = bridge._timing_pub.messages[0]
    assert timing.stream == "dev/objects"
    assert timing.source_selected is True
    assert bridge._counters["objects"].source_selected == 1
    assert bridge._counters["objects"].arrival_fallback == 0


def test_dev_source_less_lidar_uses_arrival_for_full_normalized_and_audit():
    bridge = _bridge_for_stream(
        "lidar2d", "lidar2d", (30, 40), has_full=True
    )

    bridge._decode_and_publish("lidar2d", _lidar_packet(), arrived=1.0)

    normalized = bridge._stream_publishers["lidar2d"].messages[0]
    full = bridge._full_publishers["lidar2d"].messages[0]
    assert (normalized.header.stamp.sec, normalized.header.stamp.nanosec) == (
        30,
        40,
    )
    assert (full.header.stamp.sec, full.header.stamp.nanosec) == (30, 40)
    timing = bridge._timing_pub.messages[0]
    assert timing.has_source_stamp is False
    assert timing.source_selected is False
    assert (
        timing.selected_stamp.sec,
        timing.selected_stamp.nanosec,
    ) == (30, 40)
    assert bridge._counters["lidar2d"].arrival_fallback == 1


def test_dev_arrival_mode_never_promotes_a_valid_ego_source_stamp():
    bridge = _bridge_for_ego(
        (12, 500_000_000), timestamp_mode="arrival"
    )

    bridge._decode_and_publish("ego_status", _ego_packet(), arrived=1.0)

    message = bridge._stream_publishers["ego_status"].messages[0]
    assert (message.header.stamp.sec, message.header.stamp.nanosec) == (
        12,
        500_000_000,
    )
    timing = bridge._timing_pub.messages[0]
    assert timing.source_valid is True
    assert timing.source_selected is False
    assert (timing.selected_stamp.sec, timing.selected_stamp.nanosec) == (
        12,
        500_000_000,
    )
    assert bridge._counters["ego_status"].source_selected == 0
    assert bridge._counters["ego_status"].arrival_fallback == 1


def test_dev_object_stamp_regression_is_audited_counted_and_not_published():
    bridge = _bridge_for_stream(
        "objects",
        "objects",
        (400, 0),
        (400, 100_000_000),
        timestamp_mode="source_preferred",
    )

    bridge._decode_and_publish(
        "objects", _object_packet(sec=0, nanosec=-1), arrived=1.0
    )
    bridge._decode_and_publish(
        "objects",
        _object_packet(sec=399, nanosec=900_000_000),
        arrived=2.0,
    )

    assert len(bridge._stream_publishers["objects"].messages) == 1
    regression = bridge._timing_pub.messages[-1]
    assert regression.source_selected is True
    assert regression.stamp_regression is True
    assert regression.normalized_published is False
    counter = bridge._counters["objects"]
    assert counter.source_rejected == 1
    assert counter.arrival_fallback == 1
    assert counter.source_selected == 1
    assert counter.stamp_regressions == 1
    assert counter.dropped == 1


def test_udp_counter_update_waits_for_the_statistics_snapshot_lock():
    from ad_morai_bridge_dev.bridge.node import (
        AdMoraiDevBridge,
        StreamCounters,
    )

    bridge = object.__new__(AdMoraiDevBridge)
    bridge._lock = threading.Lock()
    bridge._counters = {"objects": StreamCounters()}
    bridge._queue = queue.Queue()
    started = threading.Event()
    completed = threading.Event()

    def enqueue():
        started.set()
        bridge._enqueue("objects", b"payload", 12.5)
        completed.set()

    bridge._lock.acquire()
    worker = threading.Thread(target=enqueue)
    try:
        worker.start()
        assert started.wait(1.0)
        was_blocked = not completed.wait(0.1)
    finally:
        bridge._lock.release()
    worker.join(timeout=1.0)

    assert was_blocked is True
    assert completed.is_set()
    assert bridge._counters["objects"].packets == 1


def test_statistics_publish_from_one_atomic_counter_snapshot(monkeypatch):
    from builtin_interfaces.msg import Time
    import ad_morai_bridge_dev.bridge.node as morai_bridge_dev_node
    from ad_morai_bridge_dev.bridge.node import (
        AdMoraiDevBridge,
        StreamCounters,
    )

    bridge = object.__new__(AdMoraiDevBridge)
    bridge._lock = threading.Lock()
    bridge._started_at = 1.0
    bridge._counters = {
        "first": StreamCounters(
            packets=1, bytes=10, first_at=10.0, last_at=18.0
        ),
        "second": StreamCounters(
            packets=2,
            bytes=20,
            malformed=3,
            dropped=4,
            source_selected=5,
            arrival_fallback=6,
            source_rejected=7,
            duplicates=8,
            stamp_regressions=9,
            first_at=10.0,
            last_at=19.0,
        ),
    }
    bridge.get_clock = lambda: type(
        "Clock",
        (),
        {
            "now": lambda self: type(
                "Now", (), {"to_msg": lambda self: Time(sec=50, nanosec=60)}
            )()
        },
    )()
    begin_mutation = threading.Event()
    mutation_done = threading.Event()

    def mutate_second_counter():
        assert begin_mutation.wait(1.0)
        with bridge._lock:
            bridge._counters["second"].packets = 999
            bridge._counters["second"].source_selected = 999
        mutation_done.set()

    class PublishingProbe(_Publisher):
        def publish(self, message):
            super().publish(message)
            if len(self.messages) == 1:
                begin_mutation.set()
                assert mutation_done.wait(1.0)

    bridge._stats_pub = PublishingProbe()
    monkeypatch.setattr(morai_bridge_dev_node.time, "monotonic", lambda: 20.0)
    worker = threading.Thread(target=mutate_second_counter)
    worker.start()
    bridge._publish_statistics()
    worker.join(timeout=1.0)

    assert mutation_done.is_set()
    second = bridge._stats_pub.messages[1]
    assert second.stream == "second"
    assert second.packets == 2
    assert second.bytes == 20
    assert second.malformed == 3
    assert second.dropped == 4
    assert second.source_selected == 5
    assert second.arrival_fallback == 6
    assert second.source_rejected == 7
    assert second.duplicates == 8
    assert second.stamp_regressions == 9
    assert second.packet_rate_hz == pytest.approx(0.2)
    assert second.byte_rate_bps == pytest.approx(2.0)
    assert second.last_packet_age_sec == pytest.approx(1.0)


def test_initialized_dev_statistics_include_all_camera_bbox_streams():
    import rclpy
    from rclpy.parameter import Parameter

    from ad_morai_bridge_dev.bridge.node import (
        AdMoraiDevBridge,
        OUTPUTS,
        STREAMS,
    )

    overrides = [
        Parameter("grpc.enabled", value=False),
        Parameter("camera_bboxes.enabled", value=False),
        *(Parameter(f"{name}.enabled", value=False) for name in STREAMS),
        *(Parameter(f"{name}.enabled", value=False) for name in OUTPUTS),
    ]
    bridge = None
    rclpy.init()
    try:
        bridge = AdMoraiDevBridge(parameter_overrides=overrides)
        bridge._stats_pub = _Publisher()

        assert all(
            policy._mode == "arrival"
            and policy._suppress_source_duplicates is False
            for policy in bridge._timestamp_policies.values()
        )
        assert all(
            policy._mode == "arrival"
            and policy._suppress_source_duplicates is False
            for policy in bridge._camera_timestamp_policies.values()
        )

        bridge._publish_statistics()

        published_streams = {
            message.stream for message in bridge._stats_pub.messages
        }
        assert {
            "camera_bbox_front",
            "camera_bbox_left",
            "camera_bbox_right",
            "camera_bbox_traffic_light",
        }.issubset(published_streams)
    finally:
        if bridge is not None:
            bridge.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_dev_receipt_timestamp_bridge_rejects_use_sim_time():
    import rclpy
    from rclpy.parameter import Parameter

    from ad_morai_bridge_dev.bridge.node import (
        AdMoraiDevBridge,
        OUTPUTS,
        STREAMS,
    )

    overrides = [
        Parameter("use_sim_time", value=True),
        Parameter("grpc.enabled", value=False),
        Parameter("camera_bboxes.enabled", value=False),
        *(Parameter(f"{name}.enabled", value=False) for name in STREAMS),
        *(Parameter(f"{name}.enabled", value=False) for name in OUTPUTS),
    ]
    rclpy.init()
    try:
        with pytest.raises(ValueError, match="use_sim_time.*receipt"):
            AdMoraiDevBridge(parameter_overrides=overrides)
    finally:
        if rclpy.ok():
            rclpy.shutdown()
