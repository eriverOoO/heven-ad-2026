import struct
import threading


CAMERA_ROLES = ("front", "left", "right", "traffic_light")
CAMERA_COUNTER_STREAMS = {
    role: f"camera_bbox_{role}" for role in CAMERA_ROLES
}


class _MessageSink:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def _install_timestamp_contract(bridge, *streams, camera_roles=()):
    from ad_morai_bridge.timestamp_policy import TimestampPolicy
    from ad_morai_bridge_dev.bridge.node import StreamCounters

    bridge._timestamp_policies = {
        stream: TimestampPolicy(
            mode="arrival",
            tolerance_sec=1.0,
            suppress_source_duplicates=False,
        )
        for stream in streams
    }
    bridge._camera_timestamp_policies = {
        role: TimestampPolicy(
            mode="arrival",
            tolerance_sec=1.0,
            suppress_source_duplicates=False,
        )
        for role in camera_roles
    }
    counter_streams = (
        *streams,
        *(CAMERA_COUNTER_STREAMS[role] for role in camera_roles),
    )
    bridge._counters = {
        stream: StreamCounters() for stream in counter_streams
    }
    bridge._lock = threading.Lock()
    bridge._timing_pub = _MessageSink()


def _ego_packet() -> bytes:
    payload = struct.pack(
        "<ii2bfi2f3f3f3f3f3f3f3ff38s",
        12,
        34,
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


def test_camera_box_raw_packets_route_to_only_the_matching_dev_topic():
    from ad_morai_interfaces.msg import RawPacket
    from ad_morai_bridge_dev.bridge.node import AdMoraiDevBridge

    published = {role: [] for role in CAMERA_ROLES}
    bridge = object.__new__(AdMoraiDevBridge)
    bridge._camera_bbox_publishers = {
        role: type(
            "Publisher",
            (),
            {"publish": lambda self, message, key=role: published[key].append(message)},
        )()
        for role in CAMERA_ROLES
    }
    bridge.get_logger = lambda: type(
        "Logger", (), {"warning": lambda self, _message: None}
    )()
    _install_timestamp_contract(bridge, camera_roles=CAMERA_ROLES)

    values = tuple(float(index) + 0.25 for index in range(28))
    packet = (
        b"BOX"
        + struct.pack("<iiii", 4, 5, 0, 1)
        + struct.pack("<28fBBB", *values, 7, 8, 9)
        + b"\r\n"
    )
    message = RawPacket()
    message.header.stamp.sec = 4
    message.header.stamp.nanosec = 100
    message.stream = "camera_traffic_light"
    message.data = packet

    bridge._on_camera_raw("traffic_light", "camera_traffic_light", message)

    assert {role: len(items) for role, items in published.items()} == {
        "front": 0,
        "left": 0,
        "right": 0,
        "traffic_light": 1,
    }
    assert published["traffic_light"][0].header.frame_id == (
        "camera_traffic_light_optical_frame"
    )
    assert (
        published["traffic_light"][0].header.stamp.sec,
        published["traffic_light"][0].header.stamp.nanosec,
    ) == (4, 100)
    timing = bridge._timing_pub.messages[-1]
    assert timing.stream == "dev/camera/traffic_light"
    assert (timing.header.stamp.sec, timing.header.stamp.nanosec) == (4, 100)
    assert timing.source_valid is True
    assert timing.source_selected is False

    bridge._on_camera_raw("traffic_light", "camera_traffic_light", message)
    assert len(published["traffic_light"]) == 2
    assert bridge._timing_pub.messages[-1].duplicate is True
    assert bridge._timing_pub.messages[-1].normalized_published is True

    invalid_source = bytearray(packet)
    struct.pack_into("<i", invalid_source, 3 + 4, 1_000_000_000)
    message.header.stamp.nanosec = 200
    message.data = bytes(invalid_source)
    bridge._on_camera_raw("traffic_light", "camera_traffic_light", message)
    assert len(published["traffic_light"]) == 3
    assert (
        published["traffic_light"][-1].header.stamp.sec,
        published["traffic_light"][-1].header.stamp.nanosec,
    ) == (4, 200)
    assert bridge._timing_pub.messages[-1].source_rejected is True
    assert bridge._timing_pub.messages[-1].source_selected is False

    message.data = b"MOR" + b"jpeg-fragment"
    bridge._on_camera_raw("traffic_light", "camera_traffic_light", message)
    assert len(published["traffic_light"]) == 3

    malformed = b"BOXtruncated"
    message.data = malformed
    bridge._on_camera_raw("traffic_light", "camera_traffic_light", message)
    assert len(published["traffic_light"]) == 3

    message.stream = "camera_right"
    message.data = packet
    bridge._on_camera_raw("traffic_light", "camera_traffic_light", message)
    assert len(published["traffic_light"]) == 3

    counter = bridge._counters["camera_bbox_traffic_light"]
    assert counter.packets == 4
    assert counter.bytes == (
        len(packet) * 2 + len(invalid_source) + len(malformed)
    )
    assert counter.malformed == 1
    assert counter.source_selected == 0
    assert counter.arrival_fallback == 3
    assert counter.source_rejected == 1
    assert counter.duplicates == 1
    assert counter.stamp_regressions == 0
    assert counter.dropped == 0


def test_simulator_commands_select_one_existing_sender_and_send_once():
    from ad_morai_interfaces_dev.msg import SimulatorCommand
    from ad_morai_bridge_dev.bridge.node import AdMoraiDevBridge

    packets = []

    class Sender:
        def send(self, packet):
            packets.append(packet)

    bridge = object.__new__(AdMoraiDevBridge)
    bridge._senders = {"lamp_control": Sender()}
    bridge.get_logger = lambda: type(
        "Logger", (), {"error": lambda self, _message: None}
    )()

    unknown = SimulatorCommand()
    unknown.command_type = "not_configured"
    unknown.json_payload = "{}"
    bridge._on_simulator_command(unknown)
    assert packets == []

    command = SimulatorCommand()
    command.command_type = "lamp_control"
    command.json_payload = '{"turn_signal": 1, "emergency_signal": 0}'
    bridge._on_simulator_command(command)
    assert len(packets) == 1


def test_intersection_packet_does_not_invent_traffic_light_metadata():
    from builtin_interfaces.msg import Time
    from ad_morai_bridge_dev.bridge.node import AdMoraiDevBridge
    from ad_morai_bridge.codecs.common import encode_envelope

    published = []
    bridge = object.__new__(AdMoraiDevBridge)
    bridge._stream_configs = {
        "intersection_status": {"frame_id": "map", "mode": "intersection"}
    }
    bridge._stream_publishers = {
        "intersection_status": type(
            "Publisher", (), {"publish": lambda self, message: published.append(message)}
        )()
    }
    bridge.get_clock = lambda: type(
        "Clock",
        (),
        {
            "now": lambda self: type(
                "Now",
                (),
                {"to_msg": lambda self: Time(sec=1, nanosec=2)},
            )()
        },
    )()
    _install_timestamp_contract(bridge, "intersection_status")

    packet = encode_envelope(b"#IntStatus$", struct.pack("<hhf", 1, 16, 3.5))
    bridge._decode_and_publish("intersection_status", packet)

    assert len(published) == 1
    message = published[0]
    assert message.intersection_index == "1"
    assert list(message.traffic_light_indices) == []
    assert list(message.traffic_light_types) == []
    assert list(message.traffic_light_statuses) == [16]
    assert list(message.remaining_times) == [3.5]


def test_shared_ego_status_uses_dev_arrival_header_and_udp_device_stamp():
    from builtin_interfaces.msg import Time
    from ad_morai_bridge_dev.bridge.node import AdMoraiDevBridge

    shared = []
    full = []
    bridge = object.__new__(AdMoraiDevBridge)
    bridge._stream_configs = {
        "ego_status": {"frame_id": "map", "mode": "ego"}
    }
    bridge._stream_publishers = {
        "ego_status": type(
            "Publisher", (), {"publish": lambda self, message: shared.append(message)}
        )()
    }
    bridge._full_publishers = {
        "ego_status": type(
            "Publisher", (), {"publish": lambda self, message: full.append(message)}
        )()
    }
    bridge.get_clock = lambda: type(
        "Clock",
        (),
        {
            "now": lambda self: type(
                "Now",
                (),
                {"to_msg": lambda self: Time(sec=100, nanosec=200)},
            )()
        },
    )()
    _install_timestamp_contract(bridge, "ego_status")

    bridge._decode_and_publish("ego_status", _ego_packet())

    assert len(shared) == 1
    message = shared[0]
    assert (message.header.stamp.sec, message.header.stamp.nanosec) == (100, 200)
    assert message.has_device_stamp is True
    assert (message.device_stamp.sec, message.device_stamp.nanosec) == (12, 34)
    assert len(full) == 1


def test_constructor_closes_partial_transports_and_base_node_on_initialize_error(
    monkeypatch,
):
    import rclpy
    from rclpy.node import Node

    from ad_morai_bridge_dev.bridge.node import AdMoraiDevBridge

    closed = []
    base_destroyed = []

    class Resource:
        def __init__(self, name):
            self.name = name

        def close(self):
            closed.append(self.name)

    def fail_initialize(bridge):
        bridge._receivers["receiver"] = Resource("receiver")
        bridge._senders["sender"] = Resource("sender")
        raise RuntimeError("initialize failed")

    original_destroy_node = Node.destroy_node

    def record_base_destroy(node):
        base_destroyed.append(node)
        return original_destroy_node(node)

    monkeypatch.setattr(AdMoraiDevBridge, "_initialize", fail_initialize)
    monkeypatch.setattr(Node, "destroy_node", record_base_destroy)

    rclpy.init()
    try:
        try:
            AdMoraiDevBridge()
        except RuntimeError as exc:
            assert str(exc) == "initialize failed"
        else:
            raise AssertionError("constructor accepted failed initialization")
    finally:
        if rclpy.ok():
            rclpy.shutdown()

    assert closed == ["receiver", "sender"]
    assert len(base_destroyed) == 1


def test_close_transports_continues_after_one_oserror():
    from ad_morai_bridge_dev.bridge.node import AdMoraiDevBridge

    closed = []

    class Resource:
        def __init__(self, name, fails=False):
            self.name = name
            self.fails = fails

        def close(self):
            closed.append(self.name)
            if self.fails:
                raise OSError("close failed")

    bridge = object.__new__(AdMoraiDevBridge)
    bridge._closed = False
    bridge._receivers = {
        "bad": Resource("bad receiver", fails=True),
        "good": Resource("good receiver"),
    }
    bridge._senders = {
        "first": Resource("first sender"),
        "second": Resource("second sender"),
    }
    bridge.get_logger = lambda: type(
        "Logger", (), {"error": lambda self, _message: None}
    )()

    bridge._close_transports()

    assert closed == [
        "bad receiver",
        "good receiver",
        "first sender",
        "second sender",
    ]
    assert bridge._receivers == {}
    assert bridge._senders == {}
