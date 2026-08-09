from array import array
from contextlib import contextmanager
from importlib import import_module

import pytest
import rclpy
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField


def boundary_module():
    return import_module("ad_morai_bridge.point_time_zero_node")


def field(name, offset, datatype, count=1):
    result = PointField()
    result.name = name
    result.offset = offset
    result.datatype = datatype
    result.count = count
    return result


def cloud(*, width=2, height=2, point_step=16, row_step=40):
    message = PointCloud2()
    message.header.stamp.sec = 123
    message.header.stamp.nanosec = 456
    message.header.frame_id = "lidar_link"
    message.height = height
    message.width = width
    message.fields = [
        field("x", 0, PointField.FLOAT32),
        field("intensity", 4, PointField.FLOAT32),
        field("time", 8, PointField.FLOAT32),
        field("ring", 12, PointField.UINT16),
    ]
    message.is_bigendian = False
    message.point_step = point_step
    message.row_step = row_step
    message.data = array(
        "B", ((index * 37 + 11) % 256 for index in range(row_step * height))
    )
    message.is_dense = True
    return message


def metadata(message):
    return (
        message.header.stamp.sec,
        message.header.stamp.nanosec,
        message.header.frame_id,
        message.height,
        message.width,
        tuple(
            (item.name, item.offset, item.datatype, item.count)
            for item in message.fields
        ),
        message.is_bigendian,
        message.point_step,
        message.row_step,
        message.is_dense,
    )


def time_byte_indices(message):
    return {
        row * message.row_step + point * message.point_step + 8 + byte
        for row in range(message.height)
        for point in range(message.width)
        for byte in range(4)
    }


def test_zero_boundary_preserves_schema_header_padding_and_non_time_bytes():
    module = boundary_module()
    message = cloud()
    before_metadata = metadata(message)
    before_data = bytes(message.data)

    result = module.zero_point_time_field(message)

    assert result is message
    assert metadata(message) == before_metadata
    zeroed = time_byte_indices(message)
    assert all(
        value == 0
        for index, value in enumerate(message.data)
        if index in zeroed
    )
    assert all(
        value == before_data[index]
        for index, value in enumerate(message.data)
        if index not in zeroed
    )
    # Both organized-cloud row paddings remain byte-for-byte unchanged.
    assert bytes(message.data[32:40]) == before_data[32:40]
    assert bytes(message.data[72:80]) == before_data[72:80]


def test_zero_boundary_is_endian_independent_and_accepts_empty_cloud():
    module = boundary_module()
    message = cloud(width=1, height=1, row_step=16)
    message.is_bigendian = True
    module.zero_point_time_field(message)
    assert bytes(message.data[8:12]) == b"\x00\x00\x00\x00"

    empty = cloud(width=0, height=1, row_step=0)
    empty.data = array("B")
    assert module.zero_point_time_field(empty) is empty
    assert bytes(empty.data) == b""


def malformed_cases():
    missing = cloud()
    missing.fields = [item for item in missing.fields if item.name != "time"]

    duplicate = cloud()
    duplicate.fields.append(field("time", 4, PointField.FLOAT32))

    wrong_datatype = cloud()
    wrong_datatype_time = next(
        item for item in wrong_datatype.fields if item.name == "time"
    )
    wrong_datatype_time.datatype = PointField.UINT32

    wrong_count = cloud()
    wrong_count_time = next(
        item for item in wrong_count.fields if item.name == "time"
    )
    wrong_count_time.count = 2

    outside_point = cloud()
    outside_time = next(
        item for item in outside_point.fields if item.name == "time"
    )
    outside_time.offset = 14

    zero_point_step = cloud()
    zero_point_step.point_step = 0

    short_row = cloud()
    short_row.row_step = 31
    short_row.data = array("B", short_row.data[:62])

    short_data = cloud()
    short_data.data = array("B", short_data.data[:-1])

    extra_data = cloud()
    extra_data.data.append(99)

    return [
        (missing, "exactly one time field"),
        (duplicate, "exactly one time field"),
        (wrong_datatype, "FLOAT32"),
        (wrong_count, "count"),
        (outside_point, "point_step"),
        (zero_point_step, "point_step"),
        (short_row, "row_step"),
        (short_data, "data length"),
        (extra_data, "data length"),
    ]


@pytest.mark.parametrize(("message", "match"), malformed_cases())
def test_malformed_cloud_is_rejected_before_modification(message, match):
    module = boundary_module()
    before = bytes(message.data)

    with pytest.raises(module.PointTimeFieldError, match=match):
        module.zero_point_time_field(message)

    assert bytes(message.data) == before


@contextmanager
def running_boundary():
    module = boundary_module()
    rclpy.init()
    node = None
    try:
        node = module.PointTimeZeroBoundary()
        yield module, node
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_boundary_node_owns_only_internal_input_and_public_output():
    with running_boundary() as (module, node):
        assert node.get_name() == "ad_point_time_zero_boundary"
        subscriptions = node.get_subscriptions_info_by_topic(
            module.INPUT_TOPIC
        )
        publishers = node.get_publishers_info_by_topic(module.OUTPUT_TOPIC)
        assert len(subscriptions) == 1
        assert len(publishers) == 1
        assert subscriptions[0].topic_type == "sensor_msgs/msg/PointCloud2"
        assert publishers[0].topic_type == "sensor_msgs/msg/PointCloud2"
        assert (
            subscriptions[0].qos_profile.reliability
            == ReliabilityPolicy.BEST_EFFORT
        )
        assert (
            publishers[0].qos_profile.reliability
            == ReliabilityPolicy.BEST_EFFORT
        )
        assert (
            subscriptions[0].qos_profile.durability
            == DurabilityPolicy.VOLATILE
        )
        assert (
            publishers[0].qos_profile.durability
            == DurabilityPolicy.VOLATILE
        )
        assert (
            module.INPUT_TOPIC
            == "/ad/sensors/lidar/points_with_synthetic_time"
        )
        assert module.OUTPUT_TOPIC == "/ad/sensors/lidar/points"


def test_boundary_node_drops_malformed_cloud_without_publication(monkeypatch):
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

    with running_boundary() as (_module, node):
        publisher = CapturePublisher()
        logger = CaptureLogger()
        node._publisher = publisher
        monkeypatch.setattr(node, "get_logger", lambda: logger)
        invalid = cloud()
        invalid.fields = [
            item for item in invalid.fields if item.name != "time"
        ]

        node._on_cloud(invalid)
        valid = cloud()
        node._on_cloud(valid)

        assert len(logger.warnings) == 1
        assert "dropped malformed PointCloud2" in logger.warnings[0]
        assert publisher.messages == [valid]
        assert all(
            valid.data[index] == 0 for index in time_byte_indices(valid)
        )
