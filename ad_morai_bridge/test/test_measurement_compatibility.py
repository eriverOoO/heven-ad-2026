from contextlib import contextmanager
import math

from ad_morai_interfaces.msg import ImuPacket
import pytest
import rclpy
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy

from ad_morai_bridge.measurement_compatibility import (
    INPUT_TOPIC,
    OUTPUT_TOPIC,
    ExactImuRepeatFilter,
    MeasurementCompatibilityNode,
)


def imu_packet(
    *,
    device_stamp: tuple[int, int] | None = (10, 20),
    receipt_stamp: tuple[int, int] = (100, 200),
    orientation: tuple[float, float, float, float] = (1.0, 2.0, 2.0, 1.0),
    angular_velocity: tuple[float, float, float] = (0.1, 0.2, 0.3),
    linear_acceleration: tuple[float, float, float] = (1.1, 1.2, 1.3),
) -> ImuPacket:
    message = ImuPacket()
    message.header.stamp.sec, message.header.stamp.nanosec = receipt_stamp
    message.header.frame_id = "imu_link"
    message.has_device_stamp = device_stamp is not None
    if device_stamp is not None:
        message.device_stamp.sec, message.device_stamp.nanosec = device_stamp
    (
        message.orientation.x,
        message.orientation.y,
        message.orientation.z,
        message.orientation.w,
    ) = orientation
    (
        message.angular_velocity.x,
        message.angular_velocity.y,
        message.angular_velocity.z,
    ) = angular_velocity
    (
        message.linear_acceleration.x,
        message.linear_acceleration.y,
        message.linear_acceleration.z,
    ) = linear_acceleration
    return message


def test_accept_normalizes_quaternion_and_preserves_receipt_header_and_vectors():
    repeat_filter = ExactImuRepeatFilter()

    result = repeat_filter.accept(imu_packet())

    assert result is not None
    assert (result.header.stamp.sec, result.header.stamp.nanosec) == (100, 200)
    assert result.header.frame_id == "imu_link"
    assert (
        result.orientation.x,
        result.orientation.y,
        result.orientation.z,
        result.orientation.w,
    ) == pytest.approx(
        (
            1.0 / math.sqrt(10.0),
            2.0 / math.sqrt(10.0),
            2.0 / math.sqrt(10.0),
            1.0 / math.sqrt(10.0),
        )
    )
    assert (
        result.angular_velocity.x,
        result.angular_velocity.y,
        result.angular_velocity.z,
    ) == (0.1, 0.2, 0.3)
    assert (
        result.linear_acceleration.x,
        result.linear_acceleration.y,
        result.linear_acceleration.z,
    ) == (1.1, 1.2, 1.3)
    assert repeat_filter.published == 1
    assert repeat_filter.exact_repeats == 0
    assert repeat_filter.invalid_device_stamps == 0
    assert repeat_filter.epoch_resets == 0


def test_accept_normalizes_large_finite_quaternion_without_overflow():
    repeat_filter = ExactImuRepeatFilter()

    result = repeat_filter.accept(
        imu_packet(orientation=(1.0e308, 0.0, 0.0, 1.0e308))
    )

    assert result is not None
    assert (
        result.orientation.x,
        result.orientation.y,
        result.orientation.z,
        result.orientation.w,
    ) == pytest.approx((math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)))


def test_exact_repeat_requires_same_valid_stamp_and_byte_identical_payload():
    repeat_filter = ExactImuRepeatFilter()
    first = imu_packet()
    exact_repeat = imu_packet(receipt_stamp=(100, 300))
    changed_payload = imu_packet(
        receipt_stamp=(100, 400), angular_velocity=(0.1, 0.2, 0.4)
    )

    assert repeat_filter.accept(first) is not None
    assert repeat_filter.accept(exact_repeat) is None
    changed = repeat_filter.accept(changed_payload)

    assert changed is not None
    assert changed.header.stamp.nanosec == 400
    assert changed.angular_velocity.z == 0.4
    assert repeat_filter.published == 2
    assert repeat_filter.exact_repeats == 1


def test_signed_zero_payload_change_at_same_stamp_is_not_byte_identical():
    repeat_filter = ExactImuRepeatFilter()

    assert repeat_filter.accept(
        imu_packet(angular_velocity=(0.0, 0.2, 0.3))
    ) is not None
    changed = repeat_filter.accept(
        imu_packet(angular_velocity=(-0.0, 0.2, 0.3))
    )

    assert changed is not None
    assert math.copysign(1.0, changed.angular_velocity.x) == -1.0
    assert repeat_filter.published == 2
    assert repeat_filter.exact_repeats == 0


def test_missing_or_invalid_device_stamp_publishes_and_invalidates_repeat_state():
    repeat_filter = ExactImuRepeatFilter()

    assert repeat_filter.accept(imu_packet()) is not None
    assert repeat_filter.accept(imu_packet(device_stamp=None)) is not None
    # The missing-stamp packet was the preceding accepted packet, so the same
    # valid stamp and payload cannot be treated as its exact repeat.
    assert repeat_filter.accept(imu_packet()) is not None
    assert repeat_filter.accept(imu_packet(device_stamp=(-1, 20))) is not None

    assert repeat_filter.published == 4
    assert repeat_filter.exact_repeats == 0
    assert repeat_filter.invalid_device_stamps == 2
    assert repeat_filter.epoch_resets == 0


def test_regressed_device_stamp_starts_new_epoch_before_repeat_suppression():
    repeat_filter = ExactImuRepeatFilter()

    assert repeat_filter.accept(imu_packet(device_stamp=(10, 20))) is not None
    assert repeat_filter.accept(imu_packet(device_stamp=(9, 999))) is not None
    assert repeat_filter.accept(imu_packet(device_stamp=(9, 999))) is None

    assert repeat_filter.published == 2
    assert repeat_filter.exact_repeats == 1
    assert repeat_filter.invalid_device_stamps == 0
    assert repeat_filter.epoch_resets == 1


@pytest.mark.parametrize(
    "orientation",
    [
        (0.0, 0.0, 0.0, 0.0),
        (math.nan, 0.0, 0.0, 1.0),
        (math.inf, 0.0, 0.0, 1.0),
    ],
)
def test_invalid_quaternion_is_rejected_without_advancing_filter_state(
    orientation,
):
    repeat_filter = ExactImuRepeatFilter()

    with pytest.raises(ValueError, match="quaternion"):
        repeat_filter.accept(imu_packet(orientation=orientation))

    assert repeat_filter.published == 0
    assert repeat_filter.exact_repeats == 0


@contextmanager
def running_node():
    rclpy.init()
    node = None
    try:
        node = MeasurementCompatibilityNode()
        yield node
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_node_owns_only_full_packet_input_and_compatible_output():
    with running_node() as node:
        subscriptions = node.get_subscriptions_info_by_topic(INPUT_TOPIC)
        publishers = node.get_publishers_info_by_topic(OUTPUT_TOPIC)

        assert node.get_name() == "ad_measurement_compatibility"
        assert len(subscriptions) == 1
        assert len(publishers) == 1
        assert subscriptions[0].topic_type == "ad_morai_interfaces/msg/ImuPacket"
        assert publishers[0].topic_type == "sensor_msgs/msg/Imu"
        assert subscriptions[0].qos_profile.reliability == ReliabilityPolicy.BEST_EFFORT
        assert publishers[0].qos_profile.reliability == ReliabilityPolicy.BEST_EFFORT
        assert subscriptions[0].qos_profile.durability == DurabilityPolicy.VOLATILE
        assert publishers[0].qos_profile.durability == DurabilityPolicy.VOLATILE


def test_node_publishes_accepted_messages_and_suppresses_only_exact_repeats():
    class CapturePublisher:
        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    with running_node() as node:
        publisher = CapturePublisher()
        node._publisher = publisher

        node._on_imu(imu_packet())
        node._on_imu(imu_packet(receipt_stamp=(100, 300)))
        node._on_imu(imu_packet(linear_acceleration=(1.1, 1.2, 1.4)))

        assert len(publisher.messages) == 2
        assert node._filter.published == 2
        assert node._filter.exact_repeats == 1
