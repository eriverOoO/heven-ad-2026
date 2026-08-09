"""Optional MORAI-only IMU compatibility stream for localization consumers."""

import math
import struct

from ad_morai_interfaces.msg import ImuPacket
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu


INPUT_TOPIC = "/ad/sensors/imu/full"
OUTPUT_TOPIC = "/ad/localization/input/imu_compatible"
_NANOSECONDS_PER_SECOND = 1_000_000_000


def _device_stamp(message: ImuPacket) -> tuple[int, int] | None:
    if not message.has_device_stamp:
        return None
    stamp = message.device_stamp
    if stamp.sec < 0 or not 0 <= stamp.nanosec < _NANOSECONDS_PER_SECOND:
        return None
    return stamp.sec, stamp.nanosec


def _payload_bytes(message: ImuPacket) -> bytes:
    """Return the exact floating-point payload bits in message field order."""

    return struct.pack(
        "<10d",
        message.orientation.x,
        message.orientation.y,
        message.orientation.z,
        message.orientation.w,
        message.angular_velocity.x,
        message.angular_velocity.y,
        message.angular_velocity.z,
        message.linear_acceleration.x,
        message.linear_acceleration.y,
        message.linear_acceleration.z,
    )


def _normalized_imu(message: ImuPacket) -> Imu:
    quaternion = (
        message.orientation.x,
        message.orientation.y,
        message.orientation.z,
        message.orientation.w,
    )
    if not all(math.isfinite(value) for value in quaternion):
        raise ValueError("IMU quaternion must contain only finite values")
    scale = max(abs(value) for value in quaternion)
    if scale == 0.0:
        raise ValueError("IMU quaternion norm is too small")
    scaled_quaternion = tuple(value / scale for value in quaternion)
    scaled_norm_squared = sum(
        value * value for value in scaled_quaternion
    )
    if (
        not math.isfinite(scaled_norm_squared)
        or scaled_norm_squared <= 0.0
    ):
        raise ValueError("IMU quaternion computed norm is invalid")
    scaled_norm = math.sqrt(scaled_norm_squared)
    if scale < 1.0e-6 / scaled_norm:
        raise ValueError("IMU quaternion norm is too small")
    normalized_quaternion = tuple(
        value / scaled_norm for value in scaled_quaternion
    )
    if not all(math.isfinite(value) for value in normalized_quaternion):
        raise ValueError("IMU quaternion computed norm is invalid")

    result = Imu()
    result.header.stamp.sec = message.header.stamp.sec
    result.header.stamp.nanosec = message.header.stamp.nanosec
    result.header.frame_id = message.header.frame_id
    (
        result.orientation.x,
        result.orientation.y,
        result.orientation.z,
        result.orientation.w,
    ) = normalized_quaternion
    (
        result.angular_velocity.x,
        result.angular_velocity.y,
        result.angular_velocity.z,
    ) = (
        message.angular_velocity.x,
        message.angular_velocity.y,
        message.angular_velocity.z,
    )
    (
        result.linear_acceleration.x,
        result.linear_acceleration.y,
        result.linear_acceleration.z,
    ) = (
        message.linear_acceleration.x,
        message.linear_acceleration.y,
        message.linear_acceleration.z,
    )
    return result


class ExactImuRepeatFilter:
    """Suppress only exact payload repeats within one valid device-time epoch."""

    def __init__(self) -> None:
        self.published = 0
        self.exact_repeats = 0
        self.invalid_device_stamps = 0
        self.epoch_resets = 0
        self._last_device_stamp: tuple[int, int] | None = None
        self._last_payload: bytes | None = None

    def accept(self, message: ImuPacket) -> Imu | None:
        if not isinstance(message, ImuPacket):
            raise TypeError("message must be an ImuPacket")

        # Validate and convert before advancing repeat state. A malformed
        # quaternion must not make a later valid packet look repeated.
        result = _normalized_imu(message)
        device_stamp = _device_stamp(message)
        payload = _payload_bytes(message)

        if device_stamp is None:
            self.invalid_device_stamps += 1
            self._last_device_stamp = None
            self._last_payload = None
        else:
            if (
                self._last_device_stamp is not None
                and device_stamp < self._last_device_stamp
            ):
                self.epoch_resets += 1
                self._last_device_stamp = None
                self._last_payload = None
            if (
                device_stamp == self._last_device_stamp
                and payload == self._last_payload
            ):
                self.exact_repeats += 1
                return None
            self._last_device_stamp = device_stamp
            self._last_payload = payload

        self.published += 1
        return result


class MeasurementCompatibilityNode(Node):
    def __init__(self) -> None:
        super().__init__("ad_measurement_compatibility")
        input_topic = str(self.declare_parameter("input_topic", INPUT_TOPIC).value)
        output_topic = str(
            self.declare_parameter("output_topic", OUTPUT_TOPIC).value
        )
        self._filter = ExactImuRepeatFilter()
        self._publisher = self.create_publisher(
            Imu, output_topic, qos_profile_sensor_data
        )
        self._subscription = self.create_subscription(
            ImuPacket,
            input_topic,
            self._on_imu,
            qos_profile_sensor_data,
        )
        self._invalid_payloads = 0

    def _on_imu(self, message: ImuPacket) -> None:
        try:
            compatible = self._filter.accept(message)
        except ValueError as exc:
            self._invalid_payloads += 1
            count = self._invalid_payloads
            if count <= 3 or count & (count - 1) == 0:
                self.get_logger().warning(
                    f"dropped invalid IMU payload #{count}: {exc}"
                )
            return
        if compatible is not None:
            self._publisher.publish(compatible)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = MeasurementCompatibilityNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
