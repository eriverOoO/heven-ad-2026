from array import array
from numbers import Integral

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField


INPUT_TOPIC = "/ad/sensors/lidar/points_with_synthetic_time"
OUTPUT_TOPIC = "/ad/sensors/lidar/points"
_FLOAT32_SIZE = 4


class PointTimeFieldError(ValueError):
    pass


def _validated_layout(message: PointCloud2) -> tuple[int, int, int, int]:
    if not isinstance(message, PointCloud2):
        raise PointTimeFieldError("message must be a PointCloud2")

    time_fields = [item for item in message.fields if item.name == "time"]
    if len(time_fields) != 1:
        raise PointTimeFieldError(
            "PointCloud2 must have exactly one time field"
        )
    time_field = time_fields[0]
    if time_field.datatype != PointField.FLOAT32:
        raise PointTimeFieldError("PointCloud2 time field must be FLOAT32")
    if time_field.count != 1:
        raise PointTimeFieldError("PointCloud2 time field count must be one")

    point_step = message.point_step
    row_step = message.row_step
    width = message.width
    height = message.height
    if not all(
        isinstance(value, Integral)
        for value in (point_step, row_step, width, height)
    ):
        raise PointTimeFieldError("PointCloud2 dimensions must be integers")
    if point_step <= 0:
        raise PointTimeFieldError("PointCloud2 point_step must be positive")
    if time_field.offset + _FLOAT32_SIZE > point_step:
        raise PointTimeFieldError("PointCloud2 time field exceeds point_step")
    if row_step < width * point_step:
        raise PointTimeFieldError(
            "PointCloud2 row_step is shorter than its row"
        )
    expected_size = row_step * height
    if len(message.data) != expected_size:
        raise PointTimeFieldError(
            "PointCloud2 data length does not equal row_step * height"
        )
    return time_field.offset, point_step, row_step, width


def zero_point_time_field(message: PointCloud2) -> PointCloud2:
    """Set every FLOAT32 ``time`` field to +0 without changing other bytes."""

    time_offset, point_step, row_step, width = _validated_layout(message)
    zeros = array("B", [0]) * width
    for row in range(message.height):
        row_base = row * row_step
        for byte_offset in range(_FLOAT32_SIZE):
            start = row_base + time_offset + byte_offset
            stop = start + width * point_step
            message.data[start:stop:point_step] = zeros
    return message


class PointTimeZeroBoundary(Node):
    def __init__(self):
        super().__init__("ad_point_time_zero_boundary")
        self._drop_count = 0
        self._publisher = self.create_publisher(
            PointCloud2, OUTPUT_TOPIC, qos_profile_sensor_data
        )
        self._subscription = self.create_subscription(
            PointCloud2,
            INPUT_TOPIC,
            self._on_cloud,
            qos_profile_sensor_data,
        )

    def _on_cloud(self, message: PointCloud2) -> None:
        try:
            zero_point_time_field(message)
        except PointTimeFieldError as exc:
            self._drop_count += 1
            count = self._drop_count
            if count <= 3 or count & (count - 1) == 0:
                self.get_logger().warning(
                    f"dropped malformed PointCloud2 #{count}: {exc}"
                )
            return
        self._publisher.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PointTimeZeroBoundary()
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
