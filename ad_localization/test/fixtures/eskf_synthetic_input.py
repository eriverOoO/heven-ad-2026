"""Deterministic ROS 2 sensor input used by the opt-in ESKF integration test."""

from __future__ import annotations

from ad_morai_interfaces.msg import EgoVehicleStatus
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, NavSatFix, NavSatStatus


class SyntheticLocalizationInput(Node):
    """Publish a stationary-compatible IMU, fixed GNSS, and directed wheel speed."""

    def __init__(self):
        super().__init__("eskf_synthetic_input")
        self.gps_enabled = True
        self.status_enabled = True
        self.speed_mps = 0.0
        self.acceleration_x_mps2 = 0.0
        self.gear = 4
        self._tick = 0
        self.gps_publisher = self.create_publisher(
            NavSatFix, "/ad/sensors/gps/fix", qos_profile_sensor_data
        )
        self.imu_publisher = self.create_publisher(
            Imu, "/ad/sensors/imu/data", qos_profile_sensor_data
        )
        self.status_publisher = self.create_publisher(
            EgoVehicleStatus, "/ad/vehicle/status", qos_profile_sensor_data
        )
        # Exercise the minimum supported hardware contract directly.  The
        # stationary initializer must use elapsed time as its primary evidence,
        # rather than silently requiring a 50 Hz IMU through its sample floor.
        self.timer = self.create_timer(0.05, self._publish)

    def inputs_are_matched(self) -> bool:
        return (
            self.gps_publisher.get_subscription_count() >= 1
            and self.imu_publisher.get_subscription_count() >= 2
            and self.status_publisher.get_subscription_count() >= 1
        )

    def _publish(self):
        stamp = self.get_clock().now().to_msg()

        imu = Imu()
        imu.header.stamp = stamp
        imu.header.frame_id = "imu_link"
        imu.orientation.w = 1.0
        imu.orientation_covariance[0] = 0.01
        imu.orientation_covariance[4] = 0.01
        imu.orientation_covariance[8] = 0.01
        imu.angular_velocity_covariance[0] = 0.01
        imu.angular_velocity_covariance[4] = 0.01
        imu.angular_velocity_covariance[8] = 0.01
        imu.linear_acceleration.x = self.acceleration_x_mps2
        imu.linear_acceleration.z = 9.80665
        imu.linear_acceleration_covariance[0] = 0.01
        imu.linear_acceleration_covariance[4] = 0.01
        imu.linear_acceleration_covariance[8] = 0.01
        self.imu_publisher.publish(imu)

        status = EgoVehicleStatus()
        status.header.stamp = stamp
        status.header.frame_id = "base_link"
        status.velocity.x = self.speed_mps
        status.gear = self.gear
        if self.status_enabled:
            self.status_publisher.publish(status)

        if self.gps_enabled and self._tick % 4 == 0:
            fix = NavSatFix()
            fix.header.stamp = stamp
            fix.header.frame_id = "gps_link"
            fix.status.status = NavSatStatus.STATUS_FIX
            fix.status.service = NavSatStatus.SERVICE_GPS
            fix.latitude = 37.2390904486269
            fix.longitude = 126.773066537479
            fix.altitude = 29.5769634246826
            self.gps_publisher.publish(fix)
        self._tick += 1
