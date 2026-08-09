"""Convert decoded MORAI protocol records into ROS messages."""

import math

from builtin_interfaces.msg import Time
from sensor_msgs.msg import Imu, NavSatFix, NavSatStatus, TimeReference

from ad_morai_interfaces.msg import (
    Collision,
    CollisionArray,
    EgoVehicleStatus,
    GpsGga,
    GpsRmc,
    ImuPacket,
    SensorTiming,
)

from .protocol_records import (
    CollisionArrayRecord,
    EgoStatusRecord,
    GpggaRecord,
    GpsFixRecord,
    GprmcRecord,
    ImuRecord,
)
from .timestamp_policy import TimestampDecision, is_valid_ros_stamp


def time_message(stamp: tuple[int, int] | None) -> Time:
    message = Time()
    if stamp is not None:
        message.sec = int(stamp[0])
        message.nanosec = int(stamp[1])
    return message


def _floats(values):
    return tuple(float(value) for value in values)


def _radians(values):
    return tuple(math.radians(float(value)) for value in values)


def collision_array_message(
    record: CollisionArrayRecord, frame_id: str
) -> CollisionArray:
    result = CollisionArray()
    result.header.stamp = time_message(record.stamp)
    result.header.frame_id = frame_id
    for item in record.collisions:
        message = Collision()
        message.object_type = item.object_type
        message.object_id = item.object_id
        message.position.x, message.position.y, message.position.z = _floats(
            item.position
        )
        (
            message.global_offset.x,
            message.global_offset.y,
            message.global_offset.z,
        ) = _floats(item.global_offset)
        result.collisions.append(message)
    return result


def ego_status_message(
    record: EgoStatusRecord,
    frame_id: str,
    header_stamp: tuple[int, int],
) -> EgoVehicleStatus:
    message = EgoVehicleStatus()
    message.header.stamp = time_message(header_stamp)
    message.header.frame_id = frame_id
    message.has_device_stamp = is_valid_ros_stamp(record.stamp)
    if message.has_device_stamp:
        message.device_stamp = time_message(record.stamp)
    message.ctrl_mode = record.ctrl_mode
    message.gear = record.gear
    message.signed_velocity = float(record.signed_velocity) / 3.6
    message.map_data_id = record.map_data_id
    message.accel = record.accel
    message.brake = record.brake
    message.size.x, message.size.y, message.size.z = _floats(record.size)
    message.overhang = record.overhang
    message.wheelbase = record.wheelbase
    message.rear_overhang = record.rear_overhang
    message.position.x, message.position.y, message.position.z = _floats(
        record.position
    )
    message.rpy.x, message.rpy.y, message.rpy.z = _radians(record.rpy)
    message.velocity.x, message.velocity.y, message.velocity.z = tuple(
        float(value) / 3.6 for value in record.velocity
    )
    (
        message.angular_velocity.x,
        message.angular_velocity.y,
        message.angular_velocity.z,
    ) = _radians(record.angular_velocity)
    (
        message.acceleration.x,
        message.acceleration.y,
        message.acceleration.z,
    ) = _floats(record.acceleration)
    message.steering = math.radians(float(record.steering))
    message.link_id = record.link_id
    message.has_tire_metrics = len(record.tire_metrics) == 12

    tire = record.tire_metrics + (0.0,) * (12 - len(record.tire_metrics))
    (
        message.tire_lateral_force_fl,
        message.tire_lateral_force_fr,
        message.tire_lateral_force_rl,
        message.tire_lateral_force_rr,
        message.side_slip_angle_fl,
        message.side_slip_angle_fr,
        message.side_slip_angle_rl,
        message.side_slip_angle_rr,
        message.tire_cornering_stiffness_fl,
        message.tire_cornering_stiffness_fr,
        message.tire_cornering_stiffness_rl,
        message.tire_cornering_stiffness_rr,
    ) = _floats(tire[:12])
    return message


def imu_message(record: ImuRecord, frame_id: str) -> Imu:
    message = Imu()
    message.header.stamp = time_message(record.stamp)
    message.header.frame_id = frame_id

    quaternion = _floats(record.orientation_xyzw)
    if not all(math.isfinite(value) for value in quaternion):
        raise ValueError("IMU quaternion must contain only finite values")
    norm_squared = sum(value * value for value in quaternion)
    if norm_squared < 1e-12:
        raise ValueError("IMU quaternion norm is too small")
    norm = math.sqrt(norm_squared)
    (
        message.orientation.x,
        message.orientation.y,
        message.orientation.z,
        message.orientation.w,
    ) = tuple(value / norm for value in quaternion)

    angular_velocity = _floats(record.angular_velocity)
    linear_acceleration = _floats(record.linear_acceleration)
    if not all(
        math.isfinite(value) for value in (*angular_velocity, *linear_acceleration)
    ):
        raise ValueError("IMU vectors must contain only finite values")
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


def imu_packet_message(
    record: ImuRecord,
    frame_id: str,
    arrival_stamp: tuple[int, int],
) -> ImuPacket:
    message = ImuPacket()
    message.header.stamp = time_message(arrival_stamp)
    message.header.frame_id = frame_id
    message.has_device_stamp = is_valid_ros_stamp(record.stamp)
    if message.has_device_stamp:
        message.device_stamp = time_message(record.stamp)

    quaternion = _floats(record.orientation_xyzw)
    angular_velocity = _floats(record.angular_velocity)
    linear_acceleration = _floats(record.linear_acceleration)
    if not all(
        math.isfinite(value)
        for value in (*quaternion, *angular_velocity, *linear_acceleration)
    ):
        raise ValueError("full IMU message values must be finite")
    (
        message.orientation.x,
        message.orientation.y,
        message.orientation.z,
        message.orientation.w,
    ) = quaternion
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


def gps_rmc_message(
    record: GprmcRecord,
    frame_id: str,
    arrival_stamp: tuple[int, int],
) -> GpsRmc:
    message = GpsRmc()
    message.header.stamp = time_message(arrival_stamp)
    message.header.frame_id = frame_id
    message.sentence = record.sentence
    message.utc = record.utc
    message.valid = record.valid
    message.latitude = record.latitude
    message.longitude = record.longitude
    message.has_speed = record.speed_knots is not None
    if record.speed_knots is not None:
        message.speed_knots = record.speed_knots
    message.has_track = record.track_degrees is not None
    if record.track_degrees is not None:
        message.track_degrees = record.track_degrees
    message.date = record.date
    message.has_magnetic_variation = record.magnetic_variation_degrees is not None
    if record.magnetic_variation_degrees is not None:
        message.magnetic_variation_degrees = record.magnetic_variation_degrees
    message.magnetic_variation_direction = record.magnetic_variation_direction
    message.mode_indicator = record.mode_indicator
    message.has_checksum = record.checksum is not None
    if record.checksum is not None:
        message.checksum = record.checksum
    return message


def gps_gga_message(
    record: GpggaRecord,
    frame_id: str,
    arrival_stamp: tuple[int, int],
) -> GpsGga:
    message = GpsGga()
    message.header.stamp = time_message(arrival_stamp)
    message.header.frame_id = frame_id
    message.sentence = record.sentence
    message.utc = record.utc
    message.latitude = record.latitude
    message.longitude = record.longitude
    message.fix_quality = record.fix_quality
    message.satellites = record.satellites
    message.has_hdop = record.hdop is not None
    if record.hdop is not None:
        message.hdop = record.hdop
    message.has_altitude = record.altitude is not None
    if record.altitude is not None:
        message.altitude = record.altitude
    message.altitude_unit = record.altitude_unit
    message.has_geoid_separation = record.geoid_separation is not None
    if record.geoid_separation is not None:
        message.geoid_separation = record.geoid_separation
    message.geoid_unit = record.geoid_unit
    message.has_differential_age = record.differential_age is not None
    if record.differential_age is not None:
        message.differential_age = record.differential_age
    message.station_id = record.station_id
    message.has_checksum = record.checksum is not None
    if record.checksum is not None:
        message.checksum = record.checksum
    return message


def gps_fix_message(
    record: GpsFixRecord,
    frame_id: str,
    stamp: tuple[int, int] | None,
) -> NavSatFix:
    message = NavSatFix()
    message.header.stamp = time_message(stamp)
    message.header.frame_id = frame_id
    message.status.status = (
        NavSatStatus.STATUS_FIX if record.status > 0 else NavSatStatus.STATUS_NO_FIX
    )
    message.status.service = NavSatStatus.SERVICE_GPS
    message.latitude = record.latitude
    message.longitude = record.longitude
    message.altitude = record.altitude
    if record.hdop is not None:
        variance = (record.hdop * 3.0) ** 2
        message.position_covariance[0] = variance
        message.position_covariance[4] = variance
        message.position_covariance[8] = variance * 4.0
        message.position_covariance_type = NavSatFix.COVARIANCE_TYPE_APPROXIMATED
    return message


def gps_time_reference_message(
    frame_id: str,
    arrival_stamp: tuple[int, int],
    source_stamp: tuple[int, int],
) -> TimeReference:
    message = TimeReference()
    message.header.stamp = time_message(arrival_stamp)
    message.header.frame_id = frame_id
    message.time_ref = time_message(source_stamp)
    message.source = "MORAI GPRMC date plus whole UTC second"
    return message


def sensor_timing_message(
    stream: str,
    frame_id: str,
    arrival_stamp: tuple[int, int],
    source_stamp: tuple[int, int] | None,
    decision: TimestampDecision,
) -> SensorTiming:
    message = SensorTiming()
    message.header.stamp = time_message(arrival_stamp)
    message.header.frame_id = frame_id
    message.stream = stream
    message.has_source_stamp = source_stamp is not None
    if source_stamp is not None:
        message.source_sec = int(source_stamp[0])
        message.source_nanosec = int(source_stamp[1])
    message.selected_stamp = time_message(decision.selected_stamp)
    message.source_valid = decision.source_valid
    message.source_selected = decision.source_selected
    message.source_rejected = decision.source_rejected
    message.duplicate = decision.duplicate
    message.stamp_regression = decision.stamp_regression
    message.normalized_published = decision.publish_normalized
    return message
