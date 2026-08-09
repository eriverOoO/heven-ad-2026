import math

import pytest

from ad_morai_bridge.message_conversion import (
    collision_array_message,
    ego_status_message,
    gps_fix_message,
    gps_gga_message,
    gps_rmc_message,
    imu_message,
    imu_packet_message,
)
from ad_morai_bridge.protocol_records import (
    CollisionArrayRecord,
    CollisionRecord,
    EgoStatusRecord,
    GpggaRecord,
    GpsFixRecord,
    GprmcRecord,
    ImuRecord,
)


def test_ego_status_converts_protocol_units_exactly_once():
    record = EgoStatusRecord(
        stamp=(12, 345),
        ctrl_mode=2,
        gear=4,
        signed_velocity=36.0,
        map_data_id=17,
        accel=0.25,
        brake=0.5,
        size=(4.6, 1.8, 1.5),
        overhang=0.9,
        wheelbase=2.7,
        rear_overhang=1.0,
        position=(101.25, -22.5, 3.0),
        rpy=(180.0, -90.0, 45.0),
        velocity=(3.6, -7.2, 10.8),
        angular_velocity=(360.0, -180.0, 90.0),
        acceleration=(1.25, -2.5, 9.81),
        steering=-30.0,
        link_id="L17",
        tire_metrics=tuple(float(index) for index in range(1, 13)),
    )

    message = ego_status_message(record, "map", (100, 200))

    assert (message.header.stamp.sec, message.header.stamp.nanosec) == (100, 200)
    assert message.header.frame_id == "map"
    assert message.has_device_stamp is True
    assert (message.device_stamp.sec, message.device_stamp.nanosec) == (12, 345)
    assert message.signed_velocity == pytest.approx(10.0)
    assert (message.velocity.x, message.velocity.y, message.velocity.z) == pytest.approx(
        (1.0, -2.0, 3.0)
    )
    assert (message.rpy.x, message.rpy.y, message.rpy.z) == pytest.approx(
        (math.pi, -math.pi / 2.0, math.pi / 4.0)
    )
    assert (
        message.angular_velocity.x,
        message.angular_velocity.y,
        message.angular_velocity.z,
    ) == pytest.approx((2.0 * math.pi, -math.pi, math.pi / 2.0))
    assert message.steering == pytest.approx(-math.pi / 6.0)

    assert (message.size.x, message.size.y, message.size.z) == pytest.approx(record.size)
    assert message.overhang == pytest.approx(record.overhang)
    assert message.wheelbase == pytest.approx(record.wheelbase)
    assert message.rear_overhang == pytest.approx(record.rear_overhang)
    assert (message.position.x, message.position.y, message.position.z) == pytest.approx(
        record.position
    )
    assert (
        message.acceleration.x,
        message.acceleration.y,
        message.acceleration.z,
    ) == pytest.approx(record.acceleration)
    assert message.accel == pytest.approx(record.accel)
    assert message.brake == pytest.approx(record.brake)
    assert message.map_data_id == record.map_data_id
    assert message.link_id == record.link_id
    assert message.has_tire_metrics is True
    assert message.tire_cornering_stiffness_rr == pytest.approx(12.0)


def test_ego_status_marks_absent_tire_metrics_without_fabricating_presence():
    record = EgoStatusRecord(
        stamp=(1, 2),
        ctrl_mode=2,
        gear=4,
        signed_velocity=0.0,
        map_data_id=0,
        accel=0.0,
        brake=0.0,
        size=(0.0, 0.0, 0.0),
        overhang=0.0,
        wheelbase=0.0,
        rear_overhang=0.0,
        position=(0.0, 0.0, 0.0),
        rpy=(0.0, 0.0, 0.0),
        velocity=(0.0, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 0.0),
        acceleration=(0.0, 0.0, 0.0),
        steering=0.0,
        link_id="",
    )

    message = ego_status_message(record, "map", (3, 4))

    assert message.has_tire_metrics is False
    assert message.tire_lateral_force_fl == 0.0
    assert message.tire_cornering_stiffness_rr == 0.0


def test_ego_status_marks_missing_device_stamp_and_keeps_arrival_header():
    record = EgoStatusRecord(
        stamp=None,
        ctrl_mode=2,
        gear=4,
        signed_velocity=0.0,
        map_data_id=0,
        accel=0.0,
        brake=0.0,
        size=(0.0, 0.0, 0.0),
        overhang=0.0,
        wheelbase=0.0,
        rear_overhang=0.0,
        position=(0.0, 0.0, 0.0),
        rpy=(0.0, 0.0, 0.0),
        velocity=(0.0, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 0.0),
        acceleration=(0.0, 0.0, 0.0),
        steering=0.0,
        link_id="",
    )

    message = ego_status_message(record, "map", (7, 8))

    assert (message.header.stamp.sec, message.header.stamp.nanosec) == (7, 8)
    assert message.has_device_stamp is False
    assert (message.device_stamp.sec, message.device_stamp.nanosec) == (0, 0)


def test_collision_records_convert_without_field_loss():
    hit = CollisionRecord(1, 9, (1.0, 2.0, 3.0), (100.0, 200.0, -4.0))
    message = collision_array_message(CollisionArrayRecord((6, 7), (hit,)), "map")

    assert (message.header.stamp.sec, message.header.stamp.nanosec) == (6, 7)
    assert message.header.frame_id == "map"
    assert message.collisions[0].object_type == 1
    assert message.collisions[0].object_id == 9
    assert (
        message.collisions[0].position.x,
        message.collisions[0].position.y,
        message.collisions[0].position.z,
    ) == pytest.approx((1.0, 2.0, 3.0))
    assert (
        message.collisions[0].global_offset.x,
        message.collisions[0].global_offset.y,
        message.collisions[0].global_offset.z,
    ) == pytest.approx((100.0, 200.0, -4.0))


def test_imu_normalizes_a_finite_nonzero_quaternion():
    message = imu_message(
        ImuRecord((1, 2), (0.0, 0.0, 0.0, 2.0), (1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
        "imu_link",
    )

    assert message.header.frame_id == "imu_link"
    assert (message.orientation.x, message.orientation.y, message.orientation.z) == (
        0.0,
        0.0,
        0.0,
    )
    assert message.orientation.w == pytest.approx(1.0)
    assert (
        message.angular_velocity.x,
        message.angular_velocity.y,
        message.angular_velocity.z,
    ) == pytest.approx((1.0, 2.0, 3.0))
    assert (
        message.linear_acceleration.x,
        message.linear_acceleration.y,
        message.linear_acceleration.z,
    ) == pytest.approx((4.0, 5.0, 6.0))


def test_imu_packet_preserves_arrival_and_optional_device_stamp():
    record = ImuRecord(
        (12, 34),
        (0.1, 0.2, 0.3, 0.9),
        (4.0, 5.0, 6.0),
        (7.0, 8.0, 9.0),
    )

    message = imu_packet_message(record, "imu_link", (100, 200))

    assert (message.header.stamp.sec, message.header.stamp.nanosec) == (100, 200)
    assert message.header.frame_id == "imu_link"
    assert message.has_device_stamp is True
    assert (message.device_stamp.sec, message.device_stamp.nanosec) == (12, 34)
    assert (
        message.orientation.x,
        message.orientation.y,
        message.orientation.z,
        message.orientation.w,
    ) == pytest.approx(record.orientation_xyzw)
    assert (
        message.angular_velocity.x,
        message.angular_velocity.y,
        message.angular_velocity.z,
    ) == pytest.approx(record.angular_velocity)
    assert (
        message.linear_acceleration.x,
        message.linear_acceleration.y,
        message.linear_acceleration.z,
    ) == pytest.approx(record.linear_acceleration)


def test_imu_packet_marks_missing_device_stamp_and_leaves_zero_time():
    record = ImuRecord(
        None,
        (0.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )

    message = imu_packet_message(record, "imu_link", (3, 4))

    assert message.has_device_stamp is False
    assert (message.device_stamp.sec, message.device_stamp.nanosec) == (0, 0)


@pytest.mark.parametrize(
    "orientation",
    [
        (0.0, 0.0, 0.0, 0.0),
        (math.nan, 0.0, 0.0, 1.0),
        (0.0, math.inf, 0.0, 1.0),
    ],
)
def test_imu_fails_closed_on_invalid_quaternion(orientation):
    record = ImuRecord(None, orientation, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="quaternion"):
        imu_message(record, "imu_link")


def test_gps_fix_uses_standard_message_and_hdop_covariance():
    record = GpsFixRecord(37.0, 127.0, 12.0, 1, 8, 0.9, 1.0, 2.0)
    message = gps_fix_message(record, "gps_link", (3, 4))

    assert (message.header.stamp.sec, message.header.stamp.nanosec) == (3, 4)
    assert message.header.frame_id == "gps_link"
    assert message.latitude == pytest.approx(37.0)
    assert message.longitude == pytest.approx(127.0)
    assert message.altitude == pytest.approx(12.0)
    assert message.status.status == 0
    assert message.position_covariance[0] == pytest.approx((0.9 * 3.0) ** 2)


def test_full_gps_messages_preserve_optional_fields_and_presence_flags():
    rmc = GprmcRecord(
        utc="123519.25",
        valid=True,
        latitude=-48.1173,
        longitude=-11.5167,
        speed_knots=22.4,
        track_degrees=84.4,
        date="230394",
        magnetic_variation_degrees=3.1,
        magnetic_variation_direction="E",
        mode_indicator="D",
        checksum=0x5A,
        sentence="$GPRMC,123519.25,...*5A",
    )
    gga = GpggaRecord(
        utc="123520.50",
        latitude=48.1173,
        longitude=11.5167,
        fix_quality=2,
        satellites=11,
        hdop=0.8,
        altitude=545.4,
        altitude_unit="M",
        geoid_separation=46.9,
        geoid_unit="M",
        differential_age=1.5,
        station_id="0042",
        checksum=0x6B,
        sentence="$GPGGA,123520.50,...*6B",
    )

    rmc_message = gps_rmc_message(rmc, "gps_link", (7, 8))
    gga_message = gps_gga_message(gga, "gps_link", (9, 10))

    assert rmc_message.sentence == rmc.sentence
    assert gga_message.sentence == gga.sentence
    assert rmc_message.has_speed and rmc_message.speed_knots == pytest.approx(22.4)
    assert rmc_message.has_track and rmc_message.track_degrees == pytest.approx(84.4)
    assert rmc_message.has_magnetic_variation
    assert rmc_message.magnetic_variation_degrees == pytest.approx(3.1)
    assert rmc_message.magnetic_variation_direction == "E"
    assert rmc_message.mode_indicator == "D"
    assert rmc_message.has_checksum and rmc_message.checksum == 0x5A
    assert gga_message.has_hdop and gga_message.hdop == pytest.approx(0.8)
    assert gga_message.has_altitude and gga_message.altitude == pytest.approx(545.4)
    assert gga_message.has_geoid_separation
    assert gga_message.geoid_separation == pytest.approx(46.9)
    assert gga_message.has_differential_age
    assert gga_message.differential_age == pytest.approx(1.5)
    assert gga_message.station_id == "0042"
    assert gga_message.has_checksum and gga_message.checksum == 0x6B
