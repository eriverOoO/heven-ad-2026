import math

import pytest
from ad_morai_interfaces_dev.msg import (
    CameraBoundingBoxArray,
    EgoVehicleStatus,
    Lidar2D,
    ObjectStatusArray,
    VehicleCollisionArray,
)
from sensor_msgs.msg import LaserScan

from ad_morai_bridge_dev.bridge.message_conversion import (
    camera_bounding_boxes_message,
    dev_ego_status_message,
    laser_scan_message,
    lidar2d_message,
    object_array_message,
    vehicle_collision_array_message,
)
from ad_morai_bridge_dev.bridge.protocol_records import (
    CameraBoundingBoxArrayRecord,
    CameraBoundingBoxRecord,
    Lidar2DRecord,
    ObjectArrayRecord,
    ObjectRecord,
    VehicleCollisionArrayRecord,
    VehicleCollisionObjectRecord,
    VehicleCollisionRecord,
)
from ad_morai_bridge.protocol_records import EgoStatusRecord


def test_dev_object_converter_uses_dev_interface_and_si_units() -> None:
    record = ObjectArrayRecord(
        (4, 5),
        (
            ObjectRecord(
                3,
                1,
                (1.0, 2.0, 3.0),
                90.0,
                (4.0, 2.0, 1.0),
                1.0,
                3.0,
                1.0,
                (36.0, 0.0, 0.0),
                (0.5, 0.0, 0.0),
                "L1",
            ),
        ),
    )

    message = object_array_message(record, "map")

    assert isinstance(message, ObjectStatusArray)
    assert message.header.stamp.sec == 4
    assert message.header.stamp.nanosec == 5
    assert message.header.frame_id == "map"
    assert message.objects[0].unique_id == 3
    assert message.objects[0].object_type == 1
    assert (
        message.objects[0].position.x,
        message.objects[0].position.y,
        message.objects[0].position.z,
    ) == pytest.approx((1.0, 2.0, 3.0))
    assert message.objects[0].heading == pytest.approx(math.pi / 2.0)
    assert (
        message.objects[0].size.x,
        message.objects[0].size.y,
        message.objects[0].size.z,
    ) == pytest.approx((4.0, 2.0, 1.0))
    assert message.objects[0].overhang == pytest.approx(1.0)
    assert message.objects[0].wheelbase == pytest.approx(3.0)
    assert message.objects[0].rear_overhang == pytest.approx(1.0)
    assert message.objects[0].velocity.x == pytest.approx(10.0)
    assert message.objects[0].acceleration.x == pytest.approx(0.5)
    assert message.objects[0].link_id == "L1"


def test_dev_laser_scan_converter_uses_standard_message_and_si_units() -> None:
    record = Lidar2DRecord(
        aux=(0.0, 0.0, 0.0),
        distances_m=(1.0, 2.0, 3.0, 4.0),
        intensities=(0, 64, 128, 255),
    )

    message = laser_scan_message(record, "lidar2d_link", (6, 7))

    assert isinstance(message, LaserScan)
    assert message.header.stamp.sec == 6
    assert message.header.stamp.nanosec == 7
    assert message.header.frame_id == "lidar2d_link"
    assert message.angle_min == pytest.approx(-math.pi)
    assert message.angle_increment == pytest.approx(math.pi / 2.0)
    assert message.angle_max == pytest.approx(math.pi / 2.0)
    assert tuple(message.ranges) == pytest.approx(record.distances_m)
    assert tuple(message.intensities) == pytest.approx(record.intensities)


def test_camera_bbox_converter_preserves_every_field() -> None:
    box = CameraBoundingBoxRecord(
        corners_3d=tuple(float(index) for index in range(24)),
        bounding_box_2d=(24.0, 25.0, 26.0, 27.0),
        group=7,
        class_id=8,
        subclass_id=9,
    )

    message = camera_bounding_boxes_message(
        CameraBoundingBoxArrayRecord((4, 5), (box,)), "camera_front_optical_frame"
    )

    assert isinstance(message, CameraBoundingBoxArray)
    assert (message.header.stamp.sec, message.header.stamp.nanosec) == (4, 5)
    assert message.header.frame_id == "camera_front_optical_frame"
    assert tuple(message.boxes[0].corners_3d) == pytest.approx(box.corners_3d)
    assert tuple(message.boxes[0].bounding_box_2d) == pytest.approx(
        box.bounding_box_2d
    )
    assert (
        message.boxes[0].group,
        message.boxes[0].class_id,
        message.boxes[0].subclass_id,
    ) == (7, 8, 9)


def test_vehicle_collision_converter_applies_si_units_exactly_once() -> None:
    first = VehicleCollisionObjectRecord(
        1,
        11,
        (1.0, 2.0, 3.0),
        90.0,
        (4.0, 5.0, 6.0),
        (36.0, -7.2, 3.6),
        (0.1, 0.2, 0.3),
    )
    second = VehicleCollisionObjectRecord(
        2,
        22,
        (7.0, 8.0, 9.0),
        -45.0,
        (10.0, 11.0, 12.0),
        (0.0, 18.0, -36.0),
        (0.4, 0.5, 0.6),
    )

    message = vehicle_collision_array_message(
        VehicleCollisionArrayRecord((VehicleCollisionRecord(first, second),)),
        "map",
        (6, 7),
    )

    assert isinstance(message, VehicleCollisionArray)
    assert (message.header.stamp.sec, message.header.stamp.nanosec) == (6, 7)
    assert message.collisions[0].first.heading == pytest.approx(math.pi / 2.0)
    assert (
        message.collisions[0].first.velocity.x,
        message.collisions[0].first.velocity.y,
        message.collisions[0].first.velocity.z,
    ) == pytest.approx((10.0, -2.0, 1.0))
    assert (
        message.collisions[0].second.acceleration.x,
        message.collisions[0].second.acceleration.y,
        message.collisions[0].second.acceleration.z,
    ) == pytest.approx((0.4, 0.5, 0.6))


def test_full_dev_ego_converter_uses_dev_interface_and_presence_flag() -> None:
    record = EgoStatusRecord(
        stamp=(12, 34),
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
        position=(1.0, 2.0, 3.0),
        rpy=(180.0, 0.0, 90.0),
        velocity=(3.6, 7.2, 10.8),
        angular_velocity=(180.0, 0.0, 90.0),
        acceleration=(0.1, 0.2, 0.3),
        steering=30.0,
        link_id="L17",
        tire_metrics=tuple(float(index) for index in range(1, 13)),
    )

    message = dev_ego_status_message(record, "map")

    assert isinstance(message, EgoVehicleStatus)
    assert message.has_tire_metrics is True
    assert message.signed_velocity == pytest.approx(10.0)
    assert message.rpy.z == pytest.approx(math.pi / 2.0)
    assert (message.velocity.x, message.velocity.y, message.velocity.z) == pytest.approx(
        (1.0, 2.0, 3.0)
    )
    assert message.tire_cornering_stiffness_rr == pytest.approx(12.0)


def test_full_lidar2d_converter_preserves_aux_ranges_and_intensities() -> None:
    record = Lidar2DRecord(
        aux=(1.25, 2.5, 3.75),
        distances_m=tuple(float(index) / 10.0 for index in range(360)),
        intensities=tuple(index % 256 for index in range(360)),
    )

    message = lidar2d_message(record, "lidar2d_link", (6, 7))

    assert isinstance(message, Lidar2D)
    assert tuple(message.aux) == pytest.approx(record.aux)
    assert tuple(message.distances_m) == pytest.approx(record.distances_m)
    assert tuple(message.intensities) == record.intensities
