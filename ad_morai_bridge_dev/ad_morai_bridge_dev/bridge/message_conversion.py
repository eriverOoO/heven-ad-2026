"""Convert development MORAI protocol records into ROS messages."""

import math

from ad_morai_interfaces_dev.msg import (
    CameraBoundingBox,
    CameraBoundingBoxArray,
    EgoVehicleStatus,
    Lidar2D,
    ObjectStatus,
    ObjectStatusArray,
    VehicleCollision,
    VehicleCollisionArray,
    VehicleCollisionObject,
)
from sensor_msgs.msg import LaserScan

from ad_morai_bridge.message_conversion import time_message
from ad_morai_bridge.protocol_records import EgoStatusRecord

from ad_morai_bridge_dev.bridge.protocol_records import (
    CameraBoundingBoxArrayRecord,
    Lidar2DRecord,
    ObjectArrayRecord,
    VehicleCollisionArrayRecord,
    VehicleCollisionObjectRecord,
)


def _floats(values):
    return tuple(float(value) for value in values)


def object_array_message(
    record: ObjectArrayRecord,
    frame_id: str,
    header_stamp: tuple[int, int] | None = None,
) -> ObjectStatusArray:
    result = ObjectStatusArray()
    result.header.stamp = time_message(
        record.stamp if header_stamp is None else header_stamp
    )
    result.header.frame_id = frame_id
    for item in record.objects:
        message = ObjectStatus()
        message.unique_id = item.unique_id
        message.object_type = item.object_type
        message.position.x, message.position.y, message.position.z = _floats(
            item.position
        )
        message.heading = math.radians(float(item.heading))
        message.size.x, message.size.y, message.size.z = _floats(item.size)
        message.overhang = float(item.overhang)
        message.wheelbase = float(item.wheelbase)
        message.rear_overhang = float(item.rear_overhang)
        message.velocity.x, message.velocity.y, message.velocity.z = tuple(
            float(value) / 3.6 for value in item.velocity
        )
        (
            message.acceleration.x,
            message.acceleration.y,
            message.acceleration.z,
        ) = _floats(item.acceleration)
        message.link_id = item.link_id
        result.objects.append(message)
    return result


def laser_scan_message(
    record: Lidar2DRecord, frame_id: str, stamp: tuple[int, int]
) -> LaserScan:
    message = LaserScan()
    message.header.stamp = time_message(stamp)
    message.header.frame_id = frame_id
    message.angle_min = -math.pi
    message.angle_increment = 2.0 * math.pi / len(record.distances_m)
    message.angle_max = message.angle_min + message.angle_increment * (
        len(record.distances_m) - 1
    )
    message.range_min = 0.05
    message.range_max = 65.535
    message.ranges = list(record.distances_m)
    message.intensities = [float(value) for value in record.intensities]
    return message


def dev_ego_status_message(
    record: EgoStatusRecord,
    frame_id: str,
    header_stamp: tuple[int, int] | None = None,
) -> EgoVehicleStatus:
    message = EgoVehicleStatus()
    message.header.stamp = time_message(
        record.stamp if header_stamp is None else header_stamp
    )
    message.header.frame_id = frame_id
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
    message.rpy.x, message.rpy.y, message.rpy.z = tuple(
        math.radians(float(value)) for value in record.rpy
    )
    message.velocity.x, message.velocity.y, message.velocity.z = tuple(
        float(value) / 3.6 for value in record.velocity
    )
    (
        message.angular_velocity.x,
        message.angular_velocity.y,
        message.angular_velocity.z,
    ) = tuple(math.radians(float(value)) for value in record.angular_velocity)
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


def lidar2d_message(
    record: Lidar2DRecord,
    frame_id: str,
    arrival_stamp: tuple[int, int],
) -> Lidar2D:
    if len(record.aux) != 3:
        raise ValueError("2D LiDAR aux must contain exactly 3 values")
    if len(record.distances_m) != 360 or len(record.intensities) != 360:
        raise ValueError("2D LiDAR full arrays must contain exactly 360 values")
    message = Lidar2D()
    message.header.stamp = time_message(arrival_stamp)
    message.header.frame_id = frame_id
    message.aux = list(_floats(record.aux))
    message.distances_m = list(_floats(record.distances_m))
    message.intensities = list(record.intensities)
    return message


def camera_bounding_boxes_message(
    record: CameraBoundingBoxArrayRecord,
    frame_id: str,
    header_stamp: tuple[int, int] | None = None,
) -> CameraBoundingBoxArray:
    message = CameraBoundingBoxArray()
    message.header.stamp = time_message(
        record.stamp if header_stamp is None else header_stamp
    )
    message.header.frame_id = frame_id
    for item in record.boxes:
        box = CameraBoundingBox()
        box.corners_3d = list(_floats(item.corners_3d))
        box.bounding_box_2d = list(_floats(item.bounding_box_2d))
        box.group = item.group
        box.class_id = item.class_id
        box.subclass_id = item.subclass_id
        message.boxes.append(box)
    return message


def _collision_object_message(
    record: VehicleCollisionObjectRecord,
) -> VehicleCollisionObject:
    message = VehicleCollisionObject()
    message.object_type = record.object_type
    message.object_id = record.object_id
    message.position.x, message.position.y, message.position.z = _floats(
        record.position
    )
    message.heading = math.radians(float(record.heading))
    message.size.x, message.size.y, message.size.z = _floats(record.size)
    message.velocity.x, message.velocity.y, message.velocity.z = tuple(
        float(value) / 3.6 for value in record.velocity
    )
    (
        message.acceleration.x,
        message.acceleration.y,
        message.acceleration.z,
    ) = _floats(record.acceleration)
    return message


def vehicle_collision_array_message(
    record: VehicleCollisionArrayRecord,
    frame_id: str,
    arrival_stamp: tuple[int, int],
) -> VehicleCollisionArray:
    message = VehicleCollisionArray()
    message.header.stamp = time_message(arrival_stamp)
    message.header.frame_id = frame_id
    for item in record.collisions:
        collision = VehicleCollision()
        collision.first = _collision_object_message(item.first)
        collision.second = _collision_object_message(item.second)
        message.collisions.append(collision)
    return message
