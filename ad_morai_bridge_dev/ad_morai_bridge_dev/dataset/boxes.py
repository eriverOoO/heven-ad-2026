"""Actor ground truth -> canonical 3D box records.

The centre policy here is a direct port of the audited MORAI dataset
exporter (``tools/morai_dataset_exporter/export_morai_dataset.py::
_transform_box``): the simulator reports a vehicle pose at the rear-axle
centre on the ground, so the box centre shifts forward by
``(wheelbase + overhang - rear_overhang) / 2`` and up by ``height / 2``;
pedestrians and obstacles report a ground-centred origin and shift up
only. A parity test locks this against the exporter. No arbitrary offset
is introduced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ad_morai_bridge_dev.dataset.geometry import RigidTransform, normalize_angle
from ad_morai_bridge_dev.dataset.schema import (
    CLASS_UNKNOWN,
    RAW_TYPE_TO_CLASS,
)

VEHICLE_LENGTH_TOLERANCE_M = 0.25


@dataclass(frozen=True)
class ActorGT:
    unique_id: int
    object_type: int
    position: tuple[float, float, float]
    heading_rad: float
    size: tuple[float, float, float]
    velocity: tuple[float, float, float]
    overhang: float
    wheelbase: float
    rear_overhang: float
    link_id: str = ""


def actor_from_message(item: Any) -> ActorGT:
    return ActorGT(
        unique_id=int(item.unique_id),
        object_type=int(item.object_type),
        position=(float(item.position.x), float(item.position.y), float(item.position.z)),
        heading_rad=float(item.heading),
        size=(float(item.size.x), float(item.size.y), float(item.size.z)),
        velocity=(
            float(item.velocity.x),
            float(item.velocity.y),
            float(item.velocity.z),
        ),
        overhang=float(item.overhang),
        wheelbase=float(item.wheelbase),
        rear_overhang=float(item.rear_overhang),
        link_id=str(getattr(item, "link_id", "")),
    )


def _finite(*values: float) -> bool:
    return all(math.isfinite(v) for v in values)


def canonical_box(
    actor: ActorGT, map_to_lidar: RigidTransform | None
) -> dict[str, Any]:
    """Build one canonical GT record.

    Always carries the raw source fields and the map-frame box. The
    ``lidar_frame`` box is only added when ``map_to_lidar`` is supplied and
    every geometry check passes; individual failures flag the actor rather
    than raising.
    """

    class_name = RAW_TYPE_TO_CLASS.get(actor.object_type, CLASS_UNKNOWN)
    length, width, height = actor.size
    flags: list[str] = []

    pose_ok = _finite(*actor.position, actor.heading_rad, *actor.velocity)
    dims_ok = _finite(length, width, height) and min(length, width, height) > 0.0
    if not pose_ok:
        flags.append("nonfinite_pose")
    if not dims_ok:
        flags.append("invalid_dimensions")

    forward_offset = 0.0
    if class_name == "vehicle":
        geom = (actor.overhang, actor.wheelbase, actor.rear_overhang)
        if not (_finite(*geom) and all(v >= 0.0 for v in geom)):
            flags.append("invalid_vehicle_geometry")
        elif dims_ok and abs(sum(geom) - length) > VEHICLE_LENGTH_TOLERANCE_M:
            flags.append("vehicle_length_geometry_mismatch")
        else:
            forward_offset = (
                actor.wheelbase + actor.overhang - actor.rear_overhang
            ) / 2.0

    center_map: tuple[float, float, float] | None = None
    if pose_ok and dims_ok:
        center_map = (
            actor.position[0] + forward_offset * math.cos(actor.heading_rad),
            actor.position[1] + forward_offset * math.sin(actor.heading_rad),
            actor.position[2] + height / 2.0,
        )

    record: dict[str, Any] = {
        "actor_id": actor.unique_id,
        "raw_object_type": actor.object_type,
        "class_name": class_name,
        "class_mapped": class_name != CLASS_UNKNOWN,
        "source": {
            "position_map": list(actor.position),
            "heading_rad": actor.heading_rad,
            "size_lwh": [length, width, height],
            "velocity_map": list(actor.velocity),
            "overhang": actor.overhang,
            "wheelbase": actor.wheelbase,
            "rear_overhang": actor.rear_overhang,
            "link_id": actor.link_id,
        },
        "center_policy": (
            "rear_axle_ground_to_box_center"
            if class_name == "vehicle"
            else "ground_center_to_box_center"
        ),
        "forward_center_offset_m": forward_offset,
        "flags": flags,
    }

    if center_map is not None:
        record["map_frame"] = {
            "center": list(center_map),
            "length": length,
            "width": width,
            "height": height,
            "yaw": normalize_angle(actor.heading_rad),
        }
        if map_to_lidar is not None:
            center_lidar = map_to_lidar.apply(center_map)
            forward_lidar = map_to_lidar.apply(
                (
                    center_map[0] + math.cos(actor.heading_rad),
                    center_map[1] + math.sin(actor.heading_rad),
                    center_map[2],
                )
            )
            yaw = normalize_angle(
                math.atan2(
                    forward_lidar[1] - center_lidar[1],
                    forward_lidar[0] - center_lidar[0],
                )
            )
            record["lidar_frame"] = {
                "center": list(center_lidar),
                "length": length,
                "width": width,
                "height": height,
                "yaw": yaw,
            }

    geometry_ok = (
        pose_ok
        and dims_ok
        and not flags
        and class_name != CLASS_UNKNOWN
        and "lidar_frame" in record
    )
    record["valid_for_tracking_gt"] = bool(
        pose_ok and dims_ok and "map_frame" in record
    )
    record["valid_for_detection_gt"] = bool(geometry_ok)
    return record
