"""Typed records produced by development MORAI UDP decoders."""

from dataclasses import dataclass


Vector3 = tuple[float, float, float]
Stamp = tuple[int, int]


@dataclass(frozen=True)
class ObjectRecord:
    unique_id: int
    object_type: int
    position: Vector3
    heading: float
    size: Vector3
    overhang: float
    wheelbase: float
    rear_overhang: float
    velocity: Vector3
    acceleration: Vector3
    link_id: str


@dataclass(frozen=True)
class ObjectArrayRecord:
    stamp: Stamp
    objects: tuple[ObjectRecord, ...]


@dataclass(frozen=True)
class Lidar2DRecord:
    aux: Vector3
    distances_m: tuple[float, ...]
    intensities: tuple[int, ...]


@dataclass(frozen=True)
class CameraBoundingBoxRecord:
    corners_3d: tuple[float, ...]
    bounding_box_2d: tuple[float, ...]
    group: int
    class_id: int
    subclass_id: int


@dataclass(frozen=True)
class CameraBoundingBoxArrayRecord:
    stamp: Stamp
    boxes: tuple[CameraBoundingBoxRecord, ...]


@dataclass(frozen=True)
class VehicleCollisionObjectRecord:
    object_type: int
    object_id: int
    position: Vector3
    heading: float
    size: Vector3
    velocity: Vector3
    acceleration: Vector3


@dataclass(frozen=True)
class VehicleCollisionRecord:
    first: VehicleCollisionObjectRecord
    second: VehicleCollisionObjectRecord


@dataclass(frozen=True)
class VehicleCollisionArrayRecord:
    collisions: tuple[VehicleCollisionRecord, ...]
