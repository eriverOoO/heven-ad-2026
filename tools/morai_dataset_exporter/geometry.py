"""Rigid-transform math for the MORAI dataset exporter."""

from __future__ import annotations

import math
from dataclasses import dataclass


Point3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> Quaternion:
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _normalize(quaternion: Quaternion) -> Quaternion:
    norm = math.sqrt(sum(value * value for value in quaternion))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("invalid quaternion")
    return tuple(value / norm for value in quaternion)  # type: ignore[return-value]


def _conjugate(quaternion: Quaternion) -> Quaternion:
    return (-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3])


def _multiply(left: Quaternion, right: Quaternion) -> Quaternion:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _rotate(quaternion: Quaternion, point: Point3) -> Point3:
    normalized = _normalize(quaternion)
    value = _multiply(
        _multiply(normalized, (point[0], point[1], point[2], 0.0)),
        _conjugate(normalized),
    )
    return (value[0], value[1], value[2])


@dataclass(frozen=True)
class RigidTransform:
    """A target-from-source rigid transform."""

    translation: Point3
    rotation: Quaternion

    def apply(self, point: Point3) -> Point3:
        rotated = _rotate(self.rotation, point)
        return tuple(rotated[index] + self.translation[index] for index in range(3))

    def inverse(self) -> "RigidTransform":
        rotation = _conjugate(_normalize(self.rotation))
        translation = _rotate(rotation, tuple(-value for value in self.translation))
        return RigidTransform(translation, rotation)

    def compose(self, source_transform: "RigidTransform") -> "RigidTransform":
        """Compose target<-middle (self) with middle<-source."""
        return RigidTransform(
            self.apply(source_transform.translation),
            _normalize(_multiply(self.rotation, source_transform.rotation)),
        )


def transform_from_message(message: object) -> RigidTransform:
    transform = getattr(message, "transform")
    translation = transform.translation
    rotation = transform.rotation
    return RigidTransform(
        (float(translation.x), float(translation.y), float(translation.z)),
        (float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)),
    )
