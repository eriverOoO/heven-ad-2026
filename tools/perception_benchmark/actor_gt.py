"""MORAI actor extraction and lightweight TF graph operations."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable

from frame_alignment import NearestIndex, TimedSample


Point3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


@dataclass(frozen=True)
class RigidTransform:
    """Transform points from a child/source frame into a parent/target frame."""

    translation: Point3
    rotation: Quaternion

    def apply(self, point: Point3) -> Point3:
        rotated = _rotate(self.rotation, point)
        return tuple(rotated[index] + self.translation[index] for index in range(3))

    def inverse(self) -> "RigidTransform":
        inverse_rotation = _conjugate(_normalized(self.rotation))
        inverse_translation = _rotate(
            inverse_rotation, tuple(-value for value in self.translation)
        )
        return RigidTransform(inverse_translation, inverse_rotation)


@dataclass(frozen=True)
class TransformEdge:
    parent: str
    child: str
    transform: RigidTransform


def _normalized(quaternion: Quaternion) -> Quaternion:
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm == 0.0:
        raise ValueError("zero quaternion")
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
    q = _normalized(quaternion)
    result = _multiply(_multiply(q, (point[0], point[1], point[2], 0.0)), _conjugate(q))
    return (result[0], result[1], result[2])


def edge_from_message(transform: object) -> TransformEdge:
    value = getattr(transform, "transform")
    translation = getattr(value, "translation")
    rotation = getattr(value, "rotation")
    return TransformEdge(
        parent=str(getattr(getattr(transform, "header"), "frame_id")).lstrip("/"),
        child=str(getattr(transform, "child_frame_id")).lstrip("/"),
        transform=RigidTransform(
            (float(translation.x), float(translation.y), float(translation.z)),
            (float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)),
        ),
    )


class TransformHistory:
    def __init__(
        self,
        static_edges: Iterable[TransformEdge],
        dynamic_edges: dict[tuple[str, str], NearestIndex[TransformEdge]],
        identity_aliases: Iterable[tuple[str, str]],
    ) -> None:
        self.static_edges = list(static_edges)
        self.dynamic_edges = dynamic_edges
        self.identity_aliases = list(identity_aliases)

    def transform_point(
        self,
        point: Point3,
        source_frame: str,
        target_frame: str,
        source_ns: int,
        max_delta_ns: int,
    ) -> Point3 | None:
        source_frame = source_frame.lstrip("/")
        target_frame = target_frame.lstrip("/")
        graph: dict[str, list[tuple[str, RigidTransform]]] = {}

        def add(parent: str, child: str, transform: RigidTransform) -> None:
            graph.setdefault(child, []).append((parent, transform))
            graph.setdefault(parent, []).append((child, transform.inverse()))

        for edge in self.static_edges:
            add(edge.parent, edge.child, edge.transform)
        for index in self.dynamic_edges.values():
            sample = index.nearest(source_ns, max_delta_ns)
            if sample is not None:
                edge = sample.value
                add(edge.parent, edge.child, edge.transform)
        identity = RigidTransform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        for left, right in self.identity_aliases:
            add(left.lstrip("/"), right.lstrip("/"), identity)

        queue = deque([(source_frame, point)])
        visited = {source_frame}
        while queue:
            frame, current = queue.popleft()
            if frame == target_frame:
                return current
            for neighbor, transform in graph.get(frame, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, transform.apply(current)))
        return None


def build_dynamic_indices(
    samples: Iterable[TimedSample[TransformEdge]],
) -> dict[tuple[str, str], NearestIndex[TransformEdge]]:
    grouped: dict[tuple[str, str], list[TimedSample[TransformEdge]]] = {}
    for sample in samples:
        edge = sample.value
        grouped.setdefault((edge.parent, edge.child), []).append(sample)
    return {key: NearestIndex(values) for key, values in grouped.items()}
