#!/usr/bin/env python3
"""Build a one-source, 16-ring AV2 sparsity proxy.

AV2 Sensor sweeps store ego-motion-compensated points in the egovehicle
frame.  Physical ring elevation is therefore estimated by undoing ego motion
at each point's acquisition time and then applying the inverse static source
LiDAR extrinsic.  The proxy only uses the recovered ring identity to select
points: output XYZ remains the original compensated egovehicle XYZ.

This is a source-ring sparsity proxy, not ray-traced VLP-16 simulation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from synthetic_xyzirc import XYZIRC_DTYPE


TARGET_ELEVATIONS_DEG = np.arange(-15.0, 16.0, 2.0, dtype=np.float64)
SOURCE_LIDAR_RANGES = {
    "up_lidar": (0, 32),
    "down_lidar": (32, 64),
}


@dataclass(frozen=True)
class RingGeometry:
    source_lidar: str
    local_ring_id: int
    point_count: int
    median_elevation_deg: float
    p05_elevation_deg: float
    p95_elevation_deg: float
    mad_elevation_deg: float


@dataclass(frozen=True)
class RingSelection:
    target_channel: int
    target_elevation_deg: float
    source_lidar: str
    source_laser_number: int
    source_local_ring_id: int
    measured_elevation_deg: float
    angular_error_deg: float


def quaternion_rotation_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Return a 3x3 active rotation matrix for a normalized wxyz quaternion."""
    quaternion = np.asarray((qw, qx, qy, qz), dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if norm == 0.0:
        raise ValueError("zero quaternion")
    w, x, y, z = quaternion / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def inverse_transform_points(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Apply inverse of ``target_SE3_source`` to row-vector target points."""
    points = np.asarray(points, dtype=np.float64)
    return (points - np.asarray(translation, dtype=np.float64)) @ np.asarray(rotation)


def transform_points(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Apply ``target_SE3_source`` to row-vector source points."""
    return np.asarray(points, dtype=np.float64) @ np.asarray(rotation).T + np.asarray(
        translation, dtype=np.float64
    )


def interpolate_ego_poses(
    pose_timestamps_ns: np.ndarray,
    pose_quaternions_wxyz: np.ndarray,
    pose_translations: np.ndarray,
    query_timestamps_ns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate city_SE3_egovehicle poses at integer nanosecond queries."""
    from scipy.spatial.transform import Rotation, Slerp

    pose_timestamps_ns = np.asarray(pose_timestamps_ns, dtype=np.int64)
    query_timestamps_ns = np.asarray(query_timestamps_ns, dtype=np.int64)
    if len(pose_timestamps_ns) < 2 or np.any(np.diff(pose_timestamps_ns) <= 0):
        raise ValueError("pose timestamps must contain at least two increasing samples")
    if query_timestamps_ns.size and (
        query_timestamps_ns.min() < pose_timestamps_ns[0]
        or query_timestamps_ns.max() > pose_timestamps_ns[-1]
    ):
        raise ValueError("point acquisition timestamp is outside pose history")

    base_ns = int(pose_timestamps_ns[0])
    pose_seconds = (pose_timestamps_ns - base_ns).astype(np.float64) * 1e-9
    query_seconds = (query_timestamps_ns - base_ns).astype(np.float64) * 1e-9
    # scipy uses xyzw while AV2 feather stores wxyz.
    xyzw = np.asarray(pose_quaternions_wxyz, dtype=np.float64)[:, (1, 2, 3, 0)]
    rotations = Slerp(pose_seconds, Rotation.from_quat(xyzw))(query_seconds).as_matrix()
    translations = np.column_stack(
        [
            np.interp(query_seconds, pose_seconds, np.asarray(pose_translations)[:, axis])
            for axis in range(3)
        ]
    )
    return rotations, translations


def reconstruct_source_local_points(
    ego_points_at_reference: np.ndarray,
    offset_ns: np.ndarray,
    reference_timestamp_ns: int,
    reference_rotation: np.ndarray,
    reference_translation: np.ndarray,
    pose_timestamps_ns: np.ndarray,
    pose_quaternions_wxyz: np.ndarray,
    pose_translations: np.ndarray,
    ego_rotation_sensor: np.ndarray,
    ego_translation_sensor: np.ndarray,
) -> np.ndarray:
    """Approximate original acquisition-time points in one source sensor frame.

    AV2's compensated ego points are first returned to city coordinates using
    the sweep-reference ego pose.  The interpolated acquisition-time ego pose
    then recovers ego coordinates at ``reference_timestamp + offset_ns``.
    Finally the inverse static source extrinsic maps those points to the source
    LiDAR frame.
    """
    points = np.asarray(ego_points_at_reference, dtype=np.float64)
    acquisition_ns = np.int64(reference_timestamp_ns) + np.asarray(offset_ns, dtype=np.int64)
    city_points = transform_points(points, reference_rotation, reference_translation)
    acquisition_rotations, acquisition_translations = interpolate_ego_poses(
        pose_timestamps_ns,
        pose_quaternions_wxyz,
        pose_translations,
        acquisition_ns,
    )
    ego_at_acquisition = np.einsum(
        "ni,nij->nj", city_points - acquisition_translations, acquisition_rotations
    )
    return inverse_transform_points(
        ego_at_acquisition, ego_rotation_sensor, ego_translation_sensor
    )


def elevation_degrees(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return np.degrees(np.arctan2(points[:, 2], np.hypot(points[:, 0], points[:, 1])))


def source_laser_numbers(source_lidar: str) -> range:
    if source_lidar not in SOURCE_LIDAR_RANGES:
        raise ValueError(f"unsupported source lidar {source_lidar!r}")
    start, stop = SOURCE_LIDAR_RANGES[source_lidar]
    return range(start, stop)


def summarize_ring_geometry(
    elevations_by_laser_number: Mapping[int, Sequence[float]], source_lidar: str
) -> list[RingGeometry]:
    start, _ = SOURCE_LIDAR_RANGES[source_lidar]
    result = []
    for laser_number in source_laser_numbers(source_lidar):
        values = np.asarray(elevations_by_laser_number.get(laser_number, ()), dtype=np.float64)
        if not len(values):
            raise ValueError(f"source ring {laser_number} has no observations")
        median = float(np.median(values))
        result.append(
            RingGeometry(
                source_lidar=source_lidar,
                local_ring_id=laser_number - start,
                point_count=len(values),
                median_elevation_deg=median,
                p05_elevation_deg=float(np.percentile(values, 5)),
                p95_elevation_deg=float(np.percentile(values, 95)),
                mad_elevation_deg=float(np.median(np.abs(values - median))),
            )
        )
    return result


def select_unique_monotonic_rings(
    geometry: Sequence[RingGeometry],
    targets_deg: np.ndarray = TARGET_ELEVATIONS_DEG,
) -> list[RingSelection]:
    """Minimum-total-error, order-preserving 16-of-N source ring assignment."""
    targets = np.asarray(targets_deg, dtype=np.float64)
    rings = sorted(geometry, key=lambda row: row.median_elevation_deg)
    if len(targets) != 16 or len(rings) < len(targets):
        raise ValueError("require 16 targets and at least 16 source rings")

    n_target, n_ring = len(targets), len(rings)
    cost = np.full((n_target + 1, n_ring + 1), np.inf)
    take = np.zeros((n_target + 1, n_ring + 1), dtype=bool)
    cost[0, :] = 0.0
    for target_index in range(1, n_target + 1):
        for ring_index in range(1, n_ring + 1):
            skip_cost = cost[target_index, ring_index - 1]
            take_cost = cost[target_index - 1, ring_index - 1] + abs(
                targets[target_index - 1] - rings[ring_index - 1].median_elevation_deg
            )
            if take_cost <= skip_cost:
                cost[target_index, ring_index] = take_cost
                take[target_index, ring_index] = True
            else:
                cost[target_index, ring_index] = skip_cost

    selected_indices = []
    target_index, ring_index = n_target, n_ring
    while target_index:
        if ring_index == 0:
            raise ValueError("could not assign all target rings")
        if take[target_index, ring_index]:
            selected_indices.append(ring_index - 1)
            target_index -= 1
        ring_index -= 1
    selected_indices.reverse()

    result = []
    start, _ = SOURCE_LIDAR_RANGES[rings[0].source_lidar]
    for channel, (target, selected_index) in enumerate(zip(targets, selected_indices)):
        ring = rings[selected_index]
        result.append(
            RingSelection(
                target_channel=channel,
                target_elevation_deg=float(target),
                source_lidar=ring.source_lidar,
                source_laser_number=start + ring.local_ring_id,
                source_local_ring_id=ring.local_ring_id,
                measured_elevation_deg=ring.median_elevation_deg,
                angular_error_deg=abs(float(target) - ring.median_elevation_deg),
            )
        )
    return result


def adapt_source_rings(
    native_cloud: np.ndarray,
    laser_numbers: np.ndarray,
    selection: Sequence[RingSelection],
    *,
    source_lidar: str = "up_lidar",
) -> np.ndarray:
    """Select one source LiDAR's physical rings and relabel them as channels 0..15."""
    if native_cloud.dtype != XYZIRC_DTYPE:
        raise ValueError("native_cloud must use canonical XYZIRC_DTYPE")
    laser_numbers = np.asarray(laser_numbers)
    if len(native_cloud) != len(laser_numbers):
        raise ValueError("native cloud and laser numbers must have equal length")
    if len(selection) != 16 or len({row.source_laser_number for row in selection}) != 16:
        raise ValueError("selection must contain 16 unique source rings")
    if any(row.source_lidar != source_lidar for row in selection):
        raise ValueError("cross-sensor ring selection is prohibited")

    channel_by_ring = {row.source_laser_number: row.target_channel for row in selection}
    selected_mask = np.isin(laser_numbers, np.asarray(tuple(channel_by_ring)))
    output = native_cloud[selected_mask].copy()
    selected_lasers = laser_numbers[selected_mask]
    output["channel"] = np.asarray(
        [channel_by_ring[int(value)] for value in selected_lasers], dtype=np.uint16
    )
    return output
