"""HEVEN z-up 3D GIoU geometry for the AB3DMOT-compatible tracking core.

Per `docs/research/tracking_architecture.md`'s "AB3DMOT Integration
Decisions" §1 (yaw convention): AB3DMOT's own `Box3D.box2corners3d_camcoord`
and `dist_metrics.py` IoU/GIoU functions are camera-frame-specific (KITTI
"right x, down y, front z", rotation about the down-pointing Y axis) and
must NOT be reused for HEVEN's z-up `lidar_link`/`odom` data. This module
implements the equivalent 3D GIoU computation natively in HEVEN's
convention (+x forward, +y left, +z up, yaw = CCW rotation about +z).

The overall GIoU_3D formula (BEV intersection * height overlap for
intersection volume, enclosing-hull BEV area * height range for the GIoU
hull term) mirrors `AB3DMOT_libs/dist_metrics.py::iou(..., metric='giou_3d')`
exactly (same structure, ported deliberately for traceability), only with
z-up corners instead of camera-frame ones. Two small 2D-polygon primitives
(`polygon_clip`, `polygon_area`) are frame-agnostic textbook algorithms
already present in that same reference file (Sutherland-Hodgman clipping,
shoelace formula) and are ported here verbatim for the same reason. The
convex-hull step AB3DMOT does with `scipy.spatial.ConvexHull` is replaced
with a small self-contained monotone-chain convex hull (`convex_hull_2d`)
to avoid adding a scipy dependency for this initial baseline; the clipped
intersection polygon from `polygon_clip` on two convex inputs is already
convex or empty, so no hull step is needed there (`polygon_area` is called
on it directly).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Box3D:
    """A HEVEN z-up box: center (x, y, z), yaw (CCW about +z), and l/w/h."""

    x: float
    y: float
    z: float
    yaw: float
    length: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.z, self.yaw, self.length, self.width, self.height)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Box3D fields must be finite")
        if self.length <= 0.0 or self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("Box3D dimensions must be positive")


def bev_corners(box: Box3D) -> np.ndarray:
    """Return the 4 bird's-eye-view corners of ``box`` in CCW order.

    Corners are the box's length/width footprint rotated by ``yaw`` about
    +z, in HEVEN's z-up convention (length along the box's local +x,
    width along local +y).
    """
    half_l, half_w = box.length / 2.0, box.width / 2.0
    local = np.array(
        [
            [half_l, half_w],
            [-half_l, half_w],
            [-half_l, -half_w],
            [half_l, -half_w],
        ]
    )
    cos_yaw, sin_yaw = math.cos(box.yaw), math.sin(box.yaw)
    rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    rotated = local @ rotation.T
    return rotated + np.array([box.x, box.y])


def height_range(box: Box3D) -> tuple[float, float]:
    """Return (z_min, z_max) of ``box``, assuming ``z`` is its geometric center."""
    half_h = box.height / 2.0
    return box.z - half_h, box.z + half_h


def polygon_area(points: np.ndarray) -> float:
    """Shoelace-formula area of an ordered (convex or simple) polygon.

    Ported from `AB3DMOT_libs/dist_metrics.py::PolyArea2D` — a frame-
    agnostic 2D primitive, unchanged from the reference.
    """
    if len(points) < 3:
        return 0.0
    pts = np.asarray(points, dtype=float)
    rolled = np.roll(pts, -1, axis=0)
    return float(abs(np.sum(pts[:, 0] * rolled[:, 1] - pts[:, 1] * rolled[:, 0])) * 0.5)


def polygon_clip(subject_polygon: np.ndarray, clip_polygon: np.ndarray) -> np.ndarray | None:
    """Sutherland-Hodgman clip of ``subject_polygon`` against convex ``clip_polygon``.

    Both polygons must be CCW-ordered lists of (x, y) points. Ported from
    `AB3DMOT_libs/dist_metrics.py::polygon_clip` — a frame-agnostic 2D
    primitive, algorithm unchanged from the reference (including returning
    ``None`` for an empty result). One deliberate safety deviation: the
    reference divides by a possibly-zero denominator (parallel clip edges)
    unguarded; this port treats that degenerate case as "no intersection"
    instead of raising `ZeroDivisionError`.
    """

    def inside(p, cp1, cp2):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def compute_intersection(cp1, cp2, s, e):
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        denom = dc[0] * dp[1] - dc[1] * dp[0]
        if denom == 0.0:
            return None
        n3 = 1.0 / denom
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    output_list = list(subject_polygon)
    cp1 = clip_polygon[-1]
    for clip_vertex in clip_polygon:
        cp2 = clip_vertex
        input_list = output_list
        output_list = []
        if not input_list:
            return None
        s = input_list[-1]
        for subject_vertex in input_list:
            e = subject_vertex
            if inside(e, cp1, cp2):
                if not inside(s, cp1, cp2):
                    intersection = compute_intersection(cp1, cp2, s, e)
                    if intersection is not None:
                        output_list.append(intersection)
                output_list.append(e)
            elif inside(s, cp1, cp2):
                intersection = compute_intersection(cp1, cp2, s, e)
                if intersection is not None:
                    output_list.append(intersection)
            s = e
        cp1 = cp2
        if len(output_list) == 0:
            return None
    return np.array(output_list)


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone-chain convex hull, CCW-ordered, no duplicate closing point.

    Self-contained replacement for `scipy.spatial.ConvexHull`, used only
    for the GIoU enclosing-hull term (two boxes' combined BEV corners are
    not already convex as one set, unlike a single clipped intersection
    polygon). Standard textbook algorithm, not part of the tracking
    logic itself.
    """
    pts = sorted({(float(x), float(y)) for x, y in points})
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def giou_3d(box_a: Box3D, box_b: Box3D) -> float:
    """3D generalized IoU between two HEVEN z-up boxes.

    Same formula structure as
    `AB3DMOT_libs/dist_metrics.py::iou(..., metric='giou_3d')`:
    intersection volume = BEV-intersection-area * height-overlap; union
    volume = vol_a + vol_b - intersection; the GIoU penalty term uses the
    BEV area of the convex hull enclosing both boxes' footprints, times
    the union height range.
    """
    corners_a, corners_b = bev_corners(box_a), bev_corners(box_b)

    clipped = polygon_clip(corners_a, corners_b)
    intersection_area = polygon_area(clipped) if clipped is not None else 0.0

    hull = convex_hull_2d(np.vstack([corners_a, corners_b]))
    hull_area = polygon_area(hull)

    a_min, a_max = height_range(box_a)
    b_min, b_max = height_range(box_b)
    overlap_height = max(0.0, min(a_max, b_max) - max(a_min, b_min))
    union_height = max(a_max, b_max) - min(a_min, b_min)

    intersection_volume = intersection_area * overlap_height
    volume_a = box_a.length * box_a.width * box_a.height
    volume_b = box_b.length * box_b.width * box_b.height
    union_volume = volume_a + volume_b - intersection_volume
    hull_volume = hull_area * union_height

    if union_volume <= 0.0 or hull_volume <= 0.0:
        return -1.0

    iou_3d = intersection_volume / union_volume
    return iou_3d - (hull_volume - union_volume) / hull_volume
