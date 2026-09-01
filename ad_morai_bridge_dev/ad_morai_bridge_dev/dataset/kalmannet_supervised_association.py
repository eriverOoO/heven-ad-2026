"""Pure-NumPy supervised GT<->detection association for KalmanNet measurement
attachment.

This is **supervised label construction**, not runtime multi-object tracking.
Ground truth is used to decide which detector observation corresponds to which
GT actor so a KalmanNet training / validation dataset can be built. The runtime
system does NOT and could NOT perform this association - it has no ground
truth. Every attachment export records ``assignment_method: gt_supervised_*``.

Frame-local only: no future GT, no past/future detections, no track continuity
is used to decide a current frame's assignment.

No ROS dependency, no scipy. The oriented-BEV-IoU primitives are ported (with
attribution) from ``tools/centerpoint_offline/morai_evaluator.py`` - itself a
port of ``ad_lidar_perception/ad_lidar_perception/ab3dmot_geometry.py`` (an
AB3DMOT port). IoU here is a **diagnostic only**; assignment is driven by BEV
centre distance because the historical KalmanNet observation is a 2-D position
and the Euclidean detector supplies no yaw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

ASSOCIATION_METRIC = "bev_center_distance"
ASSIGNMENT_ALGORITHM = "hungarian_min_cost"

_LARGE = 1.0e12  # blocked pair sentinel; kept finite so the solver stays feasible


# --------------------------------------------------------------------------
# oriented-rectangle primitives (ported - see module docstring)
# --------------------------------------------------------------------------
def _bev_corners(box: np.ndarray) -> np.ndarray:
    """4 BEV corners (CCW) of ``[x, y, z, l, w, h, yaw]``; length along local +x."""
    half_l, half_w = box[3] / 2.0, box[4] / 2.0
    local = np.array(
        [
            [half_l, half_w],
            [-half_l, half_w],
            [-half_l, -half_w],
            [half_l, -half_w],
        ]
    )
    cos_yaw, sin_yaw = math.cos(box[6]), math.sin(box[6])
    rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    return local @ rotation.T + np.array([box[0], box[1]])


def _polygon_area(points: np.ndarray | None) -> float:
    if points is None or len(points) < 3:
        return 0.0
    pts = np.asarray(points, dtype=float)
    rolled = np.roll(pts, -1, axis=0)
    return float(abs(np.sum(pts[:, 0] * rolled[:, 1] - pts[:, 1] * rolled[:, 0])) * 0.5)


def _polygon_clip(subject: np.ndarray, clip: np.ndarray) -> np.ndarray | None:
    def inside(p, cp1, cp2):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def intersection(cp1, cp2, s, e):
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        denom = dc[0] * dp[1] - dc[1] * dp[0]
        if denom == 0.0:
            return None
        n3 = 1.0 / denom
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    output_list = list(subject)
    cp1 = clip[-1]
    for clip_vertex in clip:
        cp2 = clip_vertex
        input_list = output_list
        output_list = []
        if not input_list:
            return None
        s = input_list[-1]
        for e in input_list:
            if inside(e, cp1, cp2):
                if not inside(s, cp1, cp2):
                    point = intersection(cp1, cp2, s, e)
                    if point is not None:
                        output_list.append(point)
                output_list.append(e)
            elif inside(s, cp1, cp2):
                point = intersection(cp1, cp2, s, e)
                if point is not None:
                    output_list.append(point)
            s = e
        cp1 = cp2
        if not output_list:
            return None
    return np.array(output_list)


def bev_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Oriented BEV IoU of two ``[x, y, z, l, w, h, yaw]`` boxes (footprint only)."""
    a = np.asarray(box_a, dtype=float)
    b = np.asarray(box_b, dtype=float)
    clipped = _polygon_clip(_bev_corners(a), _bev_corners(b))
    inter = _polygon_area(clipped)
    if inter <= 0.0:
        return 0.0
    area_a = float(a[3] * a[4])
    area_b = float(b[3] * b[4])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


# --------------------------------------------------------------------------
# deterministic Hungarian (pure NumPy, O(n^3) augmenting-path Munkres)
# --------------------------------------------------------------------------
def hungarian_min_cost(cost: np.ndarray) -> list[tuple[int, int]]:
    """Minimum-cost perfect assignment on a rectangular non-negative matrix.

    Returns ``[(row, col), ...]`` of length ``min(n_rows, n_cols)``, sorted by
    row. Deterministic: ties resolve to the lowest column index. No scipy.
    """

    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError("cost must be 2-D")
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return []
    transposed = n_rows > n_cols
    matrix = cost.T.copy() if transposed else cost.copy()
    rows, cols = matrix.shape  # rows <= cols

    # potentials
    u = np.zeros(rows + 1)
    v = np.zeros(cols + 1)
    p = np.zeros(cols + 1, dtype=int)  # p[j] = row matched to column j (1-based; 0 = none)
    way = np.zeros(cols + 1, dtype=int)
    for i in range(1, rows + 1):
        p[0] = i
        j0 = 0
        minv = np.full(cols + 1, np.inf)
        used = np.zeros(cols + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = -1
            for j in range(1, cols + 1):
                if used[j]:
                    continue
                cur = matrix[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    pairs: list[tuple[int, int]] = []
    for j in range(1, cols + 1):
        if p[j] != 0:
            r = p[j] - 1
            c = j - 1
            pairs.append((c, r) if transposed else (r, c))
    pairs.sort()
    return pairs


# --------------------------------------------------------------------------
# frame-local supervised association
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class GtActor:
    actor_id: int
    class_name: str
    x: float
    y: float
    box_map: tuple[float, float, float, float, float, float, float] | None = None


@dataclass(frozen=True)
class Detection:
    index: int
    class_name: str
    x: float
    y: float
    score: float
    box_map: tuple[float, float, float, float, float, float, float] | None = None
    box_lidar: tuple[float, float, float, float, float, float, float] | None = None
    yaw_available: bool = False


@dataclass
class Match:
    actor_id: int
    detection_index: int
    distance_m: float
    iou: float | None


@dataclass
class FrameAssociation:
    matches: list[Match] = field(default_factory=list)
    unmatched_actor_ids: list[int] = field(default_factory=list)
    unmatched_detection_indices: list[int] = field(default_factory=list)
    gated_out_pairs: int = 0

    @property
    def matched_actor_ids(self) -> set[int]:
        return {m.actor_id for m in self.matches}

    def match_for_actor(self, actor_id: int) -> Match | None:
        for m in self.matches:
            if m.actor_id == actor_id:
                return m
        return None


def associate_frame(
    gt_actors: Sequence[GtActor],
    detections: Sequence[Detection],
    *,
    max_distance_m: float,
    class_matching: bool,
) -> FrameAssociation:
    """One-to-one supervised assignment for a single frame.

    A pair is a *candidate* only when (optionally) the classes are compatible
    and the BEV centre distance is <= ``max_distance_m``. Hungarian minimises
    total candidate distance; a non-candidate can never be forced through.
    """

    result = FrameAssociation(
        unmatched_actor_ids=[a.actor_id for a in gt_actors],
        unmatched_detection_indices=[d.index for d in detections],
    )
    if not gt_actors or not detections:
        return result

    n_gt = len(gt_actors)
    n_det = len(detections)
    cost = np.full((n_gt, n_det), _LARGE, dtype=np.float64)
    candidate = np.zeros((n_gt, n_det), dtype=bool)
    for i, actor in enumerate(gt_actors):
        for j, det in enumerate(detections):
            if class_matching and not _class_compatible(actor.class_name, det.class_name):
                continue
            dist = math.hypot(actor.x - det.x, actor.y - det.y)
            if dist > max_distance_m:
                result.gated_out_pairs += 1
                continue
            candidate[i, j] = True
            cost[i, j] = dist

    assignment = hungarian_min_cost(cost)
    matched_actor_idx: set[int] = set()
    matched_det_idx: set[int] = set()
    for i, j in assignment:
        if not candidate[i, j]:
            continue
        actor = gt_actors[i]
        det = detections[j]
        iou = None
        if (
            det.yaw_available
            and det.box_map is not None
            and actor.box_map is not None
        ):
            iou = float(bev_iou(actor.box_map, det.box_map))
        result.matches.append(
            Match(
                actor_id=actor.actor_id,
                detection_index=det.index,
                distance_m=float(cost[i, j]),
                iou=iou,
            )
        )
        matched_actor_idx.add(i)
        matched_det_idx.add(j)

    result.matches.sort(key=lambda m: m.actor_id)
    result.unmatched_actor_ids = [
        a.actor_id for k, a in enumerate(gt_actors) if k not in matched_actor_idx
    ]
    result.unmatched_detection_indices = [
        d.index for k, d in enumerate(detections) if k not in matched_det_idx
    ]
    return result


_VEHICLE_LIKE = {"vehicle", "car", "truck", "bus"}


def _class_compatible(gt_class: str, det_class: str) -> bool:
    if gt_class == det_class:
        return True
    if gt_class in _VEHICLE_LIKE and det_class in _VEHICLE_LIKE:
        return True
    return False
