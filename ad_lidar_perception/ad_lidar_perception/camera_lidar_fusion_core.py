"""POST-FREEZE EXTENSION 2, Stage 1: opt-in, EXPERIMENTAL, ROS-independent
late-fusion core for camera 2D semantic detections + LiDAR 3D detections.

Scope of this stage (deliberately minimal, per the extension brief):

    LiDAR 3D box  --project-->  2D rectangle in the camera image
    Camera 2D detection (class + confidence, e.g. from ad_camera_perception's
        COCO YOLO DynamicObstacleDetectorNode)
    --2D association (IoU or normalized-center-distance, Hungarian, gated)-->
    fused object: LiDAR geometry stays authoritative; camera contributes
        semantic class, class confidence, a visual-confirmation flag, and
        an association confidence.

This module does NOT:
  * modify or replace any LiDAR detector (Euclidean / CenterPoint),
  * train or run a multimodal neural network,
  * modify `ab3dmot_core.py`, `kalmannet_core.py`, or any production path,
  * import ROS, torch, or ultralytics at module level.

It reuses `ab3dmot_core._hungarian_matching` (the project's single existing
assignment solver) rather than adding a second matcher, per the brief.

Frame convention (HEVEN, matching `ab3dmot_geometry.py`): LiDAR `lidar_link`
is +x forward, +y left, +z up, yaw = CCW about +z. The camera optical frame
is the standard +x right, +y down, +z forward. The rigid transform between
them is supplied by the caller (from a real `/tf_static` chain) as a 4x4
homogeneous matrix; this module never invents an extrinsic.

Camera intrinsics carry an explicit `calibration_quality` tag
("A_exact" | "B_reconstructed" | "C_unknown"). For the known
`morai_cam4_20260813_163222` bag it is B_reconstructed: no CameraInfo was
recorded, so fx/fy/cx/cy are reconstructed from documented FOV + resolution
under a pinhole / zero-distortion assumption. Projection results from a
B_reconstructed model are for qualitative / mechanical inspection only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "SEMANTIC_CLASSES",
    "COCO_NAME_TO_SEMANTIC",
    "map_coco_name_to_semantic",
    "CameraIntrinsics",
    "CameraModel",
    "LidarDetection",
    "CameraDetection",
    "ProjectedBox",
    "FusedObject",
    "FusionResult",
    "FusionConfig",
    "rigid_transform_from_tf_chain",
    "project_lidar_box",
    "iou_2d",
    "normalized_center_distance",
    "associate_2d",
    "fuse",
]

_EPS_DEPTH_M = 0.05  # a corner is "in front of the camera" if optical z exceeds this

# ---------------------------------------------------------------------------
# Semantic classes
# ---------------------------------------------------------------------------

# Small internal semantic enum used across the fusion path. Kept intentionally
# short: it only needs to express what the camera can plausibly add to a LiDAR
# geometry box for downstream association / tracking work.
SEMANTIC_CLASSES = ("car", "truck", "bus", "motorcycle", "bicycle", "person", "unknown")

# COCO (Ultralytics yolo26s.pt) class name -> internal semantic class.
# COCO has no "van"; motorcycle/bicycle kept distinct; everything else -> ignored
# by `map_coco_name_to_semantic` returning None (the caller drops it).
COCO_NAME_TO_SEMANTIC = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "bicycle": "bicycle",
    "person": "person",
    "pedestrian": "person",
}


def map_coco_name_to_semantic(coco_name: str) -> Optional[str]:
    """Map an external (COCO) class name to an internal semantic class.

    Returns ``None`` for any class not in the vehicle/person set of interest
    (train, boat, traffic light, sports ball, ...), so the caller can drop it.
    """
    return COCO_NAME_TO_SEMANTIC.get(str(coco_name).strip().lower())


# ---------------------------------------------------------------------------
# Sensor model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics plus an explicit calibration-quality tag."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    calibration_quality: str = "C_unknown"
    distortion_model: str = "none"

    def __post_init__(self) -> None:
        for name in ("fx", "fy", "cx", "cy"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"CameraIntrinsics.{name} must be finite and positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("CameraIntrinsics image size must be positive")
        if self.calibration_quality not in ("A_exact", "B_reconstructed", "C_unknown"):
            raise ValueError("calibration_quality must be A_exact | B_reconstructed | C_unknown")

    @classmethod
    def from_fov(
        cls,
        width: int,
        height: int,
        horizontal_fov_deg: float,
        *,
        calibration_quality: str = "B_reconstructed",
    ) -> "CameraIntrinsics":
        """Reconstruct a square-pixel, zero-distortion pinhole model from FOV.

        This is the `morai_cam4_20260813_163222` case: no CameraInfo exists,
        so intrinsics come from documented resolution + HFOV. The default tag
        is therefore ``B_reconstructed`` on purpose.
        """
        fx = (width / 2.0) / math.tan(math.radians(horizontal_fov_deg) / 2.0)
        return cls(
            fx=fx, fy=fx, cx=width / 2.0, cy=height / 2.0,
            width=int(width), height=int(height),
            calibration_quality=calibration_quality,
        )


@dataclass(frozen=True)
class CameraModel:
    """A camera: intrinsics + the 4x4 transform mapping a homogeneous point
    in the LiDAR frame to the camera optical frame."""

    intrinsics: CameraIntrinsics
    optical_from_lidar: np.ndarray  # (4, 4)

    def __post_init__(self) -> None:
        matrix = np.asarray(self.optical_from_lidar, dtype=float)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError("optical_from_lidar must be a finite 4x4 matrix")
        object.__setattr__(self, "optical_from_lidar", matrix)

    def lidar_points_to_optical(self, points_lidar: np.ndarray) -> np.ndarray:
        """(N, 3) LiDAR-frame points -> (N, 3) camera-optical-frame points."""
        pts = np.asarray(points_lidar, dtype=float).reshape(-1, 3)
        homog = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
        return (homog @ self.optical_from_lidar.T)[:, :3]


def _quat_xyzw_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n == 0.0:
        raise ValueError("zero-norm quaternion")
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def _homog(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def rigid_transform_from_tf_chain(hops: Sequence[dict]) -> np.ndarray:
    """Compose an explicit chain of rigid hops into one 4x4 matrix mapping a
    point from the chain's input frame to its output frame.

    Each hop is ``{"translation": [x, y, z], "rotation_xyzw": [x, y, z, w],
    "invert": bool}`` and represents a ROS `/tf_static` ``parent -> child``
    entry (``p_parent = R @ p_child + t``):
      * ``invert=False``  -> this hop maps child-frame coords to parent-frame
        coords: ``p_out = R @ p_in + t``.
      * ``invert=True``   -> this hop maps parent-frame coords to child-frame
        coords: ``p_out = R^T @ (p_in - t)``.
    Hops compose left to right (first hop applied to the input point first).

    `morai_cam4` LiDAR -> front-camera-optical (the frames are siblings under
    `rear_axle_link`, so the chain mixes directions):
        [ lidar_link              (invert=False) ]   # lidar  -> rear_axle
        [ camera_front_link       (invert=True ) ]   # rear   -> camera_front_link
        [ camera_front_optical    (invert=True ) ]   # camlink-> optical
    """
    result = np.eye(4)
    for hop in hops:
        rotation = _quat_xyzw_to_rot(*[float(v) for v in hop["rotation_xyzw"]])
        translation = np.asarray([float(v) for v in hop["translation"]], dtype=float)
        if hop.get("invert", False):
            step = np.eye(4)
            step[:3, :3] = rotation.T
            step[:3, 3] = -rotation.T @ translation
        else:
            step = _homog(rotation, translation)
        result = step @ result
    return result


# ---------------------------------------------------------------------------
# Detections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LidarDetection:
    """A LiDAR 3D detection in `lidar_link`. Dimensions are NOT required to be
    positive here (unlike `Box3D`): the Euclidean detector legitimately emits
    near-degenerate 0.1 m clutter boxes on this bag, and the fusion core must
    represent and flag them, not reject them."""

    x: float
    y: float
    z: float
    length: float
    width: float
    height: float
    yaw: float = 0.0
    score: float = 1.0
    source_index: int = -1

    @property
    def is_degenerate_geometry(self) -> bool:
        return min(self.length, self.width, self.height) <= 0.15

    def corners_lidar(self) -> np.ndarray:
        """8 box corners in `lidar_link`, yaw applied about +z."""
        dx, dy, dz = self.length / 2.0, self.width / 2.0, self.height / 2.0
        local = np.array([
            [sx * dx, sy * dy, sz * dz]
            for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)
        ])
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return local @ rot.T + np.array([self.x, self.y, self.z])


@dataclass(frozen=True)
class CameraDetection:
    """A camera 2D detection in pixels, already mapped to an internal semantic
    class. `raw_class_name` keeps the external label for traceability."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    semantic_class: str
    raw_class_name: str = ""
    source_index: int = -1

    def __post_init__(self) -> None:
        if self.semantic_class not in SEMANTIC_CLASSES:
            raise ValueError(f"unknown semantic_class {self.semantic_class!r}")
        for name in ("x1", "y1", "x2", "y2", "confidence"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"CameraDetection.{name} must be finite")

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (min(self.x1, self.x2), min(self.y1, self.y2),
                max(self.x1, self.x2), max(self.y1, self.y2))

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass(frozen=True)
class ProjectedBox:
    """The 2D image rectangle a LiDAR box projects to, plus visibility flags."""

    u1: float
    v1: float
    u2: float
    v2: float
    num_corners_in_front: int
    min_depth_m: float
    projectable: bool          # >= 2 corners in front of the camera
    overlaps_image: bool       # clipped rectangle has non-empty area inside the image
    degenerate: bool           # projected side < min_projected_side_px or area < min_projected_area_px

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.u1, self.v1, self.u2, self.v2)

    @property
    def area(self) -> float:
        return max(0.0, self.u2 - self.u1) * max(0.0, self.v2 - self.v1)


@dataclass(frozen=True)
class FusionConfig:
    association_measure: str = "iou"          # "iou" | "center_distance"
    iou_gate: float = 0.10                    # minimum 2D IoU to accept a match
    center_distance_gate: float = 0.15        # max center dist / image diagonal to accept
    min_projected_side_px: float = 2.0
    min_projected_area_px: float = 16.0
    camera_min_confidence: float = 0.25
    camera_max_area_fraction: float = 0.5     # drop camera boxes bigger than this fraction of the image
    # Vertical region of interest (normalized 0..1 of image height): a camera
    # detection whose box CENTER falls outside [roi_top, roi_bottom] is dropped.
    # Default roi_bottom=0.83 removes the ego-vehicle hood band, which COCO YOLO
    # on the morai_cam4 capture persistently detects as a "car" (a real, recorded
    # Phase-3 finding). Mirrors ad_camera_perception's own crop-ROI approach.
    camera_roi_top: float = 0.0
    camera_roi_bottom: float = 0.83
    exclude_degenerate_lidar_from_projection: bool = False  # if True, degenerate LiDAR boxes are lidar_only

    def __post_init__(self) -> None:
        if self.association_measure not in ("iou", "center_distance"):
            raise ValueError("association_measure must be 'iou' or 'center_distance'")
        if not 0.0 <= self.camera_roi_top < self.camera_roi_bottom <= 1.0:
            raise ValueError("require 0 <= camera_roi_top < camera_roi_bottom <= 1")


@dataclass
class FusedObject:
    lidar: LidarDetection
    projected: ProjectedBox
    camera: Optional[CameraDetection]
    match_state: str                    # "fused" | "lidar_only" | "camera_only"
    association_measure: str
    association_score: float             # IoU (higher better) or center-distance similarity (1 - d/gate-ref)
    runner_up_score: float               # best score to a *different* camera det (ambiguity indicator)
    semantic_class: str                  # camera class if fused, else "unknown"
    semantic_confidence: float           # camera confidence if fused, else 0.0
    visual_confirmed: bool               # True iff fused with a positive-area image overlap


@dataclass
class FusionResult:
    fused: list                          # list[FusedObject], match_state == "fused"
    lidar_only: list                     # list[FusedObject], match_state == "lidar_only"
    camera_only: list                    # list[CameraDetection]
    stats: dict


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def project_lidar_box(detection: LidarDetection, camera: CameraModel, config: FusionConfig) -> ProjectedBox:
    """Project one LiDAR 3D box into the camera image as an axis-aligned 2D
    rectangle. Handles: box entirely / partly behind the camera, box partly
    outside the image, zero-area projection, non-finite inputs."""
    intr = camera.intrinsics
    try:
        corners_opt = camera.lidar_points_to_optical(detection.corners_lidar())
    except Exception:
        return ProjectedBox(0, 0, 0, 0, 0, math.inf, False, False, True)

    if not np.all(np.isfinite(corners_opt)):
        return ProjectedBox(0, 0, 0, 0, 0, math.inf, False, False, True)

    depths = corners_opt[:, 2]
    in_front = depths > _EPS_DEPTH_M
    n_front = int(in_front.sum())
    min_depth = float(depths[in_front].min()) if n_front else float(depths.max())

    if n_front < 2:
        return ProjectedBox(0, 0, 0, 0, n_front, min_depth, False, False, True)

    front = corners_opt[in_front]
    u = intr.fx * front[:, 0] / front[:, 2] + intr.cx
    v = intr.fy * front[:, 1] / front[:, 2] + intr.cy
    if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
        return ProjectedBox(0, 0, 0, 0, n_front, min_depth, False, False, True)

    u1, u2 = float(u.min()), float(u.max())
    v1, v2 = float(v.min()), float(v.max())

    cu1, cu2 = max(0.0, u1), min(float(intr.width), u2)
    cv1, cv2 = max(0.0, v1), min(float(intr.height), v2)
    overlaps = (cu2 - cu1) > 0.0 and (cv2 - cv1) > 0.0

    side = min(u2 - u1, v2 - v1)
    area = max(0.0, u2 - u1) * max(0.0, v2 - v1)
    degenerate = side < config.min_projected_side_px or area < config.min_projected_area_px

    return ProjectedBox(u1, v1, u2, v2, n_front, min_depth, True, overlaps, degenerate)


# ---------------------------------------------------------------------------
# 2D association measures
# ---------------------------------------------------------------------------


def _clip_box_to_image(box, width, height):
    x1, y1, x2, y2 = box
    return (max(0.0, min(float(width), x1)), max(0.0, min(float(height), y1)),
            max(0.0, min(float(width), x2)), max(0.0, min(float(height), y2)))


def iou_2d(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ax1, ax2 = min(ax1, ax2), max(ax1, ax2)
    ay1, ay2 = min(ay1, ay2), max(ay1, ay2)
    bx1, bx2 = min(bx1, bx2), max(bx1, bx2)
    by1, by2 = min(by1, by2), max(by1, by2)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def normalized_center_distance(box_a, box_b, image_diagonal_px: float) -> float:
    """Euclidean distance between the two box centers, normalized by the image
    diagonal. 0 = coincident centers, ~1 = opposite corners of the image."""
    ax = (box_a[0] + box_a[2]) / 2.0
    ay = (box_a[1] + box_a[3]) / 2.0
    bx = (box_b[0] + box_b[2]) / 2.0
    by = (box_b[1] + box_b[3]) / 2.0
    if image_diagonal_px <= 0.0:
        return math.inf
    return math.hypot(ax - bx, ay - by) / image_diagonal_px


# ---------------------------------------------------------------------------
# Association
# ---------------------------------------------------------------------------


def _score(proj_box, cam_box, measure, image_diagonal_px, center_gate):
    """Higher = better. For center_distance we return a bounded similarity
    ``max(0, 1 - d / center_gate)`` so the same Hungarian/gate machinery works
    for both measures."""
    if measure == "iou":
        return iou_2d(proj_box, cam_box)
    d = normalized_center_distance(proj_box, cam_box, image_diagonal_px)
    return max(0.0, 1.0 - d / center_gate) if center_gate > 0.0 else 0.0


def associate_2d(projected_boxes, camera_dets, camera_model, config):
    """One-to-one assignment between projected LiDAR rectangles and camera 2D
    boxes using `ab3dmot_core._hungarian_matching` on a negated score matrix,
    then gating. Returns (matches, unmatched_proj_idx, unmatched_cam_idx,
    score_matrix) where `matches` is a list of (proj_idx, cam_idx, score,
    runner_up_score)."""
    from ad_lidar_perception.ab3dmot_core import _hungarian_matching

    intr = camera_model.intrinsics
    diag = math.hypot(intr.width, intr.height)
    gate = config.iou_gate if config.association_measure == "iou" else config.center_distance_gate

    n_p, n_c = len(projected_boxes), len(camera_dets)
    score = np.zeros((n_p, n_c), dtype=float)
    for i, pb in enumerate(projected_boxes):
        pbox = _clip_box_to_image(pb.box, intr.width, intr.height)
        for j, cd in enumerate(camera_dets):
            score[i, j] = _score(pbox, cd.box, config.association_measure, diag,
                                 config.center_distance_gate)

    matched_proj: set[int] = set()
    matched_cam: set[int] = set()
    matches = []
    if n_p and n_c:
        for i, j in _hungarian_matching(-score):
            i, j = int(i), int(j)
            s = float(score[i, j])
            accept = s >= config.iou_gate if config.association_measure == "iou" else s > 0.0
            if not accept:
                continue
            row = score[i].copy()
            row[j] = -1.0
            runner_up = float(row.max()) if n_c > 1 else 0.0
            matches.append((i, j, s, runner_up))
            matched_proj.add(i)
            matched_cam.add(j)

    unmatched_proj = [i for i in range(n_p) if i not in matched_proj]
    unmatched_cam = [j for j in range(n_c) if j not in matched_cam]
    return matches, unmatched_proj, unmatched_cam, score


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def _prefilter_camera(camera_dets, camera_model, config):
    intr = camera_model.intrinsics
    image_area = float(intr.width * intr.height)
    roi_top_px = config.camera_roi_top * intr.height
    roi_bottom_px = config.camera_roi_bottom * intr.height
    kept, dropped = [], []
    for cd in camera_dets:
        _, y1, _, y2 = cd.box
        center_y = (y1 + y2) / 2.0
        if cd.confidence < config.camera_min_confidence:
            dropped.append((cd, "below_confidence"))
        elif image_area > 0 and cd.area / image_area > config.camera_max_area_fraction:
            dropped.append((cd, "too_large"))
        elif not (roi_top_px <= center_y <= roi_bottom_px):
            dropped.append((cd, "outside_vertical_roi_ego_hood"))
        else:
            kept.append(cd)
    return kept, dropped


def fuse(lidar_dets, camera_dets, camera_model, config=None):
    """Late-fuse LiDAR 3D detections with camera 2D semantic detections.

    LiDAR geometry (x/y/z, dimensions, yaw) is never overwritten. Camera
    contributes semantic_class / semantic_confidence / visual_confirmed /
    association_score on the fused objects only.
    """
    config = config or FusionConfig()
    kept_cam, dropped_cam = _prefilter_camera(camera_dets, camera_model, config)

    projected = [project_lidar_box(d, camera_model, config) for d in lidar_dets]

    # which LiDAR detections are eligible for a camera match this frame
    eligible_idx, ineligible_reason = [], {}
    for i, (d, pb) in enumerate(zip(lidar_dets, projected)):
        if config.exclude_degenerate_lidar_from_projection and d.is_degenerate_geometry:
            ineligible_reason[i] = "degenerate_lidar_geometry"
        elif not pb.projectable:
            ineligible_reason[i] = "behind_camera"
        elif not pb.overlaps_image:
            ineligible_reason[i] = "outside_image_fov"
        else:
            eligible_idx.append(i)

    elig_boxes = [projected[i] for i in eligible_idx]
    matches, unmatched_local, unmatched_cam_local, _ = associate_2d(
        elig_boxes, kept_cam, camera_model, config
    )

    matched_lidar = {}
    for local_i, cam_j, s, runner in matches:
        matched_lidar[eligible_idx[local_i]] = (kept_cam[cam_j], s, runner)

    fused, lidar_only = [], []
    for i, (d, pb) in enumerate(zip(lidar_dets, projected)):
        if i in matched_lidar:
            cam, s, runner = matched_lidar[i]
            fused.append(FusedObject(
                lidar=d, projected=pb, camera=cam, match_state="fused",
                association_measure=config.association_measure,
                association_score=s, runner_up_score=runner,
                semantic_class=cam.semantic_class, semantic_confidence=cam.confidence,
                visual_confirmed=pb.overlaps_image,
            ))
        else:
            lidar_only.append(FusedObject(
                lidar=d, projected=pb, camera=None, match_state="lidar_only",
                association_measure=config.association_measure,
                association_score=0.0, runner_up_score=0.0,
                semantic_class="unknown", semantic_confidence=0.0, visual_confirmed=False,
            ))

    camera_only = [kept_cam[j] for j in unmatched_cam_local]

    n_deg = sum(1 for d in lidar_dets if d.is_degenerate_geometry)
    stats = {
        "n_lidar": len(lidar_dets),
        "n_lidar_degenerate_geometry": n_deg,
        "n_camera_raw": len(camera_dets),
        "n_camera_kept": len(kept_cam),
        "n_camera_dropped": len(dropped_cam),
        "n_lidar_projectable": sum(1 for pb in projected if pb.projectable),
        "n_lidar_in_image_fov": sum(1 for pb in projected if pb.overlaps_image),
        "n_lidar_eligible": len(eligible_idx),
        "n_fused": len(fused),
        "n_lidar_only": len(lidar_only),
        "n_camera_only": len(camera_only),
        "match_rate_over_eligible": (len(fused) / len(eligible_idx)) if eligible_idx else 0.0,
        "match_rate_over_kept_camera": (len(fused) / len(kept_cam)) if kept_cam else 0.0,
        "association_measure": config.association_measure,
        "ineligible_reasons": ineligible_reason,
        "dropped_camera_reasons": [r for _, r in dropped_cam],
    }
    return FusionResult(fused=fused, lidar_only=lidar_only, camera_only=camera_only, stats=stats)
