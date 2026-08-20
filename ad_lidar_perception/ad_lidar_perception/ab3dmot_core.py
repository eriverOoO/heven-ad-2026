"""HEVEN-compatible AB3DMOT tracking core (T-1A: no ROS2 integration yet).

Implements the resolved design in
`docs/research/tracking_architecture.md` ("AB3DMOT Integration
Decisions"): reuses AB3DMOT's own `filterpy`-based 10-state/7-measurement
constant-velocity Kalman filter (`AB3DMOT_libs.kalman_filter.KF`, imported
directly from the pinned `references/ab3dmot` submodule, not forked into
HEVEN) as a frame-agnostic periodic-state filter, fed HEVEN's own z-up yaw
directly (no camera-frame conversion — see §1 of that doc). Association
cost uses HEVEN-native 3D GIoU (`ab3dmot_geometry.giou_3d`), not AB3DMOT's
camera-frame corner geometry. The transition matrix is rebuilt with the
real elapsed time (seconds) before every predict step, matching
SimpleTrack's documented deviation from AB3DMOT's implicit dt=1-per-frame
assumption (§4 of the integration-decisions doc) — this keeps the KF's
velocity state in true m/s at all times, not meters-per-frame.

Two small free functions (`_orientation_correction`, `_greedy_matching`)
are faithful ports of `AB3DMOT_libs/model.py::orientation_correction` and
`AB3DMOT_libs/matching.py::greedy_matching` — both are pure, frame-
agnostic algorithms with no camera-frame coupling, ported rather than
imported to avoid pulling in `AB3DMOT_libs.matching`'s transitive
`numba`/`scipy` dependencies (only needed there for the camera-frame IoU
path this module does not use).

T-3 adds an opt-in `AB3DMOTConfig.matcher="hungarian"` alternative
(`_hungarian_matching`, `scipy.optimize.linear_sum_assignment`) alongside
the default `"greedy"` — both consume the identical GIoU cost matrix and
the identical post-hoc `giou_gate` filter in `_associate`; nothing else
(GIoU construction, KF, lifecycle) changes with the matcher choice.

T-4 adds two opt-in `AB3DMOTConfig.association_metric` alternatives
alongside the default `"giou_3d"`: `"euclidean"` (BEV `[x, y]` center
distance, gated in meters) and `"mahalanobis"` (BEV innovation distance
using the track's own predicted covariance, gated by a chi-square
threshold). All three metrics share the identical solver dispatch
(`config.matcher`) and the identical "solve over the full matrix, then
filter accepted pairs" structure — only the pairwise matrix construction
and the accept-predicate differ per metric. Mahalanobis reuses the
reference submodule's own `KF.compute_innovation_matrix()`
(`S = H P H^T + R`) rather than re-deriving it.
"""

from __future__ import annotations

import importlib
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ad_lidar_perception.ab3dmot_config import AB3DMOTConfig
from ad_lidar_perception.ab3dmot_geometry import Box3D, giou_3d

_DEFAULT_AB3DMOT_ROOT = Path(__file__).resolve().parents[2] / "references" / "ab3dmot"


def _load_ab3dmot_kf_class(ab3dmot_root: Path | None = None) -> type:
    """Import and return `AB3DMOT_libs.kalman_filter.KF` from the pinned submodule.

    `ab3dmot_root` defaults to the source-tree-relative `references/ab3dmot`
    (correct for running T-1A's own tests from a source checkout). A ROS2
    node (T-1B) should pass this explicitly, mirroring how
    `centerpoint_detector_node.py` takes `openpcdet_root` as a parameter
    rather than assuming a relative path survives a colcon install.
    """
    root = Path(ab3dmot_root) if ab3dmot_root is not None else _DEFAULT_AB3DMOT_ROOT
    marker = root / "AB3DMOT_libs" / "kalman_filter.py"
    if not marker.is_file():
        raise FileNotFoundError(f"AB3DMOT submodule not found at {root} (missing {marker})")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    module = importlib.import_module("AB3DMOT_libs.kalman_filter")
    return module.KF


def _wrap_to_pi(angle: float) -> float:
    """Wrap ``angle`` into [-pi, pi), one 2*pi step -- ported from
    `AB3DMOT_libs/model.py::within_range` (single-step wrap, not a loop,
    matching the reference exactly)."""
    if angle >= math.pi:
        angle -= 2.0 * math.pi
    if angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _orientation_correction(theta_pre: float, theta_obs: float) -> tuple[float, float]:
    """Resolve a track's propagated yaw to the closest 180-degree-symmetric
    option before filtering against a new measurement's yaw.

    Faithful port of `AB3DMOT_libs/model.py::orientation_correction`.
    """
    theta_pre = _wrap_to_pi(theta_pre)
    theta_obs = _wrap_to_pi(theta_obs)
    if math.pi / 2.0 < abs(theta_obs - theta_pre) < math.pi * 3.0 / 2.0:
        theta_pre += math.pi
        theta_pre = _wrap_to_pi(theta_pre)
    if abs(theta_obs - theta_pre) >= math.pi * 3.0 / 2.0:
        if theta_obs > 0:
            theta_pre += math.pi * 2.0
        else:
            theta_pre -= math.pi * 2.0
    return theta_pre, theta_obs


def _hungarian_matching(cost_matrix: np.ndarray) -> np.ndarray:
    """Global-optimal bipartite assignment via `scipy.optimize.linear_sum_assignment`.

    ``cost_matrix`` is ``[num_dets, num_trks]``, lower = better -- same
    convention `_greedy_matching` consumes (the caller negates GIoU once,
    at the `_associate` call site, for both matchers identically). Returns
    an (N, 2) array of ``[det_index, trk_index]`` pairs, one per row/column
    of whichever axis is smaller (`linear_sum_assignment` always returns a
    complete matching over the smaller dimension) -- gating out invalid
    (below-`giou_gate`) pairs is the caller's job, same as for greedy;
    this function has no cost cutoff, matching `_greedy_matching`'s own
    contract exactly.
    """
    from scipy.optimize import linear_sum_assignment

    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int)
    det_indices, trk_indices = linear_sum_assignment(cost_matrix)
    return np.stack([det_indices, trk_indices], axis=1).astype(int)


def _greedy_matching(cost_matrix: np.ndarray) -> np.ndarray:
    """Greedy bipartite assignment: claim globally-lowest-cost pairs first.

    ``cost_matrix`` is [num_dets, num_trks], lower = better. Returns an
    (N, 2) array of [det_index, trk_index] pairs. Faithful port of
    `AB3DMOT_libs/matching.py::greedy_matching`.
    """
    num_dets, num_trks = cost_matrix.shape
    distance_1d = cost_matrix.reshape(-1)
    index_1d = np.argsort(distance_1d)
    index_2d = np.stack([index_1d // num_trks, index_1d % num_trks], axis=1)

    det_matched = [-1] * num_dets
    trk_matched = [-1] * num_trks
    matched_indices = []
    for det_id, trk_id in index_2d:
        det_id, trk_id = int(det_id), int(trk_id)
        if trk_matched[trk_id] == -1 and det_matched[det_id] == -1:
            trk_matched[trk_id] = det_id
            det_matched[det_id] = trk_id
            matched_indices.append([det_id, trk_id])
    if not matched_indices:
        return np.empty((0, 2), dtype=int)
    return np.asarray(matched_indices, dtype=int)


def _euclidean_bev_distance_matrix(detections: Sequence["Detection"], tracks: Sequence["Track"]) -> np.ndarray:
    """`[n_dets, n_trks]` BEV center-distance matrix, meters.

    ``d_E = sqrt((x_d - x_t)^2 + (y_d - y_t)^2)`` -- planar `[x, y]` only
    (T-4 Phase 3): box size and yaw are never read here, unlike GIoU.
    """
    matrix = np.empty((len(detections), len(tracks)))
    for d_index, detection in enumerate(detections):
        det_xy = np.array([detection.x, detection.y])
        for t_index, track in enumerate(tracks):
            trk_xy = track.predicted_bev_position()
            matrix[d_index, t_index] = float(np.linalg.norm(det_xy - trk_xy))
    return matrix


# T-4 Phase 4: a numerically invalid Mahalanobis pair (non-finite/singular
# `S`) is marked with this large-but-finite sentinel rather than `np.inf`.
# `scipy.optimize.linear_sum_assignment` treats `np.inf` as a forbidden
# edge and raises `ValueError` if any row/column has no finite option left
# (a real risk here, since a just-born track's `S` can be singular) --
# using a large finite value keeps the solver always feasible while still
# deterministically failing every gate (`d_M^2 <= mahalanobis_gate`,
# default ~11.62) by many orders of magnitude, so an invalid pair is never
# actually accepted by either matcher.
_MAHALANOBIS_INVALID_SENTINEL = 1.0e6


def _mahalanobis_bev_distance_matrix(detections: Sequence["Detection"], tracks: Sequence["Track"]) -> np.ndarray:
    """`[n_dets, n_trks]` squared Mahalanobis innovation-distance matrix.

    For each (detection, track) pair: ``y = z_xy - H x_xy`` (BEV position
    residual), ``S`` = the track's own predicted BEV innovation covariance
    (`Track.predicted_bev_innovation_covariance`, `H P H^T + R`), and
    ``d_M^2 = y^T S^-1 y`` -- solved via `np.linalg.solve` (Phase 4:
    "prefer solving the linear system rather than explicitly computing an
    inverse") rather than `np.linalg.inv`. A pair whose `S` is
    non-finite, non-symmetric beyond float tolerance, or singular/
    ill-conditioned (`LinAlgError`) is marked with
    `_MAHALANOBIS_INVALID_SENTINEL` -- deterministically excluded by every
    downstream gate rather than raising or silently producing a wrong
    finite value.
    """
    matrix = np.full((len(detections), len(tracks)), _MAHALANOBIS_INVALID_SENTINEL)
    for d_index, detection in enumerate(detections):
        det_xy = np.array([detection.x, detection.y])
        for t_index, track in enumerate(tracks):
            trk_xy = track.predicted_bev_position()
            innovation_cov = track.predicted_bev_innovation_covariance()
            if not np.all(np.isfinite(innovation_cov)):
                continue
            if not np.allclose(innovation_cov, innovation_cov.T, atol=1e-6):
                continue
            residual = det_xy - trk_xy
            try:
                solved = np.linalg.solve(innovation_cov, residual)
            except np.linalg.LinAlgError:
                continue
            distance_sq = float(residual @ solved)
            if not math.isfinite(distance_sq) or distance_sq < 0.0:
                continue
            matrix[d_index, t_index] = distance_sq
    return matrix


@dataclass(frozen=True)
class Detection:
    """One HEVEN z-up detection box, as would come from `DetectedObjects`.

    ``label``/``label_probability``/``existence_probability`` are opaque
    passthrough metadata -- never read by the KF, association, or lifecycle
    math below. They exist to carry a detection's classification and score
    onto its matched `Track`/`TrackedState`, mirroring AB3DMOT's own
    reference `Tracker` class, which carries an analogous per-detection
    ``info`` payload onto its matched track's ``self.info`` on every
    *matched* update only (`AB3DMOT_libs/model.py::update`,
    ``trk.info = info[d, :][0]``) and leaves it unchanged on coast frames --
    the same semantics `Track.update` below implements.
    """

    x: float
    y: float
    z: float
    yaw: float
    length: float
    width: float
    height: float
    label: int = 0
    label_probability: float = 0.0
    existence_probability: float = 0.0

    def as_box3d(self) -> Box3D:
        return Box3D(self.x, self.y, self.z, self.yaw, self.length, self.width, self.height)

    def as_measurement_array(self) -> np.ndarray:
        """AB3DMOT's own 7-dim measurement order: [x, y, z, theta, l, w, h]."""
        return np.array(
            [self.x, self.y, self.z, self.yaw, self.length, self.width, self.height],
            dtype=float,
        )


@dataclass
class TrackedState:
    """One published track's current state -- the T-1A-level equivalent of
    a single `TrackedObjects.objects[i]` entry (ROS conversion is T-1B)."""

    track_id: int
    x: float
    y: float
    z: float
    yaw: float
    length: float
    width: float
    height: float
    vx_mps: float
    vy_mps: float
    vz_mps: float
    position_covariance: np.ndarray  # 3x3, from KF P[0:3, 0:3]
    yaw_variance: float  # from KF P[3, 3]
    velocity_covariance: np.ndarray  # 3x3, from KF P[7:10, 7:10]
    hits: int
    time_since_update: int
    label: int  # passthrough, see Detection docstring
    label_probability: float
    existence_probability: float


class LinearKFEstimator:
    """T-7A: the pre-existing AB3DMOT-reference constant-velocity filter
    (unchanged since T-1A), wrapped behind the common estimator interface
    (`predict`/`update`/`position`/`velocity`/`yaw`/`dimensions`/
    `position_covariance`/`yaw_variance`/`velocity_covariance`/
    `predicted_bev_position`/`predicted_bev_innovation_covariance`/
    `is_finite`) so `Track` no longer hard-codes one estimator.

    State order (unchanged): `[x, y, z, yaw, l, w, h, vx, vy, vz]`. All
    behavior below is copied byte-for-byte from the pre-T-7A `Track`
    class -- see `linear_kf_audit.md` for the full audited description;
    this is purely a structural move, not a logic change.
    """

    def __init__(
        self,
        detection: Detection,
        kf_class: type,
        track_id: int,
        yaw_measurement_mode: str = "detector",
    ) -> None:
        self._kf_wrapper = kf_class(detection.as_measurement_array(), np.zeros(1), track_id)
        self.yaw_measurement_mode = yaw_measurement_mode

    @property
    def _kf(self):
        return self._kf_wrapper.kf  # the underlying filterpy.kalman.KalmanFilter

    def predict(self, dt_seconds: float) -> None:
        if not math.isfinite(dt_seconds) or dt_seconds <= 0.0:
            raise ValueError(f"predict() requires a finite, positive dt_seconds, got {dt_seconds!r}")
        transition = np.eye(10)
        transition[0, 7] = dt_seconds
        transition[1, 8] = dt_seconds
        transition[2, 9] = dt_seconds
        self._kf.F = transition
        self._kf.predict()
        self._kf.x[3, 0] = _wrap_to_pi(float(self._kf.x[3, 0]))

    def update(self, detection: Detection) -> None:
        if self.yaw_measurement_mode == "unobserved":
            # T-7A.5 Phase 6 (T-7B fix): a genuine reduced-dimension
            # update (6-dim [x,y,z,l,w,h], yaw row dropped from both H
            # and R) rather than passing a fake yaw=0 with an inflated R
            # -- Linear KF's own yaw state has no dynamics coupling it to
            # x/y/velocity (identity row in F), so it simply holds
            # whatever value it was last set to. filterpy's own
            # `KalmanFilter.update()` validates `z` against the filter's
            # *fixed* `dim_z` (7) regardless of a per-call `H` override
            # (confirmed directly: `reshape_z` raises on a 6-dim `z`
            # against `dim_z=7`) -- T-7A.5 originally called
            # `self._kf.update(measurement, R=R_reduced, H=H_reduced)`,
            # which therefore always raised `ValueError` and was never
            # actually exercised end-to-end (T-7A.5's own real runs used
            # `yaw_measurement_mode="detector"` for Linear KF, never
            # triggering this path). Found and fixed during T-7B's IMM
            # implementation (which calls this path directly): the
            # reduced update is now done manually, mirroring
            # `EKFEstimator.update()`'s own already-correct math exactly.
            measurement = np.delete(detection.as_measurement_array(), 3)
            H = np.delete(self._kf.H, 3, axis=0)
            R = np.delete(np.delete(self._kf.R, 3, axis=0), 3, axis=1)
            x = self._kf.x.reshape(-1)
            predicted_measurement = H @ x
            residual = measurement - predicted_measurement
            S = H @ self._kf.P @ H.T + R
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.inv(S + np.eye(S.shape[0]) * 1e-6)
            K = self._kf.P @ H.T @ S_inv
            self._kf.x = (x + K @ residual).reshape(10, 1)
            self._kf.P = _symmetrize((np.eye(10) - K @ H) @ self._kf.P)
        else:
            track_yaw = float(self._kf.x[3, 0])
            corrected_track_yaw, corrected_obs_yaw = _orientation_correction(track_yaw, detection.yaw)
            self._kf.x[3, 0] = corrected_track_yaw
            measurement = detection.as_measurement_array()
            measurement[3] = corrected_obs_yaw
            self._kf.update(measurement)
        self._kf.x[3, 0] = _wrap_to_pi(float(self._kf.x[3, 0]))

    @property
    def position(self) -> tuple[float, float, float]:
        x = self._kf.x.reshape(-1)
        return float(x[0]), float(x[1]), float(x[2])

    @property
    def velocity(self) -> tuple[float, float, float]:
        x = self._kf.x.reshape(-1)
        return float(x[7]), float(x[8]), float(x[9])

    @property
    def yaw(self) -> float:
        return float(self._kf.x[3, 0])

    @property
    def dimensions(self) -> tuple[float, float, float]:
        x = self._kf.x.reshape(-1)
        return float(x[4]), float(x[5]), float(x[6])

    @property
    def position_covariance(self) -> np.ndarray:
        return np.array(self._kf.P[0:3, 0:3])

    @property
    def yaw_variance(self) -> float:
        return float(self._kf.P[3, 3])

    @property
    def velocity_covariance(self) -> np.ndarray:
        return np.array(self._kf.P[7:10, 7:10])

    def predicted_bev_position(self) -> np.ndarray:
        x = self._kf.x.reshape(-1)
        return np.array([float(x[0]), float(x[1])])

    def predicted_bev_innovation_covariance(self) -> np.ndarray:
        """`S = H P H^T + R`, sliced to the BEV `[x, y]` measurement block.

        Reuses the reference submodule's own
        `KF.compute_innovation_matrix()` (unmodified,
        `references/ab3dmot/AB3DMOT_libs/kalman_filter.py`) rather than
        re-deriving `H P H^T + R` -- see `association_metric_audit.md`
        Phase 1.
        """
        innovation = self._kf_wrapper.compute_innovation_matrix()
        return np.asarray(innovation)[0:2, 0:2]

    def is_finite(self) -> bool:
        return bool(np.all(np.isfinite(self._kf.x)) and np.all(np.isfinite(self._kf.P)))


# T-7A Phase 6: below this angular rate, `EKFEstimator.predict` uses the
# numerically stable straight-line limit of the CTRV equations instead of
# dividing by `yaw_rate` -- a standard, widely-used CTRV/UKF threshold
# (matches the well-known Udacity/CTRV reference implementation's own
# `1e-4` choice), not tuned against this task's own tracking outcomes.
_EKF_OMEGA_EPS = 1.0e-4


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Force exact numerical symmetry (`(M + M^T) / 2`) -- guards against
    floating-point asymmetry drift accumulating across many predict/update
    cycles, per T-7A Phase 6's "covariance symmetry" requirement."""
    return (matrix + matrix.T) * 0.5


class EKFEstimator:
    """T-7A: opt-in planar CTRV (constant turn-rate and velocity) EKF.

    State (10-dim, index order chosen to mirror `LinearKFEstimator`'s own
    layout so both share an identical linear measurement matrix `H` --
    see `ekf_design.md`):

        [x, y, z, yaw, l, w, h, v, yaw_rate, vz]

    Measurement (unchanged from Linear KF, 7-dim):

        [x, y, z, yaw, l, w, h]

    Because every measured quantity is a *direct* state component (not a
    nonlinear function of the state), the measurement model itself is
    still linear (`z = H x`, `H` = the same 7x10 identity-block matrix
    Linear KF's own reference `H` uses) -- only the *transition* model is
    nonlinear, so only the transition Jacobian `F` needs linearizing, not
    the measurement Jacobian. `v`/`yaw_rate` are latent (never directly
    measured), matching this task's explicit "do not invent velocity/
    yaw-rate measurements" requirement.
    """

    _H = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        ],
        dtype=float,
    )

    def __init__(
        self,
        detection: Detection,
        kf_class: type | None = None,
        track_id: int | None = None,
        yaw_measurement_mode: str = "detector",
    ) -> None:
        # `kf_class`/`track_id` accepted-but-unused for a uniform
        # constructor signature with `LinearKFEstimator` (see `Track`'s
        # single estimator-construction call site).
        self.yaw_measurement_mode = yaw_measurement_mode
        birth_yaw = detection.yaw if yaw_measurement_mode == "detector" else 0.0
        # T-7A.5 Phase 4: a single detection cannot observe motion
        # heading -- start at a neutral `yaw=0, v=0` (no fabricated
        # direction) in unobserved mode; `_heading_initialized` gates the
        # one-time displacement-based initialization in `update()` below.
        self.x = np.array(
            [detection.x, detection.y, detection.z, birth_yaw,
             detection.length, detection.width, detection.height,
             0.0, 0.0, 0.0],
            dtype=float,
        ).reshape(10, 1)
        self._heading_initialized = yaw_measurement_mode == "detector"
        self._last_measurement_xy = (detection.x, detection.y)
        self._time_since_last_measurement = 0.0
        # T-7A Phase 5: P0 directly adapted from Linear KF's own
        # `kf.P *= 10.; kf.P[7:, 7:] *= 1000.` (base uncertainty, plus a
        # much larger uncertainty on the 3 "rate" states unknown from a
        # single detection) -- same reasoning, same numeric factors,
        # applied to this state's own rate-state indices [7:10] = [v,
        # yaw_rate, vz] (option 1 of Phase 5: adapt existing Linear KF
        # values consistently).
        self.P = np.eye(10) * 10.0
        self.P[7:, 7:] *= 1000.0
        # T-7A Phase 5: Q likewise directly adapted from Linear KF's own
        # `kf.Q[7:, 7:] *= 0.01` (base process noise identity, extra
        # certainty on the rate states' own process noise).
        self.Q = np.eye(10)
        self.Q[7:, 7:] *= 0.01
        # T-7A Phase 5: R reused byte-for-byte from Linear KF's own
        # (filterpy default) measurement noise -- identical 7-dim
        # measurement space, so the fairest, most directly comparable
        # baseline is the identical R, not a re-derived one.
        self.R = np.eye(7)

    def predict(self, dt_seconds: float) -> None:
        if not math.isfinite(dt_seconds) or dt_seconds <= 0.0:
            raise ValueError(f"predict() requires a finite, positive dt_seconds, got {dt_seconds!r}")
        self._time_since_last_measurement += dt_seconds
        x, y, z, yaw, length, width, height, v, yaw_rate, vz = self.x.reshape(-1)
        dt = dt_seconds

        F = np.eye(10)
        if abs(yaw_rate) > _EKF_OMEGA_EPS:
            sin_yaw, cos_yaw = math.sin(yaw), math.cos(yaw)
            new_yaw = yaw + yaw_rate * dt
            sin_new, cos_new = math.sin(new_yaw), math.cos(new_yaw)
            new_x = x + (v / yaw_rate) * (sin_new - sin_yaw)
            new_y = y + (v / yaw_rate) * (-cos_new + cos_yaw)

            F[0, 3] = (v / yaw_rate) * (cos_new - cos_yaw)
            F[0, 7] = (sin_new - sin_yaw) / yaw_rate
            F[0, 8] = (v * dt / yaw_rate) * cos_new - (v / yaw_rate ** 2) * (sin_new - sin_yaw)
            F[1, 3] = (v / yaw_rate) * (sin_new - sin_yaw)
            F[1, 7] = (-cos_new + cos_yaw) / yaw_rate
            F[1, 8] = (v * dt / yaw_rate) * sin_new - (v / yaw_rate ** 2) * (-cos_new + cos_yaw)
        else:
            # Numerically stable straight-line limit (Phase 3/6): avoids
            # dividing by a near-zero yaw_rate.
            sin_yaw, cos_yaw = math.sin(yaw), math.cos(yaw)
            new_x = x + v * cos_yaw * dt
            new_y = y + v * sin_yaw * dt

            F[0, 3] = -v * sin_yaw * dt
            F[0, 7] = cos_yaw * dt
            F[1, 3] = v * cos_yaw * dt
            F[1, 7] = sin_yaw * dt
            # Near-zero-omega ∂x'/∂yaw_rate, ∂y'/∂yaw_rate are second-order
            # in dt (e.g. -v*sin(yaw)*dt^2/2) and omitted here -- documented
            # explicitly in ekf_design.md, not silently dropped.
        F[3, 8] = dt

        new_yaw = _wrap_to_pi(yaw + yaw_rate * dt)
        new_z = z + vz * dt
        F[2, 9] = dt

        self.x = np.array(
            [new_x, new_y, new_z, new_yaw, length, width, height, v, yaw_rate, vz],
            dtype=float,
        ).reshape(10, 1)
        self.P = _symmetrize(F @ self.P @ F.T + self.Q)

    # T-7A.5 Phase 6: below this displacement (meters) between the last
    # matched-measurement position and the current one, motion heading is
    # NOT initialized -- avoids a single noisy jitter step fabricating a
    # confident-looking direction. A documented, defensible baseline (not
    # tuned against this task's own tracking outcomes) roughly matching
    # the same-scene Euclidean gate scale (`euclidean_gate_m=3.0` is the
    # *rejection* radius; this is a much smaller *confirmation* distance
    # for "real, not-just-jitter" displacement).
    _MIN_HEADING_INIT_DISPLACEMENT_M = 0.5

    def update(self, detection: Detection) -> None:
        if self.yaw_measurement_mode == "unobserved":
            self._maybe_initialize_heading(detection)
            measurement = np.delete(detection.as_measurement_array(), 3)
            H = np.delete(self._H, 3, axis=0)
            R = np.delete(np.delete(self.R, 3, axis=0), 3, axis=1)
        else:
            track_yaw = float(self.x[3, 0])
            corrected_track_yaw, corrected_obs_yaw = _orientation_correction(track_yaw, detection.yaw)
            self.x[3, 0] = corrected_track_yaw
            measurement = detection.as_measurement_array()
            measurement[3] = corrected_obs_yaw
            H = self._H
            R = self.R

        predicted_measurement = (H @ self.x).reshape(-1)
        residual = measurement - predicted_measurement
        S = H @ self.P @ H.T + R
        # Phase 6: ill-conditioned S -- fall back to a small ridge term
        # rather than crash; a genuinely singular S should not occur here
        # (R = I is always full-rank), documented defensively only.
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.inv(S + np.eye(S.shape[0]) * 1e-6)
        K = self.P @ H.T @ S_inv

        self.x = (self.x.reshape(-1) + K @ residual).reshape(10, 1)
        self.x[3, 0] = _wrap_to_pi(float(self.x[3, 0]))
        self.P = _symmetrize((np.eye(10) - K @ H) @ self.P)

        self._last_measurement_xy = (detection.x, detection.y)
        self._time_since_last_measurement = 0.0

    def _maybe_initialize_heading(self, detection: Detection) -> None:
        """T-7A.5 Phase 4/5: one-time displacement-based motion-heading
        initialization -- never repeated once `_heading_initialized` is
        set, per this task's explicit "do not continually overwrite EKF
        heading with atan2 every frame" requirement. Directly sets `yaw`
        (to the observed displacement direction) and `v` (to the observed
        displacement magnitude over the real elapsed time, always `>= 0`
        by construction -- direction lives entirely in `yaw`, avoiding the
        negative-scalar-speed representation artifact `heading_source_audit.md`
        traces to T-7A's detector-yaw-pinned-near-zero mode)."""
        if self._heading_initialized:
            return
        last_x, last_y = self._last_measurement_xy
        dx = detection.x - last_x
        dy = detection.y - last_y
        displacement = math.hypot(dx, dy)
        if displacement < self._MIN_HEADING_INIT_DISPLACEMENT_M:
            return
        theta_motion = math.atan2(dy, dx)
        elapsed = max(self._time_since_last_measurement, 1.0e-6)
        speed = displacement / elapsed
        self.x[3, 0] = _wrap_to_pi(theta_motion)
        self.x[7, 0] = speed
        self._heading_initialized = True

    @property
    def position(self) -> tuple[float, float, float]:
        x = self.x.reshape(-1)
        return float(x[0]), float(x[1]), float(x[2])

    @property
    def velocity(self) -> tuple[float, float, float]:
        """ROS-facing `vx`/`vy`/`vz`, converted from the latent CTRV
        `[v, yaw_rate]` state (`vx = v*cos(yaw)`, `vy = v*sin(yaw)`) --
        the ROS `TrackedObjects` contract (unchanged since T-1B) expects
        Cartesian velocity components, not the CTRV `[v, yaw_rate]` pair
        directly."""
        x = self.x.reshape(-1)
        yaw, v, vz = x[3], x[7], x[9]
        return float(v * math.cos(yaw)), float(v * math.sin(yaw)), float(vz)

    @property
    def yaw(self) -> float:
        return float(self.x[3, 0])

    @property
    def dimensions(self) -> tuple[float, float, float]:
        x = self.x.reshape(-1)
        return float(x[4]), float(x[5]), float(x[6])

    @property
    def speed(self) -> float:
        """The latent CTRV scalar speed `v` (T-7A diagnostics only)."""
        return float(self.x[7, 0])

    @property
    def yaw_rate(self) -> float:
        """The latent CTRV `yaw_rate` (T-7A diagnostics only)."""
        return float(self.x[8, 0])

    @property
    def position_covariance(self) -> np.ndarray:
        return np.array(self.P[0:3, 0:3])

    @property
    def yaw_variance(self) -> float:
        return float(self.P[3, 3])

    @property
    def velocity_covariance(self) -> np.ndarray:
        """3x3 Cartesian-velocity-shaped covariance for ROS-output parity
        with `LinearKFEstimator`. CTRV's own velocity uncertainty lives in
        the scalar `[v, yaw_rate]` subspace (indices 7:9), which has no
        direct 1:1 Cartesian `[vx, vy, vz]` covariance without a nonlinear
        transform of the covariance itself (out of scope here, per this
        task's "do not over-engineer") -- reported as the raw `[v,
        yaw_rate, vz]`-indexed block `P[7:10, 7:10]` instead, documented
        explicitly as *not* Cartesian-shaped like Linear KF's own."""
        return np.array(self.P[7:10, 7:10])

    def predicted_bev_position(self) -> np.ndarray:
        x = self.x.reshape(-1)
        return np.array([float(x[0]), float(x[1])])

    def predicted_bev_innovation_covariance(self) -> np.ndarray:
        S = self._H @ self.P @ self._H.T + self.R
        return np.asarray(S)[0:2, 0:2]

    def is_finite(self) -> bool:
        return bool(np.all(np.isfinite(self.x)) and np.all(np.isfinite(self.P)))


# T-7B: below this speed (m/s), a Cartesian [vx, vy] pair's own direction
# (atan2(vy, vx)) is not trusted as a motion heading -- matches the same
# order of magnitude as `EKFEstimator._MIN_HEADING_INIT_DISPLACEMENT_M`
# (0.5 m) over one canonical-replay frame interval (~0.15 s), i.e. ~3 m/s
# scaled down for a single-sample (not accumulated-displacement) speed
# reading -- a documented, untuned baseline, not searched.
_IMM_LOW_SPEED_EPS = 0.3

# T-7B Phase 3: CV's own `yaw_rate` is unmeasured/unmodeled -- when mixed
# into the common representation, its mean is 0 with this documented
# ("we genuinely don't know, but a vehicle's plausible physical turn
# rate is bounded") variance. An earlier version of this constant
# directly reused Linear KF/EKF's own P0 rate-state scaling factor
# (T-7A Phase 5: base *10, rate-state block *1000 = 10000, i.e. a ~100
# rad/s standard deviation) -- found, via the same synthetic constant-
# turn instability investigation documented on `_ctrv_native_from_common`,
# to be wildly inappropriate here: that factor was calibrated for a
# single detection's own *initial* uncertainty before any correction,
# not for CV's *per-cycle* contribution to a running IMM mix, where a
# ~100 rad/s prior swamps CTRV's own already-converged, much tighter
# posterior yaw_rate variance on every single mixing step, making the
# combined estimate re-inflate every cycle regardless of how much real
# evidence CTRV has already accumulated. `4.0` (rad^2/s^2, ~2 rad/s
# standard deviation) is a generous-but-bounded physical prior -- already
# well beyond any plausible sustained vehicle turn rate -- chosen by
# this direct empirical stability finding, not searched/tuned against
# final tracking statistics.
_IMM_CV_YAW_RATE_PRIOR_VARIANCE = 4.0

# T-7B Phase 3/4: at low speed, a CV-derived or common-derived heading is
# unreliable (division by a near-zero vector norm) -- this additional
# variance is added to the common-state yaw slot instead of trusting a
# near-singular Jacobian-propagated value.
_IMM_LOW_SPEED_YAW_VARIANCE = 10.0

_IMM_PROBABILITY_FLOOR = 1.0e-6  # T-7B Phase 11, matches the same order
# of magnitude as the existing downstream `imm_predictor.cpp`'s own
# `kMinimumProbability` floor (1e-12), loosened slightly since this IMM
# uses only 2 models (a floor this tight is unnecessary here) -- prevents
# a model's posterior probability from ever reaching exactly 0 (which
# would make it permanently unrecoverable regardless of future evidence).


def _common_state_from_cv(cv_x: np.ndarray, cv_p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """T-7B Phase 3/4: CV native `[x,y,z,yaw,l,w,h,vx,vy,vz]` (10-dim) ->
    common `[x,y,z,yaw,l,w,h,vx,vy,vz,yaw_rate]` (11-dim), Jacobian-based
    covariance propagation (not a diagonal copy).

    `yaw` is **not** copied from CV's own (in `yaw_measurement_mode=
    "unobserved"`, dynamically inert/meaningless) yaw field -- it is
    re-derived from CV's own physically-meaningful Cartesian `[vx, vy]`
    velocity (`atan2(vy, vx)`) when speed exceeds `_IMM_LOW_SPEED_EPS`,
    since that is the more trustworthy motion-heading signal CV actually
    has. `yaw_rate` has no CV analogue: zero mean, large documented prior
    variance (`_IMM_CV_YAW_RATE_PRIOR_VARIANCE`).
    """
    cv_x = cv_x.reshape(-1)
    vx, vy = cv_x[7], cv_x[8]
    speed = math.hypot(vx, vy)

    common = np.zeros(11)
    common[0:3] = cv_x[0:3]
    common[4:10] = cv_x[4:10]

    G = np.zeros((11, 10))
    G[0, 0] = G[1, 1] = G[2, 2] = 1.0
    for i in range(4, 10):
        G[i, i] = 1.0

    if speed > _IMM_LOW_SPEED_EPS:
        common[3] = math.atan2(vy, vx)
        G[3, 7] = -vy / (speed * speed)
        G[3, 8] = vx / (speed * speed)
        yaw_extra_variance = 0.0
    else:
        common[3] = cv_x[3]  # last-known (possibly stale/inert) yaw -- documented fallback
        G[3, 3] = 1.0
        yaw_extra_variance = _IMM_LOW_SPEED_YAW_VARIANCE
    # yaw_rate (row 10): no CV analogue -- zero row in G, added separately below.

    P_common = _symmetrize(G @ cv_p @ G.T)
    P_common[3, 3] += yaw_extra_variance
    P_common[10, 10] += _IMM_CV_YAW_RATE_PRIOR_VARIANCE
    return common, P_common


def _common_state_from_ctrv(ctrv_x: np.ndarray, ctrv_p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """T-7B Phase 3/4: CTRV native `[x,y,z,yaw,l,w,h,v,yaw_rate,vz]`
    (10-dim) -> common `[x,y,z,yaw,l,w,h,vx,vy,vz,yaw_rate]` (11-dim).

    `yaw` is copied directly (CTRV's own yaw already *is* motion heading
    by construction). `vx = v*cos(yaw)`, `vy = v*sin(yaw)` (exact, not
    approximate); Jacobian-based covariance propagation throughout.
    """
    ctrv_x = ctrv_x.reshape(-1)
    x, y, z, yaw, length, width, height, v, yaw_rate, vz = ctrv_x

    common = np.array([x, y, z, yaw, length, width, height,
                        v * math.cos(yaw), v * math.sin(yaw), vz, yaw_rate])

    G = np.zeros((11, 10))
    G[0, 0] = G[1, 1] = G[2, 2] = G[3, 3] = 1.0
    G[4, 4] = G[5, 5] = G[6, 6] = 1.0
    G[7, 3] = -v * math.sin(yaw)   # d(vx)/d(yaw)
    G[7, 7] = math.cos(yaw)        # d(vx)/d(v)
    G[8, 3] = v * math.cos(yaw)    # d(vy)/d(yaw)
    G[8, 7] = math.sin(yaw)        # d(vy)/d(v)
    G[9, 9] = 1.0                  # vz
    G[10, 8] = 1.0                 # yaw_rate

    P_common = _symmetrize(G @ ctrv_p @ G.T)
    return common, P_common


def _cv_native_from_common(common_x: np.ndarray, common_p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """T-7B Phase 3/4: common (11-dim) -> CV native (10-dim). A pure
    linear projection (drop `yaw_rate`), so `H` is a fixed matrix."""
    H = np.zeros((10, 11))
    for i in range(10):
        H[i, i] = 1.0
    cv_x = H @ common_x
    cv_p = _symmetrize(H @ common_p @ H.T)
    return cv_x, cv_p


def _ctrv_native_from_common(common_x: np.ndarray, common_p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """T-7B Phase 3/4: common (11-dim) -> CTRV native (10-dim).

    `v = sqrt(vx^2 + vy^2)` (speed magnitude only). **Motion heading is
    taken directly from the common state's own `yaw` slot** (`H[3,3]=1`,
    a linear copy -- symmetric with `_cv_native_from_common`'s identical
    treatment of the same slot), **not** re-derived via `atan2(vy, vx)`
    from the (possibly mixed-in-CV, direction-lagging-under-curvature)
    Cartesian velocity every mixing cycle.

    An earlier version of this function *did* re-derive yaw via
    `atan2(vy, vx)` on every reseed (matching a literal reading of this
    task's Phase 3 prompt) -- found, by direct synthetic constant-turn
    testing (Phase 14), to make `yaw_rate` wildly unstable (observed
    swings of +-8 rad/s on a true-0.6 rad/s synthetic turn): CV's own
    velocity-direction estimate lags/overshoots under curvature, and
    re-injecting that noisy direction into CTRV's own already-good,
    circular-mixed `common.yaw` every single cycle discarded a better
    signal for a worse one. The common state's `yaw` slot is already the
    IMM's own best circular-mixed heading estimate (`IMMEstimator._combine`
    mixes it correctly across the +-pi wrap) -- reusing it directly here
    is both simpler and, empirically, far more stable. `atan2(vy, vx)` is
    still used exactly once, in `_common_state_from_cv`, for CV's own
    forward (native -> common) contribution, where no better heading
    signal exists (CV's own yaw field is inert in unobserved mode).
    """
    common_x = common_x.reshape(-1)
    vx, vy = common_x[7], common_x[8]
    speed = math.hypot(vx, vy)

    H = np.zeros((10, 11))
    H[0, 0] = H[1, 1] = H[2, 2] = 1.0
    H[3, 3] = 1.0   # yaw <- common yaw directly (see docstring)
    H[4, 4] = H[5, 5] = H[6, 6] = 1.0
    H[8, 10] = 1.0  # yaw_rate <- common yaw_rate
    H[9, 9] = 1.0   # vz

    ctrv_x = np.zeros(10)
    ctrv_x[0:3] = common_x[0:3]
    ctrv_x[3] = _wrap_to_pi(common_x[3])
    ctrv_x[4:7] = common_x[4:7]
    ctrv_x[8] = common_x[10]
    ctrv_x[9] = common_x[9]
    ctrv_x[7] = speed

    if speed > _IMM_LOW_SPEED_EPS:
        H[7, 7] = vx / speed
        H[7, 8] = vy / speed
        ctrv_p = _symmetrize(H @ common_p @ H.T)
    else:
        # `d(speed)/d(vx,vy)` is singular at vx=vy=0 -- rather than
        # zeroing this row and substituting an arbitrary small constant
        # (an earlier version added a fixed `10.0`, found by direct
        # synthetic-turn debugging to catastrophically collapse the
        # reseeded speed variance from CTRV's/CV's own native ~10,000-
        # scale P0 down to 10, crippling the subsequent Kalman gain and
        # causing exactly the wild yaw_rate oscillation this task's
        # Phase 14 synthetic tests are designed to catch), use the
        # isotropic proxy `0.5*(P_vx,vx + P_vy,vy)` -- preserves the
        # actual magnitude of whatever velocity uncertainty the mixed
        # common state already carries (whether that's ~10,000 at birth
        # or a small converged value later), without trusting a specific
        # (singular) direction.
        ctrv_p = _symmetrize(H @ common_p @ H.T)
        ctrv_p[7, 7] = 0.5 * (common_p[7, 7] + common_p[8, 8])
    return ctrv_x, ctrv_p


def _gaussian_log_likelihood(residual: np.ndarray, S: np.ndarray) -> float:
    """T-7B Phase 8: numerically stable multivariate-Gaussian log-density
    of `residual` under covariance `S`, via `np.linalg.slogdet` (avoids
    computing `|S|` directly, which can overflow/underflow for larger `S`)
    and `np.linalg.solve` (avoids an explicit matrix inverse)."""
    k = residual.shape[0]
    try:
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0 or not math.isfinite(logdet):
            raise np.linalg.LinAlgError("non-positive-definite S")
        solved = np.linalg.solve(S, residual)
    except np.linalg.LinAlgError:
        ridge = S + np.eye(k) * 1e-6
        sign, logdet = np.linalg.slogdet(ridge)
        solved = np.linalg.solve(ridge, residual)
    quad = float(residual @ solved)
    return -0.5 * (quad + logdet + k * math.log(2.0 * math.pi))


class IMMEstimator:
    """T-7B: 2-model Interacting Multiple Model estimator -- Model 1 =
    `LinearKFEstimator` (CV), Model 2 = `EKFEstimator` (CTRV). Standard
    IMM cycle each `predict()`/`update()` pair (Phase 5): mixing ->
    model-specific predict -> model-specific update -> innovation
    log-likelihood -> posterior model probability -> combined output.
    Mixing/combination happen in the common representation
    (`_common_state_from_cv`/`_common_state_from_ctrv`/
    `_cv_native_from_common`/`_ctrv_native_from_common`) -- native states
    are never averaged directly (Phase 2).
    """

    def __init__(
        self,
        detection: Detection,
        kf_class: type,
        track_id: int,
        yaw_measurement_mode: str = "detector",
        cv_to_cv_probability: float = 0.95,
        ctrv_to_ctrv_probability: float = 0.95,
    ) -> None:
        self._cv = LinearKFEstimator(detection, kf_class, track_id, yaw_measurement_mode)
        self._ctrv = EKFEstimator(detection, kf_class, track_id, yaw_measurement_mode)
        self._mu = np.array([0.5, 0.5])  # [P(CV), P(CTRV)], Phase 7: neutral prior
        self._transition = np.array(
            [[cv_to_cv_probability, 1.0 - cv_to_cv_probability],
             [1.0 - ctrv_to_ctrv_probability, ctrv_to_ctrv_probability]]
        )
        self._predicted_mu = self._mu.copy()
        self.log_likelihood_cv = 0.0
        self.log_likelihood_ctrv = 0.0
        self._combined_common, self._combined_common_p = self._combine(
            *self._native_to_common(), self._mu
        )

    def _native_to_common(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cv_common, cv_common_p = _common_state_from_cv(self._cv._kf.x, self._cv._kf.P)
        ctrv_common, ctrv_common_p = _common_state_from_ctrv(self._ctrv.x, self._ctrv.P)
        return cv_common, cv_common_p, ctrv_common, ctrv_common_p

    @staticmethod
    def _combine(
        cv_common: np.ndarray, cv_common_p: np.ndarray,
        ctrv_common: np.ndarray, ctrv_common_p: np.ndarray,
        mu: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Phase 10: combined mean + mixture covariance, INCLUDING the
        between-model mean-spread term -- never a plain weighted average
        of `P_j` alone."""
        states = [cv_common, ctrv_common]
        covs = [cv_common_p, ctrv_common_p]
        # circular-safe combination for the yaw slot (index 3): mix via
        # the weighted resultant vector, not a naive linear average
        # (which is wrong across a +-pi wrap).
        sin_sum = sum(mu[j] * math.sin(states[j][3]) for j in range(2))
        cos_sum = sum(mu[j] * math.cos(states[j][3]) for j in range(2))
        combined_yaw = math.atan2(sin_sum, cos_sum)

        combined = sum(mu[j] * states[j] for j in range(2))
        combined[3] = combined_yaw

        combined_p = np.zeros((11, 11))
        for j in range(2):
            diff = states[j] - combined
            diff[3] = math.atan2(math.sin(states[j][3] - combined_yaw), math.cos(states[j][3] - combined_yaw))
            combined_p += mu[j] * (covs[j] + np.outer(diff, diff))
        return combined, _symmetrize(combined_p)

    def predict(self, dt_seconds: float) -> None:
        if not math.isfinite(dt_seconds) or dt_seconds <= 0.0:
            raise ValueError(f"predict() requires a finite, positive dt_seconds, got {dt_seconds!r}")

        # 1. Mixing probabilities (Phase 5, steps 1-3).
        predicted_mu = self._transition.T @ self._mu  # c_j = sum_i p_ij * mu_i
        predicted_mu = np.maximum(predicted_mu, _IMM_PROBABILITY_FLOOR)
        predicted_mu = predicted_mu / predicted_mu.sum()
        mixing = np.zeros((2, 2))  # mixing[i, j] = mu_{i|j}
        for j in range(2):
            for i in range(2):
                mixing[i, j] = self._transition[i, j] * self._mu[i] / predicted_mu[j]

        # 2. Mixed initial condition per destination model, in common space.
        cv_common, cv_common_p, ctrv_common, ctrv_common_p = self._native_to_common()
        states = [cv_common, ctrv_common]
        covs = [cv_common_p, ctrv_common_p]

        mixed_common = []
        mixed_common_p = []
        for j in range(2):
            w = mixing[:, j]
            sin_sum = sum(w[i] * math.sin(states[i][3]) for i in range(2))
            cos_sum = sum(w[i] * math.cos(states[i][3]) for i in range(2))
            yaw_j = math.atan2(sin_sum, cos_sum)
            mean_j = sum(w[i] * states[i] for i in range(2))
            mean_j[3] = yaw_j
            p_j = np.zeros((11, 11))
            for i in range(2):
                diff = states[i] - mean_j
                diff[3] = math.atan2(math.sin(states[i][3] - yaw_j), math.cos(states[i][3] - yaw_j))
                p_j += w[i] * (covs[i] + np.outer(diff, diff))
            mixed_common.append(mean_j)
            mixed_common_p.append(_symmetrize(p_j))

        # 3. Convert each mixed common condition back to native, seed each
        # model directly (bypassing __init__), then run each model's own predict.
        cv_x, cv_p = _cv_native_from_common(mixed_common[0], mixed_common_p[0])
        self._cv._kf.x = cv_x.reshape(10, 1)
        self._cv._kf.P = cv_p

        ctrv_x, ctrv_p = _ctrv_native_from_common(mixed_common[1], mixed_common_p[1])
        self._ctrv.x = ctrv_x.reshape(10, 1)
        self._ctrv.P = ctrv_p

        self._cv.predict(dt_seconds)
        self._ctrv.predict(dt_seconds)
        self._predicted_mu = predicted_mu

        # Interim combined estimate (queryable between predict() and
        # update(), e.g. during association) -- uses the *predicted*
        # (pre-likelihood) model probabilities.
        cv_common, cv_common_p, ctrv_common, ctrv_common_p = self._native_to_common()
        self._combined_common, self._combined_common_p = self._combine(
            cv_common, cv_common_p, ctrv_common, ctrv_common_p, self._predicted_mu
        )

    def update(self, detection: Detection) -> None:
        # Innovation/likelihood computed from each model's own PRE-update
        # (post-predict) reduced measurement model (Phase 8/9 -- reuses
        # each estimator's own already-yaw-unobserved-aware H/R exactly,
        # so both models' likelihoods are computed over the identical
        # observed measurement subspace).
        cv_residual, cv_S = self._cv_innovation(detection)
        ctrv_residual, ctrv_S = self._ctrv_innovation(detection)
        self.log_likelihood_cv = _gaussian_log_likelihood(cv_residual, cv_S)
        self.log_likelihood_ctrv = _gaussian_log_likelihood(ctrv_residual, ctrv_S)

        self._cv.update(detection)
        self._ctrv.update(detection)

        # Posterior model probability via log-sum-exp (Phase 8/11: avoids
        # underflow from exponentiating raw, possibly very negative,
        # log-likelihoods directly).
        log_weights = np.array([
            self.log_likelihood_cv + math.log(max(self._predicted_mu[0], _IMM_PROBABILITY_FLOOR)),
            self.log_likelihood_ctrv + math.log(max(self._predicted_mu[1], _IMM_PROBABILITY_FLOOR)),
        ])
        max_lw = np.max(log_weights)
        weights = np.exp(log_weights - max_lw)
        posterior = weights / weights.sum()
        posterior = np.maximum(posterior, _IMM_PROBABILITY_FLOOR)
        posterior = posterior / posterior.sum()
        self._mu = posterior

        cv_common, cv_common_p, ctrv_common, ctrv_common_p = self._native_to_common()
        self._combined_common, self._combined_common_p = self._combine(
            cv_common, cv_common_p, ctrv_common, ctrv_common_p, self._mu
        )

    def _cv_innovation(self, detection: Detection) -> tuple[np.ndarray, np.ndarray]:
        kf = self._cv._kf
        if self._cv.yaw_measurement_mode == "unobserved":
            z = np.delete(detection.as_measurement_array(), 3)
            H = np.delete(kf.H, 3, axis=0)
            R = np.delete(np.delete(kf.R, 3, axis=0), 3, axis=1)
        else:
            z = detection.as_measurement_array()
            H, R = kf.H, kf.R
        residual = z - (H @ kf.x).reshape(-1)
        S = H @ kf.P @ H.T + R
        return residual, S

    def _ctrv_innovation(self, detection: Detection) -> tuple[np.ndarray, np.ndarray]:
        est = self._ctrv
        if est.yaw_measurement_mode == "unobserved":
            z = np.delete(detection.as_measurement_array(), 3)
            H = np.delete(est._H, 3, axis=0)
            R = np.delete(np.delete(est.R, 3, axis=0), 3, axis=1)
        else:
            z = detection.as_measurement_array()
            H, R = est._H, est.R
        residual = z - (H @ est.x).reshape(-1)
        S = H @ est.P @ H.T + R
        return residual, S

    # -- common estimator interface (mirrors LinearKFEstimator/EKFEstimator) --

    @property
    def mu_cv(self) -> float:
        return float(self._mu[0])

    @property
    def mu_ctrv(self) -> float:
        return float(self._mu[1])

    @property
    def position(self) -> tuple[float, float, float]:
        c = self._combined_common
        return float(c[0]), float(c[1]), float(c[2])

    @property
    def velocity(self) -> tuple[float, float, float]:
        c = self._combined_common
        return float(c[7]), float(c[8]), float(c[9])

    @property
    def yaw(self) -> float:
        return float(self._combined_common[3])

    @property
    def dimensions(self) -> tuple[float, float, float]:
        c = self._combined_common
        return float(c[4]), float(c[5]), float(c[6])

    @property
    def speed(self) -> float:
        vx, vy = self._combined_common[7], self._combined_common[8]
        return float(math.hypot(vx, vy))

    @property
    def yaw_rate(self) -> float:
        return float(self._combined_common[10])

    @property
    def position_covariance(self) -> np.ndarray:
        return np.array(self._combined_common_p[0:3, 0:3])

    @property
    def yaw_variance(self) -> float:
        return float(self._combined_common_p[3, 3])

    @property
    def velocity_covariance(self) -> np.ndarray:
        return np.array(self._combined_common_p[7:10, 7:10])

    def predicted_bev_position(self) -> np.ndarray:
        return np.array([float(self._combined_common[0]), float(self._combined_common[1])])

    def predicted_bev_innovation_covariance(self) -> np.ndarray:
        # mu-weighted combination of each model's own BEV innovation
        # covariance -- a reasonable, documented (not rigorously derived)
        # estimate for the rare case an unobserved-mode IMM track is
        # gated by Euclidean/Mahalanobis (T-7B's own primary experiment
        # uses Euclidean only, which does not call this method).
        cv_s = self._cv.predicted_bev_innovation_covariance()
        ctrv_s = self._ctrv.predicted_bev_innovation_covariance()
        return self._mu[0] * cv_s + self._mu[1] * ctrv_s

    def is_finite(self) -> bool:
        return (
            self._cv.is_finite() and self._ctrv.is_finite()
            and bool(np.all(np.isfinite(self._combined_common)))
            and bool(np.all(np.isfinite(self._combined_common_p)))
            and bool(np.all(np.isfinite(self._mu)))
        )


class Track:
    """One tracked object: a pluggable state estimator (`LinearKFEstimator`,
    the default, or `EKFEstimator`, T-7A opt-in) plus HEVEN's own explicit
    lifecycle bookkeeping (hits / time_since_update / age).

    `Track` itself owns no estimator-specific math -- it only ever calls
    the common `predict`/`update`/`position`/`velocity`/`yaw`/
    `dimensions`/`*_covariance`/`predicted_bev_*` interface both
    estimators implement identically, so lifecycle/association code above
    this class needs no per-estimator branching.
    """

    def __init__(
        self,
        track_id: int,
        detection: Detection,
        kf_class: type,
        state_estimator: str = "linear_kf",
        yaw_measurement_mode: str = "detector",
        imm_cv_to_cv_probability: float = 0.95,
        imm_ctrv_to_ctrv_probability: float = 0.95,
    ) -> None:
        self.track_id = track_id
        if state_estimator == "imm":
            self._estimator = IMMEstimator(
                detection, kf_class, track_id, yaw_measurement_mode,
                imm_cv_to_cv_probability, imm_ctrv_to_ctrv_probability,
            )
        elif state_estimator == "ekf":
            self._estimator = EKFEstimator(detection, kf_class, track_id, yaw_measurement_mode)
        else:
            self._estimator = LinearKFEstimator(detection, kf_class, track_id, yaw_measurement_mode)
        self.hits = 1
        self.time_since_update = 0
        self.age_frames = 0
        self.birth_frame_index: int | None = None  # set by Tracker on spawn
        self.label = detection.label
        self.label_probability = detection.label_probability
        self.existence_probability = detection.existence_probability

    @property
    def _kf(self):
        """Backward-compatible accessor to the underlying filterpy filter,
        used by pre-T-7A tests that directly manipulate a Linear-KF
        track's covariance (e.g. `track._kf.P *= 100.0`). Only valid for
        `state_estimator="linear_kf"` tracks -- raises `AttributeError`
        (via the estimator's own missing `_kf`) for an EKF track, which
        is correct: EKF has no filterpy-backed filter to expose."""
        return self._estimator._kf

    def predict(self, dt_seconds: float) -> None:
        """Advance the estimator by ``dt_seconds`` real elapsed time.

        Raises ``ValueError`` on non-finite or non-positive dt rather than
        silently assuming dt=1 -- this repository's own timestamp-gating
        code (`autoware_prediction_node.cpp`) follows the same
        reject-rather-than-coerce policy for malformed timing.
        """
        self._estimator.predict(dt_seconds)
        self.time_since_update += 1
        self.age_frames += 1

    def update(self, detection: Detection) -> None:
        """Correct the estimator with a matched ``detection``."""
        self._estimator.update(detection)
        self.time_since_update = 0
        self.hits += 1
        self.label = detection.label
        self.label_probability = detection.label_probability
        self.existence_probability = detection.existence_probability

    def is_confirmed(self, min_hits: int, frame_index: int) -> bool:
        """Output-eligibility rule, exactly AB3DMOT's own `output()` condition:
        ``hits >= min_hits`` OR still within the sequence's first
        ``min_hits`` frames (the early-sequence carve-out AB3DMOT itself
        uses, preserved here per this task's "do not redesign" instruction
        even though `tracking_architecture.md` flags it as more suited to
        offline/batch evaluation than a continuous online service)."""
        return self.hits >= min_hits or frame_index <= min_hits

    def is_dead(self, max_age: int) -> bool:
        return self.time_since_update >= max_age

    def predicted_bev_position(self) -> np.ndarray:
        """Predicted (post-`predict()`, pre-`update()`) BEV `[x, y]`."""
        return self._estimator.predicted_bev_position()

    def predicted_bev_innovation_covariance(self) -> np.ndarray:
        """`S = H P H^T + R`, sliced to the BEV `[x, y]` measurement block
        -- see each estimator class's own docstring."""
        return self._estimator.predicted_bev_innovation_covariance()

    def to_tracked_state(self) -> TrackedState:
        x, y, z = self._estimator.position
        vx, vy, vz = self._estimator.velocity
        length, width, height = self._estimator.dimensions
        return TrackedState(
            track_id=self.track_id,
            x=x, y=y, z=z,
            yaw=self._estimator.yaw,
            length=length, width=width, height=height,
            vx_mps=vx, vy_mps=vy, vz_mps=vz,
            position_covariance=self._estimator.position_covariance,
            yaw_variance=self._estimator.yaw_variance,
            velocity_covariance=self._estimator.velocity_covariance,
            hits=self.hits,
            time_since_update=self.time_since_update,
            label=self.label,
            label_probability=self.label_probability,
            existence_probability=self.existence_probability,
        )


class AB3DMOTTracker:
    """Multi-object tracker: association + track birth/update/deletion.

    Association is the smallest reference-faithful path for the resolved
    baseline (greedy or Hungarian matching over a HEVEN-native 3D GIoU
    cost matrix, gated by `config.giou_gate`) -- see `_greedy_matching`,
    `_hungarian_matching`, and `ab3dmot_geometry.giou_3d`. The matcher is
    selected through `AB3DMOTConfig.matcher` (`"greedy"`, the default, or
    `"hungarian"` -- T-3); both branches share the identical GIoU matrix
    construction and post-hoc gate application in `_associate` below, so
    switching matcher never changes predict/update/lifecycle behavior.
    """

    def __init__(
        self,
        config: AB3DMOTConfig,
        ab3dmot_root: Path | None = None,
        collect_diagnostics: bool = False,
    ) -> None:
        self.config = config
        self._kf_class = _load_ab3dmot_kf_class(ab3dmot_root)
        self._tracks: list[Track] = []
        self._next_id = 1
        self._frame_index = 0
        self._last_timestamp: float | None = None
        # T-3 Phase 11/12: opt-in, additive-only per-frame instrumentation
        # (never read by predict/update/lifecycle/association math itself).
        # When enabled, `self.diagnostics` gains one dict per `step()` call
        # with match-level and timing detail -- used only by offline T-3
        # comparison tooling, not by production/ROS code paths.
        self.collect_diagnostics = collect_diagnostics
        self.diagnostics: list[dict] | None = [] if collect_diagnostics else None

    @property
    def tracks(self) -> Sequence[Track]:
        return tuple(self._tracks)

    def step(self, detections: Sequence[Detection], timestamp_seconds: float) -> list[TrackedState]:
        """Process one frame of detections at ``timestamp_seconds`` (a
        monotonic real-time clock in seconds; units are the caller's
        responsibility -- T-1B will pass a ROS `Time` converted to
        seconds). Returns the currently-confirmed, non-expired tracks."""
        if not math.isfinite(timestamp_seconds):
            raise ValueError(f"timestamp_seconds must be finite, got {timestamp_seconds!r}")
        self._frame_index += 1

        if self._last_timestamp is not None:
            dt = timestamp_seconds - self._last_timestamp
            if not math.isfinite(dt) or dt <= 0.0:
                raise ValueError(
                    "step() requires a monotonically increasing, finite timestamp "
                    f"(dt={dt!r}); refusing to silently assume dt=1"
                )
            for track in self._tracks:
                track.predict(dt)

        matched_det_indices, matched_track_indices = self._associate(detections)

        for det_index, track_index in zip(matched_det_indices, matched_track_indices):
            self._tracks[track_index].update(detections[det_index])

        matched_dets = set(matched_det_indices.tolist())
        for det_index, detection in enumerate(detections):
            if det_index in matched_dets:
                continue
            new_track = Track(
                self._next_id, detection, self._kf_class,
                self.config.state_estimator, self.config.yaw_measurement_mode,
                self.config.imm_cv_to_cv_probability, self.config.imm_ctrv_to_ctrv_probability,
            )
            new_track.birth_frame_index = self._frame_index
            self._tracks.append(new_track)
            self._next_id += 1

        outputs = [
            track.to_tracked_state()
            for track in self._tracks
            if not track.is_dead(self.config.max_age)
            and track.is_confirmed(self.config.min_hits, self._frame_index)
        ]

        self._tracks = [track for track in self._tracks if not track.is_dead(self.config.max_age)]
        self._last_timestamp = timestamp_seconds
        return outputs

    def _associate(self, detections: Sequence[Detection]) -> tuple[np.ndarray, np.ndarray]:
        metric = self.config.association_metric
        if not detections or not self._tracks:
            if self.collect_diagnostics:
                empty_entry = {
                    "frame_index": self._frame_index,
                    "association_metric": metric,
                    "n_detections": len(detections),
                    "n_tracks": len(self._tracks),
                    "n_valid_pairs": 0,
                    "n_matches": 0,
                    "total_matched_score": 0.0,
                    "mean_matched_score": None,
                    "raw_total_score": 0.0,
                    "raw_matched_pairs": [],
                    "metric_construction_ms": 0.0,
                    "solver_ms": 0.0,
                    "metric_matrix": None,
                    "matched_pairs": [],
                }
                if metric == "mahalanobis":
                    empty_entry["mahalanobis_max_distance_m"] = self.config.mahalanobis_max_distance_m
                    empty_entry["n_mahalanobis_only_valid_pairs"] = 0
                    empty_entry["hybrid_gate_ms"] = 0.0
                    empty_entry["euclid_matrix"] = None
                self.diagnostics.append(empty_entry)
            return np.empty(0, dtype=int), np.empty(0, dtype=int)

        t0 = time.perf_counter()
        if metric == "euclidean":
            metric_matrix = _euclidean_bev_distance_matrix(detections, self._tracks)
            # lower distance = better; both matchers already minimize.
            cost_matrix = metric_matrix
        elif metric == "mahalanobis":
            metric_matrix = _mahalanobis_bev_distance_matrix(detections, self._tracks)
            cost_matrix = metric_matrix
        else:
            metric_matrix = np.empty((len(detections), len(self._tracks)))
            for d_index, detection in enumerate(detections):
                det_box = detection.as_box3d()
                for t_index, track in enumerate(self._tracks):
                    predicted = track.to_tracked_state()
                    track_box = Box3D(
                        predicted.x, predicted.y, predicted.z, predicted.yaw,
                        predicted.length, predicted.width, predicted.height,
                    )
                    metric_matrix[d_index, t_index] = giou_3d(det_box, track_box)
            # GIoU is affinity (higher = better); negate once into a
            # lower-is-better cost, same convention both matchers consume.
            cost_matrix = -metric_matrix
        t1 = time.perf_counter()

        if self.config.matcher == "hungarian":
            raw_matches = _hungarian_matching(cost_matrix)
        else:
            raw_matches = _greedy_matching(cost_matrix)
        t2 = time.perf_counter()

        if metric == "giou_3d":
            def _passes_gate(det_index: int, track_index: int) -> bool:
                return metric_matrix[det_index, track_index] >= self.config.giou_gate
            valid_mask = metric_matrix >= self.config.giou_gate
        elif metric == "euclidean":
            def _passes_gate(det_index: int, track_index: int) -> bool:
                return metric_matrix[det_index, track_index] <= self.config.euclidean_gate_m
            valid_mask = metric_matrix <= self.config.euclidean_gate_m
        else:
            # T-5B: optional hybrid gate -- a pair must pass the chi-square
            # Mahalanobis gate AND (if enabled) an absolute BEV physical-
            # distance cap. The cap is a validity gate only: it never
            # changes `cost_matrix`/`metric_matrix` (still raw d_M^2), so
            # Hungarian still minimizes pure Mahalanobis distance among
            # whichever pairs pass both gates. `mahalanobis_max_distance_m
            # <= 0` reproduces pure T-4/T-5A Mahalanobis behavior exactly
            # (unchanged code path, same single condition as before).
            cap = self.config.mahalanobis_max_distance_m
            mahalanobis_valid_mask = metric_matrix <= self.config.mahalanobis_gate
            euclid_matrix = None
            hybrid_gate_ms = 0.0
            if cap > 0.0:
                t_hybrid0 = time.perf_counter()
                euclid_matrix = _euclidean_bev_distance_matrix(detections, self._tracks)
                distance_valid_mask = euclid_matrix <= cap
                hybrid_gate_ms = (time.perf_counter() - t_hybrid0) * 1000.0
                valid_mask = mahalanobis_valid_mask & distance_valid_mask

                def _passes_gate(det_index: int, track_index: int) -> bool:
                    return bool(valid_mask[det_index, track_index])
            else:
                valid_mask = mahalanobis_valid_mask
                if self.collect_diagnostics:
                    # diagnostics-only: still compute the raw BEV distance
                    # matrix so offline analysis can report the physical
                    # distance of pure-Mahalanobis accepted matches too
                    # (needed for the T-5B no-cap baseline comparison) --
                    # never used for gating when the cap is disabled.
                    euclid_matrix = _euclidean_bev_distance_matrix(detections, self._tracks)

                def _passes_gate(det_index: int, track_index: int) -> bool:
                    return metric_matrix[det_index, track_index] <= self.config.mahalanobis_gate

        accepted = [
            (det_index, track_index)
            for det_index, track_index in raw_matches
            if _passes_gate(det_index, track_index)
        ]

        if self.collect_diagnostics:
            matched_scores = [float(metric_matrix[d, t]) for d, t in accepted]
            raw_scores = [float(metric_matrix[d, t]) for d, t in raw_matches]
            diag_entry = {
                "frame_index": self._frame_index,
                "association_metric": metric,
                "n_detections": len(detections),
                "n_tracks": len(self._tracks),
                "n_valid_pairs": int(np.sum(valid_mask)),
                "n_matches": len(accepted),
                "total_matched_score": float(sum(matched_scores)),
                "mean_matched_score": (float(sum(matched_scores) / len(matched_scores)) if matched_scores else None),
                # pre-gate: the raw solver output before the metric-specific
                # gate is applied -- what the solver's own optimality
                # guarantee (Hungarian) actually covers; the gate is a
                # nonlinear post-filter that can change which pairs survive
                # differently per matcher/metric (see T-3's writeup).
                "raw_total_score": float(sum(raw_scores)),
                "raw_matched_pairs": [[int(d), int(t)] for d, t in raw_matches],
                "metric_construction_ms": (t1 - t0) * 1000.0,
                "solver_ms": (t2 - t1) * 1000.0,
                "metric_matrix": metric_matrix.tolist(),
                "matched_pairs": [[int(d), int(t)] for d, t in accepted],
            }
            if metric == "mahalanobis":
                diag_entry["mahalanobis_max_distance_m"] = self.config.mahalanobis_max_distance_m
                diag_entry["n_mahalanobis_only_valid_pairs"] = int(np.sum(mahalanobis_valid_mask))
                diag_entry["hybrid_gate_ms"] = hybrid_gate_ms
                diag_entry["euclid_matrix"] = (
                    euclid_matrix.tolist() if euclid_matrix is not None else None
                )
            self.diagnostics.append(diag_entry)

        if not accepted:
            return np.empty(0, dtype=int), np.empty(0, dtype=int)
        det_indices, track_indices = zip(*accepted)
        return np.array(det_indices, dtype=int), np.array(track_indices, dtype=int)
