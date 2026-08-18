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
"""

from __future__ import annotations

import importlib
import math
import sys
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


class Track:
    """One tracked object: an AB3DMOT-style 10-state KF plus HEVEN's own
    explicit lifecycle bookkeeping (hits / time_since_update / age).

    The reused `AB3DMOT_libs.kalman_filter.KF` instance's own `hits` /
    `time_since_update` / `id` fields (from its `Filter` base class) are
    intentionally left untouched and unread -- this class is the single
    source of truth for lifecycle state, since it also owns the real-dt
    predict wrapper AB3DMOT's own class does not provide.
    """

    def __init__(self, track_id: int, detection: Detection, kf_class: type) -> None:
        self.track_id = track_id
        # `info` is an unused pass-through slot on the reused reference
        # class (stored but never read by anything in this module) -- we
        # carry classification/score via our own label fields instead,
        # below, so both are updated together on every matched update.
        self._kf_wrapper = kf_class(detection.as_measurement_array(), np.zeros(1), track_id)
        self.hits = 1
        self.time_since_update = 0
        self.age_frames = 0
        self.birth_frame_index: int | None = None  # set by Tracker on spawn
        self.label = detection.label
        self.label_probability = detection.label_probability
        self.existence_probability = detection.existence_probability

    @property
    def _kf(self):
        return self._kf_wrapper.kf  # the underlying filterpy.kalman.KalmanFilter

    def predict(self, dt_seconds: float) -> None:
        """Advance the filter by ``dt_seconds`` real elapsed time.

        Rebuilds the state-transition matrix with the actual dt (not
        AB3DMOT's implicit dt=1-per-frame), matching SimpleTrack's
        documented `get_prediction()` deviation. Raises ``ValueError`` on
        non-finite or non-positive dt rather than silently assuming
        dt=1 -- this repository's own timestamp-gating code
        (`autoware_prediction_node.cpp`) follows the same
        reject-rather-than-coerce policy for malformed timing.
        """
        if not math.isfinite(dt_seconds) or dt_seconds <= 0.0:
            raise ValueError(f"predict() requires a finite, positive dt_seconds, got {dt_seconds!r}")
        transition = np.eye(10)
        transition[0, 7] = dt_seconds
        transition[1, 8] = dt_seconds
        transition[2, 9] = dt_seconds
        self._kf.F = transition
        self._kf.predict()
        self._kf.x[3, 0] = _wrap_to_pi(float(self._kf.x[3, 0]))
        self.time_since_update += 1
        self.age_frames += 1

    def update(self, detection: Detection) -> None:
        """Correct the filter with a matched ``detection``."""
        track_yaw = float(self._kf.x[3, 0])
        corrected_track_yaw, corrected_obs_yaw = _orientation_correction(track_yaw, detection.yaw)
        self._kf.x[3, 0] = corrected_track_yaw
        measurement = detection.as_measurement_array()
        measurement[3] = corrected_obs_yaw
        self._kf.update(measurement)
        self._kf.x[3, 0] = _wrap_to_pi(float(self._kf.x[3, 0]))
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

    def to_tracked_state(self) -> TrackedState:
        x = self._kf.x.reshape(-1)
        p = self._kf.P
        return TrackedState(
            track_id=self.track_id,
            x=float(x[0]),
            y=float(x[1]),
            z=float(x[2]),
            yaw=float(x[3]),
            length=float(x[4]),
            width=float(x[5]),
            height=float(x[6]),
            vx_mps=float(x[7]),
            vy_mps=float(x[8]),
            vz_mps=float(x[9]),
            position_covariance=np.array(p[0:3, 0:3]),
            yaw_variance=float(p[3, 3]),
            velocity_covariance=np.array(p[7:10, 7:10]),
            hits=self.hits,
            time_since_update=self.time_since_update,
            label=self.label,
            label_probability=self.label_probability,
            existence_probability=self.existence_probability,
        )


class AB3DMOTTracker:
    """Multi-object tracker: association + track birth/update/deletion.

    Association is the smallest reference-faithful path for the resolved
    baseline (greedy matching over a HEVEN-native 3D GIoU cost matrix,
    gated by `config.giou_gate`) -- see `_greedy_matching` and
    `ab3dmot_geometry.giou_3d`. The matcher is intentionally selected
    through `AB3DMOTConfig.matcher` (currently only `"greedy"` is
    implemented; `AB3DMOTConfig` rejects any other value) so a Hungarian
    or other matcher can be added later as an additional branch here
    without touching the surrounding predict/update/lifecycle code --
    not implemented now, per this task's scope.
    """

    def __init__(self, config: AB3DMOTConfig, ab3dmot_root: Path | None = None) -> None:
        self.config = config
        self._kf_class = _load_ab3dmot_kf_class(ab3dmot_root)
        self._tracks: list[Track] = []
        self._next_id = 1
        self._frame_index = 0
        self._last_timestamp: float | None = None

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
            new_track = Track(self._next_id, detection, self._kf_class)
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
        if not detections or not self._tracks:
            return np.empty(0, dtype=int), np.empty(0, dtype=int)

        giou_matrix = np.empty((len(detections), len(self._tracks)))
        for d_index, detection in enumerate(detections):
            det_box = detection.as_box3d()
            for t_index, track in enumerate(self._tracks):
                predicted = track.to_tracked_state()
                track_box = Box3D(
                    predicted.x, predicted.y, predicted.z, predicted.yaw,
                    predicted.length, predicted.width, predicted.height,
                )
                giou_matrix[d_index, t_index] = giou_3d(det_box, track_box)

        # Greedy matcher is the only implemented option (config enforces this).
        raw_matches = _greedy_matching(-giou_matrix)
        accepted = [
            (det_index, track_index)
            for det_index, track_index in raw_matches
            if giou_matrix[det_index, track_index] >= self.config.giou_gate
        ]
        if not accepted:
            return np.empty(0, dtype=int), np.empty(0, dtype=int)
        det_indices, track_indices = zip(*accepted)
        return np.array(det_indices, dtype=int), np.array(track_indices, dtype=int)
