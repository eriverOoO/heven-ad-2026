"""AV2 Motion Forecasting -> HEVEN KalmanNet Dataset Adapter v1 (Stage 0).

Deterministically converts downloaded Argoverse 2 (AV2) Motion Forecasting
scenario parquet files into the same versioned, per-segment trajectory
representation the MORAI-side adapter already produces
(``kalmannet_trajectory_adapter.py``'s ``kalmannet_morai_trajectory_v1``
shape), so both sources feed the *same* downstream KalmanNet training
convention without a second data format.

**This module trains nothing and changes no KalmanNet architecture.** It
does not import or modify ``kalmannet_core.py`` / ``kalmannet_arch2_core.py``
/ AB3DMOT / CenterPoint / the planner / prediction / the occupancy grid.
``KNET_STATE_DIM`` / ``KNET_MEAS_DIM`` are imported from
``kalmannet_trajectory_adapter.py``, never redefined here.

Audit findings this module deliberately preserves (see
``docs/perception/av2_kalmannet_adapter_v1.md`` for the full writeup):

- KalmanNet already supports real, variable ``dt`` via
  ``KalmanNetFilter.step(z, dt)`` -> analytical ``F_matrix(dt)`` /
  ``Q_matrix(dt, ...)``. ``dt`` is never added as a neural-network
  feature here.
- Missing-measurement handling is already implemented end to end as
  "prediction-only step, no learned-gain update" for short/in-sequence
  misses, and "sequence split" for a large temporal gap. This module
  reuses exactly that representation (``measurement_valid[t] = False``,
  ``measurement[t] = NaN`` -- never ``[0, 0]``, never a repeated previous
  measurement) rather than inventing a second missing-measurement
  convention.

AV2 schema facts used below were confirmed against 120 real downloaded
``train`` scenarios (not assumed from documentation): 110 timesteps per
scenario at a real, uniform 100 ms step (10 Hz); ``FOCAL_TRACK`` /
``SCORED_TRACK`` tracks span all 110 timesteps with ``observed`` flipping
``True -> False`` at exactly timestep 50 (the "50 observed + 60 future"
convention, confirmed, not hard-coded from docs alone); every other
track's ``object_states`` list is a *contiguous* sub-range of
``[0, 109]`` (no internal per-track gaps in raw AV2 data -- a "missing
measurement" in the sense this adapter's corruption layer (``CorruptionConfig``)
introduces is a synthetic construct, not something AV2 itself contains);
0 non-finite states and 0 non-contiguous-timestep tracks were observed
across all 7,358 tracks in the 120-scenario Stage-0 sample.

Output tree (mirrors ``kalmannet_trajectory_adapter.py``'s layout;
`` <split>`` is one of AV2's own ``train`` / ``val`` / ``test`` -- this
adapter never re-splits AV2 data, since AV2's own split is already
leakage-free by construction across disjoint S3 prefixes)::

    <output_root>/<split>/
      .av2_kalmannet_adapter          # marker
      metadata.json
      export_manifest.json            # schema, provenance, config, content fingerprint
      trajectory_index.jsonl          # one row per exported segment
      split_manifest.json             # AV2's own split name + scenario-leakage check result
      trajectories/<segment_id>.npz   # time-major arrays, same array names as the MORAI adapter
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ad_morai_bridge_dev.dataset.centerpoint_adapter import AdapterError
from ad_morai_bridge_dev.dataset.kalmannet_trajectory_adapter import (
    KNET_MEAS_DIM,
    KNET_MEAS_FIELDS,
    KNET_STATE_DIM,
    KNET_STATE_FIELDS,
    _atomic_write_json,
    _npz_bytes,
    _sha256_bytes,
)

ADAPTER_SCHEMA_VERSION = "av2_kalmannet_adapter_v1"

# AV2's own three top-level Motion Forecasting splits (disjoint S3
# prefixes) -- this adapter never derives a fourth split; it reuses AV2's
# split assignment verbatim.
SUPPORTED_AV2_SPLITS = ("train", "val", "test")

COORDINATE_MODE_FIRST_STATE_RELATIVE = "first_state_relative"
SUPPORTED_COORDINATE_MODES = (COORDINATE_MODE_FIRST_STATE_RELATIVE,)

# The exact per-frame numeric arrays written into every trajectories/<id>.npz.
NPZ_ARRAY_NAMES = (
    "timestamps_ns",
    "dt_s",
    "gt_state",
    "clean_position",
    "measurement",
    "measurement_valid",
    "av2_timestep",
    "av2_observed",
    "origin_xy",
)

# AV2's real ObjectType member names (confirmed live against `av2` 0.3.6's
# own ``av2.datasets.motion_forecasting.data_schema.ObjectType`` enum, and
# cross-checked against all 7,358 tracks in the Stage-0 sample -- every
# value below was actually observed, none invented).
AV2_OBJECT_TYPES = (
    "VEHICLE", "PEDESTRIAN", "MOTORCYCLIST", "CYCLIST", "BUS",
    "STATIC", "BACKGROUND", "CONSTRUCTION", "RIDERLESS_BICYCLE", "UNKNOWN",
)
AV2_TRACK_CATEGORIES = ("TRACK_FRAGMENT", "UNSCORED_TRACK", "SCORED_TRACK", "FOCAL_TRACK")

# Proposed adapter-metadata-only coarse mapping (task's own explicit
# taxonomy target: VEHICLE / PEDESTRIAN / OTHER). Never fed to KalmanNet;
# stored purely as ``coarse_object_type`` alongside the untouched
# ``av2_object_type``. See docs/perception/av2_kalmannet_adapter_v1.md
# section "Class metadata" for the rationale -- this is a proposed v1
# mapping under the task's own fixed 3-bucket constraint, not a claim
# that e.g. a motorcyclist is motion-equivalent to a static obstacle.
COARSE_CLASS_MAP = {
    "VEHICLE": "VEHICLE",
    "BUS": "VEHICLE",
    "PEDESTRIAN": "PEDESTRIAN",
    "MOTORCYCLIST": "OTHER",
    "CYCLIST": "OTHER",
    "STATIC": "OTHER",
    "BACKGROUND": "OTHER",
    "CONSTRUCTION": "OTHER",
    "RIDERLESS_BICYCLE": "OTHER",
    "UNKNOWN": "OTHER",
}
COARSE_CLASSES = ("VEHICLE", "PEDESTRIAN", "OTHER")


def coarse_object_type(av2_object_type: str) -> str:
    if av2_object_type not in COARSE_CLASS_MAP:
        raise AdapterError(f"unknown AV2 object_type {av2_object_type!r}")
    return COARSE_CLASS_MAP[av2_object_type]


def _finite(*values: float) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


# --------------------------------------------------------------------------
# lightweight, av2-package-independent source model
# --------------------------------------------------------------------------
# These mirror av2.datasets.motion_forecasting.data_schema's real field
# names 1:1 (ObjectState.observed/timestep/position/heading/velocity;
# Track.track_id/object_type/category; ArgoverseScenario.scenario_id/
# city_name/focal_track_id/timestamps_ns) but are plain, locally-defined
# dataclasses -- so every function below this line, and every unit test,
# needs no ``av2`` pip install (a heavy dependency: torch, opencv,
# kornia, ...). Only ``load_av2_scenario_parquet`` (the one real-file
# loader) imports ``av2``, and only lazily, at call time -- exactly the
# lazy-import convention this repo already uses for optional heavy
# dependencies (e.g. ``kalmannet_core``'s own lazy ``torch`` import).
@dataclass(frozen=True)
class Av2ObjectStateRecord:
    observed: bool
    timestep: int
    x: float
    y: float
    heading: float
    vx: float
    vy: float


@dataclass(frozen=True)
class Av2TrackRecord:
    track_id: str
    object_type: str   # AV2_OBJECT_TYPES member name, verbatim
    category: str       # AV2_TRACK_CATEGORIES member name, verbatim
    states: tuple[Av2ObjectStateRecord, ...]


@dataclass(frozen=True)
class Av2ScenarioRecord:
    scenario_id: str
    city_name: str
    focal_track_id: str
    timestamps_ns: tuple[int, ...]
    tracks: tuple[Av2TrackRecord, ...]


def scenario_record_from_av2(scenario: Any) -> Av2ScenarioRecord:
    """Thin, one-shot conversion from a real
    ``av2.datasets.motion_forecasting.data_schema.ArgoverseScenario`` into
    the local, av2-independent ``Av2ScenarioRecord``. No field is renamed
    or reinterpreted -- this is a type/container conversion only."""
    tracks = []
    for t in scenario.tracks:
        states = tuple(
            Av2ObjectStateRecord(
                observed=bool(st.observed),
                timestep=int(st.timestep),
                x=float(st.position[0]),
                y=float(st.position[1]),
                heading=float(st.heading),
                vx=float(st.velocity[0]),
                vy=float(st.velocity[1]),
            )
            for st in t.object_states
        )
        tracks.append(
            Av2TrackRecord(
                track_id=str(t.track_id),
                object_type=t.object_type.name,
                category=t.category.name,
                states=states,
            )
        )
    return Av2ScenarioRecord(
        scenario_id=str(scenario.scenario_id),
        city_name=str(scenario.city_name),
        focal_track_id=str(scenario.focal_track_id),
        timestamps_ns=tuple(int(v) for v in scenario.timestamps_ns),
        tracks=tuple(tracks),
    )


def load_av2_scenario_parquet(path: Path) -> Av2ScenarioRecord:
    """Loads one real ``scenario_<id>.parquet`` file via the official
    ``av2`` API (lazy import) and converts it to ``Av2ScenarioRecord``.
    Never reads the companion ``log_map_archive_<id>.json`` -- KalmanNet
    v1 needs no map geometry (confirmed: the real loader's signature is
    ``load_argoverse_scenario_parquet(scenario_path) -> ArgoverseScenario``,
    a single path argument, no map dependency)."""
    try:
        from av2.datasets.motion_forecasting.scenario_serialization import (
            load_argoverse_scenario_parquet,
        )
    except ImportError as exc:
        raise AdapterError(
            "loading a real AV2 parquet file requires the 'av2' pip package "
            "(pip install av2); not required for fixture-based tests or for "
            "any function operating on an already-constructed Av2ScenarioRecord"
        ) from exc
    scenario = load_argoverse_scenario_parquet(Path(path))
    return scenario_record_from_av2(scenario)


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SegmentConfig:
    """Mirrors ``kalmannet_trajectory_adapter.TrajectoryConfig`` exactly
    (same field names/defaults/semantics) -- deliberately not a new
    segmentation philosophy. AV2's own per-track ``object_states`` are
    already internally contiguous (confirmed on all 7,358 Stage-0 tracks),
    so in practice these thresholds are only exercised when this
    adapter's own non-finite-frame filtering removes a frame from the
    middle of a track."""

    max_gt_gap_s: float = 1.0
    max_teleport_speed_mps: float = 60.0
    min_gt_samples: int = 5

    def canonical_json(self) -> str:
        return json.dumps(
            {
                "max_gt_gap_s": self.max_gt_gap_s,
                "max_teleport_speed_mps": self.max_teleport_speed_mps,
                "min_gt_samples": self.min_gt_samples,
            },
            sort_keys=True,
        )

    def validate(self) -> None:
        if not (self.max_gt_gap_s > 0.0 and math.isfinite(self.max_gt_gap_s)):
            raise AdapterError("max_gt_gap_s must be a positive finite number")
        if not (self.max_teleport_speed_mps > 0.0 and math.isfinite(self.max_teleport_speed_mps)):
            raise AdapterError("max_teleport_speed_mps must be a positive finite number")
        if self.min_gt_samples < 2:
            raise AdapterError("min_gt_samples must be >= 2")


@dataclass(frozen=True)
class CoordinateConfig:
    """v1 supports exactly one explicit, versioned mode: first-state-
    relative translation (audit recommendation). No rotation into an
    actor heading frame, no normalization by future trajectory
    statistics -- both explicitly rejected by the audit and by this
    task's own instruction."""

    mode: str = COORDINATE_MODE_FIRST_STATE_RELATIVE

    def canonical_json(self) -> str:
        return json.dumps({"coordinate_mode": self.mode}, sort_keys=True)

    def validate(self) -> None:
        if self.mode not in SUPPORTED_COORDINATE_MODES:
            raise AdapterError(
                f"unsupported coordinate_mode {self.mode!r}; only {SUPPORTED_COORDINATE_MODES} implemented"
            )


@dataclass(frozen=True)
class CorruptionConfig:
    """Stage-0 minimal corruption only (task's own explicit scope: no
    range-dependent noise, no heavy-tail tuning, no ID-switch injection,
    no class-conditioned corruption, no MORAI-calibrated noise, no
    outliers). Every knob is independently composable and default-off;
    ``mode="none"`` (every numeric knob at its default) reproduces the
    clean AV2 position exactly, with ``measurement_valid=True``
    everywhere. Deterministic given ``seed`` -- see
    ``_segment_corruption_seed`` for the per-segment seed derivation."""

    mode: str = "none"  # "none" | "gaussian" | "dropout" | "dropout_burst" | "custom" (informational only)
    gaussian_noise_std_m: float = 0.0
    dropout_prob: float = 0.0
    dropout_burst_enabled: bool = False
    dropout_burst_prob: float = 0.0
    dropout_burst_min_len: int = 2
    dropout_burst_max_len: int = 5
    seed: int = 0

    def canonical_json(self) -> str:
        return json.dumps(
            {
                "mode": self.mode,
                "gaussian_noise_std_m": self.gaussian_noise_std_m,
                "dropout_prob": self.dropout_prob,
                "dropout_burst_enabled": self.dropout_burst_enabled,
                "dropout_burst_prob": self.dropout_burst_prob,
                "dropout_burst_min_len": self.dropout_burst_min_len,
                "dropout_burst_max_len": self.dropout_burst_max_len,
                "seed": self.seed,
            },
            sort_keys=True,
        )

    def validate(self) -> None:
        if self.mode not in ("none", "gaussian", "dropout", "dropout_burst", "custom"):
            raise AdapterError(f"unsupported corruption mode {self.mode!r}")
        if self.gaussian_noise_std_m < 0.0 or not math.isfinite(self.gaussian_noise_std_m):
            raise AdapterError("gaussian_noise_std_m must be finite and >= 0")
        if not (0.0 <= self.dropout_prob <= 1.0):
            raise AdapterError("dropout_prob must be in [0, 1]")
        if not (0.0 <= self.dropout_burst_prob <= 1.0):
            raise AdapterError("dropout_burst_prob must be in [0, 1]")
        if self.dropout_burst_min_len < 1 or self.dropout_burst_max_len < self.dropout_burst_min_len:
            raise AdapterError("dropout_burst_min_len must be >= 1 and <= dropout_burst_max_len")


def load_segment_config(path: Path | None) -> SegmentConfig:
    if path is None:
        return SegmentConfig()
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    config = SegmentConfig(
        max_gt_gap_s=float(data.get("max_gt_gap_s", 1.0)),
        max_teleport_speed_mps=float(data.get("max_teleport_speed_mps", 60.0)),
        min_gt_samples=int(data.get("min_gt_samples", 5)),
    )
    config.validate()
    return config


def load_corruption_config(path: Path | None) -> CorruptionConfig:
    if path is None:
        config = CorruptionConfig()
        config.validate()
        return config
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    config = CorruptionConfig(
        mode=str(data.get("mode", "none")),
        gaussian_noise_std_m=float(data.get("gaussian_noise_std_m", 0.0)),
        dropout_prob=float(data.get("dropout_prob", 0.0)),
        dropout_burst_enabled=bool(data.get("dropout_burst_enabled", False)),
        dropout_burst_prob=float(data.get("dropout_burst_prob", 0.0)),
        dropout_burst_min_len=int(data.get("dropout_burst_min_len", 2)),
        dropout_burst_max_len=int(data.get("dropout_burst_max_len", 5)),
        seed=int(data.get("seed", 0)),
    )
    config.validate()
    return config


# --------------------------------------------------------------------------
# segmentation (same philosophy as kalmannet_trajectory_adapter.segment_actor)
# --------------------------------------------------------------------------
@dataclass
class Av2Segment:
    segment_id: str
    scenario_id: str
    track_id: str
    segment_index: int
    av2_object_type: str
    coarse_object_type: str
    track_category: str
    city_name: str
    states: list[Av2ObjectStateRecord]
    nonfinite_frames_dropped: int = 0

    def clean_arrays(self, timestamps_ns_by_timestep: dict[int, int]) -> dict[str, np.ndarray]:
        """Per-frame numeric arrays before coordinate transform / corruption."""
        count = len(self.states)
        stamps = np.asarray(
            [timestamps_ns_by_timestep[s.timestep] for s in self.states], dtype=np.int64
        )
        dt_s = np.empty(count, dtype=np.float64)
        dt_s[0] = np.nan
        if count > 1:
            dt_s[1:] = (stamps[1:] - stamps[:-1]).astype(np.float64) / 1e9
        pos_xy = np.asarray([(s.x, s.y) for s in self.states], dtype=np.float64)
        vel_xy = np.asarray([(s.vx, s.vy) for s in self.states], dtype=np.float64)
        gt_state = np.concatenate([pos_xy, vel_xy], axis=1)  # [T, 4] = [x, y, vx, vy]
        av2_timestep = np.asarray([s.timestep for s in self.states], dtype=np.int64)
        av2_observed = np.asarray([s.observed for s in self.states], dtype=bool)
        return {
            "timestamps_ns": stamps,
            "dt_s": dt_s,
            "gt_state": gt_state,
            "clean_position": pos_xy.copy(),
            "av2_timestep": av2_timestep,
            "av2_observed": av2_observed,
        }


def extract_segments(
    scenario: Av2ScenarioRecord, config: SegmentConfig
) -> tuple[list[Av2Segment], dict[str, int]]:
    """One (or more, if split by a gap/teleport/non-finite removal) segment
    per AV2 track. Never interpolates or fabricates a position -- a
    non-finite frame is dropped (counted), never replaced. Category
    (``TRACK_FRAGMENT`` included) never gates rejection by itself; only
    actual state finiteness/continuity and ``min_gt_samples`` do."""

    timestamps_ns_by_timestep = {i: ns for i, ns in enumerate(scenario.timestamps_ns)}
    segments: list[Av2Segment] = []
    stats = {
        "tracks_seen": 0,
        "tracks_fully_rejected_nonfinite": 0,
        "frames_dropped_nonfinite": 0,
        "frames_dropped_duplicate_timestep": 0,
        "frames_dropped_backward_timestep": 0,
        "segments_emitted": 0,
    }

    for track in scenario.tracks:
        stats["tracks_seen"] += 1
        ordered = sorted(track.states, key=lambda s: s.timestep)
        cleaned: list[Av2ObjectStateRecord] = []
        for st in ordered:
            if cleaned and st.timestep == cleaned[-1].timestep:
                stats["frames_dropped_duplicate_timestep"] += 1
                continue
            if cleaned and st.timestep < cleaned[-1].timestep:
                stats["frames_dropped_backward_timestep"] += 1
                continue
            if not _finite(st.x, st.y, st.vx, st.vy, st.heading):
                stats["frames_dropped_nonfinite"] += 1
                continue
            cleaned.append(st)

        if not cleaned:
            stats["tracks_fully_rejected_nonfinite"] += 1
            continue

        coarse = coarse_object_type(track.object_type)
        raw_segments: list[list[Av2ObjectStateRecord]] = [[cleaned[0]]]
        for prev, cur in zip(cleaned[:-1], cleaned[1:]):
            dt_s = (
                timestamps_ns_by_timestep[cur.timestep] - timestamps_ns_by_timestep[prev.timestep]
            ) / 1e9
            step_m = math.hypot(cur.x - prev.x, cur.y - prev.y)
            implied_speed = step_m / dt_s if dt_s > 0 else float("inf")
            starts_new = dt_s > config.max_gt_gap_s or implied_speed > config.max_teleport_speed_mps
            if starts_new:
                raw_segments.append([cur])
            else:
                raw_segments.append(raw_segments.pop() + [cur])

        for seg_index, seg_states in enumerate(raw_segments):
            segments.append(
                Av2Segment(
                    segment_id=f"{scenario.scenario_id}__track_{track.track_id}__segment_{seg_index:03d}",
                    scenario_id=scenario.scenario_id,
                    track_id=track.track_id,
                    segment_index=seg_index,
                    av2_object_type=track.object_type,
                    coarse_object_type=coarse,
                    track_category=track.category,
                    city_name=scenario.city_name,
                    states=seg_states,
                )
            )
            stats["segments_emitted"] += 1

    segments.sort(key=lambda s: s.segment_id)
    return segments, stats


# --------------------------------------------------------------------------
# coordinate transform (first-state-relative, versioned, reversible)
# --------------------------------------------------------------------------
def apply_coordinate_transform(
    arrays: dict[str, np.ndarray], config: CoordinateConfig
) -> dict[str, np.ndarray]:
    """``x'_t = x_t - x_0``, ``y'_t = y_t - y_0``; velocity untouched
    (translation-invariant). The origin is always this segment's own
    *first* sample -- never a mean/centroid over the whole segment, so no
    future information leaks into any timestep, including the first
    (``origin == gt_state[0, :2]`` by construction). ``origin_xy`` is
    stored so the transform is always reversible."""

    config.validate()
    out = dict(arrays)
    gt_state = arrays["gt_state"].copy()
    origin_xy = gt_state[0, :2].copy()
    gt_state[:, 0] -= origin_xy[0]
    gt_state[:, 1] -= origin_xy[1]
    out["gt_state"] = gt_state
    out["origin_xy"] = origin_xy
    return out


# --------------------------------------------------------------------------
# corruption (deterministic, minimal Stage-0 set)
# --------------------------------------------------------------------------
def _segment_corruption_seed(config_seed: int, segment_id: str) -> int:
    """Deterministic per-segment seed derived from (config.seed,
    segment_id) so corruption is reproducible regardless of processing
    order/parallelism, and two different segments never accidentally
    share one RNG stream."""
    digest = hashlib.sha256(f"{config_seed}:{segment_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 1)


def apply_corruption(
    arrays: dict[str, np.ndarray], segment_id: str, config: CorruptionConfig
) -> dict[str, np.ndarray]:
    """Produces ``measurement`` [T, KNET_MEAS_DIM] and ``measurement_valid``
    [T]. Missing frames are represented as ``measurement_valid[t] = False``
    / ``measurement[t] = NaN`` -- NEVER ``[0, 0]`` and NEVER a repeated
    previous measurement, exactly matching the existing HEVEN convention
    (``kalmannet_trajectory_adapter.build_measurement_arrays``,
    ``kalmannet_measurement_attachment.py``) so the downstream
    ``kalmannet_trajectory_loader.to_kalmannet_sequence()`` conversion
    (``z_meas[t] = None`` wherever ``measurement_valid[t]`` is False)
    feeds the existing prediction-only KalmanNet behaviour unchanged."""

    config.validate()
    clean = arrays["clean_position"]
    count = clean.shape[0]
    rng = np.random.RandomState(_segment_corruption_seed(config.seed, segment_id))

    valid = np.ones(count, dtype=bool)

    if config.dropout_prob > 0.0:
        valid &= rng.random_sample(count) >= config.dropout_prob

    if config.dropout_burst_enabled and config.dropout_burst_prob > 0.0:
        i = 0
        while i < count:
            if rng.random_sample() < config.dropout_burst_prob:
                burst_len = int(rng.randint(config.dropout_burst_min_len, config.dropout_burst_max_len + 1))
                valid[i : i + burst_len] = False
                i += burst_len
            else:
                i += 1

    measurement = np.full((count, KNET_MEAS_DIM), np.nan, dtype=np.float64)
    if config.gaussian_noise_std_m > 0.0:
        noise = rng.normal(0.0, config.gaussian_noise_std_m, size=(count, 2))
        measurement[valid] = (clean + noise)[valid]
    else:
        measurement[valid] = clean[valid]

    # A frame whose underlying clean state is itself non-finite (should
    # not occur -- extract_segments already drops non-finite frames --
    # kept as a defensive backstop, never fabricated) can never be a
    # valid measurement regardless of the dropout draw above.
    finite_clean = np.all(np.isfinite(clean), axis=1)
    valid &= finite_clean
    measurement[~valid] = np.nan

    out = dict(arrays)
    out["measurement"] = measurement
    out["measurement_valid"] = valid
    return out


# --------------------------------------------------------------------------
# stats row (mirrors kalmannet_trajectory_adapter.Trajectory.stats)
# --------------------------------------------------------------------------
def segment_stats_row(
    segment: Av2Segment, arrays: dict[str, np.ndarray], config: SegmentConfig
) -> dict[str, Any]:
    dt = arrays["dt_s"][1:]
    count = arrays["gt_state"].shape[0]
    dt_ok = bool(count >= 2 and np.all(np.isfinite(dt)) and np.all(dt > 0.0))
    state_ok = bool(np.all(np.isfinite(arrays["gt_state"])))
    valid_gt = bool(count >= config.min_gt_samples and dt_ok and state_ok)
    speeds = np.hypot(arrays["gt_state"][:, 2], arrays["gt_state"][:, 3])
    n_valid_meas = int(np.count_nonzero(arrays["measurement_valid"]))
    return {
        "segment_id": segment.segment_id,
        "scenario_id": segment.scenario_id,
        "track_id": segment.track_id,
        "segment_index": segment.segment_index,
        "av2_object_type": segment.av2_object_type,
        "coarse_object_type": segment.coarse_object_type,
        "track_category": segment.track_category,
        "city_name": segment.city_name,
        "sample_count": count,
        "av2_timestep_start": int(arrays["av2_timestep"][0]),
        "av2_timestep_end": int(arrays["av2_timestep"][-1]),
        "n_observed_true": int(np.count_nonzero(arrays["av2_observed"])),
        "dt_s_median": float(np.median(dt)) if dt.size else None,
        "dt_s_max": float(np.max(dt)) if dt.size else None,
        "speed_mps_median": float(np.median(speeds)),
        "speed_mps_max": float(np.max(speeds)),
        "origin_x": float(arrays["origin_xy"][0]),
        "origin_y": float(arrays["origin_xy"][1]),
        "coordinate_mode": COORDINATE_MODE_FIRST_STATE_RELATIVE,
        "n_measurement_valid": n_valid_meas,
        "n_measurement_invalid": count - n_valid_meas,
        "valid_for_kalmannet_gt": valid_gt,
        "kalmannet_training_ready": False,
    }


def npz_export_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Selects + orders exactly ``NPZ_ARRAY_NAMES`` for the on-disk NPZ,
    matching the array-name discipline of ``kalmannet_trajectory_adapter``."""
    return {name: np.asarray(arrays[name]) for name in NPZ_ARRAY_NAMES}


# --------------------------------------------------------------------------
# scenario-level split leakage check (AV2's split is authoritative and is
# never re-derived here -- this only guards against a bug that would
# otherwise let one scenario_id's segments land in two different output
# splits within a single multi-split invocation).
# --------------------------------------------------------------------------
def check_scenario_split_leakage(scenario_to_split: dict[str, str]) -> list[str]:
    """No-op by construction for a single-split Stage-0 run (every
    scenario maps to one split here); kept as an explicit, testable
    invariant for a future multi-split combined export."""
    seen: dict[str, str] = {}
    problems: list[str] = []
    for scenario_id, split in scenario_to_split.items():
        if scenario_id in seen and seen[scenario_id] != split:
            problems.append(
                f"scenario_id {scenario_id} assigned to both {seen[scenario_id]!r} and {split!r}"
            )
        seen[scenario_id] = split
    return problems


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------
@dataclass
class ExportResult:
    manifest: dict[str, Any]
    content_fingerprint: str


class Av2KalmanNetExporter:
    MARKER = ".av2_kalmannet_adapter"

    def __init__(
        self,
        input_root: Path,
        output_root: Path,
        split: str,
        scenario_ids: list[str],
        segment_config: SegmentConfig,
        coordinate_config: CoordinateConfig,
        corruption_config: CorruptionConfig,
        adapter_commit: str,
        scenario_manifest_provenance: dict[str, Any],
        overwrite: bool = False,
    ) -> None:
        if split not in SUPPORTED_AV2_SPLITS:
            raise AdapterError(f"unsupported split {split!r}; must be one of {SUPPORTED_AV2_SPLITS}")
        self.input_root = Path(input_root).resolve()
        self.output_root = Path(output_root).resolve()
        self.split = split
        self.scenario_ids = list(scenario_ids)
        self.segment_config = segment_config
        self.coordinate_config = coordinate_config
        self.corruption_config = corruption_config
        self.adapter_commit = adapter_commit
        self.scenario_manifest_provenance = scenario_manifest_provenance
        self.overwrite = overwrite

    def _scenario_parquet_path(self, scenario_id: str) -> Path:
        return self.input_root / self.split / scenario_id / f"scenario_{scenario_id}.parquet"

    def export(self) -> ExportResult:
        self.segment_config.validate()
        self.coordinate_config.validate()
        self.corruption_config.validate()
        if not self.scenario_ids:
            raise AdapterError("no scenario ids to export - nothing to do")

        marker = self.output_root / self.MARKER
        if self.output_root.exists():
            complete = (self.output_root / "export_manifest.json").is_file()
            if not self.overwrite:
                detail = "completed export" if complete else "existing directory"
                raise AdapterError(f"{detail} exists: {self.output_root}; pass --overwrite")
            if marker.is_file():
                shutil.rmtree(self.output_root)
            else:
                raise AdapterError(f"refusing to overwrite unrecognised directory: {self.output_root}")

        staging = self.output_root.with_name(self.output_root.name + ".tmp")
        if staging.exists():
            shutil.rmtree(staging)
        (staging / "trajectories").mkdir(parents=True, exist_ok=True)
        (staging / self.MARKER).write_text("HEVEN AV2 KalmanNet adapter\n", encoding="utf-8")

        index_lines: list[str] = []
        artifact_hashes: list[str] = []
        scenario_to_split: dict[str, str] = {}
        totals = {
            "scenarios_processed": 0,
            "tracks_seen": 0,
            "tracks_fully_rejected_nonfinite": 0,
            "frames_dropped_nonfinite": 0,
            "frames_dropped_duplicate_timestep": 0,
            "frames_dropped_backward_timestep": 0,
            "segments_exported": 0,
            "segments_valid_for_kalmannet_gt": 0,
            "gt_samples_total": 0,
        }
        av2_type_counts: dict[str, int] = {}
        coarse_type_counts: dict[str, int] = {}
        track_category_counts: dict[str, int] = {}

        for scenario_id in self.scenario_ids:
            path = self._scenario_parquet_path(scenario_id)
            if not path.is_file():
                raise AdapterError(f"scenario parquet not found: {path}")
            scenario = load_av2_scenario_parquet(path)
            if scenario.scenario_id != scenario_id:
                raise AdapterError(
                    f"scenario_id mismatch: expected {scenario_id!r}, parquet contains {scenario.scenario_id!r}"
                )
            scenario_to_split[scenario_id] = self.split
            timestamps_ns_by_timestep = {i: ns for i, ns in enumerate(scenario.timestamps_ns)}

            segments, seg_stats = extract_segments(scenario, self.segment_config)
            totals["scenarios_processed"] += 1
            for key in (
                "tracks_seen", "tracks_fully_rejected_nonfinite", "frames_dropped_nonfinite",
                "frames_dropped_duplicate_timestep", "frames_dropped_backward_timestep",
            ):
                totals[key] += seg_stats[key]

            for segment in segments:
                clean_arrays = segment.clean_arrays(timestamps_ns_by_timestep)
                transformed = apply_coordinate_transform(clean_arrays, self.coordinate_config)
                corrupted = apply_corruption(transformed, segment.segment_id, self.corruption_config)
                export_arrays = npz_export_arrays(corrupted)
                npz_bytes = _npz_bytes(export_arrays)
                npz_path = staging / "trajectories" / f"{segment.segment_id}.npz"
                tmp = npz_path.with_suffix(".npz.tmp")
                tmp.write_bytes(npz_bytes)
                tmp.replace(npz_path)

                row = segment_stats_row(segment, corrupted, self.segment_config)
                row["npz_path"] = f"trajectories/{segment.segment_id}.npz"
                index_lines.append(json.dumps(row, sort_keys=True))
                artifact_hashes.append(_sha256_bytes(npz_bytes))
                artifact_hashes.append(_sha256_bytes(json.dumps(row, sort_keys=True).encode("utf-8")))

                totals["segments_exported"] += 1
                totals["segments_valid_for_kalmannet_gt"] += int(row["valid_for_kalmannet_gt"])
                totals["gt_samples_total"] += row["sample_count"]
                av2_type_counts[segment.av2_object_type] = av2_type_counts.get(segment.av2_object_type, 0) + 1
                coarse_type_counts[segment.coarse_object_type] = coarse_type_counts.get(segment.coarse_object_type, 0) + 1
                track_category_counts[segment.track_category] = track_category_counts.get(segment.track_category, 0) + 1

        if totals["segments_exported"] == 0:
            shutil.rmtree(staging)
            raise AdapterError("no segments produced by any scenario - nothing to export")

        leakage = check_scenario_split_leakage(scenario_to_split)
        if leakage:
            shutil.rmtree(staging)
            raise AdapterError(f"scenario split leakage: {leakage}")

        (staging / "trajectory_index.jsonl").write_text(
            "".join(f"{line}\n" for line in index_lines), encoding="utf-8"
        )

        split_manifest = {
            "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
            "split": self.split,
            "split_source": "av2_native (this adapter never re-splits AV2 scenarios)",
            "scenario_count": len(self.scenario_ids),
            "scenario_ids": sorted(self.scenario_ids),
            "leakage_free": True,
            "leakage_check": "check_scenario_split_leakage (no scenario_id assigned to >1 split)",
        }
        _atomic_write_json(staging / "split_manifest.json", split_manifest)

        _atomic_write_json(
            staging / "metadata.json",
            {
                "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
                "source": "argoverse2_motion_forecasting",
                "state_fields": list(KNET_STATE_FIELDS),
                "state_dim": KNET_STATE_DIM,
                "measurement_fields": list(KNET_MEAS_FIELDS),
                "measurement_dim": KNET_MEAS_DIM,
                "time_layout": "time_major_[T, field]",
                "dt_s_index0": "nan (no previous step)",
                "coordinate_mode": self.coordinate_config.mode,
                "npz_arrays": list(NPZ_ARRAY_NAMES),
                "invalid_measurement_representation": "measurement_valid[t]=False, measurement[t]=[NaN, NaN]; "
                                                        "never [0, 0], never a repeated previous measurement",
                "class_metadata_only": True,
                "class_fed_to_kalmannet": False,
            },
        )

        fingerprint = _sha256_bytes(
            "|".join(
                [
                    ADAPTER_SCHEMA_VERSION,
                    self.split,
                    self.segment_config.canonical_json(),
                    self.coordinate_config.canonical_json(),
                    self.corruption_config.canonical_json(),
                    json.dumps(sorted(self.scenario_ids)),
                    json.dumps(split_manifest, sort_keys=True),
                    *sorted(artifact_hashes),
                ]
            ).encode("utf-8")
        )

        manifest = {
            "status": "complete",
            "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
            "dataset_source": "argoverse2_motion_forecasting",
            "split": self.split,
            "adapter_repository_commit": self.adapter_commit,
            "segment_config": json.loads(self.segment_config.canonical_json()),
            "coordinate_config": json.loads(self.coordinate_config.canonical_json()),
            "corruption_config": json.loads(self.corruption_config.canonical_json()),
            "scenario_manifest_provenance": self.scenario_manifest_provenance,
            "state_fields": list(KNET_STATE_FIELDS),
            "measurement_fields": list(KNET_MEAS_FIELDS),
            "counts": totals,
            "av2_object_type_counts": dict(sorted(av2_type_counts.items())),
            "coarse_object_type_counts": dict(sorted(coarse_type_counts.items())),
            "track_category_counts": dict(sorted(track_category_counts.items())),
            "training_ready": False,
            "created_unix_time": time.time(),
            "content_fingerprint": fingerprint,
        }
        _atomic_write_json(staging / "export_manifest.json", manifest)

        staging.replace(self.output_root)
        return ExportResult(manifest, fingerprint)
