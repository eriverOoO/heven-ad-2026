"""KalmanNet MORAI Trajectory Adapter v1.

Deterministically converts the *tracking-valid* ground-truth of a canonical
``morai_tracking_dataset_v1`` (the MORAI Tracking Dataset Factory output)
into a versioned, per-trajectory dataset for later Linear-KF replay and
KalmanNet training / validation.

**This module trains nothing.** It builds only the ground-truth trajectory
side plus an explicit, entirely-absent-in-v1 measurement-attachment
contract. No KalmanNet architecture / hyperparameter, no Linear KF, no
production tracker default, no AB3DMOT / CenterPoint / planner code is
touched or imported. No synthetic noise is written as canonical data.

Frozen research context this adapter must not contradict (do not restate
as adapter output): historical T-9A KalmanNet position RMSE ~0.854 m and
tuned Linear KF ~0.855 m are effectively tied; the dense-trained KalmanNet
v2 ~1.467 m vs tuned KF ~1.456 m; a long controlled measurement dropout
favours the tuned KF (~1.23 m) over KalmanNet (~28.92 m). This adapter
produces data only - it makes no estimator claim.

Output tree (all derived; the canonical source is never modified)::

    <derived_root>/
      .kalmannet_morai_trajectory_adapter   # marker
      metadata.json            # schema, state/measurement field contract
      export_manifest.json     # adapter schema, source hash, counts, fingerprint
      trajectory_index.jsonl   # one row per trajectory: provenance + stats + flags
      split_manifest.json      # grouping key, group + trajectory counts, leakage guarantee
      splits/{train,val,test}.txt
      trajectories/<trajectory_id>.npz   # time-major GT arrays + empty measurement slots

The KalmanNet state / measurement contract is mirrored from
``ad_lidar_perception/ad_lidar_perception/kalmannet_core.py``
(``STATE_DIM = 4`` -> ``[x, y, vx, vy]``; ``MEAS_DIM = 2`` -> ``[x, y]``);
a parity test asserts the literals still agree.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ad_morai_bridge_dev.dataset.schema import SCHEMA_VERSION as SOURCE_SCHEMA_VERSION
from ad_morai_bridge_dev.dataset.schema import DatasetPaths
from ad_morai_bridge_dev.dataset.validate import validate_dataset

# Split machinery is shared verbatim with the CenterPoint adapter so a
# (scenario_id, seed) group lands in the *same* split for both derived
# datasets - deliberate shared provenance, documented in the design doc.
from ad_morai_bridge_dev.dataset.centerpoint_adapter import (
    AdapterError,
    SplitResult,
    _group_id,
    auto_split,
    check_split_leakage,
    plan_split,
)

ADAPTER_SCHEMA_VERSION = "kalmannet_morai_trajectory_v1"
MEASUREMENT_CONTRACT_VERSION = "kalmannet_morai_measurement_contract_v1"

# Mirrored from kalmannet_core.py (parity-tested against the literals).
KNET_STATE_DIM = 4
KNET_MEAS_DIM = 2
KNET_STATE_FIELDS = ("x", "y", "vx", "vy")
KNET_MEAS_FIELDS = ("x", "y")

# Every array written into each trajectories/<id>.npz (time-major [T, ...]).
NPZ_ARRAY_NAMES = (
    "timestamps_ns",
    "dt_s",
    "gt_state",
    "gt_position_map",
    "gt_velocity_map",
    "gt_yaw",
    "source_position_map",
    "finite_diff_velocity",
    "source_frame_indices",
    "source_sample_ids",
    "actor_gt_skew_ns",
    "measurement",
    "measurement_state",
    "measurement_valid",
    "measurement_score",
    "measurement_match_distance_m",
    "measurement_match_iou",
    "measurement_box_lidar",
    "measurement_source",
    "measurement_class",
)

# GT kinematics live in the MORAI map frame (absolute, planar); this is
# the same frame the historical KalmanNet trajectory study used and it
# needs no TF chain - it is exactly what ``valid_for_tracking_gt``
# guarantees on a canonical box (pose + dims + map_frame present).
PRIMARY_FRAME = "map"

# Range bins for reporting only (hypot(x, y) in the map frame), matching
# centerpoint_adapter.py's near/mid/far cut points.
RANGE_NEAR_MAX_M = 20.0
RANGE_MID_MAX_M = 45.0


# The split machinery (plan_split / auto_split / check_split_leakage) raises
# ``centerpoint_adapter.AdapterError``; this adapter reuses that one class so
# a caller catches every adapter failure with a single ``except AdapterError``.


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TrajectoryConfig:
    # A gap larger than this between consecutive tracking-valid GT samples
    # of one actor ends the current trajectory segment. The canonical
    # anchor is nominally 10 Hz (0.1 s); the default here is 10 nominal
    # periods. The observed per-run dt distribution is recorded in the
    # export manifest so this can be retuned against real data.
    max_gt_gap_s: float = 1.0
    # An implied step speed above this ends the segment (teleport / actor
    # respawn under a reused unique_id). Highway speeds plus margin.
    max_teleport_speed_mps: float = 60.0
    # A trajectory shorter than this is still exported but flagged
    # ``valid_for_kalmannet_gt = false`` (too short to fit a filter).
    min_gt_samples: int = 5

    def canonical_json(self) -> str:
        return json.dumps(
            {
                "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
                "measurement_contract_version": MEASUREMENT_CONTRACT_VERSION,
                "primary_frame": PRIMARY_FRAME,
                "state_fields": list(KNET_STATE_FIELDS),
                "measurement_fields": list(KNET_MEAS_FIELDS),
                "max_gt_gap_s": self.max_gt_gap_s,
                "max_teleport_speed_mps": self.max_teleport_speed_mps,
                "min_gt_samples": self.min_gt_samples,
            },
            sort_keys=True,
        )

    def validate(self) -> None:
        if not (self.max_gt_gap_s > 0.0 and math.isfinite(self.max_gt_gap_s)):
            raise AdapterError("max_gt_gap_s must be a positive finite number")
        if not (
            self.max_teleport_speed_mps > 0.0
            and math.isfinite(self.max_teleport_speed_mps)
        ):
            raise AdapterError("max_teleport_speed_mps must be a positive finite number")
        if self.min_gt_samples < 2:
            raise AdapterError("min_gt_samples must be >= 2")


def load_trajectory_config(path: Path | None) -> TrajectoryConfig:
    if path is None:
        return TrajectoryConfig()
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    config = TrajectoryConfig(
        max_gt_gap_s=float(data.get("max_gt_gap_s", 1.0)),
        max_teleport_speed_mps=float(data.get("max_teleport_speed_mps", 60.0)),
        min_gt_samples=int(data.get("min_gt_samples", 5)),
    )
    config.validate()
    return config


# --------------------------------------------------------------------------
# source model
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class GtObservation:
    """One tracking-valid GT sample of one actor in one frame."""

    stamp_ns: int
    frame_index: int
    sample_id: str
    actor_gt_skew_ns: int
    class_name: str
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    yaw: float
    position_map: tuple[float, float, float]


@dataclass(frozen=True)
class RunGt:
    scenario_id: str
    run_id: str
    requested_seed: int | None
    # actor unique_id -> observations (unsorted as read)
    by_actor: dict[int, list[GtObservation]]

    @property
    def group_key(self) -> tuple[str, str]:
        if self.requested_seed is None:
            return (self.run_id, self.run_id)
        return (self.scenario_id, f"seed_{self.requested_seed}")


@dataclass
class SourceReadStats:
    frames_total: int = 0
    frames_tracking_valid: int = 0
    frames_actor_status_matched: int = 0
    boxes_seen: int = 0
    boxes_tracking_valid: int = 0
    boxes_rejected_frame_not_tracking_valid: int = 0
    boxes_rejected_actor_status_not_matched: int = 0
    boxes_rejected_box_not_tracking_valid: int = 0
    boxes_rejected_no_map_frame: int = 0
    boxes_rejected_nonfinite: int = 0
    observed_dt_s: list[float] = field(default_factory=list)


def _iter_run_dirs(paths: DatasetPaths) -> Iterable[Path]:
    for scenario_dir in sorted(p for p in paths.runs_dir.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in scenario_dir.iterdir() if p.is_dir()):
            yield run_dir


def _finite(*values: float) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def read_source_runs(
    source_root: Path,
) -> tuple[list[RunGt], SourceReadStats]:
    """Every tracking-valid actor GT sample, grouped by (run, actor)."""

    paths = DatasetPaths(Path(source_root))
    runs: list[RunGt] = []
    stats = SourceReadStats()
    for run_dir in _iter_run_dirs(paths):
        manifest = json.loads((run_dir / "run_manifest.json").read_text("utf-8"))
        seed_block = manifest.get("config", {}).get("seed", {})
        requested_seed = seed_block.get("requested_seed")
        scenario_id = seed_block.get("scenario_id") or manifest["scenario_id"]
        run_id = manifest["run_id"]

        by_actor: dict[int, list[GtObservation]] = {}
        for line in (run_dir / "frames.jsonl").read_text("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            stats.frames_total += 1
            frame_tracking_valid = bool(row.get("valid_for_tracking_gt"))
            actor_gt = row.get("actor_gt") or {}
            actor_status_matched = actor_gt.get("status") == "matched"
            if frame_tracking_valid:
                stats.frames_tracking_valid += 1
            if actor_status_matched:
                stats.frames_actor_status_matched += 1

            frame_index = int(row["frame_index"])
            gt_path = run_dir / "gt" / f"{frame_index:06d}.json"
            gt = json.loads(gt_path.read_text("utf-8"))
            skew_ns = int(actor_gt.get("skew_ns") or 0)

            for box in gt.get("boxes", []):
                stats.boxes_seen += 1
                if not frame_tracking_valid:
                    stats.boxes_rejected_frame_not_tracking_valid += 1
                    continue
                if not actor_status_matched:
                    stats.boxes_rejected_actor_status_not_matched += 1
                    continue
                if not bool(box.get("valid_for_tracking_gt")):
                    stats.boxes_rejected_box_not_tracking_valid += 1
                    continue
                map_frame = box.get("map_frame")
                if not isinstance(map_frame, dict) or "center" not in map_frame:
                    stats.boxes_rejected_no_map_frame += 1
                    continue
                source = box.get("source") or {}
                center = [float(v) for v in map_frame["center"]]
                velocity = [float(v) for v in source.get("velocity_map", [])]
                position = [float(v) for v in source.get("position_map", [])]
                yaw = float(map_frame.get("yaw"))
                if len(center) != 3 or len(velocity) != 3 or len(position) != 3:
                    stats.boxes_rejected_nonfinite += 1
                    continue
                if not _finite(*center, *velocity, *position, yaw):
                    stats.boxes_rejected_nonfinite += 1
                    continue
                stats.boxes_tracking_valid += 1
                actor_id = int(box["actor_id"])
                by_actor.setdefault(actor_id, []).append(
                    GtObservation(
                        stamp_ns=int(row["lidar_header_stamp_ns"]),
                        frame_index=frame_index,
                        sample_id=str(row["sample_id"]),
                        actor_gt_skew_ns=skew_ns,
                        class_name=str(box.get("class_name")),
                        x=center[0],
                        y=center[1],
                        z=center[2],
                        vx=velocity[0],
                        vy=velocity[1],
                        vz=velocity[2],
                        yaw=yaw,
                        position_map=(position[0], position[1], position[2]),
                    )
                )

        if by_actor:
            runs.append(
                RunGt(
                    scenario_id=scenario_id,
                    run_id=run_id,
                    requested_seed=(
                        int(requested_seed) if requested_seed is not None else None
                    ),
                    by_actor=by_actor,
                )
            )
    runs.sort(key=lambda r: (r.scenario_id, r.run_id))
    return runs, stats


# --------------------------------------------------------------------------
# trajectory segmentation
# --------------------------------------------------------------------------
@dataclass
class Trajectory:
    trajectory_id: str
    scenario_id: str
    run_id: str
    requested_seed: int | None
    actor_id: int
    segment_index: int
    class_name: str
    observations: list[GtObservation]
    duplicate_stamps_dropped: int = 0
    backward_stamps_dropped: int = 0

    @property
    def group_key(self) -> tuple[str, str]:
        if self.requested_seed is None:
            return (self.run_id, self.run_id)
        return (self.scenario_id, f"seed_{self.requested_seed}")

    def npz_arrays(self) -> dict[str, np.ndarray]:
        obs = self.observations
        count = len(obs)
        stamps = np.asarray([o.stamp_ns for o in obs], dtype=np.int64)
        dt_s = np.empty(count, dtype=np.float64)
        dt_s[0] = np.nan
        if count > 1:
            dt_s[1:] = (stamps[1:] - stamps[:-1]).astype(np.float64) / 1e9

        pos_xy = np.asarray([(o.x, o.y) for o in obs], dtype=np.float64)
        vel_xy = np.asarray([(o.vx, o.vy) for o in obs], dtype=np.float64)
        gt_state = np.concatenate([pos_xy, vel_xy], axis=1)  # [T, 4] time-major

        gt_position_map = np.asarray(
            [(o.x, o.y, o.z) for o in obs], dtype=np.float64
        )
        gt_velocity_map = np.asarray(
            [(o.vx, o.vy, o.vz) for o in obs], dtype=np.float64
        )
        gt_yaw = np.asarray(
            [math.atan2(math.sin(o.yaw), math.cos(o.yaw)) for o in obs],
            dtype=np.float64,
        )
        source_position_map = np.asarray(
            [o.position_map for o in obs], dtype=np.float64
        )

        # Diagnostic only - the canonical velocity above is the native
        # MORAI source velocity, never this finite difference.
        finite_diff_velocity = np.full((count, 2), np.nan, dtype=np.float64)
        if count >= 2:
            for i in range(count):
                lo = max(0, i - 1)
                hi = min(count - 1, i + 1)
                span = (stamps[hi] - stamps[lo]) / 1e9
                if span > 0:
                    finite_diff_velocity[i] = (
                        pos_xy[hi] - pos_xy[lo]
                    ) / span

        return {
            "timestamps_ns": stamps,
            "dt_s": dt_s,
            "gt_state": gt_state,
            "gt_position_map": gt_position_map,
            "gt_velocity_map": gt_velocity_map,
            "gt_yaw": gt_yaw,
            "source_position_map": source_position_map,
            "finite_diff_velocity": finite_diff_velocity,
            "source_frame_indices": np.asarray(
                [o.frame_index for o in obs], dtype=np.int64
            ),
            "source_sample_ids": np.asarray(
                [o.sample_id for o in obs], dtype="<U128"
            ),
            "actor_gt_skew_ns": np.asarray(
                [o.actor_gt_skew_ns for o in obs], dtype=np.int64
            ),
            # --- measurement-attachment contract: entirely empty in v1 ---
            "measurement": np.full((count, KNET_MEAS_DIM), np.nan, dtype=np.float64),
            "measurement_state": np.full(
                (count, KNET_STATE_DIM), np.nan, dtype=np.float64
            ),
            "measurement_valid": np.zeros(count, dtype=bool),
            "measurement_score": np.full(count, np.nan, dtype=np.float64),
            "measurement_match_distance_m": np.full(count, np.nan, dtype=np.float64),
            "measurement_match_iou": np.full(count, np.nan, dtype=np.float64),
            "measurement_box_lidar": np.full((count, 7), np.nan, dtype=np.float64),
            "measurement_source": np.full(count, "", dtype="<U64"),
            "measurement_class": np.full(count, "", dtype="<U32"),
        }

    def stats(self, config: TrajectoryConfig) -> dict[str, Any]:
        arrays = self.npz_arrays()
        dt = arrays["dt_s"][1:]
        speeds = np.hypot(arrays["gt_velocity_map"][:, 0], arrays["gt_velocity_map"][:, 1])
        ranges = np.hypot(arrays["gt_position_map"][:, 0], arrays["gt_position_map"][:, 1])
        count = len(self.observations)
        dt_ok = bool(count >= 2 and np.all(np.isfinite(dt)) and np.all(dt > 0.0))
        state_ok = bool(np.all(np.isfinite(arrays["gt_state"])))
        valid_gt = bool(count >= config.min_gt_samples and dt_ok and state_ok)
        return {
            "trajectory_id": self.trajectory_id,
            "scenario_id": self.scenario_id,
            "run_id": self.run_id,
            "requested_seed": self.requested_seed,
            "actor_id": self.actor_id,
            "segment_index": self.segment_index,
            "class_name": self.class_name,
            "group_id": _group_id(self.group_key),
            "sample_count": count,
            "start_stamp_ns": int(self.observations[0].stamp_ns),
            "end_stamp_ns": int(self.observations[-1].stamp_ns),
            "duration_s": (
                float(self.observations[-1].stamp_ns - self.observations[0].stamp_ns)
                / 1e9
            ),
            "source_frame_start": int(self.observations[0].frame_index),
            "source_frame_end": int(self.observations[-1].frame_index),
            "duplicate_stamps_dropped": self.duplicate_stamps_dropped,
            "backward_stamps_dropped": self.backward_stamps_dropped,
            "dt_s_median": (float(np.median(dt)) if dt.size else None),
            "dt_s_p95": (
                float(np.percentile(dt, 95)) if dt.size else None
            ),
            "dt_s_max": (float(np.max(dt)) if dt.size else None),
            "speed_mps_median": float(np.median(speeds)),
            "speed_mps_max": float(np.max(speeds)),
            "range_counts": {
                "near": int(np.count_nonzero(ranges <= RANGE_NEAR_MAX_M)),
                "mid": int(
                    np.count_nonzero(
                        (ranges > RANGE_NEAR_MAX_M) & (ranges <= RANGE_MID_MAX_M)
                    )
                ),
                "far": int(np.count_nonzero(ranges > RANGE_MID_MAX_M)),
            },
            "native_velocity_used": True,
            "finite_difference_velocity_is_diagnostic_only": True,
            "measurement_contract_version": MEASUREMENT_CONTRACT_VERSION,
            "measurement_assignment_method": None,
            "measurement_frame_count": 0,
            "valid_for_kalmannet_gt": valid_gt,
            "kalmannet_training_ready": False,
        }


def segment_actor(
    run: RunGt,
    actor_id: int,
    observations: list[GtObservation],
    config: TrajectoryConfig,
) -> list[Trajectory]:
    """Order one actor's samples in time and split on gap / teleport /
    backward time. Run boundaries are absolute (this is called per run)."""

    ordered = sorted(observations, key=lambda o: o.stamp_ns)
    cleaned: list[GtObservation] = []
    duplicate_dropped = 0
    backward_dropped = 0
    for obs in ordered:
        if cleaned and obs.stamp_ns == cleaned[-1].stamp_ns:
            duplicate_dropped += 1
            continue
        if cleaned and obs.stamp_ns < cleaned[-1].stamp_ns:
            backward_dropped += 1
            continue
        cleaned.append(obs)

    if not cleaned:
        return []

    class_name = cleaned[0].class_name
    segments: list[list[GtObservation]] = [[cleaned[0]]]
    for prev, obs in zip(cleaned[:-1], cleaned[1:]):
        dt_s = (obs.stamp_ns - prev.stamp_ns) / 1e9
        step = math.hypot(obs.x - prev.x, obs.y - prev.y)
        implied_speed = step / dt_s if dt_s > 0 else float("inf")
        new_segment = (
            dt_s > config.max_gt_gap_s
            or implied_speed > config.max_teleport_speed_mps
        )
        if new_segment:
            segments.append([obs])
        else:
            segments.append(segments.pop() + [obs])

    trajectories: list[Trajectory] = []
    for seg_index, seg in enumerate(segments):
        trajectory_id = (
            f"{run.run_id}__actor_{actor_id}__segment_{seg_index:03d}"
        )
        trajectories.append(
            Trajectory(
                trajectory_id=trajectory_id,
                scenario_id=run.scenario_id,
                run_id=run.run_id,
                requested_seed=run.requested_seed,
                actor_id=actor_id,
                segment_index=seg_index,
                class_name=class_name,
                observations=seg,
                # First segment carries the drop tally for the whole actor.
                duplicate_stamps_dropped=duplicate_dropped if seg_index == 0 else 0,
                backward_stamps_dropped=backward_dropped if seg_index == 0 else 0,
            )
        )
    return trajectories


def build_trajectories(
    runs: list[RunGt], config: TrajectoryConfig
) -> list[Trajectory]:
    trajectories: list[Trajectory] = []
    for run in runs:
        for actor_id in sorted(run.by_actor):
            trajectories.extend(
                segment_actor(run, actor_id, run.by_actor[actor_id], config)
            )
    trajectories.sort(key=lambda t: t.trajectory_id)
    return trajectories


# --------------------------------------------------------------------------
# measurement attachment contract (interface only - no detector in v1)
# --------------------------------------------------------------------------
def build_measurement_arrays(
    trajectory: Trajectory,
    per_frame_measurements: dict[int, dict[str, Any]],
    *,
    assignment_method: str,
    source: str,
) -> dict[str, np.ndarray]:
    """Overlay measurement arrays onto a trajectory's GT frames.

    ``per_frame_measurements`` maps a ``source_frame_index`` to a dict with
    keys ``state`` ([x, y]), optionally ``velocity`` ([vx, vy]), ``score``,
    ``class``, ``box_lidar`` ([x, y, z, l, w, h, yaw]), ``match_distance_m``,
    ``match_iou``. Frames not present stay masked out.

    No forward-fill, no interpolation, no detector inference, no synthetic
    noise. A frame is either genuinely paired to a real detector output or
    it is ``measurement_valid = False``. This is the contract the future
    Euclidean / CenterPoint measurement pass must satisfy; v1 exports call
    this with an empty mapping (or not at all).
    """

    obs = trajectory.observations
    count = len(obs)
    out = {
        "measurement": np.full((count, KNET_MEAS_DIM), np.nan, dtype=np.float64),
        "measurement_state": np.full(
            (count, KNET_STATE_DIM), np.nan, dtype=np.float64
        ),
        "measurement_valid": np.zeros(count, dtype=bool),
        "measurement_score": np.full(count, np.nan, dtype=np.float64),
        "measurement_match_distance_m": np.full(count, np.nan, dtype=np.float64),
        "measurement_match_iou": np.full(count, np.nan, dtype=np.float64),
        "measurement_box_lidar": np.full((count, 7), np.nan, dtype=np.float64),
        "measurement_source": np.full(count, "", dtype="<U64"),
        "measurement_class": np.full(count, "", dtype="<U32"),
    }
    frame_to_row = {o.frame_index: i for i, o in enumerate(obs)}
    for frame_index, record in per_frame_measurements.items():
        if frame_index not in frame_to_row:
            raise AdapterError(
                f"measurement for frame {frame_index} not in trajectory "
                f"{trajectory.trajectory_id}"
            )
        row = frame_to_row[frame_index]
        state = [float(v) for v in record["state"]]
        if len(state) != KNET_MEAS_DIM or not _finite(*state):
            raise AdapterError(f"bad measurement state for frame {frame_index}")
        out["measurement"][row] = state
        out["measurement_state"][row, :KNET_MEAS_DIM] = state
        velocity = record.get("velocity")
        if velocity is not None:
            vel = [float(v) for v in velocity]
            if len(vel) != 2 or not _finite(*vel):
                raise AdapterError(f"bad measurement velocity for frame {frame_index}")
            out["measurement_state"][row, KNET_MEAS_DIM:] = vel
        out["measurement_valid"][row] = True
        if record.get("score") is not None:
            out["measurement_score"][row] = float(record["score"])
        if record.get("match_distance_m") is not None:
            out["measurement_match_distance_m"][row] = float(record["match_distance_m"])
        if record.get("match_iou") is not None:
            out["measurement_match_iou"][row] = float(record["match_iou"])
        box = record.get("box_lidar")
        if box is not None:
            box = [float(v) for v in box]
            if len(box) != 7:
                raise AdapterError(f"bad measurement box_lidar for frame {frame_index}")
            out["measurement_box_lidar"][row] = box
        out["measurement_source"][row] = str(source)
        out["measurement_class"][row] = str(record.get("class", ""))
    return out


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------
def _atomic_write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """A deterministic ``.npz`` (uncompressed, fixed 1980-01-01 zip stamps,
    sorted members) so a repeat export is byte-identical. ``np.load`` reads
    it exactly like a ``np.savez`` archive."""

    import io
    import zipfile

    import numpy.lib.format as npy_format

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            member = io.BytesIO()
            npy_format.write_array(
                member, np.asarray(arrays[name]), allow_pickle=False
            )
            archive.writestr(zipfile.ZipInfo(f"{name}.npy"), member.getvalue())
    return buffer.getvalue()


@dataclass
class ExportResult:
    manifest: dict[str, Any]
    split_manifest: dict[str, Any]
    content_fingerprint: str


class KalmanNetTrajectoryExporter:
    MARKER = ".kalmannet_morai_trajectory_adapter"

    def __init__(
        self,
        source_root: Path,
        output_root: Path,
        config: TrajectoryConfig,
        adapter_commit: str,
        overwrite: bool = False,
    ) -> None:
        self.source_root = Path(source_root).resolve()
        self.output_root = Path(output_root).resolve()
        self.config = config
        self.adapter_commit = adapter_commit
        self.overwrite = overwrite
        self._source_manifest_sha256: str | None = None
        self._source_dataset_id: str | None = None

    # -- source gate ---------------------------------------------------
    def validate_source(self) -> None:
        manifest_path = self.source_root / "dataset_manifest.json"
        if not manifest_path.is_file():
            raise AdapterError(f"{manifest_path} missing (not a canonical dataset)")
        source_manifest = json.loads(manifest_path.read_text("utf-8"))
        if source_manifest.get("schema_version") != SOURCE_SCHEMA_VERSION:
            raise AdapterError(
                f"source schema {source_manifest.get('schema_version')!r} != "
                f"{SOURCE_SCHEMA_VERSION!r}"
            )
        report = validate_dataset(self.source_root)
        if not report.ok:
            raise AdapterError(
                f"source dataset failed the factory validator: {report.errors[:5]}"
            )

    def source_manifest_sha256(self) -> str:
        if self._source_manifest_sha256 is None:
            self._source_manifest_sha256 = _sha256_bytes(
                (self.source_root / "dataset_manifest.json").read_bytes()
            )
        return self._source_manifest_sha256

    def source_dataset_id(self) -> str:
        if self._source_dataset_id is None:
            self._source_dataset_id = json.loads(
                (self.source_root / "dataset_manifest.json").read_text("utf-8")
            ).get("dataset_id", "unknown")
        return self._source_dataset_id

    # -- planning ----------------------------------------------------
    def plan(self, split_plan: dict[str, Any] | None) -> dict[str, Any]:
        self.validate_source()
        runs, read_stats = read_source_runs(self.source_root)
        trajectories = build_trajectories(runs, self.config)
        if not trajectories:
            raise AdapterError(
                "no tracking-valid trajectories in the source dataset - nothing "
                "to export (an empty trajectory dataset is never written)"
            )
        groups = [t.group_key for t in trajectories]
        split = (
            plan_split(groups, split_plan) if split_plan else auto_split(groups)
        )
        by_split: dict[str, int] = {"train": 0, "val": 0, "test": 0}
        dropped = 0
        for trajectory in trajectories:
            target = split.assignments.get(_group_id(trajectory.group_key))
            if target is None:
                dropped += 1
            else:
                by_split[target] += 1

        observed_dt = _observed_dt_summary(trajectories)
        return {
            "run_count": len(runs),
            "trajectory_count": len(trajectories),
            "dropped_by_split_plan": dropped,
            "split_method": split.method,
            "split_group_counts": {
                s: sum(1 for v in split.assignments.values() if v == s)
                for s in ("train", "val", "test")
            },
            "split_trajectory_counts": by_split,
            "split_warnings": split.warnings,
            "observed_dt_s": observed_dt,
            "source_read_stats": _read_stats_dict(read_stats),
            "_trajectories": trajectories,
            "_split": split,
            "_runs": runs,
        }

    # -- export ----------------------------------------------------
    def export(self, split_plan: dict[str, Any] | None) -> ExportResult:
        marker = self.output_root / self.MARKER
        if self.output_root.exists():
            complete = (self.output_root / "export_manifest.json").is_file()
            if not self.overwrite:
                detail = "completed export" if complete else "existing directory"
                raise AdapterError(
                    f"{detail} exists: {self.output_root}; pass --overwrite"
                )
            if marker.is_file():
                shutil.rmtree(self.output_root)
            else:
                raise AdapterError(
                    f"refusing to overwrite unrecognised directory: {self.output_root}"
                )

        staging = self.output_root.with_name(self.output_root.name + ".tmp")
        if staging.exists():
            shutil.rmtree(staging)
        for sub in ("trajectories", "splits"):
            (staging / sub).mkdir(parents=True, exist_ok=True)
        (staging / self.MARKER).write_text(
            "HEVEN KalmanNet MORAI trajectory adapter\n", encoding="utf-8"
        )

        preview = self.plan(split_plan)
        trajectories: list[Trajectory] = preview.pop("_trajectories")
        split: SplitResult = preview.pop("_split")
        preview.pop("_runs", None)

        split_lists: dict[str, list[str]] = {"train": [], "val": [], "test": []}
        index_lines: list[str] = []
        traj_to_split: dict[str, str] = {}
        traj_to_group: dict[str, str] = {}
        traj_to_run: dict[str, str] = {}
        artifact_hashes: list[str] = []
        stats_by_split: dict[str, list[dict[str, Any]]] = {
            s: [] for s in ("train", "val", "test")
        }
        totals = {
            "trajectories_exported": 0,
            "trajectories_valid_for_kalmannet_gt": 0,
            "gt_samples_total": 0,
            "duplicate_stamps_dropped": 0,
            "backward_stamps_dropped": 0,
            "measurement_frames_attached": 0,
        }
        class_counts: dict[str, dict[str, int]] = {
            s: {} for s in ("train", "val", "test")
        }

        for trajectory in trajectories:
            gid = _group_id(trajectory.group_key)
            target = split.assignments.get(gid)
            if target is None:
                continue
            tid = trajectory.trajectory_id
            if tid in traj_to_split:
                raise AdapterError(f"duplicate trajectory_id {tid}")

            arrays = trajectory.npz_arrays()
            npz_bytes = _npz_bytes(arrays)
            npz_path = staging / "trajectories" / f"{tid}.npz"
            tmp = npz_path.with_suffix(".npz.tmp")
            tmp.write_bytes(npz_bytes)
            tmp.replace(npz_path)

            row = trajectory.stats(self.config)
            row["split"] = target
            row["npz_path"] = f"trajectories/{tid}.npz"
            row["source_dataset_id"] = self.source_dataset_id()
            index_lines.append(json.dumps(row, sort_keys=True))

            split_lists[target].append(tid)
            traj_to_split[tid] = target
            traj_to_group[tid] = gid
            traj_to_run[tid] = trajectory.run_id
            stats_by_split[target].append(row)
            artifact_hashes.append(_sha256_bytes(npz_bytes))
            artifact_hashes.append(
                _sha256_bytes(json.dumps(row, sort_keys=True).encode("utf-8"))
            )

            totals["trajectories_exported"] += 1
            totals["trajectories_valid_for_kalmannet_gt"] += int(
                row["valid_for_kalmannet_gt"]
            )
            totals["gt_samples_total"] += row["sample_count"]
            totals["duplicate_stamps_dropped"] += row["duplicate_stamps_dropped"]
            totals["backward_stamps_dropped"] += row["backward_stamps_dropped"]
            class_counts[target][trajectory.class_name] = (
                class_counts[target].get(trajectory.class_name, 0) + 1
            )

        leakage = check_split_leakage(traj_to_split, traj_to_group, traj_to_run)
        if leakage:
            raise AdapterError(f"split leakage: {leakage}")

        for split_name, ids in split_lists.items():
            (staging / "splits" / f"{split_name}.txt").write_text(
                "".join(f"{i}\n" for i in ids), encoding="utf-8"
            )
        (staging / "trajectory_index.jsonl").write_text(
            "".join(f"{line}\n" for line in index_lines), encoding="utf-8"
        )

        _atomic_write_json(
            staging / "metadata.json",
            {
                "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
                "measurement_contract_version": MEASUREMENT_CONTRACT_VERSION,
                "primary_frame": PRIMARY_FRAME,
                "state_fields": list(KNET_STATE_FIELDS),
                "state_dim": KNET_STATE_DIM,
                "measurement_fields": list(KNET_MEAS_FIELDS),
                "measurement_dim": KNET_MEAS_DIM,
                "time_layout": "time_major_[T, field]",
                "dt_s_index0": "nan (no previous step)",
                "velocity_source": "native MORAI source.velocity_map (never finite difference)",
                "finite_diff_velocity": "diagnostic array only",
                "measurements_present": False,
                "npz_arrays": list(NPZ_ARRAY_NAMES),
            },
        )

        split_manifest = {
            "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
            "split_method": split.method,
            "grouping_key": split.grouping_key,
            "leakage_free": True,
            "shared_with_centerpoint_adapter_auto_split": (
                split.method.startswith("auto")
            ),
            "group_splits": dict(sorted(split.assignments.items())),
            "group_counts": preview["split_group_counts"],
            "trajectory_counts": {s: len(v) for s, v in split_lists.items()},
            "valid_for_kalmannet_gt_counts": {
                s: sum(1 for r in stats_by_split[s] if r["valid_for_kalmannet_gt"])
                for s in ("train", "val", "test")
            },
            "class_counts_by_split": {
                s: dict(sorted(c.items())) for s, c in class_counts.items()
            },
            "warnings": split.warnings,
            "source_run_ids": sorted(traj_to_run.values()),
        }
        _atomic_write_json(staging / "split_manifest.json", split_manifest)

        fingerprint = _sha256_bytes(
            "|".join(
                [
                    ADAPTER_SCHEMA_VERSION,
                    self.source_manifest_sha256(),
                    self.config.canonical_json(),
                    json.dumps(split_manifest, sort_keys=True),
                    *sorted(artifact_hashes),
                ]
            ).encode("utf-8")
        )

        manifest = {
            "status": "complete",
            "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
            "measurement_contract_version": MEASUREMENT_CONTRACT_VERSION,
            "source_dataset_schema_version": SOURCE_SCHEMA_VERSION,
            "source_dataset_id": self.source_dataset_id(),
            "source_dataset_manifest_sha256": self.source_manifest_sha256(),
            "adapter_repository_commit": self.adapter_commit,
            "adapter_config": json.loads(self.config.canonical_json()),
            "primary_frame": PRIMARY_FRAME,
            "state_fields": list(KNET_STATE_FIELDS),
            "measurement_fields": list(KNET_MEAS_FIELDS),
            "trajectory_count": len(traj_to_split),
            "split_trajectory_counts": {s: len(v) for s, v in split_lists.items()},
            "counts": {
                **totals,
                "run_count": preview["run_count"],
                "dropped_by_split_plan": preview["dropped_by_split_plan"],
                "source_read_stats": preview["source_read_stats"],
            },
            "observed_dt_s": preview["observed_dt_s"],
            "measurements_present": False,
            "training_ready": False,
            "content_fingerprint": fingerprint,
        }
        _atomic_write_json(staging / "export_manifest.json", manifest)

        staging.replace(self.output_root)
        return ExportResult(manifest, split_manifest, fingerprint)


def _observed_dt_summary(trajectories: list[Trajectory]) -> dict[str, Any]:
    all_dt: list[float] = []
    for trajectory in trajectories:
        dt = trajectory.npz_arrays()["dt_s"][1:]
        all_dt.extend(float(v) for v in dt if math.isfinite(v))
    if not all_dt:
        return {"count": 0}
    arr = np.asarray(all_dt, dtype=np.float64)
    return {
        "count": int(arr.size),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "nominal_rate_hz_assumed": 10.0,
        "max_gt_gap_s_derivation": "10 x nominal 0.1 s period",
    }


def _read_stats_dict(stats: SourceReadStats) -> dict[str, Any]:
    return {
        "frames_total": stats.frames_total,
        "frames_tracking_valid": stats.frames_tracking_valid,
        "frames_actor_status_matched": stats.frames_actor_status_matched,
        "boxes_seen": stats.boxes_seen,
        "boxes_tracking_valid": stats.boxes_tracking_valid,
        "boxes_rejected_frame_not_tracking_valid": (
            stats.boxes_rejected_frame_not_tracking_valid
        ),
        "boxes_rejected_actor_status_not_matched": (
            stats.boxes_rejected_actor_status_not_matched
        ),
        "boxes_rejected_box_not_tracking_valid": (
            stats.boxes_rejected_box_not_tracking_valid
        ),
        "boxes_rejected_no_map_frame": stats.boxes_rejected_no_map_frame,
        "boxes_rejected_nonfinite": stats.boxes_rejected_nonfinite,
    }
