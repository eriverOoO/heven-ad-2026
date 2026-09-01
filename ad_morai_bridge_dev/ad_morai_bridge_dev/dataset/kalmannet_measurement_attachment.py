"""KalmanNet Detector Measurement Attachment v1.

Consumes (A) a ``kalmannet_morai_trajectory_v1`` export (GT trajectory side)
and (B) sample-aligned **real detector outputs**, and produces a
``kalmannet_morai_measurement_v1`` dataset: the same GT trajectories, byte-for-
byte unchanged, with the measurement slots populated **only** where a real
detector observation was supervised-matched to the GT actor.

**This module trains nothing.** No optimizer step, no synthetic Gaussian
noise, no simulated dropout, no filled / interpolated / forward-filled
measurements, no detector inference, no KalmanNet / Linear KF / CenterPoint /
AB3DMOT / planner change, no frozen KNet/KF conclusion touched. It is a
supervised data-construction layer.

Frozen research context preserved (data only): historical T-9A KalmanNet
~0.854 m vs tuned Linear KF ~0.855 m (tied); dense v2 ~1.467 m vs ~1.456 m
position, ~2.462 vs ~2.416 m/s velocity; long controlled dropout favours the
tuned KF (~1.23 m) over KalmanNet (~28.92 m). No estimator claim is made here.

Three distinct concepts (only #2 is handled):
  1. GT presence      - actor has valid simulator ground truth.
  2. detector measurement presence - a real detection can be associated to
     the GT actor (SUPERVISED, for dataset construction).
  3. runtime tracker association - NOT implemented or evaluated here.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ad_morai_bridge_dev.dataset.geometry import RigidTransform
from ad_morai_bridge_dev.dataset.schema import SCHEMA_VERSION as SOURCE_DATASET_SCHEMA
from ad_morai_bridge_dev.dataset.schema import DatasetPaths
from ad_morai_bridge_dev.dataset.kalmannet_trajectory_adapter import (
    ADAPTER_SCHEMA_VERSION as TRAJECTORY_SCHEMA_VERSION,
    KNET_MEAS_DIM,
    KNET_STATE_DIM,
    NPZ_ARRAY_NAMES as TRAJECTORY_NPZ_ARRAYS,
    AdapterError,
)
from ad_morai_bridge_dev.dataset.kalmannet_supervised_association import (
    ASSIGNMENT_ALGORITHM,
    ASSOCIATION_METRIC,
    Detection,
    GtActor,
    associate_frame,
)
from ad_morai_bridge_dev.dataset.kalmannet_trajectory_loader import load_trajectory

MEASUREMENT_SCHEMA_VERSION = "kalmannet_morai_measurement_v1"
DETECTOR_RECORD_SCHEMA_VERSION = "kalmannet_detector_measurements_v1"

# Serialization contracts this adapter reads, both already in the repo.
HEVEN_OFFLINE_DETECTION_SCHEMA = "heven.offline_detection.v1"       # CenterPoint (prediction_bridge.py)
HEVEN_ROS_DETECTION_COMPARISON_SCHEMA = "heven.ros_detection_comparison.v1"  # Euclidean (detection_recording.py)

DETECTOR_FRAME_ID = "lidar_link"

# The measurement arrays this adapter (re)writes onto a trajectory. Everything
# else in TRAJECTORY_NPZ_ARRAYS is copied through byte/numeric-identical.
MEASUREMENT_ARRAY_NAMES = (
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
GT_ARRAY_NAMES = tuple(n for n in TRAJECTORY_NPZ_ARRAYS if n not in MEASUREMENT_ARRAY_NAMES)

MISS_REASONS = (
    "detector_frame_missing",   # no detector record for this canonical sample
    "zero_detections",          # detector ran, produced 0 detections
    "no_map_transform",         # canonical frame lacks the map->lidar chain
    "gated_out",                # a detection existed but no candidate within gate/class
    "assigned_elsewhere",       # a candidate detection went to another GT actor
)


# --------------------------------------------------------------------------
# association config
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class AssociationConfig:
    # BEV centre-distance gate. NOT the AB3DMOT runtime association gate and
    # NOT the CenterPoint evaluator's IoU threshold - it is a supervised
    # label-construction gate, defaulted from the MORAI vehicle footprint
    # (~4.6 x 1.9 m -> half-diagonal ~= 2.5 m).
    max_distance_m: float = 2.5
    assignment: str = ASSIGNMENT_ALGORITHM
    metric: str = ASSOCIATION_METRIC
    # Overridden to False automatically for a class-agnostic detector source.
    class_matching: bool = True

    def canonical_json(self, *, effective_class_matching: bool) -> str:
        return json.dumps(
            {
                "measurement_schema_version": MEASUREMENT_SCHEMA_VERSION,
                "metric": self.metric,
                "assignment": self.assignment,
                "max_distance_m": self.max_distance_m,
                "class_matching": effective_class_matching,
                "gate_provenance": (
                    "MORAI vehicle footprint half-diagonal (~2.5 m); not the "
                    "AB3DMOT runtime euclidean_gate_m and not the evaluator IoU "
                    "threshold"
                ),
            },
            sort_keys=True,
        )

    def validate(self) -> None:
        if not (self.max_distance_m > 0.0 and math.isfinite(self.max_distance_m)):
            raise AdapterError("max_distance_m must be a positive finite number")
        if self.assignment != ASSIGNMENT_ALGORITHM:
            raise AdapterError(f"only {ASSIGNMENT_ALGORITHM!r} assignment is supported")
        if self.metric != ASSOCIATION_METRIC:
            raise AdapterError(f"only {ASSOCIATION_METRIC!r} metric is supported")


def load_association_config(path: Path | None) -> AssociationConfig:
    if path is None:
        return AssociationConfig()
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    assoc = (data.get("association") or {}) if isinstance(data.get("association"), dict) else data
    config = AssociationConfig(
        max_distance_m=float(assoc.get("max_distance_m", 2.5)),
        assignment=str(assoc.get("assignment", ASSIGNMENT_ALGORITHM)),
        metric=str(assoc.get("metric", ASSOCIATION_METRIC)),
        class_matching=bool(assoc.get("class_matching", True)),
    )
    config.validate()
    return config


# --------------------------------------------------------------------------
# normalized detector record  (kalmannet_detector_measurements_v1)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class NormDetection:
    index: int
    class_name: str
    center_lidar: tuple[float, float, float]
    dims_lwh: tuple[float, float, float]
    yaw: float
    yaw_available: bool
    score: float


@dataclass(frozen=True)
class DetectorFrame:
    sample_id: str
    lidar_stamp_ns: int
    detector_frame_present: bool
    detections: tuple[NormDetection, ...]


@dataclass
class DetectorRecordSet:
    detector_source: str
    detector_config_id: str
    class_semantics: str  # "class_aware" | "class_agnostic"
    detector_manifest_sha256: str
    canonical_dataset_id: str | None
    provenance: dict[str, Any]
    completeness: dict[str, Any]
    frames: dict[str, DetectorFrame]

    @property
    def class_matching_supported(self) -> bool:
        return self.class_semantics == "class_aware"


def _wrap_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _finite(*values: float) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def centerpoint_records_from_heven_offline(
    jsonl_path: Path,
    *,
    detector_config_id: str,
    canonical_dataset_id: str | None = None,
    detector_manifest: dict[str, Any] | None = None,
) -> DetectorRecordSet:
    """CenterPoint predictions serialized by
    ``tools/centerpoint_offline/prediction_bridge.py`` (schema
    ``heven.offline_detection.v1``). No inference is run."""

    raw = Path(jsonl_path).read_bytes()
    frames: dict[str, DetectorFrame] = {}
    invalid_records = 0
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if payload.get("schema") != HEVEN_OFFLINE_DETECTION_SCHEMA:
            raise AdapterError(
                f"{jsonl_path}:{line_number}: schema {payload.get('schema')!r} != "
                f"{HEVEN_OFFLINE_DETECTION_SCHEMA!r}"
            )
        if str(payload.get("frame_id", DETECTOR_FRAME_ID)) != DETECTOR_FRAME_ID:
            raise AdapterError(
                f"{jsonl_path}:{line_number}: offline predictions must be "
                f"{DETECTOR_FRAME_ID!r}"
            )
        sample_id = str(payload["sample_id"])
        if sample_id in frames:
            raise AdapterError(f"duplicate detector record for sample {sample_id!r}")
        stamp = int(payload["source_header_stamp_ns"])
        detections: list[NormDetection] = []
        for det_index, det in enumerate(payload.get("detections", [])):
            box = [float(v) for v in det["box_lidar"][:7]]
            score = float(det["score"])
            if len(box) != 7 or not _finite(*box, score):
                invalid_records += 1
                continue
            if min(box[3:6]) <= 0.0:
                invalid_records += 1
                continue
            detections.append(
                NormDetection(
                    index=det_index,
                    class_name=str(det["class_name"]),
                    center_lidar=(box[0], box[1], box[2]),
                    dims_lwh=(box[3], box[4], box[5]),
                    yaw=_wrap_pi(box[6]),
                    yaw_available=True,
                    score=score,
                )
            )
        frames[sample_id] = DetectorFrame(
            sample_id=sample_id,
            lidar_stamp_ns=stamp,
            detector_frame_present=True,
            detections=tuple(detections),
        )

    completeness = _resolve_completeness(len(frames), detector_manifest, invalid_records)
    return DetectorRecordSet(
        detector_source="centerpoint",
        detector_config_id=str(detector_config_id),
        class_semantics="class_aware",
        detector_manifest_sha256=_sha256_bytes(raw),
        canonical_dataset_id=canonical_dataset_id,
        provenance={
            "input_schema": HEVEN_OFFLINE_DETECTION_SCHEMA,
            "input_path": str(jsonl_path),
            "detector_manifest": detector_manifest or {},
            "invalid_detection_records": invalid_records,
        },
        completeness=completeness,
        frames=frames,
    )


def euclidean_records_from_ros_comparison(
    jsonl_path: Path,
    canonical_root: Path,
    *,
    run_id: str,
    detector_config_id: str,
    canonical_dataset_id: str | None = None,
    detector_manifest: dict[str, Any] | None = None,
) -> DetectorRecordSet:
    """Euclidean detector output recorded by
    ``ad_lidar_perception/record_detected_objects.py`` (schema
    ``heven.ros_detection_comparison.v1``). That record carries a LiDAR header
    stamp and frame id but no ``sample_id`` and no yaw, so alignment is by
    exact LiDAR anchor stamp scoped to ``run_id``; ambiguity is a hard error."""

    stamp_to_sample = _canonical_stamp_index(canonical_root, run_id)
    raw = Path(jsonl_path).read_bytes()
    frames: dict[str, DetectorFrame] = {}
    unaligned_records = 0
    invalid_records = 0
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if payload.get("schema") != HEVEN_ROS_DETECTION_COMPARISON_SCHEMA:
            raise AdapterError(
                f"{jsonl_path}:{line_number}: schema {payload.get('schema')!r} != "
                f"{HEVEN_ROS_DETECTION_COMPARISON_SCHEMA!r}"
            )
        if str(payload.get("frame_id")) != DETECTOR_FRAME_ID:
            raise AdapterError(
                f"{jsonl_path}:{line_number}: Euclidean record frame_id "
                f"{payload.get('frame_id')!r} != {DETECTOR_FRAME_ID!r} - a "
                "map-frame attachment needs a lidar_link detector box"
            )
        stamp = int(payload["header_stamp_ns"])
        sample_id = stamp_to_sample.get(stamp)
        if sample_id is None:
            unaligned_records += 1
            continue
        if sample_id in frames:
            raise AdapterError(f"duplicate detector record for sample {sample_id!r}")
        detections: list[NormDetection] = []
        for det_index, obj in enumerate(payload.get("objects", [])):
            pos = [float(v) for v in obj["position"]]
            dims = [float(v) for v in obj["dimensions"]]
            score = float(obj.get("score", 1.0))
            if len(pos) != 3 or len(dims) != 3 or not _finite(*pos, *dims, score):
                invalid_records += 1
                continue
            if min(dims) <= 0.0:
                invalid_records += 1
                continue
            detections.append(
                NormDetection(
                    index=det_index,
                    class_name=str(obj.get("class_name", "unknown")),
                    center_lidar=(pos[0], pos[1], pos[2]),
                    dims_lwh=(dims[0], dims[1], dims[2]),
                    yaw=0.0,
                    yaw_available=False,
                    score=score,
                )
            )
        frames[sample_id] = DetectorFrame(
            sample_id=sample_id,
            lidar_stamp_ns=stamp,
            detector_frame_present=True,
            detections=tuple(detections),
        )

    completeness = _resolve_completeness(len(frames), detector_manifest, invalid_records)
    completeness["unaligned_records"] = unaligned_records
    return DetectorRecordSet(
        detector_source="euclidean",
        detector_config_id=str(detector_config_id),
        class_semantics="class_agnostic",
        detector_manifest_sha256=_sha256_bytes(raw),
        canonical_dataset_id=canonical_dataset_id,
        provenance={
            "input_schema": HEVEN_ROS_DETECTION_COMPARISON_SCHEMA,
            "input_path": str(jsonl_path),
            "run_id_scope": run_id,
            "alignment_key": "exact_lidar_header_stamp_ns",
            "yaw_available": False,
            "detector_manifest": detector_manifest or {},
            "invalid_detection_records": invalid_records,
            "unaligned_records": unaligned_records,
        },
        completeness=completeness,
        frames=frames,
    )


def _resolve_completeness(
    processed_frames: int,
    detector_manifest: dict[str, Any] | None,
    invalid_records: int,
) -> dict[str, Any]:
    manifest = detector_manifest or {}
    expected = int(manifest.get("expected_frames", processed_frames))
    failed = int(manifest.get("failed_frames", 0))
    complete = expected == processed_frames + failed and failed == 0
    return {
        "expected_frames": expected,
        "processed_frames": processed_frames,
        "failed_frames": failed,
        "invalid_detection_records": invalid_records,
        "complete": bool(complete),
    }


def _canonical_stamp_index(canonical_root: Path, run_id: str) -> dict[int, str]:
    """``lidar_header_stamp_ns -> sample_id`` for one run; raises on collision."""

    paths = DatasetPaths(Path(canonical_root))
    run_dirs = [
        d
        for scenario_dir in sorted(p for p in paths.runs_dir.iterdir() if p.is_dir())
        for d in sorted(p for p in scenario_dir.iterdir() if p.is_dir())
        if d.name == run_id
    ]
    if not run_dirs:
        raise AdapterError(f"run {run_id!r} not found under {paths.runs_dir}")
    if len(run_dirs) > 1:
        raise AdapterError(f"run id {run_id!r} is not unique in the canonical dataset")
    index: dict[int, str] = {}
    for line in (run_dirs[0] / "frames.jsonl").read_text("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        stamp = int(row["lidar_header_stamp_ns"])
        if stamp in index:
            raise AdapterError(
                f"run {run_id!r} has two frames at stamp {stamp} - cannot align "
                "a stamp-only detector record unambiguously"
            )
        index[stamp] = str(row["sample_id"])
    return index


# --------------------------------------------------------------------------
# canonical GT + transform per frame
# --------------------------------------------------------------------------
def _rigid_from_pair(pair: dict[str, Any]) -> RigidTransform:
    t = pair["translation"]
    q = pair["quaternion_xyzw"]
    return RigidTransform(
        (float(t[0]), float(t[1]), float(t[2])),
        (float(q[0]), float(q[1]), float(q[2]), float(q[3])),
    )


def _lidar_to_map_from_tf(tf_payload: dict[str, Any]) -> RigidTransform | None:
    """Recompose ``lidar_link -> map`` from a canonical ``tf/<frame>.json``.

    Uses exactly the pieces the factory stored (``map_to_odom``,
    ``odom_to_base``, ``static_edges``) and the same composition as
    ``transforms.build_map_to_lidar``:
    ``map_to_lidar = base_to_lidar^-1 . odom_to_base^-1 . map_to_odom``.
    """

    if tf_payload.get("status") != "matched":
        return None
    if "map_to_odom" not in tf_payload or "odom_to_base" not in tf_payload:
        return None
    static_edges = tf_payload.get("static_edges") or {}
    try:
        base_to_rear = _rigid_from_pair(static_edges["base_link->rear_axle_link"])
        rear_to_lidar = _rigid_from_pair(static_edges["rear_axle_link->lidar_link"])
    except (KeyError, TypeError):
        return None
    base_to_lidar = base_to_rear.compose(rear_to_lidar)
    map_to_odom = _rigid_from_pair(tf_payload["map_to_odom"])
    odom_to_base = _rigid_from_pair(tf_payload["odom_to_base"])
    map_to_lidar = (
        base_to_lidar.inverse().compose(odom_to_base.inverse()).compose(map_to_odom)
    )
    return map_to_lidar.inverse()


@dataclass
class CanonicalFrame:
    sample_id: str
    frame_index: int
    lidar_to_map: RigidTransform | None
    gt_actors: list[GtActor]


def _read_run_frames(run_dir: Path) -> dict[str, CanonicalFrame]:
    """Every frame of one canonical run: tracking-valid GT actors (same
    predicate as the trajectory adapter) + the lidar->map transform."""

    frames: dict[str, CanonicalFrame] = {}
    for line in (run_dir / "frames.jsonl").read_text("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        frame_index = int(row["frame_index"])
        sample_id = str(row["sample_id"])
        frame_tracking_valid = bool(row.get("valid_for_tracking_gt"))
        actor_status_matched = (row.get("actor_gt") or {}).get("status") == "matched"

        tf_payload = json.loads(
            (run_dir / "tf" / f"{frame_index:06d}.json").read_text("utf-8")
        )
        lidar_to_map = _lidar_to_map_from_tf(tf_payload)

        gt_actors: list[GtActor] = []
        if frame_tracking_valid and actor_status_matched:
            gt = json.loads(
                (run_dir / "gt" / f"{frame_index:06d}.json").read_text("utf-8")
            )
            for box in gt.get("boxes", []):
                if not bool(box.get("valid_for_tracking_gt")):
                    continue
                map_frame = box.get("map_frame")
                if not isinstance(map_frame, dict) or "center" not in map_frame:
                    continue
                center = [float(v) for v in map_frame["center"]]
                if len(center) != 3 or not _finite(*center):
                    continue
                yaw = _wrap_pi(float(map_frame.get("yaw", 0.0)))
                box_map = (
                    center[0],
                    center[1],
                    center[2],
                    float(map_frame.get("length", 0.0)),
                    float(map_frame.get("width", 0.0)),
                    float(map_frame.get("height", 0.0)),
                    yaw,
                )
                gt_actors.append(
                    GtActor(
                        actor_id=int(box["actor_id"]),
                        class_name=str(box.get("class_name")),
                        x=center[0],
                        y=center[1],
                        box_map=box_map,
                    )
                )
        frames[sample_id] = CanonicalFrame(
            sample_id=sample_id,
            frame_index=frame_index,
            lidar_to_map=lidar_to_map,
            gt_actors=gt_actors,
        )
    return frames


# --------------------------------------------------------------------------
# exporter
# --------------------------------------------------------------------------
def _atomic_write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """Deterministic ``.npz`` (identical scheme as the trajectory adapter:
    ZIP_STORED, fixed 1980-01-01 member stamps, sorted members)."""

    import io
    import zipfile

    import numpy.lib.format as npy_format

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            member = io.BytesIO()
            npy_format.write_array(member, np.asarray(arrays[name]), allow_pickle=False)
            archive.writestr(zipfile.ZipInfo(f"{name}.npy"), member.getvalue())
    return buffer.getvalue()


def _longest_true_run(mask: np.ndarray) -> int:
    best = cur = 0
    for value in mask:
        cur = cur + 1 if value else 0
        best = max(best, cur)
    return best


@dataclass
class ExportResult:
    manifest: dict[str, Any]
    content_fingerprint: str


class KalmanNetMeasurementExporter:
    MARKER = ".kalmannet_morai_measurement_dataset"

    def __init__(
        self,
        trajectory_root: Path,
        canonical_root: Path,
        output_root: Path,
        detector_records: DetectorRecordSet,
        config: AssociationConfig,
        adapter_commit: str,
        overwrite: bool = False,
        allow_incomplete_detector: bool = False,
    ) -> None:
        self.trajectory_root = Path(trajectory_root).resolve()
        self.canonical_root = Path(canonical_root).resolve()
        self.output_root = Path(output_root).resolve()
        self.detector = detector_records
        self.config = config
        self.adapter_commit = adapter_commit
        self.overwrite = overwrite
        self.allow_incomplete_detector = allow_incomplete_detector
        self._effective_class_matching = (
            config.class_matching and detector_records.class_matching_supported
        )

    # -- gates -------------------------------------------------------
    def _trajectory_manifest(self) -> dict[str, Any]:
        path = self.trajectory_root / "export_manifest.json"
        if not path.is_file():
            raise AdapterError(f"{path} missing (not a trajectory-v1 export)")
        manifest = json.loads(path.read_text("utf-8"))
        if manifest.get("adapter_schema_version") != TRAJECTORY_SCHEMA_VERSION:
            raise AdapterError(
                f"trajectory export schema {manifest.get('adapter_schema_version')!r} "
                f"!= {TRAJECTORY_SCHEMA_VERSION!r}"
            )
        if not (self.trajectory_root / ".kalmannet_morai_trajectory_adapter").is_file():
            raise AdapterError("trajectory export marker absent")
        return manifest

    def _canonical_manifest(self) -> dict[str, Any]:
        path = self.canonical_root / "dataset_manifest.json"
        if not path.is_file():
            raise AdapterError(f"{path} missing (not a canonical dataset)")
        manifest = json.loads(path.read_text("utf-8"))
        if manifest.get("schema_version") != SOURCE_DATASET_SCHEMA:
            raise AdapterError(
                f"canonical schema {manifest.get('schema_version')!r} != "
                f"{SOURCE_DATASET_SCHEMA!r}"
            )
        return manifest

    def validate_inputs(self) -> dict[str, Any]:
        traj_manifest = self._trajectory_manifest()
        canon_manifest = self._canonical_manifest()

        canonical_sha = _sha256_bytes(
            (self.canonical_root / "dataset_manifest.json").read_bytes()
        )
        traj_source_sha = traj_manifest.get("source_dataset_manifest_sha256")
        if traj_source_sha != canonical_sha:
            raise AdapterError(
                "cross-dataset pairing refused: trajectory export was derived "
                f"from a canonical dataset with manifest sha {traj_source_sha}, "
                f"not {canonical_sha}"
            )
        canon_id = canon_manifest.get("dataset_id")
        if traj_manifest.get("source_dataset_id") not in (None, canon_id):
            raise AdapterError(
                f"trajectory source_dataset_id {traj_manifest.get('source_dataset_id')!r}"
                f" != canonical dataset_id {canon_id!r}"
            )
        if self.detector.canonical_dataset_id not in (None, canon_id):
            raise AdapterError(
                f"detector records reference canonical dataset "
                f"{self.detector.canonical_dataset_id!r} != {canon_id!r}"
            )
        if not self.detector.completeness.get("complete") and not self.allow_incomplete_detector:
            raise AdapterError(
                f"detector run is incomplete ({self.detector.completeness}); pass "
                "--allow-incomplete-detector to override"
            )
        return {
            "trajectory_manifest": traj_manifest,
            "canonical_manifest": canon_manifest,
            "canonical_manifest_sha256": canonical_sha,
        }

    # -- planning ---------------------------------------------------
    def _load_trajectory_index(self) -> list[dict[str, Any]]:
        rows = []
        for line in (
            self.trajectory_root / "trajectory_index.jsonl"
        ).read_text("utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        if not rows:
            raise AdapterError("trajectory export has zero trajectories")
        return rows

    def _split_lists(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for split in ("train", "val", "test"):
            path = self.trajectory_root / "splits" / f"{split}.txt"
            out[split] = [
                line.strip()
                for line in path.read_text("utf-8").splitlines()
                if line.strip()
            ]
        return out

    def _canonical_run_frames(self, run_ids: Iterable[str]) -> dict[str, dict[str, CanonicalFrame]]:
        paths = DatasetPaths(self.canonical_root)
        wanted = set(run_ids)
        found: dict[str, dict[str, CanonicalFrame]] = {}
        for scenario_dir in sorted(p for p in paths.runs_dir.iterdir() if p.is_dir()):
            for run_dir in sorted(p for p in scenario_dir.iterdir() if p.is_dir()):
                if run_dir.name in wanted:
                    found[run_dir.name] = _read_run_frames(run_dir)
        missing = wanted - set(found)
        if missing:
            raise AdapterError(f"canonical runs missing for trajectories: {sorted(missing)}")
        return found

    def plan(self) -> dict[str, Any]:
        gate = self.validate_inputs()
        index_rows = self._load_trajectory_index()
        run_ids = sorted({row["run_id"] for row in index_rows})
        run_frames = self._canonical_run_frames(run_ids)

        # frame-local supervised association, once per (run, sample_id) that a
        # trajectory touches.
        needed: dict[tuple[str, str], set[int]] = {}
        for row in index_rows:
            traj = load_trajectory(self.trajectory_root / row["npz_path"])
            for sid in traj.source_sample_ids.tolist():
                needed.setdefault((row["run_id"], str(sid)), set())

        associations: dict[tuple[str, str], Any] = {}
        detector_frame_missing = 0
        no_map_transform = 0
        zero_detection_frames = 0
        false_positive_detections = 0
        unmatched_gt_actor_frames = 0
        match_distances: list[float] = []
        match_ious: list[float] = []
        for (run_id, sample_id) in sorted(needed):
            canon = run_frames[run_id].get(sample_id)
            if canon is None:
                raise AdapterError(
                    f"trajectory references canonical frame {run_id}/{sample_id} "
                    "that is not in the canonical dataset"
                )
            det_frame = self.detector.frames.get(sample_id)
            if det_frame is None:
                detector_frame_missing += 1
                associations[(run_id, sample_id)] = ("detector_frame_missing", None)
                continue
            if canon.lidar_to_map is None:
                no_map_transform += 1
                associations[(run_id, sample_id)] = ("no_map_transform", None)
                continue
            detections = _detections_to_map(det_frame, canon.lidar_to_map)
            if not detections:
                zero_detection_frames += 1
            assoc = associate_frame(
                canon.gt_actors,
                detections,
                max_distance_m=self.config.max_distance_m,
                class_matching=self._effective_class_matching,
            )
            false_positive_detections += len(assoc.unmatched_detection_indices)
            unmatched_gt_actor_frames += len(assoc.unmatched_actor_ids)
            for m in assoc.matches:
                match_distances.append(m.distance_m)
                if m.iou is not None:
                    match_ious.append(m.iou)
            associations[(run_id, sample_id)] = ("associated", assoc)

        return {
            "gate": gate,
            "index_rows": index_rows,
            "run_frames": run_frames,
            "associations": associations,
            "split_lists": self._split_lists(),
            "aggregate": {
                "detector_frame_missing": detector_frame_missing,
                "no_map_transform_frames": no_map_transform,
                "zero_detection_frames": zero_detection_frames,
                "false_positive_detections": false_positive_detections,
                "unmatched_gt_actor_frames": unmatched_gt_actor_frames,
                "match_distance_m": _stat(match_distances),
                "match_iou": _stat(match_ious),
            },
        }

    # -- export ---------------------------------------------------
    def export(self) -> ExportResult:
        out = self.output_root
        if out.exists():
            complete = (out / "export_manifest.json").is_file()
            if not self.overwrite:
                raise AdapterError(
                    f"{'completed export' if complete else 'directory'} exists: {out}; "
                    "pass --overwrite"
                )
            if (out / self.MARKER).is_file():
                shutil.rmtree(out)
            else:
                raise AdapterError(f"refusing to overwrite unrecognised directory: {out}")

        staging = out.with_name(out.name + ".tmp")
        if staging.exists():
            shutil.rmtree(staging)
        for sub in ("trajectories", "splits"):
            (staging / sub).mkdir(parents=True, exist_ok=True)
        (staging / self.MARKER).write_text(
            "HEVEN KalmanNet MORAI detector measurement attachment\n", encoding="utf-8"
        )

        preview = self.plan()
        index_rows = preview["index_rows"]
        associations = preview["associations"]
        split_lists = preview["split_lists"]

        traj_to_split: dict[str, str] = {}
        for split, ids in split_lists.items():
            for tid in ids:
                traj_to_split[tid] = split

        out_index_lines: list[str] = []
        artifact_hashes: list[str] = []
        totals = {
            "trajectories": 0,
            "trajectories_training_ready": 0,
            "gt_samples_total": 0,
            "measurements_valid": 0,
        }
        coverage_by_split: dict[str, list[float]] = {s: [] for s in ("train", "val", "test")}
        missing_run_lengths: list[int] = []

        for row in index_rows:
            tid = row["trajectory_id"]
            traj = load_trajectory(self.trajectory_root / row["npz_path"])
            src = dict(traj.raw)
            count = traj.length

            measurement = np.full((count, KNET_MEAS_DIM), np.nan, dtype=np.float64)
            measurement_state = np.full((count, KNET_STATE_DIM), np.nan, dtype=np.float64)
            measurement_valid = np.zeros(count, dtype=bool)
            measurement_score = np.full(count, np.nan, dtype=np.float64)
            measurement_match_distance_m = np.full(count, np.nan, dtype=np.float64)
            measurement_match_iou = np.full(count, np.nan, dtype=np.float64)
            measurement_box_lidar = np.full((count, 7), np.nan, dtype=np.float64)
            measurement_source = np.full(count, "", dtype="<U64")
            measurement_class = np.full(count, "", dtype="<U32")
            miss_reason_counts = {reason: 0 for reason in MISS_REASONS}

            for k in range(count):
                sample_id = str(traj.source_sample_ids[k])
                kind, assoc = associations[(row["run_id"], sample_id)]
                if kind == "detector_frame_missing":
                    miss_reason_counts["detector_frame_missing"] += 1
                    continue
                if kind == "no_map_transform":
                    miss_reason_counts["no_map_transform"] += 1
                    continue
                match = assoc.match_for_actor(int(row["actor_id"]))
                if match is None:
                    det_frame = self.detector.frames.get(sample_id)
                    if det_frame is not None and not det_frame.detections:
                        miss_reason_counts["zero_detections"] += 1
                    elif int(row["actor_id"]) in assoc.matched_actor_ids:
                        miss_reason_counts["assigned_elsewhere"] += 1
                    else:
                        miss_reason_counts["gated_out"] += 1
                    continue
                det = self.detector.frames[sample_id].detections[match.detection_index]
                canon = preview["run_frames"][row["run_id"]][sample_id]
                cx, cy, cz = canon.lidar_to_map.apply(det.center_lidar)
                measurement[k] = (cx, cy)
                measurement_state[k, :KNET_MEAS_DIM] = (cx, cy)
                measurement_valid[k] = True
                measurement_score[k] = det.score
                measurement_match_distance_m[k] = match.distance_m
                if match.iou is not None:
                    measurement_match_iou[k] = match.iou
                measurement_box_lidar[k] = (
                    det.center_lidar[0], det.center_lidar[1], det.center_lidar[2],
                    det.dims_lwh[0], det.dims_lwh[1], det.dims_lwh[2], det.yaw,
                )
                measurement_source[k] = self.detector.detector_source
                measurement_class[k] = det.class_name

            arrays = {name: src[name] for name in GT_ARRAY_NAMES}
            arrays.update(
                measurement=measurement,
                measurement_state=measurement_state,
                measurement_valid=measurement_valid,
                measurement_score=measurement_score,
                measurement_match_distance_m=measurement_match_distance_m,
                measurement_match_iou=measurement_match_iou,
                measurement_box_lidar=measurement_box_lidar,
                measurement_source=measurement_source,
                measurement_class=measurement_class,
            )
            npz_bytes = _npz_bytes(arrays)
            npz_path = staging / "trajectories" / f"{tid}.npz"
            tmp = npz_path.with_suffix(".npz.tmp")
            tmp.write_bytes(npz_bytes)
            tmp.replace(npz_path)

            valid_count = int(measurement_valid.sum())
            coverage = valid_count / count if count else 0.0
            longest_missing = _longest_true_run(~measurement_valid)
            missing_run_lengths.append(longest_missing)
            longest_missing_s = _missing_run_seconds(traj, measurement_valid)
            all_valid_finite = bool(
                np.all(np.isfinite(measurement[measurement_valid]))
            ) if valid_count else True
            training_ready = bool(
                row.get("valid_for_kalmannet_gt")
                and valid_count > 0
                and all_valid_finite
                and measurement.shape[1] == KNET_MEAS_DIM
            )

            out_row = dict(row)
            out_row.update(
                measurement_schema_version=MEASUREMENT_SCHEMA_VERSION,
                detector_source=self.detector.detector_source,
                detector_config_id=self.detector.detector_config_id,
                measurement_count=valid_count,
                measurement_coverage=coverage,
                longest_missing_run_frames=longest_missing,
                longest_missing_run_s=longest_missing_s,
                miss_reason_counts=miss_reason_counts,
                kalmannet_training_ready=training_ready,
            )
            out_index_lines.append(json.dumps(out_row, sort_keys=True))
            artifact_hashes.append(_sha256_bytes(npz_bytes))
            artifact_hashes.append(
                _sha256_bytes(json.dumps(out_row, sort_keys=True).encode("utf-8"))
            )

            totals["trajectories"] += 1
            totals["trajectories_training_ready"] += int(training_ready)
            totals["gt_samples_total"] += count
            totals["measurements_valid"] += valid_count
            coverage_by_split[traj_to_split[tid]].append(coverage)

        for split, ids in split_lists.items():
            (staging / "splits" / f"{split}.txt").write_text(
                "".join(f"{i}\n" for i in ids), encoding="utf-8"
            )
        (staging / "trajectory_index.jsonl").write_text(
            "".join(f"{line}\n" for line in out_index_lines), encoding="utf-8"
        )

        traj_manifest = preview["gate"]["trajectory_manifest"]
        canon_manifest = preview["gate"]["canonical_manifest"]
        assoc_config_json = self.config.canonical_json(
            effective_class_matching=self._effective_class_matching
        )

        _atomic_write_json(
            staging / "metadata.json",
            {
                "measurement_schema_version": MEASUREMENT_SCHEMA_VERSION,
                "detector_record_schema_version": DETECTOR_RECORD_SCHEMA_VERSION,
                "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
                "measurement_frame": "map",
                "measurement_fields": ["x", "y"],
                "measurement_dim": KNET_MEAS_DIM,
                "measurement_state_velocity": "nan (detector supplies no velocity; KNet z_k is position-only)",
                "gt_arrays_copied_verbatim": list(GT_ARRAY_NAMES),
                "measurement_arrays_written": list(MEASUREMENT_ARRAY_NAMES),
                "association": json.loads(assoc_config_json),
                "supervision_note": (
                    "GT is used to match detections to actors for dataset "
                    "construction. This is NOT runtime tracker association and "
                    "the runtime system cannot reproduce it."
                ),
                "assignment_method": (
                    "gt_supervised_bev_center_distance_hungarian"
                ),
            },
        )

        split_manifest = {
            "measurement_schema_version": MEASUREMENT_SCHEMA_VERSION,
            "inherited_from_trajectory_export": True,
            "split_method": "inherited_from_trajectory_v1_verbatim",
            "resplit": False,
            "trajectory_counts": {s: len(v) for s, v in split_lists.items()},
            "coverage_by_split": {
                s: _stat(coverage_by_split[s]) for s in ("train", "val", "test")
            },
            "source_trajectory_split_manifest_sha256": _sha256_bytes(
                (self.trajectory_root / "split_manifest.json").read_bytes()
            ),
        }
        _atomic_write_json(staging / "split_manifest.json", split_manifest)

        fingerprint = _sha256_bytes(
            "|".join(
                [
                    MEASUREMENT_SCHEMA_VERSION,
                    str(traj_manifest.get("content_fingerprint")),
                    self.detector.detector_manifest_sha256,
                    assoc_config_json,
                    *sorted(artifact_hashes),
                ]
            ).encode("utf-8")
        )

        aggregate = preview["aggregate"]
        manifest = {
            "status": "complete",
            "measurement_schema_version": MEASUREMENT_SCHEMA_VERSION,
            "detector_record_schema_version": DETECTOR_RECORD_SCHEMA_VERSION,
            "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
            "source_trajectory_export": {
                "content_fingerprint": traj_manifest.get("content_fingerprint"),
                "adapter_repository_commit": traj_manifest.get("adapter_repository_commit"),
            },
            "canonical_dataset": {
                "dataset_id": canon_manifest.get("dataset_id"),
                "manifest_sha256": preview["gate"]["canonical_manifest_sha256"],
            },
            "detector": {
                "source": self.detector.detector_source,
                "config_id": self.detector.detector_config_id,
                "class_semantics": self.detector.class_semantics,
                "class_matching_effective": self._effective_class_matching,
                "manifest_sha256": self.detector.detector_manifest_sha256,
                "completeness": self.detector.completeness,
                "provenance": self.detector.provenance,
            },
            "association": json.loads(assoc_config_json),
            "adapter_repository_commit": self.adapter_commit,
            "trajectory_count": totals["trajectories"],
            "split_trajectory_counts": {s: len(v) for s, v in split_lists.items()},
            "counts": {
                **totals,
                "measurement_coverage": (
                    totals["measurements_valid"] / totals["gt_samples_total"]
                    if totals["gt_samples_total"]
                    else 0.0
                ),
                "longest_natural_missing_run_frames": (
                    max(missing_run_lengths) if missing_run_lengths else 0
                ),
                **aggregate,
            },
            "measurements_present": totals["measurements_valid"] > 0,
            "content_fingerprint": fingerprint,
        }
        _atomic_write_json(staging / "export_manifest.json", manifest)

        staging.replace(out)
        return ExportResult(manifest, fingerprint)


def _detections_to_map(
    det_frame: DetectorFrame, lidar_to_map: RigidTransform
) -> list[Detection]:
    detections: list[Detection] = []
    for det in det_frame.detections:
        cx, cy, cz = lidar_to_map.apply(det.center_lidar)
        box_map = None
        if det.yaw_available:
            # rotate the box yaw into map by transforming a unit heading vector
            hx, hy, _ = lidar_to_map.apply(
                (
                    det.center_lidar[0] + math.cos(det.yaw),
                    det.center_lidar[1] + math.sin(det.yaw),
                    det.center_lidar[2],
                )
            )
            map_yaw = math.atan2(hy - cy, hx - cx)
            box_map = (
                cx, cy, cz,
                det.dims_lwh[0], det.dims_lwh[1], det.dims_lwh[2], map_yaw,
            )
        detections.append(
            Detection(
                index=det.index,
                class_name=det.class_name,
                x=cx,
                y=cy,
                score=det.score,
                box_map=box_map,
                box_lidar=tuple(det.center_lidar) + tuple(det.dims_lwh) + (det.yaw,),
                yaw_available=det.yaw_available,
            )
        )
    return detections


def _missing_run_seconds(traj, measurement_valid: np.ndarray) -> float:
    best = 0.0
    run_start = None
    stamps = traj.timestamps_ns
    for k, valid in enumerate(measurement_valid):
        if not valid:
            if run_start is None:
                run_start = k
            best = max(best, float(stamps[k] - stamps[run_start]) / 1e9)
        else:
            run_start = None
    return best


def _stat(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "p10": float(np.percentile(arr, 10)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }
