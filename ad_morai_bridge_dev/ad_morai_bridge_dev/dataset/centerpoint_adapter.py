"""CenterPoint MORAI Data Adapter v1.

Deterministically converts detection-valid frames of a canonical
``morai_tracking_dataset_v1`` (the MORAI Tracking Dataset Factory output)
into the exact LiDAR dataset contract consumed by this repository's
CenterPoint integration - the per-sample-JSON layout read by
``tools/centerpoint_offline/morai_dataset.py::MoraiHevenDatasetCore``:

    <derived_root>/
      metadata.json          # dataset_version + adapter provenance (loader reads dataset_version)
      export_manifest.json   # adapter schema, source hash, counts, content fingerprint
      sample_mapping.jsonl   # target sample_id <-> source scenario/run/frame/stamp
      split_manifest.json    # split method, grouping key, group + frame counts, leakage guarantee
      splits/{train,val,test}.txt
      labels/<sample_id>.json
      points/<sample_id>.bin  # Nx4 little-endian float32 [x, y, z, intensity]

There is **no** ``infos/`` pickle and **no** KITTI camera metadata: the
repo's loader reads neither. The canonical source dataset is never
modified; this output is strictly derived. No model is trained here.
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

ADAPTER_SCHEMA_VERSION = "centerpoint_morai_adapter_v1"

# The target contract, mirrored from tools/centerpoint_offline/morai_dataset.py.
# A parity test loads that module and asserts these still match.
TARGET_CLASS_NAMES = ("vehicle", "pedestrian", "obstacle")
TARGET_BOX_FIELDS = ("x", "y", "z", "length", "width", "height", "yaw")
TARGET_POINT_FEATURES = ("x", "y", "z", "intensity")
TARGET_GT_FRAME = "lidar_link"
# From configs/morai_heven_dataset.yaml POINT_CLOUD_RANGE (reported only; the
# adapter never crops - the model's DATA_PROCESSOR masks outside-range).
REPORTED_POINT_CLOUD_RANGE = (-4.0, -25.0, -3.0, 100.0, 25.0, 5.0)

# Adapter v1 is vehicle-only: the factory starter catalog only reliably
# ground-truths vehicles. Extend this map (config) once a scenario confirms
# another class. A source box whose class is not a key here is omitted from
# the frame (the frame is still exported) and counted.
DEFAULT_CLASS_MAP = {"vehicle": "vehicle"}


class AdapterError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class AdapterConfig:
    class_map: dict[str, str]
    dataset_version: str = "morai_centerpoint_v1"
    compute_points_in_box: bool = True

    def canonical_json(self) -> str:
        return json.dumps(
            {
                "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
                "class_map": dict(sorted(self.class_map.items())),
                "dataset_version": self.dataset_version,
                "compute_points_in_box": self.compute_points_in_box,
            },
            sort_keys=True,
        )

    def validate(self) -> None:
        if not self.class_map:
            raise AdapterError("class_map is empty")
        for target in self.class_map.values():
            if target not in TARGET_CLASS_NAMES:
                raise AdapterError(
                    f"class_map target {target!r} not in {TARGET_CLASS_NAMES}"
                )


def load_adapter_config(path: Path | None) -> AdapterConfig:
    if path is None:
        return AdapterConfig(class_map=dict(DEFAULT_CLASS_MAP))
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    config = AdapterConfig(
        class_map=dict(data.get("class_map", DEFAULT_CLASS_MAP)),
        dataset_version=str(data.get("dataset_version", "morai_centerpoint_v1")),
        compute_points_in_box=bool(data.get("compute_points_in_box", True)),
    )
    config.validate()
    return config


# --------------------------------------------------------------------------
# source frame model
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SourceFrame:
    scenario_id: str
    run_id: str
    requested_seed: int | None
    frame_index: int
    source_sample_id: str
    lidar_header_stamp_ns: int
    lidar_npz: Path
    gt_json: Path

    @property
    def group_key(self) -> tuple[str, str]:
        """Split-group key. Prefer (scenario_id, requested_seed); when the
        seed is absent in the run manifest, fall back to (run_id, run_id)."""

        if self.requested_seed is None:
            return (self.run_id, self.run_id)
        return (self.scenario_id, f"seed_{self.requested_seed}")


def _iter_run_dirs(paths: DatasetPaths) -> Iterable[Path]:
    for scenario_dir in sorted(p for p in paths.runs_dir.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in scenario_dir.iterdir() if p.is_dir()):
            yield run_dir


def read_source_frames(source_root: Path) -> tuple[list[SourceFrame], dict[str, int]]:
    """All detection-valid frames, plus an exclusion-reason tally."""

    paths = DatasetPaths(Path(source_root))
    frames: list[SourceFrame] = []
    excluded: dict[str, int] = {}
    for run_dir in _iter_run_dirs(paths):
        manifest = json.loads((run_dir / "run_manifest.json").read_text("utf-8"))
        seed_block = manifest.get("config", {}).get("seed", {})
        requested_seed = seed_block.get("requested_seed")
        scenario_id = seed_block.get("scenario_id") or manifest["scenario_id"]
        run_id = manifest["run_id"]
        for line in (run_dir / "frames.jsonl").read_text("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("valid_for_detection_gt"):
                flags = sorted(row.get("validation_flags") or [])
                reason = "+".join(flags) if flags else "not_detection_valid"
                excluded[reason] = excluded.get(reason, 0) + 1
                continue
            frames.append(
                SourceFrame(
                    scenario_id=scenario_id,
                    run_id=run_id,
                    requested_seed=(
                        int(requested_seed) if requested_seed is not None else None
                    ),
                    frame_index=int(row["frame_index"]),
                    source_sample_id=str(row["sample_id"]),
                    lidar_header_stamp_ns=int(row["lidar_header_stamp_ns"]),
                    lidar_npz=run_dir / row["lidar_filename"],
                    gt_json=run_dir / "gt" / f"{int(row['frame_index']):06d}.json",
                )
            )
    frames.sort(key=lambda f: (f.scenario_id, f.run_id, f.frame_index))
    return frames, excluded


# --------------------------------------------------------------------------
# per-frame conversion
# --------------------------------------------------------------------------
def _points_xyzi(npz_path: Path) -> tuple[np.ndarray, int, int]:
    from ad_morai_bridge_dev.dataset.pointcloud import read_frame_npz

    arrays = read_frame_npz(npz_path)
    missing = [f for f in TARGET_POINT_FEATURES if f not in arrays]
    if missing:
        raise AdapterError(f"{npz_path}: npz lacks {missing}")
    source_count = int(arrays["x"].shape[0])
    stacked = np.stack(
        [arrays[f].astype("<f4", copy=False) for f in TARGET_POINT_FEATURES], axis=1
    ).astype("<f4")
    finite = np.isfinite(stacked).all(axis=1)
    kept = np.ascontiguousarray(stacked[finite])
    return kept, source_count, int((~finite).sum())


def _points_in_oriented_box(points: np.ndarray, box: dict[str, float]) -> int:
    """Verbatim port of the offline exporter's oriented-box point count."""

    if points.size == 0:
        return 0
    dx = points[:, 0] - box["x"]
    dy = points[:, 1] - box["y"]
    cosine = math.cos(box["yaw"])
    sine = math.sin(box["yaw"])
    local_x = cosine * dx + sine * dy
    local_y = -sine * dx + cosine * dy
    local_z = points[:, 2] - box["z"]
    inside = (
        (np.abs(local_x) <= box["length"] / 2.0)
        & (np.abs(local_y) <= box["width"] / 2.0)
        & (np.abs(local_z) <= box["height"] / 2.0)
    )
    return int(inside.sum())


@dataclass
class ConvertedFrame:
    sample_id: str
    points: np.ndarray
    boxes: list[dict[str, Any]]
    source_point_count: int
    exported_finite_count: int
    nonfinite_dropped: int
    omitted_unmapped: dict[str, int]


def convert_frame(frame: SourceFrame, config: AdapterConfig) -> ConvertedFrame:
    points, source_count, dropped = _points_xyzi(frame.lidar_npz)
    gt = json.loads(frame.gt_json.read_text("utf-8"))
    boxes: list[dict[str, Any]] = []
    omitted: dict[str, int] = {}
    for src in gt.get("boxes", []):
        source_class = str(src["class_name"])
        target_class = config.class_map.get(source_class)
        if target_class is None:
            omitted[source_class] = omitted.get(source_class, 0) + 1
            continue
        lf = src.get("lidar_frame")
        if lf is None:
            raise AdapterError(
                f"{frame.gt_json}: detection-valid box {src.get('actor_id')} "
                "has no lidar_frame geometry"
            )
        cx, cy, cz = (float(v) for v in lf["center"])
        box: dict[str, Any] = {
            "class_name": target_class,
            "x": cx,
            "y": cy,
            "z": cz,
            "length": float(lf["length"]),
            "width": float(lf["width"]),
            "height": float(lf["height"]),
            "yaw": _wrap_pi(float(lf["yaw"])),
            "source_class_name": source_class,
            "source_actor_id": int(src.get("actor_id", -1)),
            "source_raw_object_type": int(src.get("raw_object_type", -1)),
        }
        values = [box[f] for f in TARGET_BOX_FIELDS]
        if not all(math.isfinite(v) for v in values):
            raise AdapterError(f"{frame.gt_json}: non-finite converted box")
        if min(box["length"], box["width"], box["height"]) <= 0.0:
            raise AdapterError(f"{frame.gt_json}: non-positive converted dims")
        if config.compute_points_in_box:
            box["num_lidar_points_inside_box"] = _points_in_oriented_box(points, box)
        boxes.append(box)
    return ConvertedFrame(
        sample_id=frame.source_sample_id,
        points=points,
        boxes=boxes,
        source_point_count=source_count,
        exported_finite_count=int(points.shape[0]),
        nonfinite_dropped=dropped,
        omitted_unmapped=omitted,
    )


def _wrap_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def in_reported_range(box: dict[str, float]) -> bool:
    lo = REPORTED_POINT_CLOUD_RANGE[:3]
    hi = REPORTED_POINT_CLOUD_RANGE[3:]
    return all(lo[i] <= box[k] <= hi[i] for i, k in enumerate(("x", "y", "z")))


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------
def _stable_bucket(group_key: tuple[str, str]) -> float:
    digest = hashlib.sha256(
        f"{ADAPTER_SCHEMA_VERSION}|{group_key[0]}|{group_key[1]}".encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16) / float(1 << 64)


@dataclass
class SplitResult:
    method: str
    grouping_key: str
    assignments: dict[str, str]  # group_id -> split
    group_members: dict[str, list[str]]  # group_id -> [group member label]
    warnings: list[str] = field(default_factory=list)


def _group_id(key: tuple[str, str]) -> str:
    return f"{key[0]}::{key[1]}"


def auto_split(
    groups: list[tuple[str, str]],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> SplitResult:
    unique = sorted(set(groups), key=_group_id)
    result = SplitResult(
        method="auto_stable_hash_sha256",
        grouping_key="(scenario_id, requested_seed) | run_id fallback",
        assignments={},
        group_members={_group_id(k): [_group_id(k)] for k in unique},
    )
    if len(unique) < 3:
        for key in unique:
            result.assignments[_group_id(key)] = "train"
        result.warnings.append(
            f"only {len(unique)} independent group(s); assigning all to train "
            "(too few groups for a leakage-safe val/test split)"
        )
        return result
    train_hi = ratios[0]
    val_hi = ratios[0] + ratios[1]
    for key in unique:
        bucket = _stable_bucket(key)
        split = "train" if bucket < train_hi else "val" if bucket < val_hi else "test"
        result.assignments[_group_id(key)] = split
    present = set(result.assignments.values())
    for required in ("train", "val", "test"):
        if required not in present:
            result.warnings.append(f"auto-split produced no {required} group")
    return result


def plan_split(
    groups: list[tuple[str, str]], plan: dict[str, Any]
) -> SplitResult:
    """Explicit plan: splits: {train: [{scenario, seeds:[...]}], ...}."""

    unique = sorted(set(groups), key=_group_id)
    result = SplitResult(
        method="explicit_plan",
        grouping_key="(scenario_id, requested_seed) | run_id fallback",
        assignments={},
        group_members={_group_id(k): [_group_id(k)] for k in unique},
    )
    wanted: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for entry in plan.get("splits", {}).get(split, []):
            scenario = str(entry["scenario"])
            for seed in entry.get("seeds", []):
                gid = _group_id((scenario, f"seed_{int(seed)}"))
                if gid in wanted:
                    raise AdapterError(f"group {gid} assigned to two splits")
                wanted[gid] = split
    for key in unique:
        gid = _group_id(key)
        if gid not in wanted:
            result.warnings.append(f"group {gid} not in the plan; dropped from export")
        else:
            result.assignments[gid] = wanted[gid]
    missing = sorted(set(wanted) - {_group_id(k) for k in unique})
    if missing:
        raise AdapterError(f"split plan references absent groups: {missing}")
    return result


def check_split_leakage(
    frame_to_split: dict[str, str],
    frame_to_group: dict[str, str],
    frame_to_run: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    group_splits: dict[str, set[str]] = {}
    run_splits: dict[str, set[str]] = {}
    for sample_id, split in frame_to_split.items():
        group_splits.setdefault(frame_to_group[sample_id], set()).add(split)
        run_splits.setdefault(frame_to_run[sample_id], set()).add(split)
    for group, splits in sorted(group_splits.items()):
        if len(splits) > 1:
            errors.append(f"group {group} spans splits {sorted(splits)}")
    for run, splits in sorted(run_splits.items()):
        if len(splits) > 1:
            errors.append(f"run {run} spans splits {sorted(splits)}")
    if len(frame_to_split) != len(set(frame_to_split)):
        errors.append("duplicate target sample_id")
    return errors


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------
def _atomic_write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class ExportResult:
    manifest: dict[str, Any]
    split_manifest: dict[str, Any]
    content_fingerprint: str


class CenterPointExporter:
    MARKER = ".centerpoint_morai_adapter"

    def __init__(
        self,
        source_root: Path,
        output_root: Path,
        config: AdapterConfig,
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

    # -- source gate ------------------------------------------------------
    def validate_source(self) -> None:
        source_manifest = json.loads(
            (self.source_root / "dataset_manifest.json").read_text("utf-8")
        )
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

    # -- planning -------------------------------------------------------
    def plan(self, split_plan: dict[str, Any] | None) -> dict[str, Any]:
        self.validate_source()
        frames, excluded = read_source_frames(self.source_root)
        groups = [f.group_key for f in frames]
        split = (
            plan_split(groups, split_plan)
            if split_plan
            else auto_split(groups)
        )
        by_split: dict[str, int] = {"train": 0, "val": 0, "test": 0}
        dropped = 0
        for frame in frames:
            gid = _group_id(frame.group_key)
            target = split.assignments.get(gid)
            if target is None:
                dropped += 1
            else:
                by_split[target] += 1
        # Every non-blank frames.jsonl row is either kept or counted once in
        # ``excluded``; no need to re-walk the source files to total them.
        source_frame_count = len(frames) + sum(excluded.values())
        return {
            "source_frame_count": source_frame_count,
            "eligible_frame_count": len(frames),
            "excluded_frame_count": source_frame_count - len(frames),
            "exclusion_reason_counts": dict(sorted(excluded.items())),
            "dropped_by_split_plan": dropped,
            "split_method": split.method,
            "split_group_counts": {
                s: sum(1 for v in split.assignments.values() if v == s)
                for s in ("train", "val", "test")
            },
            "split_frame_counts": by_split,
            "split_warnings": split.warnings,
            "_frames": frames,
            "_split": split,
        }

    # -- export --------------------------------------------------------
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
        for sub in ("points", "labels", "splits"):
            (staging / sub).mkdir(parents=True, exist_ok=True)
        (staging / self.MARKER).write_text(
            "HEVEN CenterPoint MORAI data adapter\n", encoding="utf-8"
        )

        preview = self.plan(split_plan)
        frames: list[SourceFrame] = preview.pop("_frames")
        split: SplitResult = preview.pop("_split")

        split_lists: dict[str, list[str]] = {"train": [], "val": [], "test": []}
        mapping_lines: list[str] = []
        frame_to_split: dict[str, str] = {}
        frame_to_group: dict[str, str] = {}
        frame_to_run: dict[str, str] = {}
        artifact_hashes: list[str] = []
        class_counts: dict[str, dict[str, int]] = {
            s: {} for s in ("train", "val", "test")
        }
        range_counts: dict[str, dict[str, int]] = {
            s: {"near": 0, "mid": 0, "far": 0} for s in ("train", "val", "test")
        }
        point_counts: dict[str, list[int]] = {s: [] for s in ("train", "val", "test")}
        totals = {
            "boxes_source": 0,
            "boxes_exported": 0,
            "boxes_omitted_unmapped": 0,
            "boxes_outside_reported_range": 0,
            "negative_frames": 0,
            "nonfinite_points_dropped": 0,
            "source_points": 0,
            "exported_points": 0,
        }
        omitted_by_class: dict[str, int] = {}

        for frame in frames:
            gid = _group_id(frame.group_key)
            target = split.assignments.get(gid)
            if target is None:
                continue
            converted = convert_frame(frame, self.config)
            sample_id = converted.sample_id
            if sample_id in frame_to_split:
                raise AdapterError(f"duplicate source sample_id {sample_id}")

            bin_path = staging / "points" / f"{sample_id}.bin"
            data = converted.points.astype("<f4").tobytes()
            tmp = bin_path.with_suffix(".bin.tmp")
            tmp.write_bytes(data)
            tmp.replace(bin_path)

            label = {
                "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
                "sample_id": sample_id,
                "source": {
                    "dataset_id": self.source_dataset_id(),
                    "scenario_id": frame.scenario_id,
                    "run_id": frame.run_id,
                    "requested_seed": frame.requested_seed,
                    "frame_index": frame.frame_index,
                    "source_sample_id": frame.source_sample_id,
                    "header_stamp_ns": frame.lidar_header_stamp_ns,
                },
                "points": {
                    "path": f"points/{sample_id}.bin",
                    "dtype": "float32_little_endian",
                    "feature_names": list(TARGET_POINT_FEATURES),
                    "finite_count": converted.exported_finite_count,
                    "source_point_count": converted.source_point_count,
                    "nonfinite_dropped": converted.nonfinite_dropped,
                    "canonical_fields_preserved_in_source": [
                        "x", "y", "z", "intensity", "time", "ring",
                    ],
                },
                "ground_truth": {
                    "target_frame": TARGET_GT_FRAME,
                    "box_fields": list(TARGET_BOX_FIELDS),
                    "header_stamp_ns": frame.lidar_header_stamp_ns,
                    "boxes": converted.boxes,
                },
            }
            _atomic_write_json(staging / "labels" / f"{sample_id}.json", label)

            split_lists[target].append(sample_id)
            frame_to_split[sample_id] = target
            frame_to_group[sample_id] = gid
            frame_to_run[sample_id] = frame.run_id
            mapping_lines.append(
                json.dumps(
                    {
                        "target_sample_id": sample_id,
                        "split": target,
                        **label["source"],
                    },
                    sort_keys=True,
                )
            )
            artifact_hashes.append(_sha256_bytes(data))
            artifact_hashes.append(
                _sha256_bytes(json.dumps(label, sort_keys=True).encode("utf-8"))
            )

            totals["boxes_source"] += len(converted.boxes) + sum(
                converted.omitted_unmapped.values()
            )
            totals["boxes_exported"] += len(converted.boxes)
            totals["boxes_omitted_unmapped"] += sum(converted.omitted_unmapped.values())
            totals["nonfinite_points_dropped"] += converted.nonfinite_dropped
            totals["source_points"] += converted.source_point_count
            totals["exported_points"] += converted.exported_finite_count
            for cls, count in converted.omitted_unmapped.items():
                omitted_by_class[cls] = omitted_by_class.get(cls, 0) + count
            if not converted.boxes:
                totals["negative_frames"] += 1
            point_counts[target].append(converted.exported_finite_count)
            for box in converted.boxes:
                class_counts[target][box["class_name"]] = (
                    class_counts[target].get(box["class_name"], 0) + 1
                )
                dist = math.hypot(box["x"], box["y"])
                bucket = "near" if dist <= 20.0 else "mid" if dist <= 45.0 else "far"
                range_counts[target][bucket] += 1
                if not in_reported_range(box):
                    totals["boxes_outside_reported_range"] += 1

        leakage = check_split_leakage(frame_to_split, frame_to_group, frame_to_run)
        if leakage:
            raise AdapterError(f"split leakage: {leakage}")

        for split_name, ids in split_lists.items():
            (staging / "splits" / f"{split_name}.txt").write_text(
                "".join(f"{i}\n" for i in ids), encoding="utf-8"
            )
        (staging / "sample_mapping.jsonl").write_text(
            "".join(f"{line}\n" for line in mapping_lines), encoding="utf-8"
        )
        _atomic_write_json(
            staging / "metadata.json",
            {
                "dataset_version": self.config.dataset_version,
                "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
                "class_names": list(TARGET_CLASS_NAMES),
                "box_fields": list(TARGET_BOX_FIELDS),
                "point_feature_names": list(TARGET_POINT_FEATURES),
                "gt_frame": TARGET_GT_FRAME,
                "reported_point_cloud_range": list(REPORTED_POINT_CLOUD_RANGE),
                "point_range_cropping_applied": False,
                "box_range_filtering_applied": False,
                "intensity_transform": "identity",
            },
        )

        split_manifest = {
            "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
            "split_method": split.method,
            "grouping_key": split.grouping_key,
            "leakage_free": True,
            "group_splits": dict(sorted(split.assignments.items())),
            "group_counts": preview["split_group_counts"],
            "frame_counts": {s: len(v) for s, v in split_lists.items()},
            "negative_frame_counts": {
                s: sum(
                    1
                    for sid in v
                    if not json.loads(
                        (staging / "labels" / f"{sid}.json").read_text("utf-8")
                    )["ground_truth"]["boxes"]
                )
                for s, v in split_lists.items()
            },
            "class_counts_by_split": {s: dict(sorted(c.items())) for s, c in class_counts.items()},
            "range_counts_by_split": range_counts,
            "warnings": split.warnings,
            "source_run_ids": sorted({f.run_id for f in frames}),
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
            "source_dataset_schema_version": SOURCE_SCHEMA_VERSION,
            "source_dataset_id": self.source_dataset_id(),
            "source_dataset_manifest_sha256": self.source_manifest_sha256(),
            "adapter_repository_commit": self.adapter_commit,
            "adapter_config": json.loads(self.config.canonical_json()),
            "class_map": dict(sorted(self.config.class_map.items())),
            "target_class_names": list(TARGET_CLASS_NAMES),
            "target_box_fields": list(TARGET_BOX_FIELDS),
            "target_point_features": list(TARGET_POINT_FEATURES),
            "sample_count": len(frame_to_split),
            "split_frame_counts": {s: len(v) for s, v in split_lists.items()},
            "counts": {
                "source_frame_count": preview["source_frame_count"],
                "eligible_frame_count": preview["eligible_frame_count"],
                "exported_frame_count": len(frame_to_split),
                "excluded_frame_count": preview["excluded_frame_count"],
                "exclusion_reason_counts": preview["exclusion_reason_counts"],
                "dropped_by_split_plan": preview["dropped_by_split_plan"],
                **totals,
                "boxes_omitted_by_source_class": dict(sorted(omitted_by_class.items())),
            },
            "content_fingerprint": fingerprint,
        }
        _atomic_write_json(staging / "export_manifest.json", manifest)

        staging.replace(self.output_root)
        return ExportResult(manifest, split_manifest, fingerprint)
