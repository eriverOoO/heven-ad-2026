"""Schema constants, records and validity predicates for the dataset factory.

The schema version namespaces every dataset. A later schema change must not
silently reinterpret data written under an earlier version, so the version
string is embedded in the dataset manifest, every run manifest and every
frame-index row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "morai_tracking_dataset_v1"

# Canonical topic contract. These are audited against the running graph at
# capture time; a mismatch aborts the run rather than silently substituting.
LIDAR_TOPIC = "/ad/sensors/lidar/points"
EGO_GT_TOPIC = "/ad/dev/vehicle/ego_status"
ACTOR_GT_TOPIC = "/ad/dev/objects"
ODOMETRY_TOPIC = "/ad/localization/odometry"
TF_TOPIC = "/tf"
TF_STATIC_TOPIC = "/tf_static"
DETECTED_TOPIC = "/ad/perception/objects/detected"
TRACKED_TOPIC = "/ad/perception/objects/tracked"

LIDAR_TYPE = "sensor_msgs/msg/PointCloud2"
EGO_GT_TYPE = "ad_morai_interfaces_dev/msg/EgoVehicleStatus"
ACTOR_GT_TYPE = "ad_morai_interfaces_dev/msg/ObjectStatusArray"

MAP_FRAME = "map"
LIDAR_FRAME = "lidar_link"

# Class mapping. The raw ``ObjectStatus.object_type`` is NOT the gRPC
# ObjectType enum; the 0/1/2 -> pedestrian/vehicle/obstacle mapping was only
# ever proven against one recorded scenario (see the MORAI dataset exporter
# ``class_evidence``). Every run records which scenario file justified the
# mapping; an unmapped raw type becomes CLASS_UNKNOWN and is never dropped.
CLASS_MAP_VERSION = "checkpoint14_evidence_v1"
CLASS_UNKNOWN = "CLASS_UNKNOWN"
RAW_TYPE_TO_CLASS = {0: "pedestrian", 1: "vehicle", 2: "obstacle"}

# Default synchronisation tolerances (nanoseconds). MORAI publishes LiDAR at
# ~10 Hz and the dev ego / actor GT at a similar rate; 30 ms mirrors the
# offline exporter's audited ceiling.
DEFAULT_MAX_EGO_SKEW_NS = 30_000_000
DEFAULT_MAX_ACTOR_GT_SKEW_NS = 30_000_000
DEFAULT_MAX_TF_SKEW_NS = 30_000_000

RUN_STATUS_COMPLETE = "complete"
RUN_STATUS_ABORTED = "aborted"
RUN_STATUS_ERROR = "error"


@dataclass(frozen=True)
class DatasetPaths:
    """Filesystem layout for one dataset root.

    ``<root>/dataset_manifest.json``
    ``<root>/schema.json``
    ``<root>/runs/<scenario_id>/<run_id>/run_manifest.json``
    ``<root>/runs/<scenario_id>/<run_id>/frames.jsonl``
    ``<root>/runs/<scenario_id>/<run_id>/lidar/000000.npz``
    ``<root>/runs/<scenario_id>/<run_id>/gt/000000.json``
    ``<root>/runs/<scenario_id>/<run_id>/ego/000000.json``
    ``<root>/runs/<scenario_id>/<run_id>/tf/000000.json``
    """

    root: Path

    MARKER = ".morai_tracking_dataset"

    @property
    def manifest(self) -> Path:
        return self.root / "dataset_manifest.json"

    @property
    def schema(self) -> Path:
        return self.root / "schema.json"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    def run_dir(self, scenario_id: str, run_id: str) -> Path:
        return self.runs_dir / scenario_id / run_id

    def frame_paths(self, run_dir: Path, index: int) -> dict[str, Path]:
        stem = f"{index:06d}"
        return {
            "lidar": run_dir / "lidar" / f"{stem}.npz",
            "gt": run_dir / "gt" / f"{stem}.json",
            "ego": run_dir / "ego" / f"{stem}.json",
            "tf": run_dir / "tf" / f"{stem}.json",
        }


@dataclass(frozen=True)
class SkewConfig:
    max_ego_skew_ns: int = DEFAULT_MAX_EGO_SKEW_NS
    max_actor_gt_skew_ns: int = DEFAULT_MAX_ACTOR_GT_SKEW_NS
    max_tf_skew_ns: int = DEFAULT_MAX_TF_SKEW_NS

    def as_dict(self) -> dict[str, int]:
        return {
            "max_ego_skew_ns": int(self.max_ego_skew_ns),
            "max_actor_gt_skew_ns": int(self.max_actor_gt_skew_ns),
            "max_tf_skew_ns": int(self.max_tf_skew_ns),
        }


@dataclass
class FrameRecord:
    """One dataset sample, anchored to an accepted raw LiDAR frame."""

    frame_index: int
    sample_id: str
    lidar_header_stamp_ns: int
    lidar_frame_id: str
    lidar_filename: str
    point_count: int
    point_fields: list[str]

    ego_status: str = "missing"
    ego_source_stamp_ns: int | None = None
    ego_skew_ns: int | None = None

    actor_status: str = "missing"
    actor_source_stamp_ns: int | None = None
    actor_skew_ns: int | None = None
    actor_count: int = 0
    actor_ids: list[int] = field(default_factory=list)

    tf_status: str = "missing"
    tf_source_stamp_ns: int | None = None
    tf_skew_ns: int | None = None

    lidar_frame_boxes: bool = False
    validation_flags: list[str] = field(default_factory=list)

    valid_for_tracking_gt: bool = False
    valid_for_detection_gt: bool = False

    def to_row(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "frame_index": self.frame_index,
            "sample_id": self.sample_id,
            "lidar_header_stamp_ns": self.lidar_header_stamp_ns,
            "lidar_frame_id": self.lidar_frame_id,
            "lidar_filename": self.lidar_filename,
            "point_count": self.point_count,
            "point_fields": list(self.point_fields),
            "ego": {
                "status": self.ego_status,
                "source_stamp_ns": self.ego_source_stamp_ns,
                "skew_ns": self.ego_skew_ns,
            },
            "actor_gt": {
                "status": self.actor_status,
                "source_stamp_ns": self.actor_source_stamp_ns,
                "skew_ns": self.actor_skew_ns,
                "actor_count": self.actor_count,
                "actor_ids": list(self.actor_ids),
            },
            "tf": {
                "status": self.tf_status,
                "source_stamp_ns": self.tf_source_stamp_ns,
                "skew_ns": self.tf_skew_ns,
            },
            "lidar_frame_boxes": self.lidar_frame_boxes,
            "validation_flags": list(self.validation_flags),
            "valid_for_tracking_gt": self.valid_for_tracking_gt,
            "valid_for_detection_gt": self.valid_for_detection_gt,
        }


def frame_validity(
    record: FrameRecord, boxes: list[dict[str, Any]]
) -> tuple[bool, bool, list[str]]:
    """Derive ``valid_for_tracking_gt`` / ``valid_for_detection_gt``.

    Tracking-ready needs a fresh, in-skew actor GT snapshot with at least one
    actor carrying id / pose / orientation / velocity. Detection-ready
    additionally needs a valid LiDAR<->GT transform and every retained box
    with a grounded, finite, positive-dimension, mapped-class geometry.
    """

    flags: list[str] = []
    tracking = True
    detection = True

    if record.actor_status != "matched":
        flags.append(f"actor_gt_{record.actor_status}")
        return False, False, flags
    if record.tf_status != "matched":
        flags.append(f"tf_{record.tf_status}")
        detection = False
    if record.ego_status != "matched":
        flags.append(f"ego_{record.ego_status}")
        detection = False

    trackable = [b for b in boxes if b.get("valid_for_tracking_gt")]
    if not trackable:
        flags.append("no_trackable_actor")
        tracking = False

    if not boxes:
        flags.append("no_boxes")
        detection = False
    if not record.lidar_frame_boxes:
        flags.append("no_lidar_frame_boxes")
        detection = False
    if any(b.get("class_name") == CLASS_UNKNOWN for b in boxes):
        flags.append("unmapped_class_present")
        detection = False
    if any(not b.get("valid_for_detection_gt") for b in boxes):
        flags.append("box_geometry_incomplete")
        detection = False

    return tracking, detection, flags
