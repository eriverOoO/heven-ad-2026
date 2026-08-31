"""Atomic dataset writer: frames, run manifests and the dataset manifest.

A crash must never leave a frame-index row pointing at a partial frame, so
each artifact is written to a ``.tmp`` sibling and atomically renamed, and
the ``frames.jsonl`` row is appended only after every artifact for that
frame is on disk. An interrupted run is finalised as ``aborted`` /
``error``, never ``complete``.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ad_morai_bridge_dev.dataset.pointcloud import write_frame_npz
from ad_morai_bridge_dev.dataset.schema import (
    SCHEMA_VERSION,
    DatasetPaths,
    FrameRecord,
    RUN_STATUS_ABORTED,
    RUN_STATUS_COMPLETE,
    RUN_STATUS_ERROR,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tmp.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def repo_provenance(repository: Path) -> dict[str, Any]:
    def _git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=repository,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()

    try:
        commit = _git("rev-parse", "HEAD")
        dirty = bool(_git("status", "--porcelain"))
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, dirty = "unknown", True
    return {"repository_commit": commit, "repository_dirty": dirty}


@dataclass
class RunWriter:
    """Owns one ``runs/<scenario_id>/<run_id>/`` directory."""

    paths: DatasetPaths
    scenario_id: str
    run_id: str
    run_config: dict[str, Any]
    overwrite: bool = False

    run_dir: Path = field(init=False)
    _frame_count: int = field(init=False, default=0)
    _index_stream: Any = field(init=False, default=None)
    _actor_ids: set[int] = field(init=False, default_factory=set)
    _class_counts: dict[str, int] = field(init=False, default_factory=dict)
    _tracking_frames: int = field(init=False, default=0)
    _detection_frames: int = field(init=False, default=0)
    _skew_ns: dict[str, int] = field(
        init=False, default_factory=lambda: {"ego": 0, "actor": 0, "tf": 0}
    )
    _finalized: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.run_dir = self.paths.run_dir(self.scenario_id, self.run_id)
        manifest = self.run_dir / "run_manifest.json"
        if self.run_dir.exists() and not self.overwrite:
            raise FileExistsError(f"run already exists: {self.run_dir}")
        for sub in ("lidar", "gt", "ego", "tf"):
            sub_dir = self.run_dir / sub
            if sub_dir.is_dir() and self.overwrite:
                for stale in sub_dir.iterdir():
                    if stale.is_file():
                        stale.unlink()
            sub_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.run_dir / "frames.jsonl"
        self._index_path.write_text("", encoding="utf-8")
        self._index_stream = self._index_path.open("a", encoding="utf-8")
        _atomic_write_json(
            manifest,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "running",
                "scenario_id": self.scenario_id,
                "run_id": self.run_id,
                "started_at": _utc_now(),
                "config": self.run_config,
            },
        )

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def write_frame(
        self,
        record: FrameRecord,
        point_arrays: dict[str, np.ndarray],
        gt_payload: dict[str, Any],
        ego_payload: dict[str, Any],
        tf_payload: dict[str, Any],
    ) -> None:
        frame_paths = self.paths.frame_paths(self.run_dir, record.frame_index)
        stored = write_frame_npz(frame_paths["lidar"], point_arrays)
        record.point_count = stored["point_count"]
        _atomic_write_json(frame_paths["gt"], gt_payload)
        _atomic_write_json(frame_paths["ego"], ego_payload)
        _atomic_write_json(frame_paths["tf"], tf_payload)
        # Only now is the frame durable: commit the index row.
        self._index_stream.write(json.dumps(record.to_row(), sort_keys=True) + "\n")
        self._index_stream.flush()
        os.fsync(self._index_stream.fileno())

        self._frame_count += 1
        self._actor_ids.update(record.actor_ids)
        for box in gt_payload.get("boxes", []):
            name = str(box.get("class_name"))
            self._class_counts[name] = self._class_counts.get(name, 0) + 1
        if record.valid_for_tracking_gt:
            self._tracking_frames += 1
        if record.valid_for_detection_gt:
            self._detection_frames += 1
        for key, skew in (
            ("ego", record.ego_skew_ns),
            ("actor", record.actor_skew_ns),
            ("tf", record.tf_skew_ns),
        ):
            if skew is not None:
                self._skew_ns[key] = max(self._skew_ns[key], int(skew))

    def finalize(
        self,
        status: str,
        termination_reason: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._finalized:
            raise RuntimeError("run already finalized")
        self._finalized = True
        if self._index_stream is not None:
            self._index_stream.close()
        if status not in (RUN_STATUS_COMPLETE, RUN_STATUS_ABORTED, RUN_STATUS_ERROR):
            raise ValueError(f"invalid run status: {status}")
        summary = {
            "captured_lidar_frames": self._frame_count,
            "valid_tracking_gt_frames": self._tracking_frames,
            "valid_detection_gt_frames": self._detection_frames,
            "unique_actor_ids": sorted(self._actor_ids),
            "class_counts": dict(sorted(self._class_counts.items())),
            "max_observed_skew_ns": dict(self._skew_ns),
        }
        if extra:
            summary.update(extra)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "scenario_id": self.scenario_id,
            "run_id": self.run_id,
            "finished_at": _utc_now(),
            "termination_reason": termination_reason,
            "config": self.run_config,
            "summary": summary,
        }
        _atomic_write_json(self.run_dir / "run_manifest.json", manifest)
        return manifest


def write_dataset_documents(
    paths: DatasetPaths,
    dataset_id: str,
    provenance: dict[str, Any],
    run_manifests: list[dict[str, Any]],
) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    (paths.root / DatasetPaths.MARKER).write_text(
        "HEVEN MORAI tracking dataset factory\n", encoding="utf-8"
    )
    _atomic_write_json(
        paths.schema,
        {
            "schema_version": SCHEMA_VERSION,
            "sample_anchor": "accepted raw LiDAR PointCloud2 header.stamp",
            "point_storage": "npz, one named array per PointField, dtype preserved",
            "gt_id_source": "MORAI ObjectStatus.unique_id (native stable actor id)",
            "canonical_gt_frame": "map",
            "derived_gt_frame": "lidar_link (when the transform chain is valid)",
            "validity_flags": ["valid_for_tracking_gt", "valid_for_detection_gt"],
        },
    )
    _atomic_write_json(
        paths.manifest,
        {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": dataset_id,
            "created_at": _utc_now(),
            **provenance,
            "runs": [
                {
                    "scenario_id": m["scenario_id"],
                    "run_id": m["run_id"],
                    "status": m["status"],
                    "captured_lidar_frames": m["summary"]["captured_lidar_frames"],
                    "valid_tracking_gt_frames": m["summary"][
                        "valid_tracking_gt_frames"
                    ],
                    "valid_detection_gt_frames": m["summary"][
                        "valid_detection_gt_frames"
                    ],
                }
                for m in run_manifests
            ],
        },
    )
