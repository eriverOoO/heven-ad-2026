#!/usr/bin/env python3
"""MORAI HEVEN dataset core and optional OpenPCDet registration adapter."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CLASS_NAMES = ("vehicle", "pedestrian", "obstacle")
BOX_FIELDS = ("x", "y", "z", "length", "width", "height", "yaw")
OPENPCDET_DATASET_NAME = "MoraiHevenDataset"


class MoraiHevenDatasetCore:
    """Load exported samples without importing PyTorch or OpenPCDet."""

    def __init__(
        self,
        root_path: Path | str,
        split: str = "train",
        class_names: Iterable[str] = CLASS_NAMES,
        expected_dataset_version: str | None = None,
    ) -> None:
        self.root_path = Path(root_path).resolve()
        self.class_names = tuple(class_names)
        if not self.class_names:
            raise ValueError("class_names must not be empty")
        metadata_path = self.root_path / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.dataset_version = self.metadata.get(
            "dataset_version", "unversioned_step03"
        )
        if (
            expected_dataset_version is not None
            and self.dataset_version != expected_dataset_version
        ):
            raise ValueError(
                f"dataset version {self.dataset_version!r}, expected "
                f"{expected_dataset_version!r}"
            )
        split_path = self.root_path / "splits" / f"{split}.txt"
        if not split_path.is_file():
            raise FileNotFoundError(split_path)
        self.split = split
        self.sample_ids = tuple(
            line.strip()
            for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError(f"duplicate sample IDs in {split_path}")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _load_label(self, sample_id: str) -> dict[str, Any]:
        path = self.root_path / "labels" / f"{sample_id}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        label = json.loads(path.read_text(encoding="utf-8"))
        if label.get("sample_id") != sample_id:
            raise ValueError(f"sample ID mismatch in {path}")
        if label["ground_truth"]["target_frame"] != "lidar_link":
            raise ValueError(f"non-lidar GT frame in {path}")
        return label

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_id = self.sample_ids[index]
        label = self._load_label(sample_id)
        point_info = label["points"]
        point_path = self.root_path / point_info["path"]
        points = np.fromfile(point_path, dtype="<f4")
        if points.size % 4:
            raise ValueError(f"invalid XYZI byte count: {point_path}")
        points = points.reshape(-1, 4)
        if len(points) != int(point_info["finite_count"]):
            raise ValueError(f"point count mismatch: {sample_id}")
        if not np.isfinite(points).all():
            raise ValueError(f"non-finite exported point: {sample_id}")

        boxes = label["ground_truth"]["boxes"]
        gt_boxes = np.empty((len(boxes), 7), dtype=np.float32)
        gt_names = np.empty(len(boxes), dtype=object)
        visibility = np.full(len(boxes), -1, dtype=np.int32)
        for box_index, box in enumerate(boxes):
            name = str(box["class_name"])
            if name not in self.class_names:
                raise ValueError(f"unknown class {name!r}: {sample_id}")
            values = [float(box[field]) for field in BOX_FIELDS]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"non-finite box: {sample_id}/{box_index}")
            if min(values[3:6]) <= 0.0:
                raise ValueError(f"non-positive box dimension: {sample_id}/{box_index}")
            gt_boxes[box_index] = values
            gt_names[box_index] = name
            if "num_lidar_points_inside_box" in box:
                visibility[box_index] = int(box["num_lidar_points_inside_box"])

        return {
            "frame_id": sample_id,
            "sample_id": sample_id,
            "points": points,
            "gt_boxes": gt_boxes,
            "gt_names": gt_names,
            "gt_num_lidar_points": visibility,
            "source_header_stamp_ns": int(label["source"]["header_stamp_ns"]),
            "coordinate_frame": "lidar_link",
            "source_label": label,
        }

    def first_nonempty_index(self) -> int:
        for index in range(len(self)):
            label = self._load_label(self.sample_ids[index])
            if label["ground_truth"]["boxes"]:
                return index
        raise ValueError(f"split {self.split!r} has no GT boxes")


def collate_openpcdet_contract(
    samples: list[dict[str, Any]],
    class_names: Iterable[str] = CLASS_NAMES,
) -> dict[str, Any]:
    """Collate raw samples into OpenPCDet's pre-voxel batch conventions."""
    if not samples:
        raise ValueError("samples must not be empty")
    names = tuple(class_names)
    class_ids = {name: index + 1 for index, name in enumerate(names)}
    point_batches = []
    max_boxes = max(len(sample["gt_boxes"]) for sample in samples)
    gt_boxes = np.zeros((len(samples), max_boxes, 8), dtype=np.float32)
    for batch_index, sample in enumerate(samples):
        points = np.asarray(sample["points"], dtype=np.float32)
        batch_column = np.full((len(points), 1), batch_index, dtype=np.float32)
        point_batches.append(np.concatenate((batch_column, points), axis=1))
        count = len(sample["gt_boxes"])
        if count:
            gt_boxes[batch_index, :count, :7] = sample["gt_boxes"]
            gt_boxes[batch_index, :count, 7] = [
                class_ids[str(name)] for name in sample["gt_names"]
            ]
    return {
        "batch_size": len(samples),
        "frame_id": np.asarray([sample["frame_id"] for sample in samples]),
        "points": np.concatenate(point_batches, axis=0),
        "gt_boxes": gt_boxes,
    }


def register_with_openpcdet() -> type:
    """Register the adapter at runtime without changing the upstream tree."""
    try:
        import pcdet.datasets as pcdet_datasets
        from pcdet.datasets.dataset import DatasetTemplate
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "OpenPCDet/PyTorch is unavailable; install the pinned environment "
            "and put the pinned OpenPCDet checkout on PYTHONPATH"
        ) from error

    class MoraiHevenDataset(DatasetTemplate):
        def __init__(
            self,
            dataset_cfg: Any,
            class_names: list[str],
            training: bool = True,
            root_path: Path | None = None,
            logger: Any = None,
        ) -> None:
            super().__init__(
                dataset_cfg=dataset_cfg,
                class_names=class_names,
                training=training,
                root_path=root_path,
                logger=logger,
            )
            split = str(dataset_cfg.DATA_SPLIT[self.mode])
            expected = dataset_cfg.get("EXPECTED_DATASET_VERSION", None)
            self.morai_core = MoraiHevenDatasetCore(
                self.root_path,
                split=split,
                class_names=class_names,
                expected_dataset_version=expected,
            )

        def __len__(self) -> int:
            return len(self.morai_core)

        def __getitem__(self, index: int) -> dict[str, Any]:
            raw = self.morai_core[index]
            return self.prepare_data(
                {
                    "frame_id": raw["frame_id"],
                    "points": raw["points"].copy(),
                    "gt_boxes": raw["gt_boxes"].copy(),
                    "gt_names": raw["gt_names"].copy(),
                }
            )

        def evaluation(self, det_annos: Any, class_names: Any, **kwargs: Any) -> Any:
            raise NotImplementedError(
                "STEP 05-A does not define or run a performance metric"
            )

    MoraiHevenDataset.__name__ = OPENPCDET_DATASET_NAME
    pcdet_datasets.__all__[OPENPCDET_DATASET_NAME] = MoraiHevenDataset
    return MoraiHevenDataset
