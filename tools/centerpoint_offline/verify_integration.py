#!/usr/bin/env python3
"""Validate the MORAI/OpenPCDet data contract and optionally smoke a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from morai_dataset import (
    BOX_FIELDS,
    CLASS_NAMES,
    MoraiHevenDatasetCore,
    collate_openpcdet_contract,
    make_openpcdet_dataset,
)
from openpcdet_runtime import import_smoke_components, load_batch_to_cuda


HERE = Path(__file__).resolve().parent
DEFAULT_DATA_CONFIG = HERE / "configs" / "morai_heven_dataset.yaml"
DEFAULT_MODEL_CONFIG = HERE / "configs" / "morai_centerpoint_smoke.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _identity_box_delta(sample: dict[str, Any]) -> float:
    expected = np.asarray(
        [
            [float(box[field]) for field in BOX_FIELDS]
            for box in sample["source_label"]["ground_truth"]["boxes"]
        ],
        dtype=np.float32,
    ).reshape(-1, 7)
    if expected.size == 0:
        return 0.0
    return float(np.max(np.abs(expected - sample["gt_boxes"])))


def _column_ranges(values: np.ndarray) -> list[list[float]]:
    if len(values) == 0:
        return []
    return np.stack((values.min(axis=0), values.max(axis=0)), axis=1).tolist()


def _environment_probe() -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        import torch

        result.update(
            {
                "torch_import": True,
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "torch_cuda_version": torch.version.cuda,
                "gpu": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),
            }
        )
    except (ImportError, OSError) as error:
        result.update({"torch_import": False, "error": f"{type(error).__name__}: {error}"})
    return result


def _attempt_model_smoke(
    dataset_root: Path,
    openpcdet_root: Path,
    data_config_path: Path,
    model_config_path: Path,
) -> dict[str, Any]:
    import torch
    from easydict import EasyDict

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")

    dataset_template, centerpoint_class = import_smoke_components(openpcdet_root)
    dataset_class = make_openpcdet_dataset(dataset_template)
    data_cfg = EasyDict(_load_yaml(data_config_path))
    model_cfg = EasyDict(_load_yaml(model_config_path))
    dataset = dataset_class(
        dataset_cfg=data_cfg,
        class_names=list(model_cfg.CLASS_NAMES),
        training=True,
        root_path=dataset_root,
        logger=None,
    )
    index = dataset.morai_core.first_nonempty_index()
    batch = dataset.collate_batch([dataset[index]])
    model = centerpoint_class(
        model_cfg=model_cfg.MODEL,
        num_class=len(model_cfg.CLASS_NAMES),
        dataset=dataset,
    ).cuda().eval()
    load_batch_to_cuda(batch)
    with torch.no_grad():
        predictions, _ = model(batch)
    return {
        "attempted": True,
        "success": True,
        "sample_id": str(batch["frame_id"][0]),
        "prediction_count": int(len(predictions[0]["pred_boxes"])),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--openpcdet-root", type=Path)
    parser.add_argument("--attempt-model-smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    core = MoraiHevenDatasetCore(args.dataset, split=args.split)
    first = core.first_nonempty_index()
    indices = [first]
    for index in range(len(core)):
        if index != first and len(indices) < args.batch_size:
            indices.append(index)
    samples = [core[index] for index in indices]
    batch = collate_openpcdet_contract(samples)
    data_cfg = _load_yaml(args.data_config)
    point_range = np.asarray(data_cfg["POINT_CLOUD_RANGE"], dtype=np.float64)
    voxel_size = np.asarray(
        data_cfg["DATA_PROCESSOR"][-1]["VOXEL_SIZE"], dtype=np.float64
    )
    grid_size = np.rint((point_range[3:] - point_range[:3]) / voxel_size).astype(int)

    result: dict[str, Any] = {
        "dataset": str(core.root_path),
        "dataset_version": core.dataset_version,
        "split": args.split,
        "split_samples": len(core),
        "class_names": list(CLASS_NAMES),
        "sample": {
            "sample_id": samples[0]["sample_id"],
            "points_shape": list(samples[0]["points"].shape),
            "point_column_min_max": _column_ranges(samples[0]["points"]),
            "gt_boxes_shape": list(samples[0]["gt_boxes"].shape),
            "gt_boxes": samples[0]["gt_boxes"].tolist(),
            "gt_names": [str(value) for value in samples[0]["gt_names"]],
            "gt_num_lidar_points": samples[0]["gt_num_lidar_points"].tolist(),
            "source_header_stamp_ns": samples[0]["source_header_stamp_ns"],
            "box_identity_max_abs_delta": _identity_box_delta(samples[0]),
        },
        "batch": {
            "batch_size": batch["batch_size"],
            "sample_ids": batch["frame_id"].tolist(),
            "points_shape": list(batch["points"].shape),
            "gt_boxes_shape": list(batch["gt_boxes"].shape),
            "point_batch_indices": sorted(
                int(value) for value in np.unique(batch["points"][:, 0])
            ),
        },
        "convention": {
            "frame": "lidar_link",
            "axes": "+x forward, +y left, +z up",
            "box_order": "[x, y, z, length, width, height, yaw]",
            "yaw": "counter-clockwise from +x about +z",
            "conversion": "identity",
            "point_cloud_range": point_range.tolist(),
            "voxel_size": voxel_size.tolist(),
            "grid_size_xyz": grid_size.tolist(),
        },
        "environment": _environment_probe(),
        "model_smoke": {"attempted": False, "success": False},
    }

    exit_code = 0
    if args.attempt_model_smoke:
        if args.openpcdet_root is None:
            parser.error("--attempt-model-smoke requires --openpcdet-root")
        try:
            result["model_smoke"] = _attempt_model_smoke(
                args.dataset,
                args.openpcdet_root,
                args.data_config,
                args.model_config,
            )
        except (ImportError, OSError, RuntimeError) as error:
            result["model_smoke"] = {
                "attempted": True,
                "success": False,
                "error": f"{type(error).__name__}: {error}",
            }
            exit_code = 2

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
