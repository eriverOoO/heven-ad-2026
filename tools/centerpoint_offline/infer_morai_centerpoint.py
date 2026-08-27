#!/usr/bin/env python3
"""Run reproducible offline CenterPoint inference on MORAI HEVEN frames."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from morai_dataset import CLASS_NAMES, make_openpcdet_dataset
from openpcdet_runtime import import_smoke_components, load_batch_to_cuda
from prediction_bridge import convert_openpcdet_prediction
from train_morai_centerpoint import (
    DEFAULT_DATA_CONFIG,
    DEFAULT_MODEL_CONFIG,
    load_yaml_config,
    positive_int,
    write_json_atomic,
)


def unit_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--openpcdet-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--score-threshold", type=unit_float, default=0.1)
    parser.add_argument("--max-detections", type=positive_int, default=500)
    parser.add_argument("--max-frames", type=positive_int)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    args = parser.parse_args(argv)
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    if args.summary is None:
        args.summary = args.output.with_suffix(args.output.suffix + ".summary.json")
    return args


def filter_prediction(
    prediction: dict[str, Any], score_threshold: float, max_detections: int
) -> dict[str, Any]:
    """Filter deterministically by descending score, preserving ties by input order."""
    boxes = prediction["pred_boxes"]
    scores = prediction["pred_scores"]
    labels = prediction["pred_labels"]
    score_array = scores.detach().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores)
    indices = np.flatnonzero(score_array >= score_threshold)
    if len(indices):
        indices = indices[np.argsort(-score_array[indices], kind="stable")]
    indices = indices[:max_detections]
    if hasattr(scores, "new_tensor"):
        import torch

        selected = torch.as_tensor(indices, device=scores.device, dtype=torch.long)
        return {"pred_boxes": boxes[selected], "pred_scores": scores[selected], "pred_labels": labels[selected]}
    return {"pred_boxes": boxes[indices], "pred_scores": scores[indices], "pred_labels": labels[indices]}


def _atomic_jsonl(records: list[dict[str, Any]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    import torch
    from easydict import EasyDict

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CenterPoint inference")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    data_cfg = EasyDict(load_yaml_config(args.data_config))
    model_cfg = EasyDict(load_yaml_config(args.model_config))
    if tuple(model_cfg.CLASS_NAMES) != CLASS_NAMES or model_cfg.MODEL.NAME != "CenterPoint":
        raise ValueError("inference config must use CenterPoint and the fixed MORAI class order")
    data_cfg.DATA_SPLIT.test = args.split
    dataset_template, centerpoint_class = import_smoke_components(args.openpcdet_root)
    dataset_class = make_openpcdet_dataset(dataset_template)
    dataset = dataset_class(
        dataset_cfg=data_cfg,
        class_names=list(CLASS_NAMES),
        training=False,
        root_path=args.dataset.resolve(),
        logger=None,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=dataset.collate_batch,
        pin_memory=True,
        drop_last=False,
    )
    model = centerpoint_class(model_cfg=model_cfg.MODEL, num_class=len(CLASS_NAMES), dataset=dataset).cuda()
    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cuda")
    if "model_state" not in checkpoint:
        raise KeyError("checkpoint has no model_state")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    source_by_id = {sample_id: index for index, sample_id in enumerate(dataset.morai_core.sample_ids)}
    records: list[dict[str, Any]] = []
    latencies: list[float] = []

    with torch.no_grad():
        for batch in loader:
            remaining = None if args.max_frames is None else args.max_frames - len(records)
            if remaining is not None and remaining <= 0:
                break
            load_batch_to_cuda(batch)
            torch.cuda.synchronize()
            started = time.perf_counter()
            predictions, _ = model(batch)
            torch.cuda.synchronize()
            batch_ms = (time.perf_counter() - started) * 1000.0
            count = min(len(predictions), remaining) if remaining is not None else len(predictions)
            per_frame_ms = batch_ms / len(predictions)
            for frame_offset in range(count):
                sample_id = str(batch["frame_id"][frame_offset])
                source = dataset.morai_core[source_by_id[sample_id]]
                filtered = filter_prediction(predictions[frame_offset], args.score_threshold, args.max_detections)
                frame = convert_openpcdet_prediction(
                    sample_id=sample_id,
                    source_header_stamp_ns=source["source_header_stamp_ns"],
                    prediction=filtered,
                    class_names=CLASS_NAMES,
                    frame_id=source["coordinate_frame"],
                    inference_time_ms=per_frame_ms,
                )
                records.append(asdict(frame))
                latencies.append(per_frame_ms)

    _atomic_jsonl(records, args.output.resolve())
    latency_array = np.asarray(latencies, dtype=np.float64)
    prediction_count = sum(len(record["detections"]) for record in records)
    summary = {
        "purpose": "offline_interface_validation_only",
        "frames_processed": len(records),
        "total_predictions": prediction_count,
        "mean_predictions_per_frame": prediction_count / len(records) if records else 0.0,
        "inference_latency_mean_ms": float(latency_array.mean()) if len(latency_array) else 0.0,
        "inference_latency_p50_ms": float(np.percentile(latency_array, 50)) if len(latency_array) else 0.0,
        "inference_latency_p95_ms": float(np.percentile(latency_array, 95)) if len(latency_array) else 0.0,
        "inference_latency_max_ms": float(latency_array.max()) if len(latency_array) else 0.0,
        "score_threshold": args.score_threshold,
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_model_state_loaded": True,
        "model_state_strict_load": True,
        "architecture_config_match": True,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_iteration": int(checkpoint.get("iteration", -1)),
        "model_architecture": model_cfg.MODEL.NAME,
        "class_names": list(CLASS_NAMES),
        "dataset_version": dataset.morai_core.dataset_version,
        "cuda_device": torch.cuda.get_device_name(0),
        "output_path": str(args.output.resolve()),
    }
    write_json_atomic(summary, args.summary.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
