#!/usr/bin/env python3
"""Convert OpenPCDet predictions to a detector-neutral offline contract."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np


SCHEMA = "heven.offline_detection.v1"


@dataclass(frozen=True)
class CommonDetection:
    class_name: str
    score: float
    box_lidar: tuple[float, float, float, float, float, float, float]


@dataclass(frozen=True)
class CommonDetectionFrame:
    schema: str
    sample_id: str
    source_header_stamp_ns: int
    frame_id: str
    inference_time_ms: float | None
    detections: tuple[CommonDetection, ...]


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def convert_openpcdet_prediction(
    *,
    sample_id: str,
    source_header_stamp_ns: int,
    prediction: dict[str, Any],
    class_names: Iterable[str],
    frame_id: str = "lidar_link",
    inference_time_ms: float | None = None,
) -> CommonDetectionFrame:
    """Map OpenPCDet's 1-based labels and 7D lidar boxes without geometry changes."""
    boxes = _as_numpy(prediction["pred_boxes"])
    scores = _as_numpy(prediction["pred_scores"]).reshape(-1)
    labels = _as_numpy(prediction["pred_labels"]).reshape(-1)
    names = tuple(class_names)
    if boxes.ndim != 2 or boxes.shape[1] < 7:
        raise ValueError(f"pred_boxes must be Nx7 or wider, got {boxes.shape}")
    if not (len(boxes) == len(scores) == len(labels)):
        raise ValueError("prediction array lengths differ")
    if source_header_stamp_ns <= 0:
        raise ValueError("source_header_stamp_ns must be positive")
    if not sample_id or frame_id != "lidar_link":
        raise ValueError("sample_id is required and offline predictions use lidar_link")
    if inference_time_ms is not None and (
        not math.isfinite(inference_time_ms) or inference_time_ms < 0.0
    ):
        raise ValueError("inference_time_ms must be finite and non-negative")

    detections = []
    for box, score_value, label_value in zip(boxes, scores, labels):
        score = float(score_value)
        label = int(label_value)
        values = [float(value) for value in box[:7]]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("prediction box contains non-finite values")
        if min(values[3:6]) <= 0.0:
            raise ValueError("prediction box dimensions must be positive")
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("prediction score must be in [0, 1]")
        if not 1 <= label <= len(names):
            raise ValueError(f"prediction label {label} is outside 1..{len(names)}")
        values[6] = math.atan2(math.sin(values[6]), math.cos(values[6]))
        detections.append(
            CommonDetection(names[label - 1], score, tuple(values))  # type: ignore[arg-type]
        )
    return CommonDetectionFrame(
        schema=SCHEMA,
        sample_id=sample_id,
        source_header_stamp_ns=int(source_header_stamp_ns),
        frame_id=frame_id,
        inference_time_ms=inference_time_ms,
        detections=tuple(detections),
    )


def benchmark_centers(frame: CommonDetectionFrame) -> tuple[tuple[float, float], ...]:
    """Return exactly the XY centers consumed by the STEP 02 benchmark metric."""
    return tuple((item.box_lidar[0], item.box_lidar[1]) for item in frame.detections)


def append_jsonl(path: Path | str, frame: CommonDetectionFrame) -> None:
    path = Path(path)
    payload = asdict(frame)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")


def frame_from_dict(payload: dict[str, Any]) -> CommonDetectionFrame:
    """Validate and decode one HEVEN JSONL object for offline benchmarks."""
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"unsupported offline detection schema: {payload.get('schema')!r}")
    detections = tuple(
        CommonDetection(
            class_name=str(item["class_name"]),
            score=float(item["score"]),
            box_lidar=tuple(float(value) for value in item["box_lidar"]),  # type: ignore[arg-type]
        )
        for item in payload.get("detections", [])
    )
    prediction = {
        "pred_boxes": np.asarray([item.box_lidar for item in detections], dtype=np.float32).reshape(-1, 7),
        "pred_scores": np.asarray([item.score for item in detections], dtype=np.float32),
        "pred_labels": np.asarray(
            [tuple(("vehicle", "pedestrian", "obstacle")).index(item.class_name) + 1 for item in detections],
            dtype=np.int64,
        ),
    }
    return convert_openpcdet_prediction(
        sample_id=str(payload["sample_id"]),
        source_header_stamp_ns=int(payload["source_header_stamp_ns"]),
        prediction=prediction,
        class_names=("vehicle", "pedestrian", "obstacle"),
        frame_id=str(payload.get("frame_id", "lidar_link")),
        inference_time_ms=(
            None if payload.get("inference_time_ms") is None else float(payload["inference_time_ms"])
        ),
    )


def read_jsonl(path: Path | str) -> Iterator[CommonDetectionFrame]:
    """Read records in file order; inference latency remains model-only metadata."""
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                yield frame_from_dict(payload)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid record at line {line_number}: {error}") from error


def benchmark_records(path: Path | str) -> Iterator[dict[str, Any]]:
    """Small STEP 02 adapter preserving stamp, centers, class, and score."""
    for frame in read_jsonl(path):
        yield {
            "sample_id": frame.sample_id,
            "source_header_stamp_ns": frame.source_header_stamp_ns,
            "centers_xy": benchmark_centers(frame),
            "classes": tuple(item.class_name for item in frame.detections),
            "scores": tuple(item.score for item in frame.detections),
        }
