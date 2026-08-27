#!/usr/bin/env python3
"""GT-FREE DESCRIPTIVE REPRODUCTION for Camera/LiDAR Extension 2.

This bounded runner replays the Stage-1 fusion core on stored Stage-2 detector
inputs and then invokes the canonical Stage-3 shadow/diagnostic runner. It does
not rerun a detector, download data/weights, alter ROS configuration, or make
accuracy/identity claims.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

from ad_lidar_perception.camera_lidar_fusion_core import (
    CameraDetection,
    CameraIntrinsics,
    CameraModel,
    FusionConfig,
    LidarDetection,
    fuse,
    rigid_transform_from_tf_chain,
)
from ad_lidar_perception.camera_lidar_fusion_stage2_core import (
    GeometryFilterRule,
    passes_geometry_filter,
)


EXPECTED_YOLO_SHA256 = (
    "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"
)
DEFAULT_SAMPLE_FRAMES = 12


def _existing_file(name: str, value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"{name} is not a file: {path}")
    return path


def _existing_directory(name: str, value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise SystemExit(f"{name} is not a directory: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    result = {}
    for distribution in (
        "numpy",
        "scipy",
        "matplotlib",
        "PyYAML",
        "torch",
        "ultralytics",
        "mcap",
        "mcap-ros2-support",
    ):
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[distribution] = "MISSING"
    return result


def _camera_model(tf_static: Path) -> CameraModel:
    transforms = {
        item["child_frame"]: item
        for item in json.loads(tf_static.read_text(encoding="utf-8"))
    }
    matrix = rigid_transform_from_tf_chain(
        [
            {**transforms["lidar_link"], "invert": False},
            {**transforms["camera_front_link"], "invert": True},
            {**transforms["camera_front_optical_frame"], "invert": True},
        ]
    )
    return CameraModel(
        CameraIntrinsics.from_fov(
            1280, 720, 90.0, calibration_quality="B_reconstructed"
        ),
        matrix,
    )


def _lidar(record: dict) -> LidarDetection:
    return LidarDetection(
        record["x"],
        record["y"],
        record["z"],
        record["length"],
        record["width"],
        record["height"],
        record["yaw"],
        record["score"],
        record["source_index"],
    )


def _camera(record: dict) -> CameraDetection:
    return CameraDetection(
        record["x1"],
        record["y1"],
        record["x2"],
        record["y2"],
        record["confidence"],
        record["semantic_class"],
        record["raw_class_name"],
        record["source_index"],
    )


def replay_fusion(stage2_root: Path, tf_static: Path, sample_frames: int) -> dict:
    records_path = stage2_root / "data/consecutive_frame_records.json"
    frames = json.loads(records_path.read_text(encoding="utf-8"))["frames"]
    if not 1 <= sample_frames <= len(frames):
        raise SystemExit(f"sample_frames must be in [1,{len(frames)}]")
    model = _camera_model(tf_static)
    config = FusionConfig(
        association_measure="iou",
        iou_gate=0.10,
        center_distance_gate=0.10,
        camera_min_confidence=0.25,
    )
    rule = GeometryFilterRule("stage2_conservative", min_dimension_m=0.15)
    durations = []
    expected = []
    observed = []
    deterministic = True
    for frame in frames[:sample_frames]:
        lidar = [item for item in map(_lidar, frame["lidar"]) if passes_geometry_filter(item, rule)]
        kept = set(frame["camera_kept_source_indices"])
        camera = [
            item
            for item in map(_camera, frame["camera_raw"])
            if item.source_index in kept
        ]
        started = time.perf_counter()
        first = fuse(lidar, camera, model, config)
        durations.append((time.perf_counter() - started) * 1000.0)
        second = fuse(lidar, camera, model, config)
        first_signature = sorted(
            (
                item.lidar.source_index,
                item.camera.source_index,
                round(item.association_score, 12),
            )
            for item in first.fused
        )
        second_signature = sorted(
            (
                item.lidar.source_index,
                item.camera.source_index,
                round(item.association_score, 12),
            )
            for item in second.fused
        )
        deterministic = deterministic and first_signature == second_signature
        observed.extend((frame["lidar_frame_index"], *item) for item in first_signature)
        expected.extend(
            (
                frame["lidar_frame_index"],
                item["lidar_index"],
                item["camera_source_index"],
                round(item["score"], 12),
            )
            for item in frame["iou_matches_conservative"]
        )
    observed.sort()
    expected.sort()
    if observed != expected:
        raise RuntimeError("bounded fusion replay diverged from canonical Stage-2 records")
    ordered = sorted(durations)
    p95_index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered)) - 1))
    return {
        "claim_boundary": "GT-FREE DESCRIPTIVE REPRODUCTION",
        "same_stage1_fusion_core": True,
        "sample_frames": sample_frames,
        "accepted_matches": len(observed),
        "canonical_signature_exact": True,
        "deterministic_double_replay": bool(deterministic),
        "fusion_core_latency_median_ms": statistics.median(durations),
        "fusion_core_latency_p95_ms": ordered[p95_index],
        "intrinsics_quality": "B_reconstructed",
        "geometry_rule": "min(length,width,height) > 0.15 m",
    }


def run_stage3(stage2_root: Path, runner: Path, output_root: Path) -> dict:
    stage3_output = output_root / "stage3_diagnostic"
    stage3_output.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "stage3_diagnostic.log"
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", str(output_root / "matplotlib"))
    command = [
        sys.executable,
        str(runner),
        "--stage2-root",
        str(stage2_root),
        "--output-root",
        str(stage3_output),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            check=False,
            text=True,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"Stage-3 diagnostic replay failed; see {log_path}")
    summary = json.loads(
        (stage3_output / "data/stage3_run_summary.json").read_text(encoding="utf-8")
    )
    return {
        "frames": summary["input"]["frames"],
        "lidar_detections": summary["input"]["lidar_detections"],
        "candidate_pairs": summary["selected"]["candidate_pairs"],
        "assignments": summary["selected"]["actual_assignments"],
        "candidate_pairs_with_mature_semantics": summary["selected"][
            "pairs_with_semantic_evidence"
        ],
        "candidate_pairs_with_nonzero_penalty": summary["selected"][
            "candidate_pairs_affected_by_nonzero_cost"
        ],
        "assignment_changes": summary["selected"]["baseline_assignments_changed"],
        "semantic_disabled_state_equivalent": summary["safety"][
            "semantic_disabled_state_equivalent_to_original"
        ],
        "lambda_zero_state_equivalent": summary["safety"][
            "lambda_zero_state_equivalent"
        ],
        "selected_lambda": summary["semantic_design"]["selected_lambda"],
        "verdict_preserved": "CASE_B",
        "log": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-root", required=True)
    parser.add_argument("--stage3-runner", required=True)
    parser.add_argument("--tf-static", required=True)
    parser.add_argument("--yolo-weight", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--sample-frames", type=int, default=DEFAULT_SAMPLE_FRAMES)
    args = parser.parse_args()

    stage2_root = _existing_directory("stage2_root", args.stage2_root)
    stage3_runner = _existing_file("stage3_runner", args.stage3_runner)
    tf_static = _existing_file("tf_static", args.tf_static)
    yolo_weight = _existing_file("yolo_weight", args.yolo_weight)
    weight_sha = _sha256(yolo_weight)
    if weight_sha != EXPECTED_YOLO_SHA256:
        raise SystemExit(
            f"yolo_weight SHA256 mismatch: expected {EXPECTED_YOLO_SHA256}, got {weight_sha}"
        )
    if args.output_root:
        output_root = Path(args.output_root).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
    else:
        output_root = Path(tempfile.mkdtemp(prefix="heven_extension2_repro."))

    fusion = replay_fusion(stage2_root, tf_static, args.sample_frames)
    stage3 = run_stage3(stage2_root, stage3_runner, output_root)
    result = {
        "claim_boundary": "GT-FREE DESCRIPTIVE REPRODUCTION",
        "output_root": str(output_root),
        "python": sys.executable,
        "package_versions": _package_versions(),
        "external_yolo_weight": {
            "path": str(yolo_weight),
            "sha256": weight_sha,
            "expected_sha256": EXPECTED_YOLO_SHA256,
            "download_attempted": False,
        },
        "fusion_replay": fusion,
        "semantic_diagnostic_replay": stage3,
    }
    summary_path = output_root / "extension2_reproduction_summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
