"""``ad_morai_dataset_attach_kalmannet_measurements`` - attach sample-aligned
real detector outputs to a ``kalmannet_morai_trajectory_v1`` export.

    ros2 run ad_morai_bridge_dev ad_morai_dataset_attach_kalmannet_measurements \
        --trajectories <kalmannet_morai_trajectory_v1 export> \
        --canonical-dataset <morai_tracking_dataset_v1> \
        --detections <detector-export.jsonl> \
        --detector-source centerpoint \
        --output <root> [--config attachment_config.yaml]

No detector inference is run; no KalmanNet is trained. GT state is never
modified. Missing detections stay masked - they are not filled.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from ad_morai_bridge_dev.dataset.kalmannet_measurement_attachment import (
    KalmanNetMeasurementExporter,
    centerpoint_records_from_heven_offline,
    euclidean_records_from_ros_comparison,
    load_association_config,
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _adapter_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_repository_root(),
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--canonical-dataset", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument(
        "--detector-source", required=True, choices=("centerpoint", "euclidean")
    )
    parser.add_argument("--detector-config-id", default="unspecified")
    parser.add_argument(
        "--run-id", help="required for --detector-source euclidean (stamp-scoped alignment)"
    )
    parser.add_argument(
        "--detector-manifest",
        type=Path,
        help="optional JSON {expected_frames, failed_frames} completeness manifest",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-incomplete-detector", action="store_true")
    return parser.parse_args(argv)


def _load_detector_records(args: argparse.Namespace):
    manifest = (
        json.loads(args.detector_manifest.read_text("utf-8"))
        if args.detector_manifest
        else None
    )
    if args.detector_source == "centerpoint":
        return centerpoint_records_from_heven_offline(
            args.detections,
            detector_config_id=args.detector_config_id,
            detector_manifest=manifest,
        )
    if not args.run_id:
        raise SystemExit("--run-id is required for --detector-source euclidean")
    return euclidean_records_from_ros_comparison(
        args.detections,
        args.canonical_dataset,
        run_id=args.run_id,
        detector_config_id=args.detector_config_id,
        detector_manifest=manifest,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_association_config(args.config)
    detector_records = _load_detector_records(args)

    exporter = KalmanNetMeasurementExporter(
        args.trajectories,
        args.canonical_dataset,
        args.output or args.trajectories,
        detector_records,
        config,
        _adapter_commit(),
        overwrite=args.overwrite,
        allow_incomplete_detector=args.allow_incomplete_detector,
    )

    if args.dry_run:
        preview = exporter.plan()
        summary = {
            "trajectory_count": len(preview["index_rows"]),
            "detector_frame_count": len(detector_records.frames),
            "detector_completeness": detector_records.completeness,
            "association": json.loads(
                config.canonical_json(
                    effective_class_matching=exporter._effective_class_matching
                )
            ),
            "aggregate": preview["aggregate"],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if not args.output:
        print("--output is required", file=sys.stderr)
        return 2

    result = exporter.export()
    print(
        json.dumps(
            {
                "output": str(args.output),
                "content_fingerprint": result.content_fingerprint,
                "trajectory_count": result.manifest["trajectory_count"],
                "counts": result.manifest["counts"],
                "detector": result.manifest["detector"]["source"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
