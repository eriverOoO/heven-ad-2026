"""``ad_morai_dataset_export_centerpoint`` - canonical MORAI tracking dataset
-> derived CenterPoint LiDAR dataset.

    ros2 run ad_morai_bridge_dev ad_morai_dataset_export_centerpoint \
        <source_dataset_root> --output <derived_root> [--split-plan plan.yaml]

The canonical source is never modified. No model is trained.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from ad_morai_bridge_dev.dataset.centerpoint_adapter import (
    CenterPointExporter,
    load_adapter_config,
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
    parser.add_argument("source_root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split-plan", type=Path, help="explicit train/val/test plan YAML")
    parser.add_argument("--class-map", type=Path, help="adapter class-map YAML")
    parser.add_argument(
        "--auto-ratios", default="0.7,0.15,0.15", help="train,val,test for auto-split"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_adapter_config(args.class_map)
    split_plan = (
        yaml.safe_load(args.split_plan.read_text("utf-8")) if args.split_plan else None
    )

    if args.validate_only:
        exporter = CenterPointExporter(
            args.source_root, args.source_root, config, _adapter_commit()
        )
        try:
            exporter.validate_source()
        except Exception as exc:  # noqa: BLE001
            print(f"source invalid: {exc}", file=sys.stderr)
            return 1
        print("source dataset is a valid morai_tracking_dataset_v1")
        return 0

    if not args.output:
        print("--output is required", file=sys.stderr)
        return 2

    exporter = CenterPointExporter(
        args.source_root, args.output, config, _adapter_commit(), overwrite=args.overwrite
    )

    if args.dry_run:
        preview = exporter.plan(split_plan)
        preview.pop("_frames", None)
        preview.pop("_split", None)
        print(json.dumps(preview, indent=2, sort_keys=True))
        return 0

    result = exporter.export(split_plan)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "content_fingerprint": result.content_fingerprint,
                "sample_count": result.manifest["sample_count"],
                "split_frame_counts": result.manifest["split_frame_counts"],
                "counts": result.manifest["counts"],
                "split_warnings": result.split_manifest["warnings"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
