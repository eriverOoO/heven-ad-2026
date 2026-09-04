"""``ad_morai_dataset_export_av2_kalmannet`` -- downloaded AV2 Motion
Forecasting scenarios -> derived HEVEN KalmanNet **sharded** dataset.

    ros2 run ad_morai_bridge_dev ad_morai_dataset_export_av2_kalmannet \\
        --input-root ~/datasets/av2/raw_staging \\
        --scenario-manifest ~/datasets/av2/manifests/stage0_scenarios.json \\
        --output-root ~/datasets/av2/processed/av2_kalmannet_v1 \\
        --split train

Deliberately download-free: this CLI only reads already-downloaded
``scenario_<id>.parquet`` files (see ``tools/av2_dataset_prep/
fetch_av2_stage0.py`` for the separate, read-only download tool -- task
instruction: "Download and preprocessing must remain separate").

**The canonical export writes CLEAN measurements only** (one shard NPZ
per ``--scenarios-per-shard`` scenarios, not one file per segment).
``--corruption-config``/``--seed`` here only annotate
``export_manifest.json["default_load_time_corruption_config"]`` -- the
recommended policy for a future training loader to apply at read time
(``av2_kalmannet_shard_loader.SegmentArrays.to_kalmannet_sequence()``);
neither flag changes a single byte written to the shard files.
``--materialize-corruption-debug-root``, if given, additionally writes a
SECOND, clearly-labeled corrupted shard set for debugging -- this is
never the recommended Stage-1 storage workflow.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from ad_morai_bridge_dev.dataset.av2_motion_forecasting_adapter import (
    Av2KalmanNetExporter,
    CoordinateConfig,
    SUPPORTED_AV2_SPLITS,
    load_corruption_config,
    load_segment_config,
    load_shard_config,
    materialize_corrupted_debug_copy,
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


def _load_scenario_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True,
                         help="raw staging root containing <split>/<scenario_id>/scenario_<id>.parquet")
    parser.add_argument("--scenario-manifest", type=Path, required=True,
                         help="Stage-0 scenario manifest JSON produced by tools/av2_dataset_prep/fetch_av2_stage0.py")
    parser.add_argument("--output-root", type=Path, help="derived dataset output root")
    parser.add_argument("--split", default="train", choices=SUPPORTED_AV2_SPLITS)
    parser.add_argument("--coordinate-mode", default="first_state_relative",
                         choices=["first_state_relative"])
    parser.add_argument("--segment-config", type=Path, help="segment/gap policy config YAML")
    parser.add_argument("--shard-config", type=Path, help="shard-size policy config YAML (scenarios_per_shard)")
    parser.add_argument("--corruption-config", type=Path,
                         help="recommended load-time corruption config YAML (metadata only -- see module docstring)")
    parser.add_argument("--seed", type=int, default=None,
                         help="override corruption_config.seed (metadata only, unless --materialize-corruption-debug-root is also given)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan (scenario count, configs); no export")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--materialize-corruption-debug-root", type=Path, default=None,
        help="OPTIONAL, debug only: also write a second corrupted shard set here using --corruption-config/--seed. "
             "Never the recommended Stage-1 storage workflow.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    manifest = _load_scenario_manifest(args.scenario_manifest)
    manifest_split = manifest.get("split")
    if manifest_split and manifest_split != args.split:
        print(
            f"warning: scenario manifest split={manifest_split!r} != --split={args.split!r}; "
            "using --split as authoritative",
            file=sys.stderr,
        )
    scenario_ids = sorted(entry["scenario_id"] for entry in manifest.get("scenarios", []))
    if not scenario_ids:
        print("scenario manifest contains no scenarios", file=sys.stderr)
        return 2

    segment_config = load_segment_config(args.segment_config)
    shard_config = load_shard_config(args.shard_config)
    corruption_config = load_corruption_config(args.corruption_config)
    if args.seed is not None:
        corruption_config = type(corruption_config)(
            **{**corruption_config.__dict__, "seed": args.seed}
        )
    coordinate_config = CoordinateConfig(mode=args.coordinate_mode)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "input_root": str(args.input_root),
                    "split": args.split,
                    "scenario_count": len(scenario_ids),
                    "segment_config": json.loads(segment_config.canonical_json()),
                    "coordinate_config": json.loads(coordinate_config.canonical_json()),
                    "shard_config": json.loads(shard_config.canonical_json()),
                    "estimated_shard_count": (len(scenario_ids) + shard_config.scenarios_per_shard - 1)
                    // shard_config.scenarios_per_shard,
                    "default_load_time_corruption_config": json.loads(corruption_config.canonical_json()),
                    "scenario_manifest_selection_rule": manifest.get("selection_rule"),
                    "scenario_manifest_seed": manifest.get("selection_seed"),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if not args.output_root:
        print("--output-root is required (unless --dry-run)", file=sys.stderr)
        return 2

    exporter = Av2KalmanNetExporter(
        input_root=args.input_root,
        output_root=args.output_root,
        split=args.split,
        scenario_ids=scenario_ids,
        segment_config=segment_config,
        coordinate_config=coordinate_config,
        corruption_config=corruption_config,
        adapter_commit=_adapter_commit(),
        scenario_manifest_provenance={
            "manifest_path": str(args.scenario_manifest),
            "dataset_source": manifest.get("dataset_source"),
            "bucket": manifest.get("bucket"),
            "dataset_prefix": manifest.get("dataset_prefix"),
            "split": manifest.get("split"),
            "selection_rule": manifest.get("selection_rule"),
            "selection_pool_size": manifest.get("selection_pool_size"),
            "selection_seed": manifest.get("selection_seed"),
            "scenario_count": manifest.get("scenario_count"),
        },
        shard_config=shard_config,
        overwrite=args.overwrite,
    )
    result = exporter.export()
    print(
        json.dumps(
            {
                "output": str(args.output_root),
                "content_fingerprint": result.content_fingerprint,
                "shard_count": result.manifest["shard_count"],
                "counts": result.manifest["counts"],
                "av2_object_type_counts": result.manifest["av2_object_type_counts"],
                "coarse_object_type_counts": result.manifest["coarse_object_type_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )

    if args.materialize_corruption_debug_root:
        print(
            "WARNING: materializing a DEBUG corrupted shard copy -- this is NOT the "
            "recommended Stage-1 storage workflow (corruption should be applied at load "
            "time, see av2_kalmannet_shard_loader.py)",
            file=sys.stderr,
        )
        debug_result = materialize_corrupted_debug_copy(
            clean_output_root=args.output_root,
            debug_output_root=args.materialize_corruption_debug_root,
            corruption_config=corruption_config,
            overwrite=args.overwrite,
        )
        print(json.dumps({"debug_output": str(args.materialize_corruption_debug_root),
                           "content_fingerprint": debug_result.content_fingerprint}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
