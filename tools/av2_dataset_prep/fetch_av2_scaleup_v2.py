#!/usr/bin/env python3
"""Read-only, selective AV2 Motion Forecasting Scale-Up v2 fetch tool.

Reuses `fetch_av2_stage0.py`'s own listing/selection/download/manifest
primitives verbatim (no duplicated S3/parsing logic). Generalizes
`fetch_av2_stage1.py`'s single-`--exclude-manifest` behavior to accept
**multiple** exclude manifests (repeatable `--exclude-manifest`), so the
10k TRAIN scale-up selection can be built against the UNION of every
scenario id ever used by Stage-0 (120) and the Stage-1 pilot (2,000),
rather than against just one of them.

Usage (TRAIN, excluding both prior experiments)::

    python3 fetch_av2_scaleup_v2.py --split train --pool-size 40000 \\
        --count 10000 --seed 20260920 \\
        --exclude-manifest ~/datasets/av2/manifests/stage0_scenarios.json \\
        --exclude-manifest ~/datasets/av2/manifests/stage1_pilot_scenarios.json \\
        --dest ~/datasets/av2/raw_staging_scaleup_v2/train \\
        --manifest-out ~/datasets/av2/manifests/scaleup_v2_train_scenarios.json

Usage (official VAL, no prior VAL usage exists but the overlap check is
still run against the same exclusion union for rigor)::

    python3 fetch_av2_scaleup_v2.py --split val --pool-size 5000 \\
        --count 1000 --seed 20260921 \\
        --exclude-manifest ~/datasets/av2/manifests/stage0_scenarios.json \\
        --exclude-manifest ~/datasets/av2/manifests/stage1_pilot_scenarios.json \\
        --dest ~/datasets/av2/raw_staging_scaleup_v2/official_val \\
        --manifest-out ~/datasets/av2/manifests/scaleup_v2_official_val_scenarios.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_av2_stage0 import (  # noqa: E402
    Av2FetchError,
    SUPPORTED_SPLITS,
    build_manifest,
    download_scenario_parquet,
    list_scenario_id_pool,
    scenario_remote_key,
    select_stage0,
)


def load_excluded_scenario_ids(exclude_manifest_paths: list[str]) -> set[str]:
    """Reads any number of Stage-0-shaped scenario manifests (the exact
    ``av2_stage{0,1}_scenario_manifest_v1`` schema both prior fetch tools
    write) and returns the UNION of every scenario id they already used.
    A missing manifest path is a hard error -- this must never silently
    treat "manifest not found" as "nothing to exclude"."""
    excluded: set[str] = set()
    for path in exclude_manifest_paths:
        p = Path(path).expanduser()
        if not p.is_file():
            raise Av2FetchError(f"exclude manifest not found: {p}")
        manifest = json.loads(p.read_text("utf-8"))
        excluded |= {s["scenario_id"] for s in manifest["scenarios"]}
    return excluded


def select_scaleup(pool: list[str], count: int, seed: int, excluded: set[str]) -> list[str]:
    """Filters `excluded` out of `pool` BEFORE sampling (never samples
    then discards), then delegates to the unchanged Stage-0 selection
    rule (uniform random sample without replacement, seeded)."""
    available = [sid for sid in pool if sid not in excluded]
    if count > len(available):
        raise Av2FetchError(
            f"requested count={count} exceeds available pool size={len(available)} "
            f"after excluding {len(excluded)} previously-used scenario ids from a pool of {len(pool)}"
        )
    return select_stage0(available, count, seed)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="train", choices=SUPPORTED_SPLITS)
    parser.add_argument("--pool-size", type=int, required=True,
                         help="scenario-id prefixes to list before exclusion + sampling (listing only)")
    parser.add_argument("--count", type=int, required=True, help="scenario count to select and download")
    parser.add_argument("--seed", type=int, required=True, help="deterministic selection seed")
    parser.add_argument("--exclude-manifest", action="append", default=[],
                         help="path to a prior scenario manifest whose ids must never reappear "
                              "(repeatable -- pass once per prior experiment)")
    parser.add_argument("--dest", required=True, help="raw staging root")
    parser.add_argument("--manifest-out", required=True, help="path to write the reproducibility manifest")
    parser.add_argument("--dry-run", action="store_true", help="list + select only, do not download")
    parser.add_argument("--max-bytes", type=int, default=10 * 1024 * 1024 * 1024)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    dest_root = Path(args.dest).expanduser()
    manifest_out = Path(args.manifest_out).expanduser()

    excluded = load_excluded_scenario_ids(args.exclude_manifest)
    print(f"[0/3] Loaded {len(excluded)} previously-used scenario ids to exclude "
          f"from {len(args.exclude_manifest)} manifest(s)", file=sys.stderr)

    print(f"[1/3] Listing up to {args.pool_size} scenario ids under split={args.split!r} ...", file=sys.stderr)
    pool = list_scenario_id_pool(args.split, args.pool_size)
    print(f"      listed pool size = {len(pool)}", file=sys.stderr)

    overlap = set(pool) & excluded
    print(f"      {len(overlap)} of the {len(excluded)} excluded ids were present in this pool", file=sys.stderr)

    selected = select_scaleup(pool, args.count, args.seed, excluded)
    assert not (set(selected) & excluded), "internal error: excluded scenario id leaked into selection"
    print(f"[2/3] Selected {len(selected)} scenarios (seed={args.seed}), "
          f"0 overlap with prior experiments (verified)", file=sys.stderr)

    if args.dry_run:
        print("[dry-run] no download performed. Selected scenario ids:", file=sys.stderr)
        for sid in selected:
            print(f"  {sid}  ->  {scenario_remote_key(args.split, sid)}")
        return 0

    print(f"[3/3] Downloading {len(selected)} scenario parquet files to {dest_root} ...", file=sys.stderr)
    results = []
    running_bytes = 0
    failed: list[dict] = []
    for i, sid in enumerate(selected, 1):
        try:
            result = download_scenario_parquet(args.split, sid, dest_root)
        except Exception as exc:  # noqa: BLE001 -- record and continue, never crash the whole 10k run on one bad key
            failed.append({"scenario_id": sid, "error": str(exc)})
            continue
        results.append(result)
        running_bytes += 0 if result.already_present else result.bytes_downloaded
        if running_bytes > args.max_bytes:
            raise Av2FetchError(f"hard stop: downloaded {running_bytes} bytes, exceeds --max-bytes={args.max_bytes}")
        if i % 200 == 0 or i == len(selected):
            print(f"      {i}/{len(selected)} done, {running_bytes} bytes downloaded this run, "
                  f"{len(failed)} failed so far", file=sys.stderr)

    manifest = build_manifest(args.split, args.pool_size, args.seed, results)
    manifest["manifest_schema_version"] = "av2_scaleup_v2_scenario_manifest_v1"
    manifest["excluded_manifests"] = [Path(p).name for p in args.exclude_manifest]
    manifest["excluded_scenario_count"] = len(excluded)
    manifest["prior_overlap_in_pool"] = len(overlap)
    manifest["prior_overlap_in_selection"] = 0
    manifest["failed_scenario_ids"] = failed
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Manifest written to {manifest_out}", file=sys.stderr)
    print(f"Total bytes downloaded this run: "
          f"{sum(0 if r.already_present else r.bytes_downloaded for r in results)}", file=sys.stderr)
    if failed:
        print(f"WARNING: {len(failed)} scenario(s) failed to download -- see manifest failed_scenario_ids",
              file=sys.stderr)
        return 2
    if len(results) != args.count:
        print(f"WARNING: expected {args.count} results, got {len(results)}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
