#!/usr/bin/env python3
"""Read-only, selective AV2 Motion Forecasting Stage-1 fetch tool.

Reuses `fetch_av2_stage0.py`'s own listing/selection/download primitives
verbatim (no duplicated S3/parsing logic) and adds exactly one new
behavior Stage-0 didn't need: **excluding Stage-0's own 120 already-used
scenario IDs from the Stage-1 pool before sampling**, so a Stage-0
scenario can never appear in a new Stage-1 split (train, val, OR test) --
eliminating any chance of Stage-0/Stage-1 leakage or confusion, per this
task's own "safest default: exclude all 120 Stage-0 scenario IDs" policy.

Stage-0 used `pool_size=2000, seed=20260905` (see
``ad_morai_bridge_dev/config/dataset_factory/av2_motion_forecasting_adapter/
stage0_scenarios.manifest.json``) -- reusing that same 2000-scenario pool
for a NEW 2,000-scenario Stage-1 selection would leave too little room
after excluding the 120 already-used ids, so Stage-1 lists a materially
larger pool (default 10,000) before excluding + sampling.

Usage::

    python3 fetch_av2_stage1.py --split train --pool-size 10000 \\
        --count 2000 --seed 20260910 \\
        --exclude-manifest ~/datasets/av2/manifests/stage0_scenarios.json \\
        --dest ~/datasets/av2/raw_staging_stage1 \\
        --manifest-out ~/datasets/av2/manifests/stage1_scenarios.json
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


def load_excluded_scenario_ids(exclude_manifest_path: Path) -> set[str]:
    """Reads a Stage-0-shaped scenario manifest (the exact
    `av2_stage0_scenario_manifest_v1` schema `fetch_av2_stage0.py` writes)
    and returns the set of scenario ids it already used -- these are
    excluded from ANY Stage-1 pool before sampling, so they can never
    reappear in Stage-1 train, val, or test."""
    manifest = json.loads(Path(exclude_manifest_path).expanduser().read_text("utf-8"))
    return {s["scenario_id"] for s in manifest["scenarios"]}


def select_stage1(pool: list[str], count: int, seed: int, excluded: set[str]) -> list[str]:
    """Filters `excluded` out of `pool` BEFORE sampling (never samples
    then discards -- a discard-after-sample approach could silently
    return fewer than `count` scenarios or bias the selection toward
    pool positions least likely to collide), then delegates to the
    Stage-0 selection rule unchanged (uniform random sample without
    replacement, seeded)."""
    available = [sid for sid in pool if sid not in excluded]
    if count > len(available):
        raise Av2FetchError(
            f"requested count={count} exceeds available pool size={len(available)} "
            f"after excluding {len(excluded)} Stage-0 scenario ids from a pool of {len(pool)}"
        )
    return select_stage0(available, count, seed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="train", choices=SUPPORTED_SPLITS)
    parser.add_argument("--pool-size", type=int, default=10000,
                         help="scenario-id prefixes to list before exclusion + sampling (listing only)")
    parser.add_argument("--count", type=int, required=True, help="Stage-1 scenario count to select and download")
    parser.add_argument("--seed", type=int, required=True,
                         help="deterministic selection seed -- MUST differ from Stage-0's 20260905")
    parser.add_argument("--exclude-manifest", required=True,
                         help="path to the Stage-0 scenario manifest whose scenario ids must never reappear")
    parser.add_argument("--dest", required=True, help="raw staging root")
    parser.add_argument("--manifest-out", required=True, help="path to write the Stage-1 reproducibility manifest")
    parser.add_argument("--dry-run", action="store_true", help="list + select only, do not download")
    parser.add_argument("--max-bytes", type=int, default=10 * 1024 * 1024 * 1024)
    args = parser.parse_args(argv)

    if args.seed == 20260905:
        raise Av2FetchError("--seed must differ from Stage-0's own selection seed (20260905)")

    dest_root = Path(args.dest).expanduser()
    manifest_out = Path(args.manifest_out).expanduser()

    excluded = load_excluded_scenario_ids(args.exclude_manifest)
    print(f"[0/3] Loaded {len(excluded)} Stage-0 scenario ids to exclude from {args.exclude_manifest}",
          file=sys.stderr)

    print(f"[1/3] Listing up to {args.pool_size} scenario ids under split={args.split!r} ...", file=sys.stderr)
    pool = list_scenario_id_pool(args.split, args.pool_size)
    print(f"      listed pool size = {len(pool)}", file=sys.stderr)

    overlap = set(pool) & excluded
    print(f"      {len(overlap)} of the {len(excluded)} excluded Stage-0 ids were present in this pool "
          f"(expected, since Stage-0 itself was drawn from an earlier listing of the same live bucket)",
          file=sys.stderr)

    selected = select_stage1(pool, args.count, args.seed, excluded)
    assert not (set(selected) & excluded), "internal error: excluded scenario id leaked into Stage-1 selection"
    print(f"[2/3] Selected {len(selected)} Stage-1 scenarios (seed={args.seed}), 0 overlap with Stage-0 (verified)",
          file=sys.stderr)

    if args.dry_run:
        print("[dry-run] no download performed. Selected scenario ids:", file=sys.stderr)
        for sid in selected:
            print(f"  {sid}  ->  {scenario_remote_key(args.split, sid)}")
        return 0

    print(f"[3/3] Downloading {len(selected)} scenario parquet files to {dest_root} ...", file=sys.stderr)
    results = []
    running_bytes = 0
    for i, sid in enumerate(selected, 1):
        result = download_scenario_parquet(args.split, sid, dest_root)
        results.append(result)
        running_bytes += 0 if result.already_present else result.bytes_downloaded
        if running_bytes > args.max_bytes:
            raise Av2FetchError(f"hard stop: downloaded {running_bytes} bytes, exceeds --max-bytes={args.max_bytes}")
        if i % 50 == 0 or i == len(selected):
            print(f"      {i}/{len(selected)} done, {running_bytes} bytes downloaded this run", file=sys.stderr)

    manifest = build_manifest(args.split, args.pool_size, args.seed, results)
    manifest["manifest_schema_version"] = "av2_stage1_scenario_manifest_v1"
    manifest["excluded_stage0_manifest"] = str(Path(args.exclude_manifest).expanduser())
    manifest["excluded_stage0_scenario_count"] = len(excluded)
    manifest["stage0_overlap_in_pool"] = len(overlap)
    manifest["stage0_overlap_in_selection"] = 0
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Manifest written to {manifest_out}", file=sys.stderr)
    print(f"Total bytes downloaded this run: "
          f"{sum(0 if r.already_present else r.bytes_downloaded for r in results)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
