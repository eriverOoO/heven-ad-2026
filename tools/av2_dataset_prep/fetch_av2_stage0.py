#!/usr/bin/env python3
"""Read-only, selective Argoverse 2 (AV2) Motion Forecasting Stage-0 fetch tool.

Deliberately separate from the AV2 -> HEVEN KalmanNet dataset adapter
(``ad_morai_bridge_dev/ad_morai_bridge_dev/dataset/av2_motion_forecasting_adapter*.py``):
this tool only lists and downloads raw AV2 scenario files; it performs no
KalmanNet-related parsing, transformation, or corruption. "Download and
preprocessing must remain separate" (task instruction).

No AWS account, ``s5cmd``, ``aws-cli``, or ``boto3`` is required: the
``argoverse`` S3 bucket permits anonymous, unsigned reads, so this module
uses only the Python standard library (``urllib.request`` + a minimal REST
``ListObjectsV2`` XML parse) to (1) list scenario-id prefixes under a
split, (2) deterministically select a bounded Stage-0 subset, and (3)
download only the ``scenario_<id>.parquet`` file for each selected
scenario -- never the companion ``log_map_archive_<id>.json`` map file,
since the KalmanNet v1 adapter needs no map geometry (audit finding,
``docs/perception/av2_kalmannet_adapter_v1.md``).

Resolved bucket layout (verified live against the real bucket, not
assumed from documentation -- the official docs disagreed on
"motion_forecasting" vs "motion-forecasting"; the real key uses a
hyphen)::

    s3://argoverse/datasets/av2/motion-forecasting/{train,val,test}/<scenario_id>/
        scenario_<scenario_id>.parquet   (needed)
        log_map_archive_<scenario_id>.json  (NOT downloaded by this tool)

Usage (network calls only when explicitly invoked; ``--dry-run`` performs
listing only, no download)::

    python3 fetch_av2_stage0.py --split train --pool-size 2000 \\
        --count 120 --seed 20260905 \\
        --dest ~/datasets/av2/raw_staging \\
        --manifest-out ~/datasets/av2/manifests/stage0_scenarios.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

BUCKET_HOST = "https://argoverse.s3.amazonaws.com"
BUCKET_NAME = "argoverse"
DATASET_PREFIX = "datasets/av2/motion-forecasting"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
SUPPORTED_SPLITS = ("train", "val", "test")
SCENARIO_PARQUET_TEMPLATE = "scenario_{scenario_id}.parquet"
MAP_ARCHIVE_TEMPLATE = "log_map_archive_{scenario_id}.json"


class Av2FetchError(RuntimeError):
    pass


def _list_page(prefix: str, continuation_token: str | None = None, max_keys: int = 1000) -> ET.Element:
    params = f"list-type=2&delimiter=/&prefix={prefix}&max-keys={max_keys}"
    if continuation_token:
        params += f"&continuation-token={urllib.request.quote(continuation_token, safe='')}"
    url = f"{BUCKET_HOST}/?{params}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        body = resp.read()
    return ET.fromstring(body)


def parse_listing_xml(xml_text: bytes | str) -> tuple[list[str], list[str], str | None]:
    """Pure parser (no network): returns (common_prefixes, object_keys, next_continuation_token).

    Split out from ``_list_page`` specifically so it is unit-testable
    against a fixture XML string with no network access.
    """
    root = ET.fromstring(xml_text)
    common_prefixes = [
        el.findtext(f"{S3_NS}Prefix") for el in root.findall(f"{S3_NS}CommonPrefixes")
    ]
    object_keys = [el.findtext(f"{S3_NS}Key") for el in root.findall(f"{S3_NS}Contents")]
    next_token = root.findtext(f"{S3_NS}NextContinuationToken")
    return (
        [p for p in common_prefixes if p],
        [k for k in object_keys if k],
        next_token,
    )


def list_scenario_id_pool(split: str, pool_size: int, *, _page_fetcher=None) -> list[str]:
    """List up to ``pool_size`` scenario-id prefixes under ``<split>/`` via
    paginated, delimiter='/' S3 listing (CommonPrefixes = one entry per
    scenario-id folder). No scenario content is downloaded here -- listing
    only. ``_page_fetcher`` is an injection seam for tests
    (``_page_fetcher(prefix, token) -> raw XML bytes``)."""
    if split not in SUPPORTED_SPLITS:
        raise Av2FetchError(f"unsupported split {split!r}; must be one of {SUPPORTED_SPLITS}")
    prefix = f"{DATASET_PREFIX}/{split}/"
    fetch = _page_fetcher or (lambda p, tok: _raw_page_bytes(p, tok))
    pool: list[str] = []
    token: str | None = None
    while len(pool) < pool_size:
        xml_bytes = fetch(prefix, token)
        common_prefixes, _keys, token = parse_listing_xml(xml_bytes)
        for cp in common_prefixes:
            # cp = "datasets/av2/motion-forecasting/train/<uuid>/"
            scenario_id = cp.rstrip("/").rsplit("/", 1)[-1]
            pool.append(scenario_id)
        if not token or not common_prefixes:
            break
    return pool[:pool_size]


def _raw_page_bytes(prefix: str, token: str | None) -> bytes:
    params = f"list-type=2&delimiter=/&prefix={prefix}&max-keys=1000"
    if token:
        params += f"&continuation-token={urllib.request.quote(token, safe='')}"
    url = f"{BUCKET_HOST}/?{params}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read()


def select_stage0(pool: list[str], count: int, seed: int) -> list[str]:
    """Deterministic selection rule: uniform random sample of ``count``
    scenario ids out of ``pool``, seeded -- never a hand-picked/"easy
    actors only" subset, and reproducible from (pool identity + seed)
    alone. ``pool`` order (raw S3 listing order, i.e. lexicographic over
    randomly-generated scenario UUIDs) carries no content-difficulty
    signal, so sampling from it is not equivalent to cherry-picking."""
    if count > len(pool):
        raise Av2FetchError(f"requested count={count} exceeds pool size={len(pool)}")
    rng = random.Random(seed)
    return sorted(rng.sample(pool, count))


@dataclass
class ScenarioFetchResult:
    scenario_id: str
    split: str
    remote_key: str
    local_path: str
    bytes_downloaded: int
    sha256: str
    already_present: bool


def scenario_remote_key(split: str, scenario_id: str) -> str:
    return f"{DATASET_PREFIX}/{split}/{scenario_id}/{SCENARIO_PARQUET_TEMPLATE.format(scenario_id=scenario_id)}"


def download_scenario_parquet(
    split: str, scenario_id: str, dest_root: Path, *, _opener=None
) -> ScenarioFetchResult:
    """Downloads only ``scenario_<id>.parquet`` for one scenario -- never
    the map archive (KalmanNet v1 needs no map geometry)."""
    remote_key = scenario_remote_key(split, scenario_id)
    url = f"{BUCKET_HOST}/{remote_key}"
    local_dir = dest_root / split / scenario_id
    local_path = local_dir / SCENARIO_PARQUET_TEMPLATE.format(scenario_id=scenario_id)
    if local_path.is_file() and local_path.stat().st_size > 0:
        digest = hashlib.sha256(local_path.read_bytes()).hexdigest()
        return ScenarioFetchResult(
            scenario_id, split, remote_key, str(local_path), local_path.stat().st_size, digest, True
        )
    local_dir.mkdir(parents=True, exist_ok=True)
    opener = _opener or (lambda u: urllib.request.urlopen(u, timeout=60))
    with opener(url) as resp:
        data = resp.read()
    tmp_path = local_path.with_suffix(local_path.suffix + ".part")
    tmp_path.write_bytes(data)
    tmp_path.rename(local_path)
    digest = hashlib.sha256(data).hexdigest()
    return ScenarioFetchResult(scenario_id, split, remote_key, str(local_path), len(data), digest, False)


def build_manifest(
    split: str,
    pool_size: int,
    seed: int,
    results: list[ScenarioFetchResult],
) -> dict:
    total_bytes = sum(r.bytes_downloaded for r in results)
    return {
        "manifest_schema_version": "av2_stage0_scenario_manifest_v1",
        "dataset_source": "argoverse2_motion_forecasting",
        "bucket": BUCKET_NAME,
        "dataset_prefix": DATASET_PREFIX,
        "split": split,
        "selection_rule": "uniform_random_sample_without_replacement_from_listed_pool",
        "selection_pool_size": pool_size,
        "selection_seed": seed,
        "scenario_count": len(results),
        "total_bytes_downloaded": total_bytes,
        "created_unix_time": time.time(),
        "scenarios": [
            {
                "scenario_id": r.scenario_id,
                "split": r.split,
                "remote_key": r.remote_key,
                "bytes": r.bytes_downloaded,
                "sha256": r.sha256,
                "already_present": r.already_present,
            }
            for r in results
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=SUPPORTED_SPLITS)
    parser.add_argument("--pool-size", type=int, default=2000,
                         help="how many scenario-id prefixes to list before sampling (listing only, no download)")
    parser.add_argument("--count", type=int, default=120, help="Stage-0 scenario count to select and download")
    parser.add_argument("--seed", type=int, default=20260905, help="deterministic selection seed")
    parser.add_argument("--dest", required=True, help="raw staging root, e.g. ~/datasets/av2/raw_staging")
    parser.add_argument("--manifest-out", required=True, help="path to write the reproducibility manifest JSON")
    parser.add_argument("--dry-run", action="store_true", help="list + select only, do not download")
    parser.add_argument("--max-bytes", type=int, default=10 * 1024 * 1024 * 1024,
                         help="hard stop: abort if projected/actual download would exceed this many bytes")
    args = parser.parse_args(argv)

    dest_root = Path(args.dest).expanduser()
    manifest_out = Path(args.manifest_out).expanduser()

    print(f"[1/3] Listing up to {args.pool_size} scenario ids under split={args.split!r} ...", file=sys.stderr)
    pool = list_scenario_id_pool(args.split, args.pool_size)
    print(f"      listed pool size = {len(pool)}", file=sys.stderr)

    selected = select_stage0(pool, args.count, args.seed)
    print(f"[2/3] Selected {len(selected)} scenarios (seed={args.seed}) for split={args.split!r}", file=sys.stderr)

    if args.dry_run:
        print("[dry-run] no download performed. Selected scenario ids:", file=sys.stderr)
        for sid in selected:
            print(f"  {sid}  ->  {scenario_remote_key(args.split, sid)}")
        return 0

    print(f"[3/3] Downloading {len(selected)} scenario parquet files to {dest_root} ...", file=sys.stderr)
    results: list[ScenarioFetchResult] = []
    running_bytes = 0
    for i, sid in enumerate(selected, 1):
        result = download_scenario_parquet(args.split, sid, dest_root)
        results.append(result)
        running_bytes += 0 if result.already_present else result.bytes_downloaded
        if running_bytes > args.max_bytes:
            raise Av2FetchError(
                f"hard stop: downloaded {running_bytes} bytes, exceeds --max-bytes={args.max_bytes}"
            )
        if i % 20 == 0 or i == len(selected):
            print(f"      {i}/{len(selected)} done, {running_bytes} bytes downloaded this run", file=sys.stderr)

    manifest = build_manifest(args.split, args.pool_size, args.seed, results)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Manifest written to {manifest_out}", file=sys.stderr)
    print(f"Total bytes downloaded this run: {sum(0 if r.already_present else r.bytes_downloaded for r in results)}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
