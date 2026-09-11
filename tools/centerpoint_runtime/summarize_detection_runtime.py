#!/usr/bin/env python3
"""Summarize existing DetectedObjects JSONL without inventing a frame rate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def summarize(records: list[dict]) -> dict:
    if not records:
        raise ValueError("no detection records")
    stamps = [int(record["header_stamp_ns"]) for record in records]
    latencies = [float(record["latency_ms"]) for record in records]
    if any(right <= left for left, right in zip(stamps, stamps[1:])):
        raise ValueError("header stamps must be strictly increasing")
    duration_sec = (stamps[-1] - stamps[0]) / 1e9
    return {
        "schema": "heven.centerpoint_runtime_summary.v1",
        "backend": records[0]["backend"],
        "messages": len(records),
        "output_hz": (len(records) - 1) / duration_sec if duration_sec > 0 else None,
        "latency_mean_ms": sum(latencies) / len(latencies),
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "latency_p99_ms": percentile(latencies, 0.99),
        "latency_max_ms": max(latencies),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.records.open(encoding="utf-8") as stream:
        summary = summarize([json.loads(line) for line in stream if line.strip()])
    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
