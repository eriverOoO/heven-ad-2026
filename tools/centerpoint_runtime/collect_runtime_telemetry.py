#!/usr/bin/env python3
"""Collect bounded 1 Hz host/GPU telemetry for a candidate runtime session."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path


GPU_QUERY = "utilization.gpu,memory.used,memory.free"


def meminfo() -> dict[str, int]:
    values = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.strip().split()[0]) * 1024
    return {key: values.get(key, 0) for key in ("MemAvailable", "SwapTotal", "SwapFree")}


def gpu_sample() -> dict[str, int | None]:
    if shutil.which("nvidia-smi") is None:
        return {"gpu_utilization_percent": None, "vram_used_mib": None, "vram_free_mib": None}
    completed = subprocess.run(
        ["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, check=False,
    )
    try:
        values = [int(value.strip()) for value in completed.stdout.splitlines()[0].split(",")]
        return {"gpu_utilization_percent": values[0], "vram_used_mib": values[1], "vram_free_mib": values[2]}
    except (IndexError, ValueError):
        return {"gpu_utilization_percent": None, "vram_used_mib": None, "vram_free_mib": None}


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-sec", type=float, default=60.0)
    parser.add_argument("--interval-sec", type=float, default=1.0)
    args = parser.parse_args()
    if args.duration_sec <= 0 or args.interval_sec <= 0:
        parser.error("duration and interval must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    samples = []
    deadline = time.monotonic() + args.duration_sec
    with args.output.open("w", encoding="utf-8") as stream:
        while time.monotonic() < deadline:
            memory = meminfo()
            sample = {"wall_time_ns": time.time_ns(), **memory, **gpu_sample()}
            samples.append(sample)
            stream.write(json.dumps(sample, sort_keys=True) + "\n")
            stream.flush()
            time.sleep(args.interval_sec)
    summary = {
        "samples": len(samples),
        "gpu_utilization_p50": percentile([row["gpu_utilization_percent"] for row in samples if row["gpu_utilization_percent"] is not None], 0.50),
        "gpu_utilization_p95": percentile([row["gpu_utilization_percent"] for row in samples if row["gpu_utilization_percent"] is not None], 0.95),
        "vram_used_mib_peak": max((row["vram_used_mib"] for row in samples if row["vram_used_mib"] is not None), default=None),
        "mem_available_bytes_min": min(row["MemAvailable"] for row in samples),
        "swap_used_bytes_peak": max(row["SwapTotal"] - row["SwapFree"] for row in samples),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
