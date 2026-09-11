"""KalmanNet Production Candidate Freeze + Runtime Integration Readiness v1.

Section 15: estimator-only latency/resource smoke test (not a
microbenchmark research task). Measures the cost of one
`AB3DMOTTracker.step()` call with N simultaneous, already-live tracks,
for `linear_kf` and `kalmannet` (the frozen production candidate),
at N in {1, 10, 50, 100}.

Usage:
    python3 latency_benchmark.py <kalmannet_checkpoint.pt> [--iterations N]
"""
from __future__ import annotations

import argparse
import json
import resource
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ad_lidar_perception"))

from ad_lidar_perception.ab3dmot_config import AB3DMOTConfig
from ad_lidar_perception.ab3dmot_core import AB3DMOTTracker, Detection


def _make_detections(n: int, t: float) -> list[Detection]:
    # Spread objects far enough apart (10 m) that the euclidean gate
    # never causes cross-track confusion, so every step is a clean
    # N-independent-track update, not an association stress test.
    return [
        Detection(x=float(i) * 10.0 + 0.05 * t, y=0.0, z=1.0, yaw=0.0, length=4.0, width=2.0, height=1.5)
        for i in range(n)
    ]


def build_tracker(state_estimator: str, kalmannet_checkpoint: str = "") -> AB3DMOTTracker:
    config = AB3DMOTConfig(
        association_metric="euclidean",
        matcher="hungarian",
        euclidean_gate_m=3.0,
        yaw_measurement_mode="unobserved",
        state_estimator=state_estimator,
        kalmannet_checkpoint=kalmannet_checkpoint,
        kalmannet_device="cpu",
    )
    return AB3DMOTTracker(config)


def benchmark(state_estimator: str, n_tracks: int, iterations: int, kalmannet_checkpoint: str = "") -> dict:
    tracker = build_tracker(state_estimator, kalmannet_checkpoint)
    t = 0.0
    # Birth n_tracks tracks first (not timed).
    t += 0.1
    tracker.step(_make_detections(n_tracks, t), t)

    rss_before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    durations_ms: list[float] = []
    for _ in range(iterations):
        t += 0.1
        start = time.perf_counter()
        tracker.step(_make_detections(n_tracks, t), t)
        durations_ms.append((time.perf_counter() - start) * 1000.0)
    rss_after_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    durations_ms.sort()
    p95_index = min(len(durations_ms) - 1, int(0.95 * len(durations_ms)))
    return {
        "state_estimator": state_estimator,
        "n_tracks": n_tracks,
        "iterations": iterations,
        "mean_ms": statistics.mean(durations_ms),
        "median_ms": statistics.median(durations_ms),
        "p95_ms": durations_ms[p95_index],
        "max_ms": durations_ms[-1],
        "rss_delta_kb": rss_after_kb - rss_before_kb,
        "n_live_tracks_at_end": len(tracker._tracks),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kalmannet_checkpoint", type=Path)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--track-counts", type=int, nargs="+", default=[1, 10, 50, 100])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    results = []
    for n in args.track_counts:
        for label, kwargs in [
            ("linear_kf", {"state_estimator": "linear_kf"}),
            ("kalmannet", {
                "state_estimator": "kalmannet",
                "kalmannet_checkpoint": str(args.kalmannet_checkpoint),
            }),
        ]:
            r = benchmark(n_tracks=n, iterations=args.iterations, **kwargs)
            results.append(r)
            print(
                f"{label:10s} n_tracks={n:4d}  mean={r['mean_ms']:.4f} ms  "
                f"p95={r['p95_ms']:.4f} ms  max={r['max_ms']:.4f} ms  "
                f"rss_delta={r['rss_delta_kb']} kB"
            )

    if args.output:
        args.output.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
