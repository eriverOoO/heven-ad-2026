#!/usr/bin/env python3
"""Prepare up-only random controls matched to ring16 radial counts exactly."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from prepare_av2_xyzirc import save
from prepare_sampling_control import UP_LIDAR_MAX, UP_LIDAR_MIN, up_lidar_mask
from publish_av2_xyzirc import load_npz


BIN_WIDTH_M = 5.0
ROI_CORNER_RADIUS_M = float(np.hypot(76.8, 76.8))


def radial_bin_indices(cloud: np.ndarray, width_m: float = BIN_WIDTH_M) -> np.ndarray:
    """5 m bins through the square CenterPoint ROI plus an overflow bin."""
    radius = np.hypot(cloud["x"].astype(np.float64), cloud["y"].astype(np.float64))
    normal_bins = int(np.ceil(ROI_CORNER_RADIUS_M / width_m))
    result = np.floor(radius / width_m).astype(np.int64)
    result[result >= normal_bins] = normal_bins
    return result


def frame_rng(seed: int, timestamp_ns: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence((int(seed), int(timestamp_ns), 0xE5)))


def stratified_sample(up_cloud: np.ndarray, ring_cloud: np.ndarray, seed: int, timestamp_ns: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    up_bins, ring_bins = radial_bin_indices(up_cloud), radial_bin_indices(ring_cloud)
    selected = []
    rng = frame_rng(seed, timestamp_ns)
    for bin_id in range(max(int(up_bins.max(initial=0)), int(ring_bins.max(initial=0))) + 1):
        pool = np.flatnonzero(up_bins == bin_id)
        needed = int(np.count_nonzero(ring_bins == bin_id))
        if len(pool) < needed:
            raise ValueError(f"bin {bin_id}: up pool {len(pool)} < ring target {needed}")
        if needed:
            selected.append(rng.choice(pool, size=needed, replace=False))
    indices = np.sort(np.concatenate(selected) if selected else np.empty(0, dtype=np.int64))
    output = up_cloud[indices].copy()
    if len(output) != len(ring_cloud):
        raise AssertionError("total count mismatch")
    if not np.array_equal(np.bincount(radial_bin_indices(output), minlength=int(ring_bins.max(initial=0)) + 1), np.bincount(ring_bins, minlength=int(ring_bins.max(initial=0)) + 1)):
        raise AssertionError("radial histogram mismatch")
    return output, indices, ring_bins


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--log-id", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(11, 29, 47))
    args = parser.parse_args()
    root = args.derived_root
    paths = sorted((root / "native").glob("*.npz"))
    manifest = {"log_id": args.log_id, "source": "up_lidar laser_number 0..31", "bin_width_m": BIN_WIDTH_M, "roi_corner_radius_m": ROI_CORNER_RADIUS_M, "overflow_bin": True, "gt_used_during_sampling": False, "seeds": args.seeds, "frames": []}
    for path in paths:
        timestamp = int(path.stem)
        native, meta = load_npz(path)
        ring, ring_meta = load_npz(root / "source_ring_vlp16_v2" / path.name)
        if int(meta["timestamp_ns"]) != timestamp or int(ring_meta["timestamp_ns"]) != timestamp:
            raise ValueError(f"timestamp mismatch: {timestamp}")
        up = native[up_lidar_mask(native)].copy()
        if np.any((up["channel"] < UP_LIDAR_MIN) | (up["channel"] >= UP_LIDAR_MAX)):
            raise AssertionError("non-up point entered pool")
        row = {"timestamp_ns": timestamp, "up_points": len(up), "ring_points": len(ring), "ring_radial_counts": np.bincount(radial_bin_indices(ring)).tolist(), "seeds": {}}
        for seed in args.seeds:
            mode = f"up_distance_stratified_seed{seed}"
            output, indices, _ = stratified_sample(up, ring, seed, timestamp)
            save(root / mode / path.name, output, timestamp, args.log_id, mode)
            row["seeds"][str(seed)] = {"output_points": len(output), "radial_counts": np.bincount(radial_bin_indices(output)).tolist(), "without_replacement": len(indices) == len(np.unique(indices)), "index_sha256": hashlib.sha256(indices.astype(np.int64).tobytes()).hexdigest(), "unchanged_source_points": bool(np.array_equal(output, up[indices]))}
        manifest["frames"].append(row)
    out = root / "manifests" / f"{args.log_id}_distance_stratified_control_v1.json"
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(out), "frames": len(paths), "seeds": args.seeds}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
