#!/usr/bin/env python3
"""Prepare up-LiDAR and deterministic count-matched sampling control NPZs.

The source ``native`` cloud is the AV2 aggregate.  This tool creates only
derived files: up_lidar_full uses channels 0..31, and each random control is a
without-replacement subset of that up-only pool whose count exactly matches
the existing one-source 16-ring proxy for the same timestamp.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from prepare_av2_xyzirc import save
from publish_av2_xyzirc import load_npz


UP_LIDAR_MIN = 0
UP_LIDAR_MAX = 32


def up_lidar_mask(cloud: np.ndarray) -> np.ndarray:
    """Return the verified AV2 up_lidar channel 0..31 selection mask."""
    return (cloud["channel"] >= UP_LIDAR_MIN) & (cloud["channel"] < UP_LIDAR_MAX)


def timestamp_rng(seed: int, timestamp_ns: int) -> np.random.Generator:
    """Make frame selection reproducible without coupling all frames' draws."""
    return np.random.default_rng(np.random.SeedSequence((int(seed), int(timestamp_ns))))


def sample_up_lidar(up_cloud: np.ndarray, target_count: int, seed: int, timestamp_ns: int) -> tuple[np.ndarray, np.ndarray]:
    if target_count < 0 or target_count > len(up_cloud):
        raise ValueError(f"target_count {target_count} is outside 0..{len(up_cloud)}")
    indices = timestamp_rng(seed, timestamp_ns).choice(len(up_cloud), size=target_count, replace=False)
    # Sorting avoids an artificial random ordering confound; membership remains random.
    indices.sort()
    return up_cloud[indices].copy(), indices


def point_digest(cloud: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(cloud).tobytes()).hexdigest()


def validate_sample(up_cloud: np.ndarray, sampled: np.ndarray, indices: np.ndarray, target_count: int) -> dict:
    if len(sampled) != target_count or len(indices) != target_count:
        raise ValueError("sample size does not equal target count")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("random sample contains duplicate source indices")
    if np.any(~up_lidar_mask(sampled)):
        raise ValueError("random control contains non-up-lidar point")
    if not np.array_equal(sampled, up_cloud[indices]):
        raise ValueError("sampled source coordinates or attributes changed")
    return {
        "exact_target_count": True,
        "without_replacement": True,
        "up_lidar_only": True,
        "coordinates_and_attributes_unchanged": True,
        "source_index_sha256": hashlib.sha256(indices.astype(np.int64).tobytes()).hexdigest(),
        "cloud_sha256": point_digest(sampled),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(11, 29, 47))
    parser.add_argument("--log-id", required=True)
    args = parser.parse_args()
    root = args.derived_root
    native_paths = sorted((root / "native").glob("*.npz"))
    if not native_paths:
        raise ValueError(f"no native NPZ files below {root}")
    proxy_root = root / "source_ring_vlp16_v2"
    if not proxy_root.is_dir():
        raise ValueError(f"missing corrected source-ring root: {proxy_root}")

    manifest = {"log_id": args.log_id, "up_lidar_channels": [0, 31], "seeds": args.seeds, "frames": []}
    for native_path in native_paths:
        timestamp_ns = int(native_path.stem)
        native, metadata = load_npz(native_path)
        proxy, proxy_metadata = load_npz(proxy_root / native_path.name)
        if int(metadata["timestamp_ns"]) != timestamp_ns or int(proxy_metadata["timestamp_ns"]) != timestamp_ns:
            raise ValueError(f"timestamp metadata mismatch: {timestamp_ns}")
        up = native[up_lidar_mask(native)].copy()
        if not len(up):
            raise ValueError(f"no up_lidar points: {timestamp_ns}")
        save(root / "up_lidar_full" / native_path.name, up, timestamp_ns, args.log_id, "up_lidar_full")
        frame = {
            "timestamp_ns": timestamp_ns,
            "native_points": len(native),
            "up_lidar_points": len(up),
            "ring16_points": len(proxy),
            "random": {},
        }
        for seed in args.seeds:
            mode = f"up_random_matched_seed{seed}"
            sampled, indices = sample_up_lidar(up, len(proxy), seed, timestamp_ns)
            frame["random"][str(seed)] = validate_sample(up, sampled, indices, len(proxy))
            save(root / mode / native_path.name, sampled, timestamp_ns, args.log_id, mode)
        manifest["frames"].append(frame)
    path = root / "manifests" / f"{args.log_id}_sampling_control_v1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(path), "frames": len(native_paths), "seeds": args.seeds}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
