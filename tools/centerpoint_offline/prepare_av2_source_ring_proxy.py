#!/usr/bin/env python3
"""Prepare a motion-aware, one-source AV2 16-ring proxy and geometry manifest."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from av2_source_ring_adapter import (
    SOURCE_LIDAR_RANGES,
    adapt_source_rings,
    elevation_degrees,
    interpolate_ego_poses,
    inverse_transform_points,
    quaternion_rotation_matrix,
    reconstruct_source_local_points,
    select_unique_monotonic_rings,
    summarize_ring_geometry,
)
from prepare_av2_xyzirc import native_cloud, save


MODE = "source_ring_vlp16_v2"


def pose_arrays(table) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    columns = table.to_pydict()
    timestamps = np.asarray(columns["timestamp_ns"], dtype=np.int64)
    quaternions = np.column_stack(
        [np.asarray(columns[name], dtype=np.float64) for name in ("qw", "qx", "qy", "qz")]
    )
    translations = np.column_stack(
        [np.asarray(columns[name], dtype=np.float64) for name in ("tx_m", "ty_m", "tz_m")]
    )
    order = np.argsort(timestamps)
    return timestamps[order], quaternions[order], translations[order]


def calibration_by_name(table) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    columns = table.to_pydict()
    result = {}
    for index, name in enumerate(columns["sensor_name"]):
        if name not in SOURCE_LIDAR_RANGES:
            continue
        rotation = quaternion_rotation_matrix(
            *[float(columns[field][index]) for field in ("qw", "qx", "qy", "qz")]
        )
        translation = np.asarray(
            [float(columns[field][index]) for field in ("tx_m", "ty_m", "tz_m")]
        )
        result[name] = rotation, translation
    missing = set(SOURCE_LIDAR_RANGES) - set(result)
    if missing:
        raise ValueError(f"missing source LiDAR calibration: {sorted(missing)}")
    return result


def distribution(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0, "median": None, "mad": None, "p05": None, "p95": None}
    median = float(np.median(values))
    return {
        "count": len(values),
        "median": median,
        "mad": float(np.median(np.abs(values - median))),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def collect_geometry(lidar_paths: list[Path], log_dir: Path) -> tuple[dict, dict]:
    import pyarrow.feather as feather

    pose_timestamps, pose_quaternions, pose_translations = pose_arrays(
        feather.read_table(log_dir / "city_SE3_egovehicle.feather")
    )
    calibrations = calibration_by_name(
        feather.read_table(log_dir / "calibration" / "egovehicle_SE3_sensor.feather")
    )
    approximate: dict[int, list[np.ndarray]] = defaultdict(list)
    reconstructed: dict[int, list[np.ndarray]] = defaultdict(list)

    for lidar_path in lidar_paths:
        timestamp_ns = int(lidar_path.stem)
        table = feather.read_table(
            lidar_path, columns=["x", "y", "z", "laser_number", "offset_ns"]
        )
        xyz = np.column_stack(
            [table[name].to_numpy().astype(np.float64, copy=False) for name in ("x", "y", "z")]
        )
        laser_number = table["laser_number"].to_numpy().astype(np.int64, copy=False)
        offset_ns = table["offset_ns"].to_numpy().astype(np.int64, copy=False)
        ref_rotation, ref_translation = interpolate_ego_poses(
            pose_timestamps,
            pose_quaternions,
            pose_translations,
            np.asarray([timestamp_ns], dtype=np.int64),
        )
        for source_name, (start, stop) in SOURCE_LIDAR_RANGES.items():
            mask = (laser_number >= start) & (laser_number < stop)
            source_rotation, source_translation = calibrations[source_name]
            method_a = inverse_transform_points(xyz[mask], source_rotation, source_translation)
            method_b = reconstruct_source_local_points(
                xyz[mask],
                offset_ns[mask],
                timestamp_ns,
                ref_rotation[0],
                ref_translation[0],
                pose_timestamps,
                pose_quaternions,
                pose_translations,
                source_rotation,
                source_translation,
            )
            method_a_elevation = elevation_degrees(method_a)
            method_b_elevation = elevation_degrees(method_b)
            source_lasers = laser_number[mask]
            for value in range(start, stop):
                ring_mask = source_lasers == value
                approximate[value].append(method_a_elevation[ring_mask])
                reconstructed[value].append(method_b_elevation[ring_mask])

    approximate_all = {key: np.concatenate(value) for key, value in approximate.items()}
    reconstructed_all = {key: np.concatenate(value) for key, value in reconstructed.items()}
    geometry = {
        source_name: summarize_ring_geometry(reconstructed_all, source_name)
        for source_name in SOURCE_LIDAR_RANGES
    }
    diagnostics = {
        "method_a_reference_time_static_inverse": {
            str(key): distribution(value) for key, value in sorted(approximate_all.items())
        },
        "method_b_acquisition_time_reconstruction": {
            str(key): distribution(value) for key, value in sorted(reconstructed_all.items())
        },
    }
    return geometry, diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-lidar", choices=tuple(SOURCE_LIDAR_RANGES), default="up_lidar")
    parser.add_argument("--log-id")
    args = parser.parse_args()

    import pyarrow.feather as feather

    lidar_paths = sorted((args.log_dir / "sensors" / "lidar").glob("*.feather"))
    if not lidar_paths:
        raise ValueError(f"no LiDAR sweeps found below {args.log_dir}")
    log_id = args.log_id or args.log_dir.name
    geometry, diagnostics = collect_geometry(lidar_paths, args.log_dir)
    selection = select_unique_monotonic_rings(geometry[args.source_lidar])

    per_frame = []
    for lidar_path in lidar_paths:
        table = feather.read_table(
            lidar_path, columns=["x", "y", "z", "intensity", "laser_number"]
        )
        native = native_cloud(table)
        laser_numbers = table["laser_number"].to_numpy().astype(np.int64, copy=False)
        proxy = adapt_source_rings(
            native, laser_numbers, selection, source_lidar=args.source_lidar
        )
        timestamp_ns = int(lidar_path.stem)
        save(
            args.output_root / MODE / f"{timestamp_ns}.npz",
            proxy,
            timestamp_ns,
            log_id,
            MODE,
        )
        channel_counts = np.bincount(proxy["channel"], minlength=16)
        per_frame.append(
            {
                "timestamp_ns": timestamp_ns,
                "native_points": len(native),
                "proxy_points": len(proxy),
                "retention": len(proxy) / len(native) if len(native) else 0.0,
                "occupied_beams": int(np.count_nonzero(channel_counts)),
                "points_per_channel": channel_counts.tolist(),
            }
        )

    up = geometry["up_lidar"]
    down = geometry["down_lidar"]
    pair_differences = [
        abs(up[index].median_elevation_deg - down[index].median_elevation_deg)
        for index in range(32)
    ]
    errors = [row.angular_error_deg for row in selection]
    manifest = {
        "mode": MODE,
        "source_log_id": log_id,
        "source_lidar": args.source_lidar,
        "source_mapping": {
            "up_lidar": "laser_number 0..31; local_ring_id = laser_number",
            "down_lidar": "laser_number 32..63; local_ring_id = laser_number - 32",
            "evidence": (
                "AV2 aggregate schema plus acquisition-time source-local ring bands and "
                "matching up/down local-ring elevations"
            ),
        },
        "motion_compensation": {
            "offset_semantics": "acquisition timestamp = sweep filename timestamp + offset_ns",
            "geometry_method": (
                "city_SE3_egovehicle(reference) then inverse city_SE3_egovehicle(acquisition), "
                "then inverse egovehicle_SE3_sensor"
            ),
            "output_coordinates": "original motion-compensated AV2 egovehicle XYZ",
        },
        "ring_geometry": {
            name: [asdict(row) for row in rows] for name, rows in geometry.items()
        },
        "reference_vs_acquisition_diagnostics": diagnostics,
        "paired_up_down_ring_median_difference_deg": {
            "mean": float(np.mean(pair_differences)),
            "max": float(np.max(pair_differences)),
        },
        "selection": [asdict(row) for row in selection],
        "angular_mismatch_deg": {
            "mean": float(np.mean(errors)),
            "max": float(np.max(errors)),
        },
        "horizontal_reduction": "none",
        "channel_semantics": "0..15 sorted by target elevation -15..+15 degrees",
        "frames": per_frame,
    }
    manifest_path = args.output_root / "manifests" / f"{log_id}_{MODE}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_path), **manifest["angular_mismatch_deg"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
