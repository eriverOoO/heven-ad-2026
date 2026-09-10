#!/usr/bin/env python3
"""Legacy ego-elevation structured-sparsity adapter.

The input is an ``(N, 4)`` float array (x, y, z, intensity) in one LiDAR
frame.  This deliberately does not implement an AV2 dataset reader: local AV2
storage currently contains motion-forecasting assets, not a verified LiDAR
sensor corpus.  The caller owns frame normalization and provenance.

This preserved ``ego_elevation_stress_v1`` implementation measures elevation
about the AV2 rear-axle ego origin.  It is deterministic and useful as a
structured stress-test, but it is not a physical VLP-16 proxy.  New physical
ring selection belongs in :mod:`av2_source_ring_adapter`.  Original AV2 files
are never written.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from synthetic_xyzirc import XYZIRC_DTYPE


# Nominal, interleaved VLP-16 elevations.  They remain an explicit assumption
# until calibrated against a recorded MORAI cloud/configuration.
NOMINAL_VLP16_ELEVATIONS_DEG = np.asarray(
    [-15.0, 1.0, -13.0, 3.0, -11.0, 5.0, -9.0, 7.0,
     -7.0, 9.0, -5.0, 11.0, -3.0, 13.0, -1.0, 15.0], dtype=np.float32)


@dataclass(frozen=True)
class AdapterConfig:
    min_range_m: float = 0.5
    max_range_m: float = 80.0
    vertical_tolerance_deg: float = 0.7
    azimuth_bin_deg: float = 0.2
    intensity_mode: Literal["preserve", "normalize", "clip", "zero", "constant"] = "preserve"
    constant_intensity: int = 0
    return_type: int = 0
    target_elevations_deg: tuple[float, ...] = tuple(float(value) for value in NOMINAL_VLP16_ELEVATIONS_DEG)

    def validate(self) -> None:
        if not (0.0 <= self.min_range_m < self.max_range_m):
            raise ValueError("require 0 <= min_range_m < max_range_m")
        if self.vertical_tolerance_deg <= 0.0:
            raise ValueError("vertical_tolerance_deg must be positive")
        if not (0.0 < self.azimuth_bin_deg <= 360.0):
            raise ValueError("azimuth_bin_deg must be in (0, 360]")
        if len(self.target_elevations_deg) != 16:
            raise ValueError("a VLP-16-like target requires exactly 16 elevations")
        if not all(np.isfinite(self.target_elevations_deg)):
            raise ValueError("target elevations must be finite")
        if not (0 <= self.constant_intensity <= 255 and 0 <= self.return_type <= 255):
            raise ValueError("constant intensity and return_type must fit uint8")


def _validate_input(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError("points must have shape (N, 4): x, y, z, intensity")
    if not np.issubdtype(points.dtype, np.number):
        raise ValueError("points must be numeric")
    if not np.isfinite(points[:, :3]).all():
        raise ValueError("coordinates must be finite")
    return points.astype(np.float32, copy=False)


def _convert_intensity(values: np.ndarray, mode: str, constant: int) -> np.ndarray:
    finite = np.nan_to_num(values, nan=0.0, posinf=255.0, neginf=0.0)
    if mode == "zero":
        result = np.zeros(len(values), dtype=np.uint8)
    elif mode == "constant":
        result = np.full(len(values), constant, dtype=np.uint8)
    elif mode == "normalize":
        low, high = np.percentile(finite, [1.0, 99.0]) if len(values) else (0.0, 1.0)
        result = np.zeros(len(values), dtype=np.uint8) if high <= low else np.rint(np.clip((finite - low) * 255.0 / (high - low), 0, 255)).astype(np.uint8)
    elif mode in ("preserve", "clip"):
        result = np.rint(np.clip(finite, 0.0, 255.0)).astype(np.uint8)
    else:
        raise ValueError(f"unsupported intensity mode {mode!r}")
    return result


def adapt_points(points: np.ndarray, config: AdapterConfig = AdapterConfig()) -> np.ndarray:
    """Map a cloud to packed XYZIRC using range, beam and azimuth constraints."""
    config.validate()
    source = _validate_input(points)
    if len(source) == 0:
        return np.empty(0, dtype=XYZIRC_DTYPE)
    xyz = source[:, :3]
    horizontal_range = np.hypot(xyz[:, 0], xyz[:, 1])
    ranges = np.linalg.norm(xyz, axis=1)
    elevations = np.degrees(np.arctan2(xyz[:, 2], horizontal_range))
    target = np.asarray(config.target_elevations_deg, dtype=np.float32)
    beam = np.abs(elevations[:, None] - target[None, :]).argmin(axis=1)
    beam_delta = np.abs(elevations - target[beam])
    keep = ((ranges >= config.min_range_m) & (ranges <= config.max_range_m) &
            (beam_delta <= config.vertical_tolerance_deg))
    if not np.any(keep):
        return np.empty(0, dtype=XYZIRC_DTYPE)
    chosen = source[keep]
    chosen_ranges = ranges[keep]
    chosen_beam = beam[keep]
    azimuth = (np.degrees(np.arctan2(chosen[:, 1], chosen[:, 0])) + 360.0) % 360.0
    bin_count = int(round(360.0 / config.azimuth_bin_deg))
    azimuth_bin = np.floor(azimuth / config.azimuth_bin_deg).astype(np.int64) % bin_count
    # Stable sorting guarantees ties keep the earliest source point.  One ray
    # per (assigned beam, azimuth bin), preferring nearest valid return.
    order = np.lexsort((np.arange(len(chosen)), chosen_ranges, azimuth_bin, chosen_beam))
    keyed = chosen_beam[order].astype(np.int64) * bin_count + azimuth_bin[order]
    selected = order[np.r_[True, keyed[1:] != keyed[:-1]]]
    output = np.empty(len(selected), dtype=XYZIRC_DTYPE)
    output["x"], output["y"], output["z"] = (chosen[selected, axis] for axis in range(3))
    output["intensity"] = _convert_intensity(chosen[selected, 3], config.intensity_mode, config.constant_intensity)
    output["return_type"] = config.return_type
    output["channel"] = chosen_beam[selected].astype(np.uint16)
    return output


def cloud_statistics(cloud: np.ndarray) -> dict[str, object]:
    """Return small JSON-safe diagnostics for one packed XYZIRC cloud."""
    if cloud.dtype != XYZIRC_DTYPE:
        raise ValueError("cloud must use XYZIRC_DTYPE")
    ranges = np.sqrt(cloud["x"] ** 2 + cloud["y"] ** 2 + cloud["z"] ** 2)
    bins = ((0.0, 20.0), (20.0, 40.0), (40.0, 60.0), (60.0, 80.0))
    return {
        "points": int(len(cloud)), "channels_occupied": int(len(np.unique(cloud["channel"]))),
        "range_bins": {f"{low:g}-{high:g}m": int(np.count_nonzero((ranges >= low) & (ranges < high))) for low, high in bins},
        "mean_intensity": float(cloud["intensity"].mean()) if len(cloud) else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="input .npy shaped (N,4), supplied by a verified AV2 reader")
    parser.add_argument("output", type=Path, help="output .npy containing packed XYZIRC records")
    parser.add_argument("--max-range-m", type=float, default=80.0)
    parser.add_argument("--vertical-tolerance-deg", type=float, default=0.7)
    parser.add_argument("--azimuth-bin-deg", type=float, default=0.2)
    parser.add_argument("--intensity-mode", choices=("preserve", "normalize", "clip", "zero", "constant"), default="preserve")
    args = parser.parse_args()
    cloud = adapt_points(np.load(args.input), AdapterConfig(max_range_m=args.max_range_m, vertical_tolerance_deg=args.vertical_tolerance_deg, azimuth_bin_deg=args.azimuth_bin_deg, intensity_mode=args.intensity_mode))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, cloud)
    print(cloud_statistics(cloud))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
