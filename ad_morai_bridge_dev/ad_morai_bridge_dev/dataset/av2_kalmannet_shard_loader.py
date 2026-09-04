"""Loader for an ``av2_kalmannet_shard_v1`` sharded dataset
(``av2_motion_forecasting_adapter.Av2KalmanNetExporter``'s output).

Every segment stays individually recoverable -- either by id
(``get_segment``) or by streaming iteration, shard by shard
(``iter_segments``, opens each shard NPZ once and yields every segment it
contains, rather than reopening a file per segment).

**Corruption is applied HERE, at load time** -- never baked into the
on-disk shard, which always stores the clean AV2 measurement
(``measurement_data == clean position``, ``measurement_valid`` = source
finiteness). ``SegmentArrays.to_kalmannet_sequence(corruption_config,
global_seed)`` reuses ``av2_motion_forecasting_adapter.apply_corruption``
unchanged, keyed by the segment's own stable ``segment_id`` (a SHA-256
digest of ``f"{seed}:{segment_id}"`` -- never Python's process-randomized
``hash()``), so the same (segment, corruption config, seed) always
reproduces the identical corrupted measurement sequence regardless of
processing order, and two different segments never share an RNG stream.

This module trains nothing and does not modify KalmanNet architecture.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ad_morai_bridge_dev.dataset.av2_motion_forecasting_adapter import (
    CorruptionConfig,
    apply_corruption,
)


@dataclass(frozen=True)
class ShardSegmentRef:
    """One row of ``shard_index.jsonl`` -- enough to locate and slice one
    segment without opening its shard file."""

    segment_id: str
    shard_file: str
    shard_index: int
    offset_start: int
    offset_end: int
    scenario_id: str
    track_id: str
    segment_index: int
    av2_object_type: str
    coarse_object_type: str
    track_category: str
    city_name: str
    origin_x: float
    origin_y: float
    sample_count: int
    coordinate_mode: str
    valid_for_kalmannet_gt: bool


def load_shard_index(output_root: Path) -> list[ShardSegmentRef]:
    output_root = Path(output_root)
    path = output_root / "shard_index.jsonl"
    refs: list[ShardSegmentRef] = []
    for line in path.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        refs.append(
            ShardSegmentRef(
                segment_id=row["segment_id"], shard_file=row["shard_file"],
                shard_index=int(row["shard_index"]), offset_start=int(row["offset_start"]),
                offset_end=int(row["offset_end"]), scenario_id=row["scenario_id"],
                track_id=row["track_id"], segment_index=int(row["segment_index"]),
                av2_object_type=row["av2_object_type"], coarse_object_type=row["coarse_object_type"],
                track_category=row["track_category"], city_name=row["city_name"],
                origin_x=float(row["origin_x"]), origin_y=float(row["origin_y"]),
                sample_count=int(row["sample_count"]), coordinate_mode=row["coordinate_mode"],
                valid_for_kalmannet_gt=bool(row["valid_for_kalmannet_gt"]),
            )
        )
    return refs


@dataclass
class SegmentArrays:
    """One segment's clean, per-frame arrays plus its metadata -- the
    sharded-dataset equivalent of the old single-segment NPZ contents."""

    segment_id: str
    scenario_id: str
    track_id: str
    segment_index: int
    av2_object_type: str
    coarse_object_type: str
    track_category: str
    city_name: str
    origin_x: float
    origin_y: float
    valid_for_kalmannet_gt: bool
    timestamps_ns: np.ndarray       # int64 [T]
    dt_s: np.ndarray                # float64 [T], dt_s[0] = nan
    gt_state: np.ndarray            # float64 [T, 4] = [x, y, vx, vy] (already coordinate-transformed)
    clean_measurement: np.ndarray   # float64 [T, 2] -- always the clean AV2 position
    source_measurement_valid: np.ndarray  # bool [T] -- source finiteness, NOT corruption
    av2_timestep: np.ndarray        # int64 [T]
    av2_observed: np.ndarray        # bool [T]

    @property
    def sample_count(self) -> int:
        return int(self.timestamps_ns.shape[0])

    def to_kalmannet_sequence(
        self, corruption_config: CorruptionConfig | None = None, global_seed: int = 0
    ) -> dict[str, Any]:
        """Applies the load-time corruption transform (default: no
        corruption, i.e. the clean measurement passed straight through)
        and returns the SAME ``{"actor_id","frames","dt","x_true","z_meas"}``
        dict shape ``kalmannet_trajectory_loader.TrajectoryArrays.
        to_kalmannet_sequence()`` already produces --
        ``z_meas[t] = None`` wherever the (possibly corrupted) measurement
        is invalid, feeding the existing, unmodified prediction-only
        KalmanNet behaviour with zero architecture change."""

        # A caller passes either a fully-specified CorruptionConfig (its own
        # `seed` field governs), or leaves it None and relies on
        # `global_seed` alone -- the default CorruptionConfig() has every
        # numeric knob at 0/False, so with no explicit config the clean
        # measurement passes straight through regardless of seed.
        config = corruption_config if corruption_config is not None else CorruptionConfig(seed=global_seed)
        arrays = {
            "clean_position": self.clean_measurement,
            "gt_state": self.gt_state,
        }
        corrupted = apply_corruption(arrays, self.segment_id, config)

        length = self.sample_count
        dt: list[Any] = [None]
        dt.extend(float(v) for v in self.dt_s[1:])
        x_true = [[float(v) for v in row] for row in self.gt_state]
        z_meas: list[Any] = []
        valid = corrupted["measurement_valid"]
        measurement = corrupted["measurement"]
        for k in range(length):
            if bool(valid[k]):
                z_meas.append([float(v) for v in measurement[k]])
            else:
                z_meas.append(None)
        return {
            "actor_id": self.segment_id,
            "frames": list(range(length)),
            "dt": dt,
            "x_true": x_true,
            "z_meas": z_meas,
        }


_ShardCache = dict[str, dict[str, np.ndarray]]


def _load_shard_arrays(output_root: Path, shard_file: str, cache: _ShardCache | None) -> dict[str, np.ndarray]:
    if cache is not None and shard_file in cache:
        return cache[shard_file]
    with np.load(Path(output_root) / shard_file, allow_pickle=False) as data:
        arrays = {name: np.array(data[name]) for name in data.files}
    if cache is not None:
        cache[shard_file] = arrays
    return arrays


def _segment_from_shard_arrays(shard_arrays: dict[str, np.ndarray], row_index: int) -> SegmentArrays:
    lo = int(shard_arrays["segment_offsets"][row_index])
    hi = int(shard_arrays["segment_offsets"][row_index + 1])
    return SegmentArrays(
        segment_id=str(shard_arrays["segment_id"][row_index]),
        scenario_id=str(shard_arrays["scenario_id"][row_index]),
        track_id=str(shard_arrays["track_id"][row_index]),
        segment_index=int(shard_arrays["segment_index"][row_index]),
        av2_object_type=str(shard_arrays["av2_object_type"][row_index]),
        coarse_object_type=str(shard_arrays["coarse_object_type"][row_index]),
        track_category=str(shard_arrays["track_category"][row_index]),
        city_name=str(shard_arrays["city_name"][row_index]),
        origin_x=float(shard_arrays["origin_x"][row_index]),
        origin_y=float(shard_arrays["origin_y"][row_index]),
        valid_for_kalmannet_gt=bool(shard_arrays["valid_for_kalmannet_gt"][row_index]),
        timestamps_ns=shard_arrays["timestamps_ns"][lo:hi],
        dt_s=shard_arrays["dt_s"][lo:hi],
        gt_state=shard_arrays["state_data"][lo:hi],
        clean_measurement=shard_arrays["measurement_data"][lo:hi],
        source_measurement_valid=shard_arrays["measurement_valid"][lo:hi],
        av2_timestep=shard_arrays["av2_timestep"][lo:hi],
        av2_observed=shard_arrays["av2_observed"][lo:hi],
    )


def get_segment(
    output_root: Path, segment_id: str, shard_index: list[ShardSegmentRef] | None = None
) -> SegmentArrays:
    """Recovers exactly one segment by id -- ``for segment in shard: ...``
    equivalent for random access."""
    output_root = Path(output_root)
    refs = shard_index if shard_index is not None else load_shard_index(output_root)
    ref = next((r for r in refs if r.segment_id == segment_id), None)
    if ref is None:
        raise KeyError(f"segment_id {segment_id!r} not found in {output_root}")
    shard_arrays = _load_shard_arrays(output_root, ref.shard_file, cache=None)
    row_index = list(shard_arrays["segment_id"]).index(segment_id)
    return _segment_from_shard_arrays(shard_arrays, row_index)


def iter_segments(
    output_root: Path,
    shard_index: list[ShardSegmentRef] | None = None,
    shard_files: list[str] | None = None,
) -> Iterator[SegmentArrays]:
    """Streams every segment, shard by shard -- opens each shard NPZ
    exactly once (never once per segment), yielding segments in shard
    order then within-shard row order. ``shard_files`` optionally
    restricts iteration to a subset (used by the debug materialization
    helper)."""
    output_root = Path(output_root)
    refs = shard_index if shard_index is not None else load_shard_index(output_root)
    by_shard: dict[str, list[ShardSegmentRef]] = {}
    for ref in refs:
        by_shard.setdefault(ref.shard_file, []).append(ref)

    target_files = shard_files if shard_files is not None else sorted(by_shard)
    for shard_file in target_files:
        shard_arrays = _load_shard_arrays(output_root, shard_file, cache=None)
        segment_ids = list(shard_arrays["segment_id"])
        for row_index in range(len(segment_ids)):
            yield _segment_from_shard_arrays(shard_arrays, row_index)
