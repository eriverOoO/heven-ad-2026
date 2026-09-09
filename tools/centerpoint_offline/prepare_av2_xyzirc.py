#!/usr/bin/env python3
"""Convert AV2 Sensor feather sweeps to native and VLP-16-like XYZIRC NPZ."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from av2_vlp16_adapter import adapt_points
from synthetic_xyzirc import XYZIRC_DTYPE


def native_cloud(table) -> np.ndarray:
    cloud = np.empty(table.num_rows, dtype=XYZIRC_DTYPE)
    for name in ("x", "y", "z"):
        cloud[name] = table[name].to_numpy().astype(np.float32, copy=False)
    cloud["intensity"] = table["intensity"].to_numpy().astype(np.uint8, copy=False)
    cloud["return_type"] = 0
    cloud["channel"] = table["laser_number"].to_numpy().astype(np.uint16, copy=False)
    return cloud


def save(path: Path, cloud: np.ndarray, timestamp_ns: int, log_id: str, mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{name: cloud[name] for name in cloud.dtype.names},
        timestamp_ns=np.int64(timestamp_ns),
        source_log_id=np.str_(log_id),
        mode=np.str_(mode),
        coordinate_frame=np.str_("av2_egovehicle"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feather", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--log-id", required=True)
    args = parser.parse_args()

    import pyarrow.feather as feather

    table = feather.read_table(args.feather, columns=["x", "y", "z", "intensity", "laser_number"])
    native = native_cloud(table)
    points = np.column_stack(
        (native["x"], native["y"], native["z"], native["intensity"])
    ).astype(np.float32)
    adapted = adapt_points(points)
    timestamp_ns = int(args.feather.stem)
    save(args.output_root / "native" / f"{timestamp_ns}.npz", native, timestamp_ns, args.log_id, "native")
    save(args.output_root / "vlp16_like" / f"{timestamp_ns}.npz", adapted, timestamp_ns, args.log_id, "vlp16_like")
    retention = len(adapted) / len(native) if len(native) else 0.0
    print(
        f"timestamp_ns={timestamp_ns} native={len(native)} "
        f"vlp16_like={len(adapted)} retention={retention:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
