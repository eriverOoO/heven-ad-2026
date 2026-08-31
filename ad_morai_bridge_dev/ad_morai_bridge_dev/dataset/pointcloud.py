"""Lossless PointCloud2 <-> NPZ conversion.

The canonical dataset preserves every ``PointField`` the sensor publishes
(x/y/z/intensity, and MORAI's ``time`` and any ``ring`` field), each stored
as its own named array with its true dtype. A later exporter may down-cast
to ``[x, y, z, intensity]``; the canonical frame does not throw information
away.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# sensor_msgs/PointField datatype enum -> numpy scalar type.
_PF_DTYPE = {
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,
    7: np.float32,
    8: np.float64,
}
_PF_NAME = {v().dtype.name: k for k, v in _PF_DTYPE.items()}


class PointCloudError(ValueError):
    """Raised for a malformed or unsupported PointCloud2 message."""


def decode_pointcloud(message: Any) -> dict[str, np.ndarray]:
    """Return one flat ``(N,)`` array per PointField, preserving dtype.

    ``count != 1`` fields and big-endian clouds are rejected rather than
    silently reinterpreted.
    """

    if bool(getattr(message, "is_bigendian", False)):
        raise PointCloudError("big-endian PointCloud2 is unsupported")
    fields = list(message.fields)
    if not fields:
        raise PointCloudError("PointCloud2 has no fields")
    names: list[str] = []
    formats: list[str] = []
    offsets: list[int] = []
    for pf in fields:
        if int(pf.count) != 1:
            raise PointCloudError(f"field {pf.name!r} has count {pf.count} != 1")
        scalar = _PF_DTYPE.get(int(pf.datatype))
        if scalar is None:
            raise PointCloudError(f"field {pf.name!r} datatype {pf.datatype} unknown")
        names.append(str(pf.name))
        formats.append(np.dtype(scalar).str)
        offsets.append(int(pf.offset))
    if len(set(names)) != len(names):
        raise PointCloudError("duplicate PointField names")

    point_step = int(message.point_step)
    height = int(message.height)
    width = int(message.width)
    row_step = int(message.row_step)
    if point_step <= 0 or height < 0 or width < 0:
        raise PointCloudError("non-positive point_step or negative dimensions")
    if row_step < width * point_step:
        raise PointCloudError("row_step is smaller than width * point_step")
    if any(off + np.dtype(fmt).itemsize > point_step for off, fmt in zip(offsets, formats)):
        raise PointCloudError("a field extends past point_step")

    struct_dtype = np.dtype(
        {"names": names, "formats": formats, "offsets": offsets, "itemsize": point_step}
    )
    buffer = bytes(message.data)
    expected = row_step * height
    if len(buffer) < expected:
        raise PointCloudError(f"data is {len(buffer)} bytes, expected >= {expected}")
    view = np.ndarray(
        shape=(height, width),
        dtype=struct_dtype,
        buffer=buffer,
        strides=(row_step, point_step),
    )
    return {name: np.ascontiguousarray(view[name].reshape(-1)) for name in names}


def field_schema(message: Any) -> list[dict[str, Any]]:
    """Machine-readable description of the point layout for the manifest."""

    return [
        {
            "name": str(pf.name),
            "datatype": int(pf.datatype),
            "dtype": np.dtype(_PF_DTYPE[int(pf.datatype)]).name,
            "offset": int(pf.offset),
        }
        for pf in message.fields
    ]


def write_frame_npz(path: Path, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Atomically write named point arrays; return their stored schema."""

    if not arrays:
        raise PointCloudError("refusing to write an empty point frame")
    counts = {name: int(arr.shape[0]) for name, arr in arrays.items()}
    if len(set(counts.values())) != 1:
        raise PointCloudError(f"point field length mismatch: {counts}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
    tmp.replace(path)
    return {
        "point_count": next(iter(counts.values())),
        "fields": [
            {"name": name, "dtype": arr.dtype.name} for name, arr in arrays.items()
        ],
    }


def read_frame_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as handle:
        return {name: np.asarray(handle[name]) for name in handle.files}
