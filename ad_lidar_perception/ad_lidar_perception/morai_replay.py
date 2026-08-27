"""Pure helper for replaying exported MORAI XYZI frames onto a PointCloud2 topic."""

from __future__ import annotations

from typing import Any

import numpy as np


def xyzi_to_pointcloud2(
    points: np.ndarray,
    *,
    frame_id: str,
    stamp_sec: int,
    stamp_nanosec: int,
    message_types: dict[str, Any],
) -> Any:
    """Encode an (N,4) XYZI array as the exact layout ``pointcloud2_to_xyzi`` decodes."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError("points must be an (N,4) XYZI array")
    if not np.isfinite(points).all():
        raise ValueError("points must be finite")
    encoded = points.astype("<f4", copy=False)

    message = message_types["PointCloud2"]()
    message.header.frame_id = frame_id
    message.header.stamp.sec = int(stamp_sec)
    message.header.stamp.nanosec = int(stamp_nanosec)
    field_type = message_types["PointField"]
    message.fields = [
        field_type(name=name, offset=index * 4, datatype=7, count=1)
        for index, name in enumerate(("x", "y", "z", "intensity"))
    ]
    message.height = 1
    message.width = len(encoded)
    message.point_step = 16
    message.row_step = 16 * len(encoded)
    message.is_bigendian = False
    message.is_dense = True
    message.data = encoded.tobytes()
    return message
