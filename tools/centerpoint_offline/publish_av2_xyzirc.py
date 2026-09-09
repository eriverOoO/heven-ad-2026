#!/usr/bin/env python3
"""Publish one canonical AV2 XYZIRC NPZ without importing pyarrow."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from synthetic_xyzirc import XYZIRC_DTYPE, ros_fields

FIELDS = ("x", "y", "z", "intensity", "return_type", "channel")


def load_npz(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        missing = set(FIELDS) - set(data.files)
        if missing:
            raise ValueError(f"missing canonical arrays: {sorted(missing)}")
        lengths = {len(data[name]) for name in FIELDS}
        if len(lengths) != 1:
            raise ValueError("canonical arrays must have equal length")
        cloud = np.empty(lengths.pop(), dtype=XYZIRC_DTYPE)
        for name in FIELDS:
            cloud[name] = data[name]
        metadata = {
            name: data[name].item()
            for name in ("timestamp_ns", "source_log_id", "mode", "coordinate_frame")
            if name in data.files
        }
    if metadata.get("coordinate_frame") != "av2_egovehicle":
        raise ValueError("NPZ coordinate_frame must be av2_egovehicle")
    return cloud, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path)
    parser.add_argument("--topic", default="/ad/perception/lidar/points_xyzirc")
    parser.add_argument("--frame-id", default="av2_egovehicle")
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")

    import rclpy
    from rclpy.qos import QoSProfile
    from sensor_msgs.msg import PointCloud2

    cloud, metadata = load_npz(args.npz)
    rclpy.init()
    node = rclpy.create_node("centerpoint_av2_xyzirc")
    publisher = node.create_publisher(PointCloud2, args.topic, QoSProfile(depth=1))
    try:
        for _ in range(args.count):
            message = PointCloud2()
            message.header.stamp = node.get_clock().now().to_msg()
            message.header.frame_id = args.frame_id
            message.height = 1
            message.width = len(cloud)
            message.fields = ros_fields()
            message.is_bigendian = False
            message.point_step = XYZIRC_DTYPE.itemsize
            message.row_step = len(cloud) * XYZIRC_DTYPE.itemsize
            message.is_dense = True
            message.data = cloud.tobytes()
            publisher.publish(message)
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    print(
        f"mode={metadata.get('mode')} timestamp_ns={metadata.get('timestamp_ns')} "
        f"points={len(cloud)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
