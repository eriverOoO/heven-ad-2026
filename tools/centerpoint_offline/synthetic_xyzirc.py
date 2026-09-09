#!/usr/bin/env python3
"""Create a deterministic synthetic cloud matching HEVEN's XYZIRC contract.

This is a runtime smoke-test input only.  It does not represent a MORAI scan
and must never be used for accuracy or tracking evaluation.
"""
from __future__ import annotations

import argparse
import math
from typing import Final

import numpy as np


XYZIRC_DTYPE: Final = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("intensity", "u1"), ("return_type", "u1"), ("channel", "<u2"),
], align=False)


def synthetic_xyzirc(*, rings: int = 16, azimuth_samples: int = 64) -> np.ndarray:
    """Return a deterministic, non-empty cloud with the exact 16-byte layout."""
    if rings <= 0 or azimuth_samples <= 0:
        raise ValueError("rings and azimuth_samples must be positive")
    elevation_deg = np.linspace(-15.0, 15.0, rings, dtype=np.float32)
    azimuth = np.linspace(-math.pi, math.pi, azimuth_samples, endpoint=False, dtype=np.float32)
    channel, azimuth_index = np.meshgrid(np.arange(rings), np.arange(azimuth_samples), indexing="ij")
    elevation = np.deg2rad(elevation_deg[channel])
    az = azimuth[azimuth_index]
    distance = 12.0 + channel.astype(np.float32) * 0.1
    cloud = np.empty(rings * azimuth_samples, dtype=XYZIRC_DTYPE)
    cloud["x"] = (distance * np.cos(elevation) * np.cos(az)).ravel()
    cloud["y"] = (distance * np.cos(elevation) * np.sin(az)).ravel()
    cloud["z"] = (distance * np.sin(elevation)).ravel()
    cloud["intensity"] = (32 + (channel * 7) % 200).astype(np.uint8).ravel()
    cloud["return_type"] = 0
    cloud["channel"] = channel.astype(np.uint16).ravel()
    return cloud


def ros_fields():
    """Construct ROS PointFields lazily, so layout tests do not require ROS."""
    from sensor_msgs.msg import PointField
    return [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.UINT8, count=1),
        PointField(name="return_type", offset=13, datatype=PointField.UINT8, count=1),
        PointField(name="channel", offset=14, datatype=PointField.UINT16, count=1),
    ]


def publish(topic: str, frame_id: str, count: int) -> None:
    """Publish a bounded smoke cloud sequence; imports ROS only at execution."""
    import rclpy
    from rclpy.qos import QoSProfile
    from sensor_msgs.msg import PointCloud2

    rclpy.init()
    node = rclpy.create_node("centerpoint_synthetic_xyzirc")
    publisher = node.create_publisher(PointCloud2, topic, QoSProfile(depth=1))
    cloud = synthetic_xyzirc()
    try:
        for _ in range(count):
            message = PointCloud2()
            message.header.stamp = node.get_clock().now().to_msg()
            message.header.frame_id = frame_id
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/ad/perception/lidar/points_xyzirc")
    parser.add_argument("--frame-id", default="lidar_link")
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    publish(args.topic, args.frame_id, args.count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
