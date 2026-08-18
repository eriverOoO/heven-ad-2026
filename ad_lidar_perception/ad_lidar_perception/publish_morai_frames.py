#!/usr/bin/env python3
"""Publish exported MORAI XYZI frames on a PointCloud2 topic for backend comparison.

This lets Euclidean and CenterPoint be launched independently against the
exact same MORAI export frames, without requiring a rosbag. Publish to
``/ad/perception/lidar/cropped`` (CenterPoint's default input) for a
CenterPoint run; for an Euclidean run, launch
``euclidean_clustering.launch.py finite_filter_enabled:=false
finite_input_topic:=<same topic>`` so both detectors read identical data.
Publishing at this stage bypasses self-crop and ground segmentation, so an
Euclidean run will also cluster ground points -- a known, documented
comparison caveat, not a pipeline defect.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField

from ad_lidar_perception.morai_replay import xyzi_to_pointcloud2

MESSAGE_TYPES = {"PointCloud2": PointCloud2, "PointField": PointField}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        help="Explicit sample IDs; default is the first --count samples of --split.",
    )
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--topic", default="/ad/perception/lidar/cropped")
    parser.add_argument("--interval-sec", type=float, default=1.0)
    parser.add_argument(
        "--repeat", type=int, default=1, help="Replay the selected sample sequence this many times."
    )
    args = parser.parse_args(argv)
    if args.count < 1:
        parser.error("--count must be positive")
    if args.interval_sec <= 0:
        parser.error("--interval-sec must be positive")
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    share = Path(get_package_share_directory("ad_lidar_perception"))
    sys.path.insert(0, str(share / "centerpoint_offline"))
    from morai_dataset import MoraiHevenDatasetCore

    dataset_core = MoraiHevenDatasetCore(args.dataset, split=args.split)
    if args.sample_ids:
        missing = [s for s in args.sample_ids if s not in dataset_core.sample_ids]
        if missing:
            raise ValueError(f"unknown sample_id(s) for split {args.split!r}: {missing}")
        indices = [dataset_core.sample_ids.index(s) for s in args.sample_ids]
    else:
        if args.count > len(dataset_core):
            raise ValueError(f"--count {args.count} exceeds split size {len(dataset_core)}")
        indices = list(range(args.count))

    rclpy.init()
    node = Node("ad_publish_morai_frames")
    publisher = node.create_publisher(PointCloud2, args.topic, qos_profile_sensor_data)
    try:
        for repeat_index in range(args.repeat):
            for index in indices:
                sample = dataset_core[index]
                now = node.get_clock().now().to_msg()
                message = xyzi_to_pointcloud2(
                    sample["points"],
                    frame_id="lidar_link",
                    stamp_sec=now.sec,
                    stamp_nanosec=now.nanosec,
                    message_types=MESSAGE_TYPES,
                )
                publisher.publish(message)
                node.get_logger().info(
                    f"published sample_id={sample['sample_id']} points={len(sample['points'])} "
                    f"topic={args.topic} repeat={repeat_index + 1}/{args.repeat}"
                )
                time.sleep(args.interval_sec)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
