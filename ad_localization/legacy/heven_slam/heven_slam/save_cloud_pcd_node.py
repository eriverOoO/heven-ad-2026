"""Save one ROS PointCloud2 message as a binary XYZ PCD file."""

import os
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


def write_binary_xyz_pcd(points: np.ndarray, output_path: Path) -> int:
    """Write finite XYZ points to a binary PCD file and return their count."""
    xyz = np.asarray(points, dtype=np.float32).reshape((-1, 3))
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    xyz = np.ascontiguousarray(xyz.astype("<f4", copy=False))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {len(xyz)}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(xyz)}\n"
        "DATA binary\n"
    ).encode("ascii")
    with temporary_path.open("wb") as stream:
        stream.write(header)
        stream.write(xyz.tobytes(order="C"))
    os.replace(temporary_path, output_path)
    return len(xyz)


class CloudPcdSaver(Node):
    """Wait for a global cloud, save it once, and stop."""

    def __init__(self) -> None:
        super().__init__("save_cloud_pcd")
        default_directory = Path(
            os.environ.get("HEVEN_MAP_DIR", "~/.ros/heven_maps")
        ).expanduser()
        self.declare_parameter("output_path", str(default_directory / "global_map.pcd"))
        self.declare_parameter("cloud_topic", "/slam/cloud_map")
        self._output_path = Path(
            str(self.get_parameter("output_path").value)
        ).expanduser()
        topic = str(self.get_parameter("cloud_topic").value)
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._saved = False
        self.create_subscription(PointCloud2, topic, self._save, qos)
        self.get_logger().info(f"waiting for a point cloud on {topic}")

    def _save(self, message: PointCloud2) -> None:
        if self._saved:
            return
        structured = point_cloud2.read_points(
            message,
            field_names=["x", "y", "z"],
            skip_nans=True,
        )
        xyz = np.column_stack(
            (structured["x"], structured["y"], structured["z"])
        )
        count = write_binary_xyz_pcd(xyz, self._output_path)
        self._saved = True
        self.get_logger().info(f"saved {count} points to {self._output_path}")
        rclpy.shutdown()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CloudPcdSaver()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
