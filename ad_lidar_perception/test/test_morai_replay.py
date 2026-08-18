import math
import unittest
from types import SimpleNamespace

import numpy as np

from ad_lidar_perception.centerpoint_ros import pointcloud2_to_xyzi
from ad_lidar_perception.morai_replay import xyzi_to_pointcloud2


class Header:
    def __init__(self):
        self.frame_id = ""
        self.stamp = SimpleNamespace(sec=0, nanosec=0)


class PointCloud2Fake:
    def __init__(self):
        self.header = Header()
        self.fields = []
        self.height = 0
        self.width = 0
        self.point_step = 0
        self.row_step = 0
        self.is_bigendian = False
        self.is_dense = False
        self.data = b""


class PointField:
    def __init__(self, *, name, offset, datatype, count):
        self.name = name
        self.offset = offset
        self.datatype = datatype
        self.count = count


MESSAGE_TYPES = {"PointCloud2": PointCloud2Fake, "PointField": PointField}


class MoraiReplayTest(unittest.TestCase):
    def test_round_trips_through_pointcloud2_to_xyzi(self):
        points = np.array([[1.0, 2.0, 3.0, 0.5], [4.0, -5.0, 6.0, 0.9]], dtype=np.float32)
        message = xyzi_to_pointcloud2(
            points, frame_id="lidar_link", stamp_sec=12, stamp_nanosec=34, message_types=MESSAGE_TYPES
        )
        self.assertEqual(message.header.frame_id, "lidar_link")
        self.assertEqual((message.header.stamp.sec, message.header.stamp.nanosec), (12, 34))
        self.assertEqual(message.width, 2)
        self.assertEqual(message.point_step, 16)
        decoded = pointcloud2_to_xyzi(message)
        np.testing.assert_array_equal(decoded, points)

    def test_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            xyzi_to_pointcloud2(
                np.zeros((2, 3)), frame_id="lidar_link", stamp_sec=0, stamp_nanosec=0,
                message_types=MESSAGE_TYPES,
            )

    def test_rejects_nonfinite(self):
        with self.assertRaises(ValueError):
            xyzi_to_pointcloud2(
                np.array([[math.nan, 0.0, 0.0, 0.0]]), frame_id="lidar_link", stamp_sec=0,
                stamp_nanosec=0, message_types=MESSAGE_TYPES,
            )


if __name__ == "__main__":
    unittest.main()
