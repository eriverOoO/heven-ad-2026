import math
import struct
from types import SimpleNamespace
import unittest

from ad_lidar_perception.centerpoint_ros import (
    detections_to_message,
    filter_detections,
    pointcloud2_to_xyzi,
    validate_backend,
    yaw_to_quaternion,
)


class Classification:
    UNKNOWN = 0
    CAR = 1
    PEDESTRIAN = 7

    def __init__(self):
        self.label = 0
        self.probability = 0.0


class Kinematics:
    AVAILABLE = 1

    def __init__(self):
        self.pose_with_covariance = SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=0.0),
            )
        )
        self.orientation_availability = 0
        self.has_twist = True
        self.has_twist_covariance = True


class Shape:
    BOUNDING_BOX = 1


class Object:
    def __init__(self):
        self.existence_probability = 0.0
        self.classification = []
        self.kinematics = Kinematics()
        self.shape = SimpleNamespace(
            type=0, dimensions=SimpleNamespace(x=0.0, y=0.0, z=0.0)
        )


class Objects:
    def __init__(self):
        self.header = None
        self.objects = []


TYPES = {
    "DetectedObject": Object,
    "DetectedObjectKinematics": Kinematics,
    "DetectedObjects": Objects,
    "ObjectClassification": Classification,
    "Shape": Shape,
}


def cloud(points):
    fields = [
        SimpleNamespace(name=name, offset=index * 4, datatype=7, count=1)
        for index, name in enumerate(("x", "y", "z", "intensity"))
    ]
    header = SimpleNamespace(stamp=SimpleNamespace(sec=3, nanosec=4), frame_id="lidar_link")
    return SimpleNamespace(
        header=header,
        fields=fields,
        height=1,
        width=len(points),
        point_step=16,
        row_step=16 * len(points),
        is_bigendian=False,
        data=b"".join(struct.pack("<ffff", *point) for point in points),
    )


class CenterPointRosContractTest(unittest.TestCase):
    def test_pointcloud2_conversion_preserves_exact_xyzi_order(self):
        message = cloud([(1.0, 2.0, 3.0, 0.5), (math.nan, 0.0, 0.0, 1.0)])
        self.assertEqual(pointcloud2_to_xyzi(message).tolist(), [[1.0, 2.0, 3.0, 0.5]])

    def test_header_yaw_box_score_and_class_mapping(self):
        header = cloud([]).header
        detections = [
            {"class_name": "vehicle", "score": 0.9, "box_lidar": [1, 2, 3, 4, 5, 6, math.pi / 2]},
            {"class_name": "pedestrian", "score": 0.8, "box_lidar": [0, 0, 0, 1, 1, 2, 0]},
            {"class_name": "obstacle", "score": 0.7, "box_lidar": [0, 0, 0, 2, 3, 4, 0]},
        ]
        output = detections_to_message(header=header, detections=detections, message_types=TYPES)
        self.assertIs(output.header, header)
        self.assertEqual((output.header.stamp.sec, output.header.frame_id), (3, "lidar_link"))
        self.assertEqual([item.classification[0].label for item in output.objects], [1, 7, 0])
        self.assertAlmostEqual(output.objects[0].existence_probability, 0.9)
        self.assertEqual((output.objects[0].shape.dimensions.x, output.objects[0].shape.dimensions.y, output.objects[0].shape.dimensions.z), (4, 5, 6))
        quaternion = output.objects[0].kinematics.pose_with_covariance.pose.orientation
        self.assertAlmostEqual(quaternion.z, math.sqrt(0.5))
        self.assertAlmostEqual(quaternion.w, math.sqrt(0.5))
        self.assertFalse(output.objects[0].kinematics.has_twist)

    def test_empty_detections_and_max_filtering(self):
        self.assertEqual(detections_to_message(header=cloud([]).header, detections=[], message_types=TYPES).objects, [])
        source = [
            {"score": 0.4, "class_name": "vehicle"},
            {"score": 0.9, "class_name": "obstacle"},
            {"score": 0.8, "class_name": "pedestrian"},
        ]
        self.assertEqual([item["score"] for item in filter_detections(source, 0.5, 2)], [0.9, 0.8])

    def test_yaw_rejects_nonfinite(self):
        with self.assertRaises(ValueError):
            yaw_to_quaternion(math.nan)

    def test_backend_switch_contract(self):
        self.assertEqual(validate_backend("euclidean"), "euclidean")
        self.assertEqual(validate_backend("centerpoint"), "centerpoint")
        with self.assertRaises(ValueError):
            validate_backend("unknown")


if __name__ == "__main__":
    unittest.main()
