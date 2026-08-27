"""POST-FREEZE EXTENSION 2, Stage 1: tests for the pure message<->core adapter
helpers in camera_lidar_fusion_ros.py, using lightweight fake message objects
(no ROS environment needed), same style as the fusion-core tests."""
import types
import unittest

from ad_lidar_perception.camera_lidar_fusion_core import (
    CameraDetection,
    FusedObject,
    FusionResult,
    LidarDetection,
    ProjectedBox,
)
from ad_lidar_perception.camera_lidar_fusion_ros import (
    SEMANTIC_TO_AUTOWARE_LABEL,
    detected_objects_to_lidar_detections,
    detection2d_array_to_camera_detections,
    fused_result_to_detected_objects,
)


def ns(**kw):
    return types.SimpleNamespace(**kw)


def _header(sec=1, nanosec=0, frame_id="lidar_link"):
    return ns(stamp=ns(sec=sec, nanosec=nanosec), frame_id=frame_id)


def _detected_object(x, y, z, l, w, h, qz=0.0, qw=1.0, label=0, prob=1.0, shape_type=0):
    return ns(
        shape=ns(type=shape_type, dimensions=ns(x=l, y=w, z=h)),
        kinematics=ns(pose_with_covariance=ns(pose=ns(
            position=ns(x=x, y=y, z=z),
            orientation=ns(x=0.0, y=0.0, z=qz, w=qw),
        ))),
        classification=[ns(label=label, probability=prob)],
    )


class LidarAdapterTest(unittest.TestCase):
    def test_bounding_box_objects_convert_with_geometry_preserved(self):
        msg = ns(header=_header(), objects=[
            _detected_object(3.0, -1.0, 0.2, 4.5, 2.0, 1.6),
            _detected_object(10.0, 0.0, 0.0, 0.1, 0.1, 0.1),  # clutter, kept
        ])
        dets = detected_objects_to_lidar_detections(msg)
        self.assertEqual(len(dets), 2)
        self.assertEqual(dets[0].length, 4.5)
        self.assertEqual(dets[0].source_index, 0)
        self.assertTrue(dets[1].is_degenerate_geometry)

    def test_non_bounding_box_shapes_are_skipped(self):
        msg = ns(header=_header(), objects=[_detected_object(3.0, 0.0, 0.0, 4.0, 2.0, 1.6, shape_type=1)])
        self.assertEqual(detected_objects_to_lidar_detections(msg), [])

    def test_non_finite_geometry_is_skipped(self):
        msg = ns(header=_header(), objects=[_detected_object(float("nan"), 0.0, 0.0, 4.0, 2.0, 1.6)])
        self.assertEqual(detected_objects_to_lidar_detections(msg), [])

    def test_yaw_recovered_from_quaternion(self):
        import math
        yaw = 0.7
        msg = ns(header=_header(), objects=[
            _detected_object(3.0, 0.0, 0.0, 4.0, 2.0, 1.6, qz=math.sin(yaw / 2), qw=math.cos(yaw / 2)),
        ])
        self.assertAlmostEqual(detected_objects_to_lidar_detections(msg)[0].yaw, yaw, places=6)


class CameraAdapterTest(unittest.TestCase):
    def _det2d(self, cx, cy, sx, sy, class_id, score):
        return ns(bbox=ns(center=ns(position=ns(x=cx, y=cy)), size_x=sx, size_y=sy),
                  results=[ns(hypothesis=ns(class_id=class_id, score=score))])

    def test_vehicle_classes_convert_and_center_size_to_corners(self):
        msg = ns(header=_header(frame_id="camera_front_optical_frame"), detections=[
            self._det2d(100.0, 50.0, 40.0, 20.0, "car", 0.8),
        ])
        dets = detection2d_array_to_camera_detections(msg)
        self.assertEqual(len(dets), 1)
        d = dets[0]
        self.assertEqual(d.semantic_class, "car")
        self.assertEqual(d.box, (80.0, 40.0, 120.0, 60.0))
        self.assertAlmostEqual(d.confidence, 0.8)

    def test_non_traffic_classes_are_dropped(self):
        msg = ns(header=_header(), detections=[
            self._det2d(1, 1, 2, 2, "train", 0.9),
            self._det2d(1, 1, 2, 2, "sports ball", 0.9),
            self._det2d(1, 1, 2, 2, "person", 0.5),
        ])
        dets = detection2d_array_to_camera_detections(msg)
        self.assertEqual([d.semantic_class for d in dets], ["person"])

    def test_detection_with_no_results_is_skipped(self):
        msg = ns(header=_header(), detections=[ns(bbox=ns(center=ns(position=ns(x=1, y=1)), size_x=2, size_y=2), results=[])])
        self.assertEqual(detection2d_array_to_camera_detections(msg), [])


class FusedOutputTest(unittest.TestCase):
    def _fake_msg_classes(self):
        DetectedObjects = lambda: ns(header=None, objects=[])  # noqa: E731
        DetectedObject = lambda: ns(  # noqa: E731
            shape=ns(type=None, dimensions=ns(x=0.0, y=0.0, z=0.0)),
            kinematics=ns(pose_with_covariance=ns(pose=ns(position=ns(x=0.0, y=0.0, z=0.0)))),
            classification=[], existence_probability=0.0,
        )
        ObjectClassification = lambda: ns(label=0, probability=0.0)  # noqa: E731
        Shape = lambda: ns(type=0)  # noqa: E731
        return DetectedObjects, DetectedObject, ObjectClassification, Shape

    def _result(self):
        lidar_f = LidarDetection(x=20.0, y=0.0, z=0.0, length=4.3, width=2.0, height=1.6, score=1.0)
        lidar_o = LidarDetection(x=15.0, y=0.0, z=0.0, length=0.1, width=0.1, height=0.1, score=1.0)
        pb = ProjectedBox(10, 10, 50, 50, 8, 20.0, True, True, False)
        cam = CameraDetection(10, 10, 50, 50, confidence=0.7, semantic_class="car", raw_class_name="car")
        fused = FusedObject(lidar=lidar_f, projected=pb, camera=cam, match_state="fused",
                            association_measure="iou", association_score=0.6, runner_up_score=0.0,
                            semantic_class="car", semantic_confidence=0.7, visual_confirmed=True)
        lonly = FusedObject(lidar=lidar_o, projected=pb, camera=None, match_state="lidar_only",
                            association_measure="iou", association_score=0.0, runner_up_score=0.0,
                            semantic_class="unknown", semantic_confidence=0.0, visual_confirmed=False)
        return FusionResult(fused=[fused], lidar_only=[lonly], camera_only=[], stats={})

    def test_fused_object_carries_camera_class_and_preserves_lidar_geometry(self):
        DObjs, DObj, OClass, Shape = self._fake_msg_classes()
        out = fused_result_to_detected_objects(
            self._result(), _header(),
            detected_objects_cls=DObjs, detected_object_cls=DObj,
            object_classification_cls=OClass, shape_cls=Shape, include_lidar_only=True,
        )
        self.assertEqual(len(out.objects), 2)
        fused_obj = out.objects[0]
        self.assertEqual(fused_obj.classification[0].label, SEMANTIC_TO_AUTOWARE_LABEL["car"])
        self.assertAlmostEqual(fused_obj.classification[0].probability, 0.7)
        self.assertEqual(fused_obj.shape.dimensions.x, 4.3)  # LiDAR geometry preserved
        self.assertAlmostEqual(fused_obj.existence_probability, 0.5 * 1.0 + 0.5 * 0.7)
        # lidar-only object -> UNKNOWN label
        self.assertEqual(out.objects[1].classification[0].label, SEMANTIC_TO_AUTOWARE_LABEL["unknown"])

    def test_include_lidar_only_false_emits_only_fused(self):
        DObjs, DObj, OClass, Shape = self._fake_msg_classes()
        out = fused_result_to_detected_objects(
            self._result(), _header(),
            detected_objects_cls=DObjs, detected_object_cls=DObj,
            object_classification_cls=OClass, shape_cls=Shape, include_lidar_only=False,
        )
        self.assertEqual(len(out.objects), 1)


if __name__ == "__main__":
    unittest.main()
