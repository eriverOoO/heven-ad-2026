"""POST-FREEZE EXTENSION 2, Stage 1: focused tests for the ROS-independent
camera+LiDAR late-fusion core. Run via the heven-centerpoint venv (numpy only,
no torch/ROS); deliberately not wired into colcon test, same precedent as
test_kalmannet_core.py."""
import math
import unittest

import numpy as np

from ad_lidar_perception.camera_lidar_fusion_core import (
    CameraDetection,
    CameraIntrinsics,
    CameraModel,
    FusionConfig,
    LidarDetection,
    associate_2d,
    fuse,
    iou_2d,
    map_coco_name_to_semantic,
    normalized_center_distance,
    project_lidar_box,
    rigid_transform_from_tf_chain,
)


def _forward_facing_camera(width=1280, height=720, fov=90.0):
    """A camera at the LiDAR origin looking down +x_lidar: optical z = lidar x,
    optical x = -lidar y, optical y = -lidar z."""
    optical_from_lidar = np.array([
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    return CameraModel(CameraIntrinsics.from_fov(width, height, fov), optical_from_lidar)


class TransformTest(unittest.TestCase):
    def test_tf_chain_matches_manual_composition(self):
        hops = [
            {"translation": [1.15, 0.0, 1.4], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0], "invert": False},
            {"translation": [1.9, 0.0, 1.2], "rotation_xyzw": [0.0, 0.017452406, 0.0, 0.99984770], "invert": True},
            {"translation": [0.0, 0.0, 0.0], "rotation_xyzw": [0.5, -0.5, 0.5, -0.5], "invert": True},
        ]
        matrix = rigid_transform_from_tf_chain(hops)
        self.assertEqual(matrix.shape, (4, 4))
        self.assertTrue(np.allclose(matrix[3], [0, 0, 0, 1]))
        # rotation block orthonormal
        rot = matrix[:3, :3]
        self.assertTrue(np.allclose(rot @ rot.T, np.eye(3), atol=1e-9))

    def test_forward_point_maps_to_positive_optical_z(self):
        camera = _forward_facing_camera()
        pt = camera.lidar_points_to_optical(np.array([[10.0, 0.0, 0.0]]))[0]
        self.assertAlmostEqual(pt[2], 10.0)   # forward -> +z
        self.assertAlmostEqual(pt[0], 0.0)    # on axis -> centered
        self.assertAlmostEqual(pt[1], 0.0)

    def test_left_of_lidar_is_left_in_image(self):
        camera = _forward_facing_camera()
        # +y_lidar is left; should map to -x_optical (left of image center)
        pt = camera.lidar_points_to_optical(np.array([[10.0, 3.0, 0.0]]))[0]
        self.assertLess(pt[0], 0.0)


class ProjectionTest(unittest.TestCase):
    def setUp(self):
        self.camera = _forward_facing_camera()
        self.config = FusionConfig()

    def test_box_ahead_projects_near_image_center(self):
        det = LidarDetection(x=20.0, y=0.0, z=0.0, length=4.0, width=2.0, height=1.6)
        pb = project_lidar_box(det, self.camera, self.config)
        self.assertTrue(pb.projectable)
        self.assertTrue(pb.overlaps_image)
        self.assertFalse(pb.degenerate)
        cx = (pb.u1 + pb.u2) / 2.0
        self.assertAlmostEqual(cx, self.camera.intrinsics.cx, delta=5.0)
        self.assertEqual(pb.num_corners_in_front, 8)

    def test_box_entirely_behind_camera_is_not_projectable(self):
        det = LidarDetection(x=-15.0, y=0.0, z=0.0, length=4.0, width=2.0, height=1.6)
        pb = project_lidar_box(det, self.camera, self.config)
        self.assertFalse(pb.projectable)
        self.assertEqual(pb.num_corners_in_front, 0)

    def test_box_straddling_camera_plane_needs_two_front_corners(self):
        det = LidarDetection(x=0.0, y=0.0, z=0.0, length=4.0, width=2.0, height=1.6)
        pb = project_lidar_box(det, self.camera, self.config)
        # half the corners are behind -> exactly 4 in front, still projectable
        self.assertEqual(pb.num_corners_in_front, 4)
        self.assertTrue(pb.projectable)

    def test_far_lateral_box_projects_outside_image_fov(self):
        det = LidarDetection(x=10.0, y=-40.0, z=0.0, length=4.0, width=2.0, height=1.6)
        pb = project_lidar_box(det, self.camera, self.config)
        self.assertTrue(pb.projectable)
        self.assertFalse(pb.overlaps_image)

    def test_clutter_box_flagged_degenerate_geometry_in_lidar_space(self):
        det = LidarDetection(x=15.0, y=0.0, z=0.0, length=0.1, width=0.1, height=0.1)
        self.assertTrue(det.is_degenerate_geometry)
        # a real vehicle-scale box is not
        self.assertFalse(
            LidarDetection(x=15.0, y=0.0, z=0.0, length=4.0, width=2.0, height=1.6).is_degenerate_geometry
        )

    def test_tiny_far_box_flagged_degenerate_projection(self):
        det = LidarDetection(x=180.0, y=0.0, z=0.0, length=0.1, width=0.1, height=0.1)
        pb = project_lidar_box(det, self.camera, self.config)
        self.assertTrue(pb.projectable)
        self.assertTrue(pb.degenerate)

    def test_non_finite_geometry_returns_safe_unprojectable(self):
        det = LidarDetection(x=math.nan, y=0.0, z=0.0, length=4.0, width=2.0, height=1.6)
        pb = project_lidar_box(det, self.camera, self.config)
        self.assertFalse(pb.projectable)
        self.assertTrue(pb.degenerate)

    def test_projection_is_deterministic(self):
        det = LidarDetection(x=18.0, y=1.0, z=0.0, length=4.2, width=1.9, height=1.5)
        a = project_lidar_box(det, self.camera, self.config)
        b = project_lidar_box(det, self.camera, self.config)
        self.assertEqual(a.box, b.box)


class IoUTest(unittest.TestCase):
    def test_identical_boxes_iou_one(self):
        self.assertAlmostEqual(iou_2d((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)

    def test_disjoint_boxes_iou_zero(self):
        self.assertEqual(iou_2d((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_half_overlap(self):
        self.assertAlmostEqual(iou_2d((0, 0, 10, 10), (5, 0, 15, 10)), 50.0 / 150.0)

    def test_iou_handles_unordered_corners(self):
        self.assertAlmostEqual(iou_2d((10, 10, 0, 0), (0, 0, 10, 10)), 1.0)


class CenterDistanceTest(unittest.TestCase):
    def test_coincident_centers_zero(self):
        self.assertAlmostEqual(normalized_center_distance((0, 0, 10, 10), (2, 2, 8, 8), 100.0), 0.0)

    def test_normalized_by_diagonal(self):
        d = normalized_center_distance((0, 0, 0, 0), (30, 40, 30, 40), 100.0)
        self.assertAlmostEqual(d, 0.5)  # hypot(30,40)=50, /100


class AssociationTest(unittest.TestCase):
    def setUp(self):
        self.camera = _forward_facing_camera()

    def _project(self, dets, config):
        return [project_lidar_box(d, self.camera, config) for d in dets]

    def test_one_to_one_assignment_no_double_use(self):
        config = FusionConfig(association_measure="iou", iou_gate=0.05)
        lidar = [
            LidarDetection(x=20.0, y=2.0, z=0.0, length=4.0, width=2.0, height=1.6),
            LidarDetection(x=20.0, y=-2.0, z=0.0, length=4.0, width=2.0, height=1.6),
        ]
        proj = self._project(lidar, config)
        cams = [
            CameraDetection(*_box_of(proj[0]), confidence=0.9, semantic_class="car"),
            CameraDetection(*_box_of(proj[1]), confidence=0.8, semantic_class="truck"),
        ]
        matches, un_p, un_c, _ = associate_2d(proj, cams, self.camera, config)
        self.assertEqual(len(matches), 2)
        self.assertEqual(len({m[0] for m in matches}), 2)
        self.assertEqual(len({m[1] for m in matches}), 2)
        self.assertEqual(un_p, [])
        self.assertEqual(un_c, [])

    def test_gate_rejects_poor_overlap(self):
        config = FusionConfig(association_measure="iou", iou_gate=0.30)
        lidar = [LidarDetection(x=20.0, y=0.0, z=0.0, length=4.0, width=2.0, height=1.6)]
        proj = self._project(lidar, config)
        far = CameraDetection(0.0, 0.0, 20.0, 20.0, confidence=0.9, semantic_class="car")
        matches, un_p, un_c, _ = associate_2d(proj, [far], self.camera, config)
        self.assertEqual(matches, [])
        self.assertEqual(un_p, [0])
        self.assertEqual(un_c, [0])

    def test_runner_up_score_reports_ambiguity(self):
        config = FusionConfig(association_measure="iou", iou_gate=0.01)
        lidar = [LidarDetection(x=20.0, y=0.0, z=0.0, length=6.0, width=3.0, height=2.0)]
        proj = self._project(lidar, config)
        b = _box_of(proj[0])
        cam_a = CameraDetection(b[0], b[1], b[2], b[3], confidence=0.9, semantic_class="car")
        cam_b = CameraDetection(b[0] + 10, b[1] + 5, b[2] + 10, b[3] + 5, confidence=0.7, semantic_class="car")
        matches, _, _, _ = associate_2d(proj, [cam_a, cam_b], self.camera, config)
        self.assertEqual(len(matches), 1)
        self.assertGreater(matches[0][3], 0.0)  # a non-trivial runner-up exists


class SemanticMappingTest(unittest.TestCase):
    def test_vehicle_and_person_classes_map(self):
        self.assertEqual(map_coco_name_to_semantic("car"), "car")
        self.assertEqual(map_coco_name_to_semantic("Truck"), "truck")
        self.assertEqual(map_coco_name_to_semantic("bus"), "bus")
        self.assertEqual(map_coco_name_to_semantic("motorcycle"), "motorcycle")
        self.assertEqual(map_coco_name_to_semantic("person"), "person")

    def test_non_traffic_classes_drop_to_none(self):
        for name in ("train", "boat", "traffic light", "sports ball", "airplane"):
            self.assertIsNone(map_coco_name_to_semantic(name))


class FuseTest(unittest.TestCase):
    def setUp(self):
        self.camera = _forward_facing_camera()

    def test_lidar_geometry_is_never_overwritten(self):
        lidar = [LidarDetection(x=20.0, y=0.0, z=0.0, length=4.3, width=2.1, height=1.55)]
        proj = project_lidar_box(lidar[0], self.camera, FusionConfig())
        cam = CameraDetection(*_box_of(proj), confidence=0.9, semantic_class="car", raw_class_name="car")
        result = fuse(lidar, [cam], self.camera, FusionConfig(iou_gate=0.05))
        self.assertEqual(len(result.fused), 1)
        obj = result.fused[0]
        self.assertEqual(obj.lidar.length, 4.3)
        self.assertEqual(obj.lidar.width, 2.1)
        self.assertEqual(obj.semantic_class, "car")
        self.assertTrue(obj.visual_confirmed)

    def test_unmatched_camera_becomes_camera_only(self):
        lidar = [LidarDetection(x=20.0, y=0.0, z=0.0, length=4.0, width=2.0, height=1.6)]
        cam_matching = CameraDetection(*_box_of(project_lidar_box(lidar[0], self.camera, FusionConfig())),
                                       confidence=0.9, semantic_class="car")
        cam_orphan = CameraDetection(5.0, 5.0, 40.0, 40.0, confidence=0.9, semantic_class="person")
        result = fuse(lidar, [cam_matching, cam_orphan], self.camera, FusionConfig(iou_gate=0.05))
        self.assertEqual(len(result.fused), 1)
        self.assertEqual(len(result.camera_only), 1)
        self.assertEqual(result.camera_only[0].semantic_class, "person")

    def test_unmatched_lidar_becomes_lidar_only_with_reason(self):
        lidar = [LidarDetection(x=-20.0, y=0.0, z=0.0, length=4.0, width=2.0, height=1.6)]  # behind camera
        result = fuse(lidar, [], self.camera, FusionConfig())
        self.assertEqual(len(result.lidar_only), 1)
        self.assertEqual(result.stats["ineligible_reasons"], {0: "behind_camera"})
        self.assertEqual(result.stats["n_fused"], 0)

    def test_low_confidence_camera_detection_is_dropped(self):
        lidar = [LidarDetection(x=20.0, y=0.0, z=0.0, length=4.0, width=2.0, height=1.6)]
        weak = CameraDetection(*_box_of(project_lidar_box(lidar[0], self.camera, FusionConfig())),
                               confidence=0.05, semantic_class="car")
        result = fuse(lidar, [weak], self.camera, FusionConfig(camera_min_confidence=0.25, iou_gate=0.05))
        self.assertEqual(len(result.fused), 0)
        self.assertEqual(result.stats["n_camera_dropped"], 1)
        self.assertIn("below_confidence", result.stats["dropped_camera_reasons"])

    def test_huge_camera_box_is_dropped_by_area(self):
        lidar = [LidarDetection(x=20.0, y=0.0, z=0.0, length=4.0, width=2.0, height=1.6)]
        huge = CameraDetection(0.0, 0.0, 1280.0, 700.0, confidence=0.9, semantic_class="car")
        result = fuse(lidar, [huge], self.camera, FusionConfig(camera_max_area_fraction=0.5))
        self.assertEqual(result.stats["n_camera_dropped"], 1)
        self.assertIn("too_large", result.stats["dropped_camera_reasons"])

    def test_ego_hood_band_camera_box_is_dropped_by_vertical_roi(self):
        # box centered at y~647 in a 720px image -> below the default roi_bottom=0.83*720=598
        lidar = [LidarDetection(x=20.0, y=0.0, z=0.0, length=4.0, width=2.0, height=1.6)]
        hood = CameraDetection(70.0, 575.0, 1218.0, 718.0, confidence=0.9, semantic_class="car")
        result = fuse(lidar, [hood], self.camera, FusionConfig())
        self.assertEqual(result.stats["n_camera_dropped"], 1)
        self.assertIn("outside_vertical_roi_ego_hood", result.stats["dropped_camera_reasons"])
        # a real vehicle near the horizon survives
        real = CameraDetection(460.0, 351.0, 753.0, 444.0, confidence=0.5, semantic_class="car")
        result2 = fuse(lidar, [real], self.camera, FusionConfig())
        self.assertEqual(result2.stats["n_camera_kept"], 1)

    def test_fuse_is_deterministic(self):
        lidar = [
            LidarDetection(x=20.0, y=1.5, z=0.0, length=4.0, width=2.0, height=1.6),
            LidarDetection(x=15.0, y=-1.0, z=0.0, length=0.1, width=0.1, height=0.1),
        ]
        cams = [CameraDetection(*_box_of(project_lidar_box(lidar[0], self.camera, FusionConfig())),
                                confidence=0.8, semantic_class="car")]
        r1 = fuse(lidar, cams, self.camera, FusionConfig(iou_gate=0.05))
        r2 = fuse(lidar, cams, self.camera, FusionConfig(iou_gate=0.05))
        self.assertEqual(r1.stats["n_fused"], r2.stats["n_fused"])
        self.assertEqual(r1.stats["n_lidar_only"], r2.stats["n_lidar_only"])
        self.assertEqual(
            [o.association_score for o in r1.fused],
            [o.association_score for o in r2.fused],
        )

    def test_degenerate_lidar_can_be_excluded_from_projection(self):
        lidar = [LidarDetection(x=15.0, y=0.0, z=0.0, length=0.1, width=0.1, height=0.1)]
        cfg = FusionConfig(exclude_degenerate_lidar_from_projection=True)
        result = fuse(lidar, [], self.camera, cfg)
        self.assertEqual(result.stats["ineligible_reasons"], {0: "degenerate_lidar_geometry"})


def _box_of(projected):
    return (projected.u1, projected.v1, projected.u2, projected.v2)


if __name__ == "__main__":
    unittest.main()
