import math
import unittest

import numpy as np

from ad_lidar_perception.ab3dmot_geometry import Box3D, bev_corners, giou_3d, polygon_area


class Ab3dmotGeometryTest(unittest.TestCase):
    def test_identical_boxes_have_giou_one(self):
        box = Box3D(0, 0, 0, 0.0, 4.0, 2.0, 1.5)
        self.assertAlmostEqual(giou_3d(box, box), 1.0, places=6)

    def test_disjoint_far_boxes_have_negative_giou_near_minus_one(self):
        box_a = Box3D(0, 0, 0, 0.0, 1.0, 1.0, 1.0)
        box_b = Box3D(1000.0, 0, 0, 0.0, 1.0, 1.0, 1.0)
        result = giou_3d(box_a, box_b)
        self.assertLess(result, -0.99)
        self.assertGreaterEqual(result, -1.0)

    def test_partial_overlap_giou_between_zero_and_one(self):
        box_a = Box3D(0, 0, 0, 0.0, 2.0, 2.0, 2.0)
        box_b = Box3D(1.0, 0, 0, 0.0, 2.0, 2.0, 2.0)  # half-overlapping along x
        result = giou_3d(box_a, box_b)
        self.assertGreater(result, 0.0)
        self.assertLess(result, 1.0)

    def test_rotated_box_bev_corners_z_up_convention(self):
        # yaw=pi/2 should rotate the length axis (local +x) onto world +y,
        # CCW about +z -- the ROS z-up convention this module implements.
        box = Box3D(0, 0, 0, math.pi / 2.0, length=4.0, width=2.0, height=1.0)
        corners = bev_corners(box)
        # the corner nominally at local (+length/2, +width/2) should land near
        # world (-width/2, +length/2) after a +90 deg CCW rotation
        expected = np.array([-1.0, 2.0])
        self.assertTrue(np.allclose(corners[0], expected, atol=1e-9))

    def test_adjacent_non_overlapping_boxes_give_low_but_bounded_giou(self):
        box_a = Box3D(0, 0, 0, 0.0, 2.0, 2.0, 2.0)
        box_b = Box3D(4.0, 0, 0, 0.0, 2.0, 2.0, 2.0)  # touching edges, no overlap
        result = giou_3d(box_a, box_b)
        self.assertLess(result, 0.0)
        self.assertGreater(result, -1.0)

    def test_polygon_area_shoelace_unit_square(self):
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        self.assertAlmostEqual(polygon_area(square), 1.0, places=9)

    def test_box3d_rejects_nonpositive_dimensions(self):
        with self.assertRaises(ValueError):
            Box3D(0, 0, 0, 0.0, 0.0, 1.0, 1.0)

    def test_box3d_rejects_nonfinite_fields(self):
        with self.assertRaises(ValueError):
            Box3D(math.nan, 0, 0, 0.0, 1.0, 1.0, 1.0)


if __name__ == "__main__":
    unittest.main()
