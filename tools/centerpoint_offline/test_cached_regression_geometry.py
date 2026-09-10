import math
import unittest

from audit_cached_regression_geometry import (
    geometry_error,
    geometry_shift,
    quaternion_yaw,
    yaw_axis_error,
)


class CachedRegressionGeometryTest(unittest.TestCase):
    def test_quaternion_yaw_uses_av2_wxyz_contract(self):
        yaw = math.pi / 2.0
        quaternion = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
        self.assertAlmostEqual(quaternion_yaw(quaternion), yaw)

    def test_yaw_axis_error_treats_pi_flip_as_same_box_axis(self):
        self.assertAlmostEqual(yaw_axis_error(0.2, 0.2 + math.pi), 0.0)
        self.assertAlmostEqual(yaw_axis_error(0.0, math.pi / 2.0), math.pi / 2.0)

    def test_same_cell_geometry_has_zero_native_sparse_shift(self):
        box = {
            "x": 1.0, "y": 2.0, "z": 3.0,
            "length": 4.0, "width": 2.0, "height": 1.5,
            "ros_yaw": 0.4, "vx": 0.0, "vy": 0.0,
        }
        shift = geometry_shift(box, dict(box))
        self.assertTrue(all(value == 0.0 for value in shift.values()))

    def test_geometry_error_keeps_center_and_box_axis_terms_separate(self):
        prediction = {
            "x": 2.0, "y": 2.0, "z": 1.0,
            "length": 4.5, "width": 1.5, "height": 2.0,
            "ros_yaw": math.pi,
        }
        gt = {
            "x": 1.0, "y": 2.0, "z": 1.5,
            "length": 4.0, "width": 2.0, "height": 1.5,
            "quaternion": (1.0, 0.0, 0.0, 0.0),
        }
        error = geometry_error(prediction, gt)
        self.assertAlmostEqual(error["center_xy_error_m"], 1.0)
        self.assertAlmostEqual(error["z_error_m"], 0.5)
        self.assertAlmostEqual(error["yaw_axis_error_rad"], 0.0)
        self.assertAlmostEqual(error["length_abs_error_m"], 0.5)


if __name__ == "__main__":
    unittest.main()
