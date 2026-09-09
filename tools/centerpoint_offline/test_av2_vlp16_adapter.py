import unittest

import numpy as np

from av2_vlp16_adapter import AdapterConfig, adapt_points, cloud_statistics
from synthetic_xyzirc import XYZIRC_DTYPE


class Av2Vlp16AdapterTest(unittest.TestCase):
    def test_beam_assignment_range_crop_and_layout(self):
        # Elevations -15, +1, and 30 degrees; last is outside a VLP-16 beam.
        points = np.array([[10.0, 0.0, -2.679, 20.0], [10.0, 0.0, 0.175, 30.0], [10.0, 0.0, 5.774, 40.0], [100.0, 0.0, 0.0, 50.0]], dtype=np.float32)
        cloud = adapt_points(points, AdapterConfig(vertical_tolerance_deg=1.0))
        self.assertEqual(cloud.dtype, XYZIRC_DTYPE)
        self.assertEqual(cloud.dtype.itemsize, 16)
        self.assertEqual(len(cloud), 2)
        self.assertTrue(np.all((cloud["channel"] >= 0) & (cloud["channel"] < 16)))
        self.assertEqual(set(cloud["channel"].tolist()), {0, 1})

    def test_azimuth_beam_bin_keeps_nearest_deterministically(self):
        points = np.array([[10.0, 0.0, 0.175, 10.0], [5.0, 0.0, 0.087, 20.0]], dtype=np.float32)
        config = AdapterConfig(vertical_tolerance_deg=1.0, azimuth_bin_deg=1.0)
        first, second = adapt_points(points, config), adapt_points(points, config)
        self.assertEqual(len(first), 1)
        self.assertTrue(np.array_equal(first, second))
        self.assertAlmostEqual(float(first["x"][0]), 5.0, places=5)

    def test_empty_and_intensity_modes(self):
        cloud = adapt_points(np.empty((0, 4), dtype=np.float32))
        self.assertEqual(len(cloud), 0)
        self.assertIsNone(cloud_statistics(cloud)["mean_intensity"])
        point = np.array([[10.0, 0.0, 0.175, 999.0]], dtype=np.float32)
        output = adapt_points(point, AdapterConfig(vertical_tolerance_deg=1.0, intensity_mode="constant", constant_intensity=17))
        self.assertEqual(int(output["intensity"][0]), 17)

    def test_invalid_input_or_config_rejected(self):
        with self.assertRaises(ValueError):
            adapt_points(np.zeros((3, 3), dtype=np.float32))
        with self.assertRaises(ValueError):
            adapt_points(np.zeros((1, 4), dtype=np.float32), AdapterConfig(target_elevations_deg=(0.0,)))


if __name__ == "__main__":
    unittest.main()
