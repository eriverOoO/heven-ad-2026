import unittest

import numpy as np

from prepare_distance_stratified_control import radial_bin_indices, stratified_sample
from synthetic_xyzirc import XYZIRC_DTYPE


def make_cloud(radii: list[float]) -> np.ndarray:
    cloud = np.zeros(len(radii), dtype=XYZIRC_DTYPE)
    cloud["x"] = radii
    cloud["channel"] = 1
    return cloud


class DistanceStratifiedControlTest(unittest.TestCase):
    def test_samples_exact_radial_histogram_without_replacement(self):
        up = make_cloud([1, 2, 3, 6, 7, 8, 11, 12, 13, 111])
        ring = make_cloud([1, 3, 7, 12, 111])
        output, indices, target_bins = stratified_sample(up, ring, 11, 99)
        self.assertEqual(len(output), len(ring))
        self.assertEqual(len(indices), len(np.unique(indices)))
        self.assertTrue(np.array_equal(output, up[indices]))
        self.assertTrue(np.array_equal(np.bincount(radial_bin_indices(output)), np.bincount(target_bins)))

    def test_insufficient_source_bin_fails_instead_of_sampling_with_replacement(self):
        up, ring = make_cloud([1, 2]), make_cloud([1, 2, 3])
        with self.assertRaises(ValueError):
            stratified_sample(up, ring, 11, 99)

    def test_seed_is_deterministic(self):
        up = make_cloud(list(range(1, 20)))
        ring = make_cloud([1, 2, 3, 6, 7, 11, 12, 16])
        _, left, _ = stratified_sample(up, ring, 11, 99)
        _, right, _ = stratified_sample(up, ring, 11, 99)
        self.assertTrue(np.array_equal(left, right))

    def test_distinct_seeds_select_distinct_control_points(self):
        up = make_cloud(list(range(1, 101)))
        ring = make_cloud(list(range(1, 21)))
        _, left, _ = stratified_sample(up, ring, 11, 99)
        _, right, _ = stratified_sample(up, ring, 29, 99)
        self.assertFalse(np.array_equal(left, right))


if __name__ == "__main__": unittest.main()
