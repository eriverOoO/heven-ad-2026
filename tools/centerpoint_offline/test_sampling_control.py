import unittest

import numpy as np

from prepare_sampling_control import sample_up_lidar, up_lidar_mask, validate_sample
from synthetic_xyzirc import XYZIRC_DTYPE


def cloud() -> np.ndarray:
    result = np.zeros(12, dtype=XYZIRC_DTYPE)
    result["x"] = np.arange(12, dtype=np.float32)
    result["intensity"] = np.arange(12, dtype=np.uint8)
    result["channel"] = np.asarray([0, 1, 2, 3, 30, 31, 32, 33, 40, 41, 62, 63])
    return result


class SamplingControlTest(unittest.TestCase):
    def test_up_lidar_filter_keeps_only_channels_zero_through_31(self):
        values = cloud()
        up = values[up_lidar_mask(values)]
        self.assertEqual(up["channel"].tolist(), [0, 1, 2, 3, 30, 31])

    def test_random_sample_is_exact_count_without_replacement_and_immutable(self):
        up = cloud()[up_lidar_mask(cloud())]
        sampled, indices = sample_up_lidar(up, target_count=4, seed=11, timestamp_ns=99)
        validation = validate_sample(up, sampled, indices, 4)
        self.assertTrue(validation["without_replacement"])
        self.assertTrue(validation["coordinates_and_attributes_unchanged"])
        self.assertEqual(len(np.unique(indices)), 4)

    def test_seed_and_timestamp_make_sampling_deterministic_but_seeds_differ(self):
        up = cloud()[up_lidar_mask(cloud())]
        left, left_indices = sample_up_lidar(up, 4, 11, 99)
        right, right_indices = sample_up_lidar(up, 4, 11, 99)
        other, other_indices = sample_up_lidar(up, 4, 29, 99)
        self.assertTrue(np.array_equal(left, right))
        self.assertTrue(np.array_equal(left_indices, right_indices))
        self.assertFalse(np.array_equal(left_indices, other_indices))
        self.assertFalse(np.array_equal(left, other))


if __name__ == "__main__":
    unittest.main()
