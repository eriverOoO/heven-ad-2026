import unittest

import numpy as np

from synthetic_xyzirc import XYZIRC_DTYPE, synthetic_xyzirc


class SyntheticXyzircTest(unittest.TestCase):
    def test_exact_heven_layout(self):
        self.assertEqual(XYZIRC_DTYPE.itemsize, 16)
        self.assertEqual(XYZIRC_DTYPE.fields["x"][1], 0)
        self.assertEqual(XYZIRC_DTYPE.fields["intensity"][1], 12)
        self.assertEqual(XYZIRC_DTYPE.fields["return_type"][1], 13)
        self.assertEqual(XYZIRC_DTYPE.fields["channel"][1], 14)

    def test_deterministic_and_valid_channels(self):
        first = synthetic_xyzirc()
        second = synthetic_xyzirc()
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(len(first), 16 * 64)
        self.assertEqual(int(first["channel"].min()), 0)
        self.assertEqual(int(first["channel"].max()), 15)
        self.assertTrue(np.isfinite(first["x"]).all())

    def test_rejects_empty_layout(self):
        with self.assertRaises(ValueError):
            synthetic_xyzirc(rings=0)


if __name__ == "__main__":
    unittest.main()
