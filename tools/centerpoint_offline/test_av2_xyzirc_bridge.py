import tempfile
import unittest
from pathlib import Path

import numpy as np

from publish_av2_xyzirc import load_npz


class Av2XyzircBridgeTest(unittest.TestCase):
    def test_npz_layout_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.npz"
            np.savez(
                path,
                x=np.array([1], np.float32),
                y=np.array([2], np.float32),
                z=np.array([3], np.float32),
                intensity=np.array([4], np.uint8),
                return_type=np.array([0], np.uint8),
                channel=np.array([63], np.uint16),
                timestamp_ns=np.int64(7),
                source_log_id=np.str_("log"),
                mode=np.str_("native"),
                coordinate_frame=np.str_("av2_egovehicle"),
            )
            cloud, metadata = load_npz(path)
            self.assertEqual(cloud.dtype.itemsize, 16)
            self.assertEqual(int(cloud["channel"][0]), 63)
            self.assertEqual(metadata["timestamp_ns"], 7)

    def test_rejects_inconsistent_lengths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            np.savez(
                path,
                x=np.zeros(2),
                y=np.zeros(1),
                z=np.zeros(1),
                intensity=np.zeros(1),
                return_type=np.zeros(1),
                channel=np.zeros(1),
                coordinate_frame=np.str_("av2_egovehicle"),
            )
            with self.assertRaises(ValueError):
                load_npz(path)


if __name__ == "__main__":
    unittest.main()
