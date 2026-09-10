import tempfile
import unittest
from pathlib import Path

import numpy as np

from av2_source_ring_adapter import (
    RingGeometry,
    adapt_source_rings,
    inverse_transform_points,
    reconstruct_source_local_points,
    select_unique_monotonic_rings,
    source_laser_numbers,
    summarize_ring_geometry,
    transform_points,
)
from prepare_av2_xyzirc import save
from publish_av2_xyzirc import load_npz
from synthetic_xyzirc import XYZIRC_DTYPE


def geometry(local_ring_id: int, elevation: float) -> RingGeometry:
    return RingGeometry("up_lidar", local_ring_id, 100, elevation, elevation, elevation, 0.0)


class Av2SourceRingAdapterTest(unittest.TestCase):
    def test_inverse_transform_roundtrip(self):
        angle = np.radians(30.0)
        rotation = np.asarray(
            [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        )
        translation = np.asarray([1.0, 2.0, 3.0])
        source = np.asarray([[3.0, 4.0, 5.0]])
        target = transform_points(source, rotation, translation)
        self.assertTrue(np.allclose(inverse_transform_points(target, rotation, translation), source))

    def test_acquisition_time_motion_is_undone(self):
        # At t=1 s ego moved +1 m in city. A point 10 m ahead is at city x=11
        # and is represented as x=11 in the t=0 reference ego frame.
        identity = np.eye(3)
        reconstructed = reconstruct_source_local_points(
            np.asarray([[11.0, 0.0, 0.0]]),
            np.asarray([1_000_000_000], dtype=np.int64),
            0,
            identity,
            np.zeros(3),
            np.asarray([0, 1_000_000_000], dtype=np.int64),
            np.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
            np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            identity,
            np.zeros(3),
        )
        self.assertTrue(np.allclose(reconstructed, [[10.0, 0.0, 0.0]]))

    def test_source_lidar_mapping(self):
        self.assertEqual(list(source_laser_numbers("up_lidar")), list(range(32)))
        self.assertEqual(list(source_laser_numbers("down_lidar")), list(range(32, 64)))

    def test_ring_elevation_recovery_statistics(self):
        rows = summarize_ring_geometry(
            {ring: np.asarray([ring - 16.1, ring - 16.0, ring - 15.9]) for ring in range(32)},
            "up_lidar",
        )
        self.assertEqual(len(rows), 32)
        self.assertAlmostEqual(rows[16].median_elevation_deg, 0.0)
        self.assertAlmostEqual(rows[16].mad_elevation_deg, 0.1)

    def test_unique_monotonic_selection_is_deterministic(self):
        rows = [geometry(index, -16.0 + index) for index in range(32)]
        first = select_unique_monotonic_rings(rows)
        second = select_unique_monotonic_rings(rows)
        self.assertEqual(first, second)
        self.assertEqual(len({row.source_laser_number for row in first}), 16)
        self.assertEqual([row.target_channel for row in first], list(range(16)))
        measured = [row.measured_elevation_deg for row in first]
        self.assertEqual(measured, sorted(measured))

    def test_selection_preserves_xyz_and_does_not_mix_sources(self):
        cloud = np.zeros(64, dtype=XYZIRC_DTYPE)
        cloud["x"] = np.arange(64, dtype=np.float32)
        lasers = np.arange(64, dtype=np.int64)
        selection = [
            geometry(index, float(-15 + 2 * index)) for index in range(16)
        ]
        mapped = select_unique_monotonic_rings(selection)
        output = adapt_source_rings(cloud, lasers, mapped)
        self.assertEqual(len(output), 16)
        self.assertTrue(np.array_equal(output["x"], np.arange(16, dtype=np.float32)))
        self.assertEqual(output["channel"].tolist(), list(range(16)))
        with self.assertRaises(ValueError):
            adapt_source_rings(cloud, lasers, mapped, source_lidar="down_lidar")

    def test_npz_roundtrip_keeps_mode_timestamp_and_layout(self):
        cloud = np.zeros(2, dtype=XYZIRC_DTYPE)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proxy.npz"
            save(path, cloud, 123, "log", "source_ring_vlp16_v2")
            loaded, metadata = load_npz(path)
        self.assertTrue(np.array_equal(loaded, cloud))
        self.assertEqual(metadata["timestamp_ns"], 123)
        self.assertEqual(metadata["mode"], "source_ring_vlp16_v2")


if __name__ == "__main__":
    unittest.main()
