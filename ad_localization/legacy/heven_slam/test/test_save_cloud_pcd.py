from pathlib import Path

import numpy as np

from heven_slam.save_cloud_pcd_node import write_binary_xyz_pcd


def test_write_binary_xyz_pcd_filters_non_finite_points(tmp_path: Path):
    output = tmp_path / "map.pcd"
    count = write_binary_xyz_pcd(
        np.array([[1.0, 2.0, 3.0], [np.nan, 0.0, 1.0]], dtype=np.float32),
        output,
    )

    data = output.read_bytes()
    assert count == 1
    assert b"FIELDS x y z\n" in data
    assert b"POINTS 1\n" in data
    assert data.endswith(np.array([[1.0, 2.0, 3.0]], dtype="<f4").tobytes())
