import math
import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

from visualize_morai_gt import _materially_better, points_inside_box


def test_oriented_box_membership_uses_yaw_and_height():
    box = {
        "x": 0.0,
        "y": 0.0,
        "z": 1.0,
        "length": 4.0,
        "width": 2.0,
        "height": 2.0,
        "yaw": math.pi / 2.0,
    }
    points = np.array(
        [
            [0.0, 1.9, 1.0, 0.0],
            [1.1, 0.0, 1.0, 0.0],
            [0.0, 0.0, 2.1, 0.0],
            [0.0, -2.0, 0.0, 0.0],
        ]
    )
    assert points_inside_box(points, box).tolist() == [True, False, False, True]


def test_margin_is_explicit_and_symmetric():
    box = {"x": 0.0, "y": 0.0, "z": 0.0, "length": 2.0, "width": 2.0, "height": 2.0, "yaw": 0.0}
    points = np.array([[1.2, 0.0, 0.0, 0.0], [-1.2, 0.0, 0.0, 0.0]])
    assert not points_inside_box(points, box).any()
    assert points_inside_box(points, box, margin=0.25).all()


def test_materially_better_requires_absolute_and_relative_gain():
    assert _materially_better(10, 5)
    assert not _materially_better(4, 0)
    assert not _materially_better(104, 100)
