import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from actor_gt import RigidTransform
from frame_alignment import NearestIndex, TimedSample
from metrics import distance_bin, greedy_center_matches, percentile


def test_nearest_index_enforces_source_time_limit():
    index = NearestIndex([TimedSample(100, 1000, "a"), TimedSample(140, 1040, "b")])
    assert index.nearest(128, 20).value == "b"
    assert index.nearest(200, 20) is None


def test_greedy_center_matching_is_one_to_one():
    matches = greedy_center_matches([(0.0, 0.0), (0.2, 0.0)], [(0.1, 0.0)], 1.0)
    assert len(matches) == 1
    assert matches[0][2] == 0.1


def test_percentile_uses_linear_interpolation():
    assert percentile([0.0, 10.0], 95) == 9.5
    assert percentile([], 50) is None


def test_distance_bins_are_half_open():
    edges = [20.0, 40.0, 60.0]
    assert distance_bin(19.999, edges) == "0-20m"
    assert distance_bin(20.0, edges) == "20-40m"
    assert distance_bin(60.0, edges) == "60m+"


def test_rigid_transform_inverse_round_trip():
    transform = RigidTransform((1.0, 2.0, 0.0), (0.0, 0.0, 2**-0.5, 2**-0.5))
    original = (3.0, 4.0, 5.0)
    recovered = transform.inverse().apply(transform.apply(original))
    assert recovered == pytest.approx(original)


import pytest
