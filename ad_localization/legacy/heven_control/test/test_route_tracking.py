import math

import pytest

from heven_control.route_tracking import (
    cumulative_distances,
    latlon_to_utm52,
    lookahead_index,
    nearest_index,
    normalized_pure_pursuit_steer,
)


def test_cumulative_distances_follow_route_arc_length():
    route = [(0.0, 0.0), (3.0, 4.0), (6.0, 4.0)]

    assert cumulative_distances(route) == pytest.approx([0.0, 5.0, 8.0])


def test_competition_start_gps_converts_to_mgeo_local_coordinates():
    easting, northing = latlon_to_utm52(37.2410133333, 126.7743833333)

    assert easting == pytest.approx(302586.75, abs=0.05)
    assert northing == pytest.approx(4123930.31, abs=0.05)


def test_route_indices_only_move_forward():
    route = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0), (15.0, 0.0)]

    assert nearest_index(route, 4.9, 0.2, 0) == 1
    assert nearest_index(route, 4.9, 0.2, 2) == 2
    assert lookahead_index(route, 1, 8.0) == 3


def test_nearest_index_does_not_jump_across_overlapping_route():
    route = [(float(index), 0.0) for index in range(11)] + [(2.1, 0.0)]

    assert nearest_index(route, 2.1, 0.0, 2) == 11
    assert nearest_index(route, 2.1, 0.0, 2, max_search_ahead=3) == 2


def test_pure_pursuit_is_straight_for_target_ahead():
    steer = normalized_pure_pursuit_steer(
        x=0.0,
        y=0.0,
        yaw=0.0,
        target_x=10.0,
        target_y=0.0,
        wheelbase=3.0,
        lookahead=7.0,
        max_wheel_angle_rad=math.radians(40.0),
    )

    assert steer == pytest.approx(0.0)
