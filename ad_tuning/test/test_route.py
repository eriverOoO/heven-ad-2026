import math

import pytest

from ad_tuning.route import (
    cumulative_lengths,
    pose_at_progress,
    project_to_route,
    start_yaw_deg,
)


def test_projection_reports_polyline_progress_and_cte():
    points = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0))
    lengths = cumulative_lengths(points)
    projection = project_to_route(points, lengths, 8.0, 2.0)
    assert projection.progress_m == pytest.approx(8.0)
    assert projection.cte_m == pytest.approx(2.0)


def test_hint_keeps_closed_route_start_on_first_segment():
    points = ((0.0, 0.0), (10.0, 0.0), (0.0, 0.0))
    lengths = cumulative_lengths(points)
    projection = project_to_route(points, lengths, 0.0, 0.0, hint=0)
    assert projection.segment_index == 0
    assert projection.progress_m == pytest.approx(0.0)


def test_start_yaw_uses_first_nonzero_segment():
    yaw = start_yaw_deg(((1.0, 2.0), (1.0, 2.0), (2.0, 3.0)))
    assert yaw == pytest.approx(45.0)
    assert math.isfinite(start_yaw_deg(((0.0, 0.0), (0.0, -1.0))))


def test_projection_uses_heading_to_reject_nearby_opposite_branch():
    points = (
        (0.0, 0.0),
        (5.0, 0.0),
        (10.0, 0.0),
        (10.0, 1.0),
        (5.0, 1.0),
        (0.0, 1.0),
    )
    projection = project_to_route(
        points,
        cumulative_lengths(points),
        5.0,
        0.4,
        yaw_rad=math.pi,
        hint=None,
    )

    assert projection.segment_index == 3
    assert abs(projection.heading_error_rad) < 1.0e-12


def test_pose_at_progress_interpolates_terminal_position_and_heading():
    points = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0))
    lengths = cumulative_lengths(points)

    x, y, heading = pose_at_progress(points, lengths, 15.0)

    assert x == pytest.approx(10.0)
    assert y == pytest.approx(5.0)
    assert heading == pytest.approx(math.pi / 2.0)
