from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class RouteProjection:
    segment_index: int
    progress_m: float
    cte_m: float
    heading_error_rad: float = 0.0


def cumulative_lengths(points: Sequence[tuple[float, float]]) -> list[float]:
    if len(points) < 2:
        raise ValueError("route needs at least two points")
    result = [0.0]
    for first, second in zip(points, points[1:]):
        length = math.hypot(second[0] - first[0], second[1] - first[1])
        if not math.isfinite(length):
            raise ValueError("route coordinates must be finite")
        result.append(result[-1] + length)
    if result[-1] <= 0.0:
        raise ValueError("route length must be positive")
    return result


def project_to_route(
    points: Sequence[tuple[float, float]],
    lengths: Sequence[float],
    x: float,
    y: float,
    hint: int | None = None,
    forward_window: int = 240,
    backward_window: int = 20,
    yaw_rad: float | None = None,
) -> RouteProjection:
    if len(points) < 2 or len(lengths) != len(points):
        raise ValueError("route points and cumulative lengths do not match")
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("query position must be finite")
    if yaw_rad is not None and not math.isfinite(yaw_rad):
        raise ValueError("query heading must be finite")

    segment_count = len(points) - 1
    if hint is None:
        start, stop = 0, segment_count
    else:
        hint = max(0, min(int(hint), segment_count - 1))
        start = max(0, hint - backward_window)
        stop = min(segment_count, hint + forward_window)
        if stop <= start:
            stop = min(segment_count, start + 1)

    candidates: list[tuple[float, int, float, float, float]] = []
    for index in range(start, stop):
        ax, ay = points[index]
        bx, by = points[index + 1]
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        if length_sq <= 1.0e-12:
            continue
        fraction = max(
            0.0,
            min(1.0, ((x - ax) * dx + (y - ay) * dy) / length_sq),
        )
        px, py = ax + fraction * dx, ay + fraction * dy
        distance_sq = (x - px) ** 2 + (y - py) ** 2
        segment_heading = math.atan2(dy, dx)
        heading_error = (
            math.atan2(
                math.sin(segment_heading - yaw_rad),
                math.cos(segment_heading - yaw_rad),
            )
            if yaw_rad is not None
            else 0.0
        )
        candidates.append(
            (
                distance_sq,
                index,
                fraction,
                math.sqrt(length_sq),
                heading_error,
            )
        )

    if not candidates:
        raise ValueError("route contains no nonzero segment")
    best = min(candidates, key=lambda candidate: candidate[0])
    if yaw_rad is not None:
        directionally_valid = [
            candidate
            for candidate in candidates
            if abs(candidate[4]) <= math.pi / 2.0
            and math.sqrt(candidate[0]) <= math.sqrt(best[0]) + 5.0
        ]
        if directionally_valid:
            best = min(directionally_valid, key=lambda candidate: candidate[0])
    distance_sq, index, fraction, segment_length, heading_error = best
    return RouteProjection(
        segment_index=index,
        progress_m=lengths[index] + fraction * segment_length,
        cte_m=math.sqrt(distance_sq),
        heading_error_rad=heading_error,
    )


def start_yaw_deg(points: Sequence[tuple[float, float]]) -> float:
    if len(points) < 2:
        raise ValueError("route needs at least two points")
    origin = points[0]
    for point in points[1:]:
        dx, dy = point[0] - origin[0], point[1] - origin[1]
        if math.hypot(dx, dy) > 1.0e-6:
            return math.degrees(math.atan2(dy, dx))
    raise ValueError("route has no heading")


def pose_at_progress(
    points: Sequence[tuple[float, float]],
    lengths: Sequence[float],
    progress_m: float,
) -> tuple[float, float, float]:
    """Interpolate route position and heading at one cumulative distance."""
    if len(points) < 2 or len(lengths) != len(points):
        raise ValueError("route points and cumulative lengths do not match")
    if not math.isfinite(progress_m):
        raise ValueError("route progress must be finite")
    progress = min(max(0.0, progress_m), lengths[-1])
    nonzero_segments = [
        index
        for index in range(len(points) - 1)
        if lengths[index + 1] - lengths[index] > 1.0e-12
    ]
    if not nonzero_segments:
        raise ValueError("route contains no nonzero segment")
    for index in nonzero_segments:
        segment_length = lengths[index + 1] - lengths[index]
        if (
            progress <= lengths[index + 1]
            or index == nonzero_segments[-1]
        ):
            fraction = min(
                1.0,
                max(0.0, (progress - lengths[index]) / segment_length),
            )
            ax, ay = points[index]
            bx, by = points[index + 1]
            return (
                ax + fraction * (bx - ax),
                ay + fraction * (by - ay),
                math.atan2(by - ay, bx - ax),
            )
    raise RuntimeError("unreachable route interpolation state")
