"""Pure metric calculations used by the detection benchmark."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


Point2 = tuple[float, float]


def percentile(values: Iterable[float], percentage: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if not 0.0 <= percentage <= 100.0:
        raise ValueError("percentage must be in [0, 100]")
    rank = (len(ordered) - 1) * percentage / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def greedy_center_matches(
    detections: Sequence[Point2], actors: Sequence[Point2], threshold_m: float
) -> list[tuple[int, int, float]]:
    """Greedily select globally shortest one-to-one center pairs."""
    candidates = []
    for detection_index, detection in enumerate(detections):
        for actor_index, actor in enumerate(actors):
            distance = math.hypot(detection[0] - actor[0], detection[1] - actor[1])
            if distance <= threshold_m:
                candidates.append((distance, detection_index, actor_index))
    candidates.sort()
    used_detections: set[int] = set()
    used_actors: set[int] = set()
    matches = []
    for distance, detection_index, actor_index in candidates:
        if detection_index in used_detections or actor_index in used_actors:
            continue
        used_detections.add(detection_index)
        used_actors.add(actor_index)
        matches.append((detection_index, actor_index, distance))
    return matches


def distance_bin(distance_m: float, edges_m: Sequence[float]) -> str:
    lower = 0.0
    for upper in edges_m:
        if distance_m < upper:
            return f"{lower:g}-{upper:g}m"
        lower = upper
    return f"{lower:g}m+"
