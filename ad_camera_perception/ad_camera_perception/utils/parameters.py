"""Validation helpers for normalized image regions."""

from collections.abc import Iterable
from math import isfinite
from typing import Tuple


NormalizedCrop = Tuple[float, float, float, float]
NormalizedPolygon = Tuple[float, ...]


def _as_normalized_tuple(values: Iterable[float], name: str) -> Tuple[float, ...]:
    """Convert values to floats and require normalized finite coordinates."""
    try:
        coordinates = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric coordinates") from exc

    if any(
        not isfinite(coordinate) or coordinate < 0.0 or coordinate > 1.0
        for coordinate in coordinates
    ):
        raise ValueError(f"{name} coordinates must be finite and within [0.0, 1.0]")
    return coordinates


def validate_normalized_crop(values: Iterable[float]) -> NormalizedCrop:
    """Validate an ``x_min, y_min, x_max, y_max`` normalized crop."""
    coordinates = _as_normalized_tuple(values, "crop")
    if len(coordinates) != 4:
        raise ValueError("crop must contain exactly four coordinates")

    x_min, y_min, x_max, y_max = coordinates
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("crop minimum coordinates must be smaller than maximums")
    return x_min, y_min, x_max, y_max


def validate_normalized_polygon(values: Iterable[float]) -> NormalizedPolygon:
    """Validate a flat normalized polygon containing at least three points."""
    coordinates = _as_normalized_tuple(values, "polygon")
    if len(coordinates) < 6 or len(coordinates) % 2:
        raise ValueError("polygon must contain x/y pairs for at least three points")
    return coordinates
