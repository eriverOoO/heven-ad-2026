"""Tests for normalized image-region validation."""

import pytest

from ad_camera_perception.utils.parameters import (
    validate_normalized_crop,
    validate_normalized_polygon,
)


def test_valid_crop_is_converted_to_tuple():
    """A valid crop is returned as immutable float coordinates."""
    assert validate_normalized_crop([0, 0.2, 1, 1]) == (0.0, 0.2, 1.0, 1.0)


@pytest.mark.parametrize(
    "crop",
    [
        [],
        [0.0, 0.0, 1.0],
        [0.5, 0.0, 0.5, 1.0],
        [0.0, 0.8, 1.0, 0.2],
        [-0.1, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.1, 1.0],
        [0.0, 0.0, float("nan"), 1.0],
        [0.0, 0.0, float("inf"), 1.0],
    ],
)
def test_invalid_crop_is_rejected(crop):
    """Malformed or out-of-range crops are rejected."""
    with pytest.raises(ValueError):
        validate_normalized_crop(crop)


def test_valid_polygon_is_converted_to_tuple():
    """A polygon with at least three normalized points is accepted."""
    polygon = [0.1, 0.2, 0.9, 0.2, 0.5, 1.0]
    assert validate_normalized_polygon(polygon) == tuple(polygon)


@pytest.mark.parametrize(
    "polygon",
    [
        [],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 0.0, 0.5],
        [0.0, 0.0, 1.0, 0.0, 0.5, 1.1],
        [0.0, 0.0, 1.0, 0.0, float("nan"), 1.0],
    ],
)
def test_invalid_polygon_is_rejected(polygon):
    """Malformed or out-of-range polygons are rejected."""
    with pytest.raises(ValueError):
        validate_normalized_polygon(polygon)
