"""Coordinate conversion and pure-pursuit helpers."""

import math
from typing import Sequence


def cumulative_distances(
    route: Sequence[Sequence[float]],
) -> list[float]:
    """Return the cumulative planar arc distance at every route point."""
    if not route:
        return []
    distances = [0.0]
    for previous, current in zip(route, route[1:]):
        distances.append(
            distances[-1]
            + math.hypot(current[0] - previous[0], current[1] - previous[1])
        )
    return distances


def latlon_to_utm52(latitude_deg: float, longitude_deg: float) -> tuple[float, float]:
    """Convert WGS84 latitude/longitude to UTM zone 52N metres."""
    semi_major = 6378137.0
    eccentricity_squared = 0.00669438
    scale = 0.9996
    eccentricity_prime_squared = eccentricity_squared / (1.0 - eccentricity_squared)
    latitude = math.radians(latitude_deg)
    longitude = math.radians(longitude_deg)
    central_meridian = math.radians(129.0)
    sin_latitude = math.sin(latitude)
    cos_latitude = math.cos(latitude)
    tangent = math.tan(latitude)
    radius = semi_major / math.sqrt(
        1.0 - eccentricity_squared * sin_latitude * sin_latitude
    )
    t = tangent * tangent
    c = eccentricity_prime_squared * cos_latitude * cos_latitude
    a = cos_latitude * (longitude - central_meridian)
    e2 = eccentricity_squared
    meridional_arc = semi_major * (
        (1.0 - e2 / 4.0 - 3.0 * e2**2 / 64.0 - 5.0 * e2**3 / 256.0)
        * latitude
        - (3.0 * e2 / 8.0 + 3.0 * e2**2 / 32.0 + 45.0 * e2**3 / 1024.0)
        * math.sin(2.0 * latitude)
        + (15.0 * e2**2 / 256.0 + 45.0 * e2**3 / 1024.0)
        * math.sin(4.0 * latitude)
        - 35.0 * e2**3 / 3072.0 * math.sin(6.0 * latitude)
    )
    easting = scale * radius * (
        a
        + (1.0 - t + c) * a**3 / 6.0
        + (5.0 - 18.0 * t + t**2 + 72.0 * c - 58.0 * eccentricity_prime_squared)
        * a**5
        / 120.0
    ) + 500000.0
    northing = scale * (
        meridional_arc
        + radius
        * tangent
        * (
            a**2 / 2.0
            + (5.0 - t + 9.0 * c + 4.0 * c**2) * a**4 / 24.0
            + (
                61.0
                - 58.0 * t
                + t**2
                + 600.0 * c
                - 330.0 * eccentricity_prime_squared
            )
            * a**6
            / 720.0
        )
    )
    return easting, northing


def quaternion_yaw(w: float, x: float, y: float, z: float) -> float:
    """Return ROS ENU yaw from a quaternion."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def normalize_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def nearest_index(
    route: Sequence[Sequence[float]],
    x: float,
    y: float,
    start_index: int = 0,
    max_search_ahead: int | None = None,
) -> int:
    """Return the nearest route point inside a forward progress window."""
    if not route:
        raise ValueError("route is empty")
    begin = max(0, min(start_index, len(route) - 1))
    end = len(route)
    if max_search_ahead is not None:
        if max_search_ahead < 0:
            raise ValueError("max_search_ahead cannot be negative")
        end = min(len(route), begin + max_search_ahead + 1)
    return min(
        range(begin, end),
        key=lambda index: (route[index][0] - x) ** 2 + (route[index][1] - y) ** 2,
    )


def lookahead_index(
    route: Sequence[Sequence[float]], start_index: int, distance: float
) -> int:
    """Walk forward along a polyline until the requested arc distance."""
    travelled = 0.0
    index = start_index
    while index + 1 < len(route) and travelled < distance:
        travelled += math.hypot(
            route[index + 1][0] - route[index][0],
            route[index + 1][1] - route[index][1],
        )
        index += 1
    return index


def normalized_pure_pursuit_steer(
    *,
    x: float,
    y: float,
    yaw: float,
    target_x: float,
    target_y: float,
    wheelbase: float,
    lookahead: float,
    max_wheel_angle_rad: float,
) -> float:
    """Return MORAI's normalized steering request for a lookahead point."""
    bearing = math.atan2(target_y - y, target_x - x)
    alpha = normalize_angle(bearing - yaw)
    wheel_angle = math.atan2(2.0 * wheelbase * math.sin(alpha), lookahead)
    return max(-1.0, min(1.0, wheel_angle / max_wheel_angle_rad))
