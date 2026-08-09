import struct

import pytest

from ad_morai_bridge_dev.codecs.lidar import decode_lidar2d


def _lidar_packet(first_intensity: int = 0) -> bytes:
    ranges = struct.pack("<HB", 1000, first_intensity) + b"".join(
        struct.pack("<HB", 1000 + index, index % 100) for index in range(1, 360)
    )
    return (
        b"#Lidar2D$"
        + struct.pack("<i3f", len(ranges), 1.0, 2.0, 3.0)
        + ranges
        + b"\r\n"
    )


def test_dev_lidar2d_decodes_official_360_range_layout_in_metres() -> None:
    packet = _lidar_packet()
    record = decode_lidar2d(packet)

    assert len(packet) == 1107
    assert len(record.distances_m) == 360
    assert record.aux == pytest.approx((1.0, 2.0, 3.0))
    assert record.distances_m[0] == pytest.approx(1.0)
    assert record.distances_m[-1] == pytest.approx(1.359)


def test_dev_lidar2d_intensity_uses_unsigned_byte() -> None:
    assert decode_lidar2d(_lidar_packet(255)).intensities[0] == 255
