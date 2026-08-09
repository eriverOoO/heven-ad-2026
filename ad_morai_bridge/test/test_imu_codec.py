import struct

import pytest

from ad_morai_bridge.codecs.common import PacketFormatError
from ad_morai_bridge.codecs.imu import decode_imu


def _imu(timestamped: bool) -> bytes:
    values = (1.0, 0.1, 0.2, 0.3, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    payload = (struct.pack("<ii", 12, 34) if timestamped else b"") + struct.pack(
        "<10d", *values
    )
    return b"#IMUData$" + struct.pack("<i3i", len(payload), 0, 0, 0) + payload + b"\r\n"


@pytest.mark.parametrize("timestamped,size,stamp", [(False, 107, None), (True, 115, (12, 34))])
def test_imu_accepts_manual_and_live_timestamped_layout(timestamped, size, stamp):
    packet = _imu(timestamped)
    result = decode_imu(packet)

    assert len(packet) == size
    assert result.stamp == stamp
    assert result.orientation_xyzw == pytest.approx((0.1, 0.2, 0.3, 1.0))
    assert result.angular_velocity == pytest.approx((4.0, 5.0, 6.0))
    assert result.linear_acceleration == pytest.approx((7.0, 8.0, 9.0))


def test_imu_bad_quaternion_and_length_fail():
    packet = bytearray(_imu(True))
    struct.pack_into("<d", packet, 9 + 16 + 8, float("nan"))

    with pytest.raises(PacketFormatError):
        decode_imu(bytes(packet))
    with pytest.raises(PacketFormatError):
        decode_imu(_imu(True)[:-1])


def test_timestamped_imu_preserves_invalid_source_for_policy_fallback_and_audit():
    packet = bytearray(_imu(True))
    struct.pack_into("<i", packet, len(b"#IMUData$") + 16 + 4, 1_000_000_000)

    record = decode_imu(bytes(packet))

    assert record.stamp == (12, 1_000_000_000)
