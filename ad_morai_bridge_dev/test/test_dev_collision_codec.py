import math
import struct

import pytest

from ad_morai_bridge.codecs.common import PacketFormatError, encode_envelope
from ad_morai_bridge_dev.codecs.collision import decode_npc_collisions


_COLLISION_PAIR = struct.Struct("<hh13fhh13f")


def _collision_packet() -> bytes:
    first = tuple(float(value) for value in range(1, 14))
    second = tuple(float(value) for value in range(21, 34))
    populated = _COLLISION_PAIR.pack(1, 11, *first, 2, 22, *second)
    empty = _COLLISION_PAIR.pack(0, 0, *(0.0,) * 13, 0, 0, *(0.0,) * 13)
    return encode_envelope(b"#VehicleCollision$", populated + empty * 9)


def test_npc_collision_preserves_both_objects_and_all_vectors():
    packet = _collision_packet()
    record = decode_npc_collisions(packet)

    assert len(packet) == 1156
    assert len(record.collisions) == 1
    collision = record.collisions[0]
    assert (collision.first.object_type, collision.first.object_id) == (1, 11)
    assert collision.first.position == pytest.approx((1.0, 2.0, 3.0))
    assert collision.first.heading == pytest.approx(4.0)
    assert collision.first.size == pytest.approx((5.0, 6.0, 7.0))
    assert collision.first.velocity == pytest.approx((8.0, 9.0, 10.0))
    assert collision.first.acceleration == pytest.approx((11.0, 12.0, 13.0))
    assert (collision.second.object_type, collision.second.object_id) == (2, 22)
    assert collision.second.position == pytest.approx((21.0, 22.0, 23.0))
    assert collision.second.heading == pytest.approx(24.0)
    assert collision.second.size == pytest.approx((25.0, 26.0, 27.0))
    assert collision.second.velocity == pytest.approx((28.0, 29.0, 30.0))
    assert collision.second.acceleration == pytest.approx((31.0, 32.0, 33.0))


def test_npc_collision_rejects_wrong_size_and_nonfinite_values():
    with pytest.raises(PacketFormatError):
        decode_npc_collisions(_collision_packet()[:-1])

    packet = bytearray(_collision_packet())
    header_size = len(b"#VehicleCollision$") + 16
    struct.pack_into("<f", packet, header_size + 4, math.nan)
    with pytest.raises(PacketFormatError):
        decode_npc_collisions(bytes(packet))
