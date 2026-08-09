from dataclasses import replace
import math
import struct

import pytest

from ad_morai_bridge.codecs.common import PacketFormatError
from ad_morai_bridge.codecs.ego import decode_collisions, decode_ego_status, encode_ctrl_cmd
from ad_morai_bridge.protocol_records import CtrlCommandRecord


def _collision_packet() -> bytes:
    hit = struct.pack("<hh6f", 1, 7, 1.0, 2.0, 3.0, 100.0, 200.0, 0.0)
    empty = struct.pack("<hh6f", -1, 0, *(0.0,) * 6)
    payload = struct.pack("<ii", 3, 4) + hit + empty * 4
    return b"#CollisionData$" + struct.pack("<i3i", len(payload), 0, 0, 0) + payload + b"\r\n"


def _ego_packet(extended: bool) -> bytes:
    prefix = struct.pack(
        "<ii2bfi2f3f3f3f3f3f3f3ff38s",
        12,
        34,
        2,
        4,
        27.5,
        99,
        0.2,
        0.0,
        4.7,
        1.8,
        1.5,
        0.9,
        2.9,
        0.8,
        10.0,
        20.0,
        0.1,
        0.01,
        0.02,
        1.2,
        27.5,
        0.0,
        0.0,
        0.0,
        0.0,
        0.1,
        0.2,
        0.0,
        0.0,
        0.03,
        b"LINK_EGO",
    )
    payload = prefix + (struct.pack("<12f", *range(12)) if extended else b"")
    return b"#MoraiInfo$" + struct.pack("<i3i", len(payload), 0, 0, 0) + payload + b"\r\n"


def test_ctrl_cmd_type_one_packs_every_wire_field():
    packet = encode_ctrl_cmd(
        CtrlCommandRecord(
            ctrl_mode=2,
            gear=4,
            long_cmd_type=1,
            velocity=12.5,
            acceleration=1.25,
            accel=0.3,
            brake=0.4,
            steering=-0.2,
        )
    )

    assert len(packet) == 55
    assert packet.startswith(b"#MoraiCtrlCmd$")
    assert struct.unpack_from("<i", packet, 14)[0] == 23
    assert packet.endswith(b"\r\n")
    ctrl_mode, gear, long_cmd_type, velocity, acceleration, accel, brake, steering = (
        struct.unpack("<bbb5f", packet[30:-2])
    )
    assert (ctrl_mode, gear, long_cmd_type) == (2, 4, 1)
    assert (velocity, acceleration, accel, brake, steering) == pytest.approx(
        (12.5, 1.25, 0.3, 0.4, -0.2)
    )


@pytest.mark.parametrize(
    "command,expected",
    [
        (CtrlCommandRecord(accel=0.0, brake=0.0, steering=-1.0), (0.0, 0.0, -1.0)),
        (CtrlCommandRecord(accel=1.0, brake=1.0, steering=1.0), (1.0, 1.0, 1.0)),
    ],
)
def test_ctrl_cmd_accepts_inclusive_normalized_control_endpoints(command, expected):
    packet = encode_ctrl_cmd(command)

    payload = struct.unpack("<bbb5f", packet[30:-2])
    assert payload[5:] == pytest.approx(expected)


@pytest.mark.parametrize(
    "field,value",
    [
        ("ctrl_mode", 0),
        ("ctrl_mode", 3),
        ("ctrl_mode", math.nan),
        ("ctrl_mode", math.inf),
        ("gear", -1),
        ("gear", 6),
        ("gear", math.nan),
        ("gear", math.inf),
        ("long_cmd_type", 0),
        ("long_cmd_type", 2),
        ("long_cmd_type", 3),
        ("velocity", math.nan),
        ("velocity", math.inf),
        ("acceleration", math.nan),
        ("acceleration", math.inf),
        ("accel", -0.001),
        ("accel", 1.001),
        ("accel", math.nan),
        ("accel", math.inf),
        ("brake", -0.001),
        ("brake", 1.001),
        ("brake", math.nan),
        ("brake", math.inf),
        ("steering", -1.001),
        ("steering", 1.001),
        ("steering", math.nan),
        ("steering", math.inf),
    ],
)
def test_ctrl_cmd_rejects_nonfinite_and_out_of_range_fields(field, value):
    command = replace(CtrlCommandRecord(), **{field: value})

    with pytest.raises(PacketFormatError):
        encode_ctrl_cmd(command)


def test_collision_golden_181_byte_layout():
    packet = _collision_packet()
    collisions = decode_collisions(packet)

    assert len(packet) == 181
    assert collisions.stamp == (3, 4)
    assert len(collisions.collisions) == 1
    assert collisions.collisions[0].object_id == 7
    assert collisions.collisions[0].position == pytest.approx((1.0, 2.0, 3.0))
    assert collisions.collisions[0].global_offset == pytest.approx((100.0, 200.0, 0.0))


@pytest.mark.parametrize("extended,size", [(False, 181), (True, 229)])
def test_ego_status_accepts_base_and_extended_24r2_layout(extended, size):
    packet = _ego_packet(extended)
    status = decode_ego_status(packet)

    assert len(packet) == size
    assert status.stamp == (12, 34)
    assert status.signed_velocity == pytest.approx(27.5)
    assert status.link_id == "LINK_EGO"
    assert len(status.tire_metrics) == (12 if extended else 0)


@pytest.mark.parametrize(
    "builder,decoder,header,numeric_offset",
    [
        (_collision_packet, decode_collisions, b"#CollisionData$", 12),
        (lambda: _ego_packet(False), decode_ego_status, b"#MoraiInfo$", 10),
    ],
)
def test_bad_header_tail_declared_length_and_nonfinite_are_rejected(
    builder, decoder, header, numeric_offset
):
    good = builder()
    mutations = (
        lambda p: p.__setitem__(slice(0, 1), b"!"),
        lambda p: p.__setitem__(slice(-2, None), b"xx"),
        lambda p: struct.pack_into("<i", p, len(header), 1),
        lambda p: struct.pack_into("<f", p, len(header) + 16 + numeric_offset, math.nan),
    )

    for mutation in mutations:
        packet = bytearray(good)
        mutation(packet)
        with pytest.raises(PacketFormatError):
            decoder(bytes(packet))


@pytest.mark.parametrize(
    "builder,decoder,header,expected_sec",
    [
        (_collision_packet, decode_collisions, b"#CollisionData$", 3),
        (lambda: _ego_packet(False), decode_ego_status, b"#MoraiInfo$", 12),
    ],
)
def test_invalid_device_timestamp_is_preserved_for_policy_fallback_and_audit(
    builder, decoder, header, expected_sec
):
    packet = bytearray(builder())
    struct.pack_into("<i", packet, len(header) + 16 + 4, -1)

    record = decoder(bytes(packet))

    assert record.stamp == (expected_sec, -1)
