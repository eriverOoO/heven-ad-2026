import struct

import pytest

from ad_morai_bridge.codecs.common import PacketFormatError
from ad_morai_bridge.protocol_records import EgoStatusRecord
from ad_morai_bridge_dev.codecs import decode_ego_status, decode_objects


def _object_packet() -> bytes:
    item = struct.pack(
        "<hh16f38s",
        42,
        1,
        1.0,
        2.0,
        3.0,
        90.0,
        4.0,
        2.0,
        1.5,
        0.8,
        2.7,
        0.9,
        12.0,
        0.5,
        0.0,
        0.1,
        0.2,
        0.0,
        b"LINK_42",
    )
    empty = struct.pack("<hh16f38s", 0, -1, *(0.0,) * 16, b"")
    payload = struct.pack("<ii", 10, 20) + item + empty * 19
    return (
        b"#MoraiObjInfo$"
        + struct.pack("<i3i", len(payload), 0, 0, 0)
        + payload
        + b"\r\n"
    )


def _ego_packet(extended: bool) -> bytes:
    base = struct.pack(
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
    payload = base + (struct.pack("<12f", *range(12)) if extended else b"")
    return (
        b"#MoraiInfo$"
        + struct.pack("<i3i", len(payload), 0, 0, 0)
        + payload
        + b"\r\n"
    )


def test_dev_object_info_decodes_one_of_twenty_official_slots() -> None:
    packet = _object_packet()
    record = decode_objects(packet)

    assert len(packet) == 2160
    assert record.stamp == (10, 20)
    assert len(record.objects) == 1
    assert record.objects[0].unique_id == 42
    assert record.objects[0].object_type == 1
    assert record.objects[0].link_id == "LINK_42"


@pytest.mark.parametrize("extended,size", [(False, 181), (True, 229)])
def test_dev_ego_status_reuses_shared_decoder_for_both_official_forms(
    extended: bool, size: int
) -> None:
    packet = _ego_packet(extended)
    record = decode_ego_status(packet)

    assert len(packet) == size
    assert isinstance(record, EgoStatusRecord)
    assert record.signed_velocity == pytest.approx(27.5)
    assert record.link_id == "LINK_EGO"
    assert len(record.tire_metrics) == (12 if extended else 0)


def test_dev_object_info_rejects_invalid_envelope_and_numeric_fields() -> None:
    for mutation in (
        lambda packet: packet.__setitem__(slice(0, 1), b"!"),
        lambda packet: packet.__setitem__(slice(-2, None), b"xx"),
        lambda packet: struct.pack_into("<i", packet, 14, 1),
        lambda packet: struct.pack_into("<f", packet, 42, float("nan")),
    ):
        packet = bytearray(_object_packet())
        mutation(packet)
        with pytest.raises(PacketFormatError):
            decode_objects(bytes(packet))


def test_dev_object_info_preserves_invalid_raw_source_stamp_for_policy_audit():
    packet = bytearray(_object_packet())
    struct.pack_into("<i", packet, 34, -1)

    record = decode_objects(bytes(packet))

    assert record.stamp == (10, -1)
