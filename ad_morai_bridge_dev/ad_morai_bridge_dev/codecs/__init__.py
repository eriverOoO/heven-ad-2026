import struct

from ad_morai_bridge.codecs.common import (
    decode_c_string,
    decode_envelope,
    require_finite,
)
from ad_morai_bridge.codecs.ego import decode_ego_status

from ad_morai_bridge_dev.bridge.protocol_records import ObjectArrayRecord, ObjectRecord


_OBJECT = struct.Struct("<hh16f38s")


def decode_objects(packet: bytes) -> ObjectArrayRecord:
    payload, _ = decode_envelope("objects", packet, b"#MoraiObjInfo$", 2160)
    sec, nsec = struct.unpack_from("<ii", payload)
    objects = []
    offset = 8
    for _ in range(20):
        values = _OBJECT.unpack_from(payload, offset)
        offset += _OBJECT.size
        unique_id, object_type = values[:2]
        floats = values[2:18]
        require_finite("objects", floats)
        if object_type < 0 or (unique_id == 0 and object_type == 0):
            continue
        objects.append(
            ObjectRecord(
                unique_id=unique_id,
                object_type=object_type,
                position=tuple(floats[0:3]),
                heading=floats[3],
                size=tuple(floats[4:7]),
                overhang=floats[7],
                wheelbase=floats[8],
                rear_overhang=floats[9],
                velocity=tuple(floats[10:13]),
                acceleration=tuple(floats[13:16]),
                link_id=decode_c_string(values[18]),
            )
        )
    return ObjectArrayRecord((sec, nsec), tuple(objects))


__all__ = ("decode_ego_status", "decode_objects")
