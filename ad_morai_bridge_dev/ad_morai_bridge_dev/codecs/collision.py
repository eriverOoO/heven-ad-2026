import struct

from ad_morai_bridge.codecs.common import decode_envelope, require_finite

from ad_morai_bridge_dev.bridge.protocol_records import (
    VehicleCollisionArrayRecord,
    VehicleCollisionObjectRecord,
    VehicleCollisionRecord,
)


_COLLISION_PAIR = struct.Struct("<hh13fhh13f")


def _object(values) -> VehicleCollisionObjectRecord:
    object_type, object_id, *floats = values
    return VehicleCollisionObjectRecord(
        object_type=object_type,
        object_id=object_id,
        position=tuple(floats[0:3]),
        heading=floats[3],
        size=tuple(floats[4:7]),
        velocity=tuple(floats[7:10]),
        acceleration=tuple(floats[10:13]),
    )


def _empty(values) -> bool:
    object_type, object_id = values[:2]
    return object_type < 0 or (object_type == 0 and object_id == 0)


def decode_npc_collisions(packet: bytes) -> VehicleCollisionArrayRecord:
    payload, _ = decode_envelope(
        "npc_collisions", packet, b"#VehicleCollision$", 1156
    )
    collisions = []
    for offset in range(0, len(payload), _COLLISION_PAIR.size):
        values = _COLLISION_PAIR.unpack_from(payload, offset)
        first_values = values[:15]
        second_values = values[15:]
        require_finite(
            "npc_collisions", (*first_values[2:], *second_values[2:])
        )
        if _empty(first_values) and _empty(second_values):
            continue
        collisions.append(
            VehicleCollisionRecord(
                first=_object(first_values),
                second=_object(second_values),
            )
        )
    return VehicleCollisionArrayRecord(tuple(collisions))
