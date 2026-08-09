import struct

from .common import (
    PacketFormatError,
    decode_c_string,
    decode_envelope,
    encode_envelope,
    require_finite,
)
from ..protocol_records import (
    CollisionArrayRecord,
    CollisionRecord,
    CtrlCommandRecord,
    EgoStatusRecord,
)


_CTRL = struct.Struct("<bbb5f")
_COLLISION = struct.Struct("<hh6f")
_EGO_BASE = struct.Struct("<ii2bfi2f3f3f3f3f3f3f3ff38s")
_TIRE = struct.Struct("<12f")


def encode_ctrl_cmd(command: CtrlCommandRecord) -> bytes:
    stream = "ctrl_cmd"
    if command.ctrl_mode not in (1, 2):
        raise PacketFormatError(stream, "ctrl_mode must be keyboard (1) or auto (2)")
    if command.gear not in range(0, 6):
        raise PacketFormatError(stream, "gear is outside 0..5")
    if command.long_cmd_type != 1:
        raise PacketFormatError(
            stream, "competition control accepts only throttle type (1)"
        )
    values = (
        command.velocity,
        command.acceleration,
        command.accel,
        command.brake,
        command.steering,
    )
    require_finite(stream, values)
    if not 0.0 <= command.accel <= 1.0 or not 0.0 <= command.brake <= 1.0:
        raise PacketFormatError(stream, "accel and brake must be in [0, 1]")
    if not -1.0 <= command.steering <= 1.0:
        raise PacketFormatError(stream, "steering must be in [-1, 1]")
    payload = _CTRL.pack(
        command.ctrl_mode,
        command.gear,
        command.long_cmd_type,
        command.velocity,
        command.acceleration,
        command.accel,
        command.brake,
        command.steering,
    )
    return encode_envelope(b"#MoraiCtrlCmd$", payload)


def decode_collisions(packet: bytes) -> CollisionArrayRecord:
    payload, _ = decode_envelope("collisions", packet, b"#CollisionData$", 181)
    sec, nsec = struct.unpack_from("<ii", payload)
    collisions = []
    offset = 8
    for _ in range(5):
        object_type, object_id, *floats = _COLLISION.unpack_from(payload, offset)
        offset += _COLLISION.size
        require_finite("collisions", floats)
        if object_type < 0 or (object_type == 0 and object_id == 0):
            continue
        collisions.append(
            CollisionRecord(object_type, object_id, tuple(floats[:3]), tuple(floats[3:]))
        )
    return CollisionArrayRecord((sec, nsec), tuple(collisions))


def decode_ego_status(packet: bytes) -> EgoStatusRecord:
    payload, _ = decode_envelope(
        "ego_status", packet, (b"#MoraiInfo$", b"#EgoStatus$"), (181, 229)
    )
    if len(payload) not in (_EGO_BASE.size, _EGO_BASE.size + _TIRE.size):
        raise PacketFormatError("ego_status", f"unsupported payload size {len(payload)}")
    values = _EGO_BASE.unpack_from(payload)
    sec, nsec, ctrl_mode, gear, signed_velocity, map_data_id = values[:6]
    floats = values[6:-1]
    require_finite("ego_status", (signed_velocity, *floats))
    tire_metrics = (
        _TIRE.unpack_from(payload, _EGO_BASE.size) if len(payload) > _EGO_BASE.size else ()
    )
    require_finite("ego_status", tire_metrics)
    return EgoStatusRecord(
        stamp=(sec, nsec),
        ctrl_mode=ctrl_mode,
        gear=gear,
        signed_velocity=signed_velocity,
        map_data_id=map_data_id,
        accel=floats[0],
        brake=floats[1],
        size=tuple(floats[2:5]),
        overhang=floats[5],
        wheelbase=floats[6],
        rear_overhang=floats[7],
        position=tuple(floats[8:11]),
        rpy=tuple(floats[11:14]),
        velocity=tuple(floats[14:17]),
        angular_velocity=tuple(floats[17:20]),
        acceleration=tuple(floats[20:23]),
        steering=floats[23],
        link_id=decode_c_string(values[-1]),
        tire_metrics=tuple(tire_metrics),
    )
