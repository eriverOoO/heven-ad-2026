from dataclasses import dataclass
import struct
from collections.abc import Mapping

from ad_morai_bridge.codecs.common import (
    PacketFormatError,
    decode_c_string,
    decode_envelope,
    encode_c_string,
    encode_envelope,
    require_finite,
)


@dataclass(frozen=True)
class CommandLayout:
    header: bytes
    fields: tuple[str, ...]
    fmt: str
    string_widths: Mapping[str, int] | None = None
    string_padding: bytes = b"\0"


COMMAND_LAYOUTS = {
    "ego_ghost": CommandLayout(
        b"#EgoGhostCmd$",
        (
            "pose_x",
            "pose_y",
            "pose_z",
            "roll",
            "pitch",
            "yaw",
            "velocity",
            "steering",
        ),
        "<8f",
    ),
    "traffic_light": CommandLayout(
        b"#TrafficLight$",
        ("traffic_light_index", "traffic_light_status"),
        "<12sh",
        {"traffic_light_index": 12},
    ),
    "intersection": CommandLayout(
        b"#SetIntStatus$",
        ("intersection_index", "intersection_status", "intersection_time"),
        "<hhf",
    ),
    "sensor_control": CommandLayout(
        b"#SensorControl$",
        ("sensor_index", "pose_x", "pose_y", "pose_z", "roll", "pitch", "yaw"),
        "<h6f",
    ),
    "lamp_control": CommandLayout(
        b"#LampControl$", ("turn_signal", "emergency_signal"), "<bb"
    ),
    "scenario_load": CommandLayout(
        b"#ScenarioLoad$",
        (
            "filename",
            "delete_all",
            "network",
            "ego",
            "npc",
            "pedestrian",
            "object",
            "pause",
        ),
        "<30s7?",
        {"filename": 30},
        b" ",
    ),
    "save_sensor_data": CommandLayout(
        b"#SaveSensorData$",
        ("custom", "file_name", "file_dir"),
        "<?30s60s",
        {"file_name": 30, "file_dir": 60},
    ),
}

def _simple_payload(
    kind: str, layout: CommandLayout, values: Mapping[str, object]
) -> bytes:
    packed = []
    strings = layout.string_widths or {}
    for field in layout.fields:
        if field not in values:
            if field in strings:
                value: object = ""
            elif field == "pause":
                value = False
            else:
                raise PacketFormatError(kind, f"missing field {field}")
        else:
            value = values[field]
        if field in strings:
            encoded = encode_c_string(kind, field, value, strings[field])
            if layout.string_padding != b"\0":
                encoded = encoded.rstrip(b"\0").ljust(
                    strings[field], layout.string_padding
                )
            packed.append(encoded)
        else:
            packed.append(value)
    require_finite(
        kind,
        (value for value in packed if not isinstance(value, bytes)),
    )
    try:
        return struct.pack(layout.fmt, *packed)
    except (struct.error, TypeError, ValueError) as exc:
        raise PacketFormatError(kind, f"invalid field value: {exc}") from exc


def _encode_multi_ego(values: Mapping[str, object]) -> bytes:
    entries = list(values.get("vehicles", []))
    if len(entries) > 20:
        raise PacketFormatError("multi_ego", "at most 20 vehicles are supported")
    number_of_ego = int(values.get("number_of_ego", len(entries)))
    if not 0 <= number_of_ego <= 20:
        raise PacketFormatError("multi_ego", "number_of_ego is outside 0..20")
    if number_of_ego != len(entries):
        raise PacketFormatError(
            "multi_ego", f"number_of_ego {number_of_ego} != {len(entries)} vehicles"
        )
    record = struct.Struct("<h7fbb")
    payload = struct.pack(
        "<ii",
        number_of_ego,
        int(values.get("camera_index", 0)),
    )
    fields = (
        "ego_index",
        "position_x",
        "position_y",
        "position_z",
        "roll",
        "pitch",
        "yaw",
        "velocity",
        "gear",
        "ctrl_mode",
    )
    for item in entries:
        item_values = tuple(item.get(name, 0) for name in fields)
        require_finite("multi_ego", item_values)
        payload += record.pack(*item_values)
    return payload + b"\0" * (record.size * (20 - len(entries)))


def _encode_npc_ghost(values: Mapping[str, object]) -> bytes:
    entries = list(values.get("vehicles", []))
    if len(entries) > 20:
        raise PacketFormatError("npc_ghost", "at most 20 vehicles are supported")
    record = struct.Struct("<b25s6f")
    payload = b""
    for item in entries:
        pose = tuple(
            float(item.get(name, 0.0))
            for name in ("pose_x", "pose_y", "pose_z", "roll", "pitch", "yaw")
        )
        require_finite("npc_ghost", pose)
        payload += record.pack(
            int(item.get("unique_id", 0)),
            encode_c_string(
                "npc_ghost", "car_name", item.get("car_name", ""), 25
            ),
            *pose,
        )
    return payload + b"\0" * (record.size * (20 - len(entries)))


def encode_simulator_command(kind: str, values: Mapping[str, object]) -> bytes:
    try:
        if kind == "multi_ego":
            return encode_envelope(b"#MultiEgoSetting$", _encode_multi_ego(values))
        if kind == "npc_ghost":
            return encode_envelope(b"#NpcGhostCmd$", _encode_npc_ghost(values))
        layout = COMMAND_LAYOUTS.get(kind)
        if layout is None:
            raise PacketFormatError(kind, "unknown simulator command")
        return encode_envelope(layout.header, _simple_payload(kind, layout, values))
    except PacketFormatError:
        raise
    except (AttributeError, OverflowError, TypeError, ValueError, struct.error) as exc:
        raise PacketFormatError(kind, f"invalid command payload: {exc}") from exc


def decode_known_simulator_packet(kind: str, packet: bytes) -> dict[str, object]:
    if kind == "traffic_light_status":
        payload, _ = decode_envelope(kind, packet, b"#TrafficLight$", 48)
        index, light_type, status = struct.unpack("<12shh", payload)
        return {
            "traffic_light_index": decode_c_string(index),
            "traffic_light_type": light_type,
            "traffic_light_status": status,
        }
    if kind == "intersection_status":
        payload, _ = decode_envelope(
            kind, packet, (b"#IntStatus$", b"#Intersection"), (37, 39)
        )
        if len(payload) != struct.calcsize("<hhf"):
            raise PacketFormatError(kind, f"payload size {len(payload)} != 8")
        index, status, remaining = struct.unpack("<hhf", payload)
        require_finite(kind, (remaining,))
        return {
            "intersection_index": index,
            "intersection_status": status,
            "remaining_time": remaining,
        }
    raise PacketFormatError(kind, "unknown simulator packet")
