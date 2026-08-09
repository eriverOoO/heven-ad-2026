import struct

from ad_morai_bridge.codecs.common import (
    PacketFormatError,
    require_finite,
)

from ad_morai_bridge_dev.bridge.protocol_records import (
    CameraBoundingBoxArrayRecord,
    CameraBoundingBoxRecord,
)


_HEADER = struct.Struct("<iiii")
_BOUNDING_BOX = struct.Struct("<28fBBB")
_MAX_BOXES = 575


def decode_camera_bboxes(packet: bytes) -> CameraBoundingBoxArrayRecord:
    stream = "camera_bbox"
    if not packet.startswith(b"BOX"):
        raise PacketFormatError(stream, "invalid header")
    if packet[-2:] != b"\r\n":
        raise PacketFormatError(stream, "invalid tail")
    if len(packet) < 3 + _HEADER.size + 2:
        raise PacketFormatError(stream, "truncated packet")

    sec, nsec, _sequence, count = _HEADER.unpack_from(packet, 3)
    if not 0 <= count <= _MAX_BOXES:
        raise PacketFormatError(stream, f"box count {count} outside 0..{_MAX_BOXES}")
    expected_size = 3 + _HEADER.size + count * _BOUNDING_BOX.size + 2
    if len(packet) != expected_size:
        raise PacketFormatError(
            stream, f"size {len(packet)} != count-derived size {expected_size}"
        )

    boxes = []
    offset = 3 + _HEADER.size
    for _ in range(count):
        values = _BOUNDING_BOX.unpack_from(packet, offset)
        offset += _BOUNDING_BOX.size
        floats = values[:28]
        require_finite(stream, floats)
        boxes.append(
            CameraBoundingBoxRecord(
                corners_3d=tuple(floats[:24]),
                bounding_box_2d=tuple(floats[24:]),
                group=values[28],
                class_id=values[29],
                subclass_id=values[30],
            )
        )
    return CameraBoundingBoxArrayRecord((sec, nsec), tuple(boxes))
