import math
import struct

import pytest

from ad_morai_bridge.codecs.common import PacketFormatError
from ad_morai_bridge_dev.codecs.camera import decode_camera_bboxes


def _bbox_packet(values=None, *, count=1) -> bytes:
    values = values or tuple(float(index) + 0.25 for index in range(28))
    records = b"".join(
        struct.pack("<28fBBB", *values, 7, 8, 9) for _ in range(count)
    )
    return b"BOX" + struct.pack("<iiii", 4, 5, 0, count) + records + b"\r\n"


def test_camera_bbox_preserves_all_coordinates_and_class_bytes():
    values = tuple(float(index) + 0.25 for index in range(28))
    record = decode_camera_bboxes(_bbox_packet(values))

    assert record.stamp == (4, 5)
    assert record.boxes[0].corners_3d == pytest.approx(values[:24])
    assert record.boxes[0].bounding_box_2d == pytest.approx(values[24:])
    assert (
        record.boxes[0].group,
        record.boxes[0].class_id,
        record.boxes[0].subclass_id,
    ) == (7, 8, 9)


def test_camera_bbox_rejects_count_size_and_nonfinite_values():
    with pytest.raises(PacketFormatError):
        decode_camera_bboxes(_bbox_packet()[:-1])

    invalid_count = bytearray(_bbox_packet())
    struct.pack_into("<i", invalid_count, 3 + 12, 576)
    with pytest.raises(PacketFormatError):
        decode_camera_bboxes(bytes(invalid_count))

    nonfinite = list(float(index) for index in range(28))
    nonfinite[17] = math.nan
    with pytest.raises(PacketFormatError):
        decode_camera_bboxes(_bbox_packet(tuple(nonfinite)))


def test_camera_bbox_preserves_invalid_raw_source_stamp_for_policy_audit():
    invalid_stamp = bytearray(_bbox_packet())
    struct.pack_into("<i", invalid_stamp, 3 + 4, 1_000_000_000)

    record = decode_camera_bboxes(bytes(invalid_stamp))

    assert record.stamp == (4, 1_000_000_000)
