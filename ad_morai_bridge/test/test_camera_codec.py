import struct

import pytest

from ad_morai_bridge.codecs.camera import CameraAssembler
from ad_morai_bridge.codecs.common import PacketFormatError


def _image_packet(index: int, data: bytes, final: bool, stamp=(10, 20)) -> bytes:
    assert len(data) <= 64_979
    packet = (
        b"MOR"
        + struct.pack("<iiii", stamp[0], stamp[1], index, len(data))
        + data.ljust(64_979, b"\0")
        + (b"EI" if final else b"AI")
    )
    assert len(packet) == 65_000
    return packet


def test_camera_assembles_out_of_order_65000_byte_fragments_and_uses_size():
    assembler = CameraAssembler()
    jpeg = b"\xff\xd8hello-world\xff\xd9"

    assert assembler.push(_image_packet(1, jpeg[6:], True), now=1.0) is None
    result = assembler.push(_image_packet(0, jpeg[:6], False), now=1.1)

    assert result is not None
    assert result.stamp == (10, 20)
    assert result.jpeg == jpeg
    assert result.first_arrived == pytest.approx(1.0)


def test_camera_rejects_conflicting_duplicates_and_bad_jpeg():
    assembler = CameraAssembler()
    assembler.push(_image_packet(0, b"\xff\xd8a", False), now=1.0)

    with pytest.raises(PacketFormatError):
        assembler.push(_image_packet(0, b"\xff\xd8b", False), now=1.1)
    with pytest.raises(PacketFormatError):
        CameraAssembler().push(_image_packet(0, b"not-jpeg", True), now=1.0)


def test_camera_expires_incomplete_frames_and_caps_memory():
    assembler = CameraAssembler(timeout_sec=0.5, max_frame_bytes=8)
    assembler.push(_image_packet(0, b"\xff\xd8a", False, (1, 1)), now=0.0)
    assembler.push(_image_packet(0, b"\xff\xd8b", False, (2, 2)), now=1.0)

    assert assembler.incomplete_frames == 1
    with pytest.raises(PacketFormatError):
        assembler.push(_image_packet(1, b"123456", False, (2, 2)), now=1.1)


def test_camera_caps_fragment_index_and_incomplete_frame_count():
    assembler = CameraAssembler(max_fragment_index=2, max_incomplete_frames=2)

    with pytest.raises(PacketFormatError):
        assembler.push(_image_packet(3, b"", False), now=1.0)
    assembler.push(_image_packet(0, b"", False, (1, 1)), now=1.0)
    assembler.push(_image_packet(0, b"", False, (2, 2)), now=1.0)
    with pytest.raises(PacketFormatError):
        assembler.push(_image_packet(0, b"", False, (3, 3)), now=1.0)


@pytest.mark.parametrize("stamp", [(-1, 0), (0, -1), (0, 1_000_000_000)])
def test_camera_preserves_invalid_frame_stamp_for_policy_fallback_and_audit(stamp):
    result = CameraAssembler().push(
        _image_packet(0, b"\xff\xd8\xff\xd9", True, stamp), now=1.0
    )

    assert result is not None
    assert result.stamp == stamp
