from dataclasses import dataclass, field
import struct

from .common import PacketFormatError
from ..protocol_records import CameraFrameRecord


_IMAGE_PACKET_SIZE = 65_000
_IMAGE_DATA_SIZE = 64_979


@dataclass
class _IncompleteFrame:
    created_at: float
    fragments: dict[int, bytes] = field(default_factory=dict)
    final_index: int | None = None
    total_bytes: int = 0


class CameraAssembler:
    def __init__(
        self,
        timeout_sec: float = 0.5,
        max_frame_bytes: int = 16 * 1024 * 1024,
        max_fragment_index: int = 1_024,
        max_incomplete_frames: int = 32,
    ):
        self.timeout_sec = float(timeout_sec)
        self.max_frame_bytes = int(max_frame_bytes)
        self.max_fragment_index = int(max_fragment_index)
        self.max_incomplete_frames = int(max_incomplete_frames)
        self._frames: dict[tuple[int, int], _IncompleteFrame] = {}

    @property
    def incomplete_frames(self) -> int:
        return len(self._frames)

    def _expire(self, now: float) -> None:
        expired = [
            stamp
            for stamp, frame in self._frames.items()
            if now - frame.created_at > self.timeout_sec
        ]
        for stamp in expired:
            del self._frames[stamp]

    def push(self, packet: bytes, now: float) -> CameraFrameRecord | None:
        self._expire(now)
        if len(packet) != _IMAGE_PACKET_SIZE or packet[:3] != b"MOR":
            raise PacketFormatError(
                "camera",
                f"invalid image packet size/header ({len(packet)})",
            )
        sec, nsec, index, size = struct.unpack_from("<iiii", packet, 3)
        if not 0 <= index <= self.max_fragment_index or not 0 <= size <= _IMAGE_DATA_SIZE:
            raise PacketFormatError("camera", "invalid fragment index or size")
        stamp = (sec, nsec)
        data = packet[19:19 + size]
        if stamp not in self._frames and len(self._frames) >= self.max_incomplete_frames:
            raise PacketFormatError("camera", "too many incomplete frames")
        frame = self._frames.setdefault(stamp, _IncompleteFrame(now))
        previous = frame.fragments.get(index)
        if previous is not None and previous != data:
            del self._frames[stamp]
            raise PacketFormatError("camera", f"conflicting duplicate fragment {index}")
        if previous is None:
            frame.fragments[index] = data
            frame.total_bytes += len(data)
        if frame.total_bytes > self.max_frame_bytes:
            del self._frames[stamp]
            raise PacketFormatError("camera", "assembled frame exceeds memory cap")
        if packet[-2:] == b"EI":
            if frame.final_index is not None and frame.final_index != index:
                del self._frames[stamp]
                raise PacketFormatError("camera", "conflicting final fragment")
            frame.final_index = index
        if frame.final_index is None or any(
            fragment_index not in frame.fragments
            for fragment_index in range(frame.final_index + 1)
        ):
            return None
        jpeg = b"".join(frame.fragments[i] for i in range(frame.final_index + 1))
        del self._frames[stamp]
        if not (jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9")):
            raise PacketFormatError("camera", "assembled payload is not a complete JPEG")
        return CameraFrameRecord(stamp, jpeg, frame.created_at)
