import math
import struct
from collections.abc import Iterable, Sequence


TAIL = b"\r\n"


class PacketFormatError(ValueError):
    """Raised when a MORAI datagram does not match its declared wire format."""

    def __init__(self, stream: str, reason: str):
        super().__init__(f"{stream}: {reason}")
        self.stream = stream
        self.reason = reason


def decode_c_string(value: bytes) -> str:
    return value.split(b"\0", 1)[0].decode("utf-8", errors="replace")


def encode_c_string(stream: str, field: str, value: object, width: int) -> bytes:
    encoded = str(value).encode("utf-8")
    if len(encoded) > width:
        raise PacketFormatError(stream, f"{field} exceeds {width} bytes")
    return encoded.ljust(width, b"\0")


def require_finite(stream: str, values: Iterable[float]) -> None:
    if not all(math.isfinite(float(value)) for value in values):
        raise PacketFormatError(stream, "non-finite numeric value")


def require_ros_stamp(stream: str, stamp: tuple[int, int]) -> None:
    sec, nsec = stamp
    if sec < 0 or not 0 <= nsec < 1_000_000_000:
        raise PacketFormatError(stream, f"invalid ROS timestamp ({sec}, {nsec})")


def decode_envelope(
    stream: str,
    packet: bytes,
    headers: bytes | Sequence[bytes],
    expected_sizes: int | Sequence[int],
) -> tuple[bytes, tuple[int, int, int]]:
    header_values = (headers,) if isinstance(headers, bytes) else tuple(headers)
    sizes = (expected_sizes,) if isinstance(expected_sizes, int) else tuple(expected_sizes)
    if len(packet) not in sizes:
        raise PacketFormatError(stream, f"size {len(packet)} not in {sizes}")
    header = next((value for value in header_values if packet.startswith(value)), None)
    if header is None:
        raise PacketFormatError(stream, "invalid header")
    if packet[-2:] != TAIL:
        raise PacketFormatError(stream, "invalid tail")
    if len(packet) < len(header) + 18:
        raise PacketFormatError(stream, "truncated envelope")
    declared, a0, a1, a2 = struct.unpack_from("<i3i", packet, len(header))
    payload = packet[len(header) + 16:-2]
    if declared != len(payload):
        raise PacketFormatError(stream, f"declared payload {declared} != {len(payload)}")
    return payload, (a0, a1, a2)


def encode_envelope(
    header: bytes,
    payload: bytes,
    aux: tuple[int, int, int] = (0, 0, 0),
) -> bytes:
    return header + struct.pack("<i3i", len(payload), *aux) + payload + TAIL
