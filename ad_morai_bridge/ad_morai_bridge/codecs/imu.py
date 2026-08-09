import struct

from .common import PacketFormatError, decode_envelope, require_finite
from ..protocol_records import ImuRecord


_VALUES = struct.Struct("<10d")


def decode_imu(packet: bytes) -> ImuRecord:
    payload, _ = decode_envelope("imu", packet, b"#IMUData$", (107, 115))
    if len(payload) == _VALUES.size:
        stamp = None
        offset = 0
    elif len(payload) == _VALUES.size + 8:
        stamp = struct.unpack_from("<ii", payload)
        offset = 8
    else:
        raise PacketFormatError("imu", f"unsupported payload size {len(payload)}")
    values = _VALUES.unpack_from(payload, offset)
    require_finite("imu", values)
    w, x, y, z, avx, avy, avz, lax, lay, laz = values
    norm2 = w * w + x * x + y * y + z * z
    if norm2 < 1e-12:
        raise PacketFormatError("imu", "zero quaternion")
    return ImuRecord(stamp, (x, y, z, w), (avx, avy, avz), (lax, lay, laz))
