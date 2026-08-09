import struct

from ad_morai_bridge.codecs.common import (
    PacketFormatError,
    decode_envelope,
    require_finite,
)

from ad_morai_bridge_dev.bridge.protocol_records import Lidar2DRecord


def decode_lidar2d(packet: bytes) -> Lidar2DRecord:
    payload, _ = decode_envelope("lidar2d", packet, b"#Lidar2D$", 1107)
    aux = struct.unpack_from("<3f", packet, 13)
    require_finite("lidar2d", aux)
    distances = []
    intensities = []
    for offset in range(0, len(payload), 3):
        distance_mm, intensity = struct.unpack_from("<HB", payload, offset)
        distances.append(distance_mm / 1000.0)
        intensities.append(intensity)
    if len(distances) != 360:
        raise PacketFormatError("lidar2d", f"range count {len(distances)} != 360")
    return Lidar2DRecord(tuple(aux), tuple(distances), tuple(intensities))
