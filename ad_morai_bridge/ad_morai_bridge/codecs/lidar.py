from .common import PacketFormatError


def validate_velodyne_packet(packet: bytes) -> bytes:
    if not isinstance(packet, bytes):
        raise PacketFormatError("velodyne", "packet must be bytes")
    if len(packet) != 1_206:
        raise PacketFormatError("velodyne", f"size {len(packet)} != 1206")
    for offset in range(0, 1_200, 100):
        if packet[offset:offset + 2] != b"\xff\xee":
            raise PacketFormatError("velodyne", f"invalid data block flag at {offset}")
        azimuth = int.from_bytes(packet[offset + 2:offset + 4], "little")
        if not 0 <= azimuth < 36_000:
            raise PacketFormatError(
                "velodyne", f"invalid data block azimuth at {offset}: {azimuth}"
            )
    return packet


def first_block_azimuth_hundredths(packet: bytes) -> int:
    validated = validate_velodyne_packet(packet)
    return int.from_bytes(validated[2:4], "little")
