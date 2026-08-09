import pytest

from ad_morai_bridge.codecs.common import PacketFormatError
from ad_morai_bridge.codecs.lidar import validate_velodyne_packet


def test_velodyne_packet_requires_1206_bytes_and_12_data_block_flags():
    packet = bytearray(1_206)
    for offset in range(0, 1_200, 100):
        packet[offset : offset + 2] = b"\xff\xee"

    assert validate_velodyne_packet(bytes(packet)) == bytes(packet)
    packet[100:102] = b"xx"
    with pytest.raises(PacketFormatError):
        validate_velodyne_packet(bytes(packet))
    with pytest.raises(PacketFormatError):
        validate_velodyne_packet(b"short")
