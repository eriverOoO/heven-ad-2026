from decimal import Decimal
import struct

import pytest

from ad_morai_bridge.codecs.common import (
    PacketFormatError,
    decode_envelope,
    encode_c_string,
    encode_envelope,
)
from ad_morai_bridge_dev.codecs.simulator import (
    decode_known_simulator_packet,
    encode_simulator_command,
)


COMMAND_CASES = (
    (
        "ego_ghost",
        {
            "pose_x": 1,
            "pose_y": 2,
            "pose_z": 3,
            "roll": 0,
            "pitch": 0,
            "yaw": 1,
            "velocity": 5,
            "steering": 0.1,
        },
        b"#EgoGhostCmd$",
        63,
    ),
    (
        "traffic_light",
        {"traffic_light_index": "C119BS010001", "traffic_light_status": 16},
        b"#TrafficLight$",
        46,
    ),
    (
        "intersection",
        {
            "intersection_index": 1,
            "intersection_status": 4,
            "intersection_time": 3.5,
        },
        b"#SetIntStatus$",
        40,
    ),
    (
        "sensor_control",
        {
            "sensor_index": 2,
            "pose_x": 1,
            "pose_y": 2,
            "pose_z": 3,
            "roll": 0,
            "pitch": 0,
            "yaw": 90,
        },
        b"#SensorControl$",
        59,
    ),
    (
        "lamp_control",
        {"turn_signal": 1, "emergency_signal": 0},
        b"#LampControl$",
        33,
    ),
    (
        "scenario_load",
        {
            "filename": "9_to_14.json",
            "delete_all": True,
            "network": True,
            "ego": True,
            "npc": True,
            "pedestrian": True,
            "object": True,
            "pause": False,
        },
        b"#ScenarioLoad$",
        69,
    ),
    (
        "save_sensor_data",
        {"custom": True, "file_name": "smoke", "file_dir": "/tmp"},
        b"#SaveSensorData$",
        125,
    ),
    (
        "multi_ego",
        {"vehicles": [{"ego_index": 1, "velocity": 3.0}]},
        b"#MultiEgoSetting$",
        683,
    ),
    (
        "npc_ghost",
        {"vehicles": [{"unique_id": 1, "car_name": "2016_Kia_Niro"}]},
        b"#NpcGhostCmd$",
        1031,
    ),
)


@pytest.mark.parametrize("kind,values,header,expected_size", COMMAND_CASES)
def test_dev_nine_simulator_commands_match_official_header_and_size(
    kind: str,
    values: dict[str, object],
    header: bytes,
    expected_size: int,
) -> None:
    packet = encode_simulator_command(kind, values)

    assert len(packet) == expected_size
    assert packet.startswith(header)
    assert packet.endswith(b"\r\n")


def test_dev_traffic_light_status_decodes_official_fields() -> None:
    packet = encode_envelope(
        b"#TrafficLight$", struct.pack("<12shh", b"C119BS010001", 2, 16)
    )

    assert len(packet) == 48
    assert decode_known_simulator_packet("traffic_light_status", packet) == {
        "traffic_light_index": "C119BS010001",
        "traffic_light_type": 2,
        "traffic_light_status": 16,
    }


def test_scenario_load_filename_uses_official_space_padding() -> None:
    packet = encode_simulator_command("scenario_load", COMMAND_CASES[5][1])
    payload, _ = decode_envelope("scenario_load", packet, b"#ScenarioLoad$", 69)

    assert struct.unpack("<30s7?", payload)[0] == b"9_to_14.json                  "


def test_dev_intersection_status_decodes_official_fields() -> None:
    packet = encode_envelope(b"#IntStatus$", struct.pack("<hhf", 1, 4, 3.5))
    result = decode_known_simulator_packet("intersection_status", packet)

    assert len(packet) == 37
    assert result["intersection_index"] == 1
    assert result["intersection_status"] == 4
    assert result["remaining_time"] == pytest.approx(3.5)


def test_dev_unknown_command_and_oversized_shared_c_string_are_rejected() -> None:
    with pytest.raises(PacketFormatError):
        encode_simulator_command("not_real", {})
    with pytest.raises(PacketFormatError):
        encode_c_string("scenario_load", "filename", "x" * 31, 30)
    with pytest.raises(PacketFormatError):
        encode_simulator_command("scenario_load", {"filename": "x" * 31})


@pytest.mark.parametrize("kind", ("multi_ego", "npc_ghost"))
def test_dev_complex_commands_reject_more_than_twenty_entries(kind: str) -> None:
    with pytest.raises(PacketFormatError):
        encode_simulator_command(kind, {"vehicles": [{}] * 21})


@pytest.mark.parametrize("declared", (-1, 21))
def test_dev_multi_ego_rejects_declared_count_outside_supported_range(
    declared: int,
) -> None:
    with pytest.raises(PacketFormatError, match="number_of_ego is outside 0..20"):
        encode_simulator_command(
            "multi_ego", {"number_of_ego": declared, "vehicles": []}
        )


def test_dev_multi_ego_rejects_declared_count_that_differs_from_entries() -> None:
    with pytest.raises(PacketFormatError, match="number_of_ego 0 != 1 vehicles"):
        encode_simulator_command(
            "multi_ego", {"number_of_ego": 0, "vehicles": [{}]}
        )


def test_dev_simulator_commands_reject_nonfinite_numeric_fields() -> None:
    values = dict(COMMAND_CASES[0][1])
    values["velocity"] = float("nan")
    with pytest.raises(PacketFormatError):
        encode_simulator_command("ego_ghost", values)


def test_dev_simulator_commands_reject_decimal_nan() -> None:
    values = dict(COMMAND_CASES[0][1])
    values["velocity"] = Decimal("NaN")
    with pytest.raises(PacketFormatError):
        encode_simulator_command("ego_ghost", values)
