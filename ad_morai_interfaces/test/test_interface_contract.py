"""Keep MORAI competition transport constants and core types stable."""

from ad_morai_interfaces.msg import (
    BridgeStatistics,
    CtrlCmd,
    EgoVehicleStatus,
    RawPacket,
    SensorTiming,
)


def test_ctrl_cmd_constants_match_the_morai_wire_contract():
    assert CtrlCmd.CTRL_MODE_AUTO == 2
    assert CtrlCmd.LONG_CMD_THROTTLE == 1
    assert CtrlCmd.LONG_CMD_VELOCITY == 2
    assert CtrlCmd.GEAR_DRIVE == 4


def test_core_messages_construct():
    assert CtrlCmd() is not None
    assert EgoVehicleStatus() is not None
    assert RawPacket() is not None
    assert SensorTiming() is not None
    assert SensorTiming().source_valid is False
    statistics = BridgeStatistics()
    assert statistics.source_selected == 0
    assert statistics.arrival_fallback == 0
    assert statistics.source_rejected == 0
    assert statistics.duplicates == 0
    assert statistics.stamp_regressions == 0
