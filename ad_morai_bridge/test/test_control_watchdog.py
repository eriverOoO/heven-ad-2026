import inspect
import math

import pytest

from ad_morai_bridge.control_watchdog import ControlSafetyGate
from ad_morai_bridge.protocol_records import CtrlCommandRecord


def test_startup_is_silent_and_stale_command_becomes_periodic_full_brake():
    sent = []
    gate = ControlSafetyGate(sent.append, timeout_sec=0.2, fallback_interval_sec=0.1)

    assert gate.tick(10.0) is False
    assert sent == []

    command = CtrlCommandRecord(accel=0.2, brake=0.0, steering=0.1)
    gate.accept(command, now=10.0)
    assert sent == [command]
    assert gate.tick(10.19) is False
    assert gate.tick(10.21) is True
    assert sent[-1] == CtrlCommandRecord(accel=0.0, brake=1.0, steering=0.0)
    assert sent[-1].long_cmd_type == 1
    assert sent[-1].velocity == 0.0
    assert sent[-1].acceleration == 0.0
    assert gate.tick(10.25) is False
    assert gate.tick(10.32) is True


def test_stale_fallback_triggers_at_the_exact_timeout_boundary():
    sent = []
    gate = ControlSafetyGate(sent.append, timeout_sec=0.2)
    command = CtrlCommandRecord(accel=0.2)
    gate.accept(command, now=0.0)

    assert gate.tick(0.199) is False
    assert gate.tick(0.2) is True
    assert sent == [command, CtrlCommandRecord(brake=1.0)]


def test_disable_restores_startup_silence_until_new_command():
    sent = []
    gate = ControlSafetyGate(sent.append)
    gate.accept(CtrlCommandRecord(), now=1.0)

    gate.disable()

    assert gate.tick(100.0) is False
    assert len(sent) == 1


def test_emergency_stop_transmits_bounded_burst_only_after_gate_is_armed():
    sent = []
    gate = ControlSafetyGate(sent.append)

    assert not inspect.signature(gate.emergency_stop).parameters
    assert gate.emergency_stop() is False
    assert sent == []
    gate.accept(CtrlCommandRecord(ctrl_mode=2, gear=4, accel=0.3), now=1.0)
    assert gate.emergency_stop() is True

    assert len(sent) == 4
    assert sent[-3:] == [CtrlCommandRecord(brake=1.0)] * 3
    assert all(command.long_cmd_type == 1 for command in sent[-3:])
    assert all(command.velocity == 0.0 for command in sent[-3:])
    assert all(command.acceleration == 0.0 for command in sent[-3:])
    assert gate.tick(100.0) is False


@pytest.mark.parametrize("field", ["timeout_sec", "fallback_interval_sec"])
@pytest.mark.parametrize("value", [0.0, -0.1, math.nan, math.inf, -math.inf])
def test_gate_requires_finite_positive_intervals(field, value):
    kwargs = {field: value}

    with pytest.raises(ValueError):
        ControlSafetyGate(lambda _: None, **kwargs)


@pytest.mark.parametrize("now", [math.nan, math.inf, -math.inf])
def test_accept_rejects_nonfinite_time_without_sending_or_arming(now):
    sent = []
    gate = ControlSafetyGate(sent.append)

    with pytest.raises(ValueError):
        gate.accept(CtrlCommandRecord(accel=0.2), now=now)

    assert sent == []
    assert gate.tick(10.0) is False


@pytest.mark.parametrize("now", [math.nan, math.inf, -math.inf])
def test_tick_rejects_nonfinite_time_without_sending_fallback(now):
    sent = []
    gate = ControlSafetyGate(sent.append)
    gate.accept(CtrlCommandRecord(accel=0.2), now=10.0)

    with pytest.raises(ValueError):
        gate.tick(now)

    assert len(sent) == 1


def test_gate_rejects_backward_accept_and_tick_times():
    sent = []
    gate = ControlSafetyGate(sent.append)
    gate.accept(CtrlCommandRecord(accel=0.2), now=10.0)

    with pytest.raises(ValueError):
        gate.tick(9.9)
    with pytest.raises(ValueError):
        gate.accept(CtrlCommandRecord(accel=0.1), now=9.9)

    assert len(sent) == 1


def test_new_command_cannot_precede_last_fallback_time():
    sent = []
    gate = ControlSafetyGate(sent.append)
    gate.accept(CtrlCommandRecord(accel=0.2), now=10.0)
    assert gate.tick(10.21) is True

    with pytest.raises(ValueError):
        gate.accept(CtrlCommandRecord(accel=0.1), now=10.2)

    assert len(sent) == 2


@pytest.mark.parametrize("stop_method", ["disable", "emergency_stop"])
def test_disarming_preserves_time_watermark_and_forward_time_can_rearm(stop_method):
    sent = []
    gate = ControlSafetyGate(sent.append)
    gate.accept(CtrlCommandRecord(accel=0.2), now=10.0)
    getattr(gate, stop_method)()

    with pytest.raises(ValueError):
        gate.tick(9.9)
    with pytest.raises(ValueError):
        gate.accept(CtrlCommandRecord(accel=0.1), now=9.9)

    gate.accept(CtrlCommandRecord(accel=0.1), now=10.1)
    assert sent[-1] == CtrlCommandRecord(accel=0.1)
