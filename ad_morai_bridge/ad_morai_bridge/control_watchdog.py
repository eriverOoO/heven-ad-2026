from collections.abc import Callable
import math

from .protocol_records import CtrlCommandRecord


class ControlSafetyGate:
    """Stay silent until armed, then send periodic full-brake fallbacks."""

    def __init__(
        self,
        send: Callable[[CtrlCommandRecord], object],
        *,
        timeout_sec: float = 0.2,
        fallback_interval_sec: float = 0.1,
    ):
        self._send = send
        self.timeout_sec = self._positive_interval("timeout_sec", timeout_sec)
        self.fallback_interval_sec = self._positive_interval(
            "fallback_interval_sec", fallback_interval_sec
        )
        self._last_command_at: float | None = None
        self._last_fallback_at: float | None = None
        self._last_seen_at: float | None = None
        self._template: CtrlCommandRecord | None = None

    def accept(self, command: CtrlCommandRecord, now: float) -> None:
        now = self._checked_time(now)
        self._send(command)
        self._template = command
        self._last_command_at = now
        self._last_fallback_at = None
        self._last_seen_at = now

    def tick(self, now: float) -> bool:
        now = self._checked_time(now)
        if self._last_command_at is None or self._template is None:
            self._last_seen_at = now
            return False
        if now - self._last_command_at < self.timeout_sec:
            self._last_seen_at = now
            return False
        if (
            self._last_fallback_at is not None
            and now - self._last_fallback_at < self.fallback_interval_sec
        ):
            self._last_seen_at = now
            return False
        self._send(self._fallback())
        self._last_fallback_at = now
        self._last_seen_at = now
        return True

    @staticmethod
    def _positive_interval(name: str, value: float) -> float:
        parsed = float(value)
        if not math.isfinite(parsed) or parsed <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return parsed

    def _checked_time(self, value: float) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("monotonic time must be finite")
        if self._last_seen_at is not None and parsed < self._last_seen_at:
            raise ValueError("monotonic time moved backward")
        return parsed

    def _fallback(self) -> CtrlCommandRecord:
        assert self._template is not None
        return CtrlCommandRecord(
            ctrl_mode=self._template.ctrl_mode,
            gear=self._template.gear,
            long_cmd_type=1,
            velocity=0.0,
            acceleration=0.0,
            accel=0.0,
            brake=1.0,
            steering=0.0,
        )

    def emergency_stop(self) -> bool:
        if self._template is None:
            return False
        fallback = self._fallback()
        for _ in range(3):
            self._send(fallback)
        self.disable()
        return True

    def disable(self) -> None:
        self._last_command_at = None
        self._last_fallback_at = None
        self._template = None
