"""Stateful selection of source versus UDP-arrival timestamps."""

from dataclasses import dataclass
import math

from .protocol_records import Stamp


_MAX_ROS_TIME_SEC = 2_147_483_647


def is_valid_ros_stamp(stamp: Stamp | None) -> bool:
    """Return whether *stamp* can be represented by builtin_interfaces/Time."""

    if stamp is None or len(stamp) != 2:
        return False
    sec, nanosec = stamp
    if isinstance(sec, bool) or isinstance(nanosec, bool):
        return False
    if not isinstance(sec, int) or not isinstance(nanosec, int):
        return False
    return 0 <= sec <= _MAX_ROS_TIME_SEC and 0 <= nanosec < 1_000_000_000


def stamp_ns(stamp: Stamp) -> int:
    return int(stamp[0]) * 1_000_000_000 + int(stamp[1])


def source_within_arrival_window(
    source_stamp: Stamp | None,
    arrival_stamp: Stamp,
    tolerance_sec: float,
) -> bool:
    if not is_valid_ros_stamp(source_stamp) or not is_valid_ros_stamp(arrival_stamp):
        return False
    tolerance_ns = int(float(tolerance_sec) * 1_000_000_000)
    return abs(stamp_ns(source_stamp) - stamp_ns(arrival_stamp)) <= tolerance_ns


@dataclass(frozen=True)
class TimestampDecision:
    selected_stamp: Stamp
    source_valid: bool
    source_selected: bool
    arrival_fallback: bool
    source_rejected: bool
    duplicate: bool
    stamp_regression: bool
    publish_normalized: bool


class TimestampPolicy:
    """Apply the timestamp contract for one independent sensor stream.

    Accepted source time never moves backwards. A regressed/reset source clock
    therefore remains on arrival fallback until it catches the previous source
    watermark (or this policy is recreated when the bridge restarts).
    """

    def __init__(
        self,
        *,
        mode: str,
        tolerance_sec: float,
        suppress_source_duplicates: bool = False,
    ) -> None:
        if mode not in ("source_preferred", "arrival"):
            raise ValueError("timestamp mode must be 'source_preferred' or 'arrival'")
        tolerance_sec = float(tolerance_sec)
        if not math.isfinite(tolerance_sec) or tolerance_sec <= 0.0:
            raise ValueError("source timestamp tolerance must be finite and positive")
        self._mode = mode
        self._tolerance_sec = tolerance_sec
        self._suppress_source_duplicates = bool(suppress_source_duplicates)
        self._last_source_ns: int | None = None
        self._last_chosen_ns: int | None = None

    def reset(self) -> None:
        """Clear watermarks after an explicitly confirmed stream epoch reset."""

        self._last_source_ns = None
        self._last_chosen_ns = None

    def decide(
        self,
        source_stamp: Stamp | None,
        arrival_stamp: Stamp,
        *,
        reject_source: bool = False,
        publish_requires_valid_source: bool = False,
    ) -> TimestampDecision:
        if not is_valid_ros_stamp(arrival_stamp):
            raise ValueError(f"invalid arrival timestamp {arrival_stamp!r}")

        source_rejected = bool(reject_source)
        source_accepted = False
        duplicate = False

        if source_stamp is not None and not reject_source:
            if source_within_arrival_window(
                source_stamp, arrival_stamp, self._tolerance_sec
            ):
                source_ns = stamp_ns(source_stamp)
                if self._last_source_ns is None or source_ns >= self._last_source_ns:
                    duplicate = source_ns == self._last_source_ns
                    source_accepted = True
                    if not duplicate:
                        self._last_source_ns = source_ns
                else:
                    source_rejected = True
            else:
                source_rejected = True

        source_selected = self._mode == "source_preferred" and source_accepted
        selected_stamp = source_stamp if source_selected else arrival_stamp
        assert selected_stamp is not None
        selected_ns = stamp_ns(selected_stamp)
        candidate_publishable = (
            not publish_requires_valid_source or source_accepted
        )
        stamp_regression = candidate_publishable and (
            self._last_chosen_ns is not None
            and selected_ns < self._last_chosen_ns
        )
        publish_normalized = candidate_publishable and not stamp_regression and not (
            duplicate and self._suppress_source_duplicates
        )
        if publish_normalized:
            self._last_chosen_ns = selected_ns

        return TimestampDecision(
            selected_stamp=selected_stamp,
            source_valid=source_accepted,
            source_selected=source_selected,
            arrival_fallback=not source_selected,
            source_rejected=source_rejected,
            duplicate=duplicate,
            stamp_regression=stamp_regression,
            publish_normalized=publish_normalized,
        )
