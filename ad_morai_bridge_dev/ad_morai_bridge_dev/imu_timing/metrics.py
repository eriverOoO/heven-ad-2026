"""Pure, deterministic metrics for passive MORAI IMU timing observations.

This module deliberately has no ROS dependency.  The ROS evaluator adapts its
messages into these frozen value records after it has frozen a measurement
window.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any


_COUNTER_FIELDS = (
    "packets",
    "bytes",
    "malformed",
    "dropped",
    "bind_errors",
    "source_selected",
    "arrival_fallback",
    "source_rejected",
    "duplicates",
    "stamp_regressions",
)
_VIOLATION_COUNTER_FIELDS = (
    "malformed",
    "dropped",
    "bind_errors",
    "source_selected",
    "stamp_regressions",
)


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-boolean integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _positive_integer(value: object, name: str) -> int:
    value = _nonnegative_integer(value, name)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be bool")
    return value


def _stamp_tuple(values: object, name: str) -> tuple[int, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple of nanosecond stamps")
    return tuple(_nonnegative_integer(value, f"{name}[{index}]") for index, value in enumerate(values))


@dataclass(frozen=True)
class TimingAudit:
    """One audit message adapted into only fields used by the timing contract."""

    header_stamp_ns: int
    stream: str
    source_valid: bool
    source_selected: bool
    source_rejected: bool
    duplicate: bool
    stamp_regression: bool
    normalized_published: bool

    def __post_init__(self) -> None:
        _nonnegative_integer(self.header_stamp_ns, "header_stamp_ns")
        if not isinstance(self.stream, str) or not self.stream:
            raise ValueError("stream must be a non-empty string")
        for name in (
            "source_valid",
            "source_selected",
            "source_rejected",
            "duplicate",
            "stamp_regression",
            "normalized_published",
        ):
            _boolean(getattr(self, name), name)


@dataclass(frozen=True)
class BridgeCounterSnapshot:
    """The monotonic bridge counters captured at either window boundary."""

    packets: int
    bytes: int
    malformed: int
    dropped: int
    bind_errors: int
    source_selected: int
    arrival_fallback: int
    source_rejected: int
    duplicates: int
    stamp_regressions: int

    def __post_init__(self) -> None:
        for name in _COUNTER_FIELDS:
            _nonnegative_integer(getattr(self, name), name)

    def as_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in _COUNTER_FIELDS}


@dataclass(frozen=True)
class TimingWindow:
    """A frozen measurement window; all headers are receipt-time nanoseconds."""

    normalized_headers: tuple[int, ...]
    full_headers: tuple[int, ...]
    timing_headers: tuple[int, ...]
    audits: tuple[TimingAudit, ...]
    start_counters: BridgeCounterSnapshot
    end_counters: BridgeCounterSnapshot
    measurement_duration_ns: int
    normalized_discovered: bool
    full_discovered: bool
    timing_discovered: bool
    statistics_discovered: bool

    def __post_init__(self) -> None:
        for name in ("normalized_headers", "full_headers", "timing_headers"):
            _stamp_tuple(getattr(self, name), name)
        if not isinstance(self.audits, tuple) or not all(isinstance(audit, TimingAudit) for audit in self.audits):
            raise TypeError("audits must be a tuple of TimingAudit")
        if not isinstance(self.start_counters, BridgeCounterSnapshot):
            raise TypeError("start_counters must be BridgeCounterSnapshot")
        if not isinstance(self.end_counters, BridgeCounterSnapshot):
            raise TypeError("end_counters must be BridgeCounterSnapshot")
        _positive_integer(self.measurement_duration_ns, "measurement_duration_ns")
        for name in (
            "normalized_discovered",
            "full_discovered",
            "timing_discovered",
            "statistics_discovered",
        ):
            _boolean(getattr(self, name), name)
        for name in _COUNTER_FIELDS:
            if getattr(self.end_counters, name) < getattr(self.start_counters, name):
                raise ValueError(f"bridge counter regression: {name}")


def _percentile(sorted_values: tuple[int, ...], quantile: float) -> float | None:
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction)


def _distribution(values: tuple[int | float, ...]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "stddev": None, "p50": None, "p95": None, "p99": None, "max": None}
    ordered = tuple(sorted(values))
    count = len(ordered)
    mean = math.fsum(ordered) / count
    variance = math.fsum((value - mean) ** 2 for value in ordered) / count
    return {
        "count": count,
        "min": ordered[0],
        "mean": mean,
        "stddev": math.sqrt(variance),
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1],
    }


def _counter_deltas(start: BridgeCounterSnapshot, end: BridgeCounterSnapshot) -> dict[str, int]:
    return {name: getattr(end, name) - getattr(start, name) for name in _COUNTER_FIELDS}


def _advisories(
    rate_hz: float | None,
    expected_rate_hz: int,
    interval: Mapping[str, int | float | None],
    audit_count: int,
    repeat_count: int,
    burst_count: int,
    gap_count: int,
    large_gap_count: int,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if rate_hz is not None and not (expected_rate_hz * 0.9 <= rate_hz <= expected_rate_hz * 1.1):
        warnings.append("measured_rate_out_of_range")
    if audit_count and repeat_count / audit_count > 0.05:
        warnings.append("source_repeat_ratio_exceeded")
    interval_count = int(interval["count"])
    if interval_count:
        nominal_period_ns = 1_000_000_000 / expected_rate_hz
        p95 = interval["p95"]
        p99 = interval["p99"]
        if p95 is not None and p95 > 2 * nominal_period_ns:
            warnings.append("p95_interval_exceeded")
        if p99 is not None and p99 > 3 * nominal_period_ns:
            warnings.append("p99_interval_exceeded")
        if burst_count / interval_count > 0.01:
            warnings.append("burst_ratio_exceeded")
        if gap_count / interval_count > 0.01:
            warnings.append("gap_ratio_exceeded")
    if large_gap_count:
        warnings.append("large_gap_observed")
    return tuple(warnings)


def summarize_timing_window(window: TimingWindow, expected_rate_hz: int) -> dict[str, object]:
    """Return deterministic, JSON-adaptable primitive metrics plus exact Counters."""
    if not isinstance(window, TimingWindow):
        raise TypeError("window must be TimingWindow")
    expected_rate_hz = _positive_integer(expected_rate_hz, "expected_rate_hz")

    header_multisets = {
        "normalized": Counter(window.normalized_headers),
        "full": Counter(window.full_headers),
        "timing": Counter(window.timing_headers),
    }
    parity = len({frozenset(counter.items()) for counter in header_multisets.values()}) == 1
    intervals = tuple(
        later - earlier for earlier, later in zip(window.timing_headers, window.timing_headers[1:])
    )
    interval = _distribution(intervals)
    nominal_period_ns = 1_000_000_000 / expected_rate_hz
    jitter = _distribution(tuple(abs(value - nominal_period_ns) for value in intervals))
    burst_count = sum(value < 0.25 * nominal_period_ns for value in intervals)
    gap_count = sum(value > 2 * nominal_period_ns for value in intervals)
    large_gap_count = sum(value > 500_000_000 for value in intervals)
    rate_hz = len(window.timing_headers) * 1_000_000_000 / window.measurement_duration_ns
    repeat_count = sum(audit.duplicate for audit in window.audits)
    source_advance_count = sum(
        audit.source_valid and not audit.source_rejected and not audit.duplicate and not audit.stamp_regression
        for audit in window.audits
    )
    source_reject_count = sum(audit.source_rejected for audit in window.audits)
    source_selected_count = sum(audit.source_selected for audit in window.audits)
    deltas = _counter_deltas(window.start_counters, window.end_counters)
    minimum_required_samples = max(
        100, expected_rate_hz * window.measurement_duration_ns // 2_000_000_000
    )
    normalized_non_increasing_count = sum(
        later <= earlier for earlier, later in zip(window.normalized_headers, window.normalized_headers[1:])
    )
    advisory = _advisories(
        rate_hz,
        expected_rate_hz,
        interval,
        len(window.audits),
        repeat_count,
        burst_count,
        gap_count,
        large_gap_count,
    )
    return {
        "expected_rate_hz": expected_rate_hz,
        "measurement_duration_ns": window.measurement_duration_ns,
        "discovery": {
            "normalized": window.normalized_discovered,
            "full": window.full_discovered,
            "timing": window.timing_discovered,
            "statistics": window.statistics_discovered,
        },
        "sample_count": len(window.timing_headers),
        "minimum_required_samples": minimum_required_samples,
        "header_multisets": header_multisets,
        "header_multiset_parity": parity,
        "rate_hz": rate_hz,
        "interval_ns": interval,
        "jitter_ns": jitter,
        "burst_count": burst_count,
        "gap_count": gap_count,
        "large_gap_count": large_gap_count,
        "source": {
            "repeat_count": repeat_count,
            "advance_count": source_advance_count,
            "reject_count": source_reject_count,
            "selected_count": source_selected_count,
        },
        "counter_deltas": deltas,
        "normalized_non_increasing_count": normalized_non_increasing_count,
        "audit_stamp_regression_count": sum(audit.stamp_regression for audit in window.audits),
        "audit_source_selected_count": source_selected_count,
        "audit_unpublished_count": sum(not audit.normalized_published for audit in window.audits),
        "advisories": advisory,
    }


def evaluate_transport_contract(summary: Mapping[str, object]) -> tuple[int, tuple[str, ...]]:
    """Classify a summary as pass (0), violation (2), or insufficient (3)."""
    if not isinstance(summary, Mapping):
        raise TypeError("summary must be a mapping")

    insufficiency_reasons: list[str] = []
    discovery = summary.get("discovery")
    if not isinstance(discovery, Mapping) or not all(
        discovery.get(name) is True for name in ("normalized", "full", "timing", "statistics")
    ):
        insufficiency_reasons.append("publisher_discovery_incomplete")
    sample_count = summary.get("sample_count")
    minimum_required = summary.get("minimum_required_samples")
    if (
        isinstance(sample_count, bool)
        or isinstance(minimum_required, bool)
        or not isinstance(sample_count, int)
        or not isinstance(minimum_required, int)
        or sample_count < minimum_required
    ):
        insufficiency_reasons.append("insufficient_timing_samples")
    if summary.get("header_multiset_parity") is not True:
        insufficiency_reasons.append("header_multiset_mismatch")

    reasons: list[str] = []
    if summary.get("normalized_non_increasing_count", 0):
        reasons.append("normalized_stamp_non_increasing")
    if summary.get("audit_stamp_regression_count", 0):
        reasons.append("audit_stamp_regression")
    if summary.get("audit_source_selected_count", 0):
        reasons.append("audit_source_selected")
    if summary.get("audit_unpublished_count", 0):
        reasons.append("audit_unpublished")
    deltas = summary.get("counter_deltas")
    if not isinstance(deltas, Mapping):
        raise TypeError("summary counter_deltas must be a mapping")
    for field in _VIOLATION_COUNTER_FIELDS:
        value: Any = deltas.get(field, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid counter delta: {field}")
        if value:
            reasons.append(f"counter_{field}_growth")
    if reasons:
        return 2, tuple(reasons)
    if insufficiency_reasons:
        return 3, tuple(insufficiency_reasons)
    return 0, ()
