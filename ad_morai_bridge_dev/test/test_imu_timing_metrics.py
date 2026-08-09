from __future__ import annotations

from collections import Counter
import math

import pytest

from ad_morai_bridge_dev.imu_timing.metrics import (
    BridgeCounterSnapshot,
    TimingAudit,
    TimingWindow,
    evaluate_transport_contract,
    summarize_timing_window,
)


def _counters(**overrides: int) -> BridgeCounterSnapshot:
    values = {
        "packets": 1_000,
        "bytes": 100_000,
        "malformed": 0,
        "dropped": 0,
        "bind_errors": 0,
        "source_selected": 0,
        "arrival_fallback": 100,
        "source_rejected": 0,
        "duplicates": 0,
        "stamp_regressions": 0,
    }
    values.update(overrides)
    return BridgeCounterSnapshot(**values)


def _ideal_window(rate_hz: int, period_ns: int, count: int) -> TimingWindow:
    # These are literal MORAI-rate fixtures: 20 Hz/50 ms, 30 Hz/33.333333 ms,
    # and 50 Hz/20 ms.  A broken period, parity, or rate calculation changes
    # their hand-derived outcomes below.
    headers = tuple(1_000_000_000 + index * period_ns for index in range(count))
    audits = tuple(
        TimingAudit(
            header_stamp_ns=header,
            stream="imu",
            source_valid=False,
            source_selected=False,
            source_rejected=False,
            duplicate=False,
            stamp_regression=False,
            normalized_published=True,
        )
        for header in headers
    )
    return TimingWindow(
        normalized_headers=headers,
        full_headers=headers,
        timing_headers=headers,
        audits=audits,
        start_counters=_counters(),
        end_counters=_counters(packets=1_000 + count, bytes=100_000 + count * 100),
        measurement_duration_ns=10_000_000_000,
        normalized_discovered=True,
        full_discovered=True,
        timing_discovered=True,
        statistics_discovered=True,
    )


@pytest.mark.parametrize(
    ("rate_hz", "period_ns", "count", "want_rate_hz", "want_minimum_samples"),
    [
        (20, 50_000_000, 200, 20.0, 100),
        (30, 33_333_333, 300, 30.0, 150),
        (50, 20_000_000, 500, 50.0, 250),
    ],
)
def test_literal_20_30_50_hz_windows_report_exact_parity_and_timing_metrics(
    rate_hz: int, period_ns: int, count: int, want_rate_hz: float, want_minimum_samples: int
) -> None:
    """Catches wrong interval/rate/jitter math or a set-based parity check."""
    summary = summarize_timing_window(_ideal_window(rate_hz, period_ns, count), rate_hz)

    assert summary["sample_count"] == count
    assert summary["minimum_required_samples"] == want_minimum_samples
    assert summary["header_multisets"] == {
        "normalized": Counter({1_000_000_000 + index * period_ns: 1 for index in range(count)}),
        "full": Counter({1_000_000_000 + index * period_ns: 1 for index in range(count)}),
        "timing": Counter({1_000_000_000 + index * period_ns: 1 for index in range(count)}),
    }
    assert summary["header_multiset_parity"] is True
    assert summary["rate_hz"] == pytest.approx(want_rate_hz)
    assert summary["interval_ns"] == {
        "count": count - 1,
        "min": period_ns,
        "mean": float(period_ns),
        "stddev": 0.0,
        "p50": float(period_ns),
        "p95": float(period_ns),
        "p99": float(period_ns),
        "max": period_ns,
    }
    assert summary["jitter_ns"]["max"] == pytest.approx(abs(period_ns - (1_000_000_000 / rate_hz)))
    assert summary["burst_count"] == 0
    assert summary["gap_count"] == 0
    assert summary["large_gap_count"] == 0
    assert evaluate_transport_contract(summary) == (0, ())


def test_trailing_outage_uses_the_complete_frozen_window_for_receipt_rate() -> None:
    """Catches first-to-last-header rate math hiding silence after a sufficient sample burst."""
    # 100 samples at 20 Hz occupy only the first 4.95 s of this explicit 10 s
    # frozen window; the remaining approximately 5.05 s is an outage.
    summary = summarize_timing_window(_ideal_window(20, 50_000_000, 100), 20)

    assert summary["sample_count"] == 100
    assert summary["minimum_required_samples"] == 100
    assert summary["rate_hz"] == 10.0
    assert "measured_rate_out_of_range" in summary["advisories"]
    assert evaluate_transport_contract(summary) == (0, ())


def test_multiset_parity_detects_a_duplicate_replacing_a_distinct_header() -> None:
    """Catches parity implemented as a set comparison that hides duplicates."""
    window = _ideal_window(20, 50_000_000, 100)
    mismatched = TimingWindow(
        **{**window.__dict__, "full_headers": window.full_headers[:-1] + (window.full_headers[-2],)}
    )

    summary = summarize_timing_window(mismatched, 20)

    assert summary["header_multiset_parity"] is False
    assert evaluate_transport_contract(summary) == (3, ("header_multiset_mismatch",))


@pytest.mark.parametrize(
    ("field", "delta", "reason"),
    [
        ("malformed", 1, "counter_malformed_growth"),
        ("dropped", 1, "counter_dropped_growth"),
        ("bind_errors", 1, "counter_bind_errors_growth"),
        ("source_selected", 1, "counter_source_selected_growth"),
        ("stamp_regressions", 1, "counter_stamp_regressions_growth"),
    ],
)
def test_transport_counter_growth_is_a_confirmed_violation(
    field: str, delta: int, reason: str
) -> None:
    """Catches a contract evaluator that ignores harmful bridge counter growth."""
    window = _ideal_window(20, 50_000_000, 100)
    end_values = dict(window.end_counters.__dict__)
    end_values[field] += delta
    summary = summarize_timing_window(
        TimingWindow(**{**window.__dict__, "end_counters": BridgeCounterSnapshot(**end_values)}), 20
    )

    assert evaluate_transport_contract(summary) == (2, (reason,))


@pytest.mark.parametrize(
    ("audit_update", "reason"),
    [
        ({"stamp_regression": True}, "audit_stamp_regression"),
        ({"source_selected": True}, "audit_source_selected"),
        ({"normalized_published": False}, "audit_unpublished"),
    ],
)
def test_audit_contract_events_are_confirmed_violations(
    audit_update: dict[str, bool], reason: str
) -> None:
    """Catches source/regression/publication audit violations being downgraded."""
    window = _ideal_window(20, 50_000_000, 100)
    changed_audit = TimingAudit(**{**window.audits[0].__dict__, **audit_update})
    summary = summarize_timing_window(
        TimingWindow(**{**window.__dict__, "audits": (changed_audit,) + window.audits[1:]}), 20
    )

    assert evaluate_transport_contract(summary) == (2, (reason,))


def test_non_increasing_normalized_headers_are_confirmed_violations() -> None:
    """Catches acceptance of repeated or regressing normalized receipt stamps."""
    window = _ideal_window(20, 50_000_000, 100)
    headers = window.normalized_headers[:20] + (window.normalized_headers[19],) + window.normalized_headers[21:]
    summary = summarize_timing_window(
        TimingWindow(
            **{
                **window.__dict__,
                "normalized_headers": headers,
                "full_headers": headers,
                "timing_headers": headers,
            }
        ),
        20,
    )

    assert summary["normalized_non_increasing_count"] == 1
    assert evaluate_transport_contract(summary) == (2, ("normalized_stamp_non_increasing",))


@pytest.mark.parametrize(
    ("window_update", "want_reasons"),
    [
        ({"normalized_discovered": False}, ("publisher_discovery_incomplete",)),
        ({"full_discovered": False}, ("publisher_discovery_incomplete",)),
        ({"timing_discovered": False}, ("publisher_discovery_incomplete",)),
        ({"statistics_discovered": False}, ("publisher_discovery_incomplete",)),
        ({"timing_headers": tuple(range(99))}, ("insufficient_timing_samples", "header_multiset_mismatch")),
    ],
)
def test_missing_discovery_or_too_few_samples_is_insufficient_evidence(
    window_update: dict[str, object], want_reasons: tuple[str, ...]
) -> None:
    """Catches a bridge accusation when a publisher/discovery/sample prerequisite is absent."""
    window = _ideal_window(20, 50_000_000, 100)
    summary = summarize_timing_window(TimingWindow(**{**window.__dict__, **window_update}), 20)

    exit_code, reasons = evaluate_transport_contract(summary)

    assert exit_code == 3
    assert reasons == want_reasons


def test_advisory_rate_repeat_burst_and_gap_warnings_never_change_transport_exit() -> None:
    """Catches tuning advice incorrectly changing the transport contract result."""
    headers = tuple(1_000_000_000 + value for value in (0, 1_000_000, 100_000_000, 700_000_000))
    window = _ideal_window(20, 50_000_000, 100)
    changed_audits = (
        TimingAudit(**{**window.audits[0].__dict__, "duplicate": True}),
    ) + window.audits[1:4]
    summary = summarize_timing_window(
        TimingWindow(
            **{
                **window.__dict__,
                "normalized_headers": headers,
                "full_headers": headers,
                "timing_headers": headers,
                "audits": changed_audits,
                "measurement_duration_ns": 100_000_000,
            }
        ),
        20,
    )

    assert set(summary["advisories"]) == {
        "measured_rate_out_of_range",
        "source_repeat_ratio_exceeded",
        "p95_interval_exceeded",
        "p99_interval_exceeded",
        "burst_ratio_exceeded",
        "gap_ratio_exceeded",
        "large_gap_observed",
    }
    assert evaluate_transport_contract(summary) == (3, ("insufficient_timing_samples",))


def test_confirmed_violations_take_priority_over_incomplete_discovery_and_samples() -> None:
    """Catches insufficient-observation exit 3 hiding already observed transport violations."""
    window = _ideal_window(20, 50_000_000, 100)
    changed_audit = TimingAudit(
        **{
            **window.audits[0].__dict__,
            "source_selected": True,
            "normalized_published": False,
        }
    )
    end_values = dict(window.end_counters.__dict__)
    end_values["malformed"] += 1
    summary = summarize_timing_window(
        TimingWindow(
            **{
                **window.__dict__,
                "timing_headers": tuple(range(99)),
                "audits": (changed_audit,) + window.audits[1:],
                "end_counters": BridgeCounterSnapshot(**end_values),
                "full_discovered": False,
            }
        ),
        20,
    )

    assert evaluate_transport_contract(summary) == (
        2,
        ("audit_source_selected", "audit_unpublished", "counter_malformed_growth"),
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TimingAudit(0, "imu", False, False, False, False, False, True),
        lambda: BridgeCounterSnapshot(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        lambda: TimingWindow(
            (), (), (), (), _counters(), _counters(), 1, True, True, True, True
        ),
    ],
)
def test_records_are_immutable(factory: object) -> None:
    """Catches mutable observation records that could change after the window freezes."""
    record = factory()  # type: ignore[operator]
    with pytest.raises((AttributeError, TypeError)):
        record.stream = "changed"  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "bad_value",
    [True, -1, math.inf, -math.inf, math.nan, 1.5],
)
def test_records_fail_closed_for_boolean_nonfinite_or_negative_integer_inputs(bad_value: object) -> None:
    """Catches permissive numeric validation for timestamps and bridge counters."""
    with pytest.raises((TypeError, ValueError)):
        TimingAudit(bad_value, "imu", False, False, False, False, False, True)  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError)):
        BridgeCounterSnapshot(bad_value, 0, 0, 0, 0, 0, 0, 0, 0, 0)  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError)):
        TimingWindow((), (), (), (), _counters(), _counters(), bad_value, True, True, True, True)  # type: ignore[arg-type]


def test_window_fails_closed_when_a_bridge_counter_regresses() -> None:
    """Catches silently accepted start/end counter regressions that invalidate deltas."""
    with pytest.raises(ValueError, match="regress"):
        TimingWindow(
            (), (), (), (), _counters(packets=2), _counters(packets=1), 1, True, True, True, True
        )
