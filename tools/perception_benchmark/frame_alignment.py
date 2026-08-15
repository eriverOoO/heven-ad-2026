"""Timestamp alignment primitives for offline perception benchmarks."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Generic, Iterable, TypeVar


T = TypeVar("T")


def stamp_ns(stamp: object) -> int:
    """Convert a ROS builtin_interfaces/Time-like object to nanoseconds."""
    return int(getattr(stamp, "sec")) * 1_000_000_000 + int(
        getattr(stamp, "nanosec")
    )


@dataclass(frozen=True)
class TimedSample(Generic[T]):
    source_ns: int
    record_ns: int
    value: T


class NearestIndex(Generic[T]):
    """Nearest-source-stamp index with an explicit maximum time error."""

    def __init__(self, samples: Iterable[TimedSample[T]]) -> None:
        self.samples = sorted(samples, key=lambda sample: sample.source_ns)
        self._stamps = [sample.source_ns for sample in self.samples]

    def nearest(self, source_ns: int, max_delta_ns: int) -> TimedSample[T] | None:
        if not self.samples:
            return None
        position = bisect_left(self._stamps, source_ns)
        candidates = []
        if position < len(self.samples):
            candidates.append(self.samples[position])
        if position:
            candidates.append(self.samples[position - 1])
        match = min(
            candidates,
            key=lambda sample: (abs(sample.source_ns - source_ns), sample.source_ns),
        )
        if abs(match.source_ns - source_ns) > max_delta_ns:
            return None
        return match


def exact_record_times(samples: Iterable[TimedSample[object]]) -> dict[int, int]:
    """Return the earliest recorder timestamp for every source timestamp."""
    result: dict[int, int] = {}
    for sample in samples:
        current = result.get(sample.source_ns)
        if current is None or sample.record_ns < current:
            result[sample.source_ns] = sample.record_ns
    return result
