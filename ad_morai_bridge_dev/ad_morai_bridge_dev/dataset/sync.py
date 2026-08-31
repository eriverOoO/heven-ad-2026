"""LiDAR-anchored synchronisation.

Each accepted raw LiDAR frame is one dataset sample; its ``header.stamp``
is the canonical sample key. Every other source (ego GT, actor GT, dynamic
TF) is matched to that stamp by nearest source ``header.stamp`` with an
explicit maximum skew. An out-of-skew or absent source is reported as such
- never silently replaced with the latest value or a zero transform.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Timed(Generic[T]):
    source_stamp_ns: int
    value: T


@dataclass(frozen=True)
class Match(Generic[T]):
    status: str  # "matched" | "skew_exceeded" | "missing"
    value: T | None
    source_stamp_ns: int | None
    skew_ns: int | None


class TimeBuffer(Generic[T]):
    """Bounded, stamp-sorted ring of timed samples."""

    def __init__(self, capacity: int = 512) -> None:
        if capacity < 2:
            raise ValueError("capacity must be >= 2")
        self._capacity = capacity
        self._items: list[Timed[T]] = []

    def add(self, source_stamp_ns: int, value: T) -> None:
        item = Timed(int(source_stamp_ns), value)
        stamps = [it.source_stamp_ns for it in self._items]
        pos = bisect_left(stamps, item.source_stamp_ns)
        self._items.insert(pos, item)
        if len(self._items) > self._capacity:
            # Drop the oldest; the anchor only ever looks near "now".
            self._items = self._items[-self._capacity :]

    def __len__(self) -> int:
        return len(self._items)

    def nearest(self, stamp_ns: int, max_skew_ns: int) -> Match[T]:
        if not self._items:
            return Match("missing", None, None, None)
        stamps = [it.source_stamp_ns for it in self._items]
        pos = bisect_left(stamps, stamp_ns)
        candidates: list[Timed[T]] = []
        if pos < len(self._items):
            candidates.append(self._items[pos])
        if pos:
            candidates.append(self._items[pos - 1])
        best = min(
            candidates,
            key=lambda it: (abs(it.source_stamp_ns - stamp_ns), it.source_stamp_ns),
        )
        skew = abs(best.source_stamp_ns - stamp_ns)
        if skew > max_skew_ns:
            return Match("skew_exceeded", None, best.source_stamp_ns, skew)
        return Match("matched", best.value, best.source_stamp_ns, skew)


def classify_lidar_stamp(stamp_ns: int, previous_ns: int | None) -> str:
    """Accept / reject the LiDAR anchor stamp itself."""

    if stamp_ns <= 0:
        return "nonpositive"
    if previous_ns is None:
        return "first"
    if stamp_ns == previous_ns:
        return "duplicate"
    if stamp_ns < previous_ns:
        return "backward"
    return "increasing"
