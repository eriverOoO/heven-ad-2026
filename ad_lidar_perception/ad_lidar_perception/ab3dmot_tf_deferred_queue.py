"""Bounded, order-preserving deferred-processing queue for exact-stamp TF
lookups (AB3DMOT Exact-Stamp TF Deferred Processing v1).

Root cause this addresses (see docs/perception/ab3dmot_tf_deferred_processing_v1.md):
AB3DMOT requests ``odom <- detection_frame`` at the detection's EXACT source
stamp `t`. Localization publishes TF on a slightly different schedule than
the detector, so at the instant the detection callback runs, tf2 frequently
has only the transform sample *before* `t`; the sample *after* `t` (needed
for interpolation) has not arrived yet. tf2 correctly raises
``ExtrapolationException`` ("... extrapolation into the future ...") for
this -- that rejection is safety-correct (never invent geometry), but the
transform genuinely becomes available a short time later, and the pre-
existing code discarded the detection immediately instead of waiting for it.

This module holds such a detection in a small FIFO queue instead of
discarding it, and releases it -- still keyed to its ORIGINAL stamp `t` --
once tf2 can supply the exact transform at `t`. It never extrapolates,
never substitutes a different frame's transform, never blocks, and never
reorders detections relative to their own source timestamps.

Pure, ROS-independent logic (mirrors the injection pattern already used by
``ab3dmot_ros.py``/``centerpoint_ros.py``): the caller supplies a
``can_transform`` callable (typically a thin wrapper around
``tf2_ros.Buffer.can_transform(..., return_debug_tuple=True)``) and a
monotonic clock function; this module never imports ``rclpy``/``tf2_ros``
itself, so it is unit-testable with plain fakes.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Deque, Generic, List, Tuple, TypeVar

T = TypeVar("T")


class TransformAvailability(Enum):
    """Classification of one ``can_transform(..., return_debug_tuple=True)``
    result. Only ``FUTURE_DATA_PENDING`` is transient -- everything else
    (an unknown/disconnected frame, or a request whose earlier sample has
    already been evicted from tf2's fixed-size buffer -- "extrapolation
    into the *past*") is permanent for that exact stamp and must fail fast,
    exactly like the pre-existing unconditional-drop behaviour.
    """

    READY = "ready"
    FUTURE_DATA_PENDING = "future_data_pending"
    PERMANENT_FAILURE = "permanent_failure"


def classify_transform_availability(
    ok: bool, debug_message: str
) -> TransformAvailability:
    """Classify a ``tf2_ros.Buffer.can_transform(return_debug_tuple=True)``
    result. ``debug_message`` is tf2's own human-readable reason string
    (empty when ``ok`` is True).

    Measured directly against a real ``tf2_ros.Buffer`` on this repo's ROS
    Humble install: a future-side gap reads
    "Lookup would require extrapolation into the future. ..."; a
    past-side-evicted gap reads "... extrapolation into the past. ..."; an
    unknown frame reads '... frame does not exist'. Only the first is
    treated as transient.
    """
    if ok:
        return TransformAvailability.READY
    if "extrapolation into the future" in debug_message:
        return TransformAvailability.FUTURE_DATA_PENDING
    return TransformAvailability.PERMANENT_FAILURE


class ReleaseEvent(Enum):
    """Outcome of releasing one queued (or immediately-offered) detection."""

    READY = "ready"
    DEFERRED = "deferred"
    DROPPED_PERMANENT = "dropped_permanent"
    DROPPED_TIMEOUT = "dropped_timeout"


@dataclass
class _PendingDetection(Generic[T]):
    stamp_ns: int
    payload: T
    enqueued_at_s: float


@dataclass
class DeferredQueueStats:
    """Lightweight diagnostic counters (Phase 18). Intentionally cheap to
    maintain per-frame; no per-frame logging is driven from here."""

    detections_received: int = 0
    processed_immediately: int = 0
    processed_deferred: int = 0
    dropped_tf_timeout: int = 0
    dropped_permanent_tf_failure: int = 0
    dropped_queue_overflow: int = 0
    pending_queue_depth: int = 0
    max_observed_queue_depth: int = 0

    def as_dict(self) -> dict:
        return {
            "detections_received": self.detections_received,
            "processed_immediately": self.processed_immediately,
            "processed_deferred": self.processed_deferred,
            "dropped_tf_timeout": self.dropped_tf_timeout,
            "dropped_permanent_tf_failure": self.dropped_permanent_tf_failure,
            "dropped_queue_overflow": self.dropped_queue_overflow,
            "pending_queue_depth": self.pending_queue_depth,
            "max_observed_queue_depth": self.max_observed_queue_depth,
        }


class DeferredDetectionQueue(Generic[T]):
    """FIFO, timestamp-ordered, bounded queue that defers one detection
    payload until its EXACT-stamp transform becomes available.

    Contract:
      - A detection is released in FIFO order only -- never before every
        detection ahead of it in the queue has itself been released
        (processed or dropped). This guarantees non-decreasing source-
        timestamp processing order (Phase 6) as long as the caller offers
        detections in non-decreasing stamp order (already enforced
        upstream by the existing ``classify_timestamp`` duplicate/rollback
        gate).
      - Each accepted detection is released to the caller at most once
        (Phase 7): ``offer()`` returns a payload directly only when the
        queue was empty and the transform was ready immediately; every
        other admitted detection is handed back exactly once, later, via
        ``drain()``.
      - Never calls ``lookup_transform`` and never returns a transform
        itself -- it only classifies availability. The caller performs the
        actual (unchanged) ``lookup_transform`` call once this queue says
        "ready", so tf2's own interpolation reconstructs the pose AT the
        original stamp; this queue never consumes a "future" transform's
        pose directly (Phase 38).
      - Bounded: at most ``max_pending`` entries are held; exceeding it
        drops the OLDEST unresolved entry (Phase 17), never a random one.
      - Non-blocking: no sleep, no wait; ``can_transform`` is expected to
        be called with a zero timeout by the caller.
    """

    def __init__(
        self,
        *,
        max_pending: int,
        max_wait_s: float,
        can_transform: Callable[[T], Tuple[bool, str]],
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_pending <= 0:
            raise ValueError("max_pending must be > 0")
        if not (max_wait_s > 0.0):
            raise ValueError("max_wait_s must be > 0")
        self._max_pending = int(max_pending)
        self._max_wait_s = float(max_wait_s)
        self._can_transform = can_transform
        self._monotonic = monotonic
        self._queue: Deque[_PendingDetection[T]] = deque()
        self.stats = DeferredQueueStats()

    def __len__(self) -> int:
        return len(self._queue)

    def offer(self, stamp_ns: int, payload: T) -> Tuple[ReleaseEvent, T | None]:
        """Admit one newly-arrived detection.

        Returns ``(ReleaseEvent.READY, payload)`` when the queue was empty
        and the transform is available right now -- the caller should
        process it immediately, and it is never enqueued (pending stays 0,
        Phase 19). Returns ``(ReleaseEvent.DROPPED_PERMANENT, None)`` when
        the queue was empty but the transform failed for a permanent
        reason. Otherwise the detection is enqueued (queue was non-empty --
        ordering requires it to wait behind whatever is already pending,
        Phase 6/21 -- or its own transform is only future-pending) and
        ``(ReleaseEvent.DEFERRED, None)`` is returned: a later ``drain()``
        call will hand this same payload back exactly once.
        """
        self.stats.detections_received += 1
        if not self._queue:
            ok, debug = self._can_transform(payload)
            availability = classify_transform_availability(ok, debug)
            if availability is TransformAvailability.READY:
                self.stats.processed_immediately += 1
                return ReleaseEvent.READY, payload
            if availability is TransformAvailability.PERMANENT_FAILURE:
                self.stats.dropped_permanent_tf_failure += 1
                return ReleaseEvent.DROPPED_PERMANENT, None
            # FUTURE_DATA_PENDING: fall through to enqueue below.
        self._enqueue(stamp_ns, payload)
        return ReleaseEvent.DEFERRED, None

    def _enqueue(self, stamp_ns: int, payload: T) -> None:
        if len(self._queue) >= self._max_pending:
            self._queue.popleft()
            self.stats.dropped_queue_overflow += 1
        self._queue.append(
            _PendingDetection(stamp_ns=stamp_ns, payload=payload, enqueued_at_s=self._monotonic())
        )
        self.stats.pending_queue_depth = len(self._queue)
        self.stats.max_observed_queue_depth = max(
            self.stats.max_observed_queue_depth, len(self._queue)
        )

    def drain(self) -> List[Tuple[ReleaseEvent, T]]:
        """Release as many head-of-queue entries as are now resolvable, in
        strict FIFO order. Stops at the first entry that is still
        transiently pending and has not timed out -- nothing behind it may
        be released ahead of it (Phase 6/21).
        """
        released: List[Tuple[ReleaseEvent, T]] = []
        while self._queue:
            head = self._queue[0]
            ok, debug = self._can_transform(head.payload)
            availability = classify_transform_availability(ok, debug)
            if availability is TransformAvailability.READY:
                self._queue.popleft()
                self.stats.processed_deferred += 1
                released.append((ReleaseEvent.READY, head.payload))
                continue
            if availability is TransformAvailability.PERMANENT_FAILURE:
                self._queue.popleft()
                self.stats.dropped_permanent_tf_failure += 1
                released.append((ReleaseEvent.DROPPED_PERMANENT, head.payload))
                continue
            elapsed = self._monotonic() - head.enqueued_at_s
            if elapsed >= self._max_wait_s:
                self._queue.popleft()
                self.stats.dropped_tf_timeout += 1
                released.append((ReleaseEvent.DROPPED_TIMEOUT, head.payload))
                continue
            break  # still legitimately pending, not timed out: stop here.
        self.stats.pending_queue_depth = len(self._queue)
        return released
