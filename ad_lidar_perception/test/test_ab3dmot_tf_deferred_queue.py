"""Unit tests for the pure deferred-TF-processing queue (T-FIX: AB3DMOT
Exact-Stamp TF Deferred Processing v1). No rclpy/tf2_ros import -- these
run against plain fakes, exactly like ``test_ab3dmot_ros.py``.
"""

import unittest

from ad_lidar_perception.ab3dmot_tf_deferred_queue import (
    DeferredDetectionQueue,
    ReleaseEvent,
    TransformAvailability,
    classify_transform_availability,
)


class ClassifyTransformAvailabilityTest(unittest.TestCase):
    def test_ok_is_ready(self):
        self.assertEqual(
            classify_transform_availability(True, ""), TransformAvailability.READY
        )

    def test_future_extrapolation_message_is_transient(self):
        self.assertEqual(
            classify_transform_availability(
                False,
                "Lookup would require extrapolation into the future.  "
                "Requested time 11.5 but the latest data is at time 11.0",
            ),
            TransformAvailability.FUTURE_DATA_PENDING,
        )

    def test_past_extrapolation_message_is_permanent(self):
        self.assertEqual(
            classify_transform_availability(
                False,
                "Lookup would require extrapolation into the past.  "
                "Requested time 9.0 but the earliest data is at time 10.0",
            ),
            TransformAvailability.PERMANENT_FAILURE,
        )

    def test_unknown_frame_message_is_permanent(self):
        self.assertEqual(
            classify_transform_availability(
                False, 'Invalid frame ID "nonexistent" passed to canTransform'
            ),
            TransformAvailability.PERMANENT_FAILURE,
        )

    def test_arbitrary_other_message_is_permanent(self):
        self.assertEqual(
            classify_transform_availability(False, "some other tf2 failure"),
            TransformAvailability.PERMANENT_FAILURE,
        )


def _make_can_transform(available_stamps):
    """Returns a can_transform(payload)->(ok, debug) fake keyed on
    ``payload.stamp_ns in available_stamps``."""

    def can_transform(payload):
        if payload.stamp_ns in available_stamps:
            return True, ""
        return (
            False,
            "Lookup would require extrapolation into the future.  "
            f"Requested time {payload.stamp_ns}",
        )

    return can_transform


class _Payload:
    def __init__(self, stamp_ns, tag=""):
        self.stamp_ns = stamp_ns
        self.tag = tag

    def __repr__(self):
        return f"_Payload({self.stamp_ns}, {self.tag!r})"


class _FakeClock:
    def __init__(self, start=0.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, dt):
        self.now += dt


# Phase 19: immediate TF -> processed once immediately, pending == 0.
class ImmediateTransformTest(unittest.TestCase):
    def test_ready_transform_is_processed_immediately_with_empty_queue(self):
        available = {100}
        queue = DeferredDetectionQueue(
            max_pending=8, max_wait_s=1.0, can_transform=_make_can_transform(available)
        )
        event, payload = queue.offer(100, _Payload(100))
        self.assertEqual(event, ReleaseEvent.READY)
        self.assertIsNotNone(payload)
        self.assertEqual(len(queue), 0)
        self.assertEqual(queue.stats.processed_immediately, 1)
        self.assertEqual(queue.stats.processed_deferred, 0)


# Phase 20: delayed TF -> not processed, not dropped, pending == 1; once the
# transform becomes available, processed exactly once with original stamp.
class DelayedTransformTest(unittest.TestCase):
    def test_pending_until_transform_ready_then_released_once(self):
        available = set()
        can_transform = _make_can_transform(available)
        queue = DeferredDetectionQueue(max_pending=8, max_wait_s=1.0, can_transform=can_transform)

        event, payload = queue.offer(100, _Payload(100))
        self.assertEqual(event, ReleaseEvent.DEFERRED)
        self.assertIsNone(payload)
        self.assertEqual(len(queue), 1)

        # transform still not ready: drain releases nothing.
        released = queue.drain()
        self.assertEqual(released, [])
        self.assertEqual(len(queue), 1)

        # now the transform becomes available.
        available.add(100)
        released = queue.drain()
        self.assertEqual(len(released), 1)
        event, payload = released[0]
        self.assertEqual(event, ReleaseEvent.READY)
        self.assertEqual(payload.stamp_ns, 100)
        self.assertEqual(len(queue), 0)
        self.assertEqual(queue.stats.processed_deferred, 1)


# Phase 21: ordering -- D2 (already transformable) must not overtake D1.
class OrderingTest(unittest.TestCase):
    def test_later_ready_detection_does_not_overtake_earlier_pending_one(self):
        available = {200}  # D2's stamp is ready from the start; D1's is not.
        can_transform = _make_can_transform(available)
        queue = DeferredDetectionQueue(max_pending=8, max_wait_s=5.0, can_transform=can_transform)

        event, _ = queue.offer(100, _Payload(100, "D1"))
        self.assertEqual(event, ReleaseEvent.DEFERRED)

        # D2 arrives; queue is non-empty, so D2 must be enqueued behind D1
        # regardless of its own transform already being available.
        event, payload = queue.offer(200, _Payload(200, "D2"))
        self.assertEqual(event, ReleaseEvent.DEFERRED)
        self.assertIsNone(payload)
        self.assertEqual(len(queue), 2)

        # D1 still not ready: draining releases nothing, D2 stays parked.
        self.assertEqual(queue.drain(), [])

        # D1 becomes ready: draining must release D1 THEN D2, in that order,
        # within the same drain() call.
        available.add(100)
        released = queue.drain()
        self.assertEqual(len(released), 2)
        self.assertEqual(released[0][1].tag, "D1")
        self.assertEqual(released[1][1].tag, "D2")
        self.assertEqual(len(queue), 0)


# Phase 22: timeout -> dropped once, no tracker update, reason tf_timeout.
class TimeoutTest(unittest.TestCase):
    def test_never_ready_detection_is_dropped_after_max_wait(self):
        clock = _FakeClock(start=0.0)
        can_transform = lambda payload: (False, "extrapolation into the future")
        queue = DeferredDetectionQueue(
            max_pending=8, max_wait_s=0.5, can_transform=can_transform, monotonic=clock
        )
        queue.offer(100, _Payload(100))
        self.assertEqual(len(queue), 1)

        clock.advance(0.3)
        self.assertEqual(queue.drain(), [])  # not yet timed out
        self.assertEqual(len(queue), 1)

        clock.advance(0.3)  # total elapsed 0.6s > 0.5s max_wait
        released = queue.drain()
        self.assertEqual(len(released), 1)
        event, payload = released[0]
        self.assertEqual(event, ReleaseEvent.DROPPED_TIMEOUT)
        self.assertEqual(payload.stamp_ns, 100)
        self.assertEqual(len(queue), 0)
        self.assertEqual(queue.stats.dropped_tf_timeout, 1)


# Phase 23: permanent frame error -> dropped immediately, never queued
# indefinitely.
class PermanentFrameErrorTest(unittest.TestCase):
    def test_permanent_failure_at_offer_time_drops_without_enqueueing(self):
        can_transform = lambda payload: (False, 'Invalid frame ID "bogus" does not exist')
        queue = DeferredDetectionQueue(max_pending=8, max_wait_s=5.0, can_transform=can_transform)
        event, payload = queue.offer(100, _Payload(100))
        self.assertEqual(event, ReleaseEvent.DROPPED_PERMANENT)
        self.assertIsNone(payload)
        self.assertEqual(len(queue), 0)
        self.assertEqual(queue.stats.dropped_permanent_tf_failure, 1)

    def test_permanent_failure_discovered_while_queued_drops_without_full_wait(self):
        # D1 is future-pending (enqueues); once queued, its frame turns out
        # to be permanently broken (e.g. TF tree disconnected) -- it must
        # drop immediately on the next drain(), not wait for max_wait_s.
        state = {"mode": "future"}

        def can_transform(payload):
            if state["mode"] == "future":
                return False, "extrapolation into the future"
            return False, "Invalid frame ID does not exist"

        clock = _FakeClock()
        queue = DeferredDetectionQueue(
            max_pending=8, max_wait_s=10.0, can_transform=can_transform, monotonic=clock
        )
        queue.offer(100, _Payload(100))
        self.assertEqual(len(queue), 1)

        state["mode"] = "permanent"
        clock.advance(0.01)  # far less than max_wait_s
        released = queue.drain()
        self.assertEqual(len(released), 1)
        self.assertEqual(released[0][0], ReleaseEvent.DROPPED_PERMANENT)
        self.assertEqual(len(queue), 0)


# Phase 24: queue bound -- deterministic oldest-drop overflow, no growth.
class QueueBoundTest(unittest.TestCase):
    def test_overflow_drops_oldest_unresolved_deterministically(self):
        can_transform = lambda payload: (False, "extrapolation into the future")
        queue = DeferredDetectionQueue(max_pending=3, max_wait_s=100.0, can_transform=can_transform)
        for i in range(5):
            queue.offer(100 + i, _Payload(100 + i))
        self.assertEqual(len(queue), 3)
        self.assertEqual(queue.stats.dropped_queue_overflow, 2)
        remaining_stamps = [p.stamp_ns for p in (item.payload for item in queue._queue)]
        # The two oldest (100, 101) were evicted; 102/103/104 remain, oldest-first.
        self.assertEqual(remaining_stamps, [102, 103, 104])
        self.assertEqual(queue.stats.max_observed_queue_depth, 3)


# Phase 25: no duplicate -- exactly one release per admitted detection.
class NoDuplicateTest(unittest.TestCase):
    def test_repeated_drain_calls_never_release_the_same_entry_twice(self):
        available = set()
        can_transform = _make_can_transform(available)
        queue = DeferredDetectionQueue(max_pending=8, max_wait_s=5.0, can_transform=can_transform)
        queue.offer(100, _Payload(100))

        available.add(100)
        first = queue.drain()
        second = queue.drain()  # immediately again, e.g. two triggers close together
        self.assertEqual(len(first), 1)
        self.assertEqual(second, [])  # nothing left to release
        self.assertEqual(queue.stats.processed_deferred, 1)

    def test_offer_never_double_processes_when_ready_immediately(self):
        available = {100}
        can_transform = _make_can_transform(available)
        queue = DeferredDetectionQueue(max_pending=8, max_wait_s=5.0, can_transform=can_transform)
        event, payload = queue.offer(100, _Payload(100))
        self.assertEqual(event, ReleaseEvent.READY)
        # A caller processing an immediate READY result must never also see
        # it emerge from drain() -- it was never enqueued.
        self.assertEqual(queue.drain(), [])


# Phase 26: source stamp preserved through deferral.
class SourceStampPreservedTest(unittest.TestCase):
    def test_released_payload_carries_the_original_stamp_unchanged(self):
        available = set()
        can_transform = _make_can_transform(available)
        clock = _FakeClock()
        queue = DeferredDetectionQueue(
            max_pending=8, max_wait_s=5.0, can_transform=can_transform, monotonic=clock
        )
        queue.offer(12345, _Payload(12345))
        clock.advance(2.0)  # processing is artificially delayed
        available.add(12345)
        released = queue.drain()
        self.assertEqual(released[0][1].stamp_ns, 12345)


class ConstructionValidationTest(unittest.TestCase):
    def test_rejects_non_positive_max_pending(self):
        with self.assertRaises(ValueError):
            DeferredDetectionQueue(max_pending=0, max_wait_s=1.0, can_transform=lambda p: (True, ""))

    def test_rejects_non_positive_max_wait(self):
        with self.assertRaises(ValueError):
            DeferredDetectionQueue(max_pending=1, max_wait_s=0.0, can_transform=lambda p: (True, ""))


if __name__ == "__main__":
    unittest.main()
