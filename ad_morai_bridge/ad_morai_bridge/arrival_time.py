import math


NANOSECONDS_PER_SECOND = 1_000_000_000


class ReceiptClockMapper:
    """Map steady-clock socket receipt times onto one stable ROS epoch."""

    def __init__(self, *, sampled_monotonic: float, sampled_ros_ns: int) -> None:
        if not math.isfinite(sampled_monotonic):
            raise ValueError("monotonic timestamp must be finite")
        if sampled_ros_ns < 0:
            raise ValueError("ROS timestamp must not precede its epoch")
        self._ros_minus_monotonic_ns = (
            sampled_ros_ns
            - round(sampled_monotonic * NANOSECONDS_PER_SECOND)
        )

    def stamp(self, received_monotonic: float) -> tuple[int, int]:
        if not math.isfinite(received_monotonic):
            raise ValueError("monotonic timestamp must be finite")
        received_ros_ns = max(
            0,
            self._ros_minus_monotonic_ns
            + round(received_monotonic * NANOSECONDS_PER_SECOND),
        )
        return divmod(received_ros_ns, NANOSECONDS_PER_SECOND)
