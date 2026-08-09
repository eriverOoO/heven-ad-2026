"""Pure SLAM health checks shared by the runtime watchdog and tests."""

from dataclasses import dataclass
import math


@dataclass
class SlamHealthGuard:
    """Detect pose jumps and stale SLAM odometry while the vehicle is moving."""

    max_pose_step_m: float = 1.0
    max_odom_speed_mps: float = 2.0
    max_odom_age_sec: float = 1.0
    moving_speed_threshold_mps: float = 0.15
    startup_grace_sec: float = 5.0

    def __post_init__(self) -> None:
        if self.max_pose_step_m <= 0.0:
            raise ValueError("max_pose_step_m must be positive")
        if self.max_odom_age_sec <= 0.0:
            raise ValueError("max_odom_age_sec must be positive")
        if self.max_odom_speed_mps <= 0.0:
            raise ValueError("max_odom_speed_mps must be positive")
        if self.moving_speed_threshold_mps < 0.0:
            raise ValueError("moving_speed_threshold_mps cannot be negative")
        if self.startup_grace_sec < 0.0:
            raise ValueError("startup_grace_sec cannot be negative")

        self._armed_since: float | None = None
        self._last_odom_time: float | None = None
        self._last_xy: tuple[float, float] | None = None
        self._vehicle_speed_mps = 0.0

    def set_armed(self, armed: bool, now: float) -> None:
        if armed and self._armed_since is None:
            self._armed_since = now
        elif not armed:
            self._armed_since = None

    def set_vehicle_speed(self, speed_mps: float) -> None:
        self._vehicle_speed_mps = float(speed_mps)

    def observe_odom(self, x: float, y: float, now: float) -> str | None:
        if not all(math.isfinite(value) for value in (x, y, now)):
            return "SLAM odometry contains a non-finite value"

        reason = None
        if self._last_xy is not None:
            step = math.hypot(x - self._last_xy[0], y - self._last_xy[1])
            if step > self.max_pose_step_m:
                reason = (
                    f"SLAM pose jumped {step:.3f} m "
                    f"(limit {self.max_pose_step_m:.3f} m)"
                )
            elapsed = now - self._last_odom_time
            if elapsed > 0.0 and step / elapsed > self.max_odom_speed_mps:
                reason = (
                    f"SLAM odometry speed is {step / elapsed:.3f} m/s "
                    f"(limit {self.max_odom_speed_mps:.3f} m/s)"
                )

        self._last_xy = (x, y)
        self._last_odom_time = now
        return reason

    def evaluate(self, now: float) -> str | None:
        if self._armed_since is None:
            return None
        if now - self._armed_since < self.startup_grace_sec:
            return None
        if abs(self._vehicle_speed_mps) <= self.moving_speed_threshold_mps:
            return None
        if self._last_odom_time is None:
            return "SLAM odometry is missing while the armed vehicle is moving"

        age = now - self._last_odom_time
        if age > self.max_odom_age_sec:
            return (
                f"SLAM odometry is stale by {age:.3f} s "
                f"(limit {self.max_odom_age_sec:.3f} s)"
            )
        return None
