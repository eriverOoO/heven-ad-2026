"""Capture session state machine (ROS-free core).

Drives: wait-for-inputs -> settle -> capture -> terminate. The rclpy node
(:mod:`ad_morai_bridge_dev.dataset.capture_node`) is a thin shell that pumps
deserialised messages into :class:`CaptureSession`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ad_morai_bridge_dev.dataset.frame_builder import FrameAssembler
from ad_morai_bridge_dev.dataset.schema import (
    RUN_STATUS_ABORTED,
    RUN_STATUS_COMPLETE,
    RUN_STATUS_ERROR,
    SkewConfig,
)
from ad_morai_bridge_dev.dataset.sync import TimeBuffer, classify_lidar_stamp
from ad_morai_bridge_dev.dataset.transforms import StaticExtrinsics
from ad_morai_bridge_dev.dataset.writer import RunWriter


def _stamp_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


@dataclass(frozen=True)
class CaptureLimits:
    frame_count: int | None = None
    duration_ns: int | None = None
    settle_ns: int = 1_000_000_000

    def __post_init__(self) -> None:
        if self.frame_count is None and self.duration_ns is None:
            raise ValueError("a frame_count or duration_ns limit is required")


class CaptureSession:
    PHASE_WAIT = "waiting_for_inputs"
    PHASE_SETTLE = "settling"
    PHASE_CAPTURE = "capturing"
    PHASE_DONE = "done"

    def __init__(
        self,
        writer: RunWriter,
        static: StaticExtrinsics,
        skew: SkewConfig,
        limits: CaptureLimits,
        scenario_file: str,
        scenario_sha256: str,
        buffer_capacity: int = 512,
        require_ego: bool = True,
        require_actor_gt: bool = True,
        require_tf: bool = True,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self._writer = writer
        self._assembler = FrameAssembler(
            writer.run_id, static, skew, scenario_file, scenario_sha256
        )
        self._skew = skew
        self._limits = limits
        self._log = logger or (lambda _msg: None)
        self._require = {
            "ego": require_ego,
            "actor": require_actor_gt,
            "tf": require_tf,
        }
        self._ego = TimeBuffer(buffer_capacity)
        self._actor = TimeBuffer(buffer_capacity)
        self._tf = TimeBuffer(buffer_capacity)
        self.phase = self.PHASE_WAIT
        self._settle_start_ns: int | None = None
        self._first_capture_ns: int | None = None
        self._prev_lidar_ns: int | None = None
        self._point_field_schema: list[dict[str, Any]] | None = None
        self._rejected: dict[str, int] = {}
        self._finalized = False

    # -- source ingestion ------------------------------------------------
    def on_ego(self, message: Any) -> None:
        self._ego.add(_stamp_ns(message.header.stamp), message)

    def on_actor_gt(self, message: Any) -> None:
        self._actor.add(_stamp_ns(message.header.stamp), message)

    def on_tf(self, message: Any) -> None:
        for item in message.transforms:
            if (
                str(item.header.frame_id).lstrip("/") == "odom"
                and str(item.child_frame_id).lstrip("/") == "base_link"
            ):
                self._tf.add(_stamp_ns(item.header.stamp), item)

    # -- anchor --------------------------------------------------------
    def _inputs_ready(self) -> bool:
        ok = True
        if self._require["ego"] and len(self._ego) == 0:
            ok = False
        if self._require["actor"] and len(self._actor) == 0:
            ok = False
        if self._require["tf"] and len(self._tf) == 0:
            ok = False
        return ok

    def _reject(self, reason: str) -> None:
        self._rejected[reason] = self._rejected.get(reason, 0) + 1

    def on_lidar(self, message: Any) -> str:
        """Process one LiDAR frame. Returns the action taken."""

        if self.phase == self.PHASE_DONE:
            return "ignored_after_done"

        stamp_ns = _stamp_ns(message.header.stamp)
        klass = classify_lidar_stamp(stamp_ns, self._prev_lidar_ns)
        if klass in ("nonpositive", "duplicate", "backward"):
            self._reject(f"lidar_stamp_{klass}")
            return f"rejected_{klass}"
        self._prev_lidar_ns = stamp_ns

        if self.phase == self.PHASE_WAIT:
            if not self._inputs_ready():
                self._reject("inputs_not_ready")
                return "waiting"
            self.phase = self.PHASE_SETTLE
            self._settle_start_ns = stamp_ns
            self._log("inputs ready; settling")

        if self.phase == self.PHASE_SETTLE:
            assert self._settle_start_ns is not None
            if stamp_ns - self._settle_start_ns < self._limits.settle_ns:
                self._reject("settle_period")
                return "settling"
            self.phase = self.PHASE_CAPTURE
            self._first_capture_ns = stamp_ns
            self._log("settle complete; capturing")

        if self.phase != self.PHASE_CAPTURE:
            return "waiting"

        frame_index = self._writer.frame_count
        ego_match = self._ego.nearest(stamp_ns, self._skew.max_ego_skew_ns)
        actor_match = self._actor.nearest(stamp_ns, self._skew.max_actor_gt_skew_ns)
        tf_match = self._tf.nearest(stamp_ns, self._skew.max_tf_skew_ns)

        if self._point_field_schema is None:
            self._point_field_schema = FrameAssembler.field_schema(message)

        assembled = self._assembler.assemble(
            frame_index, message, ego_match, actor_match, tf_match
        )
        self._writer.write_frame(
            assembled.record,
            assembled.point_arrays,
            assembled.gt_payload,
            assembled.ego_payload,
            assembled.tf_payload,
        )
        self._log(
            f"frame {frame_index}: {assembled.record.point_count} pts, "
            f"tracking_gt={assembled.record.valid_for_tracking_gt} "
            f"detection_gt={assembled.record.valid_for_detection_gt}"
        )

        if self._reached_limit(stamp_ns):
            self.phase = self.PHASE_DONE
            return "captured_final"
        return "captured"

    def _reached_limit(self, stamp_ns: int) -> bool:
        if (
            self._limits.frame_count is not None
            and self._writer.frame_count >= self._limits.frame_count
        ):
            return True
        if (
            self._limits.duration_ns is not None
            and self._first_capture_ns is not None
            and stamp_ns - self._first_capture_ns >= self._limits.duration_ns
        ):
            return True
        return False

    @property
    def complete(self) -> bool:
        return self.phase == self.PHASE_DONE

    @property
    def finalized(self) -> bool:
        return self._finalized

    # -- finalisation --------------------------------------------------
    def finalize(self, reason: str, error: str | None = None) -> dict[str, Any]:
        if self._finalized:
            raise RuntimeError("session already finalized")
        self._finalized = True
        if error is not None:
            status = RUN_STATUS_ERROR
        elif self.phase == self.PHASE_DONE:
            status = RUN_STATUS_COMPLETE
        else:
            status = RUN_STATUS_ABORTED
        extra = {
            "point_field_schema": self._point_field_schema or [],
            "rejected_frames": dict(sorted(self._rejected.items())),
            "final_phase": self.phase,
        }
        if error is not None:
            extra["error"] = error
        return self._writer.finalize(status, reason, extra)
