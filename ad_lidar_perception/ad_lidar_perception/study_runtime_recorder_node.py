#!/usr/bin/env python3
"""GT-free descriptive recorder for the fixed-window study replay."""

from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

import rclpy
from ad_interfaces.msg import PredictedObjectArray
from autoware_perception_msgs.msg import DetectedObjects, TrackedObjects
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock


def _seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _summary(values):
    return {
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p90": _percentile(values, 0.90),
    }


class StudyRuntimeRecorder(Node):
    def __init__(self):
        super().__init__("ad_study_runtime_recorder")
        self.declare_parameter("variant", "")
        self.declare_parameter("output_path", "")
        self.declare_parameter("duration_sec", 60.0)
        self.declare_parameter("source_start_sec", 0.0)
        self.variant = str(self.get_parameter("variant").value)
        self.output_path = Path(str(self.get_parameter("output_path").value)).expanduser()
        self.duration = float(self.get_parameter("duration_sec").value)
        requested_start = float(self.get_parameter("source_start_sec").value)
        if not self.output_path.is_absolute() or self.duration <= 0.0:
            raise ValueError("output_path must be absolute and duration_sec must be positive")
        self.start_time = requested_start if requested_start > 0.0 else None
        self.started = False
        self.finished = False
        self.counts = {"detected": [], "tracked": [], "predicted": [], "dynamic_ogm": []}
        self.stamps = {key: [] for key in self.counts}
        self.dynamic_nonempty = 0
        self.seen_ids = set()
        self.previous_ids = set()
        self.births = 0
        self.deletions = 0
        self.first_seen = {}
        self.last_seen = {}
        self.last_state = {}
        self.jitter_by_id = {}
        self.tracked_frame_index = 0
        self.create_subscription(Clock, "/clock", self._on_clock, qos_profile_sensor_data)
        self.create_subscription(DetectedObjects, "/ad/perception/objects/detected", self._detected, qos_profile_sensor_data)
        self.create_subscription(TrackedObjects, "/ad/perception/objects/tracked", self._tracked, qos_profile_sensor_data)
        self.create_subscription(PredictedObjectArray, "/ad/perception/objects/predicted", self._predicted, qos_profile_sensor_data)
        self.create_subscription(OccupancyGrid, "/ad/perception/occupancy/dynamic", self._dynamic, qos_profile_sensor_data)

    def _within_window(self):
        return self.started and not self.finished

    def _on_clock(self, message):
        now = _seconds(message.clock)
        if self.start_time is None:
            self.start_time = now
        if not self.started and now >= self.start_time:
            self.started = True
            self.get_logger().info(
                f"study runtime window started at source time {self.start_time:.9f} for {self.duration:.1f}s"
            )
        if self.started and not self.finished and now - self.start_time >= self.duration:
            self._write(now)

    def _record(self, key, header, count):
        stamp = _seconds(header.stamp)
        if (
            not self._within_window()
            or stamp < self.start_time
            or stamp >= self.start_time + self.duration
        ):
            return False
        self.counts[key].append(int(count))
        self.stamps[key].append(stamp)
        return True

    def _detected(self, message):
        self._record("detected", message.header, len(message.objects))

    def _predicted(self, message):
        self._record("predicted", message.header, len(message.objects))

    def _dynamic(self, message):
        if self._record("dynamic_ogm", message.header, len(message.data)):
            self.dynamic_nonempty += int(any(value > 0 for value in message.data))

    def _tracked(self, message):
        if not self._record("tracked", message.header, len(message.objects)):
            return
        stamp = _seconds(message.header.stamp)
        current_ids = set()
        for tracked in message.objects:
            track_id = bytes(tracked.object_id.uuid).hex()
            current_ids.add(track_id)
            if track_id not in self.seen_ids:
                self.seen_ids.add(track_id)
                self.births += 1
                self.first_seen[track_id] = stamp
            self.last_seen[track_id] = stamp
            pose = tracked.kinematics.pose_with_covariance.pose.position
            velocity = tracked.kinematics.twist_with_covariance.twist.linear
            previous = self.last_state.get(track_id)
            if previous is not None and previous[0] == self.tracked_frame_index - 1:
                dt = stamp - previous[1]
                if dt > 0.0:
                    residual = math.hypot(
                        pose.x - (previous[2] + previous[4] * dt),
                        pose.y - (previous[3] + previous[5] * dt),
                    )
                    self.jitter_by_id.setdefault(track_id, []).append(residual)
            self.last_state[track_id] = (
                self.tracked_frame_index, stamp, pose.x, pose.y, velocity.x, velocity.y
            )
        self.deletions += len(self.previous_ids - current_ids)
        self.previous_ids = current_ids
        self.tracked_frame_index += 1

    def _hz(self, key):
        stamps = self.stamps[key]
        span = max(stamps) - min(stamps) if len(stamps) > 1 else 0.0
        return (len(stamps) - 1) / span if span > 0.0 else None

    def _write(self, end_time):
        self.finished = True
        duration = min(self.duration, max(0.0, end_time - self.start_time))
        jitter = [
            value for values in self.jitter_by_id.values() if len(values) >= 2
            for value in values
        ]
        lifetimes = [self.last_seen[key] - self.first_seen[key] for key in self.seen_ids]
        result = {
            "schema_version": 1,
            "variant": self.variant,
            "source_start_sec": self.start_time,
            "source_duration_sec": duration,
            "gt_available": False,
            "detection": {
                "messages": len(self.counts["detected"]),
                "detections_total": sum(self.counts["detected"]),
                "detections_per_frame": _summary(self.counts["detected"]),
                "publish_hz": self._hz("detected"),
            },
            "tracking": {
                "messages": len(self.counts["tracked"]),
                "nonempty_frames": sum(value > 0 for value in self.counts["tracked"]),
                "tracks_total": sum(self.counts["tracked"]),
                "tracks_per_frame": _summary(self.counts["tracked"]),
                "publish_hz": self._hz("tracked"),
                "unique_track_ids": len(self.seen_ids),
                "track_births": self.births,
                "track_deletions_inferred": self.deletions,
                "id_churn_created_per_min": self.births * 60.0 / duration if duration else None,
                "id_churn_deleted_per_min": self.deletions * 60.0 / duration if duration else None,
                "observed_lifetime_sec_mean": statistics.fmean(lifetimes) if lifetimes else None,
                "observed_lifetime_sec_median": statistics.median(lifetimes) if lifetimes else None,
                "motion_jitter_median_m": statistics.median(jitter) if jitter else None,
                "motion_jitter_p90_m": _percentile(jitter, 0.90),
                "motion_jitter_sample_count": len(jitter),
            },
            "prediction": {
                "messages": len(self.counts["predicted"]),
                "publish_hz": self._hz("predicted"),
            },
            "dynamic_occupancy": {
                "messages": len(self.counts["dynamic_ogm"]),
                "nonempty_frames": self.dynamic_nonempty,
                "nonempty_ratio": self.dynamic_nonempty / len(self.counts["dynamic_ogm"])
                if self.counts["dynamic_ogm"] else None,
            },
            "notes": [
                "No GT: motion jitter is a temporal smoothness diagnostic, not position error.",
                "Track deletions and ID churn are inferred lifecycle activity, not ID switches.",
                "Hz uses source header-stamp span, not wall-clock replay rate.",
            ],
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        markdown = self.output_path.with_suffix(".md")
        d, t = result["detection"], result["tracking"]
        markdown.write_text(
            "| Variant | Detection Hz | Detections/frame | Tracking Hz | Tracks/frame | Unique IDs | ID churn/min | Jitter median/p90 (m) | Prediction Hz | Dynamic OGM non-empty |\n"
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
            f"| {self.variant} | {d['publish_hz'] or 0:.3f} | {d['detections_per_frame']['mean'] or 0:.3f} | "
            f"{t['publish_hz'] or 0:.3f} | {t['tracks_per_frame']['mean'] or 0:.3f} | {t['unique_track_ids']} | "
            f"{t['id_churn_created_per_min'] or 0:.2f} | {t['motion_jitter_median_m'] or 0:.4f} / {t['motion_jitter_p90_m'] or 0:.4f} | "
            f"{result['prediction']['publish_hz'] or 0:.3f} | {(result['dynamic_occupancy']['nonempty_ratio'] or 0) * 100:.1f}% |\n",
            encoding="utf-8",
        )
        self.get_logger().info(f"study runtime metrics written: {self.output_path}")


def main():
    rclpy.init()
    node = StudyRuntimeRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
