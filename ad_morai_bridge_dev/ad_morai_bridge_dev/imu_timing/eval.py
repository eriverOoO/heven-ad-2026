"""Passive ROS 2 collection for the MORAI IMU timing contract.

The bridge is deliberately never controlled here.  This node only subscribes
to the four competition surfaces and turns frozen observations into the pure
``imu_timing_metrics`` contract inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import time
from typing import Any, Sequence

from ad_morai_bridge_dev.imu_timing.metrics import (
    BridgeCounterSnapshot,
    TimingAudit,
    TimingWindow,
    evaluate_transport_contract,
    summarize_timing_window,
)


def _ros_dependencies() -> tuple[object, object, object, object, object, tuple[object, ...]]:
    """Load ROS only when the executable constructs its runtime boundary."""
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import Imu

    from ad_morai_interfaces.msg import BridgeStatistics, ImuPacket, SensorTiming

    return (
        rclpy,
        Node,
        QoSProfile,
        ReliabilityPolicy,
        qos_profile_sensor_data,
        (Imu, ImuPacket, SensorTiming, BridgeStatistics),
    )


NORMALIZED_TOPIC = "/ad/sensors/imu/data"
FULL_TOPIC = "/ad/sensors/imu/full"
TIMING_TOPIC = "/ad/sensors/timing"
STATISTICS_TOPIC = "/ad/udp/statistics"
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_NANOSECONDS_PER_SECOND = 1_000_000_000
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


@dataclass(frozen=True)
class ReportArtifact:
    run_directory: Path
    report: Path


def _positive_finite_seconds(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return numeric


def _seconds_to_ns(value: float, name: str) -> int:
    numeric = _positive_finite_seconds(value, name)
    result = int(round(numeric * _NANOSECONDS_PER_SECOND))
    if result <= 0:
        raise ValueError(f"{name} is too small to represent in nanoseconds")
    return result


def _stamp_ns(header: object) -> int:
    stamp = getattr(header, "stamp", None)
    seconds = getattr(stamp, "sec", None)
    nanoseconds = getattr(stamp, "nanosec", None)
    if (
        isinstance(seconds, bool)
        or isinstance(nanoseconds, bool)
        or not isinstance(seconds, int)
        or not isinstance(nanoseconds, int)
        or seconds < 0
        or not 0 <= nanoseconds < _NANOSECONDS_PER_SECOND
    ):
        raise ValueError("message header stamp must be a non-negative ROS time")
    return seconds * _NANOSECONDS_PER_SECOND + nanoseconds


def _counter_snapshot(message: object) -> BridgeCounterSnapshot:
    values: dict[str, int] = {}
    for name in _COUNTER_FIELDS:
        value = getattr(message, name, None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"BridgeStatistics.{name} must be a non-negative integer")
        values[name] = value
    return BridgeCounterSnapshot(**values)


def _timing_audit(message: object) -> TimingAudit:
    return TimingAudit(
        header_stamp_ns=_stamp_ns(getattr(message, "header", None)),
        stream=getattr(message, "stream", None),
        source_valid=getattr(message, "source_valid", None),
        source_selected=getattr(message, "source_selected", None),
        source_rejected=getattr(message, "source_rejected", None),
        duplicate=getattr(message, "duplicate", None),
        stamp_regression=getattr(message, "stamp_regression", None),
        normalized_published=getattr(message, "normalized_published", None),
    )


def validate_run_id(run_id: object) -> str:
    if not isinstance(run_id, str) or _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("run_id is unsafe")
    return run_id


def _configured_data_root() -> Path:
    raw = os.environ.get("AD_DATA_DIR")
    if raw is None or not raw.strip():
        raise ValueError("AD_DATA_DIR must be explicitly configured")
    configured = Path(raw).expanduser()
    if not configured.is_absolute():
        raise ValueError("AD_DATA_DIR must be an absolute path")
    try:
        root = configured.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"AD_DATA_DIR does not exist: {configured}") from exc
    if not root.is_dir():
        raise ValueError("AD_DATA_DIR must name a directory")
    return root


_DIRECTORY_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _open_directory_at(parent_fd: int, name: str, *, create: bool) -> int:
    try:
        return os.open(name, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_fd)
    except FileNotFoundError:
        if not create:
            raise ValueError(f"required artifact directory is missing: {name}") from None
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except FileExistsError:
            pass
        try:
            return os.open(name, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_fd)
        except OSError as exc:
            raise ValueError(f"unsafe artifact directory: {name}") from exc
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            raise ValueError(f"unsafe artifact directory: {name}") from exc
        raise


def _open_directory_walk(root: Path, parts: tuple[str, ...], *, create: bool) -> int:
    """Return a held FD reached without following any artifact component."""
    try:
        current = os.open(root, _DIRECTORY_OPEN_FLAGS)
    except OSError as exc:
        raise ValueError(f"unable to open AD_DATA_DIR safely: {root}") from exc
    try:
        for part in parts:
            next_fd = _open_directory_at(current, part, create=create)
            os.close(current)
            current = next_fd
        return current
    except BaseException:
        os.close(current)
        raise


def create_report_artifact(run_id: str) -> ReportArtifact:
    """Reserve one non-reusable report directory below ``AD_DATA_DIR``."""
    root = _configured_data_root()
    run_id = validate_run_id(run_id)
    base_parts = ("experiments", "morai_imu_timing")
    base_fd = _open_directory_walk(root, base_parts, create=True)
    try:
        try:
            os.mkdir(run_id, mode=0o700, dir_fd=base_fd)
            os.fsync(base_fd)
        except FileExistsError as exc:
            raise FileExistsError("run directory already exists") from exc
        run_fd = _open_directory_at(base_fd, run_id, create=False)
        os.close(run_fd)
    finally:
        os.close(base_fd)
    run_directory = root / "experiments" / "morai_imu_timing" / run_id
    report = run_directory / "report.json"
    return ReportArtifact(run_directory=run_directory, report=report)


def _validate_report_target(path: Path) -> tuple[Path, tuple[str, ...]]:
    root = _configured_data_root()
    target = Path(path)
    if not target.is_absolute():
        raise ValueError("report path must be absolute")
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise ValueError("report path resolves outside AD_DATA_DIR") from exc
    if (
        len(relative.parts) != 4
        or relative.parts[:2] != ("experiments", "morai_imu_timing")
        or relative.parts[3] != "report.json"
    ):
        raise ValueError("report path must be inside a MORAI IMU run directory")
    validate_run_id(relative.parts[2])
    return root, relative.parts[:3]


def sensor_profile_provenance(path: Path) -> dict[str, object]:
    """Validate an operator-supplied MORAI profile and record provenance only."""
    source = Path(path)
    try:
        descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError as exc:
        raise ValueError(f"sensor profile does not exist: {source}") from exc
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError("sensor profile symlink is not allowed") from exc
        raise ValueError(f"invalid sensor profile: {source}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("sensor profile must be a regular file")
        canonical_path = os.readlink(f"/proc/self/fd/{descriptor}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        content = b"".join(chunks)
        document = json.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid sensor profile: {source}") from exc
    finally:
        os.close(descriptor)
    imu_list = document.get("IMUList") if isinstance(document, dict) else None
    if not isinstance(imu_list, list) or not imu_list:
        raise ValueError("sensor profile IMUList must be a non-empty list")
    periods: list[dict[str, object]] = []
    for index, item in enumerate(imu_list):
        configuration = item.get("ic") if isinstance(item, dict) else None
        period = configuration.get("sensorPeriod") if isinstance(configuration, dict) else None
        if isinstance(period, bool) or not isinstance(period, (int, float)):
            raise ValueError(f"IMUList/{index}/ic/sensorPeriod must be finite and positive")
        try:
            period_value = float(period)
        except OverflowError as exc:
            raise ValueError(f"IMUList/{index}/ic/sensorPeriod must be finite and positive") from exc
        if not math.isfinite(period_value) or period_value <= 0.0:
            raise ValueError(f"IMUList/{index}/ic/sensorPeriod must be finite and positive")
        identifiers = [item[key] for key in ("m_SensorUniqueID", "UNIQUEID") if key in item]
        if (
            not identifiers
            or any(isinstance(value, bool) or not isinstance(value, int) for value in identifiers)
            or len(set(identifiers)) != 1
        ):
            raise ValueError(f"IMUList/{index} must contain one agreeing integer sensor identifier")
        periods.append(
            {
                "index": index,
                "unique_id": identifiers[0],
                "sensor_period_sec": period_value,
            }
        )
    return {
        "path": canonical_path,
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
        "imu_periods": periods,
    }


class TimingCollector:
    """Accumulate callbacks until one immutable measurement window is frozen."""

    def __init__(self, *, warmup_ns: int, duration_ns: int, drain_ns: int):
        self._warmup_ns = _seconds_to_ns(warmup_ns / _NANOSECONDS_PER_SECOND, "warmup")
        self._duration_ns = _seconds_to_ns(duration_ns / _NANOSECONDS_PER_SECOND, "duration")
        self._drain_ns = _seconds_to_ns(drain_ns / _NANOSECONDS_PER_SECOND, "drain")
        self._phase = "waiting"
        self._surfaces = {name: False for name in ("normalized", "full", "timing", "statistics")}
        self._ready_ns: int | None = None
        self._measurement_start_ns: int | None = None
        self._measurement_end_ns: int | None = None
        self._freeze_ns: int | None = None
        self._header_range: tuple[int, int] | None = None
        self._normalized: list[tuple[int, int]] = []
        self._full: list[tuple[int, int]] = []
        self._timing: list[tuple[int, TimingAudit]] = []
        self._statistics: list[tuple[int, BridgeCounterSnapshot]] = []
        self._frozen: TimingWindow | None = None
        self._startup_timed_out = False

    @property
    def phase(self) -> str:
        return self._phase

    def _mark_surface(self, name: str, receipt_ns: int) -> None:
        if self._phase == "frozen":
            return
        self._surfaces[name] = True
        if self._phase == "waiting" and all(self._surfaces.values()):
            self._ready_ns = receipt_ns
            self._phase = "warmup"

    def on_normalized(self, message: object, *, receipt_ns: int) -> None:
        if self._phase == "frozen":
            return
        self._mark_surface("normalized", receipt_ns)
        self._normalized.append((receipt_ns, _stamp_ns(getattr(message, "header", None))))

    def on_full(self, message: object, *, receipt_ns: int) -> None:
        if self._phase == "frozen":
            return
        self._mark_surface("full", receipt_ns)
        self._full.append((receipt_ns, _stamp_ns(getattr(message, "header", None))))

    def on_timing(self, message: object, *, receipt_ns: int) -> None:
        if self._phase == "frozen" or getattr(message, "stream", None) != "imu":
            return
        self._mark_surface("timing", receipt_ns)
        self._timing.append((receipt_ns, _timing_audit(message)))

    def on_statistics(self, message: object, *, receipt_ns: int) -> None:
        if self._phase == "frozen" or getattr(message, "stream", None) != "imu":
            return
        self._mark_surface("statistics", receipt_ns)
        self._statistics.append((receipt_ns, _counter_snapshot(message)))

    def tick(self, receipt_ns: int) -> None:
        if self._phase == "frozen" or self._phase == "waiting":
            return
        assert self._ready_ns is not None
        warmup_end = self._ready_ns + self._warmup_ns
        if self._phase == "warmup":
            if receipt_ns < warmup_end:
                return
            self._measurement_start_ns = warmup_end
            self._measurement_end_ns = warmup_end + self._duration_ns
            self._phase = "measuring"
        assert self._measurement_end_ns is not None
        if self._phase == "measuring" and receipt_ns >= self._measurement_end_ns:
            self._phase = "draining"
        if self._phase == "draining" and receipt_ns >= self._measurement_end_ns + self._drain_ns:
            self._freeze(receipt_ns)

    def force_startup_timeout(self, receipt_ns: int) -> None:
        if self._phase != "waiting":
            raise ValueError("startup timeout is valid only before all surfaces arrive")
        self._startup_timed_out = True
        self._measurement_start_ns = receipt_ns
        self._measurement_end_ns = receipt_ns + self._duration_ns
        self._freeze(receipt_ns)

    def _freeze(self, receipt_ns: int) -> None:
        if self._frozen is not None:
            return
        assert self._measurement_start_ns is not None
        assert self._measurement_end_ns is not None
        start = self._measurement_start_ns
        end = self._measurement_end_ns
        pre = [item for item in self._statistics if item[0] < start]
        post = [item for item in self._statistics if item[0] >= end]
        start_counters = pre[-1][1] if pre else BridgeCounterSnapshot(**{name: 0 for name in _COUNTER_FIELDS})
        end_counters = post[-1][1] if post else start_counters
        in_window_timing = [
            audit.header_stamp_ns
            for when, audit in self._timing
            if start <= when < end
        ]
        if in_window_timing:
            self._header_range = (min(in_window_timing), max(in_window_timing))
            lower_header, upper_header = self._header_range
            header_in_range = lambda stamp: lower_header <= stamp <= upper_header
        else:
            header_in_range = lambda _stamp: False
        self._frozen = TimingWindow(
            normalized_headers=tuple(stamp for _when, stamp in self._normalized if header_in_range(stamp)),
            full_headers=tuple(stamp for _when, stamp in self._full if header_in_range(stamp)),
            timing_headers=tuple(audit.header_stamp_ns for _when, audit in self._timing if header_in_range(audit.header_stamp_ns)),
            audits=tuple(audit for _when, audit in self._timing if header_in_range(audit.header_stamp_ns)),
            start_counters=start_counters,
            end_counters=end_counters,
            measurement_duration_ns=end - start,
            normalized_discovered=self._surfaces["normalized"],
            full_discovered=self._surfaces["full"],
            timing_discovered=self._surfaces["timing"],
            statistics_discovered=self._surfaces["statistics"] and bool(pre) and bool(post),
        )
        self._freeze_ns = receipt_ns
        self._phase = "frozen"

    def frozen_window(self) -> TimingWindow:
        if self._frozen is None:
            raise ValueError("measurement window has not frozen")
        return self._frozen

    def lifecycle(self) -> dict[str, int | bool | None]:
        return {
            "ready_monotonic_ns": self._ready_ns,
            "measurement_start_monotonic_ns": self._measurement_start_ns,
            "measurement_end_monotonic_ns": self._measurement_end_ns,
            "header_range_start_ns": self._header_range[0] if self._header_range else None,
            "header_range_end_ns": self._header_range[1] if self._header_range else None,
            "freeze_monotonic_ns": self._freeze_ns,
            "startup_timed_out": self._startup_timed_out,
        }


def _encode_json(value: Any) -> Any:
    if isinstance(value, Counter):
        return [
            {"stamp_ns": stamp, "count": count}
            for stamp, count in sorted(value.items())
        ]
    if isinstance(value, dict):
        return {str(key): _encode_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_encode_json(item) for item in value]
    if isinstance(value, list):
        return [_encode_json(item) for item in value]
    return value


def evaluate_frozen_window(window: TimingWindow, *, expected_rate_hz: int) -> tuple[dict[str, object], int]:
    summary = summarize_timing_window(window, expected_rate_hz)
    exit_code, reasons = evaluate_transport_contract(summary)
    return {
        "summary": _encode_json(summary),
        "reasons": list(reasons),
        "advisories": list(summary["advisories"]),
    }, exit_code


def write_report(path: Path, payload: dict[str, object]) -> None:
    root, directory_parts = _validate_report_target(path)
    try:
        serialized = (json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"report payload is not strict JSON: {exc}") from exc
    run_fd = _open_directory_walk(root, directory_parts, create=False)
    temporary_name = f".report-{secrets.token_hex(16)}.tmp"
    temporary_fd: int | None = None
    try:
        temporary_fd = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=run_fd,
        )
        with os.fdopen(temporary_fd, "wb", closefd=True) as temporary:
            temporary_fd = None
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            os.link(
                temporary_name,
                "report.json",
                src_dir_fd=run_fd,
                dst_dir_fd=run_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise FileExistsError("report already exists") from exc
        os.fsync(run_fd)
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        try:
            os.unlink(temporary_name, dir_fd=run_fd)
        except FileNotFoundError:
            pass
        os.close(run_fd)


class ImuTimingEvaluationNode:
    """Thin ROS adapter; all contract decisions remain in the pure evaluator."""

    def __init__(self, collector: TimingCollector):
        (
            _rclpy,
            node_type,
            qos_profile_type,
            reliability_policy,
            sensor_qos,
            messages,
        ) = _ros_dependencies()
        imu_type, imu_packet_type, sensor_timing_type, statistics_type = messages
        self._node = node_type("ad_morai_imu_timing_eval")
        self._collector = collector
        try:
            reliable_stats = qos_profile_type(depth=20, reliability=reliability_policy.RELIABLE)
            self._subscriptions = [
                self._node.create_subscription(imu_type, NORMALIZED_TOPIC, self._on_normalized, sensor_qos),
                self._node.create_subscription(imu_packet_type, FULL_TOPIC, self._on_full, sensor_qos),
                self._node.create_subscription(sensor_timing_type, TIMING_TOPIC, self._on_timing, sensor_qos),
                self._node.create_subscription(statistics_type, STATISTICS_TOPIC, self._on_statistics, reliable_stats),
            ]
            self._timer = self._node.create_timer(0.01, self._on_timer)
        except BaseException:
            self._node.destroy_node()
            raise

    @property
    def ros_node(self) -> object:
        return self._node

    def destroy_node(self) -> None:
        self._node.destroy_node()

    def _on_normalized(self, message: object) -> None:
        self._collector.on_normalized(message, receipt_ns=time.monotonic_ns())

    def _on_full(self, message: object) -> None:
        self._collector.on_full(message, receipt_ns=time.monotonic_ns())

    def _on_timing(self, message: object) -> None:
        self._collector.on_timing(message, receipt_ns=time.monotonic_ns())

    def _on_statistics(self, message: object) -> None:
        self._collector.on_statistics(message, receipt_ns=time.monotonic_ns())

    def _on_timer(self) -> None:
        self._collector.tick(time.monotonic_ns())


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-rate-hz", type=int, choices=(20, 30, 50), required=True)
    parser.add_argument("--duration-sec", type=float, required=True)
    parser.add_argument("--warmup-sec", type=float, required=True)
    parser.add_argument("--startup-timeout-sec", type=float, required=True)
    parser.add_argument("--drain-sec", type=float, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sensor-profile", type=Path)
    arguments = parser.parse_args(argv)
    try:
        for name in ("duration_sec", "warmup_sec", "startup_timeout_sec", "drain_sec"):
            _positive_finite_seconds(getattr(arguments, name), f"--{name.replace('_', '-')}")
        validate_run_id(arguments.run_id)
        if arguments.sensor_profile is not None:
            sensor_profile_provenance(arguments.sensor_profile)
    except ValueError as exc:
        parser.error(str(exc))
    return arguments


def _window_observations(window: TimingWindow) -> dict[str, object]:
    return {
        "normalized_headers": list(window.normalized_headers),
        "full_headers": list(window.full_headers),
        "timing_headers": list(window.timing_headers),
        "timing_audits": [
            {
                "header_stamp_ns": audit.header_stamp_ns,
                "stream": audit.stream,
                "source_valid": audit.source_valid,
                "source_selected": audit.source_selected,
                "source_rejected": audit.source_rejected,
                "duplicate": audit.duplicate,
                "stamp_regression": audit.stamp_regression,
                "normalized_published": audit.normalized_published,
            }
            for audit in window.audits
        ],
        "start_counters": window.start_counters.as_dict(),
        "end_counters": window.end_counters.as_dict(),
    }


def main(args: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(args)
    artifact = create_report_artifact(arguments.run_id)
    profile = sensor_profile_provenance(arguments.sensor_profile) if arguments.sensor_profile else None
    collector = TimingCollector(
        warmup_ns=_seconds_to_ns(arguments.warmup_sec, "--warmup-sec"),
        duration_ns=_seconds_to_ns(arguments.duration_sec, "--duration-sec"),
        drain_ns=_seconds_to_ns(arguments.drain_sec, "--drain-sec"),
    )
    rclpy: object | None = None
    node: ImuTimingEvaluationNode | None = None
    try:
        rclpy, _node_type, _qos_type, _reliability, _sensor_qos, _messages = _ros_dependencies()
        rclpy.init(args=None)
        node = ImuTimingEvaluationNode(collector)
        started_ns = time.monotonic_ns()
        while collector.phase != "frozen":
            rclpy.spin_once(node.ros_node, timeout_sec=0.05)
            now_ns = time.monotonic_ns()
            if collector.phase == "waiting" and now_ns - started_ns >= _seconds_to_ns(arguments.startup_timeout_sec, "--startup-timeout-sec"):
                collector.force_startup_timeout(now_ns)
        window = collector.frozen_window()
    finally:
        try:
            if node is not None:
                node.destroy_node()
        finally:
            if rclpy is not None:
                try_shutdown = getattr(rclpy, "try_shutdown", None)
                if callable(try_shutdown):
                    try_shutdown()
                elif getattr(rclpy, "ok", lambda: False)():
                    rclpy.shutdown()
    evaluated, exit_code = evaluate_frozen_window(window, expected_rate_hz=arguments.expected_rate_hz)
    report = {
        "schema_version": 1,
        "cli": {
            "expected_rate_hz": arguments.expected_rate_hz,
            "duration_sec": arguments.duration_sec,
            "warmup_sec": arguments.warmup_sec,
            "startup_timeout_sec": arguments.startup_timeout_sec,
            "drain_sec": arguments.drain_sec,
            "run_id": arguments.run_id,
        },
        "sensor_profile_provenance": profile,
        "lifecycle": collector.lifecycle(),
        "observations": _window_observations(window),
        "summary": evaluated["summary"],
        "reasons": evaluated["reasons"],
        "advisories": evaluated["advisories"],
        "transport_exit_code": exit_code,
    }
    write_report(artifact.report, report)
    print(json.dumps({"report": str(artifact.report), "transport_exit_code": exit_code}, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
