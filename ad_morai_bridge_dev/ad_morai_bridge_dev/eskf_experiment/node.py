"""Safety state machine and ROS entry point for parallel ESKF experiments.

The command executor is intentionally ROS-free so every fail-closed path can be
unit tested without a simulator. The ROS orchestration is defined below it;
only the reviewed closed-loop profile can authorize live vehicle control.
"""

from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from typing import Callable, Sequence

from ad_morai_bridge_dev.eskf_experiment.types import (
    ActorIdentity,
    CommandPhase,
    SafetyLimits,
    TruthSample,
    VehicleCommand,
    truth_freshness_error,
)

from ad_morai_bridge_dev.eskf_experiment.artifacts import (
    JsonlRecorder,
    create_run_artifacts,
    file_manifest,
    write_json,
)
from ad_morai_bridge_dev.eskf_experiment.closed_loop import (
    ClosedLoopPulseExecutor,
    load_closed_loop_pulse,
)
from ad_morai_bridge_dev.eskf_experiment.metrics import (
    align_parallel_candidate_samples,
    apply_fixed_offsets,
    estimate_settle_offsets,
    summarize_candidate,
)
from ad_morai_bridge_dev.eskf_experiment.types import CandidateSample


class ExperimentAbort(RuntimeError):
    """A safety invariant stopped an experiment."""


class ExperimentCleanupError(RuntimeError):
    """Preserve both the primary failure and an unverified-stop failure."""

    def __init__(self, primary_error: BaseException, cleanup_error: BaseException):
        super().__init__(
            f"experiment failed with {type(primary_error).__name__}: {primary_error}; "
            f"cleanup also failed with {type(cleanup_error).__name__}: {cleanup_error}"
        )
        self.primary_error = primary_error
        self.cleanup_error = cleanup_error


_TINY_PROFILE = (
    CommandPhase("settle", 0.25, 0.0, 1.0),
    CommandPhase("coast", 2.0, 0.0, 0.0),
    CommandPhase("accelerate", 0.25, 0.03, 0.0),
    CommandPhase("coast", 0.50, 0.0, 0.0),
    CommandPhase("brake", 2.0, 0.0, 1.0),
    CommandPhase("stopped", 0.25, 0.0, 1.0),
)

_PULSE10_PROFILE = (
    CommandPhase("settle", 0.25, 0.0, 1.0),
    CommandPhase("coast", 2.0, 0.0, 0.0),
    CommandPhase("accelerate", 0.50, 0.10, 0.0),
    CommandPhase("coast", 0.25, 0.0, 0.0),
    CommandPhase("brake", 2.0, 0.0, 1.0),
    CommandPhase("stopped", 0.50, 0.0, 1.0),
)

_PULSE05_PROFILE = (
    CommandPhase("settle", 0.25, 0.0, 1.0),
    CommandPhase("coast", 2.0, 0.0, 0.0),
    CommandPhase("accelerate", 0.50, 0.05, 0.0),
    CommandPhase("coast", 0.25, 0.0, 0.0),
    CommandPhase("brake", 2.0, 0.0, 1.0),
    CommandPhase("stopped", 0.50, 0.0, 1.0),
)

_MILD_PROFILE = (
    CommandPhase("settle", 2.0, 0.0, 1.0),
    CommandPhase("coast", 2.0, 0.0, 0.0),
    CommandPhase("accelerate", 2.0, 0.10, 0.0),
    CommandPhase("coast", 1.0, 0.0, 0.0),
    CommandPhase("brake", 2.0, 0.0, 0.20),
    CommandPhase("stopped", 3.0, 0.0, 1.0),
)


def fixed_command_profile(name: str) -> tuple[CommandPhase, ...]:
    """Return a named, reviewed command waveform; never parse free-form input."""
    profiles = {
        "stationary": (),
        "brake_check": (CommandPhase("stopped", 1.0, 0.0, 1.0),),
        "tiny": _TINY_PROFILE,
        "pulse05": _PULSE05_PROFILE,
        "pulse10": _PULSE10_PROFILE,
        "mild": _MILD_PROFILE,
        "extended": _MILD_PROFILE
        + (
            CommandPhase("coast", 2.0, 0.0, 0.0),
            CommandPhase("accelerate", 2.0, 0.15, 0.0),
            CommandPhase("coast", 1.0, 0.0, 0.0),
            CommandPhase("brake", 2.0, 0.0, 0.35),
            CommandPhase("stopped", 3.0, 0.0, 1.0),
        ),
    }
    try:
        return profiles[name]
    except KeyError as exc:
        raise ValueError(f"unknown fixed command profile: {name!r}") from exc


def drive_is_authorized(
    config_enabled: bool, launch_enabled: bool, profile: str
) -> bool:
    return bool(
        config_enabled
        and launch_enabled
        and profile == "closed_loop_pulse"
    )


def nonself_collision_ids(
    truth: TruthSample, actor: ActorIdentity
) -> tuple[str, ...]:
    return tuple(
        object_id
        for object_id in truth.collision_object_ids
        if object_id and object_id != actor.id_value
    )


def _speed_mps(truth: TruthSample) -> float:
    return math.sqrt(sum(value * value for value in truth.world_velocity_xyz))


def _distance_m(lhs: TruthSample, rhs: TruthSample) -> float:
    return math.sqrt(
        sum(
            (left - right) * (left - right)
            for left, right in zip(lhs.position_xyz, rhs.position_xyz)
        )
    )


def _toward(current: float, target: float, maximum_step: float) -> float:
    if target > current:
        return min(target, current + maximum_step)
    return max(target, current - maximum_step)


def _slew_command(
    current: VehicleCommand,
    target: VehicleCommand,
    maximum_step: float,
) -> VehicleCommand:
    throttle_target = target.throttle
    brake_target = target.brake
    if current.brake > 0.0 and throttle_target > 0.0:
        throttle_target = 0.0
    if current.throttle > 0.0 and brake_target > 0.0:
        brake_target = 0.0
    throttle = _toward(current.throttle, throttle_target, maximum_step)
    brake = _toward(current.brake, brake_target, maximum_step)
    if throttle > 0.0 and brake > 0.0:
        # Finish releasing the active pedal before engaging the other one.
        if current.brake > 0.0:
            throttle = 0.0
        else:
            brake = 0.0
    return VehicleCommand(throttle=throttle, brake=brake, steer=0.0)


class BoundedProfileExecutor:
    """Execute fixed MORAI commands with unconditional verified braking."""

    def __init__(
        self,
        client,
        actor: ActorIdentity,
        limits: SafetyLimits,
        *,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], None] = time.sleep,
        on_truth: Callable[[str, TruthSample], None] | None = None,
        on_command: Callable[[str, VehicleCommand, int], None] | None = None,
        health_check: Callable[[int], None] | None = None,
        abort_requested: Callable[[], bool] | None = None,
    ) -> None:
        self._client = client
        self._actor = actor
        self._limits = limits
        self._monotonic_ns = monotonic_ns
        self._sleep = sleep
        self._on_truth = on_truth or (lambda _phase, _sample: None)
        self._on_command = on_command or (
            lambda _phase, _command, _receipt_ns: None
        )
        self._health_check = health_check or (lambda _now_ns: None)
        self._abort_requested = abort_requested or (lambda: False)

    def run(self, phases: Sequence[CommandPhase]) -> tuple[TruthSample, ...]:
        self._validate_limits()
        phases = tuple(phases)
        reviewed = {
            fixed_command_profile(name)
            for name in (
                "brake_check",
                "tiny",
                "pulse05",
                "pulse10",
                "mild",
                "extended",
            )
        }
        if phases not in reviewed:
            raise ValueError("executor accepts only a reviewed fixed profile")
        samples: list[TruthSample] = []
        period_sec = 1.0 / self._limits.command_rate_hz
        maximum_step = (
            self._limits.maximum_command_delta_per_sec * period_sec
        )
        current = VehicleCommand(0.0, 1.0, 0.0)
        primary_error: BaseException | None = None
        try:
            self._on_command(
                "preflight", VehicleCommand(0.0, 1.0, 0.0), self._monotonic_ns()
            )
            self._client.full_brake()
            start = self._sample_and_validate(None, start=None)
            if _speed_mps(start) > self._limits.maximum_start_speed_mps:
                raise ExperimentAbort(
                    "Ego start speed exceeds maximum_start_speed_mps"
                )
            previous_truth = start
            cumulative_travel_m = 0.0
            for phase in phases:
                deadline_ns = self._monotonic_ns() + round(
                    phase.duration_sec * 1.0e9
                )
                target = VehicleCommand(phase.throttle, phase.brake, 0.0)
                while self._monotonic_ns() < deadline_ns:
                    if self._abort_requested():
                        raise ExperimentAbort("experiment abort requested")
                    truth = self._sample_and_validate(phase.name, start=start)
                    if self._monotonic_ns() >= deadline_ns:
                        break
                    cumulative_travel_m += _distance_m(truth, previous_truth)
                    previous_truth = truth
                    if cumulative_travel_m > self._limits.maximum_travel_m:
                        raise ExperimentAbort(
                            "Ego exceeded maximum cumulative travel"
                        )
                    samples.append(truth)
                    current = _slew_command(current, target, maximum_step)
                    self._on_command(phase.name, current, self._monotonic_ns())
                    self._client.send_command(current)
                    remaining_sec = (
                        deadline_ns - self._monotonic_ns()
                    ) * 1.0e-9
                    self._sleep(min(period_sec, max(0.0, remaining_sec)))
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                self._cleanup_full_brake()
            except BaseException as cleanup_error:
                if primary_error is not None:
                    raise ExperimentCleanupError(
                        primary_error, cleanup_error
                    ) from cleanup_error
                raise
        return tuple(samples)

    def _sample_and_validate(
        self, phase: str | None, *, start: TruthSample | None
    ) -> TruthSample:
        truth = self._client.get_truth()
        values = (
            *truth.position_xyz,
            *truth.orientation_xyzw,
            *truth.world_velocity_xyz,
            *truth.world_acceleration_xyz,
        )
        if not all(math.isfinite(value) for value in values):
            raise ExperimentAbort("MORAI truth contains a non-finite value")
        # Preserve every finite safety-triggering observation in raw evidence.
        # Non-finite values are intentionally not serialized because strict
        # JSON artifacts reject NaN/Inf.
        self._on_truth(phase or "preflight", truth)
        now_ns = self._monotonic_ns()
        freshness_error = truth_freshness_error(
            truth, now_ns, self._limits.truth_stale_timeout_sec
        )
        if freshness_error is not None:
            raise ExperimentAbort(freshness_error)
        if truth.gear_mode != "GEAR_MODE_D":
            raise ExperimentAbort(f"Ego is not in drive gear: {truth.gear_mode}")
        if nonself_collision_ids(truth, self._actor):
            raise ExperimentAbort("MORAI reports a non-self collision")
        speed = _speed_mps(truth)
        if speed > self._limits.maximum_speed_mps:
            raise ExperimentAbort("Ego exceeded maximum_speed_mps")
        if start is not None and _distance_m(truth, start) > self._limits.maximum_travel_m:
            raise ExperimentAbort("Ego exceeded maximum_travel_m")
        self._health_check(now_ns)
        del phase
        return truth

    def _cleanup_full_brake(self) -> None:
        period_sec = 1.0 / self._limits.command_rate_hz
        for _ in range(5):
            self._on_command(
                "cleanup", VehicleCommand(0.0, 1.0, 0.0), self._monotonic_ns()
            )
            self._client.full_brake()
            self._sleep(period_sec)
        stable_since_ns: int | None = None
        deadline_ns = self._monotonic_ns() + int(
            max(2.0, self._limits.stopped_stable_duration_sec * 4.0) * 1.0e9
        )
        while self._monotonic_ns() <= deadline_ns:
            truth = self._client.get_truth()
            if _speed_mps(truth) <= self._limits.stopped_speed_mps:
                stable_since_ns = stable_since_ns or self._monotonic_ns()
                if self._monotonic_ns() - stable_since_ns >= int(
                    self._limits.stopped_stable_duration_sec * 1.0e9
                ):
                    return
            else:
                stable_since_ns = None
            self._on_command(
                "cleanup", VehicleCommand(0.0, 1.0, 0.0), self._monotonic_ns()
            )
            self._client.full_brake()
            self._sleep(period_sec)
        raise ExperimentAbort("full-brake cleanup could not verify stable stop")

    def _validate_limits(self) -> None:
        values = asdict(self._limits)
        if not all(
            math.isfinite(float(value)) and float(value) > 0.0
            for value in values.values()
        ):
            raise ValueError("all safety limits must be positive and finite")


def _message_stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _vector3(message) -> tuple[float, float, float]:
    return (float(message.x), float(message.y), float(message.z))


def _quaternion(message) -> tuple[float, float, float, float]:
    return (
        float(message.x),
        float(message.y),
        float(message.z),
        float(message.w),
    )


def _evaluation_errors(errors):
    """Exclude pre-initialization transients while preserving them in raw data."""
    return tuple(error for error in errors if error.phase != "initialization")


def _stopped_evaluation_phase(profile: str) -> str:
    if profile == "closed_loop_pulse":
        return "closed_loop_stop"
    return "stopped"


def _git_snapshot(repository_root: Path) -> dict[str, object]:
    def run(*arguments: str) -> bytes:
        completed = subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return completed.stdout

    head = run("rev-parse", "HEAD").decode("utf-8").strip()
    status = run("status", "--porcelain=v1", "-z")
    diff = run("diff", "--binary", "--no-ext-diff")
    untracked = tuple(
        entry
        for entry in run("ls-files", "--others", "--exclude-standard", "-z").split(
            b"\0"
        )
        if entry
    )
    hasher = hashlib.sha256(status + b"\0" + diff)
    for relative_bytes in sorted(untracked):
        relative = Path(os.fsdecode(relative_bytes))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("git returned an unsafe untracked path")
        path = repository_root / relative
        hasher.update(b"\0untracked-path\0" + relative_bytes)
        if path.is_symlink():
            hasher.update(b"\0symlink\0" + os.fsencode(os.readlink(path)))
        elif path.is_file():
            hasher.update(b"\0regular\0" + path.read_bytes())
        else:
            # Never open special files while producing evidence. Their path and
            # lstat metadata still make this state explicit and deterministic.
            metadata = path.lstat()
            hasher.update(
                f"\0special\0{metadata.st_mode:o}\0{metadata.st_size}".encode(
                    "ascii"
                )
            )
    digest = hasher.hexdigest()
    return {
        "head": head,
        "dirty": bool(status),
        "worktree_sha256": digest,
    }


def _morai_build_files(active_sensor_file: Path) -> dict[str, Path]:
    """Find immutable MORAI build evidence from the active sensor path."""
    sensor_path = Path(active_sensor_file)
    data_root = next(
        (
            parent
            for parent in sensor_path.parents
            if parent.name.endswith("_Data")
        ),
        None,
    )
    if data_root is None:
        return {}
    launcher_stem = data_root.name[: -len("_Data")]
    candidates = {
        "simulator:app_info": data_root / "app.info",
        "simulator:assembly_csharp": (
            data_root / "Managed" / "Assembly-CSharp.dll"
        ),
        "simulator:launcher": data_root.parent / f"{launcher_stem}.x86_64",
    }
    return {name: path for name, path in candidates.items() if path.is_file()}


def _connect_safe_client(
    target: str,
    timeout_sec: float,
    limits: SafetyLimits,
    on_truth: Callable[[str, TruthSample], None] | None = None,
):
    from ad_morai_bridge_dev.eskf_experiment.grpc import SafeMoraiExperimentClient
    from ad_morai_bridge_dev.simulator_grpc.client import MoraiGrpcClient
    from ad_morai_bridge_dev.simulator_grpc.descriptors import MoraiApi

    raw_client = MoraiGrpcClient.connect(
        MoraiApi.load(), target, default_timeout=timeout_sec
    )
    return SafeMoraiExperimentClient(
        raw_client,
        timeout_sec=timeout_sec,
        truth_timeout_sec=min(timeout_sec, limits.truth_stale_timeout_sec),
        stopped_speed_mps=limits.stopped_speed_mps,
        stopped_stable_duration_sec=limits.stopped_stable_duration_sec,
        command_entry_stable_duration_sec=limits.stopped_stable_duration_sec,
        cleanup_poll_interval_sec=1.0 / limits.command_rate_hz,
        brake_attempts=max(
            5,
            math.ceil(
                limits.stopped_stable_duration_sec * limits.command_rate_hz
            )
            + 2,
        ),
        on_truth=on_truth,
    )


def _evidence_capabilities() -> dict[str, bool]:
    """Disclose which reviewed diagnostics are and are not in artifacts."""
    return {
        "bias_covariance_persisted": False,
        "estimator_diagnostic_counters_persisted": False,
        "cleanup_truth_callback_enabled": True,
    }


def _truth_frame_contract() -> dict[str, str]:
    """Make the one inferred MORAI frame convention explicit in artifacts."""
    return {
        "world_velocity_source": "ActorState.global_velocity",
        "world_acceleration_source": "ActorState.acceleration",
        "actor_acceleration_input_frame": "body_inferred",
        "world_acceleration_transform": "actor_rotation",
    }


def _load_runtime_document(path: Path) -> dict[str, object]:
    import yaml

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("invalid ESKF parallel experiment config")
    return document


def _runtime_boolean(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be a boolean")
    return value


def _enabled_candidates(document: dict[str, object]) -> tuple[str, ...]:
    raw = document.get("candidates")
    if not isinstance(raw, dict):
        raise ValueError("experiment candidates must be a mapping")
    result = tuple(
        str(name)
        for name, config in raw.items()
        if isinstance(config, dict)
        and _runtime_boolean(
            config.get("enabled", False), f"candidate {name} enabled"
        )
    )
    if not result:
        raise ValueError("experiment requires at least one enabled candidate")
    return result


def _safety_limits(document: dict[str, object]) -> SafetyLimits:
    raw = document.get("safety")
    if not isinstance(raw, dict):
        raise ValueError("experiment safety must be a mapping")
    return SafetyLimits(
        maximum_start_speed_mps=float(raw["maximum_start_speed_mps"]),
        maximum_speed_mps=float(raw["maximum_speed_mps"]),
        maximum_travel_m=float(raw["maximum_travel_m"]),
        truth_stale_timeout_sec=float(raw["truth_stale_timeout_sec"]),
        estimator_stale_timeout_sec=float(raw["estimator_stale_timeout_sec"]),
        command_rate_hz=float(raw["command_rate_hz"]),
        maximum_command_delta_per_sec=float(raw["maximum_command_delta_per_sec"]),
        stopped_speed_mps=float(raw["stopped_speed_mps"]),
        stopped_stable_duration_sec=float(raw["stopped_stable_duration_sec"]),
    )


class EskfExperimentNode:
    """Thin facade whose implementation subclasses rclpy.Node at runtime."""

    def __new__(cls, *args, **kwargs):
        del cls
        return _RosEskfExperimentNode(*args, **kwargs)


try:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data

    from ad_morai_interfaces.msg import CollisionArray
    from geometry_msgs.msg import PoseStamped, Vector3Stamped
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Float64MultiArray
except ImportError:  # pragma: no cover - pure unit tests can import without ROS.
    rclpy = None
    MultiThreadedExecutor = None
    Node = object


class _RosEskfExperimentNode(Node):
    def __init__(
        self,
        *,
        client_factory: Callable[
            [
                str,
                float,
                SafetyLimits,
                Callable[[str, TruthSample], None] | None,
            ],
            object,
        ] = _connect_safe_client,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if rclpy is None:
            raise RuntimeError("ROS 2 Python libraries are unavailable")
        super().__init__("eskf_ab_harness")
        self.declare_parameter("run_id", "stationary_ab")
        self.declare_parameter("experiment_config", "")
        self.declare_parameter("eskf_config", "")
        self.declare_parameter("grpc_target", "127.0.0.1:7789")
        self.declare_parameter("grpc_timeout_sec", 2.0)
        self.declare_parameter("drive_enabled", False)
        self.declare_parameter("profile", "")
        self.declare_parameter("stationary_duration_sec", 15.0)
        self.declare_parameter("initialization_timeout_sec", 15.0)
        self.declare_parameter("maximum_pair_age_sec", 0.10)
        self.declare_parameter("data_root", "")
        self.declare_parameter("repository_root", "")
        self.declare_parameter("active_sensor_file", "")

        self._run_id = self.get_parameter("run_id").value
        self._experiment_path = Path(
            self.get_parameter("experiment_config").value
        )
        self._eskf_path = Path(self.get_parameter("eskf_config").value)
        self._grpc_target = self.get_parameter("grpc_target").value
        self._grpc_timeout_sec = float(
            self.get_parameter("grpc_timeout_sec").value
        )
        self._launch_drive_enabled = bool(
            self.get_parameter("drive_enabled").value
        )
        self._stationary_duration_sec = float(
            self.get_parameter("stationary_duration_sec").value
        )
        self._initialization_timeout_sec = float(
            self.get_parameter("initialization_timeout_sec").value
        )
        self._maximum_pair_age_ns = int(
            float(self.get_parameter("maximum_pair_age_sec").value) * 1.0e9
        )
        data_root = str(self.get_parameter("data_root").value).strip()
        if data_root:
            if not Path(data_root).is_absolute():
                raise ValueError("data_root must be absolute")
            os.environ["AD_DATA_DIR"] = data_root
        self._repository_root = Path(
            str(self.get_parameter("repository_root").value)
        )
        self._active_sensor_file = Path(
            str(self.get_parameter("active_sensor_file").value)
        )
        if not self._experiment_path.is_file() or not self._eskf_path.is_file():
            raise ValueError("experiment_config and eskf_config must be files")
        if not self._active_sensor_file.is_file():
            raise ValueError("active_sensor_file must identify the live sensor profile")

        self._document = _load_runtime_document(self._experiment_path)
        self._candidates = _enabled_candidates(self._document)
        self._limits = _safety_limits(self._document)
        configured_profile = str(self._document.get("profile", "stationary"))
        requested_profile = str(self.get_parameter("profile").value).strip()
        self._profile = requested_profile or configured_profile
        self._config_drive_enabled = _runtime_boolean(
            self._document.get("drive_enabled", False), "drive_enabled"
        )
        # Validate the requested name even when driving remains disabled.
        self._closed_loop_config = None
        if self._profile == "closed_loop_pulse":
            self._closed_loop_config = load_closed_loop_pulse(
                self._document, self._limits
            )
        else:
            fixed_command_profile(self._profile)

        self._client_factory = client_factory
        self._monotonic_ns = monotonic_ns
        self._abort = threading.Event()
        self._lock = threading.Lock()
        self._phase = "initialization"
        self._canonical_odometry = None
        self._last_collision_ns = 0
        self._collision_ids: tuple[int, ...] = ()
        self._candidate_samples: dict[str, list[CandidateSample]] = {
            name: [] for name in self._candidates
        }
        self._truth_samples: list[TruthSample] = []
        self._latest_candidate_ns = {name: 0 for name in self._candidates}
        self._candidate_initialized = {name: False for name in self._candidates}
        self._accel_bias = {name: (0.0, 0.0, 0.0) for name in self._candidates}
        self._gyro_bias = {name: (0.0, 0.0, 0.0) for name in self._candidates}
        self._recorder: JsonlRecorder | None = None

        reliable = QoSProfile(depth=20)
        reliable.reliability = ReliabilityPolicy.RELIABLE
        namespace = f"/ad/experiment/eskf/{self._run_id}"
        self._initial_publishers = {}
        # Do not shadow rclpy.Node._subscriptions.  rclpy owns that registry;
        # duplicating entries there makes destroy_node() destroy an entity twice.
        self._experiment_subscriptions = []
        for name in self._candidates:
            prefix = f"{namespace}/{name}"
            node_prefix = f"{namespace}/eskf_{name}"
            self._initial_publishers[name] = self.create_publisher(
                PoseStamped, f"{prefix}/initial_pose", reliable
            )
            self._experiment_subscriptions.append(
                self.create_subscription(
                    Odometry,
                    f"{prefix}/odometry",
                    lambda message, candidate=name: self._on_candidate_odometry(
                        candidate, message
                    ),
                    reliable,
                )
            )
            self._experiment_subscriptions.append(
                self.create_subscription(
                    Float64MultiArray,
                    f"{node_prefix}/debug/initialization",
                    lambda message, candidate=name: self._on_initialization(
                        candidate, message
                    ),
                    reliable,
                )
            )
            self._experiment_subscriptions.append(
                self.create_subscription(
                    Vector3Stamped,
                    f"{node_prefix}/current_accel_bias",
                    lambda message, candidate=name: self._on_accel_bias(
                        candidate, message
                    ),
                    reliable,
                )
            )
            self._experiment_subscriptions.append(
                self.create_subscription(
                    Vector3Stamped,
                    f"{node_prefix}/current_gyro_bias",
                    lambda message, candidate=name: self._on_gyro_bias(
                        candidate, message
                    ),
                    reliable,
                )
            )
        self._experiment_subscriptions.extend(
            [
                self.create_subscription(
                    Odometry,
                    "/ad/localization/odometry",
                    self._on_canonical_odometry,
                    reliable,
                ),
                self.create_subscription(
                    CollisionArray,
                    "/ad/safety/collisions",
                    self._on_collisions,
                    qos_profile_sensor_data,
                ),
            ]
        )

    def request_abort(self, reason: str) -> None:
        if not self._abort.is_set():
            self.get_logger().error(f"ESKF experiment abort requested: {reason}")
            self._abort.set()

    def _on_canonical_odometry(self, message) -> None:
        with self._lock:
            self._canonical_odometry = message

    def _on_collisions(self, message) -> None:
        with self._lock:
            self._last_collision_ns = self._monotonic_ns()
            self._collision_ids = tuple(
                int(collision.object_id) for collision in message.collisions
            )

    def _on_accel_bias(self, candidate: str, message) -> None:
        with self._lock:
            self._accel_bias[candidate] = _vector3(message.vector)

    def _on_gyro_bias(self, candidate: str, message) -> None:
        with self._lock:
            self._gyro_bias[candidate] = _vector3(message.vector)

    def _on_initialization(self, candidate: str, message) -> None:
        if message.data and int(message.data[0]) == 1:
            with self._lock:
                self._candidate_initialized[candidate] = True

    def _on_candidate_odometry(self, candidate: str, message) -> None:
        receipt_ns = self._monotonic_ns()
        with self._lock:
            sample = CandidateSample(
                candidate=candidate,
                phase=self._phase,
                receipt_monotonic_ns=receipt_ns,
                header_stamp_ns=_message_stamp_ns(message.header.stamp),
                position_xyz=_vector3(message.pose.pose.position),
                orientation_xyzw=_quaternion(message.pose.pose.orientation),
                body_velocity_xyz=_vector3(message.twist.twist.linear),
                accel_bias_xyz=self._accel_bias[candidate],
                gyro_bias_xyz=self._gyro_bias[candidate],
                # The current upstream diagnostic topic exposes nominal bias but
                # not its covariance.  -1 is a finite explicit unavailable sentinel.
                accel_bias_covariance_xyz=(-1.0, -1.0, -1.0),
                gyro_bias_covariance_xyz=(-1.0, -1.0, -1.0),
            )
            self._candidate_samples[candidate].append(sample)
            self._latest_candidate_ns[candidate] = receipt_ns
            recorder = self._recorder
        if recorder is not None:
            recorder.write("candidate", asdict(sample))

    def _wait_until(self, predicate, timeout_sec: float, description: str) -> None:
        deadline = self._monotonic_ns() + int(timeout_sec * 1.0e9)
        while self._monotonic_ns() < deadline:
            if self._abort.is_set():
                raise ExperimentAbort("abort requested while waiting")
            if predicate():
                return
            time.sleep(0.05)
        raise ExperimentAbort(f"timeout waiting for {description}")

    def _inputs_ready(self) -> bool:
        with self._lock:
            canonical_ready = self._canonical_odometry is not None
            collision_ready = self._last_collision_ns > 0
        subscribers_ready = all(
            publisher.get_subscription_count() >= 1
            for publisher in self._initial_publishers.values()
        )
        return canonical_ready and collision_ready and subscribers_ready

    def _publish_common_initial_pose(self) -> None:
        with self._lock:
            odometry = self._canonical_odometry
        if odometry is None:
            raise ExperimentAbort("canonical odometry is unavailable")
        pose = PoseStamped()
        pose.header = odometry.header
        pose.pose = odometry.pose.pose
        values = (
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        )
        if not all(math.isfinite(value) for value in values):
            raise ExperimentAbort("canonical initial pose is non-finite")
        for _ in range(5):
            for publisher in self._initial_publishers.values():
                publisher.publish(pose)
            time.sleep(0.10)

    def _candidates_ready(self) -> bool:
        with self._lock:
            return all(
                self._latest_candidate_ns[name] > 0
                and self._candidate_initialized[name]
                for name in self._candidates
            )

    def _health_check(self, now_ns: int) -> None:
        with self._lock:
            collision_age = now_ns - self._last_collision_ns
            collision_ids = self._collision_ids
            candidate_ages = {
                name: now_ns - value
                for name, value in self._latest_candidate_ns.items()
            }
        if collision_age > int(self._limits.truth_stale_timeout_sec * 1.0e9):
            raise ExperimentAbort("ROS collision stream is stale")
        if collision_ids:
            raise ExperimentAbort(f"ROS collision detected: {collision_ids}")
        maximum_age = int(self._limits.estimator_stale_timeout_sec * 1.0e9)
        stale = [name for name, age in candidate_ages.items() if age > maximum_age]
        if stale:
            raise ExperimentAbort(f"candidate estimator output is stale: {stale}")

    def _record_truth(self, phase: str, truth: TruthSample) -> None:
        with self._lock:
            self._phase = phase
            self._truth_samples.append(truth)
            recorder = self._recorder
        if recorder is not None:
            recorder.write("truth", {"phase": phase, **asdict(truth)})

    def _record_command(
        self, phase: str, command: VehicleCommand, receipt_monotonic_ns: int
    ) -> None:
        with self._lock:
            recorder = self._recorder
        if recorder is not None:
            recorder.write(
                "command",
                {
                    "receipt_monotonic_ns": receipt_monotonic_ns,
                    "phase": phase,
                    **asdict(command),
                },
            )

    def _collect_stationary(self, client, duration_sec: float) -> None:
        period_sec = 1.0 / self._limits.command_rate_hz
        start_ns = self._monotonic_ns()
        settle_end_ns = start_ns + int(min(5.0, duration_sec / 3.0) * 1.0e9)
        end_ns = start_ns + int(duration_sec * 1.0e9)
        first_truth = None
        previous_truth = None
        cumulative_travel_m = 0.0
        while self._monotonic_ns() < end_ns:
            if self._abort.is_set():
                raise ExperimentAbort("abort requested")
            phase = "settle" if self._monotonic_ns() < settle_end_ns else "stopped"
            truth = client.get_truth()
            values = (
                *truth.position_xyz,
                *truth.orientation_xyzw,
                *truth.world_velocity_xyz,
                *truth.world_acceleration_xyz,
            )
            if not all(math.isfinite(value) for value in values):
                raise ExperimentAbort("MORAI truth contains a non-finite value")
            self._record_truth(phase, truth)
            now_ns = self._monotonic_ns()
            freshness_error = truth_freshness_error(
                truth, now_ns, self._limits.truth_stale_timeout_sec
            )
            if freshness_error is not None:
                raise ExperimentAbort(freshness_error)
            if truth.gear_mode != "GEAR_MODE_D":
                raise ExperimentAbort(
                    f"Ego is not in drive gear: {truth.gear_mode}"
                )
            if nonself_collision_ids(truth, client.actor_identity):
                raise ExperimentAbort("MORAI reports a non-self collision")
            if _speed_mps(truth) > self._limits.maximum_start_speed_mps:
                raise ExperimentAbort("Ego moved during stationary collection")
            first_truth = first_truth or truth
            if previous_truth is not None:
                cumulative_travel_m += _distance_m(truth, previous_truth)
            previous_truth = truth
            if _distance_m(truth, first_truth) > self._limits.maximum_travel_m:
                raise ExperimentAbort("Ego left the stationary travel bound")
            if cumulative_travel_m > self._limits.maximum_travel_m:
                raise ExperimentAbort("Ego exceeded cumulative stationary travel")
            self._health_check(now_ns)
            time.sleep(period_sec)

    def _write_aligned_and_summary(self, artifacts) -> dict[str, object]:
        rows = []
        summaries = {}
        with self._lock:
            truth = tuple(self._truth_samples)
            candidate_samples = {
                name: tuple(values)
                for name, values in self._candidate_samples.items()
            }
        aligned_by_candidate = align_parallel_candidate_samples(
            candidate_samples,
            truth,
            maximum_pair_age_ns=self._maximum_pair_age_ns,
        )
        stopped_phase = _stopped_evaluation_phase(self._profile)
        for name, samples in candidate_samples.items():
            aligned = aligned_by_candidate[name]
            offsets = estimate_settle_offsets(aligned)
            errors = [apply_fixed_offsets(sample, offsets) for sample in aligned]
            evaluation_errors = _evaluation_errors(errors)
            summaries[name] = {
                "frame_offsets": asdict(offsets),
                "metrics": summarize_candidate(
                    evaluation_errors,
                    stopped_phase=stopped_phase,
                    stopped_velocity_threshold_mps=self._limits.stopped_speed_mps,
                ),
                "bias_covariance_available": False,
                "metrics_excluded_phases": ["initialization"],
                "stopped_phase_used": stopped_phase,
            }
            rows.extend(asdict(error) for error in errors)
        fieldnames = tuple(rows[0]) if rows else ()
        with artifacts.aligned.open("w", encoding="utf-8", newline="") as stream:
            if fieldnames:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
        summary = {
            "schema_version": 1,
            "run_id": self._run_id,
            "profile": self._profile,
            "evidence_capabilities": _evidence_capabilities(),
            "parallel_alignment": {
                "mode": "common_output_header_shared_truth",
                "received_output_sample_counts": {
                    name: len(samples)
                    for name, samples in candidate_samples.items()
                },
                "paired_output_header_count": (
                    len(next(iter(aligned_by_candidate.values())))
                    if aligned_by_candidate
                    else 0
                ),
                "same_truth_for_each_paired_header": True,
                "exact_input_payload_identity_persisted": False,
                "candidate_input_receive_counters_persisted": False,
            },
            "candidate_summaries": summaries,
        }
        return summary

    def _manifest(
        self,
        actor,
        control_mode: str | None,
        *,
        final_control_mode: str | None = None,
        control_mode_safety: dict[str, object] | None = None,
        status: str = "running",
        cleanup_status: str = "pending",
    ) -> dict[str, object]:
        sources = {
            "experiment_config": self._experiment_path,
            "eskf_config": self._eskf_path,
            "active_sensor_file": self._active_sensor_file,
        }
        sources.update(_morai_build_files(self._active_sensor_file))
        if self._repository_root.is_dir():
            patch_directory = (
                self._repository_root
                / "patches"
                / "kalman-filter-localization-ros2"
            )
            for patch in sorted(patch_directory.glob("*.patch")):
                sources[f"dependency_overlay:{patch.name}"] = patch
        return {
            "schema_version": 1,
            "run_id": self._run_id,
            "actor_identity": asdict(actor) if actor is not None else None,
            "initial_control_mode": control_mode,
            "final_control_mode": final_control_mode,
            "control_mode_safety": control_mode_safety
            or {
                "command_control_mode": None,
                "pre_waveform_stable_stop_status": "not_requested",
                "cleanup_stable_stop_status": "not_started",
                "restoration_status": "not_required",
                "restore_skipped_reason": None,
                "post_restore_stop_status": "not_required",
                "last_brake_rpc_status": "not_attempted",
            },
            "status": status,
            "cleanup_status": cleanup_status,
            "profile": self._profile,
            "config_drive_enabled": self._config_drive_enabled,
            "launch_drive_enabled": self._launch_drive_enabled,
            "prohibited_rpcs_used": False,
            "evidence_capabilities": _evidence_capabilities(),
            "truth_frame_contract": _truth_frame_contract(),
            "parallel_input_contract": {
                "mode": "same_topic_simultaneous",
                "shared_inputs": self._document.get("shared_inputs"),
                "exact_message_payload_identity_persisted": False,
                "candidate_input_receive_counters_persisted": False,
            },
            "rpc_audit_contract": {
                "enforcement": "closed_wrapper_allowlist",
                "runtime_rpc_call_log_persisted": False,
                "prohibited_rpcs_used_is_structural_claim": True,
            },
            "allowed_rpcs": [
                "morai_sim_api.actor.Actor/GetAllActorsState",
                "morai_sim_api.actor.Actor/GetActorState",
                "morai_sim_api.actor.Actor/GetVehicleControlMode",
                "morai_sim_api.actor.Actor/SetVehicleControlMode",
                "morai_sim_api.actor.Actor/ControlVehicle",
            ],
            "candidate_parameters": self._document["candidates"],
            "files": file_manifest(sources),
            "repository": (
                _git_snapshot(self._repository_root)
                if (self._repository_root / ".git").exists()
                else None
            ),
        }

    def run_experiment(self) -> dict[str, object]:
        artifacts = create_run_artifacts(self._run_id)
        client = None
        actor = None
        initial_control_mode = None
        final_control_mode = None
        control_mode_safety = None
        summary = None
        primary_error: BaseException | None = None
        cleanup_error: BaseException | None = None
        write_json(artifacts.manifest, self._manifest(None, None))
        with JsonlRecorder(artifacts.raw) as recorder:
            self._recorder = recorder
            try:
                self._wait_until(
                    self._inputs_ready,
                    self._initialization_timeout_sec,
                    "canonical inputs and candidate subscriptions",
                )
                client = self._client_factory(
                    self._grpc_target,
                    self._grpc_timeout_sec,
                    self._limits,
                    self._record_truth,
                )
                actor = client.discover_ego()
                initial_control_mode = client.initial_control_mode
                truth = client.get_truth()
                if nonself_collision_ids(truth, actor):
                    raise ExperimentAbort("MORAI reports a non-self collision")
                if _speed_mps(truth) > self._limits.maximum_start_speed_mps:
                    raise ExperimentAbort("Ego must be stopped before initialization")
                self._publish_common_initial_pose()
                self._wait_until(
                    self._candidates_ready,
                    self._initialization_timeout_sec,
                    "all ESKF candidate outputs",
                )
                write_json(
                    artifacts.manifest,
                    self._manifest(
                        actor,
                        initial_control_mode,
                        control_mode_safety=client.safety_status,
                    ),
                )

                if drive_is_authorized(
                    self._config_drive_enabled,
                    self._launch_drive_enabled,
                    self._profile,
                ):
                    # Always establish a multi-second fixed frame offset before motion.
                    self._collect_stationary(client, 3.0)
                    client.enter_command_control()
                    executor_arguments = {
                        "monotonic_ns": self._monotonic_ns,
                        "on_truth": self._record_truth,
                        "on_command": self._record_command,
                        "health_check": self._health_check,
                        "abort_requested": self._abort.is_set,
                    }
                    if self._profile == "closed_loop_pulse":
                        executor = ClosedLoopPulseExecutor(
                            client,
                            actor,
                            self._limits,
                            **executor_arguments,
                        )
                        executor.run(self._closed_loop_config)
                    else:
                        executor = BoundedProfileExecutor(
                            client,
                            actor,
                            self._limits,
                            **executor_arguments,
                        )
                        executor.run(fixed_command_profile(self._profile))
                else:
                    self._collect_stationary(client, self._stationary_duration_sec)
                summary = self._write_aligned_and_summary(artifacts)
                summary.setdefault(
                    "evidence_capabilities", _evidence_capabilities()
                )
                summary.setdefault("truth_frame_contract", _truth_frame_contract())
            except BaseException as exc:
                primary_error = exc
            finally:
                try:
                    if client is not None:
                        close_succeeded = False
                        try:
                            client.close()
                            close_succeeded = True
                        except BaseException as exc:
                            cleanup_error = exc
                        try:
                            control_mode_safety = client.safety_status
                        except BaseException as exc:
                            if cleanup_error is None:
                                cleanup_error = exc
                        try:
                            final_control_mode = client.final_control_mode
                        except BaseException as exc:
                            if close_succeeded and cleanup_error is None:
                                cleanup_error = exc
                        if (
                            close_succeeded
                            and cleanup_error is None
                            and initial_control_mode is not None
                            and final_control_mode != initial_control_mode
                        ):
                            cleanup_error = ExperimentAbort(
                                "MORAI vehicle control mode changed during cleanup"
                            )
                finally:
                    # SafeMoraiExperimentClient.close() emits cleanup and
                    # post-restore truth through _record_truth. Keep the raw
                    # recorder attached until every final safety sample lands.
                    self._recorder = None

        resolved_cleanup_status = (
            "not_required"
            if client is None
            else ("verified" if cleanup_error is None else "failed")
        )
        write_json(
            artifacts.manifest,
            self._manifest(
                actor,
                initial_control_mode,
                final_control_mode=final_control_mode,
                control_mode_safety=control_mode_safety,
                status=(
                    "complete"
                    if primary_error is None and cleanup_error is None
                    else "failed"
                ),
                cleanup_status=resolved_cleanup_status,
            ),
        )

        if primary_error is not None or cleanup_error is not None:
            effective_error: BaseException
            if primary_error is not None and cleanup_error is not None:
                effective_error = ExperimentCleanupError(
                    primary_error, cleanup_error
                )
            else:
                effective_error = primary_error or cleanup_error  # type: ignore[assignment]
            failure = {
                "schema_version": 1,
                "run_id": self._run_id,
                "profile": self._profile,
                "status": "failed",
                "cleanup_status": resolved_cleanup_status,
                "failure_type": type(effective_error).__name__,
                "failure": str(effective_error),
            }
            if summary is not None:
                failure["partial_metrics"] = summary
            write_json(artifacts.summary, failure)
            raise effective_error

        if summary is None:
            raise RuntimeError("experiment ended without a summary")
        summary.update(
            {
                "status": "complete",
                "cleanup_status": "verified",
                "initial_control_mode": initial_control_mode,
                "final_control_mode": final_control_mode,
            }
        )
        write_json(artifacts.summary, summary)
        return summary


def main(args=None) -> int:
    if rclpy is None:
        raise RuntimeError("ROS 2 Python libraries are unavailable")
    rclpy.init(args=args)
    node = _RosEskfExperimentNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    old_handlers = {}

    def request_abort(signum, _frame):
        node.request_abort(signal.Signals(signum).name)

    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, request_abort)
    spin_thread.start()
    result = 0
    try:
        summary = node.run_experiment()
        node.get_logger().info(
            "ESKF A/B experiment complete: "
            + json.dumps(summary, sort_keys=True, separators=(",", ":"))
        )
    except BaseException as exc:
        node.get_logger().error(f"ESKF A/B experiment failed: {exc}")
        result = 1
    finally:
        executor.shutdown()
        spin_thread.join(timeout=5.0)
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    return result
