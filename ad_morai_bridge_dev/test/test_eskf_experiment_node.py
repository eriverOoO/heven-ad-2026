from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace

import pytest

import ad_morai_bridge_dev.eskf_experiment.node as experiment_node
from ad_morai_bridge_dev.eskf_experiment.node import (
    BoundedProfileExecutor,
    ExperimentAbort,
    ExperimentCleanupError,
    _RosEskfExperimentNode,
    _evaluation_errors,
    _stopped_evaluation_phase,
    drive_is_authorized,
    fixed_command_profile,
    nonself_collision_ids,
)
from ad_morai_bridge_dev.eskf_experiment.types import (
    ActorIdentity,
    SafetyLimits,
    TruthSample,
    VehicleCommand,
)


class FakeClock:
    def __init__(self):
        self.nanoseconds = 0

    def monotonic_ns(self):
        return self.nanoseconds

    def sleep(self, duration_sec):
        self.nanoseconds += int(duration_sec * 1_000_000_000)


def _truth(
    clock,
    *,
    x=0.0,
    speed=0.0,
    collisions=(),
    gear="GEAR_MODE_D",
):
    return TruthSample(
        receipt_monotonic_ns=clock.monotonic_ns(),
        rpc_start_monotonic_ns=clock.monotonic_ns(),
        simulator_timestamp=None,
        position_xyz=(x, 0.0, 0.0),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        world_velocity_xyz=(speed, 0.0, 0.0),
        world_acceleration_xyz=(0.0, 0.0, 0.0),
        gear_mode=gear,
        throttle=0.0,
        brake=0.0,
        steer=0.0,
        collision_object_ids=tuple(collisions),
    )


class FakeSafeClient:
    def __init__(self, clock, states, *, fail_non_brake=False):
        self.clock = clock
        self.states = list(states)
        self.last_state = self.states[-1]
        self.fail_non_brake = fail_non_brake
        self.commands = []
        self.full_brakes = 0
        self.actor_identity = ActorIdentity("Ego", "OBJECT_TYPE_VEHICLE", "")

    def get_truth(self):
        if self.states:
            self.last_state = self.states.pop(0)
        self.last_state = replace(
            self.last_state,
            rpc_start_monotonic_ns=self.clock.monotonic_ns(),
            receipt_monotonic_ns=self.clock.monotonic_ns(),
        )
        return self.last_state

    def send_command(self, command):
        self.commands.append(command)
        if self.fail_non_brake and command.throttle > 0.0:
            raise RuntimeError("injected RPC failure")

    def full_brake(self):
        self.full_brakes += 1
        self.commands.append(VehicleCommand(throttle=0.0, brake=1.0, steer=0.0))


@pytest.fixture
def limits():
    return SafetyLimits(
        maximum_start_speed_mps=0.05,
        maximum_speed_mps=0.5,
        maximum_travel_m=0.25,
        truth_stale_timeout_sec=0.25,
        estimator_stale_timeout_sec=0.25,
        command_rate_hz=20.0,
        maximum_command_delta_per_sec=0.5,
        stopped_speed_mps=0.02,
        stopped_stable_duration_sec=0.10,
    )


def test_drive_requires_both_config_and_launch_opt_in_and_a_nonstationary_profile():
    assert not drive_is_authorized(False, True, "tiny")
    assert not drive_is_authorized(True, False, "tiny")
    assert not drive_is_authorized(True, True, "stationary")
    assert not drive_is_authorized(True, True, "tiny")
    assert drive_is_authorized(True, True, "closed_loop_pulse")


def test_closed_loop_profile_is_not_treated_as_an_open_loop_pedal_waveform():
    with pytest.raises(ValueError, match="profile"):
        fixed_command_profile("closed_loop_pulse")


def test_profiles_are_fixed_and_never_accept_arbitrary_commands():
    assert fixed_command_profile("stationary") == ()
    tiny = fixed_command_profile("tiny")
    assert tiny
    assert all(phase.throttle <= 0.03 for phase in tiny)
    assert all(
        phase.name in {"settle", "accelerate", "coast", "brake", "stopped"}
        for phase in tiny
    )
    with pytest.raises(ValueError, match="profile"):
        fixed_command_profile("throttle=1.0")


def test_safe_client_connection_receives_yaml_stop_limits_and_truth_callback(
    monkeypatch, limits
):
    import ad_morai_bridge_dev.eskf_experiment.grpc as grpc_wrapper
    import ad_morai_bridge_dev.simulator_grpc.client as grpc_client
    import ad_morai_bridge_dev.simulator_grpc.descriptors as grpc_descriptors

    raw_client = object()
    captured = {}

    def callback(_phase, _truth):
        pass

    monkeypatch.setattr(
        grpc_descriptors.MoraiApi, "load", staticmethod(lambda: "api")
    )
    monkeypatch.setattr(
        grpc_client.MoraiGrpcClient,
        "connect",
        staticmethod(
            lambda api, target, default_timeout: (
                captured.update(
                    api=api,
                    target=target,
                    default_timeout=default_timeout,
                )
                or raw_client
            )
        ),
    )

    class FakeSafeClient:
        def __init__(self, client, **kwargs):
            captured["client"] = client
            captured.update(kwargs)

    monkeypatch.setattr(grpc_wrapper, "SafeMoraiExperimentClient", FakeSafeClient)

    live_limits = replace(limits, stopped_stable_duration_sec=1.0)
    experiment_node._connect_safe_client(
        "fixture:7789", 2.5, live_limits, callback
    )

    assert captured == {
        "api": "api",
        "target": "fixture:7789",
        "default_timeout": 2.5,
        "client": raw_client,
        "timeout_sec": 2.5,
        "truth_timeout_sec": live_limits.truth_stale_timeout_sec,
        "stopped_speed_mps": live_limits.stopped_speed_mps,
        "stopped_stable_duration_sec": live_limits.stopped_stable_duration_sec,
        "command_entry_stable_duration_sec": (
            live_limits.stopped_stable_duration_sec
        ),
        "cleanup_poll_interval_sec": 1.0 / live_limits.command_rate_hz,
        "brake_attempts": max(
            5,
            math.ceil(
                live_limits.stopped_stable_duration_sec
                * live_limits.command_rate_hz
            )
            + 2,
        ),
        "on_truth": callback,
    }
    assert captured["brake_attempts"] == 22


def _initialize_git_repository(path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"], check=True
    )
    (path / "tracked-a").write_text("a\n", encoding="utf-8")
    (path / "tracked-b").write_text("b\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-qm", "fixture"], check=True
    )


def test_git_snapshot_hashes_untracked_file_contents_not_only_status_name(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    _initialize_git_repository(repository)
    untracked = repository / "evidence.txt"
    untracked.write_text("first\n", encoding="utf-8")
    first = experiment_node._git_snapshot(repository)

    untracked.write_text("second\n", encoding="utf-8")
    second = experiment_node._git_snapshot(repository)

    assert first["dirty"] is True
    assert first["worktree_sha256"] != second["worktree_sha256"]


def test_git_snapshot_hashes_untracked_symlink_target(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    _initialize_git_repository(repository)
    link = repository / "evidence-link"
    link.symlink_to("tracked-a")
    first = experiment_node._git_snapshot(repository)

    link.unlink()
    link.symlink_to("tracked-b")
    second = experiment_node._git_snapshot(repository)

    assert first["worktree_sha256"] != second["worktree_sha256"]


def test_morai_build_files_are_derived_from_the_active_sensor_tree(tmp_path):
    launcher_root = tmp_path / "MoraiLauncher_Lin"
    data_root = launcher_root / "MoraiLauncher_Lin_Data"
    sensor_file = data_root / "SaveFile" / "Sensor" / "active.json"
    app_info = data_root / "app.info"
    assembly = data_root / "Managed" / "Assembly-CSharp.dll"
    launcher = launcher_root / "MoraiLauncher_Lin.x86_64"
    for path in (sensor_file, app_info, assembly, launcher):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding="utf-8")

    assert experiment_node._morai_build_files(sensor_file) == {
        "simulator:app_info": app_info,
        "simulator:assembly_csharp": assembly,
        "simulator:launcher": launcher,
    }


def test_morai_build_files_are_empty_outside_a_morai_data_tree(tmp_path):
    sensor_file = tmp_path / "sensor.json"
    sensor_file.write_text("fixture\n", encoding="utf-8")

    assert experiment_node._morai_build_files(sensor_file) == {}


def test_pulse10_is_a_reviewed_bounded_deadband_probe():
    pulse = fixed_command_profile("pulse10")

    assert tuple(phase.name for phase in pulse) == (
        "settle",
        "coast",
        "accelerate",
        "coast",
        "brake",
        "stopped",
    )
    assert max(phase.throttle for phase in pulse) == pytest.approx(0.10)
    assert sum(
        phase.duration_sec
        for phase in pulse
        if phase.throttle > 0.0
    ) == pytest.approx(0.50)
    assert all(phase.throttle == 0.0 or phase.brake == 0.0 for phase in pulse)


def test_pulse05_is_a_reviewed_lower_energy_repeatability_probe():
    pulse = fixed_command_profile("pulse05")

    assert tuple(phase.name for phase in pulse) == (
        "settle",
        "coast",
        "accelerate",
        "coast",
        "brake",
        "stopped",
    )
    assert max(phase.throttle for phase in pulse) == pytest.approx(0.05)
    assert sum(
        phase.duration_sec
        for phase in pulse
        if phase.throttle > 0.0
    ) == pytest.approx(0.50)
    assert all(phase.throttle == 0.0 or phase.brake == 0.0 for phase in pulse)


def test_actor_self_collision_is_recorded_but_not_an_abort():
    actor = ActorIdentity("Ego", "OBJECT_TYPE_VEHICLE", "")
    clock = FakeClock()
    assert nonself_collision_ids(_truth(clock, collisions=("Ego",)), actor) == ()
    assert nonself_collision_ids(
        _truth(clock, collisions=("Ego", "NPC-7")), actor
    ) == ("NPC-7",)


def test_initialization_samples_remain_raw_but_are_excluded_from_metrics():
    errors = (
        SimpleNamespace(phase="initialization"),
        SimpleNamespace(phase="settle"),
        SimpleNamespace(phase="closed_loop_track"),
    )

    selected = _evaluation_errors(errors)

    assert [item.phase for item in selected] == [
        "settle",
        "closed_loop_track",
    ]


def test_closed_loop_summary_uses_the_post_motion_stop_phase():
    assert _stopped_evaluation_phase("closed_loop_pulse") == "closed_loop_stop"
    assert _stopped_evaluation_phase("stationary") == "stopped"


def test_rpc_failure_always_reaches_repeated_full_brake_cleanup(limits):
    clock = FakeClock()
    states = [_truth(clock) for _ in range(100)]
    client = FakeSafeClient(clock, states, fail_non_brake=True)
    actor = ActorIdentity("Ego", "OBJECT_TYPE_VEHICLE", "")
    executor = BoundedProfileExecutor(
        client,
        actor,
        limits,
        monotonic_ns=clock.monotonic_ns,
        sleep=clock.sleep,
    )

    with pytest.raises(RuntimeError, match="injected"):
        executor.run(fixed_command_profile("tiny"))

    assert client.full_brakes >= 5


def test_fixed_profile_rejects_slow_truth_rpc_and_reaches_brake_cleanup(limits):
    clock = FakeClock()

    class SlowSafeClient(FakeSafeClient):
        def get_truth(self):
            if self.states:
                self.last_state = self.states.pop(0)
            rpc_start_ns = self.clock.monotonic_ns()
            self.clock.sleep(limits.truth_stale_timeout_sec + 0.01)
            return replace(
                self.last_state,
                rpc_start_monotonic_ns=rpc_start_ns,
                receipt_monotonic_ns=self.clock.monotonic_ns(),
            )

    client = SlowSafeClient(clock, [_truth(clock)])
    executor = BoundedProfileExecutor(
        client,
        client.actor_identity,
        limits,
        monotonic_ns=clock.monotonic_ns,
        sleep=clock.sleep,
    )

    with pytest.raises(ExperimentAbort, match="stale"):
        executor.run(fixed_command_profile("tiny"))

    assert client.full_brakes >= 6


def test_fixed_profile_phases_use_absolute_deadlines_with_slow_truth(limits):
    clock = FakeClock()

    class DelayedSafeClient(FakeSafeClient):
        def __init__(self, clock, states):
            super().__init__(clock, states)
            self.truth_calls = 0

        def get_truth(self):
            if self.states:
                self.last_state = self.states.pop(0)
            rpc_start_ns = self.clock.monotonic_ns()
            if self.truth_calls >= 46:
                self.clock.sleep(0.20)
            self.truth_calls += 1
            return replace(
                self.last_state,
                rpc_start_monotonic_ns=rpc_start_ns,
                receipt_monotonic_ns=self.clock.monotonic_ns(),
            )

    client = DelayedSafeClient(clock, [_truth(clock) for _ in range(200)])
    command_phases = []
    executor = BoundedProfileExecutor(
        client,
        client.actor_identity,
        limits,
        monotonic_ns=clock.monotonic_ns,
        sleep=clock.sleep,
        on_command=lambda phase, _command, _stamp: command_phases.append(phase),
    )

    executor.run(fixed_command_profile("tiny"))

    assert command_phases.count("accelerate") == 1


def test_unsafe_preflight_start_still_reaches_verified_brake_cleanup(limits):
    clock = FakeClock()
    client = FakeSafeClient(
        clock,
        [_truth(clock, speed=0.10)] + [_truth(clock) for _ in range(20)],
    )
    executor = BoundedProfileExecutor(
        client,
        ActorIdentity("Ego", "OBJECT_TYPE_VEHICLE", ""),
        limits,
        monotonic_ns=clock.monotonic_ns,
        sleep=clock.sleep,
    )

    with pytest.raises(ExperimentAbort, match="start"):
        executor.run(fixed_command_profile("tiny"))

    assert client.full_brakes >= 5


def test_arbitrary_phase_sequence_is_rejected_before_any_vehicle_rpc(limits):
    clock = FakeClock()
    client = FakeSafeClient(clock, [_truth(clock)])
    executor = BoundedProfileExecutor(
        client,
        ActorIdentity("Ego", "OBJECT_TYPE_VEHICLE", ""),
        limits,
        monotonic_ns=clock.monotonic_ns,
        sleep=clock.sleep,
    )

    from ad_morai_bridge_dev.eskf_experiment.types import CommandPhase

    with pytest.raises(ValueError, match="reviewed fixed profile"):
        executor.run((CommandPhase("accelerate", 10.0, 1.0, 0.0),))

    assert client.commands == []


def test_speed_and_travel_guards_abort_and_brake(limits):
    clock = FakeClock()
    actor = ActorIdentity("Ego", "OBJECT_TYPE_VEHICLE", "")
    for unsafe in (
        [_truth(clock), _truth(clock, speed=0.6)],
        [_truth(clock), _truth(clock, x=0.30)],
    ):
        client = FakeSafeClient(clock, unsafe + [_truth(clock) for _ in range(20)])
        executor = BoundedProfileExecutor(
            client,
            actor,
            limits,
            monotonic_ns=clock.monotonic_ns,
            sleep=clock.sleep,
        )
        with pytest.raises(ExperimentAbort):
            executor.run(fixed_command_profile("tiny"))
        assert client.full_brakes >= 5


def test_finite_speed_limit_violation_is_recorded_before_abort(limits):
    clock = FakeClock()
    client = FakeSafeClient(
        clock,
        [_truth(clock), _truth(clock, speed=0.6)]
        + [_truth(clock) for _ in range(20)],
    )
    recorded = []
    executor = BoundedProfileExecutor(
        client,
        client.actor_identity,
        limits,
        monotonic_ns=clock.monotonic_ns,
        sleep=clock.sleep,
        on_truth=lambda phase, truth: recorded.append((phase, truth)),
    )

    with pytest.raises(ExperimentAbort, match="maximum_speed_mps"):
        executor.run(fixed_command_profile("tiny"))

    assert any(
        phase == "settle" and truth.world_velocity_xyz[0] == pytest.approx(0.6)
        for phase, truth in recorded
    )


def test_cumulative_travel_guard_catches_motion_inside_start_radius(limits):
    clock = FakeClock()
    positions = (0.0, 0.10, 0.0, 0.10, 0.0)
    client = FakeSafeClient(
        clock,
        [_truth(clock, x=value) for value in positions]
        + [_truth(clock) for _ in range(20)],
    )
    executor = BoundedProfileExecutor(
        client,
        ActorIdentity("Ego", "OBJECT_TYPE_VEHICLE", ""),
        limits,
        monotonic_ns=clock.monotonic_ns,
        sleep=clock.sleep,
    )

    with pytest.raises(ExperimentAbort, match="cumulative"):
        executor.run(fixed_command_profile("tiny"))

    assert client.full_brakes >= 5


def test_command_slew_is_bounded_and_steering_stays_zero(limits):
    clock = FakeClock()
    states = [_truth(clock) for _ in range(200)]
    client = FakeSafeClient(clock, states)
    actor = ActorIdentity("Ego", "OBJECT_TYPE_VEHICLE", "")
    executor = BoundedProfileExecutor(
        client,
        actor,
        limits,
        monotonic_ns=clock.monotonic_ns,
        sleep=clock.sleep,
    )

    executor.run(fixed_command_profile("tiny"))

    controlled = [command for command in client.commands if command.brake < 1.0]
    maximum_step = limits.maximum_command_delta_per_sec / limits.command_rate_hz
    previous = VehicleCommand(0.0, 1.0, 0.0)
    for command in controlled:
        assert command.steer == 0.0
        assert abs(command.throttle - previous.throttle) <= maximum_step + 1e-12
        assert abs(command.brake - previous.brake) <= maximum_step + 1e-12
        previous = command
    assert all(math.isfinite(command.throttle) for command in client.commands)


def test_candidate_readiness_requires_both_debug_initialization_and_odometry():
    node = object.__new__(_RosEskfExperimentNode)
    node._lock = threading.Lock()
    node._candidates = ("baseline", "production_bias")
    node._latest_candidate_ns = {"baseline": 100, "production_bias": 100}
    node._candidate_initialized = {
        "baseline": False,
        "production_bias": False,
    }

    node._on_initialization("baseline", SimpleNamespace(data=[]))
    node._on_initialization("baseline", SimpleNamespace(data=[0.0]))
    assert not node._candidate_initialized["baseline"]

    node._on_initialization("baseline", SimpleNamespace(data=[1.0]))
    assert not node._candidates_ready()

    node._on_initialization("production_bias", SimpleNamespace(data=[1.0]))
    assert node._candidates_ready()

    node._latest_candidate_ns["baseline"] = 0
    assert not node._candidates_ready()


@pytest.mark.parametrize(
    ("states", "failure"),
    [
        (({"speed": 0.06},), "moved"),
        (({"collisions": ("NPC-7",)},), "collision"),
        (({"speed": float("nan")},), "non-finite"),
        (({"gear": "GEAR_MODE_R"},), "drive gear"),
        (({}, {"x": 0.30}), "stationary travel bound"),
        (
            ({}, {"x": 0.10}, {}, {"x": 0.10}),
            "cumulative stationary travel",
        ),
    ],
)
def test_stationary_collection_enforces_motion_and_collision_guards(
    limits, monkeypatch, states, failure
):
    clock = FakeClock()
    truths = [_truth(clock, **state) for state in states]
    client = FakeSafeClient(clock, truths + [_truth(clock) for _ in range(20)])
    node = object.__new__(_RosEskfExperimentNode)
    node._limits = limits
    node._monotonic_ns = clock.monotonic_ns
    node._abort = threading.Event()
    node._health_check = lambda _now_ns: None
    node._record_truth = lambda _phase, _truth_sample: None
    monkeypatch.setattr(experiment_node.time, "sleep", clock.sleep)

    with pytest.raises(ExperimentAbort, match=failure):
        node._collect_stationary(client, 1.0)


@pytest.mark.parametrize(
    ("receipt_offset_ns", "failure"),
    [(-1_000_000_000, "stale"), (1_000_000_000, "future")],
)
def test_stationary_collection_rejects_invalid_truth_receipt_time(
    limits, monkeypatch, receipt_offset_ns, failure
):
    clock = FakeClock()
    clock.nanoseconds = 1_000_000_000
    truth = replace(
        _truth(clock),
        receipt_monotonic_ns=clock.monotonic_ns() + receipt_offset_ns,
    )

    class FixedTruthClient:
        actor_identity = ActorIdentity("Ego", "OBJECT_TYPE_VEHICLE", "")

        def get_truth(self):
            return truth

    node = object.__new__(_RosEskfExperimentNode)
    node._limits = limits
    node._monotonic_ns = clock.monotonic_ns
    node._abort = threading.Event()
    node._health_check = lambda _now_ns: None
    node._record_truth = lambda _phase, _truth_sample: None
    monkeypatch.setattr(experiment_node.time, "sleep", clock.sleep)

    with pytest.raises(ExperimentAbort, match=failure):
        node._collect_stationary(FixedTruthClient(), 0.1)


def test_stationary_collection_rejects_slow_truth_rpc(
    limits, monkeypatch
):
    clock = FakeClock()
    clock.nanoseconds = 1_000_000_000
    truth = replace(
        _truth(clock),
        rpc_start_monotonic_ns=(
            clock.monotonic_ns()
            - round((limits.truth_stale_timeout_sec + 0.01) * 1.0e9)
        ),
    )

    class FixedTruthClient:
        actor_identity = ActorIdentity("Ego", "OBJECT_TYPE_VEHICLE", "")

        def get_truth(self):
            return truth

    node = object.__new__(_RosEskfExperimentNode)
    node._limits = limits
    node._monotonic_ns = clock.monotonic_ns
    node._abort = threading.Event()
    node._health_check = lambda _now_ns: None
    node._record_truth = lambda _phase, _truth_sample: None
    monkeypatch.setattr(experiment_node.time, "sleep", clock.sleep)

    with pytest.raises(ExperimentAbort, match="slow RPC"):
        node._collect_stationary(FixedTruthClient(), 0.1)


def test_every_executor_control_write_has_a_raw_command_callback(limits):
    clock = FakeClock()
    client = FakeSafeClient(clock, [_truth(clock) for _ in range(200)])
    callbacks = []
    executor = BoundedProfileExecutor(
        client,
        client.actor_identity,
        limits,
        monotonic_ns=clock.monotonic_ns,
        sleep=clock.sleep,
        on_command=lambda phase, command, receipt_ns: callbacks.append(
            (phase, command, receipt_ns)
        ),
    )

    executor.run(fixed_command_profile("tiny"))

    assert len(callbacks) == len(client.commands)
    assert callbacks[0][0] == "preflight"
    assert any(phase == "accelerate" for phase, _command, _stamp in callbacks)
    assert callbacks[-1][0] == "cleanup"
    assert all(command.steer == 0.0 for _phase, command, _stamp in callbacks)


class LifecycleClient:
    def __init__(
        self,
        clock,
        events,
        *,
        final_control_mode="CONTROL_MODE_KEYBOARD",
        close_error=None,
    ):
        self._clock = clock
        self._events = events
        self._final_control_mode = final_control_mode
        self._close_error = close_error
        self._cleanup_truth_callback = None
        self._safety_status = {
            "command_control_mode": None,
            "pre_waveform_stable_stop_status": "not_requested",
            "cleanup_stable_stop_status": "not_started",
            "restoration_status": "not_required",
            "restore_skipped_reason": None,
            "post_restore_stop_status": "not_required",
            "last_brake_rpc_status": "not_attempted",
        }
        self.actor_identity = ActorIdentity("Ego", "OBJECT_TYPE_VEHICLE", "")
        self.initial_control_mode = "CONTROL_MODE_KEYBOARD"

    def discover_ego(self):
        self._events.append("discover")
        return self.actor_identity

    def get_truth(self):
        self._events.append("truth")
        return _truth(self._clock)

    def get_control_mode(self):
        self._events.append("legacy_get_control_mode")
        return self._final_control_mode

    def enter_command_control(self):
        self._events.append("enter_command_control")
        self._safety_status.update(
            {
                "command_control_mode": "VEHICLE_CONTROL_AUTO_MODE",
                "pre_waveform_stable_stop_status": "verified",
                "restoration_status": "pending",
                "last_brake_rpc_status": "verified",
            }
        )

    @property
    def safety_status(self):
        self._events.append("safety_status")
        return dict(self._safety_status)

    @property
    def final_control_mode(self):
        self._events.append("final_control_mode")
        return self._final_control_mode

    def close(self):
        self._events.append("close_verified_brake")
        if self._cleanup_truth_callback is not None:
            self._cleanup_truth_callback("cleanup", _truth(self._clock))
        if self._close_error is not None:
            self._safety_status["cleanup_stable_stop_status"] = "failed"
            raise self._close_error
        self._safety_status["cleanup_stable_stop_status"] = "verified"
        if self._safety_status["restoration_status"] == "pending":
            self._safety_status["restoration_status"] = "verified"
            self._safety_status["post_restore_stop_status"] = "verified"


def _lifecycle_node(tmp_path, monkeypatch, limits, client, events):
    data_root = tmp_path / "data"
    monkeypatch.setenv("AD_DATA_DIR", str(data_root))
    experiment_config = tmp_path / "experiment.yaml"
    eskf_config = tmp_path / "eskf.yaml"
    sensor_file = tmp_path / "sensor.json"
    for path in (experiment_config, eskf_config, sensor_file):
        path.write_text("fixture\n", encoding="utf-8")

    node = object.__new__(_RosEskfExperimentNode)
    node._run_id = "lifecycle"
    node._experiment_path = experiment_config
    node._eskf_path = eskf_config
    node._active_sensor_file = sensor_file
    node._repository_root = tmp_path / "not-a-repository"
    node._document = {"candidates": {"baseline": {"enabled": True}}}
    node._profile = "stationary"
    node._config_drive_enabled = False
    node._launch_drive_enabled = False
    node._stationary_duration_sec = 1.0
    node._initialization_timeout_sec = 1.0
    node._grpc_target = "fixture:7789"
    node._grpc_timeout_sec = 1.0
    node._limits = limits
    node._monotonic_ns = client._clock.monotonic_ns

    def record_truth(phase, truth):
        events.append(f"truth:{phase}")
        _RosEskfExperimentNode._record_truth(node, phase, truth)

    node._record_truth = record_truth
    node._record_command = (
        lambda phase, _command, _stamp: events.append(f"command:{phase}")
    )
    node._health_check = lambda _stamp: None
    node._abort = threading.Event()
    node._lock = threading.Lock()
    node._phase = "initialization"
    node._truth_samples = []
    node._recorder = None

    def client_factory(_target, _timeout, passed_limits, on_truth):
        assert passed_limits is limits
        client._cleanup_truth_callback = on_truth
        return client

    node._client_factory = client_factory
    node._inputs_ready = lambda: True
    node._candidates_ready = lambda: True
    node._wait_until = lambda predicate, _timeout, _description: (
        None if predicate() else (_ for _ in ()).throw(AssertionError("not ready"))
    )
    node._publish_common_initial_pose = lambda: events.append("initial_pose")
    node._collect_stationary = lambda _client, _duration: events.append("collect")
    node._write_aligned_and_summary = lambda _artifacts: (
        events.append("metrics")
        or {
            "schema_version": 1,
            "run_id": "lifecycle",
            "profile": "stationary",
            "candidate_summaries": {},
        }
    )
    return node, data_root


def test_success_is_persisted_only_after_final_mode_check_and_verified_close(
    tmp_path, monkeypatch, limits
):
    events = []
    clock = FakeClock()
    client = LifecycleClient(clock, events)
    node, data_root = _lifecycle_node(
        tmp_path, monkeypatch, limits, client, events
    )
    real_write_json = experiment_node.write_json

    def recording_write(path, payload):
        events.append(f"write:{Path(path).name}:{payload.get('status', 'running')}")
        real_write_json(path, payload)

    monkeypatch.setattr(experiment_node, "write_json", recording_write)

    summary = node.run_experiment()

    assert summary["status"] == "complete"
    assert summary["cleanup_status"] == "verified"
    assert summary["initial_control_mode"] == "CONTROL_MODE_KEYBOARD"
    assert summary["final_control_mode"] == "CONTROL_MODE_KEYBOARD"
    close_index = events.index("close_verified_brake")
    assert events.index("final_control_mode") > close_index
    assert "legacy_get_control_mode" not in events
    assert events.index("write:manifest.json:complete") > close_index
    assert events.index("write:summary.json:complete") > close_index
    persisted = json.loads(
        (data_root / "experiments/eskf/lifecycle/summary.json").read_text()
    )
    assert persisted["status"] == "complete"


def test_cleanup_truth_callback_is_recorded_before_raw_recorder_closes(
    tmp_path, monkeypatch, limits
):
    events = []
    client = LifecycleClient(FakeClock(), events)
    node, data_root = _lifecycle_node(
        tmp_path, monkeypatch, limits, client, events
    )

    node.run_experiment()

    assert events.index("truth:cleanup") > events.index("close_verified_brake")
    raw_lines = [
        json.loads(line)
        for line in (
            data_root / "experiments/eskf/lifecycle/raw.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        row["stream"] == "truth" and row["phase"] == "cleanup"
        for row in raw_lines
    )


def test_artifacts_disclose_unavailable_covariance_and_counter_evidence(
    tmp_path, monkeypatch, limits
):
    events = []
    client = LifecycleClient(FakeClock(), events)
    node, data_root = _lifecycle_node(
        tmp_path, monkeypatch, limits, client, events
    )

    node.run_experiment()

    manifest = json.loads(
        (data_root / "experiments/eskf/lifecycle/manifest.json").read_text()
    )
    summary = json.loads(
        (data_root / "experiments/eskf/lifecycle/summary.json").read_text()
    )
    expected = {
        "bias_covariance_persisted": False,
        "estimator_diagnostic_counters_persisted": False,
        "cleanup_truth_callback_enabled": True,
    }
    assert manifest["evidence_capabilities"] == expected
    assert summary["evidence_capabilities"] == expected
    frame_contract = {
        "world_velocity_source": "ActorState.global_velocity",
        "world_acceleration_source": "ActorState.acceleration",
        "actor_acceleration_input_frame": "body_inferred",
        "world_acceleration_transform": "actor_rotation",
    }
    assert manifest["truth_frame_contract"] == frame_contract
    assert summary["truth_frame_contract"] == frame_contract


def test_control_mode_change_fails_the_run_but_still_closes_client(
    tmp_path, monkeypatch, limits
):
    events = []
    client = LifecycleClient(
        FakeClock(), events, final_control_mode="CONTROL_MODE_AUTONOMOUS"
    )
    node, data_root = _lifecycle_node(
        tmp_path, monkeypatch, limits, client, events
    )

    with pytest.raises(ExperimentAbort, match="control mode changed"):
        node.run_experiment()

    assert "close_verified_brake" in events
    summary = json.loads(
        (data_root / "experiments/eskf/lifecycle/summary.json").read_text()
    )
    assert summary["status"] == "failed"
    assert summary["cleanup_status"] == "failed"


def test_legacy_fixed_profile_cannot_enter_live_command_control(
    tmp_path, monkeypatch, limits
):
    events = []
    client = LifecycleClient(FakeClock(), events)
    node, data_root = _lifecycle_node(
        tmp_path, monkeypatch, limits, client, events
    )
    node._profile = "pulse10"
    node._config_drive_enabled = True
    node._launch_drive_enabled = True

    class RecordingExecutor:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("legacy fixed executor must remain disabled")

        def run(self, _phases):
            raise AssertionError("legacy fixed executor must remain disabled")

    monkeypatch.setattr(
        experiment_node, "BoundedProfileExecutor", RecordingExecutor
    )

    node.run_experiment()

    assert "collect" in events
    assert "enter_command_control" not in events
    assert "profile_run" not in events
    assert "close_verified_brake" in events
    manifest = json.loads(
        (data_root / "experiments/eskf/lifecycle/manifest.json").read_text()
    )
    assert (
        "morai_sim_api.actor.Actor/SetVehicleControlMode"
        in manifest["allowed_rpcs"]
    )
    assert manifest["control_mode_safety"] == {
        "command_control_mode": None,
        "pre_waveform_stable_stop_status": "not_requested",
        "cleanup_stable_stop_status": "verified",
        "restoration_status": "not_required",
        "restore_skipped_reason": None,
        "post_restore_stop_status": "not_required",
        "last_brake_rpc_status": "not_attempted",
    }


def test_closed_loop_drive_uses_feedback_executor_after_verified_entry(
    tmp_path, monkeypatch, limits
):
    events = []
    client = LifecycleClient(FakeClock(), events)
    node, _data_root = _lifecycle_node(
        tmp_path, monkeypatch, limits, client, events
    )
    node._profile = "closed_loop_pulse"
    node._config_drive_enabled = True
    node._launch_drive_enabled = True
    node._closed_loop_config = object()

    class RecordingClosedLoopExecutor:
        def __init__(self, *_args, **_kwargs):
            events.append("closed_loop_executor_created")

        def run(self, config):
            assert config is node._closed_loop_config
            events.append("closed_loop_run")

    monkeypatch.setattr(
        experiment_node,
        "ClosedLoopPulseExecutor",
        RecordingClosedLoopExecutor,
    )

    node.run_experiment()

    assert events.index("collect") < events.index("enter_command_control")
    assert events.index("enter_command_control") < events.index("closed_loop_run")
    assert events.index("closed_loop_run") < events.index("close_verified_brake")


def test_primary_and_close_failures_are_both_preserved_in_artifacts_and_error(
    tmp_path, monkeypatch, limits
):
    events = []
    client = LifecycleClient(
        FakeClock(), events, close_error=RuntimeError("brake verification failed")
    )
    node, data_root = _lifecycle_node(
        tmp_path, monkeypatch, limits, client, events
    )

    def fail_collection(_client, _duration):
        raise ValueError("candidate collection failed")

    node._collect_stationary = fail_collection

    with pytest.raises(ExperimentCleanupError) as caught:
        node.run_experiment()

    assert isinstance(caught.value.primary_error, ValueError)
    assert isinstance(caught.value.cleanup_error, RuntimeError)
    assert "candidate collection failed" in str(caught.value)
    assert "brake verification failed" in str(caught.value)
    manifest = json.loads(
        (data_root / "experiments/eskf/lifecycle/manifest.json").read_text()
    )
    summary = json.loads(
        (data_root / "experiments/eskf/lifecycle/summary.json").read_text()
    )
    assert manifest["status"] == "failed"
    assert manifest["cleanup_status"] == "failed"
    assert summary["failure_type"] == "ExperimentCleanupError"
    assert summary["cleanup_status"] == "failed"
    assert "candidate collection failed" in summary["failure"]
    assert "brake verification failed" in summary["failure"]


def test_failure_before_client_creation_marks_cleanup_not_required_consistently(
    tmp_path, monkeypatch, limits
):
    events = []
    node, data_root = _lifecycle_node(
        tmp_path,
        monkeypatch,
        limits,
        LifecycleClient(FakeClock(), events),
        events,
    )

    def fail_wait(_predicate, _timeout, _description):
        raise ExperimentAbort("inputs unavailable")

    node._wait_until = fail_wait

    with pytest.raises(ExperimentAbort, match="inputs unavailable"):
        node.run_experiment()

    manifest = json.loads(
        (data_root / "experiments/eskf/lifecycle/manifest.json").read_text()
    )
    summary = json.loads(
        (data_root / "experiments/eskf/lifecycle/summary.json").read_text()
    )
    assert manifest["cleanup_status"] == "not_required"
    assert summary["cleanup_status"] == "not_required"
    assert "close_verified_brake" not in events
