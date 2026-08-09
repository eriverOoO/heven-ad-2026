import json
import os
import signal
import subprocess

import pytest

import ad_morai_bridge_dev.perception.bag as perception_bag
from ad_morai_bridge_dev.perception.bag import (
    REQUIRED_TOPICS,
    PerceptionBagRecorder,
    run_recorder_session,
    prepare_recording,
)
from ad_morai_bridge_dev.perception.validation_contract import (
    canonical_config_paths,
    canonical_launch_arguments,
    load_dependency_pins,
    verify_dependency_sources,
)


EXPECTED_TOPICS = (
    "/ad/sensors/lidar/raw",
    "/ad/sensors/lidar/points",
    "/ad/perception/lidar/cropped",
    "/ad/perception/lidar/cloud",
    "/ad/perception/lidar/ground",
    "/ad/perception/lidar/nonground",
    "/ad/perception/lidar/clusters",
    "/ad/perception/objects/detected",
    "/ad/perception/objects/tracked",
    "/ad/perception/objects/predicted",
    "/ad/perception/objects/prediction_debug",
    "/tf",
    "/tf_static",
    "/ad/localization/odometry",
    "/ad/dev/objects",
    "/ad/dev/vehicle/ego_status",
    "/ad/sensors/timing",
)


class RecordingProcess:
    def __init__(self):
        self.signals = []
        self.wait_timeouts = []
        self.returncode = None
        self.pid = 731

    def send_signal(self, value):
        self.signals.append(value)
        self.returncode = 0

    def wait(self, timeout):
        self.wait_timeouts.append(timeout)
        return self.returncode

    def poll(self):
        return self.returncode


def test_manifest_covers_every_stage_truth_localization_and_timing(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    profile = repository_root / "profile.yaml"
    profile.write_text("schema_version: 1\n", encoding="utf-8")

    prepared = prepare_recording(
        data_root=data_root,
        repository_root=repository_root,
        profile=profile,
        actor_preset="roundabout_loop",
        run_id="deterministic-run",
        git_commit="0123456789abcdef0123456789abcdef01234567",
        dependency_revisions={
            "autoware_universe": "a" * 40,
            "muSSP": "b" * 40,
        },
    )

    assert REQUIRED_TOPICS == EXPECTED_TOPICS
    manifest = json.loads(prepared.manifest.read_text(encoding="utf-8"))
    assert tuple(manifest["topics"]) == EXPECTED_TOPICS
    assert manifest["actor_preset"] == "roundabout_loop"
    assert manifest["git_commit"] == "0123456789abcdef0123456789abcdef01234567"
    assert manifest["profile"]["sha256"] == (
        "5c9536c0f64193f535c425baf9f2c7431b89eb46bab5fb85d4bfcffabb565f43"
    )
    assert manifest["dependency_revisions"] == {
        "autoware_universe": "a" * 40,
        "muSSP": "b" * 40,
    }
    assert manifest["record_command"] == [
        "ros2",
        "bag",
        "record",
        "--output",
        str(prepared.bag_directory),
        *EXPECTED_TOPICS,
    ]
    assert prepared.run_directory.parent == (
        data_root / "experiments" / "morai_classical_tracking"
    )


def test_recorder_writes_manifest_before_process_and_stops_with_sigint(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    profile = repository_root / "profile.yaml"
    profile.write_text("schema_version: 1\n", encoding="utf-8")
    prepared = prepare_recording(
        data_root=data_root,
        repository_root=repository_root,
        profile=profile,
        actor_preset="highway_loop",
        run_id="ordered-start",
        git_commit="0" * 40,
        dependency_revisions={"autoware_universe": "1" * 40, "muSSP": "2" * 40},
    )
    process = RecordingProcess()

    def process_factory(command, **kwargs):
        assert prepared.manifest.is_file()
        assert command == list(prepared.command)
        assert kwargs == {"start_new_session": True}
        return process

    group_signals = []
    recorder = PerceptionBagRecorder(
        prepared,
        process_factory=process_factory,
        group_signal=lambda pid, value: (
            group_signals.append((pid, value)),
            setattr(process, "returncode", 130),
        ),
    )
    recorder.start()
    recorder.stop(timeout_sec=2.5)

    assert group_signals == [(process.pid, signal.SIGINT)]
    assert process.wait_timeouts == [2.5]


def test_recording_refuses_unset_source_tree_or_reused_destination(
    tmp_path, monkeypatch
):
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    profile = repository_root / "profile.yaml"
    profile.write_text("schema_version: 1\n", encoding="utf-8")
    monkeypatch.delenv("AD_DATA_DIR", raising=False)
    common = {
        "repository_root": repository_root,
        "profile": profile,
        "actor_preset": "roundabout_loop",
        "git_commit": "0" * 40,
        "dependency_revisions": {
            "autoware_universe": "1" * 40,
            "muSSP": "2" * 40,
        },
    }

    with pytest.raises(ValueError, match="AD_DATA_DIR"):
        prepare_recording(data_root=None, run_id="missing-root", **common)
    with pytest.raises(ValueError, match="source tree"):
        prepare_recording(
            data_root=repository_root,
            run_id="source-fallback",
            **common,
        )

    data_root = tmp_path / "data"
    data_root.mkdir()
    prepare_recording(data_root=data_root, run_id="duplicate", **common)
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_recording(data_root=data_root, run_id="duplicate", **common)


def test_recording_rejects_intermediate_symlink_into_source_tree(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    source_destination = repository_root / "redirected-output"
    source_destination.mkdir()
    (data_root / "experiments").symlink_to(
        source_destination, target_is_directory=True
    )
    profile = repository_root / "profile.yaml"
    profile.write_text("schema_version: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="symlink"):
        prepare_recording(
            data_root=data_root,
            repository_root=repository_root,
            profile=profile,
            actor_preset="roundabout_loop",
            run_id="must-not-enter-source",
            git_commit="0" * 40,
            dependency_revisions={
                "autoware_universe": "1" * 40,
                "muSSP": "2" * 40,
            },
        )

    assert list(source_destination.iterdir()) == []


def test_prepare_failure_removes_only_its_new_empty_run_directory(
    tmp_path, monkeypatch
):
    data_root = tmp_path / "data"
    data_root.mkdir()
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    profile = repository_root / "profile.yaml"
    profile.write_text("schema_version: 1\n", encoding="utf-8")
    base = data_root / "experiments" / "morai_classical_tracking"
    base.mkdir(parents=True)
    sentinel = base / "preserve-me"
    sentinel.mkdir()

    monkeypatch.setattr(
        perception_bag,
        "_write_new_manifest",
        lambda *_args: (_ for _ in ()).throw(OSError("write failed")),
    )
    with pytest.raises(OSError, match="write failed"):
        prepare_recording(
            data_root=data_root,
            repository_root=repository_root,
            profile=profile,
            actor_preset="roundabout_loop",
            run_id="failed-invocation",
            git_commit="0" * 40,
            dependency_revisions={
                "autoware_universe": "1" * 40,
                "muSSP": "2" * 40,
            },
        )

    assert sentinel.is_dir()
    assert not (base / "failed-invocation").exists()


class EscalatingProcess:
    def __init__(self, outcomes):
        self.pid = 991
        self.returncode = None
        self.outcomes = list(outcomes)
        self.wait_timeouts = []

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        self.returncode = outcome
        return outcome

    def poll(self):
        return self.returncode


class ExitRaceProcess:
    def __init__(self):
        self.pid = 992
        self.returncode = None
        self.wait_timeouts = []

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        if self.returncode is None:
            raise subprocess.TimeoutExpired("ros2", timeout)
        return self.returncode

    def poll(self):
        return self.returncode


def _prepared(tmp_path, run_id="lifecycle"):
    data_root = tmp_path / "data"
    data_root.mkdir()
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    profile = repository_root / "profile.yaml"
    profile.write_text("schema_version: 1\n", encoding="utf-8")
    return prepare_recording(
        data_root=data_root,
        repository_root=repository_root,
        profile=profile,
        actor_preset="roundabout_loop",
        run_id=run_id,
        git_commit="0" * 40,
        dependency_revisions={
            "autoware_universe": "1" * 40,
            "muSSP": "2" * 40,
        },
    )


def test_stop_escalates_process_group_and_is_idempotent(tmp_path):
    prepared = _prepared(tmp_path)
    timed_out = subprocess.TimeoutExpired("ros2", 1.0)
    process = EscalatingProcess((timed_out, timed_out, -signal.SIGKILL))
    group_signals = []
    recorder = PerceptionBagRecorder(
        prepared,
        process_factory=lambda *_args, **_kwargs: process,
        group_signal=lambda pid, value: group_signals.append((pid, value)),
    )
    recorder.start()

    recorder.stop(
        timeout_sec=1.0,
        escalation_timeout_sec=2.0,
        abort_reason="wrapper received SIGTERM",
    )
    recorder.stop(abort_reason="duplicate stop")

    assert group_signals == [
        (process.pid, signal.SIGINT),
        (process.pid, signal.SIGTERM),
        (process.pid, signal.SIGKILL),
    ]
    assert process.wait_timeouts == [1.0, 2.0, 2.0]
    manifest = json.loads(prepared.manifest.read_text(encoding="utf-8"))
    assert manifest["recording_status"] == {
        "abort_reason": "wrapper received SIGTERM",
        "exit_code": -signal.SIGKILL,
        "phase": "aborted",
        "termination_signal": "SIGKILL",
    }


@pytest.mark.parametrize(
    "race_signal, exit_code",
    (
        (signal.SIGINT, 0),
        (signal.SIGTERM, 7),
        (signal.SIGKILL, -signal.SIGKILL),
    ),
)
def test_stop_persists_child_exit_race_at_every_escalation(
    tmp_path, race_signal, exit_code
):
    prepared = _prepared(tmp_path)
    process = ExitRaceProcess()
    group_signals = []

    def group_signal(pid, requested_signal):
        group_signals.append((pid, requested_signal))
        if requested_signal == race_signal:
            process.returncode = exit_code
            raise ProcessLookupError("process group exited")

    recorder = PerceptionBagRecorder(
        prepared,
        process_factory=lambda *_args, **_kwargs: process,
        group_signal=group_signal,
    )
    recorder.start()

    recorder.stop(timeout_sec=1.0, escalation_timeout_sec=2.0)
    recorder.stop()

    expected_signals = [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]
    assert group_signals == [
        (process.pid, value)
        for value in expected_signals[: expected_signals.index(race_signal) + 1]
    ]
    manifest = json.loads(prepared.manifest.read_text(encoding="utf-8"))
    assert manifest["recording_status"] == {
        "abort_reason": None,
        "exit_code": exit_code,
        "phase": "completed" if exit_code == 0 else "child_exited",
        "termination_signal": None,
    }


def test_unexpected_group_signal_error_is_persisted_and_stop_can_retry(tmp_path):
    prepared = _prepared(tmp_path)
    process = ExitRaceProcess()
    attempts = []

    def group_signal(_pid, requested_signal):
        attempts.append(requested_signal)
        if len(attempts) == 1:
            raise PermissionError("not permitted")
        process.returncode = 130

    recorder = PerceptionBagRecorder(
        prepared,
        process_factory=lambda *_args, **_kwargs: process,
        group_signal=group_signal,
    )
    recorder.start()

    with pytest.raises(PermissionError, match="not permitted"):
        recorder.stop(abort_reason="wrapper received SIGTERM")
    failed = json.loads(prepared.manifest.read_text(encoding="utf-8"))
    assert failed["recording_status"]["phase"] == "termination_failed"
    assert "PermissionError: not permitted" in (
        failed["recording_status"]["abort_reason"]
    )

    recorder.stop(abort_reason="retry cleanup")
    recorder.stop()
    assert attempts == [signal.SIGINT, signal.SIGINT]
    completed = json.loads(prepared.manifest.read_text(encoding="utf-8"))
    assert completed["recording_status"]["phase"] == "aborted"
    assert completed["recording_status"]["exit_code"] == 130


def test_normal_wait_persists_child_exit_and_stop_does_not_signal(tmp_path):
    prepared = _prepared(tmp_path)
    process = EscalatingProcess((0,))
    group_signals = []
    recorder = PerceptionBagRecorder(
        prepared,
        process_factory=lambda *_args, **_kwargs: process,
        group_signal=lambda pid, value: group_signals.append((pid, value)),
    )
    recorder.start()

    assert recorder.wait() == 0
    recorder.stop()

    assert group_signals == []
    manifest = json.loads(prepared.manifest.read_text(encoding="utf-8"))
    assert manifest["recording_status"]["phase"] == "completed"
    assert manifest["recording_status"]["exit_code"] == 0


def test_spawn_failure_is_persisted_without_a_live_child(tmp_path):
    prepared = _prepared(tmp_path)

    def fail_spawn(*_args, **_kwargs):
        raise OSError("spawn denied")

    recorder = PerceptionBagRecorder(prepared, process_factory=fail_spawn)
    with pytest.raises(OSError, match="spawn denied"):
        recorder.start()
    recorder.stop()

    manifest = json.loads(prepared.manifest.read_text(encoding="utf-8"))
    assert manifest["recording_status"]["phase"] == "spawn_failed"
    assert manifest["recording_status"]["abort_reason"] == "spawn denied"


@pytest.mark.parametrize(
    "wrapper_signal", (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
)
def test_wrapper_signals_always_stop_recorder_and_record_reason(
    monkeypatch, wrapper_signal
):
    installed = {}
    monkeypatch.setattr(
        signal,
        "signal",
        lambda signum, handler: installed.setdefault(signum, handler),
    )

    class SessionRecorder:
        def __init__(self):
            self.stop_reasons = []

        def start(self):
            return None

        def wait(self):
            installed[wrapper_signal](wrapper_signal, None)

        def stop(self, *, abort_reason=None, **_kwargs):
            self.stop_reasons.append(abort_reason)

    recorder = SessionRecorder()
    result = run_recorder_session(recorder)

    assert result == 128 + wrapper_signal
    assert recorder.stop_reasons == [
        f"wrapper received {signal.Signals(wrapper_signal).name}"
    ]


def test_manifest_records_shared_launch_all_configs_and_runtime_provenance(
    tmp_path
):
    data_root = tmp_path / "data"
    data_root.mkdir()
    repository_root = tmp_path / "repo"
    perception_share = repository_root / "ad_lidar_perception"
    description_share = repository_root / "ad_description"
    perception_share.mkdir(parents=True)
    description_share.mkdir()
    profile = perception_share / "profile.yaml"
    profile.write_text("profile\n", encoding="utf-8")
    config_paths = canonical_config_paths(
        perception_share, description_share
    )
    for index, path in enumerate(config_paths.values()):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"config-{index}\n", encoding="utf-8")
    dependencies_lock = repository_root / "dependencies.repos"
    dependencies_lock.write_text("lock\n", encoding="utf-8")
    pins = {"autoware_universe": "a" * 40, "muSSP": "b" * 40}
    runtime = {
        "checker": {
            "message": (
                "Autoware perception selection verified "
                "(tracker=autoware)."
            ),
            "result": "passed",
        },
        "mussp": {
            "package": "mussp",
            "prefix": "/overlay/install/mussp",
            "version": "0.1.0",
        },
        "tracker": {
            "package": "autoware_multi_object_tracker",
            "prefix": "/overlay/install/autoware_multi_object_tracker",
            "version": "0.51.0",
        },
    }

    prepared = prepare_recording(
        data_root=data_root,
        repository_root=repository_root,
        profile=profile,
        actor_preset="roundabout_loop",
        run_id="full-provenance",
        git_commit="0" * 40,
        dependency_revisions=pins,
        dependency_pins=pins,
        dependency_lock=dependencies_lock,
        launch_arguments=canonical_launch_arguments(
            perception_share, description_share
        ),
        config_paths=config_paths,
        runtime_provenance=runtime,
    )

    manifest = json.loads(prepared.manifest.read_text(encoding="utf-8"))
    assert manifest["launch_arguments"] == canonical_launch_arguments(
        perception_share, description_share
    )
    assert set(manifest["configs"]) == {
        "adaptive_euclidean_cluster",
        "autoware_lock",
        "composition",
        "ground_segmentation",
        "prediction",
        "self_crop",
        "sensor_mounts",
        "tracker",
        "vehicle_parameters",
    }
    assert all(
        len(record["sha256"]) == 64
        for record in manifest["configs"].values()
    )
    assert manifest["runtime_provenance"] == runtime
    assert manifest["dependency_lock"]["path"] == str(dependencies_lock)
    assert len(manifest["dependency_lock"]["sha256"]) == 64


def test_dependency_pins_reject_revision_and_dirty_source_drift(tmp_path):
    lock = tmp_path / "dependencies.repos"
    lock.write_text(
        "repositories:\n"
        "  autoware_universe:\n"
        "    type: git\n"
        "    url: https://example.invalid/autoware.git\n"
        f"    version: {'a' * 40}\n"
        "  muSSP:\n"
        "    type: git\n"
        "    url: https://example.invalid/mussp.git\n"
        f"    version: {'b' * 40}\n",
        encoding="utf-8",
    )
    sources = tmp_path / "src"
    (sources / "autoware_universe").mkdir(parents=True)
    (sources / "muSSP").mkdir()
    pins = load_dependency_pins(lock)
    state = {
        "autoware_universe": ["a" * 40, ""],
        "muSSP": ["b" * 40, ""],
    }

    def runner(command, **_kwargs):
        name = os.path.basename(command[2])
        value = state[name].pop(0)
        return subprocess.CompletedProcess(command, 0, value + "\n", "")

    assert verify_dependency_sources(
        sources, pins, command_runner=runner
    ) == pins

    state["autoware_universe"] = ["c" * 40, ""]
    state["muSSP"] = ["b" * 40, ""]
    with pytest.raises(ValueError, match="autoware_universe revision drift"):
        verify_dependency_sources(sources, pins, command_runner=runner)

    state["autoware_universe"] = ["a" * 40, " M tracked.cpp"]
    state["muSSP"] = ["b" * 40, ""]
    with pytest.raises(ValueError, match="autoware_universe is dirty"):
        verify_dependency_sources(sources, pins, command_runner=runner)


def test_prepare_rejects_dependency_revision_drift_before_spawn(tmp_path):
    prepared_inputs = _prepared(tmp_path, run_id="baseline")
    assert prepared_inputs.manifest.is_file()
    data_root = tmp_path / "other-data"
    data_root.mkdir()
    repository_root = tmp_path / "other-repo"
    repository_root.mkdir()
    profile = repository_root / "profile.yaml"
    profile.write_text("profile\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match dependencies.repos"):
        prepare_recording(
            data_root=data_root,
            repository_root=repository_root,
            profile=profile,
            actor_preset="roundabout_loop",
            run_id="drifted",
            git_commit="0" * 40,
            dependency_revisions={
                "autoware_universe": "c" * 40,
                "muSSP": "b" * 40,
            },
            dependency_pins={
                "autoware_universe": "a" * 40,
                "muSSP": "b" * 40,
            },
        )
