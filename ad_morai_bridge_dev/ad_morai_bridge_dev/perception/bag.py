"""Fail-closed rosbag recorder for MORAI classical perception validation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import signal
import stat
import subprocess
from typing import Callable, Mapping, Sequence

from ad_morai_bridge_dev.perception.validation_contract import (
    canonical_config_paths,
    canonical_launch_arguments,
    collect_runtime_provenance,
    config_records,
    file_record,
    load_dependency_pins,
    verify_dependency_sources,
)


REQUIRED_TOPICS = (
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

_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_DEPENDENCIES = frozenset({"autoware_universe", "muSSP"})


@dataclass(frozen=True)
class PreparedRecording:
    run_directory: Path
    bag_directory: Path
    manifest: Path
    command: tuple[str, ...]


def _resolved_directory(path: Path, name: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{name} does not exist: {candidate}") from error
    if not resolved.is_dir():
        raise ValueError(f"{name} must name a directory")
    return resolved


def _reject_existing_symlinks(path: Path, name: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            status = os.lstat(current)
        except FileNotFoundError:
            return
        except OSError as error:
            raise ValueError(f"{name} is inaccessible: {current}") from error
        if stat.S_ISLNK(status.st_mode):
            raise ValueError(f"{name} contains a symlink: {current}")


def _configured_data_root(data_root: Path | None) -> Path:
    if data_root is None:
        configured = os.environ.get("AD_DATA_DIR", "").strip()
        if not configured:
            raise ValueError("AD_DATA_DIR must be explicitly configured")
        data_root = Path(configured)
    candidate = Path(data_root).expanduser()
    if not candidate.is_absolute():
        raise ValueError("AD_DATA_DIR must be an absolute path")
    _reject_existing_symlinks(candidate, "AD_DATA_DIR")
    return _resolved_directory(candidate, "AD_DATA_DIR")


def _safe_run_id(value: str | None) -> str:
    if value is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        value = f"{stamp}-{secrets.token_hex(4)}"
    if not isinstance(value, str) or _RUN_ID.fullmatch(value) is None:
        raise ValueError("run_id is unsafe")
    return value


def _profile_record(path: Path) -> dict[str, str]:
    source = Path(path).expanduser()
    if not source.is_absolute():
        source = source.resolve()
    try:
        payload = source.read_bytes()
        resolved = source.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"profile is unreadable: {source}") from error
    if not resolved.is_file():
        raise ValueError("profile must be a regular file")
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _revision(value: object, name: str) -> str:
    if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
        raise ValueError(f"{name} must be a full lowercase Git commit")
    return value


def _open_directory(path: Path) -> int:
    return os.open(
        path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )


def _open_or_create_directory(parent_fd: int, name: str) -> int:
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_fd)
    except FileExistsError:
        pass
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
    except OSError as error:
        raise ValueError(
            f"recording destination contains a symlink or non-directory: {name}"
        ) from error
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError(f"recording destination is not a directory: {name}")
    return descriptor


def _write_new_manifest(
    run_fd: int, manifest: Mapping[str, object]
) -> None:
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    temporary_name = ".manifest.json.tmp"
    descriptor = os.open(
        temporary_name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=run_fd,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(
            temporary_name,
            "manifest.json",
            src_dir_fd=run_fd,
            dst_dir_fd=run_fd,
        )
        os.fsync(run_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=run_fd)
        except FileNotFoundError:
            pass


def prepare_recording(
    *,
    data_root: Path | None,
    repository_root: Path,
    profile: Path,
    actor_preset: str,
    git_commit: str,
    dependency_revisions: Mapping[str, str],
    dependency_pins: Mapping[str, str] | None = None,
    dependency_lock: Path | None = None,
    run_id: str | None = None,
    launch_arguments: Mapping[str, str] | None = None,
    config_paths: Mapping[str, Path] | None = None,
    runtime_provenance: Mapping[str, object] | None = None,
) -> PreparedRecording:
    """Reserve one unique run and persist its command before rosbag starts."""
    root = _configured_data_root(data_root)
    repository = _resolved_directory(repository_root, "repository_root")
    try:
        root.relative_to(repository)
    except ValueError:
        pass
    else:
        raise ValueError("AD_DATA_DIR must not resolve inside the source tree")
    if not isinstance(actor_preset, str) or not actor_preset.strip():
        raise ValueError("actor_preset must be nonempty")
    commit = _revision(git_commit, "git_commit")
    if set(dependency_revisions) != _DEPENDENCIES:
        raise ValueError("dependency_revisions must contain Autoware and muSSP")
    revisions = {
        name: _revision(dependency_revisions[name], name)
        for name in sorted(_DEPENDENCIES)
    }
    if dependency_pins is not None:
        if set(dependency_pins) != _DEPENDENCIES:
            raise ValueError("dependency_pins must contain Autoware and muSSP")
        pins = {
            name: _revision(dependency_pins[name], f"{name} pin")
            for name in sorted(_DEPENDENCIES)
        }
        if revisions != pins:
            raise ValueError(
                "dependency state does not match dependencies.repos"
            )
    config_manifest = config_records(config_paths) if config_paths else {}
    dependency_lock_manifest = (
        file_record(dependency_lock, "dependencies.repos")
        if dependency_lock is not None
        else None
    )
    runtime_manifest = dict(runtime_provenance or {})
    if runtime_manifest and set(runtime_manifest) != {
        "checker",
        "mussp",
        "tracker",
    }:
        raise ValueError("runtime_provenance is incomplete")
    identifier = _safe_run_id(run_id)
    base = root / "experiments" / "morai_classical_tracking"
    run_directory = base / identifier
    root_fd = None
    experiments_fd = None
    base_fd = None
    run_fd = None
    run_created = False
    try:
        root_fd = _open_directory(root)
        experiments_fd = _open_or_create_directory(root_fd, "experiments")
        base_fd = _open_or_create_directory(
            experiments_fd, "morai_classical_tracking"
        )
        try:
            os.mkdir(identifier, mode=0o700, dir_fd=base_fd)
        except FileExistsError as error:
            raise FileExistsError("run directory already exists") from error
        run_created = True
        run_fd = os.open(
            identifier,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=base_fd,
        )
        actual_run_directory = Path(
            os.readlink(f"/proc/self/fd/{run_fd}")
        ).resolve(strict=True)
        try:
            actual_run_directory.relative_to(repository)
        except ValueError:
            pass
        else:
            raise ValueError(
                "recording destination resolves inside the source tree"
            )

        run_directory = actual_run_directory
        bag_directory = run_directory / "bag"
        manifest_path = run_directory / "manifest.json"
        command = (
            "ros2",
            "bag",
            "record",
            "--output",
            str(bag_directory),
            *REQUIRED_TOPICS,
        )
        manifest = {
            "schema_version": 1,
            "run_id": identifier,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": commit,
            "profile": _profile_record(profile),
            "actor_preset": actor_preset,
            "dependency_revisions": revisions,
            "dependency_lock": dependency_lock_manifest,
            "launch_arguments": dict(sorted((launch_arguments or {}).items())),
            "configs": config_manifest,
            "runtime_provenance": runtime_manifest,
            "topics": list(REQUIRED_TOPICS),
            "bag_directory": str(bag_directory),
            "record_command": list(command),
            "recording_status": {
                "abort_reason": None,
                "exit_code": None,
                "phase": "prepared",
                "termination_signal": None,
            },
        }
        _write_new_manifest(run_fd, manifest)
        return PreparedRecording(
            run_directory=run_directory,
            bag_directory=bag_directory,
            manifest=manifest_path,
            command=command,
        )
    except Exception:
        if run_created and base_fd is not None:
            if run_fd is not None:
                try:
                    os.unlink(".manifest.json.tmp", dir_fd=run_fd)
                except FileNotFoundError:
                    pass
            try:
                os.rmdir(identifier, dir_fd=base_fd)
            except OSError:
                pass
        raise
    finally:
        for descriptor in (run_fd, base_fd, experiments_fd, root_fd):
            if descriptor is not None:
                os.close(descriptor)


class PerceptionBagRecorder:
    def __init__(
        self,
        prepared: PreparedRecording,
        *,
        process_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
        group_signal: Callable[[int, int], None] = os.killpg,
    ) -> None:
        self._prepared = prepared
        self._process_factory = process_factory
        self._group_signal = group_signal
        self._process = None
        self._status_final = False
        self._stopped = False

    def _persist_status(
        self,
        phase: str,
        *,
        exit_code: int | None = None,
        abort_reason: str | None = None,
        termination_signal: str | None = None,
        final: bool = False,
    ) -> None:
        _reject_existing_symlinks(
            self._prepared.run_directory, "run_directory"
        )
        run_fd = _open_directory(self._prepared.run_directory)
        try:
            manifest_fd = os.open(
                "manifest.json",
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=run_fd,
            )
            with os.fdopen(manifest_fd, "r", encoding="utf-8") as stream:
                document = json.load(stream)
            document["recording_status"] = {
                "abort_reason": abort_reason,
                "exit_code": exit_code,
                "phase": phase,
                "termination_signal": termination_signal,
            }
            encoded = json.dumps(document, indent=2, sort_keys=True) + "\n"
            temporary_name = ".manifest.status.tmp"
            temporary_fd = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=run_fd,
            )
            try:
                with os.fdopen(
                    temporary_fd, "w", encoding="utf-8"
                ) as stream:
                    temporary_fd = -1
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(
                    temporary_name,
                    "manifest.json",
                    src_dir_fd=run_fd,
                    dst_dir_fd=run_fd,
                )
                os.fsync(run_fd)
            finally:
                if temporary_fd >= 0:
                    os.close(temporary_fd)
                try:
                    os.unlink(temporary_name, dir_fd=run_fd)
                except FileNotFoundError:
                    pass
        finally:
            os.close(run_fd)
        if final:
            self._status_final = True

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("recorder is already started")
        if not self._prepared.manifest.is_file():
            raise RuntimeError("recording manifest is missing")
        try:
            self._process = self._process_factory(
                list(self._prepared.command), start_new_session=True
            )
        except Exception as error:
            self._persist_status(
                "spawn_failed",
                abort_reason=str(error),
                final=True,
            )
            raise
        self._persist_status("recording")

    def wait(self) -> int:
        if self._process is None:
            raise RuntimeError("recorder is not started")
        exit_code = int(self._process.wait())
        phase = "completed" if exit_code == 0 else "child_exited"
        self._persist_status(phase, exit_code=exit_code, final=True)
        return exit_code

    def stop(
        self,
        *,
        timeout_sec: float = 10.0,
        escalation_timeout_sec: float = 5.0,
        abort_reason: str | None = None,
    ) -> None:
        if self._stopped or self._status_final:
            return
        if self._process is None:
            return
        return_code = self._process.poll()
        if return_code is not None:
            phase = "completed" if return_code == 0 else "child_exited"
            self._persist_status(
                phase, exit_code=int(return_code), final=True
            )
            self._stopped = True
            return

        reason = abort_reason or "recorder stop requested"
        termination_signal = None
        for requested_signal, timeout in (
            (signal.SIGINT, timeout_sec),
            (signal.SIGTERM, escalation_timeout_sec),
            (signal.SIGKILL, escalation_timeout_sec),
        ):
            termination_signal = signal.Signals(requested_signal).name
            try:
                self._group_signal(self._process.pid, requested_signal)
            except ProcessLookupError:
                return_code = self._process.poll()
                if return_code is None:
                    try:
                        return_code = int(
                            self._process.wait(timeout=timeout)
                        )
                    except subprocess.TimeoutExpired as error:
                        failure_reason = (
                            f"{reason}; process group disappeared but child "
                            "did not exit"
                        )
                        self._persist_status(
                            "termination_failed",
                            abort_reason=failure_reason,
                            termination_signal=termination_signal,
                        )
                        raise RuntimeError(failure_reason) from error
                phase = (
                    "completed" if int(return_code) == 0 else "child_exited"
                )
                self._persist_status(
                    phase,
                    exit_code=int(return_code),
                    final=True,
                )
                self._stopped = True
                return
            except OSError as error:
                failure_reason = (
                    f"{reason}; {type(error).__name__}: {error}"
                )
                self._persist_status(
                    "termination_failed",
                    abort_reason=failure_reason,
                    termination_signal=termination_signal,
                )
                raise
            try:
                return_code = int(self._process.wait(timeout=timeout))
                break
            except subprocess.TimeoutExpired:
                continue
            except OSError as error:
                failure_reason = (
                    f"{reason}; {type(error).__name__}: {error}"
                )
                self._persist_status(
                    "termination_failed",
                    abort_reason=failure_reason,
                    termination_signal=termination_signal,
                )
                raise
        else:
            self._persist_status(
                "termination_failed",
                abort_reason=reason,
                termination_signal=termination_signal,
                final=True,
            )
            self._stopped = True
            raise RuntimeError("rosbag process group survived SIGKILL timeout")
        self._persist_status(
            "aborted",
            exit_code=return_code,
            abort_reason=reason,
            termination_signal=termination_signal,
            final=True,
        )
        self._stopped = True


class _WrapperSignal(RuntimeError):
    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(signal.Signals(signum).name)


def run_recorder_session(
    recorder: PerceptionBagRecorder,
    *,
    on_started: Callable[[], None] | None = None,
) -> int:
    """Run one recorder with wrapper-signal cleanup guaranteed in finally."""
    handled_signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    previous_handlers = {}

    def handle_signal(signum, _frame):
        raise _WrapperSignal(signum)

    for signum in handled_signals:
        previous_handlers[signum] = signal.signal(signum, handle_signal)
    abort_reason = None
    try:
        recorder.start()
        if on_started is not None:
            on_started()
        return recorder.wait()
    except _WrapperSignal as error:
        abort_reason = f"wrapper received {signal.Signals(error.signum).name}"
        return 128 + error.signum
    except BaseException as error:
        abort_reason = f"wrapper exception: {type(error).__name__}: {error}"
        raise
    finally:
        try:
            recorder.stop(abort_reason=abort_reason)
        finally:
            for signum, previous in previous_handlers.items():
                signal.signal(signum, previous)


def _git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return _revision(result.stdout.strip(), str(path))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--actor-preset", required=True)
    parser.add_argument("--run-id")
    arguments = parser.parse_args(argv)
    repository = arguments.repository_root.resolve()
    workspace_sources = repository.parent
    dependencies_lock = repository / "dependencies.repos"
    dependency_pins = load_dependency_pins(dependencies_lock)
    dependency_revisions = verify_dependency_sources(
        workspace_sources, dependency_pins
    )
    from ament_index_python.packages import get_package_share_directory

    perception_share = Path(
        get_package_share_directory("ad_lidar_perception")
    )
    description_share = Path(
        get_package_share_directory("ad_description")
    )
    launch_arguments = canonical_launch_arguments(
        perception_share, description_share
    )
    configs = canonical_config_paths(perception_share, description_share)
    runtime_provenance = collect_runtime_provenance(perception_share)
    prepared = prepare_recording(
        data_root=None,
        repository_root=repository,
        profile=arguments.profile,
        actor_preset=arguments.actor_preset,
        run_id=arguments.run_id,
        git_commit=_git_revision(repository),
        dependency_revisions=dependency_revisions,
        dependency_pins=dependency_pins,
        dependency_lock=dependencies_lock,
        launch_arguments=launch_arguments,
        config_paths=configs,
        runtime_provenance=runtime_provenance,
    )
    recorder = PerceptionBagRecorder(prepared)
    return run_recorder_session(
        recorder,
        on_started=lambda: print(str(prepared.run_directory), flush=True),
    )


if __name__ == "__main__":
    raise SystemExit(main())
