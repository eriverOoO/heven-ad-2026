"""Contained, deterministic artifact storage for ESKF experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import Mapping


_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_ARTIFACT_NAMES = {"manifest.json", "raw.jsonl", "aligned.csv", "summary.json"}


@dataclass(frozen=True)
class RunArtifacts:
    run_directory: Path
    manifest: Path
    raw: Path
    aligned: Path
    summary: Path


def validate_run_id(run_id: object) -> str:
    """Return a valid run ID or reject values unsafe for an artifact path."""
    if not isinstance(run_id, str) or _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError(
            "run_id must contain only letters, digits, dot, underscore, or dash"
        )
    return run_id


def create_run_artifacts(run_id: str) -> RunArtifacts:
    """Create a run directory below the explicitly configured AD_DATA_DIR."""
    root = _configured_data_root()
    run_id = validate_run_id(run_id)
    run_directory = root / "experiments" / "eskf" / run_id
    resolved_run_directory = run_directory.resolve(strict=False)
    if not _is_relative_to(resolved_run_directory, root):
        raise ValueError("run directory resolves outside AD_DATA_DIR")
    run_directory.parent.mkdir(parents=True, exist_ok=True)
    run_directory.mkdir(exist_ok=False)
    resolved_after_creation = run_directory.resolve(strict=True)
    if not _is_relative_to(resolved_after_creation, root):
        raise ValueError("run directory resolves outside AD_DATA_DIR")
    return RunArtifacts(
        run_directory=run_directory,
        manifest=run_directory / "manifest.json",
        raw=run_directory / "raw.jsonl",
        aligned=run_directory / "aligned.csv",
        summary=run_directory / "summary.json",
    )


class JsonlRecorder:
    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        _validate_run_artifact_target(self._path, required_name="raw.jsonl")
        self._stream = self._path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self._closed = False

    def write(self, stream: str, record: Mapping[str, object]) -> None:
        if self._closed:
            raise ValueError("JSONL recorder is closed")
        receipt_time = record.get("receipt_monotonic_ns")
        if not isinstance(receipt_time, int) or isinstance(receipt_time, bool):
            raise ValueError("receipt_monotonic_ns must be an integer")
        if not stream:
            raise ValueError("stream must be nonempty")
        reserved_fields = {"schema_version", "stream"}.intersection(record)
        if reserved_fields:
            raise ValueError(
                f"record contains reserved field(s): {sorted(reserved_fields)}"
            )
        payload = {"schema_version": 1, "stream": stream, **record}
        try:
            line = json.dumps(
                payload,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"record is not strict JSON: {exc}") from exc
        with self._lock:
            self._stream.write(line + "\n")
            self._stream.flush()

    def close(self) -> None:
        if not self._closed:
            self._stream.close()
            self._closed = True

    def __enter__(self) -> "JsonlRecorder":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_manifest(files: Mapping[str, Path]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for name, source in sorted(files.items()):
        path = Path(source).resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"manifest source is not a file: {path}")
        result[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return result


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    target = Path(path)
    _validate_run_artifact_target(target)
    try:
        serialized = json.dumps(
            payload,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise ValueError(f"payload is not strict JSON: {exc}") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        delete=False,
    ) as temporary:
        temporary.write(serialized)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, target)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _configured_data_root() -> Path:
    raw = os.environ.get("AD_DATA_DIR")
    if raw is None or not raw.strip():
        raise ValueError("AD_DATA_DIR must be explicitly configured")
    configured = Path(raw).expanduser()
    if not configured.is_absolute():
        raise ValueError("AD_DATA_DIR must be an absolute path")
    return configured.resolve(strict=False)


def _validate_run_artifact_target(
    path: Path, *, required_name: str | None = None
) -> None:
    root = _configured_data_root()
    resolved = path.resolve(strict=False)
    if not _is_relative_to(resolved, root):
        raise ValueError("artifact target resolves outside AD_DATA_DIR")
    relative = resolved.relative_to(root)
    parts = relative.parts
    valid_run_id = False
    if len(parts) == 4:
        try:
            validate_run_id(parts[2])
            valid_run_id = True
        except ValueError:
            pass
    valid_shape = (
        len(parts) == 4
        and parts[0] == "experiments"
        and parts[1] == "eskf"
        and valid_run_id
        and parts[3] in _ARTIFACT_NAMES
    )
    if not valid_shape or (required_name is not None and parts[-1] != required_name):
        raise ValueError("artifact target must be inside an ESKF run directory")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
