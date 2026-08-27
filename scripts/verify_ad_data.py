#!/usr/bin/env python3
"""Validate the curated, versioned assets below ``ad_data``."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
from pathlib import Path
import re
import subprocess
import sys

import yaml


CHECKSUM_PATTERN = re.compile(r"^([0-9a-f]{64})  ([^\n]+)$")
IPV4_PATTERN = re.compile(r"(?<![0-9])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9])")
MACHINE_PATH_PATTERN = re.compile(r"(?:/home/|/Users/)[^/\s\"']+/")
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1\n"
FORBIDDEN_DIRECTORY_NAMES = {
    "backups",
    "experiments",
    "logs",
    "SensorData",
    "workers",
    "__pycache__",
}
FORBIDDEN_SUFFIXES = {
    ".bag",
    ".db",
    ".db3",
    ".jsonl",
    ".log",
    ".mcap",
    ".pcap",
    ".pid",
    ".pyc",
    ".sqlite",
    ".sqlite3",
    ".wal",
}
FORBIDDEN_FILE_NAMES = {
    "events.csv",
    "raw_samples.csv",
    "run_state.json",
    "trials.csv",
}
LOCAL_ONLY_DIRECTORY_NAMES = {"local_archive"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative_path(raw: str) -> Path | None:
    path = Path(raw)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        return None
    return path


def _load_checksums(path: Path, errors: list[str]) -> dict[Path, str]:
    checksums: dict[Path, str] = {}
    if not path.is_file():
        errors.append(f"missing checksum file: {path}")
        return checksums

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        match = CHECKSUM_PATTERN.fullmatch(line)
        if match is None:
            errors.append(f"invalid SHA256SUMS line {line_number}: {line!r}")
            continue
        digest, raw_relative = match.groups()
        relative = _safe_relative_path(raw_relative)
        if relative is None:
            errors.append(f"unsafe checksum path: {raw_relative!r}")
            continue
        if relative in checksums:
            errors.append(f"duplicate checksum path: {relative.as_posix()}")
            continue
        checksums[relative] = digest
    return checksums


def _load_manifest(path: Path, errors: list[str]) -> dict:
    if not path.is_file():
        errors.append(f"missing manifest: {path}")
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        errors.append(f"invalid manifest YAML: {exc}")
        return {}
    if not isinstance(loaded, dict):
        errors.append("manifest must be a YAML mapping")
        return {}
    if loaded.get("schema_version") != 1:
        errors.append("manifest schema_version must be 1")
    collections = loaded.get("collections")
    if not isinstance(collections, list) or not collections:
        errors.append("manifest collections must be a non-empty list")
    else:
        required = {"id", "path", "kind", "source", "license"}
        for index, collection in enumerate(collections):
            if not isinstance(collection, dict) or not required.issubset(collection):
                errors.append(
                    f"manifest collection {index} must contain {sorted(required)}"
                )
    if not isinstance(loaded.get("lfs_paths", []), list):
        errors.append("manifest lfs_paths must be a list")
    return loaded


def _validate_structured_files(data_root: Path, files: set[Path], errors: list[str]):
    for relative in sorted(files):
        path = data_root / relative
        if path.suffix.lower() == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                errors.append(f"invalid JSON {relative.as_posix()}: {exc}")
        elif path.suffix.lower() in {".yaml", ".yml"}:
            try:
                yaml.safe_load(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, yaml.YAMLError) as exc:
                errors.append(f"invalid YAML {relative.as_posix()}: {exc}")


def _validate_network_templates(
    data_root: Path, files: set[Path], errors: list[str]
):
    for relative in sorted(files):
        if relative.parts[:3] != ("morai", "SaveFile", "Network"):
            continue
        path = data_root / relative
        if path.suffix.lower() != ".json":
            continue
        text = path.read_text(encoding="utf-8")
        for raw_address in IPV4_PATTERN.findall(text):
            try:
                address = ipaddress.ip_address(raw_address)
            except ValueError:
                errors.append(
                    f"invalid IPv4 address in network template {relative}: {raw_address}"
                )
                continue
            if not (address.is_loopback or address.is_unspecified):
                errors.append(
                    "machine-specific IPv4 address in network template "
                    f"{relative.as_posix()}: {raw_address}"
                )


def _validate_machine_specific_paths(
    data_root: Path, files: set[Path], errors: list[str]
):
    for relative in sorted(files):
        path = data_root / relative
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeError, OSError):
            continue
        if MACHINE_PATH_PATTERN.search(text):
            errors.append(
                f"machine-specific absolute path in curated data: {relative.as_posix()}"
            )


def _validate_lfs(
    repository_root: Path,
    data_root: Path,
    manifest: dict,
    errors: list[str],
):
    raw_paths = manifest.get("lfs_paths", [])
    if not isinstance(raw_paths, list):
        return
    for raw_relative in raw_paths:
        if not isinstance(raw_relative, str):
            errors.append(f"manifest LFS path must be a string: {raw_relative!r}")
            continue
        relative = _safe_relative_path(raw_relative)
        if relative is None:
            errors.append(f"unsafe LFS path: {raw_relative!r}")
            continue
        path = data_root / relative
        if not path.is_file():
            errors.append(f"missing LFS file: {relative.as_posix()}")
            continue
        with path.open("rb") as stream:
            if stream.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX:
                errors.append(f"LFS payload is missing: {relative.as_posix()}")

        if (repository_root / ".git").exists():
            tracked_path = (Path("ad_data") / relative).as_posix()
            result = subprocess.run(
                ["git", "check-attr", "filter", "--", tracked_path],
                cwd=repository_root,
                text=True,
                capture_output=True,
                check=False,
            )
            if not result.stdout.rstrip().endswith(": lfs"):
                errors.append(f"LFS attribute is missing: {tracked_path}")


def verify_repository(repository_root: Path) -> tuple[int, list[str]]:
    data_root = repository_root / "ad_data"
    errors: list[str] = []
    if not data_root.is_dir():
        return 0, [f"missing curated data directory: {data_root}"]

    manifest = _load_manifest(data_root / "manifest.yaml", errors)
    checksums = _load_checksums(data_root / "SHA256SUMS", errors)
    discovered = set()
    for path in data_root.rglob("*"):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        relative = path.relative_to(data_root)
        if relative.parts[0] in LOCAL_ONLY_DIRECTORY_NAMES:
            continue
        discovered.add(relative)
    listed = set(checksums)
    for relative in sorted(discovered - listed):
        errors.append(f"file missing from SHA256SUMS: {relative.as_posix()}")
    for relative in sorted(listed - discovered):
        errors.append(f"checksum references missing file: {relative.as_posix()}")

    for relative in sorted(discovered):
        path = data_root / relative
        if (
            any(part in FORBIDDEN_DIRECTORY_NAMES for part in relative.parts)
            or path.suffix.lower() in FORBIDDEN_SUFFIXES
            or path.name in FORBIDDEN_FILE_NAMES
        ):
            errors.append(f"forbidden runtime data: {relative.as_posix()}")
        expected = checksums.get(relative)
        if expected is not None:
            actual = _sha256(path)
            if actual != expected:
                errors.append(
                    f"checksum mismatch: {relative.as_posix()} "
                    f"expected {expected}, got {actual}"
                )

    collections = manifest.get("collections", [])
    if isinstance(collections, list):
        for collection in collections:
            if not isinstance(collection, dict) or not isinstance(
                collection.get("path"), str
            ):
                continue
            relative = _safe_relative_path(collection["path"])
            if relative is None:
                errors.append(f"unsafe collection path: {collection['path']!r}")
            elif not (data_root / relative).exists():
                errors.append(f"missing collection path: {relative.as_posix()}")

    _validate_structured_files(data_root, discovered, errors)
    _validate_network_templates(data_root, discovered, errors)
    _validate_machine_specific_paths(data_root, discovered, errors)
    _validate_lfs(repository_root, data_root, manifest, errors)
    return len(discovered), errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root containing ad_data",
    )
    arguments = parser.parse_args()
    file_count, errors = verify_repository(arguments.root.resolve())
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"verified {file_count} curated data files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
