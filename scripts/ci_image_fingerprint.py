#!/usr/bin/env python3
"""Compute the content key for the HEVEN CI dependency image."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys


FORMAT_VERSION = b"heven-ci-dependencies-v1\0"
REQUIRED_FILES = (
    Path("docker/ci/Dockerfile"),
    Path("docker/ci/rebuild-revision"),
    Path("dependencies.repos"),
    Path("scripts/apply_dependency_patches.sh"),
)
EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".venv",
    "build",
    "install",
    "log",
    "worktrees",
}


def is_excluded(relative_path: Path) -> bool:
    return any(part in EXCLUDED_DIRECTORY_NAMES for part in relative_path.parts)


def dependency_inputs(root: Path) -> list[Path]:
    missing = [path for path in REQUIRED_FILES if not (root / path).is_file()]
    if missing:
        raise ValueError(
            "missing dependency fingerprint input: "
            + ", ".join(path.as_posix() for path in missing)
        )

    package_manifests = [
        path.relative_to(root)
        for path in root.rglob("package.xml")
        if not is_excluded(path.relative_to(root))
    ]
    if not package_manifests:
        raise ValueError("missing dependency fingerprint input: **/package.xml")

    dependency_patches = [
        path.relative_to(root)
        for path in (root / "patches").rglob("*.patch")
        if path.is_file()
    ]
    if not dependency_patches:
        raise ValueError("missing dependency fingerprint input: patches/**/*.patch")

    return sorted(set(REQUIRED_FILES) | set(package_manifests) | set(dependency_patches))


def calculate_fingerprint(root: Path) -> tuple[str, list[Path]]:
    inputs = dependency_inputs(root)
    digest = hashlib.sha256(FORMAT_VERSION)
    for relative_path in inputs:
        contents = (root / relative_path).read_bytes()
        encoded_path = relative_path.as_posix().encode("utf-8")
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest(), inputs


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--list-inputs",
        action="store_true",
        help="write the normalized input paths to stderr",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    root = arguments.root.resolve()
    try:
        fingerprint, inputs = calculate_fingerprint(root)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2

    if arguments.list_inputs:
        for path in inputs:
            print(path.as_posix(), file=sys.stderr)
    print(fingerprint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
