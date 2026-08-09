"""Canonical launch and provenance contract for MORAI perception validation."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import subprocess
from typing import Callable, Mapping, Sequence
import xml.etree.ElementTree as ElementTree

import yaml


_COMMIT = re.compile(r"[0-9a-f]{40}")
_DEPENDENCIES = ("autoware_universe", "muSSP")


def canonical_launch_arguments(
    perception_share: Path, description_share: Path
) -> dict[str, str]:
    """Return the complete fixed launch arguments used by launch and recorder."""
    perception = Path(perception_share)
    description = Path(description_share)
    return {
        "platform_profile": "morai",
        "composition_config": str(
            perception / "config" / "lidar_perception_morai_classical.yaml"
        ),
        "start_ground_segmentation": "true",
        "deskew_enabled": "false",
        "deskew_mode": "3d",
        "self_crop_enabled": "true",
        "patchwork_leveling_enabled": "false",
        "finite_filter_enabled": "false",
        "densifier_enabled": "false",
        "point_layout_adapter_enabled": "false",
        "ground_config": str(
            perception
            / "config"
            / "preprocessing"
            / "ground_segmentation.yaml"
        ),
        "sensor_config": str(
            description / "config" / "sensor_mounts.yaml"
        ),
        "sensor_profile": "",
    }


def canonical_config_paths(
    perception_share: Path, description_share: Path
) -> dict[str, Path]:
    """Return every configuration file effective in the fixed graph."""
    perception = Path(perception_share)
    description = Path(description_share)
    return {
        "adaptive_euclidean_cluster": (
            perception
            / "config"
            / "clustering"
            / "adaptive_euclidean_cluster.yaml"
        ),
        "autoware_lock": (
            perception / "config" / "autoware_perception.lock.yaml"
        ),
        "composition": (
            perception / "config" / "lidar_perception_morai_classical.yaml"
        ),
        "ground_segmentation": (
            perception
            / "config"
            / "preprocessing"
            / "ground_segmentation.yaml"
        ),
        "prediction": (
            perception / "config" / "tracking" / "prediction.yaml"
        ),
        "self_crop": (
            perception / "config" / "preprocessing" / "self_crop.yaml"
        ),
        "sensor_mounts": description / "config" / "sensor_mounts.yaml",
        "tracker": perception / "config" / "tracking" / "autoware.yaml",
        "vehicle_parameters": (
            description / "config" / "vehicle_parameters.yaml"
        ),
    }


def file_record(path: Path, name: str) -> dict[str, str]:
    source = Path(path)
    try:
        resolved = source.resolve(strict=True)
        payload = resolved.read_bytes()
    except OSError as error:
        raise ValueError(f"{name} is unreadable: {source}") from error
    if not resolved.is_file():
        raise ValueError(f"{name} must be a regular file")
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def config_records(paths: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    if set(paths) != set(canonical_config_paths(Path("p"), Path("d"))):
        raise ValueError("config_paths must contain every canonical config")
    return {
        name: file_record(paths[name], f"config {name}")
        for name in sorted(paths)
    }


def load_dependency_pins(path: Path) -> dict[str, str]:
    try:
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        repositories = document["repositories"]
    except (OSError, KeyError, TypeError, yaml.YAMLError) as error:
        raise ValueError(f"invalid dependencies.repos: {error}") from error
    pins = {}
    for name in _DEPENDENCIES:
        try:
            revision = repositories[name]["version"]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"dependencies.repos is missing {name}"
            ) from error
        if not isinstance(revision, str) or _COMMIT.fullmatch(revision) is None:
            raise ValueError(f"dependencies.repos {name} pin is not a commit")
        pins[name] = revision
    return pins


def verify_dependency_sources(
    workspace_sources: Path,
    pins: Mapping[str, str],
    *,
    command_runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> dict[str, str]:
    if set(pins) != set(_DEPENDENCIES):
        raise ValueError("dependency pins must contain Autoware and muSSP")
    verified = {}
    for name in _DEPENDENCIES:
        source = Path(workspace_sources) / name
        revision_result = command_runner(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        revision = revision_result.stdout.strip()
        if revision != pins[name]:
            raise ValueError(
                f"{name} revision drift: expected {pins[name]}, got {revision}"
            )
        status_result = command_runner(
            [
                "git",
                "-C",
                str(source),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if status_result.stdout.strip():
            raise ValueError(f"{name} is dirty")
        verified[name] = revision
    return verified


def _installed_package_record(
    package: str,
    prefixes: Sequence[Path],
    *,
    expected_version: str,
) -> dict[str, str]:
    for prefix in prefixes:
        package_xml = Path(prefix) / "share" / package / "package.xml"
        if not package_xml.is_file():
            continue
        try:
            root = ElementTree.parse(package_xml).getroot()
            version = root.findtext("version")
        except (ElementTree.ParseError, OSError) as error:
            raise ValueError(f"installed {package} metadata is invalid") from error
        if version != expected_version:
            raise ValueError(
                f"installed {package} version drift: "
                f"expected {expected_version}, got {version}"
            )
        return {
            "package": package,
            "prefix": str(Path(prefix).resolve()),
            "version": version,
        }
    raise ValueError(f"installed package is unavailable: {package}")


def collect_runtime_provenance(
    perception_share: Path,
    *,
    prefixes: Sequence[Path] | None = None,
) -> dict[str, object]:
    """Verify installed tracker artifacts and return exact selected prefixes."""
    from ad_lidar_perception.autoware_provenance import (
        load_lock,
        verify_selection,
    )
    from ad_lidar_perception.selection import load_selection

    selected_prefixes = tuple(prefixes or ())
    if not selected_prefixes:
        selected_prefixes = tuple(
            Path(item)
            for item in os.environ.get("AMENT_PREFIX_PATH", "").split(
                os.pathsep
            )
            if item
        )
    configs = canonical_config_paths(perception_share, Path("unused"))
    lock = load_lock(configs["autoware_lock"])
    selection = load_selection(configs["composition"])
    verified = verify_selection(
        selection,
        lock_path=configs["autoware_lock"],
        prefixes=selected_prefixes,
    )
    if verified.tracker is None:
        raise ValueError("Autoware tracker was not selected")
    return {
        "checker": {
            "message": (
                "Autoware perception selection verified "
                "(tracker=autoware)."
            ),
            "result": "passed",
        },
        "mussp": _installed_package_record(
            "mussp", selected_prefixes, expected_version="0.1.0"
        ),
        "tracker": _installed_package_record(
            "autoware_multi_object_tracker",
            selected_prefixes,
            expected_version=lock["tracker"]["version"],
        ),
    }
