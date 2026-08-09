"""Filesystem-only provenance checks for optional Autoware perception."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import hmac
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Any, Iterable, Mapping, Sequence
import xml.etree.ElementTree as ElementTree

from .selection import (
    PerceptionSelection,
    SelectionError,
    load_selection,
    load_yaml_payload,
)


_LOCK_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "autoware",
        "weights_policy",
        "messages",
        "detectors",
        "tracker",
    }
)
_AUTOWARE_KEYS = frozenset(
    {
        "meta_release",
        "universe",
        "autoware_msgs",
        "manifest_url",
        "artifact_role_url",
    }
)
_WEIGHTS_KEYS = frozenset(
    {
        "license_status",
        "redistribution_allowed_by_project",
        "required_ack_env",
    }
)
_MESSAGES_KEYS = frozenset({"package", "version"})
_DETECTOR_KEYS = frozenset(
    {
        "package",
        "version",
        "executable",
        "launch",
        "installed_files",
        "model_dir",
        "model_name",
        "ml_package",
        "class_remapper",
        "artifacts",
        "engines",
    }
)
_TRACKER_KEYS = frozenset(
    {
        "package",
        "version",
        "executable",
        "launch",
        "installed_files",
        "prediction_package",
        "prediction_executable",
    }
)
_ARTIFACT_KEYS = frozenset({"path", "sha256"})
_UPSTREAM_LEAF_KEYS = frozenset({"path", "sha256"})
_DETECTOR_BACKENDS = frozenset(
    {
        "centerpoint_tiny",
        "centerpoint",
        "transfusion",
        "bevfusion_lidar",
    }
)
_ARTIFACT_ROLES = {
    "centerpoint_tiny": frozenset(
        {"encoder_onnx", "head_onnx", "ml_package", "class_remapper"}
    ),
    "centerpoint": frozenset(
        {"encoder_onnx", "head_onnx", "ml_package", "class_remapper"}
    ),
    "transfusion": frozenset({"onnx", "ml_package", "class_remapper"}),
    "bevfusion_lidar": frozenset(
        {"onnx", "ml_package", "class_remapper"}
    ),
}
_PACKAGE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")
_PINNED_AUTOWARE = {
    "meta_release": "1.8.0",
    "universe": "0.51.0",
    "autoware_msgs": "1.13.0",
    "manifest_url": (
        "https://github.com/autowarefoundation/autoware/blob/1.8.0/"
        "repositories/autoware.repos"
    ),
    "artifact_role_url": (
        "https://github.com/autowarefoundation/autoware/blob/1.8.0/"
        "ansible/roles/artifacts/tasks/main.yaml"
    ),
}
_PINNED_WEIGHT_POLICY = {
    "license_status": "unresolved_upstream_artifact_license",
    "redistribution_allowed_by_project": False,
    "required_ack_env": "AD_AUTOWARE_MODEL_LICENSE_REVIEWED",
}
_PINNED_LOCK_SHA256 = (
    "b8d94641c0af0dd9b48a61a738ee7c6ab5a83cca3b6d38a0651d4ec00d20e42e"
)


class LockError(ValueError):
    """Raised when the checked-in provenance lock is malformed."""


class VerificationError(RuntimeError):
    """Raised once with every deterministic provenance failure."""

    def __init__(self, issues: Iterable[str]):
        ordered = tuple(sorted(set(issues)))
        self.issues = ordered
        body = "\n".join(f"- {issue}" for issue in ordered)
        super().__init__(
            "Autoware perception verification failed:"
            + (f"\n{body}" if body else "")
        )


@dataclass(frozen=True)
class DetectorRuntime:
    backend: str
    package: str
    executable: str
    launch_path: Path
    model_path: Path
    ml_package_path: Path
    class_remapper_path: Path
    build_only: bool


@dataclass(frozen=True)
class TrackerRuntime:
    package: str
    executable: str
    launch_path: Path


@dataclass(frozen=True)
class VerifiedSelection:
    detector: DetectorRuntime | None
    tracker: TrackerRuntime | None


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise LockError(f"{name} must be a mapping")
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], name: str
) -> None:
    actual = set(value)
    missing = set(expected) - actual
    if missing:
        raise LockError(
            f"{name} is missing keys: {', '.join(sorted(map(str, missing)))}"
        )
    unknown = actual - set(expected)
    if unknown:
        raise LockError(
            f"{name} has unknown keys: {', '.join(sorted(map(str, unknown)))}"
        )


def _string(value: Any, name: str) -> str:
    if type(value) is not str:
        raise LockError(f"{name} must be a string")
    if not value:
        raise LockError(f"{name} must not be empty")
    return value


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise LockError(f"{name} must be a boolean")
    return value


def _package(value: Any, name: str) -> str:
    package_name = _string(value, name)
    if _PACKAGE_NAME.fullmatch(package_name) is None:
        raise LockError(f"{name} must be a valid ROS package name")
    return package_name


def _safe_relative(value: Any, name: str, *, basename: bool = False) -> str:
    text = _string(value, name)
    parts = text.split("/")
    if (
        "\x00" in text
        or "\\" in text
        or text.startswith("/")
        or _WINDOWS_DRIVE.match(text) is not None
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise LockError(f"{name} must be a safe relative POSIX path")
    parsed = PurePosixPath(text)
    if parsed.is_absolute() or (basename and len(parsed.parts) != 1):
        raise LockError(f"{name} must be a safe relative POSIX path")
    return text


def _string_list(
    value: Any,
    name: str,
    *,
    nonempty: bool = True,
) -> list[str]:
    if type(value) is not list:
        raise LockError(f"{name} must be a list")
    if nonempty and not value:
        raise LockError(f"{name} must not be empty")
    result = []
    for index, item in enumerate(value):
        result.append(_safe_relative(item, f"{name}[{index}]"))
    if len(set(result)) != len(result):
        raise LockError(f"{name} must not contain duplicates")
    return result


def _sha256(value: Any, name: str) -> str:
    digest = _string(value, name)
    if _SHA256.fullmatch(digest) is None:
        raise LockError(
            f"{name} must be 64 lowercase hexadecimal characters"
        )
    return digest


def _upstream_leaf(value: Any, name: str) -> dict[str, Any]:
    record = _mapping(value, name)
    _exact_keys(record, _UPSTREAM_LEAF_KEYS, name)
    _safe_relative(record["path"], f"{name}.path")
    _sha256(record["sha256"], f"{name}.sha256")
    return record


def _upstream_leaves(value: Any, name: str) -> list[dict[str, Any]]:
    if type(value) is not list:
        raise LockError(f"{name} must be a list")
    if not value:
        raise LockError(f"{name} must not be empty")
    records = [
        _upstream_leaf(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    ]
    paths = [record["path"] for record in records]
    if len(set(paths)) != len(paths):
        raise LockError(f"{name} must not contain duplicate paths")
    return records


def load_lock(path: Path | str) -> dict[str, Any]:
    """Load and structurally validate a pinned Autoware provenance lock."""
    try:
        payload = Path(path).read_bytes()
    except OSError as error:
        raise LockError(f"{path}: {error.strerror}") from error
    actual_lock_sha256 = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(actual_lock_sha256, _PINNED_LOCK_SHA256):
        raise LockError(
            "lock content does not match the trusted checked-in lock"
        )
    try:
        document = load_yaml_payload(payload)
    except SelectionError as error:
        raise LockError(str(error)) from error

    root = _mapping(document, "root")
    _exact_keys(root, _LOCK_ROOT_KEYS, "root")
    schema_version = root["schema_version"]
    if type(schema_version) is not int:
        raise LockError("schema_version must be an integer")
    if schema_version != 1:
        raise LockError("schema_version must equal 1")

    autoware = _mapping(root["autoware"], "autoware")
    weights = _mapping(root["weights_policy"], "weights_policy")
    messages = _mapping(root["messages"], "messages")
    detectors = _mapping(root["detectors"], "detectors")
    tracker = _mapping(root["tracker"], "tracker")
    _exact_keys(autoware, _AUTOWARE_KEYS, "autoware")
    _exact_keys(weights, _WEIGHTS_KEYS, "weights_policy")
    _exact_keys(messages, _MESSAGES_KEYS, "messages")
    _exact_keys(detectors, _DETECTOR_BACKENDS, "detectors")
    _exact_keys(tracker, _TRACKER_KEYS, "tracker")

    for key in _AUTOWARE_KEYS:
        _string(autoware[key], f"autoware.{key}")
        if autoware[key] != _PINNED_AUTOWARE[key]:
            raise LockError(
                f"autoware.{key} must equal {_PINNED_AUTOWARE[key]}"
            )
    _string(weights["license_status"], "weights_policy.license_status")
    redistribution = _boolean(
        weights["redistribution_allowed_by_project"],
        "weights_policy.redistribution_allowed_by_project",
    )
    if redistribution:
        raise LockError(
            "weights_policy.redistribution_allowed_by_project must be false"
        )
    _string(weights["required_ack_env"], "weights_policy.required_ack_env")
    for key, expected in _PINNED_WEIGHT_POLICY.items():
        if weights[key] != expected:
            raise LockError(
                f"weights_policy.{key} must equal {expected}"
            )
    messages_package = _package(messages["package"], "messages.package")
    if messages_package != "autoware_perception_msgs":
        raise LockError(
            "messages.package must equal autoware_perception_msgs"
        )
    _string(messages["version"], "messages.version")
    if messages["version"] != autoware["autoware_msgs"]:
        raise LockError(
            "messages.version must equal autoware.autoware_msgs"
        )

    for backend in sorted(_DETECTOR_BACKENDS):
        name = f"detectors.{backend}"
        detector = _mapping(detectors[backend], name)
        _exact_keys(detector, _DETECTOR_KEYS, name)
        _package(detector["package"], f"{name}.package")
        _string(detector["version"], f"{name}.version")
        if detector["version"] != autoware["universe"]:
            raise LockError(
                f"{name}.version must equal autoware.universe"
            )
        _safe_relative(
            detector["executable"], f"{name}.executable", basename=True
        )
        _upstream_leaf(detector["launch"], f"{name}.launch")
        _upstream_leaves(
            detector["installed_files"], f"{name}.installed_files"
        )
        _safe_relative(
            detector["model_dir"], f"{name}.model_dir", basename=True
        )
        model_name = _safe_relative(
            detector["model_name"], f"{name}.model_name", basename=True
        )
        if model_name != backend:
            raise LockError(f"{name}.model_name must equal {backend}")
        ml_package = _safe_relative(
            detector["ml_package"], f"{name}.ml_package"
        )
        class_remapper = _safe_relative(
            detector["class_remapper"], f"{name}.class_remapper"
        )
        engines = _string_list(detector["engines"], f"{name}.engines")
        if not engines:
            raise LockError(f"{name}.engines must not be empty")

        artifacts = _mapping(detector["artifacts"], f"{name}.artifacts")
        _exact_keys(
            artifacts,
            _ARTIFACT_ROLES[backend],
            f"{name}.artifacts",
        )
        for role in sorted(_ARTIFACT_ROLES[backend]):
            artifact_name = f"{name}.artifacts.{role}"
            artifact = _mapping(artifacts[role], artifact_name)
            _exact_keys(artifact, _ARTIFACT_KEYS, artifact_name)
            artifact_path = _safe_relative(
                artifact["path"], f"{artifact_name}.path"
            )
            _sha256(artifact["sha256"], f"{artifact_name}.sha256")
            if role == "ml_package" and artifact_path != ml_package:
                raise LockError(
                    f"{artifact_name}.path must equal {name}.ml_package"
                )
            if role == "class_remapper" and artifact_path != class_remapper:
                raise LockError(
                    f"{artifact_name}.path must equal {name}.class_remapper"
                )

    _package(tracker["package"], "tracker.package")
    _string(tracker["version"], "tracker.version")
    if tracker["version"] != autoware["universe"]:
        raise LockError("tracker.version must equal autoware.universe")
    _safe_relative(
        tracker["executable"], "tracker.executable", basename=True
    )
    _upstream_leaf(tracker["launch"], "tracker.launch")
    _upstream_leaves(tracker["installed_files"], "tracker.installed_files")
    _package(tracker["prediction_package"], "tracker.prediction_package")
    _safe_relative(
        tracker["prediction_executable"],
        "tracker.prediction_executable",
        basename=True,
    )
    return root


def resolve_data_root(
    explicit: Path | str | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the explicit or environment-provided canonical data root."""
    environment = os.environ if environ is None else environ
    if explicit is not None and str(explicit):
        selected = Path(explicit)
    elif environment.get("AD_DATA_DIR"):
        selected = Path(environment["AD_DATA_DIR"])
    else:
        raise ValueError("set data_root or AD_DATA_DIR")
    return selected.expanduser().resolve()


def _prefixes(
    supplied: Sequence[Path | str] | None,
    environ: Mapping[str, str],
) -> tuple[Path, ...]:
    if supplied is not None:
        return tuple(Path(prefix) for prefix in supplied)
    value = environ.get("AMENT_PREFIX_PATH", "")
    return tuple(
        Path(prefix) for prefix in value.split(os.pathsep) if prefix
    )


def _find_package_prefix(
    package_name: str, prefixes: Sequence[Path]
) -> Path | None:
    for prefix in prefixes:
        package_xml = prefix / "share" / package_name / "package.xml"
        if os.path.lexists(package_xml):
            return prefix
    return None


def _secure_regular_file(
    root: Path,
    relative: str | PurePosixPath,
    label: str,
    issues: list[str],
    *,
    executable: bool = False,
    allow_final_symlink: bool = False,
) -> Path | None:
    relative_path = PurePosixPath(str(relative))
    current = root
    try:
        root_stat = os.lstat(root)
    except OSError as error:
        issues.append(f"{label}: root {root} is unavailable: {error.strerror}")
        return None
    if stat.S_ISLNK(root_stat.st_mode):
        issues.append(f"{label}: root {root} is a symlink")
        return None
    if not stat.S_ISDIR(root_stat.st_mode):
        issues.append(f"{label}: root {root} is not a directory")
        return None

    for index, part in enumerate(relative_path.parts):
        current = current / part
        try:
            status = os.lstat(current)
        except OSError as error:
            issues.append(
                f"{label}: {current} is unavailable: {error.strerror}"
            )
            return None
        final = index == len(relative_path.parts) - 1
        if stat.S_ISLNK(status.st_mode):
            if final and allow_final_symlink:
                try:
                    target_status = os.stat(current)
                except OSError as error:
                    issues.append(
                        f"{label}: {current} is a broken symlink:"
                        f" {error.strerror}"
                    )
                    return None
                if (
                    not stat.S_ISREG(target_status.st_mode)
                    or target_status.st_size <= 0
                ):
                    issues.append(
                        f"{label}: {current} symlink target must be a"
                        " non-empty regular file"
                    )
                    return None
                if executable and target_status.st_mode & 0o111 == 0:
                    issues.append(
                        f"{label}: {current} symlink target is not executable"
                    )
                    return None
                return current
            issues.append(f"{label}: {current} is a symlink")
            return None
        if not final:
            if not stat.S_ISDIR(status.st_mode):
                issues.append(
                    f"{label}: intermediate {current} is not a directory"
                )
                return None
            continue
        if not stat.S_ISREG(status.st_mode) or status.st_size <= 0:
            issues.append(
                f"{label}: {current} must be a non-empty regular file"
            )
            return None
        if executable and status.st_mode & 0o111 == 0:
            issues.append(f"{label}: {current} is not executable")
            return None
    return current


def _read_regular_bytes(
    path: Path,
    label: str,
    issues: list[str],
    *,
    allow_final_symlink: bool = False,
) -> bytes | None:
    flags = os.O_RDONLY
    if not allow_final_symlink and hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        issues.append(f"{label}: could not open {path}: {error.strerror}")
        return None
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode) or status.st_size <= 0:
            issues.append(f"{label}: {path} must be a non-empty regular file")
            return None
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(descriptor)


def _sha256_regular(
    path: Path,
    label: str,
    issues: list[str],
    *,
    allow_final_symlink: bool = False,
) -> str | None:
    flags = os.O_RDONLY
    if not allow_final_symlink and hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        issues.append(f"{label}: could not open {path}: {error.strerror}")
        return None
    digest = hashlib.sha256()
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode) or status.st_size <= 0:
            issues.append(f"{label}: {path} must be a non-empty regular file")
            return None
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _package_version(
    prefix: Path,
    package_name: str,
    expected: str,
    issues: list[str],
) -> None:
    label = f"{package_name} package.xml"
    package_xml = _secure_regular_file(
        prefix,
        PurePosixPath("share") / package_name / "package.xml",
        label,
        issues,
        allow_final_symlink=True,
    )
    if package_xml is None:
        return
    payload = _read_regular_bytes(
        package_xml,
        label,
        issues,
        allow_final_symlink=True,
    )
    if payload is None:
        return
    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError as error:
        issues.append(f"{label}: invalid XML: {error}")
        return
    installed_name = root.findtext("name")
    version = root.findtext("version")
    if installed_name != package_name:
        issues.append(
            f"{package_name} package.xml name {installed_name!r}"
            f" != {package_name!r}"
        )
    if version != expected:
        issues.append(f"{package_name} version {version} != {expected}")


def _verify_package(
    contract: Mapping[str, Any],
    prefixes: Sequence[Path],
    issues: list[str],
    *,
    require_executable: bool = True,
    require_leaves: bool = True,
) -> tuple[Path, Path | None] | None:
    package_name = contract["package"]
    prefix = _find_package_prefix(package_name, prefixes)
    if prefix is None:
        issues.append(
            f"package {package_name} was not found in AMENT_PREFIX_PATH"
        )
        return None
    _package_version(prefix, package_name, contract["version"], issues)

    if require_executable:
        _secure_regular_file(
            prefix,
            PurePosixPath("lib") / package_name / contract["executable"],
            f"{package_name} executable {contract['executable']}",
            issues,
            executable=True,
            allow_final_symlink=True,
        )

    launch_path = None
    if require_leaves:
        records = [contract["launch"], *contract["installed_files"]]
        for index, record in enumerate(records):
            relative = record["path"]
            label = f"{package_name} installed leaf {relative}"
            installed_path = _secure_regular_file(
                prefix,
                PurePosixPath("share") / package_name / relative,
                label,
                issues,
                allow_final_symlink=True,
            )
            if installed_path is None:
                continue
            actual = _sha256_regular(
                installed_path,
                label,
                issues,
                allow_final_symlink=True,
            )
            if actual is not None and not hmac.compare_digest(
                actual, record["sha256"]
            ):
                issues.append(
                    f"{label} SHA-256 mismatch:"
                    f" expected {record['sha256']}, got {actual}"
                )
            if index == 0:
                launch_path = installed_path
    return prefix, launch_path


def _default_lock_path(prefixes: Sequence[Path]) -> Path | None:
    prefix = _find_package_prefix("ad_lidar_perception", prefixes)
    if prefix is None:
        return None
    return (
        prefix
        / "share"
        / "ad_lidar_perception"
        / "config"
        / "autoware_perception.lock.yaml"
    )


def verify_selection(
    selection: PerceptionSelection,
    *,
    lock_path: Path | str | None = None,
    prefixes: Sequence[Path | str] | None = None,
    data_root: Path | str | None = None,
    environ: Mapping[str, str] | None = None,
) -> VerifiedSelection:
    """Verify only the packages and artifacts selected for this process."""
    if (
        selection.detector.backend == "none"
        and selection.tracker.backend == "none"
    ):
        return VerifiedSelection(detector=None, tracker=None)

    environment = os.environ if environ is None else environ
    search_prefixes = _prefixes(prefixes, environment)
    selected_lock = (
        Path(lock_path)
        if lock_path is not None
        else _default_lock_path(search_prefixes)
    )
    if selected_lock is None:
        raise VerificationError(
            ["could not locate the installed Autoware perception lock file"]
        )
    try:
        lock = load_lock(selected_lock)
    except LockError as error:
        raise VerificationError([f"lock file invalid: {error}"]) from error

    issues: list[str] = []
    detector_runtime = None
    tracker_runtime = None
    detector_backend = selection.detector.backend

    if detector_backend == "euclidean_cluster":
        # The model-free adaptive clusterer is built inside this package.
        # Only the shared Autoware message ABI and downstream tracker need
        # external provenance verification here.
        messages = lock["messages"]
        _verify_package(
            messages,
            search_prefixes,
            issues,
            require_executable=False,
            require_leaves=False,
        )
    elif detector_backend == "none":
        issues.append("tracker requires a non-none detector")
    else:
        acknowledgement = lock["weights_policy"]["required_ack_env"]
        if environment.get(acknowledgement) != "1":
            issues.append(f"{acknowledgement} must equal exactly 1")

        detector = lock["detectors"][detector_backend]
        detector_install = _verify_package(
            detector, search_prefixes, issues
        )
        messages = lock["messages"]
        _verify_package(
            messages,
            search_prefixes,
            issues,
            require_executable=False,
            require_leaves=False,
        )

        resolved_data_root = resolve_data_root(
            data_root, environ=environment
        )
        model_relative = (
            PurePosixPath(selection.detector.model_subdir.as_posix())
            / detector["model_dir"]
        )
        model_path = resolved_data_root.joinpath(*model_relative.parts)
        for role in sorted(detector["artifacts"]):
            artifact = detector["artifacts"][role]
            relative = model_relative / artifact["path"]
            artifact_path = _secure_regular_file(
                resolved_data_root,
                relative,
                f"{detector_backend} artifact {role}",
                issues,
            )
            if artifact_path is None:
                continue
            actual = _sha256_regular(
                artifact_path,
                f"{detector_backend} artifact {role}",
                issues,
            )
            if actual is not None and not hmac.compare_digest(
                actual, artifact["sha256"]
            ):
                issues.append(
                    f"{detector_backend} artifact {role} SHA-256 mismatch:"
                    f" expected {artifact['sha256']}, got {actual}"
                )

        for engine in detector["engines"]:
            engine_relative = model_relative / engine
            if (
                selection.detector.build_only
                and not os.path.lexists(
                    resolved_data_root.joinpath(*engine_relative.parts)
                )
            ):
                continue
            _secure_regular_file(
                resolved_data_root,
                engine_relative,
                f"{detector_backend} engine {engine}",
                issues,
            )

        if detector_install is not None:
            _, launch_path = detector_install
            if launch_path is not None:
                detector_runtime = DetectorRuntime(
                    backend=detector_backend,
                    package=detector["package"],
                    executable=detector["executable"],
                    launch_path=launch_path,
                    model_path=model_path,
                    ml_package_path=(
                        model_path / detector["ml_package"]
                    ),
                    class_remapper_path=(
                        model_path / detector["class_remapper"]
                    ),
                    build_only=selection.detector.build_only,
                )

    if selection.tracker.backend == "autoware":
        tracker = lock["tracker"]
        tracker_install = _verify_package(
            tracker, search_prefixes, issues
        )
        prediction_contract = {
            "package": tracker["prediction_package"],
            "version": None,
            "executable": tracker["prediction_executable"],
        }
        prediction_prefix = _find_package_prefix(
            prediction_contract["package"], search_prefixes
        )
        if prediction_prefix is None:
            issues.append(
                f"package {prediction_contract['package']} was not found in"
                " AMENT_PREFIX_PATH"
            )
        else:
            _secure_regular_file(
                prediction_prefix,
                PurePosixPath("lib")
                / prediction_contract["package"]
                / prediction_contract["executable"],
                "ad_lidar_perception installed leaf "
                + tracker["prediction_executable"],
                issues,
                executable=True,
                allow_final_symlink=True,
            )
        if tracker_install is not None:
            _, tracker_launch = tracker_install
            if tracker_launch is not None:
                tracker_runtime = TrackerRuntime(
                    package=tracker["package"],
                    executable=tracker["executable"],
                    launch_path=tracker_launch,
                )

    if issues:
        raise VerificationError(issues)
    return VerifiedSelection(
        detector=detector_runtime,
        tracker=tracker_runtime,
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify pinned optional Autoware perception packages "
            "and artifacts."
        )
    )
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--prefix",
        action="append",
        type=Path,
        help="Ament install prefix; repeat to preserve overlay order.",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> int:
    """CLI entry point; verification only, with no download/build behavior."""
    arguments = _argument_parser().parse_args(argv)
    try:
        selection = load_selection(arguments.selection)
        result = verify_selection(
            selection,
            lock_path=arguments.lock,
            prefixes=arguments.prefix,
            data_root=arguments.data_root,
            environ=environ,
        )
    except (SelectionError, VerificationError) as error:
        if isinstance(error, VerificationError):
            diagnostic = str(error)
        else:
            diagnostic = str(
                VerificationError([f"selection invalid: {error}"])
            )
        print(diagnostic, file=sys.stderr)
        return 1

    selected = []
    if result.detector is not None:
        selected.append(f"detector={result.detector.backend}")
    if result.tracker is not None:
        selected.append("tracker=autoware")
    suffix = (
        ", ".join(selected)
        if selected
        else "no optional backend selected"
    )
    print(f"Autoware perception selection verified ({suffix}).")
    return 0


__all__ = [
    "DetectorRuntime",
    "LockError",
    "TrackerRuntime",
    "VerificationError",
    "VerifiedSelection",
    "load_lock",
    "main",
    "resolve_data_root",
    "verify_selection",
]
