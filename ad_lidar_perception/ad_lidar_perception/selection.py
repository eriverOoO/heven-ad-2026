"""Strict loading for the pinned perception backend selection."""

from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
import re
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode


class SelectionError(ValueError):
    """Raised when a perception selection document is invalid."""


class UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys at every depth."""

    def construct_mapping(
        self, node: MappingNode, deep: bool = False
    ) -> dict[Any, Any]:
        if not isinstance(node, MappingNode):
            return super().construct_mapping(node, deep=deep)

        self.flatten_mapping(node)
        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in seen
            except TypeError as error:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found an unhashable mapping key",
                    key_node.start_mark,
                ) from error
            if duplicate:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            seen.add(key)

        return super().construct_mapping(node, deep=deep)


@dataclass(frozen=True)
class DetectorSelection:
    backend: str
    model_subdir: Path
    build_only: bool


@dataclass(frozen=True)
class TrackerSelection:
    backend: str


@dataclass(frozen=True)
class OccupancySelection:
    static_enabled: bool
    dynamic_enabled: bool
    publish_combined: bool


@dataclass(frozen=True)
class PerceptionSelection:
    schema_version: int
    detector: DetectorSelection
    tracker: TrackerSelection
    occupancy: OccupancySelection


_ROOT_KEYS = frozenset(
    {"schema_version", "detector", "tracker", "occupancy"}
)
_DETECTOR_KEYS = frozenset({"backend", "model_subdir", "build_only"})
_TRACKER_KEYS = frozenset({"backend"})
_OCCUPANCY_KEYS = frozenset(
    {"static_enabled", "dynamic_enabled", "publish_combined"}
)
_DETECTOR_BACKENDS = (
    "none",
    "euclidean_cluster",
    "centerpoint_tiny",
    "centerpoint",
    "transfusion",
    "bevfusion_lidar",
)
_TRACKER_BACKENDS = ("none", "autoware")
_WINDOWS_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:")


def load_yaml_payload(payload: bytes | str) -> Any:
    """Parse one YAML payload while rejecting duplicate keys everywhere."""
    try:
        text = (
            payload.decode("utf-8")
            if isinstance(payload, bytes)
            else payload
        )
    except UnicodeError as error:
        raise SelectionError(f"invalid UTF-8 YAML: {error}") from error
    if type(text) is not str:
        raise SelectionError("YAML payload must be bytes or a string")

    if "\x00" in text:
        raise SelectionError("YAML document contains a NUL byte")

    try:
        return yaml.load(text, Loader=UniqueKeySafeLoader)
    except yaml.YAMLError as error:
        raise SelectionError(f"invalid YAML: {error}") from error


def load_yaml_document(path: Path | str) -> Any:
    """Read and strictly parse one YAML document exactly once."""
    yaml_path = Path(path)
    try:
        payload = yaml_path.read_bytes()
    except OSError as error:
        raise SelectionError(f"could not read {yaml_path}: {error}") from error
    return load_yaml_payload(payload)


def _require_mapping(value: Any, name: str) -> dict[Any, Any]:
    if type(value) is not dict:
        raise SelectionError(f"{name} must be a mapping")
    return value


def _format_keys(keys: set[Any]) -> str:
    return ", ".join(sorted((str(key) for key in keys)))


def _require_exact_keys(
    mapping: dict[Any, Any], expected: frozenset[str], name: str
) -> None:
    actual = set(mapping)
    missing = set(expected) - actual
    if missing:
        raise SelectionError(
            f"{name} is missing keys: {_format_keys(missing)}"
        )

    unknown = actual - set(expected)
    if unknown:
        raise SelectionError(
            f"{name} has unknown keys: {_format_keys(unknown)}"
        )


def _require_present_keys(
    mapping: dict[Any, Any], expected: frozenset[str], name: str
) -> None:
    missing = set(expected) - set(mapping)
    if missing:
        raise SelectionError(
            f"{name} is missing keys: {_format_keys(missing)}"
        )


def _reject_unknown_keys(
    mapping: dict[Any, Any], expected: frozenset[str], name: str
) -> None:
    unknown = set(mapping) - set(expected)
    if unknown:
        raise SelectionError(
            f"{name} has unknown keys: {_format_keys(unknown)}"
        )


def _require_string(value: Any, name: str) -> str:
    if type(value) is not str:
        raise SelectionError(f"{name} must be a string")
    return value


def _require_boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise SelectionError(f"{name} must be a boolean")
    return value


def _require_backend(
    value: Any, name: str, supported: tuple[str, ...]
) -> str:
    backend = _require_string(value, name)
    if backend not in supported:
        choices = ", ".join(supported)
        raise SelectionError(f"{name} must be one of: {choices}")
    return backend


def _require_safe_model_subdir(value: Any) -> Path:
    name = "detector.model_subdir"
    text = _require_string(value, name)
    parts = text.split("/")
    unsafe = (
        not text
        or "\x00" in text
        or "\\" in text
        or text.startswith("/")
        or _WINDOWS_DRIVE_PREFIX.match(text) is not None
        or any(part in ("", ".", "..") for part in parts)
    )
    if unsafe:
        raise SelectionError(
            f"{name} must be a safe relative POSIX path"
        )

    path = PurePosixPath(text)
    if path.is_absolute():
        raise SelectionError(
            f"{name} must be a safe relative POSIX path"
        )
    return Path(*path.parts)


def load_selection(path: Path | str) -> PerceptionSelection:
    """Load and validate a schema-version-1 perception selection."""
    try:
        document = load_yaml_document(path)
    except SelectionError as error:
        if "NUL byte" in str(error):
            raise SelectionError(
                "detector.model_subdir must be a safe relative POSIX path"
            ) from error
        raise

    root = _require_mapping(document, "root")
    _require_present_keys(root, _ROOT_KEYS, "root")

    schema_version = root["schema_version"]
    if type(schema_version) is not int:
        raise SelectionError("schema_version must be an integer")
    if schema_version != 1:
        raise SelectionError("schema_version must equal 1")

    detector_data = _require_mapping(root["detector"], "detector")
    tracker_data = _require_mapping(root["tracker"], "tracker")
    occupancy_data = _require_mapping(root["occupancy"], "occupancy")
    _reject_unknown_keys(root, _ROOT_KEYS, "root")
    _require_exact_keys(detector_data, _DETECTOR_KEYS, "detector")
    _require_exact_keys(tracker_data, _TRACKER_KEYS, "tracker")
    _require_exact_keys(occupancy_data, _OCCUPANCY_KEYS, "occupancy")

    detector = DetectorSelection(
        backend=_require_backend(
            detector_data["backend"],
            "detector.backend",
            _DETECTOR_BACKENDS,
        ),
        model_subdir=_require_safe_model_subdir(
            detector_data["model_subdir"]
        ),
        build_only=_require_boolean(
            detector_data["build_only"], "detector.build_only"
        ),
    )
    tracker = TrackerSelection(
        backend=_require_backend(
            tracker_data["backend"], "tracker.backend", _TRACKER_BACKENDS
        )
    )
    occupancy = OccupancySelection(
        static_enabled=_require_boolean(
            occupancy_data["static_enabled"], "occupancy.static_enabled"
        ),
        dynamic_enabled=_require_boolean(
            occupancy_data["dynamic_enabled"], "occupancy.dynamic_enabled"
        ),
        publish_combined=_require_boolean(
            occupancy_data["publish_combined"],
            "occupancy.publish_combined",
        ),
    )

    if tracker.backend != "none" and detector.backend == "none":
        raise SelectionError("tracker requires a non-none detector")
    if occupancy.dynamic_enabled and tracker.backend != "autoware":
        raise SelectionError("dynamic occupancy requires tracker autoware")
    if occupancy.publish_combined and not occupancy.static_enabled:
        raise SelectionError("combined occupancy requires static occupancy")

    return PerceptionSelection(
        schema_version=schema_version,
        detector=detector,
        tracker=tracker,
        occupancy=occupancy,
    )
