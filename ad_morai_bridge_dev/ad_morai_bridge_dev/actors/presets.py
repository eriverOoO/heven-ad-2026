"""Strict, side-effect-free actor preset loading."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


@dataclass(frozen=True)
class ActorPreset:
    name: str
    client_key: str
    map_id: str
    actor_type: str
    behavior: str
    request_id: str
    model_name: str
    label: str
    xyz: tuple[float, float, float]
    rpy_deg: tuple[float, float, float]
    velocity: float
    decision_range: float
    route_links: tuple[tuple[str, int], ...]


_PRESET_FIELDS = {
    "name",
    "map_id",
    "actor_type",
    "behavior",
    "request_id",
    "model_name",
    "label",
    "xyz",
    "rpy_deg",
    "velocity",
    "decision_range",
    "route_links",
}
_ROUTE_LINK_FIELDS = {"id", "waypoint_idx"}
_FORBIDDEN_KEYS = {
    "destroy_all",
    "destroy_all_actors",
    "load_map",
    "map_load",
    "scenario_load",
    "simulation_pause",
    "simulation_resume",
    "simulation_start",
    "simulation_stop",
}


def load_presets(source: str | Path) -> dict[str, ActorPreset]:
    path = Path(source)
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load actor presets {path}: {exc}") from exc
    root = _mapping(document, "actor preset document")
    if _find_forbidden_key(root):
        raise ValueError("actor presets contain a forbidden operation key")
    if root.get("schema_version") != 1:
        raise ValueError("actor preset schema_version must be 1")
    if set(root) != {"schema_version", "client_key", "presets"}:
        raise ValueError("actor preset document has unexpected top-level fields")
    client_key = _text(root.get("client_key"), "client_key")
    raw_presets = root.get("presets")
    if not isinstance(raw_presets, list) or not raw_presets:
        raise ValueError("actor preset document requires a nonempty presets array")

    presets: dict[str, ActorPreset] = {}
    request_ids: set[str] = set()
    for value in raw_presets:
        preset = _parse_preset(value, client_key)
        if preset.name in presets:
            raise ValueError(f"duplicate preset name: {preset.name}")
        if preset.request_id in request_ids:
            raise ValueError(f"duplicate preset request_id: {preset.request_id}")
        presets[preset.name] = preset
        request_ids.add(preset.request_id)
    return presets


def _parse_preset(value: object, client_key: str) -> ActorPreset:
    mapping = _mapping(value, "actor preset")
    if set(mapping) != _PRESET_FIELDS:
        raise ValueError("actor preset has missing or unexpected fields")
    actor_type = _text(mapping.get("actor_type"), "actor_type")
    if actor_type != "vehicle":
        raise ValueError("actor preset supports only actor_type vehicle")
    behavior = _text(mapping.get("behavior"), "behavior")
    if behavior != "physical_ai":
        raise ValueError("actor preset supports only behavior physical_ai")
    velocity = _finite(mapping.get("velocity"), "velocity")
    if velocity < 0.0:
        raise ValueError("velocity must be nonnegative")
    decision_range = _finite(mapping.get("decision_range"), "decision_range")
    if decision_range <= 0.0:
        raise ValueError("decision_range must be positive")
    raw_links = mapping.get("route_links")
    if not isinstance(raw_links, list) or not raw_links:
        raise ValueError("route_links must be a nonempty array")
    route_links = tuple(_parse_link(item) for item in raw_links)
    return ActorPreset(
        name=_text(mapping.get("name"), "name"),
        client_key=client_key,
        map_id=_text(mapping.get("map_id"), "map_id"),
        actor_type=actor_type,
        behavior=behavior,
        request_id=_text(mapping.get("request_id"), "request_id"),
        model_name=_text(mapping.get("model_name"), "model_name"),
        label=_text(mapping.get("label"), "label"),
        xyz=_triplet(mapping.get("xyz"), "xyz"),
        rpy_deg=_triplet(mapping.get("rpy_deg"), "rpy_deg"),
        velocity=velocity,
        decision_range=decision_range,
        route_links=route_links,
    )


def _parse_link(value: object) -> tuple[str, int]:
    mapping = _mapping(value, "route link")
    if set(mapping) != _ROUTE_LINK_FIELDS:
        raise ValueError("route link has missing or unexpected fields")
    waypoint_idx = mapping.get("waypoint_idx")
    if isinstance(waypoint_idx, bool) or not isinstance(waypoint_idx, int):
        raise ValueError("waypoint_idx must be an integer")
    if waypoint_idx < 0:
        raise ValueError("waypoint_idx must be nonnegative")
    return _text(mapping.get("id"), "route link ID"), waypoint_idx


def _find_forbidden_key(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).lower() in _FORBIDDEN_KEYS or _find_forbidden_key(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_find_forbidden_key(child) for child in value)
    return False


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _triplet(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{name} must contain three finite values")
    return tuple(_finite(item, name) for item in value)
