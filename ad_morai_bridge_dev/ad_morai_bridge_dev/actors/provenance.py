"""Deterministic validation for MORAI actor-preset map evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import yaml

from ad_morai_bridge_dev.actors.presets import load_presets


_SOURCE_FIELDS = {"map_id", "link_set_path_hint", "link_set_sha256"}
_ROUTE_FIELDS = {"closed", "links"}
_LINK_FIELDS = {
    "id",
    "waypoint_idx",
    "from_node",
    "to_node",
    "first_point",
    "initial_heading_deg",
}


def validate_actor_preset_provenance(
    preset_path: str | Path,
    provenance_path: str | Path,
    *,
    source_link_set: str | Path | None = None,
) -> dict[str, object]:
    """Validate preset routes against committed and optional source evidence."""
    presets = load_presets(preset_path)
    root = _load_yaml(provenance_path, "actor preset provenance")
    if set(root) != {"schema_version", "source", "routes"}:
        raise ValueError("provenance has missing or unexpected top-level fields")
    if root.get("schema_version") != 1:
        raise ValueError("provenance schema_version must be 1")

    source = _mapping(root.get("source"), "provenance source")
    if set(source) != _SOURCE_FIELDS:
        raise ValueError("provenance source has missing or unexpected fields")
    map_id = _text(source.get("map_id"), "source map_id")
    _text(source.get("link_set_path_hint"), "source link_set_path_hint")
    expected_sha256 = _sha256_text(
        source.get("link_set_sha256"), "source link_set_sha256"
    )
    if {preset.map_id for preset in presets.values()} != {map_id}:
        raise ValueError("preset map_id does not exactly match provenance source")

    raw_routes = _mapping(root.get("routes"), "provenance routes")
    if set(raw_routes) != set(presets):
        raise ValueError("provenance route names do not exactly match presets")

    evidence_occurrences: list[Mapping[str, object]] = []
    summary_routes: dict[str, dict[str, object]] = {}
    for name in sorted(presets):
        preset = presets[name]
        route = _mapping(raw_routes[name], f"provenance route {name}")
        if set(route) != _ROUTE_FIELDS:
            raise ValueError(f"provenance route {name} has unexpected fields")
        closed = route.get("closed")
        if type(closed) is not bool:
            raise ValueError(f"provenance route {name} closed must be boolean")
        raw_links = route.get("links")
        if not isinstance(raw_links, list) or not raw_links:
            raise ValueError(f"provenance route {name} links must be nonempty")
        links = [
            _provenance_link(value, f"provenance route {name} link")
            for value in raw_links
        ]
        link_ids = tuple(_text(link["id"], "provenance link id") for link in links)
        link_occurrences = tuple(
            (link_id, _waypoint_idx(link["waypoint_idx"], "provenance waypoint"))
            for link_id, link in zip(link_ids, links)
        )
        if link_occurrences != preset.route_links:
            raise ValueError(
                f"provenance route {name} waypoint occurrences do not match preset"
            )
        if not _same_triplet(links[0]["first_point"], preset.xyz):
            raise ValueError(f"provenance route {name} spawn point does not match preset")
        if not math.isclose(
            _finite(links[0]["initial_heading_deg"], "initial heading"),
            preset.rpy_deg[2],
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"provenance route {name} heading does not match preset")
        for previous, following in zip(links, links[1:]):
            if previous["to_node"] != following["from_node"]:
                raise ValueError(f"provenance route {name} is not node-adjacent")
        calculated_closed = links[-1]["to_node"] == links[0]["from_node"]
        if closed is not calculated_closed:
            raise ValueError(f"provenance route {name} closed flag is incorrect")
        evidence_occurrences.extend(links)
        summary_routes[name] = {"closed": closed, "link_count": len(links)}

    if source_link_set is not None:
        _validate_source_link_set(
            Path(source_link_set), expected_sha256, evidence_occurrences
        )

    return {
        "map_id": map_id,
        "preset_count": len(presets),
        "routes": summary_routes,
        "source_link_set_sha256": expected_sha256,
    }


def _load_yaml(path_value: str | Path, name: str) -> Mapping[str, Any]:
    path = Path(path_value)
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load {name} {path}: {exc}") from exc
    return _mapping(value, name)


def _provenance_link(value: object, name: str) -> Mapping[str, object]:
    link = _mapping(value, name)
    if set(link) != _LINK_FIELDS:
        raise ValueError(f"{name} has missing or unexpected fields")
    parsed: dict[str, object] = {
        "id": _text(link.get("id"), f"{name} id"),
        "waypoint_idx": _waypoint_idx(
            link.get("waypoint_idx"), f"{name} waypoint_idx"
        ),
        "from_node": _text(link.get("from_node"), f"{name} from_node"),
        "to_node": _text(link.get("to_node"), f"{name} to_node"),
        "first_point": _triplet(link.get("first_point"), f"{name} first_point"),
        "initial_heading_deg": _finite(
            link.get("initial_heading_deg"), f"{name} initial_heading_deg"
        ),
    }
    return parsed


def _validate_source_link_set(
    path: Path,
    expected_sha256: str,
    evidence_occurrences: Sequence[Mapping[str, object]],
) -> None:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read source link_set {path}: {exc}") from exc
    actual_sha256 = hashlib.sha256(content).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "source link_set SHA-256 does not match provenance: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    try:
        raw_links = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"source link_set is not valid JSON: {exc}") from exc
    if not isinstance(raw_links, list):
        raise ValueError("source link_set must be an array")

    evidence_ids = {str(evidence["id"]) for evidence in evidence_occurrences}
    selected: dict[str, Mapping[str, Any]] = {}
    for value in raw_links:
        link = _mapping(value, "source link")
        link_id = _text(link.get("idx"), "source link idx")
        if link_id in evidence_ids:
            if link_id in selected:
                raise ValueError(f"source link_set has duplicate link {link_id}")
            selected[link_id] = link
    if set(selected) != evidence_ids:
        missing = sorted(evidence_ids - set(selected))
        raise ValueError(f"source link_set is missing provenance links: {missing}")

    for evidence in evidence_occurrences:
        link_id = str(evidence["id"])
        source = selected[link_id]
        if _text(source.get("from_node_idx"), "source from_node_idx") != evidence[
            "from_node"
        ]:
            raise ValueError(f"source from_node differs for link {link_id}")
        if _text(source.get("to_node_idx"), "source to_node_idx") != evidence[
            "to_node"
        ]:
            raise ValueError(f"source to_node differs for link {link_id}")
        points = source.get("points")
        if not isinstance(points, list):
            raise ValueError(f"source link {link_id} points must be an array")
        waypoint_idx = _waypoint_idx(
            evidence["waypoint_idx"], f"source link {link_id} waypoint_idx"
        )
        if waypoint_idx >= len(points):
            raise ValueError(f"source link {link_id} waypoint is out of range")
        if waypoint_idx + 1 >= len(points):
            raise ValueError(
                f"source link {link_id} waypoint has no following heading point"
            )
        waypoint = _triplet(
            points[waypoint_idx], f"source link {link_id} waypoint"
        )
        following = _triplet(
            points[waypoint_idx + 1], f"source link {link_id} heading point"
        )
        if not _same_triplet(waypoint, evidence["first_point"]):
            raise ValueError(f"source waypoint differs for link {link_id}")
        heading = math.degrees(
            math.atan2(
                following[1] - waypoint[1],
                following[0] - waypoint[0],
            )
        )
        if not math.isclose(
            heading,
            _finite(evidence["initial_heading_deg"], "provenance heading"),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"source heading differs for link {link_id}")


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _sha256_text(value: object, name: str) -> str:
    text = _text(value, name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return text


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _waypoint_idx(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _triplet(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain three finite values")
    if len(value) != 3:
        raise ValueError(f"{name} must contain three finite values")
    return tuple(_finite(item, name) for item in value)


def _same_triplet(left: object, right: object) -> bool:
    left_triplet = _triplet(left, "left triplet")
    right_triplet = _triplet(right, "right triplet")
    return all(
        math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9)
        for a, b in zip(left_triplet, right_triplet)
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="validate MORAI actor presets against route provenance"
    )
    parser.add_argument("presets", type=Path)
    parser.add_argument("provenance", type=Path)
    parser.add_argument("--source-link-set", type=Path)
    arguments = parser.parse_args(argv)
    try:
        summary = validate_actor_preset_provenance(
            arguments.presets,
            arguments.provenance,
            source_link_set=arguments.source_link_set,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
