#!/usr/bin/env python3
"""Derive a deterministic, legal multi-lane route corridor from MGeo data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import zipfile


REQUIRED_MGEO_FILES = ("global_info.json", "node_set.json", "link_set.json")
SCHEMA_VERSION = 1
DEFAULT_MATCH_TOLERANCE_M = 0.05


@dataclass(frozen=True)
class Link:
    identifier: str
    from_node_id: str
    to_node_id: str
    points: tuple[tuple[float, float, float], ...]
    max_speed: float
    width_start: float
    width_end: float
    can_move_left_lane: bool
    left_destination_id: str | None
    can_move_right_lane: bool
    right_destination_id: str | None


@dataclass(frozen=True)
class PointCandidate:
    link_id: str
    point_index: int
    distance_m: float


def _read_mgeo_files(source: str | os.PathLike[str]) -> dict[str, bytes]:
    path = Path(source)
    if path.is_dir():
        payloads: dict[str, bytes] = {}
        for name in REQUIRED_MGEO_FILES:
            file_path = path / name
            if not file_path.is_file():
                raise ValueError(f"MGeo directory is missing {name}: {path}")
            try:
                payloads[name] = file_path.read_bytes()
            except OSError as exc:
                raise ValueError(f"failed to read {file_path}: {exc}") from exc
        return payloads

    if not path.is_file():
        raise ValueError(f"MGeo input does not exist: {path}")
    try:
        with zipfile.ZipFile(path) as archive:
            selected: dict[str, zipfile.ZipInfo] = {}
            for member in archive.infolist():
                name = Path(member.filename).name
                if name not in REQUIRED_MGEO_FILES or member.is_dir():
                    continue
                if name in selected:
                    raise ValueError(f"MGeo ZIP contains duplicate {name}")
                selected[name] = member
            missing = [name for name in REQUIRED_MGEO_FILES if name not in selected]
            if missing:
                raise ValueError(f"MGeo ZIP is missing {', '.join(missing)}")
            return {name: archive.read(selected[name]) for name in REQUIRED_MGEO_FILES}
    except zipfile.BadZipFile as exc:
        raise ValueError(f"MGeo input must be a directory or ZIP archive: {path}") from exc
    except OSError as exc:
        raise ValueError(f"failed to read MGeo ZIP {path}: {exc}") from exc


def _parse_json(payload: bytes, name: str) -> Any:
    try:
        return json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON in {name}: {exc}") from exc


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be a finite number")
    return result


def _point(value: Any, context: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{context} must contain exactly three coordinates")
    return tuple(_finite_number(component, context) for component in value)  # type: ignore[return-value]


def _records(document: Any, name: str) -> list[Mapping[str, Any]]:
    if not isinstance(document, list):
        raise ValueError(f"{name} root must be a JSON array")
    if not all(isinstance(record, dict) for record in document):
        raise ValueError(f"{name} entries must be JSON objects")
    return document


def _identifier(record: Mapping[str, Any], context: str) -> str:
    value = record.get("idx")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} has no nonempty idx")
    return value


def _parse_nodes(document: Any) -> set[str]:
    nodes: set[str] = set()
    for record in _records(document, "node_set.json"):
        identifier = _identifier(record, "node_set entry")
        if identifier in nodes:
            raise ValueError(f"duplicate node ID: {identifier}")
        _point(record.get("point"), f"node {identifier} point")
        nodes.add(identifier)
    return nodes


def _optional_destination(record: Mapping[str, Any], field: str, context: str) -> str | None:
    value = record.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} {field} must be a nonempty ID or null")
    return value


def _parse_links(document: Any, node_ids: set[str]) -> dict[str, Link]:
    links: dict[str, Link] = {}
    for record in _records(document, "link_set.json"):
        identifier = _identifier(record, "link_set entry")
        if identifier in links:
            raise ValueError(f"duplicate link ID: {identifier}")
        from_node_id = record.get("from_node_idx")
        to_node_id = record.get("to_node_idx")
        if not isinstance(from_node_id, str) or not isinstance(to_node_id, str):
            raise ValueError(f"link {identifier} has invalid endpoint IDs")
        if from_node_id not in node_ids or to_node_id not in node_ids:
            raise ValueError(f"link {identifier} references a missing node")
        raw_points = record.get("points")
        if not isinstance(raw_points, list) or not raw_points:
            raise ValueError(f"link {identifier} has no points")
        points = tuple(
            _point(point, f"link {identifier} point {index}")
            for index, point in enumerate(raw_points)
        )
        if len({point for point in points}) < 2:
            raise ValueError(f"link {identifier} has no usable segment")
        left_permitted = record.get("can_move_left_lane")
        right_permitted = record.get("can_move_right_lane")
        if not isinstance(left_permitted, bool) or not isinstance(right_permitted, bool):
            raise ValueError(f"link {identifier} lane-change permissions must be booleans")
        links[identifier] = Link(
            identifier=identifier,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            points=points,
            max_speed=_finite_number(record.get("max_speed"), f"link {identifier} max_speed"),
            width_start=_finite_number(record.get("width_start"), f"link {identifier} width_start"),
            width_end=_finite_number(record.get("width_end"), f"link {identifier} width_end"),
            can_move_left_lane=left_permitted,
            left_destination_id=_optional_destination(
                record, "left_lane_change_dst_link_idx", f"link {identifier}"
            ),
            can_move_right_lane=right_permitted,
            right_destination_id=_optional_destination(
                record, "right_lane_change_dst_link_idx", f"link {identifier}"
            ),
        )
    if not links:
        raise ValueError("link_set.json contains no links")
    return links


def _parse_global_path(path: str | os.PathLike[str]) -> tuple[bytes, list[tuple[float, float, float]]]:
    source = Path(path)
    try:
        payload = source.read_bytes()
        text = payload.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"failed to read global path {source}: {exc}") from exc
    points: list[tuple[float, float, float]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        fields = raw_line.replace(",", " ").split()
        if not fields:
            continue
        if len(fields) != 3:
            raise ValueError(
                f"global path line {line_number} must contain exactly x y z"
            )
        try:
            point = tuple(float(field) for field in fields)
        except ValueError as exc:
            raise ValueError(
                f"global path line {line_number} contains a non-numeric coordinate"
            ) from exc
        if not all(math.isfinite(component) for component in point):
            raise ValueError(
                f"global path line {line_number} contains a non-finite coordinate"
            )
        points.append(point)  # type: ignore[arg-type]
    if not points:
        raise ValueError("global path contains no points")
    return payload, points


def _grid_key(point: tuple[float, float, float], cell_size: float) -> tuple[int, int, int]:
    return tuple(math.floor(component / cell_size) for component in point)  # type: ignore[return-value]


def _point_candidates(
    links: Mapping[str, Link], path_points: Sequence[tuple[float, float, float]], tolerance_m: float
) -> list[list[PointCandidate]]:
    if not math.isfinite(tolerance_m) or tolerance_m <= 0.0:
        raise ValueError("match tolerance must be a positive finite distance")
    grid: dict[tuple[int, int, int], list[PointCandidate]] = {}
    for link in links.values():
        for index, point in enumerate(link.points):
            grid.setdefault(_grid_key(point, tolerance_m), []).append(
                PointCandidate(link.identifier, index, 0.0)
            )

    output: list[list[PointCandidate]] = []
    distance_epsilon = 1e-12
    for path_index, point in enumerate(path_points):
        key = _grid_key(point, tolerance_m)
        nearby: list[PointCandidate] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for candidate in grid.get((key[0] + dx, key[1] + dy, key[2] + dz), []):
                        link_point = links[candidate.link_id].points[candidate.point_index]
                        distance = math.dist(point, link_point)
                        if distance <= tolerance_m:
                            nearby.append(
                                PointCandidate(candidate.link_id, candidate.point_index, distance)
                            )
        if not nearby:
            raise ValueError(
                f"global path point {path_index + 1} does not match an MGeo link point "
                f"within {tolerance_m:.6g} m"
            )
        nearest_distance = min(candidate.distance_m for candidate in nearby)
        output.append(sorted(
            (
                candidate
                for candidate in nearby
                if candidate.distance_m <= nearest_distance + distance_epsilon
            ),
            key=lambda candidate: (candidate.link_id, candidate.point_index),
        ))
    return output


def _links_are_continuous(
    previous: Link, current: Link, match_tolerance_m: float,
) -> bool:
    return (
        previous.to_node_id == current.from_node_id
        and math.dist(previous.points[-1], current.points[0]) <= match_tolerance_m
    )


def _is_legal_transition(
    previous: PointCandidate,
    current: PointCandidate,
    links: Mapping[str, Link],
    *,
    match_tolerance_m: float = DEFAULT_MATCH_TOLERANCE_M,
) -> bool:
    if previous.link_id == current.link_id:
        return current.point_index >= previous.point_index
    return _links_are_continuous(
        links[previous.link_id], links[current.link_id], match_tolerance_m
    )


def _match_primary_link_runs(
    candidates_by_point: Sequence[Sequence[PointCandidate]],
    links: Mapping[str, Link],
    *,
    match_tolerance_m: float = DEFAULT_MATCH_TOLERANCE_M,
) -> list[str]:
    # The labels retain link-point progress; this resolves coincident MGeo points
    # by an ordered graph path instead of arbitrary nearest-owner selection.
    labels: dict[tuple[str, int], tuple[tuple[float, int, str, int], PointCandidate | None]] = {}
    for candidate in candidates_by_point[0]:
        labels[(candidate.link_id, candidate.point_index)] = (
            (candidate.distance_m, 0, candidate.link_id, candidate.point_index),
            None,
        )
    history: list[dict[tuple[str, int], PointCandidate | None]] = [
        {state: predecessor for state, (_, predecessor) in labels.items()}
    ]

    for path_index, candidates in enumerate(candidates_by_point[1:], start=2):
        next_labels: dict[tuple[str, int], tuple[tuple[float, int, str, int], PointCandidate]] = {}
        for current in candidates:
            best: tuple[tuple[float, int, str, int], PointCandidate] | None = None
            for (previous_link_id, previous_point_index), (cost, _) in labels.items():
                previous = PointCandidate(previous_link_id, previous_point_index, 0.0)
                if not _is_legal_transition(
                    previous,
                    current,
                    links,
                    match_tolerance_m=match_tolerance_m,
                ):
                    continue
                next_cost = (
                    cost[0] + current.distance_m,
                    cost[1] + int(previous.link_id != current.link_id),
                    previous.link_id,
                    previous.point_index,
                )
                proposal = (next_cost, previous)
                if best is None or proposal[0] < best[0]:
                    best = proposal
            if best is not None:
                next_labels[(current.link_id, current.point_index)] = best
        if not next_labels:
            raise ValueError(
                f"global path has no continuous MGeo transition at point {path_index}"
            )
        labels = next_labels
        history.append({state: predecessor for state, (_, predecessor) in labels.items()})

    final_state = min(labels, key=lambda state: labels[state][0])
    selected: list[PointCandidate] = [PointCandidate(final_state[0], final_state[1], 0.0)]
    for previous_history in reversed(history[1:]):
        predecessor = previous_history[(selected[-1].link_id, selected[-1].point_index)]
        assert predecessor is not None
        selected.append(predecessor)
    selected.reverse()
    runs: list[str] = []
    for candidate in selected:
        if not runs or runs[-1] != candidate.link_id:
            runs.append(candidate.link_id)
    return runs


def _split_adjacent_sequences(
    primary_ids: Sequence[str],
    links: Mapping[str, Link],
    side: str,
    *,
    match_tolerance_m: float = DEFAULT_MATCH_TOLERANCE_M,
) -> list[list[str]]:
    sequences: list[list[str]] = []
    current_sequence: list[str] = []
    for primary_id in primary_ids:
        link = links[primary_id]
        permitted = link.can_move_left_lane if side == "left" else link.can_move_right_lane
        destination_id = link.left_destination_id if side == "left" else link.right_destination_id
        if not permitted:
            if current_sequence:
                sequences.append(current_sequence)
                current_sequence = []
            continue
        if destination_id is None:
            raise ValueError(f"link {primary_id} permits {side} lane change without destination")
        if destination_id not in links:
            raise ValueError(f"link {primary_id} references missing {side} destination {destination_id}")
        if not current_sequence:
            current_sequence = [destination_id]
        elif current_sequence[-1] != destination_id:
            if _links_are_continuous(
                links[current_sequence[-1]],
                links[destination_id],
                match_tolerance_m,
            ):
                current_sequence.append(destination_id)
            else:
                sequences.append(current_sequence)
                current_sequence = [destination_id]
    if current_sequence:
        sequences.append(current_sequence)
    return sequences


def _link_samples_with_ratio(
    link: Link,
) -> Iterable[tuple[tuple[float, float, float], float, float, float]]:
    distances = [0.0]
    for previous, current in zip(link.points, link.points[1:]):
        distances.append(distances[-1] + math.hypot(current[0] - previous[0], current[1] - previous[1]))
    total = distances[-1]
    if total <= 0.0:
        raise ValueError(f"link {link.identifier} has no usable planar segment")
    for point, distance in zip(link.points, distances):
        ratio = distance / total
        width = link.width_start + ratio * (link.width_end - link.width_start)
        yield point, width, link.max_speed / 3.6, ratio


def _link_samples(
    link: Link,
) -> Iterable[tuple[tuple[float, float, float], float, float]]:
    for point, width, speed_limit, _ratio in _link_samples_with_ratio(link):
        yield point, width, speed_limit


def _primary_link_station_ranges(
    primary_ids: Sequence[str],
    links: Mapping[str, Link],
) -> dict[str, tuple[float, float]]:
    station = 0.0
    ranges: dict[str, tuple[float, float]] = {}
    for primary_id in primary_ids:
        samples = list(_link_samples_with_ratio(links[primary_id]))
        length = sum(
            math.hypot(
                current[0][0] - previous[0][0],
                current[0][1] - previous[0][1],
            )
            for previous, current in zip(samples, samples[1:])
        )
        if length <= 0.0:
            raise ValueError(f"link {primary_id} has no usable planar segment")
        ranges[primary_id] = (station, station + length)
        station += length
    return ranges


def _adjacent_link_station_ranges(
    primary_ids: Sequence[str],
    links: Mapping[str, Link],
    side: str,
) -> dict[str, tuple[float, float]]:
    primary_ranges = _primary_link_station_ranges(primary_ids, links)
    aligned: dict[str, tuple[float, float]] = {}
    for primary_id in primary_ids:
        primary = links[primary_id]
        permitted = (
            primary.can_move_left_lane
            if side == "left"
            else primary.can_move_right_lane
        )
        destination_id = (
            primary.left_destination_id
            if side == "left"
            else primary.right_destination_id
        )
        if not permitted or destination_id is None:
            continue
        start_s, end_s = primary_ranges[primary_id]
        if destination_id in aligned:
            previous_start, previous_end = aligned[destination_id]
            aligned[destination_id] = (
                min(previous_start, start_s),
                max(previous_end, end_s),
            )
        else:
            aligned[destination_id] = (start_s, end_s)
    return aligned


def _lane_document(
    lane_sequence_id: str,
    source_link_ids: Sequence[str],
    links: Mapping[str, Link],
    adjacent_lane_sequence_ids: Mapping[str, Sequence[str]],
    aligned_link_stations: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    samples: list[tuple[tuple[float, float, float], float, float, float | None]] = []
    fallback_station = 0.0
    for source_link_id in source_link_ids:
        link_samples = list(_link_samples_with_ratio(links[source_link_id]))
        natural_length = sum(
            math.hypot(
                current[0][0] - previous[0][0],
                current[0][1] - previous[0][1],
            )
            for previous, current in zip(link_samples, link_samples[1:])
        )
        aligned_range = (
            aligned_link_stations.get(source_link_id)
            if aligned_link_stations is not None
            else None
        )
        for point, width, speed_limit, ratio in link_samples:
            route_station = (
                aligned_range[0] + ratio * (aligned_range[1] - aligned_range[0])
                if aligned_range is not None
                else fallback_station + ratio * natural_length
            )
            sample = (point, width, speed_limit, route_station)
            if samples and sample[0] == samples[-1][0]:
                continue
            samples.append(sample)
        fallback_station = (
            aligned_range[1]
            if aligned_range is not None
            else fallback_station + natural_length
        )
    if len(samples) < 2:
        raise ValueError(f"lane sequence {lane_sequence_id} has fewer than two usable points")

    route_s = [sample[3] for sample in samples]
    for previous, current in zip(route_s, route_s[1:]):
        if current <= previous:
            raise ValueError(f"lane sequence {lane_sequence_id} has a zero-length segment")
    yaws: list[float] = []
    for index in range(len(samples)):
        first = samples[max(index - 1, 0)][0]
        last = samples[min(index + 1, len(samples) - 1)][0]
        yaws.append(math.atan2(last[1] - first[1], last[0] - first[0]))
    curvatures = [0.0] * len(samples)
    for index in range(1, len(samples) - 1):
        delta_yaw = math.remainder(yaws[index + 1] - yaws[index - 1], 2.0 * math.pi)
        delta_s = route_s[index + 1] - route_s[index - 1]
        curvatures[index] = delta_yaw / delta_s

    points = [
        {
            "curvature_inv_m": curvatures[index],
            "left_width_m": width / 2.0,
            "right_width_m": width / 2.0,
            "route_s_m": route_s[index],
            "speed_limit_mps": speed_limit,
            "x_m": point[0],
            "y_m": point[1],
            "yaw_rad": yaws[index],
            "z_m": point[2],
        }
        for index, (point, width, speed_limit, _station) in enumerate(samples)
    ]
    return {
        "adjacent_lane_sequence_ids": {
            side: list(ids) for side, ids in sorted(adjacent_lane_sequence_ids.items()) if ids
        },
        "lane_sequence_id": lane_sequence_id,
        "points": points,
        "source_link_attributes": [
            {
                "id": links[source_link_id].identifier,
                "max_speed": links[source_link_id].max_speed,
                "width_end": links[source_link_id].width_end,
                "width_start": links[source_link_id].width_start,
            }
            for source_link_id in source_link_ids
        ],
        "source_link_ids": list(source_link_ids),
    }


def build_route_corridor(
    mgeo: str | os.PathLike[str],
    global_path: str | os.PathLike[str],
    *,
    match_tolerance_m: float = DEFAULT_MATCH_TOLERANCE_M,
) -> dict[str, Any]:
    """Build a cache document without writing it to disk."""

    mgeo_payloads = _read_mgeo_files(mgeo)
    global_info = _parse_json(mgeo_payloads["global_info.json"], "global_info.json")
    if not isinstance(global_info, dict):
        raise ValueError("global_info.json root must be a JSON object")
    node_ids = _parse_nodes(_parse_json(mgeo_payloads["node_set.json"], "node_set.json"))
    links = _parse_links(_parse_json(mgeo_payloads["link_set.json"], "link_set.json"), node_ids)
    global_path_payload, path_points = _parse_global_path(global_path)
    primary_ids = _match_primary_link_runs(
        _point_candidates(links, path_points, match_tolerance_m),
        links,
        match_tolerance_m=match_tolerance_m,
    )

    adjacent_sequences: dict[str, list[tuple[str, list[str]]]] = {}
    for side in ("left", "right"):
        adjacent_sequences[side] = [
            (f"route:0:{side}:{index}", sequence)
            for index, sequence in enumerate(
                _split_adjacent_sequences(
                    primary_ids,
                    links,
                    side,
                    match_tolerance_m=match_tolerance_m,
                ),
                start=1,
            )
        ]
    primary_adjacent = {
        side: [lane_id for lane_id, _ in sequences]
        for side, sequences in adjacent_sequences.items()
        if sequences
    }
    lanes = [_lane_document("route:0", primary_ids, links, primary_adjacent)]
    for side in ("left", "right"):
        aligned_stations = _adjacent_link_station_ranges(
            primary_ids, links, side
        )
        opposite = "right" if side == "left" else "left"
        for lane_id, source_ids in adjacent_sequences[side]:
            lanes.append(
                _lane_document(
                    lane_id,
                    source_ids,
                    links,
                    {opposite: ["route:0"]},
                    aligned_stations,
                )
            )
    return {
        "frame_id": "map",
        "lanes": lanes,
        "primary_lane_sequence_id": "route:0",
        "schema_version": SCHEMA_VERSION,
        "source_sha256": {
            "global_info.json": hashlib.sha256(mgeo_payloads["global_info.json"]).hexdigest(),
            "global_path": hashlib.sha256(global_path_payload).hexdigest(),
            "link_set.json": hashlib.sha256(mgeo_payloads["link_set.json"]).hexdigest(),
            "node_set.json": hashlib.sha256(mgeo_payloads["node_set.json"]).hexdigest(),
        },
    }


def write_route_corridor(document: Mapping[str, Any], output: str | os.PathLike[str]) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"output must not be a symlink: {path}")
    payload = json.dumps(
        document, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        stream = os.fdopen(descriptor, "wb")
        descriptor = -1
        with stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        if descriptor != -1:
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mgeo", required=True, help="MGeo ZIP archive or directory")
    parser.add_argument("--global-path", required=True, help="whitespace/comma-separated x y z path")
    parser.add_argument("--output", required=True, help="route_corridor.json output path")
    parser.add_argument(
        "--match-tolerance-m", type=float, default=DEFAULT_MATCH_TOLERANCE_M,
        help="maximum nearest MGeo point distance (default: 0.05)",
    )
    arguments = parser.parse_args(argv)
    try:
        write_route_corridor(
            build_route_corridor(
                arguments.mgeo,
                arguments.global_path,
                match_tolerance_m=arguments.match_tolerance_m,
            ),
            arguments.output,
        )
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
