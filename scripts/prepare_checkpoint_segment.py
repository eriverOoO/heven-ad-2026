#!/usr/bin/env python3
"""Extract an inclusive checkpoint-to-checkpoint global-path segment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Sequence


def load_path(path: str | Path) -> list[tuple[float, float, float]]:
    points = []
    for line_number, raw in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = raw.split()
        if not fields:
            continue
        if len(fields) not in (2, 3):
            raise ValueError(
                f"invalid path line {line_number}: expected 2 or 3 numbers"
            )
        try:
            values = tuple(float(value) for value in fields)
        except ValueError as exc:
            raise ValueError(
                f"invalid path line {line_number}: non-numeric coordinate"
            ) from exc
        point = values if len(values) == 3 else (*values, 0.0)
        if not all(math.isfinite(value) for value in point):
            raise ValueError(
                f"invalid path line {line_number}: non-finite coordinate"
            )
        points.append(point)
    if len(points) < 2:
        raise ValueError("source path needs at least two points")
    return points


def load_checkpoints(path: str | Path) -> dict[int, tuple[float, float]]:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = document["checkpoints"]
        checkpoints = {
            int(entry["number"]): (
                float(entry["position"]["x_m"]),
                float(entry["position"]["y_m"]),
            )
            for entry in entries
        }
    except (
        OSError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(f"invalid checkpoints file {path}: {exc}") from exc
    if not checkpoints:
        raise ValueError("checkpoints file is empty")
    return checkpoints


def _nearest_indices(
    points: Sequence[tuple[float, float, float]],
    xy: tuple[float, float],
) -> tuple[list[int], float]:
    distances = [
        math.hypot(point[0] - xy[0], point[1] - xy[1])
        for point in points
    ]
    minimum = min(distances)
    return (
        [
            index
            for index, distance in enumerate(distances)
            if math.isclose(distance, minimum, abs_tol=1.0e-9)
        ],
        minimum,
    )


def extract_segment(
    points: Sequence[tuple[float, float, float]],
    checkpoints: dict[int, tuple[float, float]],
    start_checkpoint: int,
    end_checkpoint: int,
    *,
    maximum_endpoint_error_m: float = 0.5,
) -> tuple[list[tuple[float, float, float]], int, int]:
    if start_checkpoint not in checkpoints or end_checkpoint not in checkpoints:
        raise ValueError("requested checkpoint is missing")
    start_matches, start_error = _nearest_indices(
        points, checkpoints[start_checkpoint]
    )
    end_matches, end_error = _nearest_indices(
        points, checkpoints[end_checkpoint]
    )
    if max(start_error, end_error) > maximum_endpoint_error_m:
        raise ValueError(
            "checkpoint does not lie on the source path within "
            f"{maximum_endpoint_error_m:.3f} m"
        )

    # Use the last tied start and first tied end so adjacent duplicate
    # checkpoint points do not create a zero-length first/last segment.
    start_index = max(start_matches)
    end_index = min(end_matches)
    if start_index >= end_index:
        raise ValueError("end checkpoint must follow start checkpoint")

    segment = []
    for point in points[start_index : end_index + 1]:
        if segment and math.dist(segment[-1], point) <= 1.0e-9:
            continue
        segment.append(point)
    if len(segment) < 2:
        raise ValueError("checkpoint segment has no usable geometry")
    return segment, start_index, end_index


def segment_length(points: Sequence[tuple[float, float, float]]) -> float:
    return sum(
        math.hypot(second[0] - first[0], second[1] - first[1])
        for first, second in zip(points, points[1:])
    )


def build_route_corridor(
    points: Sequence[tuple[float, float, float]],
    *,
    global_path_sha256: str,
    lane_sequence_id: str,
    lane_half_width_m: float = 1.75,
    speed_limit_mps: float = 16.25,
) -> dict:
    if len(global_path_sha256) != 64:
        raise ValueError("global path SHA-256 must contain 64 hex characters")
    if not lane_sequence_id:
        raise ValueError("lane sequence ID must not be empty")
    if lane_half_width_m <= 0.0 or speed_limit_mps < 0.0:
        raise ValueError("lane dimensions and speed limit are invalid")

    stations = [0.0]
    headings = []
    for first, second in zip(points, points[1:]):
        distance = math.hypot(
            second[0] - first[0], second[1] - first[1]
        )
        if distance <= 1.0e-9:
            raise ValueError("route corridor cannot contain duplicate points")
        stations.append(stations[-1] + distance)
        headings.append(
            math.atan2(second[1] - first[1], second[0] - first[0])
        )
    headings.append(headings[-1])

    curvatures = []
    for index in range(len(points)):
        first = max(0, index - 1)
        second = min(len(points) - 1, index + 1)
        ds = stations[second] - stations[first]
        delta_heading = math.remainder(
            headings[second] - headings[first], 2.0 * math.pi
        )
        curvatures.append(delta_heading / ds if ds > 0.0 else 0.0)

    lane_points = [
        {
            "curvature_inv_m": curvature,
            "left_width_m": lane_half_width_m,
            "right_width_m": lane_half_width_m,
            "route_s_m": station,
            "speed_limit_mps": speed_limit_mps,
            "x_m": point[0],
            "y_m": point[1],
            "yaw_rad": heading,
            "z_m": point[2],
        }
        for point, station, heading, curvature in zip(
            points, stations, headings, curvatures
        )
    ]
    return {
        "frame_id": "map",
        "lanes": [
            {
                "adjacent_lane_sequence_ids": {
                    "left": [],
                    "right": [],
                },
                "lane_sequence_id": lane_sequence_id,
                "points": lane_points,
                "source_link_ids": [f"global-path-segment:{lane_sequence_id}"],
            }
        ],
        "primary_lane_sequence_id": lane_sequence_id,
        "schema_version": 1,
        "source_sha256": {"global_path": global_path_sha256},
    }


def write_atomic(path: str | Path, points: Sequence[tuple[float, ...]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(" ".join(map(str, point)) + "\n" for point in points)
    if output.exists() and output.read_text(encoding="utf-8") == content:
        return
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        delete=False,
    ) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    os.replace(temporary, output)


def write_json_atomic(path: str | Path, document: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") == content:
        return
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        delete=False,
    ) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    os.replace(temporary, output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corridor-output", type=Path)
    arguments = parser.parse_args()

    segment, start_index, end_index = extract_segment(
        load_path(arguments.source),
        load_checkpoints(arguments.checkpoints),
        arguments.start,
        arguments.end,
    )
    write_atomic(arguments.output, segment)
    path_digest = hashlib.sha256(arguments.output.read_bytes()).hexdigest()
    if arguments.corridor_output is not None:
        write_json_atomic(
            arguments.corridor_output,
            build_route_corridor(
                segment,
                global_path_sha256=path_digest,
                lane_sequence_id=(
                    f"checkpoint:{arguments.start}-{arguments.end}"
                ),
            ),
        )
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "corridor_output": (
                    str(arguments.corridor_output)
                    if arguments.corridor_output is not None
                    else None
                ),
                "global_path_sha256": path_digest,
                "start_checkpoint": arguments.start,
                "end_checkpoint": arguments.end,
                "source_start_index": start_index,
                "source_end_index": end_index,
                "point_count": len(segment),
                "length_m": segment_length(segment),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
