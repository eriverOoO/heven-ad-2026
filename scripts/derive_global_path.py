#!/usr/bin/env python3
"""Create a separately stored, resampled global path without changing its source."""

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Iterable, Sequence


Point = tuple[float, float, float]
FileIdentity = tuple[int, int]
_MAX_OUTPUT_POINTS = 10_000_000


@dataclass(frozen=True)
class DerivedPath:
    points: tuple[Point, ...]
    closed: bool


def _validate_parameters(spacing_m: float, smooth_passes: int, duplicate_tolerance_m: float) -> None:
    if not math.isfinite(spacing_m) or spacing_m <= 0.0:
        raise ValueError("spacing_m must be finite and positive")
    if not isinstance(smooth_passes, int) or smooth_passes < 0:
        raise ValueError("smooth_passes must be a non-negative integer")
    if not math.isfinite(duplicate_tolerance_m) or duplicate_tolerance_m <= 0.0:
        raise ValueError("duplicate_tolerance_m must be finite and positive")


def _point_from_values(values: Sequence[float]) -> Point:
    if len(values) not in (2, 3):
        raise ValueError("each point must have two or three coordinates")
    point = (float(values[0]), float(values[1]), float(values[2]) if len(values) == 3 else 0.0)
    if not all(math.isfinite(value) for value in point):
        raise ValueError("path coordinates must be finite")
    return point


def _xy_distance(first: Point, second: Point) -> float:
    distance = math.hypot(second[0] - first[0], second[1] - first[1])
    if not math.isfinite(distance):
        raise ValueError("path segment length must be finite")
    return distance


def _remove_consecutive_duplicates(points: Iterable[Point], tolerance_m: float) -> list[Point]:
    result: list[Point] = []
    for point in points:
        if not result or _xy_distance(result[-1], point) > tolerance_m:
            result.append(point)
    return result


def _resample_segment(start: Point, end: Point, fraction: float) -> Point:
    if not math.isfinite(fraction) or fraction < 0.0 or fraction > 1.0:
        raise ValueError("path interpolation fraction must be finite and bounded")
    if fraction == 0.0:
        return start
    if fraction == 1.0:
        return end
    result = tuple(
        (1.0 - fraction) * start[index] + fraction * end[index]
        for index in range(3)
    )
    if not all(math.isfinite(value) for value in result):
        raise ValueError("path interpolation result must be finite")
    return result  # type: ignore[return-value]


def _cumulative_lengths(points: Sequence[Point], closed: bool) -> list[float]:
    cumulative = [0.0]
    segment_count = len(points) if closed else len(points) - 1
    for index in range(segment_count):
        length = _xy_distance(points[index], points[(index + 1) % len(points)])
        if length <= 0.0:
            raise ValueError("path segments must have positive XY length")
        total = cumulative[-1] + length
        if not math.isfinite(total):
            raise ValueError("total path length must be finite")
        if total <= cumulative[-1]:
            raise ValueError("each path segment must advance cumulative length")
        cumulative.append(total)
    return cumulative


def _checked_sample_count(count: int) -> None:
    if count > _MAX_OUTPUT_POINTS:
        raise ValueError(
            f"derived path would exceed {_MAX_OUTPUT_POINTS} output points"
        )


def _sample_distances(
    total_length: float,
    spacing_m: float,
    closed: bool,
    duplicate_tolerance_m: float,
) -> list[float]:
    ratio = total_length / spacing_m
    if not math.isfinite(ratio):
        raise ValueError("path length to spacing ratio must be finite")

    if closed:
        interval_count = max(2, round(ratio))
        _checked_sample_count(interval_count + 1)
        return [
            (index / interval_count) * total_length
            for index in range(interval_count)
        ]

    full_step_count = math.floor(ratio)
    _checked_sample_count(full_step_count + 2)
    distances = [0.0]
    for index in range(1, full_step_count + 1):
        distance = index * spacing_m
        if not math.isfinite(distance):
            raise ValueError("sample distance must be finite")
        endpoint_tolerance = (
            duplicate_tolerance_m
            + math.ulp(total_length)
            + math.ulp(distance)
        )
        if distance >= total_length or total_length - distance <= endpoint_tolerance:
            break
        distances.append(distance)
    return distances


def _sample_points(
    points: Sequence[Point],
    closed: bool,
    cumulative: Sequence[float],
    distances: Sequence[float],
) -> list[Point]:
    sampled: list[Point] = []
    segment_index = 0
    segment_count = len(cumulative) - 1
    for distance in distances:
        if not math.isfinite(distance) or distance < 0.0 or distance >= cumulative[-1]:
            raise ValueError("sample distance must be finite and inside the path")
        while (
            segment_index + 1 < segment_count
            and distance >= cumulative[segment_index + 1]
        ):
            segment_index += 1
        segment_start = cumulative[segment_index]
        segment_length = cumulative[segment_index + 1] - segment_start
        fraction = (distance - segment_start) / segment_length
        sampled.append(
            _resample_segment(
                points[segment_index],
                points[(segment_index + 1) % len(points)],
                fraction,
            )
        )
    return sampled


def _resample(
    points: Sequence[Point],
    closed: bool,
    spacing_m: float,
    duplicate_tolerance_m: float,
) -> list[Point]:
    cumulative = _cumulative_lengths(points, closed)
    distances = _sample_distances(
        cumulative[-1],
        spacing_m,
        closed,
        duplicate_tolerance_m,
    )
    sampled = _sample_points(points, closed, cumulative, distances)
    if not closed:
        sampled.append(points[-1])
    return sampled


def _smooth_once(points: Sequence[Point], closed: bool) -> list[Point]:
    if len(points) < 3:
        return list(points)

    smoothed: list[Point] = []
    for index, point in enumerate(points):
        if not closed and index in (0, len(points) - 1):
            smoothed.append(point)
            continue
        previous = points[(index - 1) % len(points)]
        following = points[(index + 1) % len(points)]
        smoothed_point = tuple(
            0.25 * previous[axis] + 0.5 * point[axis] + 0.25 * following[axis]
            for axis in range(3)
        )
        if not all(math.isfinite(value) for value in smoothed_point):
            raise ValueError("smoothed path coordinates must be finite")
        smoothed.append(smoothed_point)  # type: ignore[arg-type]
    return smoothed


def derive_path(
    points: Iterable[Sequence[float]],
    *,
    spacing_m: float,
    smooth_passes: int,
    duplicate_tolerance_m: float = 1.0e-6,
) -> DerivedPath:
    """Normalize, resample, and optionally smooth a supplied path in memory."""

    _validate_parameters(spacing_m, smooth_passes, duplicate_tolerance_m)
    parsed = [_point_from_values(point) for point in points]
    if len(parsed) < 2:
        raise ValueError("path requires at least two useful points")

    deduplicated = _remove_consecutive_duplicates(parsed, duplicate_tolerance_m)
    if len(deduplicated) < 2:
        raise ValueError("path must contain at least two distinct XY points")

    closed = _xy_distance(deduplicated[0], deduplicated[-1]) <= duplicate_tolerance_m
    if closed:
        deduplicated.pop()
    if len(deduplicated) < 2:
        raise ValueError("closed path must contain at least two distinct XY points")

    result = _resample(
        deduplicated,
        closed,
        spacing_m,
        duplicate_tolerance_m,
    )
    for _ in range(smooth_passes):
        result = _smooth_once(result, closed)
    if closed:
        result.append(result[0])
    if len(result) < 2:
        raise ValueError("derived path requires at least two points")
    return DerivedPath(tuple(result), closed)


def _parse_snapshot(source_bytes: bytes, source_name: str) -> tuple[list[Point], int]:
    try:
        text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"cannot decode input path {source_name}: {error}") from error

    delimiter: str | None = None
    column_count: int | None = None
    points: list[Point] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        current_delimiter = "comma" if "," in line else "whitespace"
        if delimiter is None:
            delimiter = current_delimiter
        elif delimiter != current_delimiter:
            raise ValueError(f"line {line_number}: mixed delimiters")
        fields = [field.strip() for field in line.split(",")] if delimiter == "comma" else line.split()
        if column_count is None:
            column_count = len(fields)
        elif column_count != len(fields):
            raise ValueError(f"line {line_number}: mixed coordinate columns")
        if len(fields) not in (2, 3) or any(not field for field in fields):
            raise ValueError(f"line {line_number}: expected two or three coordinates")
        try:
            point = _point_from_values(tuple(float(field) for field in fields))
        except ValueError as error:
            raise ValueError(f"line {line_number}: {error}") from error
        points.append(point)
    if len(points) < 2:
        raise ValueError("input path requires at least two useful points")
    return points, len(points)


def _format_path(derived: DerivedPath) -> bytes:
    return "".join(
        f"{x:.17g} {y:.17g} {z:.17g}\n" for x, y, z in derived.points
    ).encode("utf-8")


def _sidecar_path(output_path: Path) -> Path:
    return Path(f"{output_path}.json")


def _identity_from_stat(stat_result: os.stat_result) -> FileIdentity:
    return stat_result.st_dev, stat_result.st_ino


def _unlink_if_owned(path: Path, expected_identity: FileIdentity) -> None:
    try:
        current_identity = _identity_from_stat(path.stat(follow_symlinks=False))
    except FileNotFoundError:
        return
    if current_identity == expected_identity:
        path.unlink()


def _write_fully(file_descriptor: int, payload: bytes) -> None:
    with os.fdopen(file_descriptor, "wb", closefd=False) as stream:
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = stream.write(view[offset:])
            if written is None or written <= 0:
                raise OSError("temporary file write made no progress")
            offset += written
        stream.flush()
        os.fsync(file_descriptor)


def _create_written_temp(destination: Path, payload: bytes) -> tuple[Path, FileIdentity]:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    identity = _identity_from_stat(os.fstat(descriptor))
    try:
        _write_fully(descriptor, payload)
        os.close(descriptor)
        descriptor = -1
    except BaseException:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        _unlink_if_owned(temporary, identity)
        raise
    return temporary, identity


def _publish_no_clobber(
    temporary: Path,
    destination: Path,
    expected_identity: FileIdentity,
) -> None:
    os.link(temporary, destination, follow_symlinks=False)
    published_identity = _identity_from_stat(
        destination.stat(follow_symlinks=False)
    )
    if published_identity != expected_identity:
        raise OSError(f"published file identity changed unexpectedly: {destination}")


def _verify_published_file(
    path: Path,
    expected_identity: FileIdentity,
    expected_bytes: bytes,
) -> None:
    with path.open("rb") as stream:
        if _identity_from_stat(os.fstat(stream.fileno())) != expected_identity:
            raise OSError(f"published file was replaced before verification: {path}")
        if stream.read() != expected_bytes:
            raise OSError(f"published file bytes differ from prepared data: {path}")


def _write_new_files(output_path: Path, output_bytes: bytes, sidecar_bytes: bytes) -> None:
    sidecar_path = _sidecar_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_files: list[tuple[Path, FileIdentity]] = []
    published_files: list[tuple[Path, FileIdentity]] = []
    try:
        output_temp = _create_written_temp(output_path, output_bytes)
        temporary_files.append(output_temp)
        sidecar_temp = _create_written_temp(sidecar_path, sidecar_bytes)
        temporary_files.append(sidecar_temp)

        published_files.append((output_path, output_temp[1]))
        _publish_no_clobber(output_temp[0], output_path, output_temp[1])
        published_files.append((sidecar_path, sidecar_temp[1]))
        _publish_no_clobber(sidecar_temp[0], sidecar_path, sidecar_temp[1])

        _verify_published_file(output_path, output_temp[1], output_bytes)
        _verify_published_file(sidecar_path, sidecar_temp[1], sidecar_bytes)
    except BaseException:
        for path, identity in reversed(published_files):
            _unlink_if_owned(path, identity)
        raise
    finally:
        for path, identity in temporary_files:
            _unlink_if_owned(path, identity)


def derive_file(
    input_path: Path,
    output_path: Path,
    *,
    spacing_m: float,
    smooth_passes: int,
    duplicate_tolerance_m: float,
) -> DerivedPath:
    """Derive a path into two new files while leaving the input untouched."""

    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must differ")
    sidecar_path = _sidecar_path(output_path)
    if output_path.exists() or sidecar_path.exists():
        raise ValueError("output path or sidecar already exists")

    try:
        source_bytes = input_path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read input path {input_path}: {error}") from error
    source_points, source_point_count = _parse_snapshot(
        source_bytes, str(input_path)
    )
    derived = derive_path(
        source_points,
        spacing_m=spacing_m,
        smooth_passes=smooth_passes,
        duplicate_tolerance_m=duplicate_tolerance_m,
    )
    output_bytes = _format_path(derived)
    sidecar = {
        "closed": derived.closed,
        "derived_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "derived_point_count": len(derived.points),
        "duplicate_tolerance_m": duplicate_tolerance_m,
        "smooth_passes": smooth_passes,
        "source_point_count": source_point_count,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "spacing_m": spacing_m,
    }
    _write_new_files(
        output_path,
        output_bytes,
        json.dumps(sidecar, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    return derived


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--spacing-m", type=float, default=0.5)
    parser.add_argument("--smooth-passes", type=int, default=1)
    parser.add_argument("--duplicate-tolerance-m", type=float, default=1.0e-6)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _argument_parser().parse_args(argv)
    try:
        derive_file(
            arguments.input,
            arguments.output,
            spacing_m=arguments.spacing_m,
            smooth_passes=arguments.smooth_passes,
            duplicate_tolerance_m=arguments.duplicate_tolerance_m,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
