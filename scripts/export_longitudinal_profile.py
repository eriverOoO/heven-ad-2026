#!/usr/bin/env python3
"""Convert profiler CSV reports into Profile Stanley ROS parameters."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CurvePoint:
    speed_mps: float
    value_mps2: float
    delay_s: float = 0.0


def _number(row: dict[str, str], field: str) -> float | None:
    text = row.get(field, "").strip()
    if not text:
        return None
    value = float(text)
    return value if math.isfinite(value) else None


def load_curve(
    profile_csv: Path,
    *,
    command_kind: str,
    command_percent: int,
) -> list[CurvePoint]:
    if command_kind not in {"accelerator", "brake"}:
        raise ValueError(f"unsupported command kind: {command_kind}")
    points: list[CurvePoint] = []
    with profile_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if (
                row.get("command_kind") != command_kind
                or int(row.get("command_percent", "-1")) != command_percent
                or row.get("status") != "complete"
            ):
                continue
            speed_kph = _number(row, "speed_kph")
            if command_kind == "accelerator":
                value = _number(row, "median_acceleration_mps2")
                delay = _number(row, "median_command_echo_delay_sec") or 0.0
            else:
                value = _number(
                    row, "coast_normalized_brake_deceleration_mps2"
                )
                if value is None:
                    value = _number(row, "median_mean_deceleration_mps2")
                delay = max(
                    _number(row, "median_command_echo_delay_sec") or 0.0,
                    _number(row, "median_deceleration_onset_delay_sec") or 0.0,
                )
            if (
                speed_kph is None
                or speed_kph < 0.0
                or value is None
                or value <= 0.0
                or delay < 0.0
            ):
                continue
            points.append(CurvePoint(speed_kph / 3.6, value, delay))

    points.sort(key=lambda point: point.speed_mps)
    deduplicated: list[CurvePoint] = []
    for point in points:
        if deduplicated and math.isclose(
            point.speed_mps, deduplicated[-1].speed_mps, abs_tol=1.0e-12
        ):
            raise ValueError(
                f"duplicate speed in {command_kind} profile: "
                f"{point.speed_mps:.6g} m/s"
            )
        deduplicated.append(point)
    if len(deduplicated) < 2:
        raise ValueError(
            f"{command_kind} command {command_percent}% needs at least "
            "two complete positive measurements"
        )
    return deduplicated


def interpolate(points: list[CurvePoint], speed_mps: float) -> float:
    if speed_mps <= points[0].speed_mps:
        return points[0].value_mps2
    if speed_mps >= points[-1].speed_mps:
        return points[-1].value_mps2
    for lower, upper in zip(points, points[1:]):
        if speed_mps <= upper.speed_mps:
            fraction = (speed_mps - lower.speed_mps) / (
                upper.speed_mps - lower.speed_mps
            )
            return lower.value_mps2 + fraction * (
                upper.value_mps2 - lower.value_mps2
            )
    raise AssertionError("interpolation range was not covered")


def build_parameters(
    accelerator: list[CurvePoint],
    brake: list[CurvePoint],
) -> dict[str, object]:
    minimum_speed = max(accelerator[0].speed_mps, brake[0].speed_mps)
    maximum_speed = min(accelerator[-1].speed_mps, brake[-1].speed_mps)
    if maximum_speed <= minimum_speed:
        raise ValueError("accelerator and brake profiles have no overlapping range")
    speeds = sorted(
        {
            point.speed_mps
            for point in (*accelerator, *brake)
            if minimum_speed <= point.speed_mps <= maximum_speed
        }
    )
    if len(speeds) < 2:
        raise ValueError("profile overlap needs at least two speed samples")
    prefix = "profile_stanley.longitudinal_profile"
    return {
        f"{prefix}.speed_mps": speeds,
        f"{prefix}.acceleration_mps2": [
            interpolate(accelerator, speed) for speed in speeds
        ],
        f"{prefix}.deceleration_mps2": [
            interpolate(brake, speed) for speed in speeds
        ],
        f"{prefix}.braking_delay_s": max(point.delay_s for point in brake),
    }


def render_yaml(parameters: dict[str, object]) -> str:
    lines = []
    for name, value in parameters.items():
        if isinstance(value, list):
            rendered = json.dumps(
                [round(float(item), 9) for item in value], separators=(", ", ": ")
            )
        else:
            rendered = f"{float(value):.9g}"
        lines.append(f"{name}: {rendered}")
    return "\n".join(lines) + "\n"


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accelerator-profile", type=Path, required=True)
    parser.add_argument("--accelerator-percent", type=int, required=True)
    parser.add_argument("--brake-profile", type=Path, required=True)
    parser.add_argument("--brake-percent", type=int, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(arguments)


def main(arguments: Iterable[str] | None = None) -> int:
    args = parse_args(arguments)
    parameters = build_parameters(
        load_curve(
            args.accelerator_profile,
            command_kind="accelerator",
            command_percent=args.accelerator_percent,
        ),
        load_curve(
            args.brake_profile,
            command_kind="brake",
            command_percent=args.brake_percent,
        ),
    )
    text = render_yaml(parameters)
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
