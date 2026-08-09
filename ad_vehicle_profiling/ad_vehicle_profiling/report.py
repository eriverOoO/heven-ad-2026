from __future__ import annotations

import argparse
import csv
from html import escape
import hashlib
import json
from pathlib import Path
import statistics
from typing import Iterable

from .experiment import TrialSummary
from .storage import RunStore


def render_label(value: object) -> str:
    return escape(str(value), quote=True)


def _median(
    summaries: Iterable[TrialSummary],
    attribute: str,
) -> float | None:
    values = [
        value
        for summary in summaries
        if summary.valid
        and (value := getattr(summary, attribute)) is not None
    ]
    return statistics.median(values) if values else None


def _acceleration_source(
    summaries: Iterable[TrialSummary],
) -> str | None:
    sources = {
        summary.acceleration_source
        for summary in summaries
        if summary.valid and summary.acceleration_source != "unavailable"
    }
    if len(sources) == 1:
        return next(iter(sources))
    if sources:
        return "mixed"
    return None


def _quality_flags(summaries: Iterable[TrialSummary]) -> list[str]:
    return sorted(
        {
            flag
            for summary in summaries
            for flag in summary.quality_flags.split(";")
            if flag
        }
    )


def _repeatability_mad(
    summaries: Iterable[TrialSummary],
    attribute: str,
) -> float | None:
    values = [
        value
        for summary in summaries
        if summary.valid
        and (value := getattr(summary, attribute)) is not None
    ]
    if not values:
        return None
    median_value = statistics.median(values)
    return statistics.median(
        abs(value - median_value) for value in values
    )


def build_report(store: RunStore) -> dict[str, object]:
    manifest_bytes = (store.run_directory / "manifest.json").read_bytes()
    cells = []
    for cell in store.cells:
        summaries = store.summaries(cell)
        status = store.cell_status(cell)
        cells.append(
            {
                "speed_kph": cell.speed_kph,
                "command_kind": cell.command_kind,
                "command_percent": cell.command_percent,
                "status": "incomplete" if status == "pending" else status,
                "valid_trial_count": store.valid_trial_count(cell),
                "attempted_trial_count": store.attempted_trial_count(cell),
                "median_acceleration_mps2": _median(
                    summaries, "effective_acceleration_mps2"
                ),
                "median_simulator_acceleration_mps2": _median(
                    summaries, "median_acceleration_mps2"
                ),
                "acceleration_source": _acceleration_source(summaries),
                "median_stddev_mps2": _median(
                    summaries, "acceleration_stddev_mps2"
                ),
                "median_mad_mps2": _median(
                    summaries, "acceleration_mad_mps2"
                ),
                "median_velocity_derived_acceleration_mps2": _median(
                    summaries, "velocity_derived_acceleration_mps2"
                ),
                "median_cross_check_disagreement_mps2": _median(
                    summaries, "cross_check_disagreement_mps2"
                ),
                "median_peak_abs_jerk_mps3": _median(
                    summaries, "peak_abs_jerk_mps3"
                ),
                "median_mean_deceleration_mps2": _median(
                    summaries, "mean_deceleration_mps2"
                ),
                "mean_deceleration_repeatability_mad_mps2": (
                    _repeatability_mad(
                        summaries, "mean_deceleration_mps2"
                    )
                ),
                "median_peak_deceleration_mps2": _median(
                    summaries, "peak_deceleration_mps2"
                ),
                "median_p95_deceleration_mps2": _median(
                    summaries, "p95_deceleration_mps2"
                ),
                "median_command_echo_delay_sec": _median(
                    summaries, "command_echo_delay_sec"
                ),
                "median_deceleration_onset_delay_sec": _median(
                    summaries, "deceleration_onset_delay_sec"
                ),
                "median_initial_speed_mps": _median(
                    summaries, "initial_speed_mps"
                ),
                "median_final_speed_mps": _median(
                    summaries, "final_speed_mps"
                ),
                "median_speed_drop_mps": _median(
                    summaries, "speed_drop_mps"
                ),
                "median_distance_travelled_m": _median(
                    summaries, "distance_travelled_m"
                ),
                "median_measurement_duration_sec": _median(
                    summaries, "measurement_duration_sec"
                ),
                "median_sample_rate_hz": _median(
                    summaries, "sample_rate_hz"
                ),
                "median_mean_brake_echo_error": _median(
                    summaries, "mean_brake_echo_error"
                ),
                "quality_flags": _quality_flags(summaries),
            }
        )
    coast_deceleration_by_speed = {
        cell["speed_kph"]: cell["median_mean_deceleration_mps2"]
        for cell in cells
        if cell["command_kind"] == "brake"
        and cell["command_percent"] == 0
    }
    for cell in cells:
        coast = (
            coast_deceleration_by_speed.get(cell["speed_kph"])
            if cell["command_kind"] == "brake"
            else None
        )
        total = cell["median_mean_deceleration_mps2"]
        cell["coast_baseline_deceleration_mps2"] = coast
        cell["coast_normalized_brake_deceleration_mps2"] = (
            total - coast
            if total is not None and coast is not None
            else None
        )
    statuses = [cell["status"] for cell in cells]
    return {
        "schema_version": 2,
        "run_id": store.manifest.get("run_id", ""),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "units": {
            "speed": "km/h",
            "command": "percent",
            "acceleration": "m/s^2",
            "jerk": "m/s^3",
            "latency": "s",
            "distance": "m",
        },
        "axes": {
            "speed_kph": sorted({cell.speed_kph for cell in store.cells}),
            "command_percent": sorted(
                {cell.command_percent for cell in store.cells}
            ),
            "command_kind": sorted(
                {cell.command_kind for cell in store.cells}
            ),
        },
        "progress": {
            "cell_count": len(cells),
            "complete_cell_count": statuses.count("complete"),
            "incomplete_cell_count": statuses.count("incomplete"),
            "attempt_limit_cell_count": statuses.count("attempt_limit"),
            "limiter_bound_cell_count": statuses.count("limiter_bound"),
            "total_attempted_trials": sum(
                cell["attempted_trial_count"] for cell in cells
            ),
            "total_valid_trials": sum(
                cell["valid_trial_count"] for cell in cells
            ),
        },
        "cells": cells,
    }


def _color(value: float | None) -> str:
    if value is None:
        return "#d1d5db"
    clipped = max(-10.0, min(10.0, value))
    if clipped >= 0.0:
        intensity = int(235 - 150 * clipped / 10.0)
        return f"#55{intensity:02x}55"
    intensity = int(235 - 150 * abs(clipped) / 10.0)
    return f"#55{intensity:02x}e6"


def render_heatmap(
    profile: dict[str, object],
    command_kind: str,
    output_path: Path,
    *,
    value_field: str | None = None,
    title: str | None = None,
) -> None:
    if command_kind not in {"accelerator", "brake"}:
        raise ValueError(f"unsupported command kind: {command_kind}")
    speeds = list(profile["axes"]["speed_kph"])
    commands = list(profile["axes"]["command_percent"])
    if value_field is None:
        value_field = (
            "median_mean_deceleration_mps2"
            if command_kind == "brake"
            else "median_acceleration_mps2"
        )
    values = {
        (cell["speed_kph"], cell["command_percent"]): cell[value_field]
        for cell in profile["cells"]
        if cell["command_kind"] == command_kind
    }
    left, top = 90, 70
    cell_width, cell_height = 42, 22
    width = left + len(speeds) * cell_width + 40
    height = top + len(commands) * cell_height + 90
    if title is None:
        title = (
            "Accelerator net acceleration"
            if command_kind == "accelerator"
            else "Brake mean deceleration"
        )
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" '
                f'font-size="18">{title} (m/s²)</text>'
        ),
    ]
    for row, command in enumerate(commands):
        y = top + row * cell_height
        parts.append(
            f'<text x="{left - 8}" y="{y + 16}" text-anchor="end" '
            f'font-size="10">{command}</text>'
        )
        for column, speed in enumerate(speeds):
            x = left + column * cell_width
            value = values.get((speed, command))
            label = "no data" if value is None else f"{value:.2f} m/s²"
            parts.extend(
                [
                    (
                        f'<rect x="{x}" y="{y}" width="{cell_width}" '
                        f'height="{cell_height}" fill="{_color(value)}" '
                        'stroke="#ffffff" stroke-width="1">'
                    ),
                    f"<title>{render_label(label)}</title>",
                    "</rect>",
                ]
            )
    for column, speed in enumerate(speeds):
        x = left + column * cell_width + cell_width / 2
        parts.append(
            f'<text x="{x}" y="{top + len(commands) * cell_height + 18}" '
            f'text-anchor="middle" font-size="9">{speed}</text>'
        )
    parts.extend(
        [
            (
                f'<text x="{left + len(speeds) * cell_width / 2}" '
                f'y="{height - 25}" text-anchor="middle" font-size="13">'
                "Initial speed (km/h)</text>"
            ),
            (
                f'<text x="18" y="{top + len(commands) * cell_height / 2}" '
                'text-anchor="middle" font-size="13" '
                'transform="rotate(-90 18 '
                f'{top + len(commands) * cell_height / 2})">'
                f'{"Accelerator" if command_kind == "accelerator" else "Brake"} '
                "command (%)</text>"
            ),
            "</svg>",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_report(run_directory: Path) -> dict[str, Path]:
    store = RunStore.resume(Path(run_directory))
    profile = build_report(store)
    json_path = store.run_directory / "profile.json"
    csv_path = store.run_directory / "profile.csv"
    plots_directory = store.run_directory / "plots"

    json_path.write_text(
        json.dumps(profile, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    cells = profile["cells"]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    outputs = {
        "json": json_path,
        "csv": csv_path,
    }
    for command_kind in profile["axes"]["command_kind"]:
        heatmap = plots_directory / f"{command_kind}_heatmap.svg"
        render_heatmap(profile, command_kind, heatmap)
        outputs[f"{command_kind}_heatmap"] = heatmap
        if command_kind == "brake":
            net_heatmap = plots_directory / "brake_coast_normalized_heatmap.svg"
            render_heatmap(
                profile,
                command_kind,
                net_heatmap,
                value_field="coast_normalized_brake_deceleration_mps2",
                title="Coast-normalized brake deceleration",
            )
            outputs["brake_coast_normalized_heatmap"] = net_heatmap
    return outputs


def main(args=None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate MORAI longitudinal profiling reports"
    )
    parser.add_argument("run_directory", type=Path)
    parsed = parser.parse_args(args)
    outputs = write_report(parsed.run_directory)
    for name, path in outputs.items():
        print(f"{name}: {path}")
