from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "export_longitudinal_profile.py"
SPEC = importlib.util.spec_from_file_location("export_longitudinal_profile", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
profile_export = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = profile_export
SPEC.loader.exec_module(profile_export)


FIELDS = [
    "speed_kph",
    "command_kind",
    "command_percent",
    "status",
    "median_acceleration_mps2",
    "median_mean_deceleration_mps2",
    "coast_normalized_brake_deceleration_mps2",
    "median_command_echo_delay_sec",
    "median_deceleration_onset_delay_sec",
]


def write_profile(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_builds_one_interpolated_envelope_and_conservative_brake_delay(tmp_path):
    accelerator_csv = tmp_path / "accelerator.csv"
    brake_csv = tmp_path / "brake.csv"
    write_profile(
        accelerator_csv,
        [
            {
                "speed_kph": speed,
                "command_kind": "accelerator",
                "command_percent": 40,
                "status": "complete",
                "median_acceleration_mps2": value,
            }
            for speed, value in ((0, 1.0), (10, 2.0), (20, 3.0))
        ],
    )
    write_profile(
        brake_csv,
        [
            {
                "speed_kph": speed,
                "command_kind": "brake",
                "command_percent": 20,
                "status": "complete",
                "coast_normalized_brake_deceleration_mps2": value,
                "median_command_echo_delay_sec": echo,
                "median_deceleration_onset_delay_sec": onset,
            }
            for speed, value, echo, onset in (
                (0, 0.5, 0.02, 0.1),
                (20, 1.5, 0.03, 0.2),
            )
        ],
    )

    parameters = profile_export.build_parameters(
        profile_export.load_curve(
            accelerator_csv, command_kind="accelerator", command_percent=40
        ),
        profile_export.load_curve(
            brake_csv, command_kind="brake", command_percent=20
        ),
    )

    prefix = "profile_stanley.longitudinal_profile"
    assert parameters[f"{prefix}.speed_mps"] == pytest.approx(
        [0.0, 10.0 / 3.6, 20.0 / 3.6]
    )
    assert parameters[f"{prefix}.acceleration_mps2"] == pytest.approx(
        [1.0, 2.0, 3.0]
    )
    assert parameters[f"{prefix}.deceleration_mps2"] == pytest.approx(
        [0.5, 1.0, 1.5]
    )
    assert parameters[f"{prefix}.braking_delay_s"] == pytest.approx(0.2)

    rendered = profile_export.render_yaml(parameters)
    arrays = {
        line.split(": ", 1)[0]: json.loads(line.split(": ", 1)[1])
        for line in rendered.splitlines()
        if line.endswith("]")
    }
    assert arrays[f"{prefix}.speed_mps"][1] == pytest.approx(2.777777778)


def test_rejects_an_incomplete_or_too_sparse_selected_command(tmp_path):
    profile_csv = tmp_path / "profile.csv"
    write_profile(
        profile_csv,
        [
            {
                "speed_kph": 10,
                "command_kind": "accelerator",
                "command_percent": 40,
                "status": "complete",
                "median_acceleration_mps2": 1.0,
            },
            {
                "speed_kph": 20,
                "command_kind": "accelerator",
                "command_percent": 40,
                "status": "incomplete",
                "median_acceleration_mps2": 2.0,
            },
        ],
    )

    with pytest.raises(ValueError, match="at least two complete"):
        profile_export.load_curve(
            profile_csv, command_kind="accelerator", command_percent=40
        )
