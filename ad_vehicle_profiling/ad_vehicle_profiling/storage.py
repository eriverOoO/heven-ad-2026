from __future__ import annotations

import csv
from dataclasses import asdict, fields
import json
import os
from pathlib import Path
from typing import Iterable, Mapping

from .experiment import (
    ExperimentCell,
    TrialSample,
    TrialSummary,
    build_cells,
    needs_more_trials,
)


_TRIAL_FIELDS = (
    "speed_kph",
    "command_kind",
    "command_percent",
    *(field.name for field in fields(TrialSummary)),
)
_REQUIRED_TRIAL_FIELDS = (
    "speed_kph",
    "command_kind",
    "command_percent",
    "valid",
    "sample_count",
)
_RAW_FIELDS = (
    "speed_kph",
    "command_kind",
    "command_percent",
    "trial_index",
    *(field.name for field in fields(TrialSample)),
)
_EVENT_FIELDS = (
    "unix_time_sec",
    "monotonic_time_sec",
    "event",
    "phase",
    "speed_kph",
    "command_kind",
    "command_percent",
    "trial_index",
    "detail",
)
_TERMINAL_CLASSIFICATIONS = {"limiter_bound", "unreachable"}


def _atomic_json(path: Path, document: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _cell_key(cell: ExperimentCell) -> str:
    return (
        f"{cell.speed_kph}:{cell.command_kind}:{cell.command_percent}"
    )


class RunStore:
    def __init__(
        self,
        run_directory: Path,
        manifest: Mapping[str, object],
        cells: Iterable[ExperimentCell],
    ) -> None:
        self.run_directory = run_directory
        self.manifest = dict(manifest)
        self.cells = tuple(cells)
        self._summaries = {cell: [] for cell in self.cells}
        self._classifications: dict[ExperimentCell, str] = {}

    @classmethod
    def create(
        cls,
        run_directory: Path,
        manifest: Mapping[str, object],
        *,
        cells: Iterable[ExperimentCell] | None = None,
    ) -> "RunStore":
        run_directory = Path(run_directory)
        if run_directory.exists() and any(run_directory.iterdir()):
            raise FileExistsError(f"run directory is not empty: {run_directory}")
        run_directory.mkdir(parents=True, exist_ok=True)
        selected_cells = tuple(cells if cells is not None else build_cells())
        document = dict(manifest)
        document["cells"] = [asdict(cell) for cell in selected_cells]
        _atomic_json(run_directory / "manifest.json", document)
        _atomic_json(run_directory / "classifications.json", {})
        with (run_directory / "trials.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            csv.DictWriter(stream, fieldnames=_TRIAL_FIELDS).writeheader()
            stream.flush()
            os.fsync(stream.fileno())
        with (run_directory / "raw_samples.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            csv.DictWriter(stream, fieldnames=_RAW_FIELDS).writeheader()
            stream.flush()
            os.fsync(stream.fileno())
        with (run_directory / "events.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            csv.DictWriter(stream, fieldnames=_EVENT_FIELDS).writeheader()
            stream.flush()
            os.fsync(stream.fileno())
        return cls(run_directory, document, selected_cells)

    @classmethod
    def resume(cls, run_directory: Path) -> "RunStore":
        run_directory = Path(run_directory)
        manifest = json.loads(
            (run_directory / "manifest.json").read_text(encoding="utf-8")
        )
        cells = tuple(
            ExperimentCell(
                int(item["speed_kph"]),
                str(item["command_kind"]),
                int(item["command_percent"]),
            )
            for item in manifest["cells"]
        )
        store = cls(run_directory, manifest, cells)
        store._load_trials()
        events_path = run_directory / "events.csv"
        if not events_path.exists():
            with events_path.open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                csv.DictWriter(
                    stream, fieldnames=_EVENT_FIELDS
                ).writeheader()
                stream.flush()
                os.fsync(stream.fileno())
        classification_path = run_directory / "classifications.json"
        if classification_path.exists():
            raw = json.loads(classification_path.read_text(encoding="utf-8"))
            keyed_cells = {_cell_key(cell): cell for cell in cells}
            store._classifications = {
                keyed_cells[key]: value
                for key, value in raw.items()
                if key in keyed_cells
            }
        return store

    def _load_trials(self) -> None:
        path = self.run_directory / "trials.csv"
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for index, row in enumerate(rows):
            if any(
                row.get(field) is None for field in _REQUIRED_TRIAL_FIELDS
            ):
                if index == len(rows) - 1:
                    continue
                raise ValueError("malformed non-trailing trials.csv row")
            cell = ExperimentCell(
                int(row["speed_kph"]),
                row["command_kind"],
                int(row["command_percent"]),
            )
            if cell not in self._summaries:
                raise ValueError(f"trial references unknown cell: {cell}")
            self._summaries[cell].append(self._summary_from_row(row))

    @staticmethod
    def _optional_float(value: str | None) -> float | None:
        return None if value in ("", None) else float(value)

    @classmethod
    def _summary_from_row(cls, row: Mapping[str, str]) -> TrialSummary:
        median_acceleration = cls._optional_float(
            row["median_acceleration_mps2"]
        )
        velocity_derived = cls._optional_float(
            row["velocity_derived_acceleration_mps2"]
        )
        effective_acceleration = cls._optional_float(
            row.get("effective_acceleration_mps2")
        )
        acceleration_source = row.get("acceleration_source") or ""
        if effective_acceleration is None:
            if (
                median_acceleration is not None
                and velocity_derived is not None
                and abs(median_acceleration) <= 0.02
                and abs(velocity_derived) >= 0.05
            ):
                effective_acceleration = velocity_derived
                acceleration_source = "velocity_derived"
            elif median_acceleration is not None:
                effective_acceleration = median_acceleration
                acceleration_source = "simulator_field"
            else:
                acceleration_source = "unavailable"
        integer_fields = {"sample_count", "baseline_sample_count"}
        string_fields = {"acceleration_source", "quality_flags"}
        values: dict[str, object] = {}
        for field in fields(TrialSummary):
            name = field.name
            if name == "valid":
                values[name] = row.get(name) == "1"
            elif name == "rejection_reason":
                values[name] = row.get(name) or None
            elif name in integer_fields:
                values[name] = int(row.get(name) or 0)
            elif name in string_fields:
                values[name] = row.get(name) or ""
            else:
                values[name] = cls._optional_float(row.get(name))
        values["median_acceleration_mps2"] = median_acceleration
        values["velocity_derived_acceleration_mps2"] = velocity_derived
        values["effective_acceleration_mps2"] = effective_acceleration
        values["acceleration_source"] = acceleration_source
        return TrialSummary(**values)

    def append_trial(
        self, cell: ExperimentCell, summary: TrialSummary
    ) -> None:
        if cell not in self._summaries:
            raise ValueError(f"unknown experiment cell: {cell}")
        row = {
            "speed_kph": cell.speed_kph,
            "command_kind": cell.command_kind,
            "command_percent": cell.command_percent,
            **{
                key: (
                    "1"
                    if key == "valid" and value
                    else "0"
                    if key == "valid"
                    else ""
                    if value is None
                    else value
                )
                for key, value in asdict(summary).items()
            },
        }
        with (self.run_directory / "trials.csv").open(
            "a", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=_TRIAL_FIELDS)
            writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
        self._summaries[cell].append(summary)

    def append_event(
        self,
        *,
        unix_time_sec: float,
        monotonic_time_sec: float,
        event: str,
        phase: str,
        cell: ExperimentCell,
        trial_index: int,
        detail: str = "",
    ) -> None:
        with (self.run_directory / "events.csv").open(
            "a", encoding="utf-8", newline=""
        ) as stream:
            csv.DictWriter(stream, fieldnames=_EVENT_FIELDS).writerow(
                {
                    "unix_time_sec": unix_time_sec,
                    "monotonic_time_sec": monotonic_time_sec,
                    "event": event,
                    "phase": phase,
                    "speed_kph": cell.speed_kph,
                    "command_kind": cell.command_kind,
                    "command_percent": cell.command_percent,
                    "trial_index": trial_index,
                    "detail": detail,
                }
            )
            stream.flush()
            os.fsync(stream.fileno())

    def append_samples(
        self,
        cell: ExperimentCell,
        *,
        trial_index: int,
        samples: Iterable[TrialSample],
    ) -> None:
        if cell not in self._summaries:
            raise ValueError(f"unknown experiment cell: {cell}")
        with (self.run_directory / "raw_samples.csv").open(
            "a", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=_RAW_FIELDS)
            for sample in samples:
                writer.writerow(
                    {
                        "speed_kph": cell.speed_kph,
                        "command_kind": cell.command_kind,
                        "command_percent": cell.command_percent,
                        "trial_index": trial_index,
                        **asdict(sample),
                    }
                )
            stream.flush()
            os.fsync(stream.fileno())

    def write_state(self, state: Mapping[str, object]) -> None:
        _atomic_json(self.run_directory / "run_state.json", dict(state))

    def read_state(self) -> dict[str, object]:
        state_path = self.run_directory / "run_state.json"
        if not state_path.exists():
            return {}
        return json.loads(state_path.read_text(encoding="utf-8"))

    def valid_trial_count(self, cell: ExperimentCell) -> int:
        return sum(summary.valid for summary in self._summaries[cell])

    def attempted_trial_count(self, cell: ExperimentCell) -> int:
        return len(self._summaries[cell])

    def summaries(self, cell: ExperimentCell) -> tuple[TrialSummary, ...]:
        return tuple(self._summaries[cell])

    def classify_cell(self, cell: ExperimentCell, status: str) -> None:
        if cell not in self._summaries:
            raise ValueError(f"unknown experiment cell: {cell}")
        if status not in _TERMINAL_CLASSIFICATIONS:
            raise ValueError(f"unsupported terminal classification: {status}")
        self._classifications[cell] = status
        _atomic_json(
            self.run_directory / "classifications.json",
            {
                _cell_key(item): classification
                for item, classification in self._classifications.items()
            },
        )

    def cell_status(self, cell: ExperimentCell) -> str:
        if cell in self._classifications:
            return self._classifications[cell]
        minimum = int(self.manifest["minimum_valid_trials"])
        maximum = int(self.manifest["maximum_attempts"])
        attempts = self.attempted_trial_count(cell)
        valid_count = self.valid_trial_count(cell)
        needs_repeat = needs_more_trials(
            self._summaries[cell],
            attempted_count=min(attempts, maximum - 1),
            minimum=minimum,
            maximum=maximum,
            mad_limit=float(self.manifest.get("mad_limit_mps2", 0.15)),
            disagreement_limit=float(
                self.manifest.get("cross_check_limit_mps2", 0.2)
            ),
            repeatability_mad_limit=float(
                self.manifest.get(
                    "repeatability_mad_limit_mps2", 0.5
                )
            ),
        )
        if attempts >= maximum:
            return (
                "complete"
                if valid_count >= minimum and not needs_repeat
                else "attempt_limit"
            )
        if needs_repeat:
            return "pending"
        if valid_count >= minimum:
            return "complete"
        return "pending"

    def pending_cells(self) -> tuple[ExperimentCell, ...]:
        return tuple(
            cell for cell in self.cells if self.cell_status(cell) == "pending"
        )
