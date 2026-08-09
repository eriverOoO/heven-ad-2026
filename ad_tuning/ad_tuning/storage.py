from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import yaml


def is_reusable_trial(
    row: Mapping[str, object],
    route_digest: str,
    search_space_version: str,
) -> bool:
    """Return whether a persisted row is a valid sampler observation."""
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        return False
    infrastructure_failure = any(
        bool(metrics.get(name))
        for name in ("aborted", "reset_failed", "disconnected")
    )
    return (
        row.get("route_digest") == route_digest
        and row.get("status") == "completed"
        and row.get("search_space_version") == search_space_version
        and not infrastructure_failure
    )


class TrialStorage:
    def __init__(
        self,
        output_dir: Path,
        algorithm: str = "profile_stanley",
        worker_id: str = "local",
    ) -> None:
        if not algorithm or not algorithm.replace("_", "").isalnum():
            raise ValueError("algorithm must be a safe file-name component")
        safe_worker_id = (
            worker_id.replace("_", "").replace("-", "").replace(".", "")
        )
        if not worker_id or not safe_worker_id.isalnum():
            raise ValueError("worker_id must be a safe file-name component")
        self.worker_id = worker_id
        self.output_root = output_dir
        self.output_dir = output_dir / "workers" / worker_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trials_path = self.output_dir / f"{algorithm}_trials.jsonl"
        self.best_path = self.output_dir / f"best_{algorithm}_optuna.yaml"
        self.best_metadata_path = (
            self.output_dir / f"best_{algorithm}_optuna.json"
        )
        self.trajectory_dir = self.output_dir / f"{algorithm}_trajectories"
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        self.pending_dir = self.output_dir / "pending_database_results"
        self.pending_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[dict[str, object]]:
        if not self.trials_path.exists():
            return []
        rows = []
        for line in self.trials_path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
        return rows

    def append(self, row: Mapping[str, object]) -> None:
        with self.trials_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def write_best(
        self,
        row: Mapping[str, object],
        fixed_parameters: Mapping[str, float],
    ) -> None:
        parameters = dict(row["parameters"])
        parameters.update(fixed_parameters)
        document = {"ad_planner": {"ros__parameters": parameters}}
        temporary = self.best_path.with_suffix(".yaml.tmp")
        temporary.write_text(
            yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
        )
        temporary.replace(self.best_path)
        metadata = {
            "trial": row["trial"],
            "feasible": row["feasible"],
            "cost": row["cost"],
            "route_digest": row["route_digest"],
            "experiment_fingerprint": row.get("experiment_fingerprint", ""),
            "worker_id": row.get("worker_id", self.worker_id),
            "trajectory": row.get("trajectory", ""),
            "metrics": row["metrics"],
            "parameters": parameters,
        }
        metadata_temporary = self.best_metadata_path.with_suffix(".json.tmp")
        metadata_temporary.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        metadata_temporary.replace(self.best_metadata_path)

    def write_trajectory(
        self, trial: int, samples: Sequence[tuple[object, ...]]
    ) -> str:
        path = self.trajectory_dir / f"trial_{trial:04d}.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                (
                    "sim_elapsed_s",
                    "wall_elapsed_s",
                    "sim_time_s",
                    "header_time_s",
                    "base_x_m",
                    "base_y_m",
                    "control_x_m",
                    "control_y_m",
                    "speed_mps",
                    "target_speed_mps",
                    "throttle",
                    "brake",
                    "steering_normalized",
                    "actual_steering_rad",
                    "lateral_acceleration_mps2",
                    "front_cte_m",
                    "rear_cte_m",
                    "progress_m",
                    "segment",
                    "local_planner_active",
                    "local_planner_failure",
                    "dwa_failure_reason",
                    "collision_count",
                )
            )
            for sample in samples:
                writer.writerow(sample)
        return str(path.resolve())

    def stage_pending(self, row: Mapping[str, object]) -> Path:
        """Persist a result before Optuna commits it to the remote database."""
        trial = int(row["trial"])
        path = self.pending_dir / f"trial_{trial:08d}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(dict(row), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return path

    def acknowledge_pending(self, trial: int) -> None:
        path = self.pending_dir / f"trial_{int(trial):08d}.json"
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def pending_results(self) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        for path in sorted(self.pending_dir.glob("trial_*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if isinstance(value, dict):
                results.append(value)
        return results
