#!/usr/bin/env python3
"""Deterministically reanalyze parallel closed-loop ESKF A/B artifacts."""

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


BASELINE = "baseline"
PRODUCTION = "production_bias"
CANDIDATES = (BASELINE, PRODUCTION)
PULSE_PHASES = (
    "closed_loop_release",
    "closed_loop_track",
    "closed_loop_stop",
)
VECTOR_COLUMNS = (
    "body_velocity_error_xyz",
    "world_velocity_error_xyz",
    "position_error_xyz",
    "attitude_error_rpy_rad",
)


class AnalysisError(RuntimeError):
    """Raised when an input artifact cannot satisfy the analysis contract."""


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_number(value, context):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise AnalysisError(f"{context} contains a non-finite value")
    return result


def _vector(value, context):
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError) as error:
            raise AnalysisError(f"{context} is not a valid vector") from error
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise AnalysisError(f"{context} must contain exactly three values")
    return tuple(
        _finite_number(component, f"{context}[{index}]")
        for index, component in enumerate(value)
    )


def _load_json(path):
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisError(f"cannot read JSON source {path}: {error}") from error
    if not isinstance(value, dict):
        raise AnalysisError(f"JSON source {path} must contain an object")
    return value


def _load_jsonl(path):
    rows = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise AnalysisError(
                        f"invalid JSON at {path}:{line_number}: {error}"
                    ) from error
                if not isinstance(row, dict):
                    raise AnalysisError(
                        f"record at {path}:{line_number} must be an object"
                    )
                rows.append(row)
    except OSError as error:
        raise AnalysisError(f"cannot read JSONL source {path}: {error}") from error
    if not rows:
        raise AnalysisError(f"JSONL source {path} is empty")
    return rows


def _load_csv(path):
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"candidate", "phase", *VECTOR_COLUMNS}
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise AnalysisError(
                    f"CSV source {path} is missing columns: {sorted(missing)}"
                )
            rows = list(reader)
    except (OSError, csv.Error) as error:
        raise AnalysisError(f"cannot read CSV source {path}: {error}") from error
    if not rows:
        raise AnalysisError(f"CSV source {path} is empty")
    return rows


def _source(path, row_count):
    return {
        "path": str(path.resolve()),
        "row_count": row_count,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _rmse(values):
    if not values:
        raise AnalysisError("RMSE requires at least one selected sample")
    return math.sqrt(math.fsum(value * value for value in values) / len(values))


def _candidate_metrics(rows, candidate):
    selected = [
        row
        for row in rows
        if row["candidate"] == candidate and row["phase"] != "initialization"
    ]
    if not selected:
        raise AnalysisError(
            f"candidate {candidate!r} has no rows after initialization exclusion"
        )
    parsed = {
        column: [
            _vector(row[column], f"{candidate}.{column}") for row in selected
        ]
        for column in VECTOR_COLUMNS
    }
    body = parsed["body_velocity_error_xyz"]
    attitude = parsed["attitude_error_rpy_rad"]
    return {
        "attitude_rmse_degrees": {
            "pitch": math.degrees(_rmse([value[1] for value in attitude])),
            "roll": math.degrees(_rmse([value[0] for value in attitude])),
            "yaw": math.degrees(_rmse([value[2] for value in attitude])),
        },
        "body_velocity_rmse_mps": {
            "x": _rmse([value[0] for value in body]),
            "y": _rmse([value[1] for value in body]),
            "z": _rmse([value[2] for value in body]),
        },
        "sample_count": len(selected),
        "world_z_velocity_rmse_mps": _rmse(
            [value[2] for value in parsed["world_velocity_error_xyz"]]
        ),
        "z_position_rmse_m": _rmse(
            [value[2] for value in parsed["position_error_xyz"]]
        ),
    }


def _ordered_truth(raw_rows):
    truth = [row for row in raw_rows if row.get("stream") == "truth"]
    if not truth:
        raise AnalysisError("raw artifact contains no truth records")
    try:
        truth.sort(key=lambda row: int(row["receipt_monotonic_ns"]))
    except (KeyError, TypeError, ValueError) as error:
        raise AnalysisError("truth record has invalid receipt_monotonic_ns") from error
    return truth


def _cumulative_travel(truth, context):
    positions = [
        _vector(row.get("position_xyz"), f"{context}.position_xyz")
        for row in truth
    ]
    return math.fsum(
        math.dist(first, second)
        for first, second in zip(positions, positions[1:])
    )


def _trajectory(raw_rows, actor_id):
    truth = _ordered_truth(raw_rows)
    pulse_truth = [row for row in truth if row.get("phase") in PULSE_PHASES]
    if not pulse_truth:
        raise AnalysisError("raw artifact contains no closed-loop pulse truth")
    speeds = [
        math.sqrt(math.fsum(value * value for value in _vector(
            row.get("world_velocity_xyz"), "truth.world_velocity_xyz"
        )))
        for row in truth
    ]
    command_throttles = [
        _finite_number(row.get("throttle"), "command.throttle")
        for row in raw_rows
        if row.get("stream") == "command"
    ]
    if not command_throttles:
        raise AnalysisError("raw artifact contains no command records")
    nonself_collisions = sorted(
        {
            str(object_id)
            for row in truth
            for object_id in row.get("collision_object_ids", [])
            if str(object_id) != actor_id
        }
    )
    durations = []
    for row in truth:
        try:
            receipt = int(row["receipt_monotonic_ns"])
            start = int(row["rpc_start_monotonic_ns"])
        except (KeyError, TypeError, ValueError) as error:
            raise AnalysisError("truth record has invalid RPC timestamps") from error
        duration = receipt - start
        if duration < 0:
            raise AnalysisError("truth RPC duration is negative")
        durations.append(duration)
    return {
        "all_persisted_truth_cumulative_travel_m": _cumulative_travel(
            truth, "truth"
        ),
        "maximum_command_throttle": max(command_throttles),
        "maximum_truth_speed_mps": max(speeds),
        "nonself_collision_ids": nonself_collisions,
        "pulse_phase_cumulative_travel_m": _cumulative_travel(
            pulse_truth, "pulse_truth"
        ),
        "truth_rpc_maximum_duration_ns": max(durations),
    }


def _run_result(run_dir):
    run_dir = run_dir.resolve()
    aligned_path = run_dir / "aligned.csv"
    raw_path = run_dir / "raw.jsonl"
    manifest_path = run_dir / "manifest.json"
    for path in (aligned_path, raw_path, manifest_path):
        if not path.is_file():
            raise AnalysisError(f"required source does not exist: {path}")
    aligned_rows = _load_csv(aligned_path)
    raw_rows = _load_jsonl(raw_path)
    manifest = _load_json(manifest_path)
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise AnalysisError(f"manifest {manifest_path} has no run_id")
    actor_id = str(manifest.get("actor_identity", {}).get("id_value", ""))
    if not actor_id:
        raise AnalysisError(f"manifest {manifest_path} has no actor identity")
    trajectory = _trajectory(raw_rows, actor_id)
    control_safety = manifest.get("control_mode_safety", {})
    initial_mode = manifest.get("initial_control_mode")
    final_mode = manifest.get("final_control_mode")
    safety = {
        "cleanup_status": manifest.get("cleanup_status"),
        "cleanup_verified": (
            manifest.get("cleanup_status") == "verified"
            and control_safety.get("cleanup_stable_stop_status") == "verified"
        ),
        "control_mode_safety": control_safety,
        "final_control_mode": final_mode,
        "initial_control_mode": initial_mode,
        "keyboard_to_keyboard_restoration_verified": (
            initial_mode == "VEHICLE_CONTROL_KEYBOARD"
            and final_mode == "VEHICLE_CONTROL_KEYBOARD"
            and control_safety.get("restoration_status") == "verified"
            and control_safety.get("post_restore_stop_status") == "verified"
        ),
        "prohibited_rpcs_used": bool(manifest.get("prohibited_rpcs_used", False)),
        "status": manifest.get("status"),
    }
    return {
        "manifest_provenance": {
            "candidate_parameters": manifest.get("candidate_parameters"),
            "repository": manifest.get("repository"),
            "schema_version": manifest.get("schema_version"),
        },
        "metrics": {
            candidate: _candidate_metrics(aligned_rows, candidate)
            for candidate in CANDIDATES
        },
        "run_id": run_id,
        "safety": safety,
        "sources": {
            "aligned_csv": _source(aligned_path, len(aligned_rows)),
            "manifest_json": _source(manifest_path, 1),
            "raw_jsonl": _source(raw_path, len(raw_rows)),
        },
        "trajectory": trajectory,
    }


def _mean_candidate_metrics(runs, candidate):
    values = [run["metrics"][candidate] for run in runs]
    return {
        "attitude_rmse_degrees": {
            axis: statistics.fmean(
                value["attitude_rmse_degrees"][axis] for value in values
            )
            for axis in ("pitch", "roll", "yaw")
        },
        "body_velocity_rmse_mps": {
            axis: statistics.fmean(
                value["body_velocity_rmse_mps"][axis] for value in values
            )
            for axis in ("x", "y", "z")
        },
        "world_z_velocity_rmse_mps": statistics.fmean(
            value["world_z_velocity_rmse_mps"] for value in values
        ),
        "z_position_rmse_m": statistics.fmean(
            value["z_position_rmse_m"] for value in values
        ),
    }


def _relative_change(production, baseline):
    def change(candidate_value, baseline_value):
        if baseline_value == 0.0:
            return None
        return 100.0 * (candidate_value / baseline_value - 1.0)

    return {
        "attitude_rmse_degrees": {
            axis: change(
                production["attitude_rmse_degrees"][axis],
                baseline["attitude_rmse_degrees"][axis],
            )
            for axis in ("pitch", "roll", "yaw")
        },
        "body_velocity_rmse_mps": {
            axis: change(
                production["body_velocity_rmse_mps"][axis],
                baseline["body_velocity_rmse_mps"][axis],
            )
            for axis in ("x", "y", "z")
        },
        "world_z_velocity_rmse_mps": change(
            production["world_z_velocity_rmse_mps"],
            baseline["world_z_velocity_rmse_mps"],
        ),
        "z_position_rmse_m": change(
            production["z_position_rmse_m"], baseline["z_position_rmse_m"]
        ),
    }


def analyze(run_directories, *, generated_at=None):
    runs = [_run_result(Path(path)) for path in run_directories]
    runs.sort(key=lambda run: run["run_id"])
    run_ids = [run["run_id"] for run in runs]
    if len(set(run_ids)) != len(run_ids):
        raise AnalysisError("run_id values must be unique")
    mean_metrics = {
        candidate: _mean_candidate_metrics(runs, candidate)
        for candidate in CANDIDATES
    }
    maximum_speeds = [
        run["trajectory"]["maximum_truth_speed_mps"] for run in runs
    ]
    all_travel = [
        run["trajectory"]["all_persisted_truth_cumulative_travel_m"]
        for run in runs
    ]
    pulse_travel = [
        run["trajectory"]["pulse_phase_cumulative_travel_m"] for run in runs
    ]
    throttles = [
        run["trajectory"]["maximum_command_throttle"] for run in runs
    ]
    speed_mean = statistics.fmean(maximum_speeds)
    speed_cv = (
        0.0
        if speed_mean == 0.0
        else 100.0 * statistics.pstdev(maximum_speeds) / speed_mean
    )
    script_path = Path(__file__).resolve()
    return {
        "mean_of_per_run_rmse": mean_metrics,
        "metric_contract": {
            "candidate_run_aggregation": "arithmetic mean of per-run RMSE",
            "coefficient_of_variation": (
                "100 * population_standard_deviation / arithmetic_mean (ddof=0)"
            ),
            "rmse_formula": "sqrt(sum(error_i^2) / N)",
            "row_selection": "phase != 'initialization'",
        },
        "production_relative_change_percent_vs_baseline": _relative_change(
            mean_metrics[PRODUCTION], mean_metrics[BASELINE]
        ),
        "provenance": {
            "deterministic": True,
            "generated_at": generated_at,
            "generator_path": str(script_path),
            "generator_sha256": _sha256(script_path),
            "run_order": run_ids,
            "source_contract": "manifest.json + aligned.csv + raw.jsonl",
        },
        "runs": runs,
        "safety": {
            "completed_runs": sum(
                run["safety"]["status"] == "complete" for run in runs
            ),
            "nonself_collision_runs": sum(
                bool(run["trajectory"]["nonself_collision_ids"])
                for run in runs
            ),
            "prohibited_rpc_runs": sum(
                run["safety"]["prohibited_rpcs_used"] for run in runs
            ),
            "verified_cleanup_runs": sum(
                run["safety"]["cleanup_verified"] for run in runs
            ),
            "verified_keyboard_to_keyboard_restoration_runs": sum(
                run["safety"]["keyboard_to_keyboard_restoration_verified"]
                for run in runs
            ),
        },
        "schema_version": 1,
        "trajectory": {
            "all_persisted_truth_cumulative_travel_m_range": [
                min(all_travel),
                max(all_travel),
            ],
            "maximum_command_throttle_range": [min(throttles), max(throttles)],
            "maximum_truth_speed_mps_range": [
                min(maximum_speeds),
                max(maximum_speeds),
            ],
            "maximum_truth_speed_population_cv_percent": speed_cv,
            "pulse_phase_cumulative_travel_m_range": [
                min(pulse_travel),
                max(pulse_travel),
            ],
            "pulse_phases": list(PULSE_PHASES),
        },
        "truth_rpc_duration_ns": {
            "overall_maximum": max(
                run["trajectory"]["truth_rpc_maximum_duration_ns"]
                for run in runs
            ),
            "per_run_maximum": {
                run["run_id"]: run["trajectory"][
                    "truth_rpc_maximum_duration_ns"
                ]
                for run in runs
            },
        },
    }


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Reanalyze closed-loop baseline/production_bias ESKF run directories"
        )
    )
    parser.add_argument("run_directories", nargs="+")
    parser.add_argument(
        "--generated-at",
        help="ISO-8601 generation timestamp to persist in provenance",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    try:
        result = analyze(
            args.run_directories,
            generated_at=args.generated_at,
        )
        serialized = json.dumps(
            result,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    except (AnalysisError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
