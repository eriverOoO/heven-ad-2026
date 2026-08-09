import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "analyze_eskf_closed_loop_ab.py"
CSV_FIELDS = (
    "candidate",
    "phase",
    "body_velocity_error_xyz",
    "world_velocity_error_xyz",
    "position_error_xyz",
    "attitude_error_rpy_rad",
)


def _candidate_row(candidate, phase, scale, production=False):
    if production:
        body = (scale, 2.0 * scale, scale)
        world = (0.0, 0.0, scale)
        position = (0.0, 0.0, 3.0 * scale)
        attitude = (0.05 * scale, 0.1 * scale, 0.15 * scale)
    else:
        body = (3.0 * scale, 4.0 * scale, scale)
        world = (0.0, 0.0, 2.0 * scale)
        position = (0.0, 0.0, 6.0 * scale)
        attitude = (0.1 * scale, 0.2 * scale, 0.3 * scale)
    return {
        "candidate": candidate,
        "phase": phase,
        "body_velocity_error_xyz": repr(body),
        "world_velocity_error_xyz": repr(world),
        "position_error_xyz": repr(position),
        "attitude_error_rpy_rad": repr(attitude),
    }


def _write_run(parent, run_id, scale, *, nonself_collision=False, nan_truth=False):
    run_dir = parent / run_id
    run_dir.mkdir()
    rows = [
        {
            **_candidate_row("baseline", "initialization", 100.0),
            "body_velocity_error_xyz": repr((0.0, 0.0, 999.0)),
        },
        _candidate_row("baseline", "closed_loop_track", scale),
        {
            **_candidate_row("baseline", "closed_loop_stop", scale),
            "body_velocity_error_xyz": repr((0.0, 0.0, 3.0 * scale)),
            "world_velocity_error_xyz": repr((0.0, 0.0, 4.0 * scale)),
            "position_error_xyz": repr((0.0, 0.0, 8.0 * scale)),
        },
        _candidate_row(
            "production_bias", "closed_loop_track", scale, production=True
        ),
        {
            **_candidate_row(
                "production_bias", "closed_loop_stop", scale, production=True
            ),
            "position_error_xyz": repr((0.0, 0.0, 4.0 * scale)),
        },
    ]
    aligned = run_dir / "aligned.csv"
    with aligned.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    positions = (0.0, 1.0, 3.0, 6.0, 10.0)
    speeds = (0.0, 1.0, 2.0, 1.0, 0.0)
    phases = (
        "settle",
        "closed_loop_release",
        "closed_loop_track",
        "closed_loop_stop",
        "client_cleanup_stop",
    )
    raw_rows = []
    for index, (position, speed, phase) in enumerate(
        zip(positions, speeds, phases)
    ):
        collision_ids = ["Ego"]
        if nonself_collision and index == 2:
            collision_ids.append("Cone")
        velocity_x = float("nan") if nan_truth and index == 2 else speed * scale
        raw_rows.append(
            {
                "schema_version": 1,
                "stream": "truth",
                "phase": phase,
                "receipt_monotonic_ns": 1000 + index * 100,
                "rpc_start_monotonic_ns": 990 + index * 100,
                "position_xyz": [position * scale, 0.0, 0.0],
                "world_velocity_xyz": [velocity_x, 0.0, 0.0],
                "collision_object_ids": collision_ids,
            }
        )
    raw_rows.append(
        {
            "schema_version": 1,
            "stream": "command",
            "phase": "closed_loop_track",
            "receipt_monotonic_ns": 1300,
            "throttle": 0.1 * scale,
            "brake": 0.0,
            "steer": 0.0,
        }
    )
    raw = run_dir / "raw.jsonl"
    raw.write_text(
        "".join(json.dumps(row) + "\n" for row in raw_rows),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "complete",
        "cleanup_status": "verified",
        "initial_control_mode": "VEHICLE_CONTROL_KEYBOARD",
        "final_control_mode": "VEHICLE_CONTROL_KEYBOARD",
        "prohibited_rpcs_used": False,
        "actor_identity": {"id_value": "Ego"},
        "candidate_parameters": {
            "baseline": {
                "parameters": {"initial_imu_acc_bias_covariance": 0.0}
            },
            "production_bias": {
                "parameters": {"initial_imu_acc_bias_covariance": 0.01}
            },
        },
        "control_mode_safety": {
            "pre_waveform_stable_stop_status": "verified",
            "cleanup_stable_stop_status": "verified",
            "restoration_status": "verified",
            "post_restore_stop_status": "verified",
            "last_brake_rpc_status": "verified",
        },
        "repository": {"head": "abc", "worktree_sha256": "def"},
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return run_dir


def _run(*run_dirs, generated_at=None):
    arguments = [sys.executable, str(SCRIPT)]
    if generated_at is not None:
        arguments.extend(["--generated-at", generated_at])
    arguments.extend(str(path) for path in run_dirs)
    return subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        text=True,
    )


def test_generator_reanalyzes_sources_with_explicit_deterministic_provenance(
    tmp_path,
):
    run_b = _write_run(tmp_path, "run_b", 2.0)
    run_a = _write_run(tmp_path, "run_a", 1.0, nonself_collision=True)

    first = _run(run_b, run_a)
    second = _run(run_a, run_b)

    assert first.returncode == 0, first.stderr
    assert first.stderr == ""
    assert first.stdout == second.stdout
    result = json.loads(first.stdout, parse_constant=lambda value: 1 / 0)

    assert result["schema_version"] == 1
    assert result["provenance"]["generated_at"] is None
    assert result["provenance"]["run_order"] == ["run_a", "run_b"]
    assert result["metric_contract"] == {
        "candidate_run_aggregation": "arithmetic mean of per-run RMSE",
        "coefficient_of_variation": (
            "100 * population_standard_deviation / arithmetic_mean (ddof=0)"
        ),
        "rmse_formula": "sqrt(sum(error_i^2) / N)",
        "row_selection": "phase != 'initialization'",
    }

    run_a_result = result["runs"][0]
    assert run_a_result["run_id"] == "run_a"
    assert run_a_result["sources"]["aligned_csv"]["row_count"] == 5
    assert run_a_result["sources"]["raw_jsonl"]["row_count"] == 6
    assert run_a_result["sources"]["manifest_json"]["row_count"] == 1
    aligned_path = run_a / "aligned.csv"
    assert run_a_result["sources"]["aligned_csv"]["path"] == str(
        aligned_path.resolve()
    )
    assert run_a_result["sources"]["aligned_csv"]["sha256"] == hashlib.sha256(
        aligned_path.read_bytes()
    ).hexdigest()

    baseline = run_a_result["metrics"]["baseline"]
    production = run_a_result["metrics"]["production_bias"]
    assert baseline["sample_count"] == 2
    assert baseline["body_velocity_rmse_mps"]["z"] == math.sqrt(5.0)
    assert baseline["world_z_velocity_rmse_mps"] == math.sqrt(10.0)
    assert baseline["z_position_rmse_m"] == math.sqrt(50.0)
    assert baseline["attitude_rmse_degrees"]["pitch"] == math.degrees(0.2)
    assert production["body_velocity_rmse_mps"] == {
        "x": 1.0,
        "y": 2.0,
        "z": 1.0,
    }

    mean_metrics = result["mean_of_per_run_rmse"]
    assert mean_metrics["baseline"]["body_velocity_rmse_mps"]["z"] == (
        math.sqrt(5.0) + math.sqrt(20.0)
    ) / 2.0
    assert mean_metrics["production_bias"]["z_position_rmse_m"] == (
        math.sqrt(12.5) + math.sqrt(50.0)
    ) / 2.0

    assert result["trajectory"] == {
        "all_persisted_truth_cumulative_travel_m_range": [10.0, 20.0],
        "maximum_command_throttle_range": [0.1, 0.2],
        "maximum_truth_speed_mps_range": [2.0, 4.0],
        "maximum_truth_speed_population_cv_percent": 100.0 / 3.0,
        "pulse_phase_cumulative_travel_m_range": [5.0, 10.0],
        "pulse_phases": [
            "closed_loop_release",
            "closed_loop_track",
            "closed_loop_stop",
        ],
    }
    assert result["truth_rpc_duration_ns"] == {
        "overall_maximum": 10,
        "per_run_maximum": {"run_a": 10, "run_b": 10},
    }
    assert result["safety"] == {
        "completed_runs": 2,
        "nonself_collision_runs": 1,
        "prohibited_rpc_runs": 0,
        "verified_cleanup_runs": 2,
        "verified_keyboard_to_keyboard_restoration_runs": 2,
    }

    stamped = _run(run_a, run_b, generated_at="2026-08-02T12:45:15+09:00")
    assert stamped.returncode == 0, stamped.stderr
    assert json.loads(stamped.stdout)["provenance"]["generated_at"] == (
        "2026-08-02T12:45:15+09:00"
    )


def test_generator_rejects_nonfinite_truth_instead_of_emitting_nan(tmp_path):
    run_dir = _write_run(tmp_path, "bad_run", 1.0, nan_truth=True)

    completed = _run(run_dir)

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "non-finite" in completed.stderr
