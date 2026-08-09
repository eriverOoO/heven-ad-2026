import json
from pathlib import Path

from ad_vehicle_profiling.experiment import (
    DEFAULT_SPEEDS_KPH,
    TrialSummary,
    build_cells,
)
from ad_vehicle_profiling.report import (
    build_report,
    render_heatmap,
    render_label,
    write_report,
)
from ad_vehicle_profiling.storage import RunStore


def _summary(acceleration=1.0):
    return TrialSummary(
        valid=True,
        rejection_reason=None,
        sample_count=10,
        mean_acceleration_mps2=acceleration,
        median_acceleration_mps2=acceleration,
        acceleration_stddev_mps2=0.02,
        acceleration_mad_mps2=0.01,
        minimum_acceleration_mps2=acceleration - 0.1,
        maximum_acceleration_mps2=acceleration + 0.1,
        velocity_derived_acceleration_mps2=acceleration + 0.02,
        cross_check_disagreement_mps2=0.02,
        peak_abs_jerk_mps3=1.5,
        effective_acceleration_mps2=acceleration,
        acceleration_source="simulator_field",
        mean_deceleration_mps2=acceleration,
    )


def _store(tmp_path):
    cells = build_cells()
    store = RunStore.create(
        tmp_path / "run",
        {
            "run_id": "report-test",
            "minimum_valid_trials": 3,
            "maximum_attempts": 7,
        },
        cells=cells,
    )
    for acceleration in (0.9, 1.0, 1.1):
        store.append_trial(cells[0], _summary(acceleration))
    return store


def test_report_emits_every_matrix_cell(tmp_path):
    report = build_report(_store(tmp_path))

    assert report["axes"]["speed_kph"] == list(DEFAULT_SPEEDS_KPH)
    assert report["axes"]["command_percent"] == list(range(0, 101, 10))
    assert report["axes"]["command_kind"] == ["brake"]
    assert len(report["cells"]) == 39 * 11
    assert report["cells"][0]["status"] == "complete"
    assert report["cells"][0]["median_acceleration_mps2"] == 1.0
    assert (
        report["cells"][0][
            "coast_normalized_brake_deceleration_mps2"
        ]
        == 0.0
    )
    assert report["cells"][1]["status"] == "incomplete"
    assert report["cells"][1]["median_acceleration_mps2"] is None
    assert report["progress"]["cell_count"] == 429


def test_report_uses_effective_velocity_derived_acceleration(tmp_path):
    cell = build_cells()[0]
    store = RunStore.create(
        tmp_path / "run",
        {
            "run_id": "velocity-derived",
            "minimum_valid_trials": 1,
            "maximum_attempts": 1,
        },
        cells=(cell,),
    )
    store.append_trial(
        cell,
        TrialSummary(
            **{
                **_summary(0.0).__dict__,
                "velocity_derived_acceleration_mps2": 1.25,
                "cross_check_disagreement_mps2": 1.25,
                "effective_acceleration_mps2": 1.25,
                "acceleration_source": "velocity_derived",
            }
        ),
    )

    reported = build_report(store)["cells"][0]

    assert reported["median_acceleration_mps2"] == 1.25
    assert reported["median_simulator_acceleration_mps2"] == 0.0
    assert reported["acceleration_source"] == "velocity_derived"


def test_svg_escapes_labels_and_contains_axis_units(tmp_path):
    profile = build_report(_store(tmp_path))
    output = tmp_path / "heatmap.svg"

    render_heatmap(profile, "brake", output)

    text = output.read_text()
    assert "km/h" in text
    assert "m/s²" in text
    assert "Brake command (%)" in text
    assert render_label("<unsafe>") == "&lt;unsafe&gt;"


def test_write_report_creates_json_csv_and_configured_heatmap(tmp_path):
    store = _store(tmp_path)

    outputs = write_report(store.run_directory)

    assert outputs["json"] == store.run_directory / "profile.json"
    assert outputs["csv"] == store.run_directory / "profile.csv"
    assert outputs["brake_heatmap"].is_file()
    assert outputs["brake_coast_normalized_heatmap"].is_file()
    document = json.loads(outputs["json"].read_text())
    assert document["schema_version"] == 2
    assert len(document["manifest_sha256"]) == 64
    assert len(document["cells"]) == 429


def test_report_output_is_deterministic_for_same_store(tmp_path):
    store = _store(tmp_path)

    first = json.dumps(build_report(store), sort_keys=True)
    second = json.dumps(build_report(RunStore.resume(store.run_directory)), sort_keys=True)

    assert first == second
