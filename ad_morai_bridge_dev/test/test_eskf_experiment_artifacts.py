import hashlib
import json
import pytest

from ad_morai_bridge_dev.eskf_experiment import artifacts as artifact_module
from ad_morai_bridge_dev.eskf_experiment.artifacts import (
    JsonlRecorder,
    create_run_artifacts,
    file_manifest,
    write_json,
)


def test_run_artifacts_require_explicit_absolute_ad_data_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("AD_DATA_DIR", raising=False)
    with pytest.raises(ValueError, match="AD_DATA_DIR"):
        create_run_artifacts("run-001")

    monkeypatch.setenv("AD_DATA_DIR", "relative/data")
    with pytest.raises(ValueError, match="absolute"):
        create_run_artifacts("run-001")

    monkeypatch.setenv("AD_DATA_DIR", str(tmp_path))
    artifacts = create_run_artifacts("run-001")
    assert artifacts.run_directory == tmp_path / "experiments" / "eskf" / "run-001"


@pytest.mark.parametrize("run_id", ("../escape", "a/b", ".", "", "run id"))
def test_run_artifacts_refuse_paths_outside_configured_root(monkeypatch, tmp_path, run_id):
    monkeypatch.setenv("AD_DATA_DIR", str(tmp_path))

    with pytest.raises(ValueError, match="run_id"):
        create_run_artifacts(run_id)

    assert not (tmp_path.parent / "escape").exists()


def test_run_artifacts_refuse_symlink_escape(monkeypatch, tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / "experiments").symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("AD_DATA_DIR", str(tmp_path))

    with pytest.raises(ValueError, match="outside AD_DATA_DIR"):
        create_run_artifacts("run-001")


def test_artifact_contract_creates_only_the_four_named_run_files(monkeypatch, tmp_path):
    monkeypatch.setenv("AD_DATA_DIR", str(tmp_path))

    artifacts = create_run_artifacts("20260802T120000Z")

    assert artifacts.manifest == artifacts.run_directory / "manifest.json"
    assert artifacts.raw == artifacts.run_directory / "raw.jsonl"
    assert artifacts.aligned == artifacts.run_directory / "aligned.csv"
    assert artifacts.summary == artifacts.run_directory / "summary.json"
    assert artifacts.run_directory.is_dir()
    assert list(artifacts.run_directory.iterdir()) == []


def test_run_artifacts_refuse_to_reuse_an_existing_run_id(monkeypatch, tmp_path):
    monkeypatch.setenv("AD_DATA_DIR", str(tmp_path))
    first = create_run_artifacts("unique-run")

    with pytest.raises(FileExistsError):
        create_run_artifacts("unique-run")

    assert first.run_directory.is_dir()
    assert list(first.run_directory.iterdir()) == []


@pytest.mark.parametrize("run_id", ("run-001", "2026.08.02_ab", "A" * 128))
def test_public_run_id_validator_returns_valid_string(run_id):
    assert artifact_module.validate_run_id(run_id) == run_id


@pytest.mark.parametrize("run_id", (None, True, 123, "../escape", "A" * 129))
def test_public_run_id_validator_rejects_invalid_values(run_id):
    with pytest.raises(ValueError, match="run_id"):
        artifact_module.validate_run_id(run_id)


def test_jsonl_recorder_preserves_monotonic_ros_and_simulator_timestamps(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("AD_DATA_DIR", str(tmp_path))
    artifacts = create_run_artifacts("run-raw")

    with JsonlRecorder(artifacts.raw) as recorder:
        recorder.write(
            "truth",
            {
                "receipt_monotonic_ns": 1_234_567_890,
                "rpc_start_monotonic_ns": 1_234_560_000,
                "header_stamp_ns": None,
                "simulator_timestamp": 998877,
                "position_xyz": [1.0, 2.0, 3.0],
            },
        )

    record = json.loads(artifacts.raw.read_text(encoding="utf-8"))
    assert record == {
        "schema_version": 1,
        "stream": "truth",
        "receipt_monotonic_ns": 1_234_567_890,
        "rpc_start_monotonic_ns": 1_234_560_000,
        "header_stamp_ns": None,
        "simulator_timestamp": 998877,
        "position_xyz": [1.0, 2.0, 3.0],
    }


def test_jsonl_recorder_rejects_missing_or_non_integer_receipt_time(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("AD_DATA_DIR", str(tmp_path))
    artifacts = create_run_artifacts("run-invalid")

    with JsonlRecorder(artifacts.raw) as recorder:
        with pytest.raises(ValueError, match="receipt_monotonic_ns"):
            recorder.write("truth", {"simulator_timestamp": 1})
        with pytest.raises(ValueError, match="receipt_monotonic_ns"):
            recorder.write("truth", {"receipt_monotonic_ns": 1.5})


@pytest.mark.parametrize("reserved_field", ("schema_version", "stream"))
def test_jsonl_recorder_rejects_reserved_field_overrides(
    monkeypatch, tmp_path, reserved_field
):
    monkeypatch.setenv("AD_DATA_DIR", str(tmp_path))
    artifacts = create_run_artifacts(f"reserved-{reserved_field}")

    with JsonlRecorder(artifacts.raw) as recorder:
        with pytest.raises(ValueError, match="reserved"):
            recorder.write(
                "truth",
                {
                    "receipt_monotonic_ns": 1,
                    reserved_field: "caller-controlled",
                },
            )

    assert artifacts.raw.read_text(encoding="utf-8") == ""


def test_file_manifest_is_sorted_and_hashes_exact_file_bytes(tmp_path):
    alpha = tmp_path / "alpha.yaml"
    beta = tmp_path / "beta.json"
    alpha.write_bytes(b"gravity: 9.80665\n")
    beta.write_bytes(b'{"candidate":"baseline"}\n')

    manifest = file_manifest({"z-config": beta, "a-config": alpha})

    assert list(manifest) == ["a-config", "z-config"]
    assert manifest["a-config"] == {
        "path": str(alpha.resolve()),
        "sha256": hashlib.sha256(b"gravity: 9.80665\n").hexdigest(),
        "size_bytes": 17,
    }
    assert manifest["z-config"]["sha256"] == hashlib.sha256(
        b'{"candidate":"baseline"}\n'
    ).hexdigest()


def test_write_json_is_deterministic_and_refuses_nonfinite_values(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("AD_DATA_DIR", str(tmp_path))
    first = create_run_artifacts("first").manifest
    second = create_run_artifacts("second").manifest

    write_json(first, {"z": 2, "a": {"value": 1}})
    write_json(second, {"a": {"value": 1}, "z": 2})

    assert first.read_bytes() == second.read_bytes()
    assert first.read_text(encoding="utf-8") == (
        '{\n  "a": {\n    "value": 1\n  },\n  "z": 2\n}\n'
    )
    with pytest.raises(ValueError, match="JSON"):
        write_json(create_run_artifacts("invalid").summary, {"value": float("nan")})


def test_write_json_requires_ad_data_dir_even_for_an_absolute_target(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("AD_DATA_DIR", raising=False)

    with pytest.raises(ValueError, match="AD_DATA_DIR"):
        write_json(tmp_path / "summary.json", {"ok": True})


def test_writers_refuse_targets_outside_run_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("AD_DATA_DIR", str(tmp_path))
    create_run_artifacts("run-contained")

    with pytest.raises(ValueError, match="run directory"):
        JsonlRecorder(tmp_path / "raw.jsonl")
    with pytest.raises(ValueError, match="run directory"):
        write_json(tmp_path / "summary.json", {"ok": True})
