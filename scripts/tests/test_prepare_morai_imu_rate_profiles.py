import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))
import prepare_morai_imu_rate_profiles as profiles


PERIODS = {
    "20hz": 0.05000000074505806,
    "30hz": 0.03333333507180214,
    "50hz": 0.019999999552965164,
}


def _source_document():
    return {
        "IMUList": [
            {
                "m_SensorUniqueID": 7,
                "ic": {
                    "sensorPeriod": 0.019999999552965164,
                    "topic": "/imu/target",
                },
                "name": "target-imu",
            },
            {
                "m_SensorUniqueID": 8,
                "ic": {
                    "sensorPeriod": 0.10000000149011612,
                    "topic": "/imu/untouched",
                },
                "name": "other-imu",
            },
        ],
        "GPSList": [{"UNIQUEID": 21, "sensorPeriod": 0.05}],
        "scene": {"name": "literal fixture", "version": 4},
    }


def _write_source(path: Path, document=None) -> Path:
    path.write_text(
        json.dumps(document or _source_document(), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _changed_pointers(before, after, pointer=""):
    if isinstance(before, dict) and isinstance(after, dict):
        pointers = []
        for key in sorted(set(before) | set(after)):
            child = f"{pointer}/{str(key).replace('~', '~0').replace('/', '~1')}"
            if key not in before or key not in after:
                pointers.append(child)
            else:
                pointers.extend(_changed_pointers(before[key], after[key], child))
        return pointers
    if isinstance(before, list) and isinstance(after, list):
        pointers = []
        for index, (old, new) in enumerate(zip(before, after)):
            pointers.extend(_changed_pointers(old, new, f"{pointer}/{index}"))
        if len(before) != len(after):
            pointers.append(pointer)
        return pointers
    return [] if before == after else [pointer]


def test_prepare_profiles_clones_only_selected_period_with_exact_hashes(tmp_path):
    source = _write_source(tmp_path / "sensor.json")
    original_bytes = source.read_bytes()
    original_document = json.loads(original_bytes)

    manifest = profiles.prepare_profiles(source, tmp_path / "profiles", 7)

    assert source.read_bytes() == original_bytes
    assert manifest["source"]["sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert manifest["selected_imu"] == {
        "json_pointer": "/IMUList/0/ic/sensorPeriod",
        "old_sensor_period": 0.019999999552965164,
        "unique_id": 7,
    }
    assert set(manifest["profiles"]) == {"20hz", "30hz", "50hz"}
    for label, period in PERIODS.items():
        output = tmp_path / "profiles" / f"sensor__imu_{label}.json"
        output_bytes = output.read_bytes()
        prepared = json.loads(output_bytes)
        expected = copy.deepcopy(original_document)
        expected["IMUList"][0]["ic"]["sensorPeriod"] = period
        expected_changes = (
            []
            if period == original_document["IMUList"][0]["ic"]["sensorPeriod"]
            else ["/IMUList/0/ic/sensorPeriod"]
        )

        assert prepared == expected
        assert _changed_pointers(original_document, prepared) == expected_changes
        assert manifest["profiles"][label] == {
            "json_pointer": "/IMUList/0/ic/sensorPeriod",
            "sensor_period": period,
            "sha256": hashlib.sha256(output_bytes).hexdigest(),
            "semantic_diff": {
                "changed_json_pointers": expected_changes,
                "only_selected_sensor_period_changed": True,
            },
        }

    manifest_bytes = (tmp_path / "profiles" / "imu_rate_profiles_manifest.json").read_bytes()
    assert manifest == json.loads(manifest_bytes)


def test_prepare_profiles_rejects_missing_or_duplicate_selected_id(tmp_path):
    missing = _write_source(tmp_path / "missing.json")
    with pytest.raises(ValueError, match="exactly one IMU"):
        profiles.prepare_profiles(missing, tmp_path / "missing-output", 99)

    duplicate_document = _source_document()
    duplicate_document["IMUList"].append(
        {
            "m_SensorUniqueID": 7,
            "ic": {"sensorPeriod": 0.05, "topic": "/imu/duplicate"},
            "name": "duplicate-target",
        }
    )
    duplicate = _write_source(tmp_path / "duplicate.json", duplicate_document)
    with pytest.raises(ValueError, match="exactly one IMU"):
        profiles.prepare_profiles(duplicate, tmp_path / "duplicate-output", 7)


def test_prepare_profiles_accepts_only_verified_legacy_uniqueid_spelling(tmp_path):
    legacy = _source_document()
    for index, imu in enumerate(legacy["IMUList"]):
        imu["UNIQUEID"] = 17 + index
        del imu["m_SensorUniqueID"]
    source = _write_source(tmp_path / "legacy.json", legacy)

    manifest = profiles.prepare_profiles(source, tmp_path / "legacy-output", 17)

    assert manifest["selected_imu"]["unique_id"] == 17


def test_prepare_profiles_fails_closed_for_missing_invalid_or_disagreeing_ids(tmp_path):
    missing = _source_document()
    del missing["IMUList"][0]["m_SensorUniqueID"]
    with pytest.raises(ValueError, match="identifier"):
        profiles.prepare_profiles(
            _write_source(tmp_path / "missing-id.json", missing),
            tmp_path / "missing-id-output",
            7,
        )

    invalid = _source_document()
    invalid["IMUList"][0]["m_SensorUniqueID"] = "7"
    with pytest.raises(ValueError, match="must be an integer"):
        profiles.prepare_profiles(
            _write_source(tmp_path / "invalid-id.json", invalid),
            tmp_path / "invalid-id-output",
            7,
        )

    disagreeing = _source_document()
    disagreeing["IMUList"][0]["UNIQUEID"] = 17
    with pytest.raises(ValueError, match="disagree"):
        profiles.prepare_profiles(
            _write_source(tmp_path / "disagreeing-id.json", disagreeing),
            tmp_path / "disagreeing-id-output",
            7,
        )


def test_checked_in_morai_profile_selects_id_7_without_writing_beside_source(
    tmp_path,
):
    source = (
        Path(__file__).resolve().parents[2]
        / "ad_data/morai/SaveFile/Sensor/25.S4.MolitComp03/"
        "SensorInfo_2023_Hyundai_Ioniq5.json"
    )
    original_bytes = source.read_bytes()

    manifest = profiles.prepare_profiles(source, tmp_path / "profiles", 7)

    assert source.read_bytes() == original_bytes
    assert manifest["selected_imu"]["unique_id"] == 7
    assert (tmp_path / "profiles" / "SensorInfo_2023_Hyundai_Ioniq5__imu_20hz.json").exists()


def test_prepare_profiles_rejects_non_finite_or_invalid_imu_schema(tmp_path):
    non_finite = _source_document()
    non_finite["IMUList"][0]["ic"]["sensorPeriod"] = float("inf")
    source = _write_source(tmp_path / "invalid.json", non_finite)

    with pytest.raises(ValueError, match="finite"):
        profiles.prepare_profiles(source, tmp_path / "output", 7)
    assert not (tmp_path / "output").exists()

    malformed = _write_source(tmp_path / "malformed.json", {"IMUList": []})
    with pytest.raises(ValueError, match="IMUList"):
        profiles.prepare_profiles(malformed, tmp_path / "malformed-output", 7)

    too_large = _source_document()
    too_large["IMUList"][0]["ic"]["sensorPeriod"] = 10**400
    oversized = _write_source(tmp_path / "oversized.json", too_large)
    with pytest.raises(ValueError, match="finite"):
        profiles.prepare_profiles(oversized, tmp_path / "oversized-output", 7)


def test_prepare_profiles_is_idempotent_but_refuses_conflicting_output(tmp_path):
    source = _write_source(tmp_path / "sensor.json")
    output_dir = tmp_path / "profiles"

    first = profiles.prepare_profiles(source, output_dir, 7)
    second = profiles.prepare_profiles(source, output_dir, 7)

    assert second == first
    conflict = output_dir / "sensor__imu_30hz.json"
    conflict.write_text('{"conflict": true}\n', encoding="utf-8")
    manifest_before = (output_dir / "imu_rate_profiles_manifest.json").read_bytes()

    with pytest.raises(FileExistsError, match="conflicting output"):
        profiles.prepare_profiles(source, output_dir, 7)
    assert conflict.read_text(encoding="utf-8") == '{"conflict": true}\n'
    assert (output_dir / "imu_rate_profiles_manifest.json").read_bytes() == manifest_before


def test_prepare_profiles_preserves_source_when_output_directory_contains_it(tmp_path):
    source = _write_source(tmp_path / "sensor__imu_20hz.json")
    original_bytes = source.read_bytes()

    profiles.prepare_profiles(source, tmp_path, 7)

    assert source.read_bytes() == original_bytes
    assert (tmp_path / "sensor__imu_20hz__imu_20hz.json").exists()


def test_main_accepts_explicit_arguments_and_prints_manifest(tmp_path, capsys):
    source = _write_source(tmp_path / "sensor.json")

    result = profiles.main(
        [
            "--source",
            str(source),
            "--output-dir",
            str(tmp_path / "profiles"),
            "--imu-unique-id",
            "7",
        ]
    )

    assert result == 0
    assert json.loads(capsys.readouterr().out)["selected_imu"]["unique_id"] == 7
