from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
VERIFIER = REPOSITORY_ROOT / "scripts" / "verify_ad_data.py"


def _write_valid_repository(root: Path) -> Path:
    data_root = root / "ad_data"
    network = data_root / "morai" / "SaveFile" / "Network" / "NetworkInfo.json"
    network.parent.mkdir(parents=True)
    network.write_text(
        json.dumps({"host": "127.0.0.1", "port": 2368}),
        encoding="utf-8",
    )

    scenario = data_root / "morai" / "SaveFile" / "Scenario" / "checkpoint1.json"
    scenario.parent.mkdir(parents=True)
    scenario.write_text(json.dumps({"version": "1.0"}), encoding="utf-8")

    lfs_payload = data_root / "morai" / "fixtures" / "sample.bin"
    lfs_payload.parent.mkdir(parents=True)
    lfs_payload.write_bytes(b"binary sensor fixture\x00" * 100)

    manifest = {
        "schema_version": 1,
        "collections": [
            {
                "id": "morai_savefile",
                "path": "morai/SaveFile",
                "kind": "source",
                "source": "MORAI 25.S4 SaveFile",
                "license": "MORAI internal competition asset",
            },
            {
                "id": "sensor_fixture",
                "path": "morai/fixtures",
                "kind": "fixture",
                "source": "MORAI 25.S4 SensorData",
                "license": "MORAI internal competition asset",
            },
        ],
        "lfs_paths": ["morai/fixtures/sample.bin"],
    }
    manifest_path = data_root / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    checksum_paths = sorted(
        path for path in data_root.rglob("*") if path.is_file()
    )
    checksum_lines = []
    for path in checksum_paths:
        relative = path.relative_to(data_root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checksum_lines.append(f"{digest}  {relative}\n")
    (data_root / "SHA256SUMS").write_text("".join(checksum_lines), encoding="utf-8")
    return data_root


def _run_verifier(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VERIFIER), "--root", str(root)],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_repository_curated_assets_use_existing_runtime_paths():
    required = (
        "ad_data/map/global_info.json",
        "ad_data/map/node_set.json",
        "ad_data/map/link_set.json",
        "ad_data/map/lane_graph.json",
        "ad_data/map/route_corridor.json",
        "ad_data/path/2026_molit_comp_global_path.txt",
        "ad_data/paths/cp14_to_cp15.txt",
    )

    missing = [path for path in required if not (REPOSITORY_ROOT / path).is_file()]

    assert missing == []


def test_valid_curated_data_tree_passes(tmp_path: Path):
    _write_valid_repository(tmp_path)

    result = _run_verifier(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "verified 4 curated data files" in result.stdout


def test_local_archive_is_outside_the_curated_data_contract(tmp_path: Path):
    data_root = _write_valid_repository(tmp_path)
    archive = data_root / "local_archive" / "historical_workspace"
    archive.mkdir(parents=True)
    (archive / "run.db3").write_bytes(b"local rosbag payload")
    (archive / "machine-specific.json").write_text(
        '{"path": "/home/operator/private/result"}',
        encoding="utf-8",
    )

    result = _run_verifier(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "verified 4 curated data files" in result.stdout


def test_modified_file_fails_checksum_validation(tmp_path: Path):
    data_root = _write_valid_repository(tmp_path)
    scenario = data_root / "morai" / "SaveFile" / "Scenario" / "checkpoint1.json"
    scenario.write_text(json.dumps({"version": "modified"}), encoding="utf-8")

    result = _run_verifier(tmp_path)

    assert result.returncode == 1
    assert "checksum mismatch" in result.stderr


def test_runtime_database_is_rejected_even_when_checksummed(tmp_path: Path):
    data_root = _write_valid_repository(tmp_path)
    database = data_root / "tuning" / "study.sqlite3"
    database.parent.mkdir(parents=True)
    database.write_bytes(b"not a real sqlite database")
    digest = hashlib.sha256(database.read_bytes()).hexdigest()
    with (data_root / "SHA256SUMS").open("a", encoding="utf-8") as stream:
        stream.write(f"{digest}  tuning/study.sqlite3\n")

    result = _run_verifier(tmp_path)

    assert result.returncode == 1
    assert "forbidden runtime data" in result.stderr


def test_machine_specific_network_ip_is_rejected(tmp_path: Path):
    data_root = _write_valid_repository(tmp_path)
    network = data_root / "morai" / "SaveFile" / "Network" / "NetworkInfo.json"
    network.write_text(
        json.dumps({"host": "192.168.0.42", "port": 2368}),
        encoding="utf-8",
    )
    checksum_file = data_root / "SHA256SUMS"
    lines = [
        line
        for line in checksum_file.read_text(encoding="utf-8").splitlines()
        if not line.endswith("  morai/SaveFile/Network/NetworkInfo.json")
    ]
    digest = hashlib.sha256(network.read_bytes()).hexdigest()
    lines.append(f"{digest}  morai/SaveFile/Network/NetworkInfo.json")
    checksum_file.write_text("\n".join(sorted(lines)) + "\n", encoding="utf-8")

    result = _run_verifier(tmp_path)

    assert result.returncode == 1
    assert "machine-specific IPv4 address" in result.stderr


def test_machine_specific_absolute_home_path_is_rejected(tmp_path: Path):
    data_root = _write_valid_repository(tmp_path)
    scenario = data_root / "morai" / "SaveFile" / "Scenario" / "checkpoint1.json"
    scenario.write_text(
        json.dumps({"cache": "/home/developer/private/cache.bin"}),
        encoding="utf-8",
    )
    checksum_file = data_root / "SHA256SUMS"
    lines = [
        line
        for line in checksum_file.read_text(encoding="utf-8").splitlines()
        if not line.endswith("  morai/SaveFile/Scenario/checkpoint1.json")
    ]
    digest = hashlib.sha256(scenario.read_bytes()).hexdigest()
    lines.append(f"{digest}  morai/SaveFile/Scenario/checkpoint1.json")
    checksum_file.write_text("\n".join(sorted(lines)) + "\n", encoding="utf-8")

    result = _run_verifier(tmp_path)

    assert result.returncode == 1
    assert "machine-specific absolute path" in result.stderr


def test_malformed_json_is_rejected(tmp_path: Path):
    data_root = _write_valid_repository(tmp_path)
    scenario = data_root / "morai" / "SaveFile" / "Scenario" / "checkpoint1.json"
    scenario.write_text("{broken", encoding="utf-8")
    checksum_file = data_root / "SHA256SUMS"
    lines = [
        line
        for line in checksum_file.read_text(encoding="utf-8").splitlines()
        if not line.endswith("  morai/SaveFile/Scenario/checkpoint1.json")
    ]
    digest = hashlib.sha256(scenario.read_bytes()).hexdigest()
    lines.append(f"{digest}  morai/SaveFile/Scenario/checkpoint1.json")
    checksum_file.write_text("\n".join(sorted(lines)) + "\n", encoding="utf-8")

    result = _run_verifier(tmp_path)

    assert result.returncode == 1
    assert "invalid JSON" in result.stderr


def test_lfs_pointer_without_payload_is_rejected(tmp_path: Path):
    data_root = _write_valid_repository(tmp_path)
    payload = data_root / "morai" / "fixtures" / "sample.bin"
    payload.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:0123456789abcdef\n"
        "size 2048\n",
        encoding="utf-8",
    )
    checksum_file = data_root / "SHA256SUMS"
    lines = [
        line
        for line in checksum_file.read_text(encoding="utf-8").splitlines()
        if not line.endswith("  morai/fixtures/sample.bin")
    ]
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    lines.append(f"{digest}  morai/fixtures/sample.bin")
    checksum_file.write_text("\n".join(sorted(lines)) + "\n", encoding="utf-8")

    result = _run_verifier(tmp_path)

    assert result.returncode == 1
    assert "LFS payload is missing" in result.stderr
