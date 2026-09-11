import hashlib
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
DOWNLOAD_SCRIPT = ROOT / "scripts" / "download_centerpoint_checkpoint.sh"
MANIFEST = ROOT / "models" / "experimental" / "manifest.yaml"
RUNNER = ROOT / "scripts" / "run_camera_lidar_tracking.sh"

EXPECTED_SHA256 = (
    "466c8181a377682e032bb32579c8ddb65807b5feebd8625a750b1d5538ddbc95"
)


def _manifest_entry():
    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    return next(
        m for m in data["models"] if m["file"] == "centerpoint_t14_reproduction.pth"
    )


def test_download_script_is_executable_and_portable():
    source = DOWNLOAD_SCRIPT.read_text(encoding="utf-8")

    assert DOWNLOAD_SCRIPT.stat().st_mode & 0o111
    assert "/home/" not in source
    assert "models/experimental/centerpoint_t14_reproduction.pth" in source
    assert EXPECTED_SHA256 in source


def test_download_script_default_dest_matches_runner_default():
    download_source = DOWNLOAD_SCRIPT.read_text(encoding="utf-8")
    runner_source = RUNNER.read_text(encoding="utf-8")

    assert (
        "models/experimental/centerpoint_t14_reproduction.pth" in download_source
    )
    assert (
        "models/experimental/centerpoint_t14_reproduction.pth" in runner_source
    )


def test_download_script_rejects_unknown_option():
    result = subprocess.run(
        [str(DOWNLOAD_SCRIPT), "--not-a-real-option"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0


def test_download_script_help_does_not_require_network():
    result = subprocess.run(
        [str(DOWNLOAD_SCRIPT), "--help"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "--dest" in result.stderr
    assert "--force" in result.stderr


def test_manifest_records_matching_sha256_and_size():
    entry = _manifest_entry()

    assert entry["sha256"] == EXPECTED_SHA256
    assert entry["file_size_bytes"] == 93474618
    assert entry["download_url"].startswith(
        "https://github.com/eriverOoO/heven-ad-2026/releases/download/"
    )


def test_manifest_documents_train_eval_overlap_limitation():
    entry = _manifest_entry()
    limitations = " ".join(entry["known_limitations"])

    assert "100%" in limitations
    assert "overlap" in limitations.lower()


def test_manifest_does_not_claim_moraivalidation():
    manifest_text = MANIFEST.read_text(encoding="utf-8").lower()

    assert "validated on morai" not in manifest_text
    assert "optimal for morai" not in manifest_text


def test_verify_sha256_helper_matches_a_known_file(tmp_path):
    sample = tmp_path / "sample.bin"
    sample.write_bytes(b"heven-centerpoint-checkpoint-fixture")
    expected = hashlib.sha256(sample.read_bytes()).hexdigest()

    result = subprocess.run(
        ["sha256sum", "--", str(sample)],
        check=True,
        text=True,
        capture_output=True,
    )
    actual = result.stdout.split()[0]

    assert actual == expected
