"""KalmanNet Production Candidate Freeze v1: validates
config/kalmannet/production_candidate.yaml against the actual checkpoint
file it names, so the manifest can never silently drift from reality.

The last two cases (hidden_size / state-measurement dims) require torch
and are run via the torch-enabled venv, same established pattern as
test_ab3dmot_kalmannet.py -- never wired into colcon/system-Python test
suites, which have no torch.
"""
from __future__ import annotations

import hashlib
import re
import sys
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "ad_lidar_perception" / "config" / "kalmannet" / "production_candidate.yaml"


def _load_manifest() -> dict:
    return yaml.safe_load(MANIFEST_PATH.read_text())


class ManifestSchemaTest(unittest.TestCase):
    def test_manifest_parses_as_valid_yaml(self):
        manifest = _load_manifest()
        self.assertIsInstance(manifest, dict)

    def test_manifest_has_no_machine_specific_absolute_path(self):
        text = MANIFEST_PATH.read_text()
        self.assertNotIn("/home/", text)
        self.assertIsNone(re.search(r"[A-Za-z]:\\\\", text))

    def test_checkpoint_file_is_a_relative_path(self):
        manifest = _load_manifest()
        checkpoint_file = manifest["checkpoint_file"]
        self.assertFalse(Path(checkpoint_file).is_absolute())
        self.assertTrue(checkpoint_file.startswith("models/experimental/"))

    def test_checkpoint_file_exists_in_repository(self):
        manifest = _load_manifest()
        checkpoint_path = REPO_ROOT / manifest["checkpoint_file"]
        self.assertTrue(checkpoint_path.is_file(), f"{checkpoint_path} does not exist")

    def test_recorded_sha256_matches_the_actual_checkpoint_bytes(self):
        manifest = _load_manifest()
        checkpoint_path = REPO_ROOT / manifest["checkpoint_file"]
        actual_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        self.assertEqual(actual_sha256, manifest["checkpoint_sha256"])

    def test_status_declares_rejected_and_diagnostic_variants_explicitly(self):
        manifest = _load_manifest()
        status = manifest["status"]
        self.assertIn("rejected", status["motion_focused_model"].lower())
        self.assertIn("diagnostic", status["mild_variable_dt_model"].lower())
        self.assertIn("rejected", status["mixed_50_model"].lower())
        self.assertIn("pending", status["morai_final_validation"].lower())

    def test_selection_rationale_does_not_claim_morai_validation(self):
        manifest = _load_manifest()
        text = (manifest["selection_rationale"] + manifest["status"]["morai_final_validation"]).lower()
        self.assertNotIn("validated on morai", text)
        self.assertNotIn("morai-validated", text)


class ManifestVsRuntimeTest(unittest.TestCase):
    """Requires torch -- run via the torch-enabled venv."""

    def test_state_and_measurement_definitions_match_runtime_dims(self):
        manifest = _load_manifest()
        self.assertEqual(manifest["state_definition"], ["x", "y", "vx", "vy"])
        self.assertEqual(manifest["measurement_definition"], ["x", "y"])
        sys.path.insert(0, str(REPO_ROOT / "ad_lidar_perception"))
        from ad_lidar_perception.kalmannet_core import STATE_DIM, MEAS_DIM
        self.assertEqual(len(manifest["state_definition"]), STATE_DIM)
        self.assertEqual(len(manifest["measurement_definition"]), MEAS_DIM)

    def test_hidden_size_matches_the_actual_checkpoint(self):
        manifest = _load_manifest()
        checkpoint_path = REPO_ROOT / manifest["checkpoint_file"]
        sys.path.insert(0, str(REPO_ROOT / "ad_lidar_perception"))
        from ad_lidar_perception.ab3dmot_core import load_kalmannet_network
        _, provenance = load_kalmannet_network(str(checkpoint_path), device="cpu")
        self.assertEqual(provenance["hidden_size"], manifest["hidden_size"])
        self.assertEqual(provenance["checkpoint_sha256"], manifest["checkpoint_sha256"])


if __name__ == "__main__":
    unittest.main()
