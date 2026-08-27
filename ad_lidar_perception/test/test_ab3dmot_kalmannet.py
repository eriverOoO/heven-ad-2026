"""T-9B: tests for the opt-in KalmanNetEstimator / kalmannet state
estimator. Uses a tiny, deterministic, synthetic checkpoint fixture
(hidden_size=4) rather than the large real DENSE-KALMANNET-v2 artifact,
per this task's explicit "do not require the huge external artifact in
every repository unit test" instruction. Requires torch -- run via the
torch-enabled venv, same established pattern as test_kalmannet_core.py
(never wired into colcon/system-Python test suites, which have no torch).
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ad_lidar_perception.ab3dmot_config import AB3DMOTConfig
from ad_lidar_perception.ab3dmot_core import (
    AB3DMOTTracker, Detection, KalmanNetEstimator, Track,
    _load_ab3dmot_kf_class, load_kalmannet_network,
)
from ad_lidar_perception.kalmannet_core import KalmanNetGRU, set_seed

KF_CLASS = _load_ab3dmot_kf_class()


def make_tiny_checkpoint(path: Path, hidden_size: int = 4, seed: int = 0) -> None:
    set_seed(seed)
    net = KalmanNetGRU(hidden_size=hidden_size)
    torch.save(net.state_dict(), path)


def make_detection(x=10.0, y=5.0, z=1.0, yaw=0.0, l=4.0, w=2.0, h=1.5):
    return Detection(x=x, y=y, z=z, yaw=yaw, length=l, width=w, height=h)


class CheckpointLoadingTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.ckpt_path = Path(self.tmpdir.name) / "tiny.pt"
        make_tiny_checkpoint(self.ckpt_path, hidden_size=4)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_valid_checkpoint_loads(self):
        net, prov = load_kalmannet_network(str(self.ckpt_path), device="cpu")
        self.assertEqual(prov["hidden_size"], 4)
        self.assertIn("checkpoint_sha256", prov)
        self.assertEqual(len(prov["checkpoint_sha256"]), 64)

    def test_missing_checkpoint_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_kalmannet_network(str(Path(self.tmpdir.name) / "nope.pt"), device="cpu")

    def test_wrong_architecture_metadata_raises(self):
        bad_path = Path(self.tmpdir.name) / "bad_arch.pt"
        torch.save({"output_fc.weight": torch.zeros(3, 4), "output_fc.bias": torch.zeros(3)}, bad_path)
        with self.assertRaises(RuntimeError):
            load_kalmannet_network(str(bad_path), device="cpu")

    def test_corrupted_checkpoint_raises(self):
        corrupt_path = Path(self.tmpdir.name) / "corrupt.pt"
        corrupt_path.write_bytes(b"not a real checkpoint")
        with self.assertRaises(Exception):
            load_kalmannet_network(str(corrupt_path), device="cpu")

    def test_missing_expected_key_raises(self):
        bad_path = Path(self.tmpdir.name) / "missing_key.pt"
        torch.save({"some_other_key": torch.zeros(1)}, bad_path)
        with self.assertRaises(RuntimeError):
            load_kalmannet_network(str(bad_path), device="cpu")

    def test_cpu_loading_succeeds(self):
        net, prov = load_kalmannet_network(str(self.ckpt_path), device="cpu")
        self.assertEqual(prov["device"], "cpu")
        for p in net.parameters():
            self.assertEqual(p.device.type, "cpu")

    def test_deterministic_repeated_inference(self):
        net, _ = load_kalmannet_network(str(self.ckpt_path), device="cpu")
        est1 = KalmanNetEstimator(make_detection(), KF_CLASS, 1, "unobserved", net, "cpu")
        est2 = KalmanNetEstimator(make_detection(), KF_CLASS, 2, "unobserved", net, "cpu")
        for est in (est1, est2):
            est.predict(0.15)
            est.update(make_detection(x=10.2))
        self.assertEqual(est1.position, est2.position)
        self.assertEqual(est1.velocity, est2.velocity)


class ConfigValidationTest(unittest.TestCase):
    def test_default_state_estimator_unchanged(self):
        cfg = AB3DMOTConfig()
        self.assertEqual(cfg.state_estimator, "linear_kf")
        self.assertEqual(cfg.kalmannet_checkpoint, "")

    def test_kalmannet_requires_checkpoint(self):
        with self.assertRaises(ValueError):
            AB3DMOTConfig(state_estimator="kalmannet")

    def test_kalmannet_rejects_mahalanobis(self):
        with self.assertRaises(ValueError):
            AB3DMOTConfig(state_estimator="kalmannet", kalmannet_checkpoint="/tmp/x.pt",
                           association_metric="mahalanobis")

    def test_kalmannet_accepts_euclidean(self):
        cfg = AB3DMOTConfig(state_estimator="kalmannet", kalmannet_checkpoint="/tmp/x.pt",
                             association_metric="euclidean")
        self.assertEqual(cfg.state_estimator, "kalmannet")

    def test_non_kalmannet_estimator_ignores_checkpoint_field(self):
        # linear_kf/ekf/imm must not require or load any checkpoint.
        cfg = AB3DMOTConfig(state_estimator="linear_kf")
        self.assertEqual(cfg.kalmannet_checkpoint, "")


class TrackerNoCheckpointLoadWhenNotSelectedTest(unittest.TestCase):
    def test_linear_kf_tracker_never_touches_kalmannet_loader(self):
        cfg = AB3DMOTConfig(state_estimator="linear_kf")
        tracker = AB3DMOTTracker(cfg)
        self.assertIsNone(tracker._kalmannet_net)
        self.assertIsNone(tracker.kalmannet_provenance)


class MultiTrackIsolationTest(unittest.TestCase):
    """The mandatory Phase 13 test, using the tiny fixture so it runs fast
    as part of the standard suite."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.ckpt_path = Path(self.tmpdir.name) / "tiny.pt"
        make_tiny_checkpoint(self.ckpt_path, hidden_size=4)
        self.net, _ = load_kalmannet_network(str(self.ckpt_path), device="cpu")

    def tearDown(self):
        self.tmpdir.cleanup()

    def _make(self, x, y):
        return KalmanNetEstimator(make_detection(x=x, y=y), KF_CLASS, 1, "unobserved", self.net, "cpu")

    def test_interleaved_track_never_changes_the_other(self):
        est_a_alone = self._make(0.0, 0.0)
        positions_alone = []
        for i in range(8):
            est_a_alone.predict(0.15)
            est_a_alone.update(make_detection(x=float(i), y=0.0))
            positions_alone.append(est_a_alone.position)

        est_a = self._make(0.0, 0.0)
        est_b = self._make(100.0, 100.0)
        positions_interleaved = []
        for i in range(8):
            est_a.predict(0.15)
            est_a.update(make_detection(x=float(i), y=0.0))
            positions_interleaved.append(est_a.position)
            est_b.predict(0.15)
            est_b.update(make_detection(x=100.0, y=100.0 - i))

        for p1, p2 in zip(positions_alone, positions_interleaved):
            self.assertEqual(p1, p2)

    def test_new_track_after_death_gets_fresh_hidden_state(self):
        est_a = self._make(0.0, 0.0)
        for i in range(5):
            est_a.predict(0.15)
            est_a.update(make_detection(x=float(i), y=0.0))
        h_a = est_a._hidden_state.clone()
        self.assertFalse(torch.equal(h_a, torch.zeros_like(h_a)))

        est_c = self._make(50.0, 50.0)  # "new track" after A's (implicit) death
        h_c = est_c._hidden_state.clone()
        self.assertTrue(torch.equal(h_c, torch.zeros_like(h_c)))
        self.assertFalse(torch.equal(h_c, h_a))

    def test_track_id_reuse_does_not_leak_state(self):
        est_a = self._make(0.0, 0.0)
        for i in range(5):
            est_a.predict(0.15)
            est_a.update(make_detection(x=float(i), y=0.0))
        # same track_id reused for a brand-new estimator -- must not inherit A's state
        est_new = KalmanNetEstimator(make_detection(x=0.0, y=0.0), KF_CLASS, 1, "unobserved", self.net, "cpu")
        self.assertTrue(torch.equal(est_new._hidden_state, torch.zeros_like(est_new._hidden_state)))


class LifecycleTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.ckpt_path = Path(self.tmpdir.name) / "tiny.pt"
        make_tiny_checkpoint(self.ckpt_path, hidden_size=4)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _tracker(self):
        cfg = AB3DMOTConfig(association_metric="euclidean", matcher="hungarian", euclidean_gate_m=3.0,
                             yaw_measurement_mode="unobserved", state_estimator="kalmannet",
                             kalmannet_checkpoint=str(self.ckpt_path), kalmannet_device="cpu")
        return AB3DMOTTracker(cfg)

    def test_birth_update_miss_death(self):
        tracker = self._tracker()
        out = tracker.step([make_detection(x=10.0, y=5.0)], 0.0)
        self.assertEqual(len(out), 1)
        track_id = out[0].track_id

        out = tracker.step([make_detection(x=10.2, y=5.0)], 0.15)
        self.assertEqual(out[0].track_id, track_id)
        self.assertTrue(np.isfinite(out[0].vx_mps))
        self.assertTrue(np.isfinite(out[0].vy_mps))

        out = tracker.step([], 0.30)  # missed frame -- track should coast, not die yet (max_age=2 default)
        self.assertEqual(len(out), 1)

        out = tracker.step([], 0.45)  # second consecutive miss -- exceeds default max_age=2
        self.assertEqual(len(out), 0)

    def test_no_nan_inf_over_many_steps(self):
        tracker = self._tracker()
        t = 0.0
        for i in range(30):
            t += 0.15
            det = make_detection(x=10.0 + i * 0.3, y=5.0)
            out = tracker.step([det], t)
            for state in out:
                self.assertTrue(np.isfinite(state.x) and np.isfinite(state.y))
                self.assertTrue(np.isfinite(state.vx_mps) and np.isfinite(state.vy_mps))

    def test_kalmannet_provenance_logged_once(self):
        tracker = self._tracker()
        self.assertIsNotNone(tracker.kalmannet_provenance)
        self.assertIn("checkpoint_sha256", tracker.kalmannet_provenance)
        self.assertEqual(tracker.kalmannet_provenance["hidden_size"], 4)


if __name__ == "__main__":
    unittest.main()
