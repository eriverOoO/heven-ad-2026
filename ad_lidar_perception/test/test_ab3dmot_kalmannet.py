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

    def test_expected_sha256_match_succeeds(self):
        import hashlib
        real_sha256 = hashlib.sha256(self.ckpt_path.read_bytes()).hexdigest()
        net, prov = load_kalmannet_network(
            str(self.ckpt_path), device="cpu", expected_sha256=real_sha256,
        )
        self.assertEqual(prov["checkpoint_sha256"], real_sha256)

    def test_expected_sha256_mismatch_raises_before_loading(self):
        wrong_sha256 = "0" * 64
        with self.assertRaises(RuntimeError) as ctx:
            load_kalmannet_network(str(self.ckpt_path), device="cpu", expected_sha256=wrong_sha256)
        self.assertIn("SHA-256 mismatch", str(ctx.exception))

    def test_no_expected_sha256_skips_verification(self):
        # Existing (pre-Runtime-Readiness-v1) behavior: no expected_sha256
        # argument at all -- verification stays fully opt-in.
        net, prov = load_kalmannet_network(str(self.ckpt_path), device="cpu")
        self.assertIsNotNone(net)


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

    def test_kalmannet_expected_sha256_default_disabled(self):
        cfg = AB3DMOTConfig(state_estimator="kalmannet", kalmannet_checkpoint="/tmp/x.pt")
        self.assertEqual(cfg.kalmannet_expected_sha256, "")

    def test_kalmannet_expected_sha256_accepts_valid_hex64(self):
        cfg = AB3DMOTConfig(
            state_estimator="kalmannet", kalmannet_checkpoint="/tmp/x.pt",
            kalmannet_expected_sha256="a" * 64,
        )
        self.assertEqual(cfg.kalmannet_expected_sha256, "a" * 64)

    def test_kalmannet_expected_sha256_rejects_wrong_length(self):
        with self.assertRaises(ValueError):
            AB3DMOTConfig(
                state_estimator="kalmannet", kalmannet_checkpoint="/tmp/x.pt",
                kalmannet_expected_sha256="deadbeef",
            )

    def test_kalmannet_expected_sha256_rejects_non_hex(self):
        with self.assertRaises(ValueError):
            AB3DMOTConfig(
                state_estimator="kalmannet", kalmannet_checkpoint="/tmp/x.pt",
                kalmannet_expected_sha256="z" * 64,
            )


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


class GapAndReacquireLifecycleTest(unittest.TestCase):
    """Runtime Readiness v1 sections 8E/8F: several consecutive missing
    measurements, then the measurement returns -- with an explicit
    max_age wide enough to survive the gap, so both scenarios are
    actually exercised (default max_age=2 only survives one miss)."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.ckpt_path = Path(self.tmpdir.name) / "tiny.pt"
        make_tiny_checkpoint(self.ckpt_path, hidden_size=4)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _tracker(self, max_age=6):
        cfg = AB3DMOTConfig(association_metric="euclidean", matcher="hungarian", euclidean_gate_m=3.0,
                             yaw_measurement_mode="unobserved", state_estimator="kalmannet",
                             kalmannet_checkpoint=str(self.ckpt_path), kalmannet_device="cpu",
                             max_age=max_age)
        return AB3DMOTTracker(cfg)

    def test_several_consecutive_missing_measurements_then_reacquire(self):
        tracker = self._tracker(max_age=6)
        t = 0.0
        out = tracker.step([make_detection(x=10.0, y=5.0)], t)
        track_id = out[0].track_id
        t += 0.15
        out = tracker.step([make_detection(x=10.15, y=5.0)], t)
        self.assertEqual(out[0].track_id, track_id)

        # 4 consecutive missing-measurement frames (predict-only every time).
        for _ in range(4):
            t += 0.15
            out = tracker.step([], t)
            self.assertEqual(len(out), 1)  # survives -- max_age=6
            self.assertEqual(out[0].track_id, track_id)
            self.assertTrue(np.isfinite(out[0].x) and np.isfinite(out[0].y))

        # measurement returns: same track_id, not a fresh birth.
        t += 0.15
        out = tracker.step([make_detection(x=10.9, y=5.0)], t)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].track_id, track_id)
        self.assertTrue(np.isfinite(out[0].vx_mps) and np.isfinite(out[0].vy_mps))

    def test_hidden_state_unchanged_across_missing_measurement_run(self):
        """The GRU hidden state must not be touched by predict-only frames
        -- only update() (a real matched measurement) reads/writes it, per
        KalmanNetEstimator's own documented division of responsibility."""
        tracker = self._tracker(max_age=6)
        tracker.step([make_detection(x=10.0, y=5.0)], 0.0)
        estimator = tracker._tracks[0]._estimator
        h_after_birth = estimator._hidden_state.clone()

        t = 0.15
        for _ in range(4):
            tracker.step([], t)  # predict-only: must NOT touch hidden state
            t += 0.15
        h_after_gap = estimator._hidden_state.clone()
        self.assertTrue(torch.equal(h_after_birth, h_after_gap))

        # a real update DOES change it (network was actually invoked).
        tracker.step([make_detection(x=10.9, y=5.0)], t)
        h_after_update = estimator._hidden_state.clone()
        self.assertFalse(torch.equal(h_after_gap, h_after_update))


class MultiObjectTrackerIsolationTest(unittest.TestCase):
    """Runtime Readiness v1 section 9: 3 simultaneous tracks through the
    real AB3DMOTTracker (association + lifecycle included, not just the
    raw estimator), different measurement histories, one with a
    missing-measurement gap."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.ckpt_path = Path(self.tmpdir.name) / "tiny.pt"
        make_tiny_checkpoint(self.ckpt_path, hidden_size=4)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _tracker(self):
        cfg = AB3DMOTConfig(association_metric="euclidean", matcher="hungarian", euclidean_gate_m=3.0,
                             yaw_measurement_mode="unobserved", state_estimator="kalmannet",
                             kalmannet_checkpoint=str(self.ckpt_path), kalmannet_device="cpu",
                             max_age=4)
        return AB3DMOTTracker(cfg)

    def test_three_simultaneous_tracks_with_one_gap_stay_isolated(self):
        tracker = self._tracker()
        # Birth 3 well-separated objects in one frame.
        t = 0.0
        out = tracker.step([
            make_detection(x=0.0, y=0.0),
            make_detection(x=50.0, y=0.0),
            make_detection(x=0.0, y=50.0),
        ], t)
        self.assertEqual(len(out), 3)
        by_start = {}
        for s in out:
            if abs(s.x - 0.0) < 1.0 and abs(s.y - 0.0) < 1.0:
                by_start["A"] = s.track_id
            elif abs(s.x - 50.0) < 1.0:
                by_start["B"] = s.track_id
            elif abs(s.y - 50.0) < 1.0:
                by_start["C"] = s.track_id
        self.assertEqual(len(by_start), 3)

        # Frame 2: A and B move normally; C has NO measurement this frame.
        t += 0.15
        out = tracker.step([
            make_detection(x=0.3, y=0.0),
            make_detection(x=50.3, y=0.0),
        ], t)
        self.assertEqual(len(out), 3)  # C survives as a coast (max_age=4)

        # Frames 3-4: same pattern -- A, B keep moving; C keeps missing.
        for i in range(2, 4):
            t += 0.15
            out = tracker.step([
                make_detection(x=0.3 * i, y=0.0),
                make_detection(x=50.0 + 0.3 * i, y=0.0),
            ], t)

        by_id = {s.track_id: s for s in out}
        self.assertEqual(len(by_id), 3)
        state_a = by_id[by_start["A"]]
        state_b = by_id[by_start["B"]]
        state_c = by_id[by_start["C"]]

        # A and B, which received real measurements every frame, moved
        # meaningfully; C, coasting on predict-only, is far closer to its
        # last real position than A/B are to theirs is irrelevant here --
        # the isolation property under test is that C's drift never
        # contaminated A or B's state (checked by A/B staying near their
        # OWN measurement trajectories) and vice versa.
        self.assertLess(abs(state_a.x - 0.9), 2.0)
        self.assertLess(abs(state_a.y - 0.0), 2.0)
        self.assertLess(abs(state_b.x - 50.9), 2.0)
        self.assertGreater(state_c.x, -5.0)
        self.assertLess(state_c.x, 5.0)  # C never jumped toward B's or A's moving trajectory
        for s in (state_a, state_b, state_c):
            self.assertTrue(np.isfinite(s.x) and np.isfinite(s.y))
            self.assertTrue(np.isfinite(s.vx_mps) and np.isfinite(s.vy_mps))

        # Distinct hidden-state tensors, never shared/aliased.
        estimators = [t._estimator for t in tracker._tracks]
        self.assertEqual(len(estimators), 3)
        hidden_ids = {id(e._hidden_state) for e in estimators}
        self.assertEqual(len(hidden_ids), 3)


class VariableDtRuntimeTest(unittest.TestCase):
    """Runtime Readiness v1 section 10: an irregular dt sequence through
    the real AB3DMOTTracker. The preferred checkpoint is FIXED-DT
    trained, but the runtime F/Q math must use the ACTUAL dt passed to
    predict(), never a hidden fixed-0.1 assumption -- this is a runtime
    math property, independent of what a given checkpoint was trained
    on, and is not itself a new training behavior."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.ckpt_path = Path(self.tmpdir.name) / "tiny.pt"
        make_tiny_checkpoint(self.ckpt_path, hidden_size=4)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _tracker(self):
        # Generous gate: the tiny untrained test fixture's early learned
        # gain is effectively random, so its predicted position can be
        # noisy for the first few frames -- a wide gate keeps this test
        # about dt handling, not association-gate tuning.
        cfg = AB3DMOTConfig(association_metric="euclidean", matcher="hungarian", euclidean_gate_m=50.0,
                             yaw_measurement_mode="unobserved", state_estimator="kalmannet",
                             kalmannet_checkpoint=str(self.ckpt_path), kalmannet_device="cpu")
        return AB3DMOTTracker(cfg)

    def test_irregular_dt_sequence_stays_finite(self):
        tracker = self._tracker()
        dt_sequence = [0.10, 0.11, 0.18, 0.10, 0.25, 0.09]
        t = 0.0
        out = tracker.step([make_detection(x=0.0, y=0.0)], t)
        self.assertEqual(len(out), 1)
        for i, dt in enumerate(dt_sequence):
            t += dt
            out = tracker.step([make_detection(x=float(i + 1), y=0.0)], t)
            self.assertEqual(len(out), 1)
            self.assertTrue(np.isfinite(out[0].x) and np.isfinite(out[0].y))
            self.assertTrue(np.isfinite(out[0].vx_mps) and np.isfinite(out[0].vy_mps))
            self.assertEqual(out[0].time_since_update, 0)

    def test_dt_actually_affects_the_analytical_prediction(self):
        """F_matrix(dt) is dt-dependent (position += velocity*dt); confirm
        predict() with two different dt values produces two different
        predicted positions for an otherwise-identical estimator state --
        proof the runtime never silently coerces dt to a fixed 0.1."""
        net, _ = load_kalmannet_network(str(self.ckpt_path), device="cpu")
        est_short = KalmanNetEstimator(make_detection(x=0.0, y=0.0), KF_CLASS, 1, "unobserved", net, "cpu")
        est_long = KalmanNetEstimator(make_detection(x=0.0, y=0.0), KF_CLASS, 2, "unobserved", net, "cpu")
        # Give both a nonzero velocity via one identical update first.
        for est in (est_short, est_long):
            est.predict(0.1)
            est.update(make_detection(x=1.0, y=0.0))
        pos_short = est_short.predicted_bev_position()
        est_short.predict(0.10)
        pos_after_short_predict = est_short.predicted_bev_position()
        pos_long = est_long.predicted_bev_position()
        est_long.predict(0.50)
        pos_after_long_predict = est_long.predicted_bev_position()
        # Different dt -> different analytical advance from the same start.
        self.assertFalse(np.allclose(
            pos_after_short_predict - pos_short, pos_after_long_predict - pos_long,
        ))


class AbnormalDtSafetyTest(unittest.TestCase):
    """Runtime Readiness v1 section 11: dt<=0 is rejected explicitly
    (never silently coerced to dt=1 or dt=0.1); a large-but-finite dt
    stays numerically finite (no NaN propagation)."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.ckpt_path = Path(self.tmpdir.name) / "tiny.pt"
        make_tiny_checkpoint(self.ckpt_path, hidden_size=4)
        self.net, _ = load_kalmannet_network(str(self.ckpt_path), device="cpu")

    def tearDown(self):
        self.tmpdir.cleanup()

    def _estimator(self):
        return KalmanNetEstimator(make_detection(x=0.0, y=0.0), KF_CLASS, 1, "unobserved", self.net, "cpu")

    def test_zero_dt_rejected(self):
        est = self._estimator()
        with self.assertRaises(ValueError):
            est.predict(0.0)

    def test_negative_dt_rejected(self):
        est = self._estimator()
        with self.assertRaises(ValueError):
            est.predict(-0.1)

    def test_nan_dt_rejected(self):
        est = self._estimator()
        with self.assertRaises(ValueError):
            est.predict(float("nan"))

    def test_infinite_dt_rejected(self):
        est = self._estimator()
        with self.assertRaises(ValueError):
            est.predict(float("inf"))

    def test_large_finite_dt_stays_finite(self):
        est = self._estimator()
        est.predict(120.0)  # 2 minutes -- unusually large but finite
        est.update(make_detection(x=5.0, y=0.0))
        self.assertTrue(est.is_finite())

    def test_tracker_step_rejects_non_positive_dt_at_frame_level(self):
        cfg = AB3DMOTConfig(association_metric="euclidean", matcher="hungarian",
                             yaw_measurement_mode="unobserved", state_estimator="kalmannet",
                             kalmannet_checkpoint=str(self.ckpt_path), kalmannet_device="cpu")
        tracker = AB3DMOTTracker(cfg)
        tracker.step([make_detection(x=0.0, y=0.0)], 1.0)
        with self.assertRaises(ValueError):
            tracker.step([make_detection(x=0.1, y=0.0)], 1.0)  # duplicate/zero dt
        with self.assertRaises(ValueError):
            tracker.step([make_detection(x=0.1, y=0.0)], 0.5)  # backwards


if __name__ == "__main__":
    unittest.main()
