"""T-7A: state-estimator abstraction (LinearKFEstimator/EKFEstimator) and
CTRV-EKF correctness tests, plus tracker-integration regression coverage.
"""
import math
import unittest

import numpy as np

from ad_lidar_perception.ab3dmot_config import AB3DMOTConfig
from ad_lidar_perception.ab3dmot_core import (
    AB3DMOTTracker,
    Detection,
    EKFEstimator,
    LinearKFEstimator,
    Track,
    _load_ab3dmot_kf_class,
    _wrap_to_pi,
)

KF_CLASS = _load_ab3dmot_kf_class()


def make_detection(x=0.0, y=0.0, z=0.0, yaw=0.0, length=4.0, width=2.0, height=1.5) -> Detection:
    return Detection(x=x, y=y, z=z, yaw=yaw, length=length, width=width, height=height)


def make_tracker(state_estimator="linear_kf", **overrides) -> AB3DMOTTracker:
    config = AB3DMOTConfig(
        association_metric="euclidean", matcher="hungarian", euclidean_gate_m=3.0,
        state_estimator=state_estimator, **overrides,
    )
    return AB3DMOTTracker(config)


# ---------------------------------------------------------------------------
# LINEAR KF REGRESSION
# ---------------------------------------------------------------------------
class LinearKFRegressionTest(unittest.TestCase):
    def test_1_state_order_unchanged(self):
        det = make_detection(x=1.0, y=2.0, z=0.5, yaw=0.3, length=4.0, width=2.0, height=1.5)
        track = Track(1, det, KF_CLASS, state_estimator="linear_kf")
        x = track._estimator._kf.x.reshape(-1)
        self.assertAlmostEqual(x[0], 1.0)
        self.assertAlmostEqual(x[1], 2.0)
        self.assertAlmostEqual(x[2], 0.5)
        self.assertAlmostEqual(x[3], 0.3)
        self.assertAlmostEqual(x[4], 4.0)
        self.assertAlmostEqual(x[5], 2.0)
        self.assertAlmostEqual(x[6], 1.5)
        self.assertAlmostEqual(x[7], 0.0)
        self.assertAlmostEqual(x[8], 0.0)
        self.assertAlmostEqual(x[9], 0.0)

    def test_2_real_dt_cv_prediction_unchanged(self):
        det = make_detection(x=0.0, y=0.0)
        track = Track(1, det, KF_CLASS, state_estimator="linear_kf")
        track._estimator._kf.x[7, 0] = 2.0  # vx = 2 m/s
        track.predict(0.5)
        x, y, _ = track._estimator.position
        self.assertAlmostEqual(x, 1.0, places=6)  # 2 m/s * 0.5s
        self.assertAlmostEqual(y, 0.0, places=6)

    def test_3_same_measurement_update_behavior(self):
        tracker = make_tracker(state_estimator="linear_kf")
        tracker.step([make_detection(x=0.0, y=0.0)], 1.0)
        outputs = tracker.step([make_detection(x=1.0, y=0.0)], 2.0)
        self.assertEqual(len(outputs), 1)
        self.assertAlmostEqual(outputs[0].x, 1.0, delta=0.5)

    def test_4_ros_velocity_semantics_unchanged(self):
        tracker = make_tracker(state_estimator="linear_kf")
        tracker.step([make_detection(x=0.0, y=0.0)], 1.0)
        outputs = tracker.step([make_detection(x=1.0, y=0.0)], 2.0)
        self.assertTrue(math.isfinite(outputs[0].vx_mps))
        self.assertTrue(math.isfinite(outputs[0].vy_mps))
        self.assertTrue(math.isfinite(outputs[0].vz_mps))


# ---------------------------------------------------------------------------
# EKF
# ---------------------------------------------------------------------------
class EKFTest(unittest.TestCase):
    def test_5_stationary_remains_stationary(self):
        det = make_detection(x=3.0, y=-2.0, yaw=0.5)
        est = EKFEstimator(det, KF_CLASS, 1)
        est.predict(1.0)
        x, y, _ = est.position
        self.assertAlmostEqual(x, 3.0, places=9)
        self.assertAlmostEqual(y, -2.0, places=9)
        self.assertAlmostEqual(est.yaw, 0.5, places=9)

    def test_6_straight_motion_yaw_rate_zero(self):
        det = make_detection(x=0.0, y=0.0, yaw=0.0)
        est = EKFEstimator(det, KF_CLASS, 1)
        est.x[7, 0] = 5.0  # v = 5 m/s
        est.x[8, 0] = 0.0  # yaw_rate = 0
        est.predict(2.0)
        x, y, _ = est.position
        self.assertAlmostEqual(x, 10.0, places=6)  # v*dt
        self.assertAlmostEqual(y, 0.0, places=9)

    def test_7_constant_turn_prediction_matches_closed_form(self):
        det = make_detection(x=0.0, y=0.0, yaw=0.0)
        est = EKFEstimator(det, KF_CLASS, 1)
        v, omega, dt = 4.0, 0.5, 1.0
        est.x[7, 0] = v
        est.x[8, 0] = omega
        est.predict(dt)
        x, y, _ = est.position
        expected_x = (v / omega) * (math.sin(0 + omega * dt) - math.sin(0))
        expected_y = (v / omega) * (-math.cos(0 + omega * dt) + math.cos(0))
        self.assertAlmostEqual(x, expected_x, places=9)
        self.assertAlmostEqual(y, expected_y, places=9)
        self.assertAlmostEqual(est.yaw, omega * dt, places=9)

    def test_8_correct_yaw_update(self):
        det = make_detection(x=0.0, y=0.0, yaw=0.0)
        est = EKFEstimator(det, KF_CLASS, 1)
        est.x[8, 0] = 1.0  # yaw_rate = 1 rad/s
        est.predict(1.0)
        self.assertAlmostEqual(est.yaw, 1.0, places=6)

    def test_9_z_vz_propagation(self):
        det = make_detection(x=0.0, y=0.0, z=1.0)
        est = EKFEstimator(det, KF_CLASS, 1)
        est.x[9, 0] = 0.5  # vz
        est.predict(2.0)
        _, _, z = est.position
        self.assertAlmostEqual(z, 2.0, places=6)  # 1.0 + 0.5*2.0

    def test_10_dimensions_finite_stable(self):
        det = make_detection(length=4.5, width=2.1, height=1.6)
        est = EKFEstimator(det, KF_CLASS, 1)
        est.x[7, 0] = 3.0
        est.predict(1.0)
        length, width, height = est.dimensions
        self.assertAlmostEqual(length, 4.5)
        self.assertAlmostEqual(width, 2.1)
        self.assertAlmostEqual(height, 1.6)

    def test_11_dt_scaling(self):
        det = make_detection(x=0.0, y=0.0, yaw=0.0)
        est_small = EKFEstimator(det, KF_CLASS, 1)
        est_small.x[7, 0] = 3.0
        est_small.predict(1.0)
        est_big = EKFEstimator(det, KF_CLASS, 1)
        est_big.x[7, 0] = 3.0
        est_big.predict(2.0)
        x_small, _, _ = est_small.position
        x_big, _, _ = est_big.position
        self.assertAlmostEqual(x_big, 2.0 * x_small, places=6)

    def test_12_yaw_wrap_across_pi(self):
        det = make_detection(x=0.0, y=0.0, yaw=3.0)
        est = EKFEstimator(det, KF_CLASS, 1)
        est.x[8, 0] = 1.0  # yaw_rate pushes yaw past pi
        est.predict(1.0)
        self.assertTrue(-math.pi <= est.yaw < math.pi)
        self.assertAlmostEqual(est.yaw, _wrap_to_pi(3.0 + 1.0), places=6)

    def test_13_no_nan_inf(self):
        det = make_detection(x=0.0, y=0.0)
        est = EKFEstimator(det, KF_CLASS, 1)
        est.x[7, 0] = 2.0
        est.x[8, 0] = 0.3
        for i in range(10):
            est.predict(0.15)
            est.update(make_detection(x=float(i) * 0.3, y=float(i) * 0.05))
        self.assertTrue(est.is_finite())

    def test_14_covariance_finite_symmetric(self):
        det = make_detection(x=0.0, y=0.0)
        est = EKFEstimator(det, KF_CLASS, 1)
        est.x[7, 0] = 2.0
        est.x[8, 0] = 0.4
        est.predict(0.2)
        est.update(make_detection(x=0.4, y=0.1))
        self.assertTrue(np.all(np.isfinite(est.P)))
        np.testing.assert_allclose(est.P, est.P.T, atol=1e-9)

    def test_15_deterministic_repeated_sequence(self):
        def run():
            det = make_detection(x=0.0, y=0.0)
            est = EKFEstimator(det, KF_CLASS, 1)
            est.x[7, 0] = 2.0
            est.x[8, 0] = 0.2
            for i in range(5):
                est.predict(0.15)
                est.update(make_detection(x=float(i) * 0.2, y=float(i) * 0.1))
            return est.x.copy(), est.P.copy()

        x1, p1 = run()
        x2, p2 = run()
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(p1, p2)


# ---------------------------------------------------------------------------
# MEASUREMENT
# ---------------------------------------------------------------------------
class EKFMeasurementTest(unittest.TestCase):
    def test_16_position_correction_works(self):
        det = make_detection(x=0.0, y=0.0)
        est = EKFEstimator(det, KF_CLASS, 1)
        est.predict(1.0)
        x_before, y_before, _ = est.position
        est.update(make_detection(x=5.0, y=5.0))
        x_after, y_after, _ = est.position
        self.assertGreater(abs(x_after - x_before), 0.0)
        self.assertLess(abs(x_after - 5.0), abs(x_before - 5.0))

    def test_17_yaw_correction_works(self):
        det = make_detection(x=0.0, y=0.0, yaw=0.1)
        est = EKFEstimator(det, KF_CLASS, 1)
        est.predict(1.0)
        est.update(make_detection(x=0.0, y=0.0, yaw=0.5))
        self.assertNotAlmostEqual(est.yaw, 0.1, places=3)

    def test_18_unobserved_v_yaw_rate_estimated_not_overwritten(self):
        det = make_detection(x=0.0, y=0.0, yaw=0.0)
        est = EKFEstimator(det, KF_CLASS, 1)
        self.assertAlmostEqual(est.speed, 0.0)
        for i in range(1, 8):
            est.predict(0.15)
            est.update(make_detection(x=float(i) * 0.15 * 3.0, y=0.0))
        # after several consistent ~3 m/s measurements, v should have grown
        # smoothly toward ~3, not been reset/overwritten to any single value.
        self.assertGreater(est.speed, 0.5)
        self.assertLess(est.speed, 6.0)


# ---------------------------------------------------------------------------
# TRACKER INTEGRATION
# ---------------------------------------------------------------------------
class TrackerIntegrationTest(unittest.TestCase):
    def test_19_unmatched_prediction_works(self):
        tracker = make_tracker(state_estimator="ekf")
        tracker.step([make_detection(x=0.0, y=0.0)], 1.0)
        outputs = tracker.step([make_detection(x=100.0, y=100.0)], 2.0)
        ids = {o.track_id for o in outputs}
        self.assertEqual(ids, {1, 2})  # track1 coasts (unmatched), track2 new

    def test_20_matched_update_works(self):
        tracker = make_tracker(state_estimator="ekf")
        tracker.step([make_detection(x=0.0, y=0.0)], 1.0)
        outputs = tracker.step([make_detection(x=0.3, y=0.0)], 2.0)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].track_id, 1)

    def test_21_lifecycle_unchanged(self):
        for estimator in ("linear_kf", "ekf"):
            tracker = make_tracker(state_estimator=estimator, max_age=2, min_hits=1)
            tracker.step([make_detection(x=0.0, y=0.0)], 1.0)  # time_since_update=0
            out2 = tracker.step([make_detection(x=100.0, y=100.0)], 2.0)  # far -> track1 coasts, time_since_update=1
            ids2 = {o.track_id for o in out2}
            self.assertIn(1, ids2, f"estimator={estimator}: track should still be alive at time_since_update=1")
            out3 = tracker.step([], 3.0)  # time_since_update=2 -> is_dead (>= max_age)
            ids3 = {o.track_id for o in out3}
            self.assertNotIn(1, ids3, f"estimator={estimator}: track should be dead")

    def test_22_id_uniqueness_unchanged(self):
        for estimator in ("linear_kf", "ekf"):
            tracker = make_tracker(state_estimator=estimator)
            tracker.step([make_detection(x=0.0, y=0.0), make_detection(x=50.0, y=0.0)], 1.0)
            outputs = tracker.step(
                [make_detection(x=0.1, y=0.0), make_detection(x=50.1, y=0.0), make_detection(x=200.0, y=0.0)], 2.0
            )
            ids = [o.track_id for o in outputs]
            self.assertEqual(len(ids), len(set(ids)), f"estimator={estimator}: duplicate track id")

    def test_23_default_linear_kf_behavior_unchanged(self):
        config = AB3DMOTConfig()
        self.assertEqual(config.state_estimator, "linear_kf")
        tracker = AB3DMOTTracker(config)
        tracker.step([make_detection(x=0.0, y=0.0)], 1.0)
        outputs = tracker.step([make_detection(x=0.1, y=0.0)], 2.0)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].track_id, 1)

    def test_24_unsupported_estimator_rejected(self):
        with self.assertRaises(ValueError):
            AB3DMOTConfig(state_estimator="ukf")

    def test_25_ekf_tracker_output_all_finite(self):
        tracker = make_tracker(state_estimator="ekf")
        tracker.step([make_detection(x=0.0, y=0.0), make_detection(x=5.0, y=0.0)], 1.0)
        outputs = tracker.step([make_detection(x=0.3, y=0.1), make_detection(x=5.2, y=0.0)], 2.0)
        for o in outputs:
            for v in (o.x, o.y, o.z, o.yaw, o.vx_mps, o.vy_mps, o.vz_mps):
                self.assertTrue(math.isfinite(v))
            self.assertTrue(np.all(np.isfinite(o.position_covariance)))


if __name__ == "__main__":
    unittest.main()
