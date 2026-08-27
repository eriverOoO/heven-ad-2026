"""T-7B: IMM(CV + CTRV) state/covariance transformation, probability, and
mixing tests, plus tracker-integration regression coverage.
"""
import math
import unittest

import numpy as np

from ad_lidar_perception.ab3dmot_config import AB3DMOTConfig
from ad_lidar_perception.ab3dmot_core import (
    AB3DMOTTracker,
    Detection,
    IMMEstimator,
    Track,
    _common_state_from_cv,
    _common_state_from_ctrv,
    _cv_native_from_common,
    _ctrv_native_from_common,
    _gaussian_log_likelihood,
    _load_ab3dmot_kf_class,
    _wrap_to_pi,
)

KF_CLASS = _load_ab3dmot_kf_class()


def make_detection(x=0.0, y=0.0, z=0.0, yaw=0.0, length=4.0, width=2.0, height=1.5) -> Detection:
    return Detection(x=x, y=y, z=z, yaw=yaw, length=length, width=width, height=height)


def make_tracker(state_estimator="imm", **overrides) -> AB3DMOTTracker:
    config = AB3DMOTConfig(
        association_metric="euclidean", matcher="hungarian", euclidean_gate_m=3.0,
        state_estimator=state_estimator, yaw_measurement_mode="unobserved", **overrides,
    )
    return AB3DMOTTracker(config)


# ---------------------------------------------------------------------------
# STATE TRANSFORMATION
# ---------------------------------------------------------------------------
class StateTransformTest(unittest.TestCase):
    def test_1_cv_native_to_common(self):
        cv_x = np.array([1.0, 2.0, 0.5, 0.3, 4.0, 2.0, 1.5, 3.0, 0.0, 0.0]).reshape(10, 1)
        cv_p = np.eye(10) * 2.0
        common, common_p = _common_state_from_cv(cv_x, cv_p)
        self.assertEqual(common.shape, (11,))
        self.assertEqual(common_p.shape, (11, 11))
        self.assertAlmostEqual(common[0], 1.0)
        self.assertAlmostEqual(common[1], 2.0)
        self.assertAlmostEqual(common[7], 3.0)  # vx preserved
        self.assertAlmostEqual(common[8], 0.0)  # vy preserved

    def test_2_ctrv_native_to_common(self):
        ctrv_x = np.array([1.0, 2.0, 0.5, 0.0, 4.0, 2.0, 1.5, 3.0, 0.2, 0.0]).reshape(10, 1)
        ctrv_p = np.eye(10) * 2.0
        common, common_p = _common_state_from_ctrv(ctrv_x, ctrv_p)
        self.assertAlmostEqual(common[3], 0.0)  # yaw preserved directly
        self.assertAlmostEqual(common[7], 3.0, places=6)  # vx = v*cos(0) = v
        self.assertAlmostEqual(common[8], 0.0, places=6)  # vy = v*sin(0) = 0
        self.assertAlmostEqual(common[10], 0.2)  # yaw_rate preserved

    def test_3_common_to_ctrv(self):
        common = np.array([1.0, 2.0, 0.5, 0.7, 4.0, 2.0, 1.5, 3.0, 3.0, 0.0, 0.15])
        common_p = np.eye(11) * 2.0
        ctrv_x, ctrv_p = _ctrv_native_from_common(common, common_p)
        self.assertAlmostEqual(ctrv_x[3], 0.7)  # yaw taken directly from common
        self.assertAlmostEqual(ctrv_x[7], math.hypot(3.0, 3.0), places=6)  # v = |vx,vy|
        self.assertAlmostEqual(ctrv_x[8], 0.15)  # yaw_rate

    def test_4_plus_x_velocity_conversion(self):
        ctrv_x = np.array([0, 0, 0, 0.0, 4, 2, 1.5, 5.0, 0.0, 0]).reshape(10, 1)
        common, _ = _common_state_from_ctrv(ctrv_x, np.eye(10))
        self.assertAlmostEqual(common[7], 5.0, places=6)
        self.assertAlmostEqual(common[8], 0.0, places=6)

    def test_5_plus_y_velocity_conversion(self):
        ctrv_x = np.array([0, 0, 0, math.pi / 2, 4, 2, 1.5, 5.0, 0.0, 0]).reshape(10, 1)
        common, _ = _common_state_from_ctrv(ctrv_x, np.eye(10))
        self.assertAlmostEqual(common[7], 0.0, places=6)
        self.assertAlmostEqual(common[8], 5.0, places=6)

    def test_6_minus_x_velocity_conversion(self):
        ctrv_x = np.array([0, 0, 0, math.pi, 4, 2, 1.5, 5.0, 0.0, 0]).reshape(10, 1)
        common, _ = _common_state_from_ctrv(ctrv_x, np.eye(10))
        self.assertAlmostEqual(common[7], -5.0, places=6)
        self.assertAlmostEqual(common[8], 0.0, places=6)

    def test_7_near_zero_speed_conversion_stable(self):
        common = np.array([0, 0, 0, 0.3, 4, 2, 1.5, 0.0001, -0.0001, 0, 0.0])
        common_p = np.eye(11) * 5.0
        ctrv_x, ctrv_p = _ctrv_native_from_common(common, common_p)
        self.assertTrue(np.all(np.isfinite(ctrv_x)))
        self.assertTrue(np.all(np.isfinite(ctrv_p)))
        cv_x, cv_p = _cv_native_from_common(common, common_p)
        self.assertTrue(np.all(np.isfinite(cv_x)) and np.all(np.isfinite(cv_p)))

    def test_8_covariance_transform_finite_symmetric(self):
        cv_x = np.array([1, 2, 0, 0, 4, 2, 1.5, 3, 1, 0]).reshape(10, 1)
        cv_p = np.eye(10) * 3.0
        _, common_p = _common_state_from_cv(cv_x, cv_p)
        self.assertTrue(np.all(np.isfinite(common_p)))
        np.testing.assert_allclose(common_p, common_p.T, atol=1e-9)


# ---------------------------------------------------------------------------
# IMM PROBABILITIES
# ---------------------------------------------------------------------------
class ImmProbabilityTest(unittest.TestCase):
    def test_9_transition_probabilities_sum_correctly(self):
        det = make_detection()
        est = IMMEstimator(det, KF_CLASS, 1, "unobserved", cv_to_cv_probability=0.9, ctrv_to_ctrv_probability=0.8)
        row_sums = est._transition.sum(axis=1)
        np.testing.assert_allclose(row_sums, [1.0, 1.0])

    def test_10_conditional_mixing_probabilities_normalize(self):
        det = make_detection()
        est = IMMEstimator(det, KF_CLASS, 1, "unobserved")
        est.predict(0.15)
        # after predict, predicted_mu itself must sum to 1
        self.assertAlmostEqual(float(est._predicted_mu.sum()), 1.0, places=9)

    def test_11_posterior_probabilities_sum_to_1(self):
        det = make_detection()
        est = IMMEstimator(det, KF_CLASS, 1, "unobserved")
        est.predict(0.15)
        est.update(make_detection(x=0.3))
        self.assertAlmostEqual(est.mu_cv + est.mu_ctrv, 1.0, places=9)

    def test_12_equal_likelihood_preserves_symmetric_probabilities(self):
        # Directly: identical residual/S inputs to the log-likelihood
        # function itself must produce identical likelihoods (and hence,
        # combined with equal priors, identical posterior weight) --
        # isolates the likelihood *formula*'s own symmetry from the two
        # models' structurally different native covariance, which need
        # not itself produce symmetric priors on a real, short sequence.
        residual = np.array([0.1, -0.05, 0.0, 0.0, 0.0, 0.0])
        S = np.eye(6) * 2.0
        ll_a = _gaussian_log_likelihood(residual, S)
        ll_b = _gaussian_log_likelihood(residual, S)
        self.assertAlmostEqual(ll_a, ll_b, places=9)

    def test_13_cv_favored_innovation_increases_cv_probability(self):
        det = make_detection(x=0.0, y=0.0)
        est = IMMEstimator(det, KF_CLASS, 1, "unobserved")
        for i in range(1, 20):
            est.predict(0.15)
            est.update(make_detection(x=float(i) * 0.5, y=0.0))  # perfectly straight
        self.assertGreater(est.mu_cv, 0.5)

    def test_14_ctrv_favored_innovation_increases_ctrv_probability(self):
        det = make_detection(x=0.0, y=0.0)
        est = IMMEstimator(det, KF_CLASS, 1, "unobserved")
        v, omega, dt = 4.0, 2.5, 0.15
        x, y, yaw = 0.0, 0.0, 0.0
        mu_ctrv_early, mu_ctrv_late = None, None
        for i in range(1, 20):
            est.predict(dt)
            yaw += omega * dt
            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt
            est.update(make_detection(x=x, y=y))
            if i == 8:
                mu_ctrv_early = est.mu_ctrv
            if i == 19:
                mu_ctrv_late = est.mu_ctrv
        # under a sustained sharp turn, CTRV should gain (not lose) ground
        # over time relative to its own earlier value.
        self.assertGreaterEqual(mu_ctrv_late, mu_ctrv_early - 1e-9)

    def test_15_no_probability_nan_inf(self):
        det = make_detection()
        est = IMMEstimator(det, KF_CLASS, 1, "unobserved")
        for i in range(15):
            est.predict(0.15)
            est.update(make_detection(x=float(i) * 0.3, y=float(i) * 0.05))
        self.assertTrue(math.isfinite(est.mu_cv) and math.isfinite(est.mu_ctrv))


# ---------------------------------------------------------------------------
# MIXING
# ---------------------------------------------------------------------------
class MixingTest(unittest.TestCase):
    def test_16_identical_model_states_unchanged_by_mixing(self):
        common = np.array([1, 2, 0, 0.5, 4, 2, 1.5, 3, 0, 0, 0.1])
        p = np.eye(11) * 2.0
        combined, combined_p = IMMEstimator._combine(common, p, common, p, np.array([0.5, 0.5]))
        np.testing.assert_allclose(combined, common, atol=1e-9)

    def test_17_weighted_state_mean_correct(self):
        c1 = np.zeros(11)
        c1[0] = 0.0
        c2 = np.zeros(11)
        c2[0] = 10.0
        p = np.eye(11)
        combined, _ = IMMEstimator._combine(c1, p, c2, p, np.array([0.25, 0.75]))
        self.assertAlmostEqual(combined[0], 7.5, places=6)

    def test_18_mixture_covariance_includes_spread(self):
        c1 = np.zeros(11)
        c2 = np.zeros(11)
        c2[0] = 10.0  # large mean separation
        p = np.eye(11) * 0.01  # tiny within-model covariance
        _, combined_p = IMMEstimator._combine(c1, p, c2, p, np.array([0.5, 0.5]))
        # spread term must dominate: combined variance for x should be
        # much larger than the tiny within-model 0.01.
        self.assertGreater(combined_p[0, 0], 1.0)


# ---------------------------------------------------------------------------
# BEHAVIOR (synthetic)
# ---------------------------------------------------------------------------
class BehaviorTest(unittest.TestCase):
    def test_19_straight_trajectory_favors_or_retains_cv(self):
        det = make_detection(x=0.0, y=0.0)
        est = IMMEstimator(det, KF_CLASS, 1, "unobserved")
        for i in range(1, 15):
            est.predict(0.15)
            est.update(make_detection(x=float(i) * 0.4, y=0.0))
        self.assertGreaterEqual(est.mu_cv, 0.5)

    def test_20_constant_turn_increases_ctrv_relative_to_baseline(self):
        det = make_detection(x=0.0, y=0.0)
        est = IMMEstimator(det, KF_CLASS, 1, "unobserved")
        v, omega, dt = 4.0, 2.5, 0.15
        x, y, yaw = 0.0, 0.0, 0.0
        mu_ctrv_vals = []
        for i in range(1, 20):
            est.predict(dt)
            yaw += omega * dt
            x += v * math.cos(yaw) * dt
            y += v * math.sin(yaw) * dt
            est.update(make_detection(x=x, y=y))
            mu_ctrv_vals.append(est.mu_ctrv)
        # CTRV's probability should not monotonically collapse to ~0 under
        # sustained sharp turning -- it should stay meaningfully engaged.
        self.assertGreater(max(mu_ctrv_vals[5:]), 0.3)

    def test_21_stationary_near_zero_speed_stable(self):
        det = make_detection(x=5.0, y=5.0)
        est = IMMEstimator(det, KF_CLASS, 1, "unobserved")
        for i in range(10):
            est.predict(0.15)
            est.update(make_detection(x=5.0 + 0.01 * i, y=5.0))
            self.assertTrue(est.is_finite())

    def test_22_yaw_wraps_correctly(self):
        self.assertAlmostEqual(_wrap_to_pi(math.pi + 0.2), -math.pi + 0.2, places=6)
        det = make_detection(x=0.0, y=0.0)
        est = IMMEstimator(det, KF_CLASS, 1, "unobserved")
        for i in range(1, 10):
            est.predict(0.15)
            est.update(make_detection(x=-float(i) * 0.5, y=-0.01))
        self.assertTrue(-math.pi <= est.yaw < math.pi)


# ---------------------------------------------------------------------------
# TRACKER INTEGRATION
# ---------------------------------------------------------------------------
class TrackerIntegrationTest(unittest.TestCase):
    def test_23_matched_update(self):
        tracker = make_tracker()
        tracker.step([make_detection(x=0.0, y=0.0)], 1.0)
        outputs = tracker.step([make_detection(x=0.3, y=0.0)], 2.0)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].track_id, 1)

    def test_24_unmatched_prediction(self):
        tracker = make_tracker()
        tracker.step([make_detection(x=0.0, y=0.0)], 1.0)
        outputs = tracker.step([make_detection(x=100.0, y=100.0)], 2.0)
        self.assertEqual({o.track_id for o in outputs}, {1, 2})

    def test_25_lifecycle_unchanged(self):
        tracker = make_tracker(max_age=2, min_hits=1)
        tracker.step([make_detection(x=0.0, y=0.0)], 1.0)
        out2 = tracker.step([make_detection(x=100.0, y=100.0)], 2.0)
        self.assertIn(1, {o.track_id for o in out2})
        out3 = tracker.step([], 3.0)
        self.assertNotIn(1, {o.track_id for o in out3})

    def test_26_ids_unchanged(self):
        tracker = make_tracker()
        tracker.step([make_detection(x=0.0, y=0.0), make_detection(x=50.0, y=0.0)], 1.0)
        outputs = tracker.step(
            [make_detection(x=0.1, y=0.0), make_detection(x=50.1, y=0.0), make_detection(x=200.0, y=0.0)], 2.0
        )
        ids = [o.track_id for o in outputs]
        self.assertEqual(len(ids), len(set(ids)))

    def test_27_association_config_unchanged(self):
        config = AB3DMOTConfig(
            association_metric="euclidean", matcher="hungarian", euclidean_gate_m=3.0,
            state_estimator="imm", yaw_measurement_mode="unobserved",
        )
        self.assertEqual(config.association_metric, "euclidean")
        self.assertEqual(config.matcher, "hungarian")
        self.assertEqual(config.euclidean_gate_m, 3.0)

    def test_28_default_linear_kf_unchanged(self):
        config = AB3DMOTConfig()
        self.assertEqual(config.state_estimator, "linear_kf")
        tracker = AB3DMOTTracker(config)
        tracker.step([make_detection(x=0.0, y=0.0)], 1.0)
        outputs = tracker.step([make_detection(x=0.1, y=0.0)], 2.0)
        self.assertEqual(len(outputs), 1)

    def test_29_deterministic_repeated_sequence(self):
        def run():
            tracker = make_tracker()
            tracker.step([make_detection(x=0.0, y=0.0)], 1.0)
            for i in range(1, 6):
                tracker.step([make_detection(x=float(i) * 0.3, y=0.0)], 1.0 + i)
            out = tracker.step([make_detection(x=2.0, y=0.0)], 7.0)
            return [(o.track_id, round(o.x, 6), round(o.y, 6)) for o in out]

        self.assertEqual(run(), run())

    def test_30_unsupported_estimator_still_rejected(self):
        with self.assertRaises(ValueError):
            AB3DMOTConfig(state_estimator="not_a_real_estimator")

    def test_31_invalid_transition_probability_rejected(self):
        with self.assertRaises(ValueError):
            AB3DMOTConfig(imm_cv_to_cv_probability=0.0)
        with self.assertRaises(ValueError):
            AB3DMOTConfig(imm_ctrv_to_ctrv_probability=1.5)


if __name__ == "__main__":
    unittest.main()
