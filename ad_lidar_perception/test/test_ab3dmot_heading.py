"""T-7A.5: unobserved-yaw estimator mode and displacement-based motion-
heading initialization tests.
"""
import math
import unittest

import numpy as np

from ad_lidar_perception.ab3dmot_config import AB3DMOTConfig
from ad_lidar_perception.ab3dmot_core import (
    AB3DMOTTracker,
    Detection,
    EKFEstimator,
    Track,
    _load_ab3dmot_kf_class,
    _wrap_to_pi,
)

KF_CLASS = _load_ab3dmot_kf_class()


def make_detection(x=0.0, y=0.0, z=0.0, yaw=0.0, length=4.0, width=2.0, height=1.5) -> Detection:
    return Detection(x=x, y=y, z=z, yaw=yaw, length=length, width=width, height=height)


def make_tracker(state_estimator="ekf", yaw_measurement_mode="unobserved", **overrides) -> AB3DMOTTracker:
    config = AB3DMOTConfig(
        association_metric="euclidean", matcher="hungarian", euclidean_gate_m=3.0,
        state_estimator=state_estimator, yaw_measurement_mode=yaw_measurement_mode, **overrides,
    )
    return AB3DMOTTracker(config)


class UnobservedYawTest(unittest.TestCase):
    def test_1_yaw0_does_not_force_heading_zero(self):
        # Even though every detection has yaw=0.0, once heading is
        # initialized from real +y displacement, the EKF's own yaw state
        # should reflect that motion direction, not stay pinned at 0.
        est = EKFEstimator(make_detection(x=0.0, y=0.0, yaw=0.0), KF_CLASS, 1, "unobserved")
        est.predict(1.0)
        est.update(make_detection(x=0.0, y=2.0, yaw=0.0))  # 2m +y displacement
        self.assertAlmostEqual(est.yaw, math.pi / 2.0, places=2)

    def test_2_first_observation_no_fabricated_heading(self):
        est = EKFEstimator(make_detection(x=5.0, y=5.0), KF_CLASS, 1, "unobserved")
        self.assertFalse(est._heading_initialized)
        self.assertAlmostEqual(est.yaw, 0.0)
        self.assertAlmostEqual(est.speed, 0.0)

    def test_3_second_sufficiently_displaced_observation_initializes(self):
        est = EKFEstimator(make_detection(x=0.0, y=0.0), KF_CLASS, 1, "unobserved")
        est.predict(1.0)
        est.update(make_detection(x=1.0, y=0.0))  # 1.0m > 0.5m threshold
        self.assertTrue(est._heading_initialized)

    def test_4_plus_x_motion_heading_near_zero(self):
        est = EKFEstimator(make_detection(x=0.0, y=0.0), KF_CLASS, 1, "unobserved")
        est.predict(1.0)
        est.update(make_detection(x=3.0, y=0.0))
        self.assertAlmostEqual(est.yaw, 0.0, places=2)
        self.assertGreater(est.speed, 0.0)

    def test_5_plus_y_motion_heading_near_half_pi(self):
        est = EKFEstimator(make_detection(x=0.0, y=0.0), KF_CLASS, 1, "unobserved")
        est.predict(1.0)
        est.update(make_detection(x=0.0, y=3.0))
        self.assertAlmostEqual(est.yaw, math.pi / 2.0, places=2)

    def test_6_minus_x_motion_heading_near_pi_positive_speed(self):
        est = EKFEstimator(make_detection(x=0.0, y=0.0), KF_CLASS, 1, "unobserved")
        est.predict(1.0)
        est.update(make_detection(x=-3.0, y=0.0))
        self.assertAlmostEqual(abs(est.yaw), math.pi, places=2)
        self.assertGreaterEqual(est.speed, 0.0)

    def test_7_minus_y_motion_appropriate_heading_positive_speed(self):
        est = EKFEstimator(make_detection(x=0.0, y=0.0), KF_CLASS, 1, "unobserved")
        est.predict(1.0)
        est.update(make_detection(x=0.0, y=-3.0))
        self.assertAlmostEqual(est.yaw, -math.pi / 2.0, places=2)
        self.assertGreaterEqual(est.speed, 0.0)

    def test_8_insufficient_displacement_does_not_initialize(self):
        est = EKFEstimator(make_detection(x=0.0, y=0.0), KF_CLASS, 1, "unobserved")
        est.predict(1.0)
        est.update(make_detection(x=0.1, y=0.0))  # 0.1m < 0.5m threshold
        self.assertFalse(est._heading_initialized)
        self.assertAlmostEqual(est.yaw, 0.0, places=6)

    def test_9_yaw_wrap_across_pi(self):
        self.assertAlmostEqual(_wrap_to_pi(math.pi + 0.1), -math.pi + 0.1, places=6)
        est = EKFEstimator(make_detection(x=0.0, y=0.0), KF_CLASS, 1, "unobserved")
        est.predict(1.0)
        est.update(make_detection(x=-3.0, y=-0.01))  # near-negative-x, slightly below axis
        self.assertTrue(-math.pi <= est.yaw < math.pi)

    def test_10_detector_mode_preserves_existing_behavior(self):
        est = EKFEstimator(make_detection(x=0.0, y=0.0, yaw=0.0), KF_CLASS, 1, "detector")
        est.predict(1.0)
        est.update(make_detection(x=1.0, y=0.0, yaw=0.0))
        # detector mode: yaw measurement is 0.0, so state yaw stays ~0
        # regardless of motion direction (T-7A's original documented behavior).
        self.assertAlmostEqual(est.yaw, 0.0, places=2)

    def test_11_unobserved_update_excludes_yaw_measurement(self):
        # Feed a wildly different "detector yaw" -- since unobserved mode
        # never reads detection.yaw for measurement, the update must not
        # be perturbed by it (only position/dims drive the correction).
        est = EKFEstimator(make_detection(x=0.0, y=0.0), KF_CLASS, 1, "unobserved")
        est.predict(1.0)
        est.update(make_detection(x=0.1, y=0.0, yaw=2.9))  # insufficient displacement, huge fake yaw
        self.assertAlmostEqual(est.yaw, 0.0, places=6)  # unaffected by yaw=2.9

    def test_12_dimensions_update_correctly(self):
        est = EKFEstimator(make_detection(length=4.0, width=2.0, height=1.5), KF_CLASS, 1, "unobserved")
        est.predict(1.0)
        est.update(make_detection(x=1.0, length=4.5, width=2.2, height=1.6))
        length, width, height = est.dimensions
        self.assertAlmostEqual(length, 4.5, delta=0.5)
        self.assertAlmostEqual(width, 2.2, delta=0.5)
        self.assertAlmostEqual(height, 1.6, delta=0.5)

    def test_13_no_nan_inf(self):
        est = EKFEstimator(make_detection(x=0.0, y=0.0), KF_CLASS, 1, "unobserved")
        for i in range(10):
            est.predict(0.15)
            est.update(make_detection(x=float(i) * 0.3, y=float(i) * 0.1))
        self.assertTrue(est.is_finite())

    def test_14_deterministic_repeated_sequence(self):
        def run():
            est = EKFEstimator(make_detection(x=0.0, y=0.0), KF_CLASS, 1, "unobserved")
            for i in range(5):
                est.predict(0.15)
                est.update(make_detection(x=float(i) * 0.4, y=float(i) * 0.1))
            return est.x.copy(), est.P.copy()

        x1, p1 = run()
        x2, p2 = run()
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(p1, p2)

    def test_15_lifecycle_unchanged(self):
        tracker = make_tracker(max_age=2, min_hits=1)
        tracker.step([make_detection(x=0.0, y=0.0)], 1.0)
        out2 = tracker.step([make_detection(x=100.0, y=100.0)], 2.0)
        self.assertIn(1, {o.track_id for o in out2})
        out3 = tracker.step([], 3.0)
        self.assertNotIn(1, {o.track_id for o in out3})

    def test_16_association_unchanged(self):
        tracker = make_tracker()
        tracker.step([make_detection(x=0.0, y=0.0), make_detection(x=50.0, y=0.0)], 1.0)
        outputs = tracker.step([make_detection(x=0.2, y=0.0), make_detection(x=50.2, y=0.0)], 2.0)
        ids = sorted(o.track_id for o in outputs)
        self.assertEqual(ids, [1, 2])

    def test_17_linear_kf_default_unchanged(self):
        config = AB3DMOTConfig()
        self.assertEqual(config.yaw_measurement_mode, "detector")
        tracker = AB3DMOTTracker(config)
        tracker.step([make_detection(x=0.0, y=0.0)], 1.0)
        outputs = tracker.step([make_detection(x=0.1, y=0.0)], 2.0)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].track_id, 1)

    def test_18_unsupported_mode_rejected(self):
        with self.assertRaises(ValueError):
            AB3DMOTConfig(yaw_measurement_mode="always_available")

    def test_19_tracker_integration_ekf_unobserved_no_nan(self):
        tracker = make_tracker()
        tracker.step([make_detection(x=0.0, y=0.0), make_detection(x=5.0, y=0.0)], 1.0)
        outputs = tracker.step([make_detection(x=-0.5, y=0.3), make_detection(x=5.2, y=0.0)], 2.0)
        for o in outputs:
            for v in (o.x, o.y, o.z, o.yaw, o.vx_mps, o.vy_mps, o.vz_mps):
                self.assertTrue(math.isfinite(v))


if __name__ == "__main__":
    unittest.main()
