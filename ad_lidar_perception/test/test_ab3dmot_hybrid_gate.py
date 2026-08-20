"""T-5B: covariance-aware hybrid Mahalanobis gate (chi-square AND optional
absolute BEV physical-distance cap). The physical cap is a validity gate
only -- it never changes the Hungarian cost value (still raw d_M^2).
`mahalanobis_max_distance_m <= 0` must reproduce pure T-4/T-5A Mahalanobis
behavior exactly.
"""
import unittest

import numpy as np

from ad_lidar_perception.ab3dmot_config import AB3DMOTConfig
from ad_lidar_perception.ab3dmot_core import (
    AB3DMOTTracker,
    Detection,
    Track,
    _euclidean_bev_distance_matrix,
    _load_ab3dmot_kf_class,
)

KF_CLASS = _load_ab3dmot_kf_class()


def make_detection(x=0.0, y=0.0, z=0.0, yaw=0.0, length=4.0, width=2.0, height=1.5) -> Detection:
    return Detection(x=x, y=y, z=z, yaw=yaw, length=length, width=width, height=height)


def make_track(track_id=1, x=0.0, y=0.0) -> Track:
    return Track(track_id, make_detection(x=x, y=y), KF_CLASS)


def make_tracker(**overrides) -> AB3DMOTTracker:
    config = AB3DMOTConfig(association_metric="mahalanobis", matcher="hungarian", **overrides)
    return AB3DMOTTracker(config)


class HybridGateTest(unittest.TestCase):
    def test_1_inside_both_gates_valid(self):
        tracker = make_tracker(mahalanobis_gate=1000.0, mahalanobis_max_distance_m=5.0)
        tracker.step([make_detection(x=0.0, y=0.0)], 1.0)
        outputs = tracker.step([make_detection(x=1.0, y=0.0)], 2.0)  # 1m < 5m cap, d_M2 tiny
        ids = {o.track_id for o in outputs}
        self.assertEqual(ids, {1})

    def test_2_inside_mahalanobis_outside_absolute_rejected(self):
        tracker = make_tracker(mahalanobis_gate=1_000_000.0, mahalanobis_max_distance_m=3.0)
        tracker.step([make_detection(x=0.0, y=0.0)], 1.0)
        # huge mahalanobis_gate means d_M2 always passes; 20m > 3m cap must reject.
        outputs = tracker.step([make_detection(x=20.0, y=0.0)], 2.0)
        ids = {o.track_id for o in outputs}
        self.assertEqual(ids, {1, 2})  # coasting old track 1 + new track 2

    def test_3_outside_mahalanobis_inside_absolute_rejected(self):
        tracker = make_tracker(mahalanobis_gate=1e-9, mahalanobis_max_distance_m=1000.0)
        tracker.step([make_detection(x=0.0, y=0.0)], 1.0)
        outputs = tracker.step([make_detection(x=0.5, y=0.0)], 2.0)  # inside 1000m cap, but d_M2 gate is ~0
        ids = {o.track_id for o in outputs}
        self.assertEqual(ids, {1, 2})

    def test_4_outside_both_rejected(self):
        tracker = make_tracker(mahalanobis_gate=1e-9, mahalanobis_max_distance_m=1.0)
        tracker.step([make_detection(x=0.0, y=0.0)], 1.0)
        outputs = tracker.step([make_detection(x=50.0, y=0.0)], 2.0)
        ids = {o.track_id for o in outputs}
        self.assertEqual(ids, {1, 2})

    def test_5_disabled_cap_reproduces_pure_mahalanobis(self):
        # <=0 (default) must behave identically to a config with no cap field set.
        config_nocap = AB3DMOTConfig(association_metric="mahalanobis", matcher="hungarian",
                                      mahalanobis_gate=1000.0, mahalanobis_max_distance_m=0.0)
        config_default = AB3DMOTConfig(association_metric="mahalanobis", matcher="hungarian",
                                        mahalanobis_gate=1000.0)
        self.assertEqual(config_default.mahalanobis_max_distance_m, 0.0)
        t1 = AB3DMOTTracker(config_nocap)
        t2 = AB3DMOTTracker(config_default)
        for t in (t1, t2):
            t.step([make_detection(x=0.0, y=0.0)], 1.0)
            out = t.step([make_detection(x=90.0, y=0.0)], 2.0)  # far, but huge gate accepts it
            self.assertEqual({o.track_id for o in out}, {1})  # matched despite huge distance

    def test_6_covariance_increase_reduces_d_m2_but_cap_still_applies(self):
        # Reproduces the T-5A failure-mode mechanism directly: inflate a
        # track's covariance so a very distant detection passes the
        # (loosened) chi-square gate, but the absolute cap still rejects it.
        tracker = make_tracker(mahalanobis_gate=11.62, mahalanobis_max_distance_m=5.0)
        tracker.step([make_detection(x=0.0, y=0.0)], 1.0)
        # directly inflate predicted covariance (post-predict, pre-associate
        # is not reachable from outside; instead use a huge initial P via a
        # manual track injection through the tracker's private track list).
        track = tracker._tracks[0]
        track._kf.P *= 1.0e6  # simulate the T-5A "hits=1, huge P" scenario
        outputs = tracker.step([make_detection(x=40.0, y=0.0)], 2.0)  # 40m -- inside loosened chi-square gate
        ids = {o.track_id for o in outputs}
        # With the cap, the 40m pair must be rejected even though d_M2 easily
        # passes the (now enormous-covariance) chi-square gate.
        self.assertEqual(ids, {1, 2})

        # Sanity: without the cap, the same scenario DOES accept the far match.
        tracker_nocap = make_tracker(mahalanobis_gate=11.62, mahalanobis_max_distance_m=0.0)
        tracker_nocap.step([make_detection(x=0.0, y=0.0)], 1.0)
        tracker_nocap._tracks[0]._kf.P *= 1.0e6
        outputs_nocap = tracker_nocap.step([make_detection(x=40.0, y=0.0)], 2.0)
        self.assertEqual({o.track_id for o in outputs_nocap}, {1})

    def test_7_rectangular_matrix(self):
        dets = [make_detection(x=0.0, y=0.0), make_detection(x=10.0, y=0.0), make_detection(x=20.0, y=0.0)]
        trks = [make_track(1, x=0.0, y=0.0), make_track(2, x=10.0, y=0.0)]
        matrix = _euclidean_bev_distance_matrix(dets, trks)
        self.assertEqual(matrix.shape, (3, 2))

    def test_8_unmatched_detections_and_tracks_correct(self):
        tracker = make_tracker(mahalanobis_gate=1000.0, mahalanobis_max_distance_m=2.0)
        tracker.step([make_detection(x=0.0, y=0.0), make_detection(x=100.0, y=0.0)], 1.0)
        outputs = tracker.step(
            [make_detection(x=0.5, y=0.0), make_detection(x=200.0, y=0.0)], 2.0
        )
        ids = sorted(o.track_id for o in outputs)
        self.assertEqual(ids, [1, 2, 3])  # track1 matched(close), track2 coasts, new track3 for far det

    def test_9_no_duplicate_assignment(self):
        tracker = make_tracker(mahalanobis_gate=1000.0, mahalanobis_max_distance_m=100.0)
        tracker.step(
            [make_detection(x=0.0, y=0.0), make_detection(x=5.0, y=0.0), make_detection(x=10.0, y=0.0)], 1.0
        )
        outputs = tracker.step(
            [make_detection(x=0.2, y=0.0), make_detection(x=5.2, y=0.0), make_detection(x=10.2, y=0.0)], 2.0
        )
        ids = [o.track_id for o in outputs]
        self.assertEqual(len(ids), len(set(ids)))

    def test_10_no_nan_inf(self):
        tracker = make_tracker(mahalanobis_gate=11.62, mahalanobis_max_distance_m=5.0)
        tracker.step(
            [make_detection(x=float(i), y=float(-i)) for i in range(5)], 1.0
        )
        outputs = tracker.step(
            [make_detection(x=float(i) + 0.3, y=float(-i)) for i in range(5)], 2.0
        )
        for o in outputs:
            self.assertTrue(all(np.isfinite(v) for v in (o.x, o.y, o.vx_mps, o.vy_mps)))

    def test_11_pure_giou_euclidean_behavior_unchanged(self):
        config_giou = AB3DMOTConfig()
        self.assertEqual(config_giou.association_metric, "giou_3d")
        tracker = AB3DMOTTracker(config_giou)
        tracker.step([make_detection(x=0.0, y=0.0)], 1.0)
        outputs = tracker.step([make_detection(x=0.1, y=0.0)], 2.0)
        self.assertEqual({o.track_id for o in outputs}, {1})

        config_eucl = AB3DMOTConfig(association_metric="euclidean", matcher="hungarian")
        tracker2 = AB3DMOTTracker(config_eucl)
        tracker2.step([make_detection(x=0.0, y=0.0)], 1.0)
        outputs2 = tracker2.step([make_detection(x=0.5, y=0.0)], 2.0)
        self.assertEqual({o.track_id for o in outputs2}, {1})

    def test_12_default_tracker_behavior_unchanged(self):
        config = AB3DMOTConfig()
        self.assertEqual(config.matcher, "greedy")
        self.assertEqual(config.mahalanobis_max_distance_m, 0.0)
        tracker = AB3DMOTTracker(config)
        tracker.step([make_detection(x=0.0, y=0.0)], 1.0)
        outputs = tracker.step([make_detection(x=0.1, y=0.0)], 2.0)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].track_id, 1)

    def test_13_config_rejects_non_finite_cap(self):
        with self.assertRaises(ValueError):
            AB3DMOTConfig(mahalanobis_max_distance_m=float("nan"))
        with self.assertRaises(ValueError):
            AB3DMOTConfig(mahalanobis_max_distance_m=float("inf"))


if __name__ == "__main__":
    unittest.main()
