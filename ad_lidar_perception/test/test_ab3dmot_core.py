import math
import unittest

from ad_lidar_perception.ab3dmot_config import AB3DMOTConfig
from ad_lidar_perception.ab3dmot_core import (
    AB3DMOTTracker,
    Detection,
    Track,
    _greedy_matching,
    _orientation_correction,
    _load_ab3dmot_kf_class,
)

import numpy as np


KF_CLASS = _load_ab3dmot_kf_class()


def make_tracker(**overrides) -> AB3DMOTTracker:
    config = AB3DMOTConfig(**overrides)
    return AB3DMOTTracker(config)


class KfPredictUpdateTest(unittest.TestCase):
    def test_predict_rejects_nonpositive_and_nonfinite_dt(self):
        track = Track(1, Detection(0, 0, 0, 0.0, 4.0, 2.0, 1.5), KF_CLASS)
        for bad_dt in (0.0, -0.1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                track.predict(bad_dt)

    def test_predict_advances_position_by_current_velocity_estimate(self):
        track = Track(1, Detection(0, 0, 0, 0.0, 4.0, 2.0, 1.5), KF_CLASS)
        # force a known velocity directly into the filter state to isolate
        # the predict-step transition math from estimation/convergence
        track._kf.x[7, 0] = 2.0  # vx = 2 m/s
        before = track.to_tracked_state()
        track.predict(0.5)
        after = track.to_tracked_state()
        self.assertAlmostEqual(after.x - before.x, 1.0, places=6)  # 2 m/s * 0.5 s

    def test_update_resets_time_since_update_and_increments_hits(self):
        track = Track(1, Detection(0, 0, 0, 0.0, 4.0, 2.0, 1.5), KF_CLASS)
        track.predict(0.1)
        track.predict(0.1)
        self.assertEqual(track.time_since_update, 2)
        track.update(Detection(0.2, 0, 0, 0.0, 4.0, 2.0, 1.5))
        self.assertEqual(track.time_since_update, 0)
        self.assertEqual(track.hits, 2)


class VelocityUnitsTest(unittest.TestCase):
    def test_velocity_is_meters_per_second_not_per_frame(self):
        """Same physical speed (2 m/s), fed at two different frame rates,
        must converge to the same m/s estimate -- proving the state isn't
        AB3DMOT's raw meters-per-frame."""

        def run(dt: float, steps: int) -> float:
            tracker = make_tracker()
            t = 0.0
            x = 0.0
            tracker.step([Detection(x, 0, 0, 0.0, 4.0, 2.0, 1.5)], t)
            for _ in range(steps):
                t += dt
                x += 2.0 * dt  # true speed: 2 m/s
                out = tracker.step([Detection(x, 0, 0, 0.0, 4.0, 2.0, 1.5)], t)
            return out[0].vx_mps

        vx_slow = run(dt=0.5, steps=10)   # 2 Hz
        vx_fast = run(dt=0.05, steps=100)  # 20 Hz, same total physical motion
        self.assertAlmostEqual(vx_slow, 2.0, delta=0.3)
        self.assertAlmostEqual(vx_fast, 2.0, delta=0.3)
        self.assertAlmostEqual(vx_slow, vx_fast, delta=0.3)

    def test_constant_velocity_motion_converges(self):
        tracker = make_tracker()
        t, x = 0.0, 0.0
        tracker.step([Detection(x, 0, 0, 0.0, 4.0, 2.0, 1.5)], t)
        out = None
        for _ in range(15):
            t += 0.1
            x += 3.0 * 0.1  # 3 m/s
            out = tracker.step([Detection(x, 0, 0, 0.0, 4.0, 2.0, 1.5)], t)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0].vx_mps, 3.0, delta=0.3)


class LifecycleTest(unittest.TestCase):
    def test_stable_track_id_across_frames(self):
        tracker = make_tracker()
        t, x = 0.0, 0.0
        first = tracker.step([Detection(x, 0, 0, 0.0, 4.0, 2.0, 1.5)], t)[0].track_id
        for _ in range(5):
            t += 0.1
            x += 0.1
            out = tracker.step([Detection(x, 0, 0, 0.0, 4.0, 2.0, 1.5)], t)
            self.assertEqual(len(out), 1)
            self.assertEqual(out[0].track_id, first)

    def test_two_separated_objects_stay_separate_tracks(self):
        tracker = make_tracker()
        t = 0.0
        out = tracker.step(
            [Detection(0, 0, 0, 0.0, 4.0, 2.0, 1.5), Detection(50, 50, 0, 0.0, 4.0, 2.0, 1.5)], t
        )
        ids = {ts.track_id for ts in out}
        self.assertEqual(len(ids), 2)
        for _ in range(5):
            t += 0.1
            out = tracker.step(
                [Detection(0, 0, 0, 0.0, 4.0, 2.0, 1.5), Detection(50, 50, 0, 0.0, 4.0, 2.0, 1.5)], t
            )
            self.assertEqual({ts.track_id for ts in out}, ids)

    def test_temporary_missed_detection_keeps_same_track(self):
        tracker = make_tracker()
        t = 0.0
        track_id = tracker.step([Detection(0, 0, 0, 0.0, 4.0, 2.0, 1.5)], t)[0].track_id
        t += 0.1
        tracker.step([Detection(0.1, 0, 0, 0.0, 4.0, 2.0, 1.5)], t)

        t += 0.1
        out_missed = tracker.step([], t)  # one frame with no detections at all
        self.assertEqual(len(out_missed), 1)  # still coasting, max_age=2 not hit
        self.assertEqual(out_missed[0].track_id, track_id)

        t += 0.1
        out_recovered = tracker.step([Detection(0.3, 0, 0, 0.0, 4.0, 2.0, 1.5)], t)
        self.assertEqual(len(out_recovered), 1)
        self.assertEqual(out_recovered[0].track_id, track_id)
        self.assertEqual(len(tracker.tracks), 1)

    def test_max_age_deletion_creates_new_id_on_reappearance(self):
        tracker = make_tracker(max_age=2)
        t = 0.0
        first_id = tracker.step([Detection(0, 0, 0, 0.0, 4.0, 2.0, 1.5)], t)[0].track_id

        t += 0.1
        tracker.step([], t)  # time_since_update -> 1, still alive
        self.assertEqual(len(tracker.tracks), 1)

        t += 0.1
        out = tracker.step([], t)  # time_since_update -> 2 == max_age -> dead
        self.assertEqual(len(out), 0)
        self.assertEqual(len(tracker.tracks), 0)

        t += 0.1
        out = tracker.step([Detection(0, 0, 0, 0.0, 4.0, 2.0, 1.5)], t)
        self.assertEqual(len(out), 1)
        self.assertNotEqual(out[0].track_id, first_id)


class AssociationTest(unittest.TestCase):
    def test_greedy_matching_claims_lowest_cost_pairs_first(self):
        # det0 cheapest with trk1, det1 cheapest with trk0 -> both should
        # get their preferred (lowest-cost) partner since there's no conflict
        cost = np.array([[5.0, 1.0], [1.0, 5.0]])
        matches = _greedy_matching(cost)
        pairs = {(int(d), int(t)) for d, t in matches}
        self.assertEqual(pairs, {(0, 1), (1, 0)})

    def test_greedy_matching_resolves_conflict_by_global_lowest_cost(self):
        # det0-trk0 (cost 0.1) is the single globally-cheapest pair, so it's
        # claimed first even though det1-trk0 (0.2) is also cheap; det1 then
        # falls back to its only remaining option, trk1 -- greedy_matching
        # itself has no cost cutoff (that's a separate gating step), so
        # every detection still gets matched to *some* free track here.
        cost = np.array([[0.1, 9.0], [0.2, 9.0]])
        matches = _greedy_matching(cost)
        pairs = {(int(d), int(t)) for d, t in matches}
        self.assertEqual(pairs, {(0, 0), (1, 1)})

    def test_association_pairs_each_detection_with_its_nearest_track(self):
        tracker = make_tracker()
        t = 0.0
        # two well-separated tracks
        tracker.step(
            [Detection(0, 0, 0, 0.0, 4.0, 2.0, 1.5), Detection(100, 0, 0, 0.0, 4.0, 2.0, 1.5)], t
        )
        t += 0.1
        out = tracker.step(
            [Detection(100.1, 0, 0, 0.0, 4.0, 2.0, 1.5), Detection(0.1, 0, 0, 0.0, 4.0, 2.0, 1.5)], t
        )
        by_id = {ts.track_id: ts for ts in out}
        self.assertEqual(len(by_id), 2)
        for ts in by_id.values():
            # each track should have moved only slightly, never jumped
            # ~100 m to the other object
            self.assertLess(min(abs(ts.x - 0.1), abs(ts.x - 100.1)), 1.0)


class YawWraparoundTest(unittest.TestCase):
    def test_orientation_correction_keeps_result_within_90_degrees(self):
        cases = [
            (3.0, -3.0),
            (0.1, 3.0),
            (math.pi - 0.05, -math.pi + 0.05),
            (0.0, math.pi - 0.01),
        ]
        for theta_pre, theta_obs in cases:
            corrected_pre, corrected_obs = _orientation_correction(theta_pre, theta_obs)
            self.assertLessEqual(abs(corrected_obs - corrected_pre), math.pi / 2.0 + 1e-9)

    def test_track_update_across_wraparound_does_not_diverge(self):
        track = Track(1, Detection(0, 0, 0, math.pi - 0.05, 4.0, 2.0, 1.5), KF_CLASS)
        track.predict(0.1)
        # measurement just across the +/-pi discontinuity from the track's heading
        track.update(Detection(0, 0, 0, -math.pi + 0.05, 4.0, 2.0, 1.5))
        state = track.to_tracked_state()
        self.assertTrue(math.isfinite(state.yaw))
        # the filtered yaw should stay close to the true heading (~pi),
        # not jump to something near 0 by naively averaging across the wrap
        wrapped = (state.yaw + math.pi) % (2 * math.pi) - math.pi
        self.assertLess(min(abs(wrapped - math.pi), abs(wrapped + math.pi)), 0.2)


if __name__ == "__main__":
    unittest.main()
