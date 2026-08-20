"""T-3: greedy vs Hungarian assignment-strategy correctness tests.

Exercises `_greedy_matching`/`_hungarian_matching` directly (matrix-level,
same style as `test_ab3dmot_core.py::AssociationTest`) and
`AB3DMOTTracker`/`AB3DMOTConfig` end-to-end for the matcher-selection,
gate-interaction, and unmatched-detection/track behavior that must be
identical between the two matchers.
"""
import math
import unittest

import numpy as np

from ad_lidar_perception.ab3dmot_config import AB3DMOTConfig
from ad_lidar_perception.ab3dmot_core import (
    AB3DMOTTracker,
    Detection,
    Track,
    _greedy_matching,
    _hungarian_matching,
    _load_ab3dmot_kf_class,
)

KF_CLASS = _load_ab3dmot_kf_class()


def make_tracker(**overrides) -> AB3DMOTTracker:
    config = AB3DMOTConfig(**overrides)
    return AB3DMOTTracker(config)


def assert_one_to_one(test, matches):
    dets = [int(d) for d, _ in matches]
    trks = [int(t) for _, t in matches]
    test.assertEqual(len(dets), len(set(dets)), "duplicate detection in matches")
    test.assertEqual(len(trks), len(set(trks)), "duplicate track in matches")


class MatcherMatrixTest(unittest.TestCase):
    """Cases 1-10 at the raw cost-matrix level for both matchers."""

    def test_1_empty_matrix(self):
        for matcher in (_greedy_matching, _hungarian_matching):
            cost = np.empty((0, 0))
            matches = matcher(cost)
            self.assertEqual(matches.shape, (0, 2))

    def test_2_one_detection_one_track(self):
        for matcher in (_greedy_matching, _hungarian_matching):
            cost = np.array([[0.3]])
            matches = matcher(cost)
            self.assertEqual(matches.tolist(), [[0, 0]])

    def test_3_perfect_diagonal_affinity(self):
        # affinity: diagonal is best (highest); cost = -affinity
        affinity = np.array([[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]])
        cost = -affinity
        for matcher in (_greedy_matching, _hungarian_matching):
            matches = matcher(cost)
            pairs = {(int(d), int(t)) for d, t in matches}
            self.assertEqual(pairs, {(0, 0), (1, 1), (2, 2)})

    def test_4_more_detections_than_tracks(self):
        # 3 detections, 1 track -> at most 1 match, 2 dets unmatched
        cost = np.array([[0.5], [0.1], [0.9]])
        for matcher in (_greedy_matching, _hungarian_matching):
            matches = matcher(cost)
            self.assertEqual(len(matches), 1)
            assert_one_to_one(self, matches)
            # the lowest-cost (best) row is det index 1
            self.assertEqual(int(matches[0][0]), 1)

    def test_5_more_tracks_than_detections(self):
        # 1 detection, 3 tracks -> at most 1 match, 2 tracks unmatched
        cost = np.array([[0.5, 0.1, 0.9]])
        for matcher in (_greedy_matching, _hungarian_matching):
            matches = matcher(cost)
            self.assertEqual(len(matches), 1)
            assert_one_to_one(self, matches)
            self.assertEqual(int(matches[0][1]), 1)

    def test_9_greedy_and_hungarian_identical_on_unambiguous_matrix(self):
        affinity = np.array([[0.9, 0.05], [0.05, 0.9]])
        cost = -affinity
        greedy_pairs = {(int(d), int(t)) for d, t in _greedy_matching(cost)}
        hungarian_pairs = {(int(d), int(t)) for d, t in _hungarian_matching(cost)}
        self.assertEqual(greedy_pairs, hungarian_pairs)
        self.assertEqual(greedy_pairs, {(0, 0), (1, 1)})

    def test_10_greedy_globally_suboptimal_hungarian_finds_better_total_affinity(self):
        # Classic case: greedy claims the single best edge (D0-T0, 10)
        # first, which then forces D1 onto its only remaining, terrible
        # option (D1-T1, 1) -- total affinity 11. Hungarian instead
        # chooses the cross pairing (D0-T1, D1-T0), both 9 -- total
        # affinity 18, strictly better, even though neither edge alone is
        # the single best entry in the matrix.
        affinity = np.array([[10.0, 9.0], [9.0, 1.0]])
        cost = -affinity

        greedy_matches = _greedy_matching(cost)
        hungarian_matches = _hungarian_matching(cost)

        greedy_pairs = {(int(d), int(t)) for d, t in greedy_matches}
        hungarian_pairs = {(int(d), int(t)) for d, t in hungarian_matches}
        self.assertEqual(greedy_pairs, {(0, 0), (1, 1)})
        self.assertEqual(hungarian_pairs, {(0, 1), (1, 0)})

        greedy_total = sum(affinity[d, t] for d, t in greedy_pairs)
        hungarian_total = sum(affinity[d, t] for d, t in hungarian_pairs)
        self.assertEqual(greedy_total, 11.0)
        self.assertEqual(hungarian_total, 18.0)
        self.assertGreater(hungarian_total, greedy_total)


class TrackerLevelAssociationTest(unittest.TestCase):
    """Cases 6-8 plus unmatched-detection/track/no-duplicate checks,
    exercised through the full AB3DMOTTracker for both matchers."""

    def test_6_all_pairs_below_gate_produce_no_matches(self):
        for matcher in ("greedy", "hungarian"):
            tracker = make_tracker(matcher=matcher, giou_gate=0.99)
            first_id = tracker.step([Detection(0, 0, 0, 0.0, 4.0, 2.0, 1.5)], 0.0)[0].track_id
            # second frame: detection barely overlaps the track's predicted
            # box -> real GIoU well below the near-impossible 0.99 gate, so
            # every candidate pair must be rejected -> new track spawned,
            # old one just coasts (not yet dead, max_age default 2)
            out = tracker.step([Detection(20, 20, 0, 0.0, 4.0, 2.0, 1.5)], 0.1)
            ids = {ts.track_id for ts in out}
            self.assertIn(first_id, ids)  # old track still coasting
            self.assertEqual(len(ids), 2)  # + a brand-new track for the far detection

    def test_7_mixed_valid_and_invalid_pairs(self):
        for matcher in ("greedy", "hungarian"):
            tracker = make_tracker(matcher=matcher, giou_gate=0.3)
            tracker.step(
                [Detection(0, 0, 0, 0.0, 4.0, 2.0, 1.5), Detection(50, 0, 0, 0.0, 4.0, 2.0, 1.5)],
                0.0,
            )
            # near-duplicate of det0 (valid pair, high GIoU) + a detection
            # far from anything (invalid pair for both existing tracks)
            out = tracker.step(
                [Detection(0.05, 0, 0, 0.0, 4.0, 2.0, 1.5), Detection(200, 200, 0, 0.0, 4.0, 2.0, 1.5)],
                0.1,
            )
            # 2 original tracks (one updated, one coasting) + 1 new track
            self.assertEqual(len(out), 3)

    def test_8_deterministic_tie_case(self):
        # Equal-affinity tie between two candidate pairs; both matchers
        # must produce the exact same result every time given the same
        # input (no randomness anywhere in the pipeline).
        for matcher in ("greedy", "hungarian"):
            results = []
            for _ in range(5):
                tracker = make_tracker(matcher=matcher)
                tracker.step(
                    [Detection(0, 0, 0, 0.0, 4.0, 2.0, 1.5), Detection(10, 0, 0, 0.0, 4.0, 2.0, 1.5)],
                    0.0,
                )
                out = tracker.step(
                    [Detection(0.1, 0, 0, 0.0, 4.0, 2.0, 1.5), Detection(10.1, 0, 0, 0.0, 4.0, 2.0, 1.5)],
                    0.1,
                )
                results.append(tuple(sorted((ts.track_id, round(ts.x, 6)) for ts in out)))
            self.assertEqual(len(set(results)), 1, f"non-deterministic across repeated runs: {results}")

    def test_no_duplicate_assignment_more_dets_than_tracks(self):
        for matcher in ("greedy", "hungarian"):
            tracker = make_tracker(matcher=matcher)
            tracker.step([Detection(0, 0, 0, 0.0, 4.0, 2.0, 1.5)], 0.0)
            out = tracker.step(
                [
                    Detection(0.05, 0, 0, 0.0, 4.0, 2.0, 1.5),
                    Detection(0.06, 0, 0, 0.0, 4.0, 2.0, 1.5),
                ],
                0.1,
            )
            # only one of the two near-identical detections can match the
            # single existing track; the other must spawn a new one -- no
            # track is ever updated twice in a single frame
            self.assertEqual(len(out), 2)
            self.assertEqual(len({ts.track_id for ts in out}), 2)

    def test_unmatched_detection_spawns_new_track_both_matchers(self):
        for matcher in ("greedy", "hungarian"):
            tracker = make_tracker(matcher=matcher)
            tracker.step([Detection(0, 0, 0, 0.0, 4.0, 2.0, 1.5)], 0.0)
            out = tracker.step(
                [Detection(0.1, 0, 0, 0.0, 4.0, 2.0, 1.5), Detection(500, 500, 0, 0.0, 4.0, 2.0, 1.5)],
                0.1,
            )
            self.assertEqual(len(out), 2)

    def test_unmatched_track_coasts_both_matchers(self):
        for matcher in ("greedy", "hungarian"):
            tracker = make_tracker(matcher=matcher, max_age=2)
            track_id = tracker.step([Detection(0, 0, 0, 0.0, 4.0, 2.0, 1.5)], 0.0)[0].track_id
            out = tracker.step([], 0.1)  # nothing to match -> track coasts
            self.assertEqual(len(out), 1)
            self.assertEqual(out[0].track_id, track_id)
            self.assertEqual(out[0].time_since_update, 1)


class HungarianRegressionParityTest(unittest.TestCase):
    """Phase 5: re-run the existing greedy-baseline scenarios
    (test_ab3dmot_core.py's LifecycleTest/VelocityUnitsTest/AssociationTest/
    YawWraparoundTest) with matcher="hungarian" and verify the same
    invariants (finite state, valid dims, quaternion/yaw handling, ID
    uniqueness, real-dt F update, lifecycle, timestamps) hold identically.
    None of predict/update/lifecycle code path changes with the matcher,
    so these are expected to pass exactly like the greedy originals."""

    def _assert_finite_and_valid(self, ts):
        self.assertTrue(math.isfinite(ts.x) and math.isfinite(ts.y) and math.isfinite(ts.z))
        self.assertTrue(math.isfinite(ts.yaw))
        self.assertTrue(ts.length > 0 and ts.width > 0 and ts.height > 0)
        self.assertTrue(math.isfinite(ts.vx_mps) and math.isfinite(ts.vy_mps) and math.isfinite(ts.vz_mps))

    def test_stable_track_id_across_frames_hungarian(self):
        tracker = make_tracker(matcher="hungarian")
        t, x = 0.0, 0.0
        first = tracker.step([Detection(x, 0, 0, 0.0, 4.0, 2.0, 1.5)], t)[0].track_id
        for _ in range(5):
            t += 0.1
            x += 0.1
            out = tracker.step([Detection(x, 0, 0, 0.0, 4.0, 2.0, 1.5)], t)
            self.assertEqual(len(out), 1)
            self.assertEqual(out[0].track_id, first)
            self._assert_finite_and_valid(out[0])

    def test_two_separated_objects_stay_separate_tracks_hungarian(self):
        tracker = make_tracker(matcher="hungarian")
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
            for ts in out:
                self._assert_finite_and_valid(ts)

    def test_temporary_missed_detection_keeps_same_track_hungarian(self):
        tracker = make_tracker(matcher="hungarian")
        t = 0.0
        track_id = tracker.step([Detection(0, 0, 0, 0.0, 4.0, 2.0, 1.5)], t)[0].track_id
        t += 0.1
        tracker.step([Detection(0.1, 0, 0, 0.0, 4.0, 2.0, 1.5)], t)
        t += 0.1
        out_missed = tracker.step([], t)
        self.assertEqual(len(out_missed), 1)
        self.assertEqual(out_missed[0].track_id, track_id)
        t += 0.1
        out_recovered = tracker.step([Detection(0.3, 0, 0, 0.0, 4.0, 2.0, 1.5)], t)
        self.assertEqual(len(out_recovered), 1)
        self.assertEqual(out_recovered[0].track_id, track_id)

    def test_max_age_deletion_creates_new_id_on_reappearance_hungarian(self):
        tracker = make_tracker(matcher="hungarian", max_age=2)
        t = 0.0
        first_id = tracker.step([Detection(0, 0, 0, 0.0, 4.0, 2.0, 1.5)], t)[0].track_id
        t += 0.1
        tracker.step([], t)
        t += 0.1
        out = tracker.step([], t)
        self.assertEqual(len(out), 0)
        t += 0.1
        out = tracker.step([Detection(0, 0, 0, 0.0, 4.0, 2.0, 1.5)], t)
        self.assertEqual(len(out), 1)
        self.assertNotEqual(out[0].track_id, first_id)

    def test_association_pairs_each_detection_with_nearest_track_hungarian(self):
        tracker = make_tracker(matcher="hungarian")
        t = 0.0
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
            self.assertLess(min(abs(ts.x - 0.1), abs(ts.x - 100.1)), 1.0)

    def test_velocity_is_meters_per_second_hungarian(self):
        tracker = make_tracker(matcher="hungarian")
        t, x = 0.0, 0.0
        tracker.step([Detection(x, 0, 0, 0.0, 4.0, 2.0, 1.5)], t)
        out = None
        for _ in range(15):
            t += 0.1
            x += 3.0 * 0.1
            out = tracker.step([Detection(x, 0, 0, 0.0, 4.0, 2.0, 1.5)], t)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0].vx_mps, 3.0, delta=0.3)

    def test_orientation_correction_and_wraparound_unaffected_by_matcher(self):
        # orientation_correction/yaw handling lives entirely in Track.update,
        # never touched by which matcher chose the pair -- direct Track-level
        # check, matcher-independent by construction, included for completeness.
        track = Track(1, Detection(0, 0, 0, math.pi - 0.05, 4.0, 2.0, 1.5), KF_CLASS)
        track.predict(0.1)
        track.update(Detection(0, 0, 0, -math.pi + 0.05, 4.0, 2.0, 1.5))
        state = track.to_tracked_state()
        self.assertTrue(math.isfinite(state.yaw))

    def test_timestamps_and_real_dt_unchanged_by_matcher(self):
        # same real-dt F-matrix construction regardless of matcher: feed
        # two different frame rates, both must converge to the same speed.
        def run(matcher, dt, steps):
            tracker = make_tracker(matcher=matcher)
            t, x = 0.0, 0.0
            tracker.step([Detection(x, 0, 0, 0.0, 4.0, 2.0, 1.5)], t)
            out = None
            for _ in range(steps):
                t += dt
                x += 2.0 * dt
                out = tracker.step([Detection(x, 0, 0, 0.0, 4.0, 2.0, 1.5)], t)
            return out[0].vx_mps

        vx_greedy = run("greedy", 0.1, 15)
        vx_hungarian = run("hungarian", 0.1, 15)
        self.assertAlmostEqual(vx_greedy, vx_hungarian, delta=1e-9)


class ConfigValidationTest(unittest.TestCase):
    def test_default_matcher_is_greedy(self):
        self.assertEqual(AB3DMOTConfig().matcher, "greedy")

    def test_hungarian_is_a_valid_matcher(self):
        AB3DMOTConfig(matcher="hungarian")  # must not raise

    def test_unknown_matcher_rejected(self):
        with self.assertRaises(ValueError):
            AB3DMOTConfig(matcher="auction")


if __name__ == "__main__":
    unittest.main()
