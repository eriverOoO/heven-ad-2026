import math
import unittest

from ad_lidar_perception.camera_lidar_fusion_core import CameraIntrinsics, LidarDetection
from ad_lidar_perception.camera_lidar_fusion_stage2_core import (
    GeometryFilterRule,
    SemanticObservation,
    TemporalDetection,
    TemporalLinkConfig,
    compare_match_sets,
    link_adjacent_frames,
    passes_geometry_filter,
    perturb_intrinsics,
    semantic_stability_statistics,
)


class GeometryFilterTest(unittest.TestCase):
    def test_no_thresholds_keeps_positive_box(self):
        detection = LidarDetection(0, 0, 0, 0.1, 0.1, 0.1)
        self.assertTrue(passes_geometry_filter(detection, GeometryFilterRule("none")))

    def test_min_dimension_boundary_is_rejected(self):
        detection = LidarDetection(0, 0, 0, 4.0, 2.0, 0.15)
        rule = GeometryFilterRule("conservative", min_dimension_m=0.15)
        self.assertFalse(passes_geometry_filter(detection, rule))

    def test_footprint_volume_and_aspect_thresholds(self):
        detection = LidarDetection(0, 0, 0, 4.0, 0.5, 1.0)
        self.assertFalse(passes_geometry_filter(
            detection, GeometryFilterRule("aspect", max_footprint_aspect_ratio=7.0)
        ))
        self.assertFalse(passes_geometry_filter(
            detection, GeometryFilterRule("volume", min_volume_m3=2.1)
        ))
        self.assertTrue(passes_geometry_filter(
            detection, GeometryFilterRule("pass", min_footprint_m2=2.0, min_volume_m3=2.0)
        ))


class TemporalLinkTest(unittest.TestCase):
    @staticmethod
    def det(frame, index, x, y, length=4.0, width=2.0, height=1.5):
        return TemporalDetection(frame, index, x, y, length, width, height)

    def test_links_adjacent_one_to_one_deterministically(self):
        frames = [
            [self.det(0, 0, 0, 0), self.det(0, 1, 10, 0)],
            [self.det(1, 3, 9.7, 0), self.det(1, 2, 0.2, 0)],
        ]
        config = TemporalLinkConfig(1.0, math.log(2.0))
        first = link_adjacent_frames(frames, config)
        second = link_adjacent_frames(frames, config)
        self.assertEqual(first, second)
        self.assertEqual(first[(0, 0)], first[(1, 2)])
        self.assertEqual(first[(0, 1)], first[(1, 3)])

    def test_rejects_distance_and_size_gate_failures(self):
        frames = [
            [self.det(0, 0, 0, 0)],
            [self.det(1, 0, 2, 0), self.det(1, 1, 0, 0, length=10.0)],
        ]
        links = link_adjacent_frames(frames, TemporalLinkConfig(1.0, math.log(2.0)))
        self.assertNotEqual(links[(0, 0)], links[(1, 0)])
        self.assertNotEqual(links[(0, 0)], links[(1, 1)])

    def test_track_does_not_coast_over_empty_frame(self):
        frames = [[self.det(0, 0, 0, 0)], [], [self.det(2, 0, 0, 0)]]
        links = link_adjacent_frames(frames, TemporalLinkConfig(1.0, math.log(2.0)))
        self.assertNotEqual(links[(0, 0)], links[(2, 0)])

    def test_horizontal_dimension_swap_is_consistent(self):
        frames = [
            [self.det(0, 0, 0, 0, length=4, width=2)],
            [self.det(1, 0, 0, 0, length=2, width=4)],
        ]
        links = link_adjacent_frames(frames, TemporalLinkConfig(1.0, 0.01))
        self.assertEqual(links[(0, 0)], links[(1, 0)])


class IntrinsicsAndMatchSetTest(unittest.TestCase):
    def test_intrinsics_perturbation(self):
        base = CameraIntrinsics(640, 640, 640, 360, 1280, 720, "B_reconstructed")
        result = perturb_intrinsics(base, focal_scale=1.05, cx_offset_fraction=0.01)
        self.assertEqual(result.fx, 672)
        self.assertEqual(result.fy, 672)
        self.assertEqual(result.cx, 652.8)
        self.assertEqual(result.cy, 360)
        self.assertEqual(result.calibration_quality, "B_reconstructed")

    def test_match_set_comparison(self):
        result = compare_match_sets({(1, 2), (3, 4)}, {(1, 2), (5, 6)})
        self.assertEqual(result["baseline_retained"], 1)
        self.assertEqual(result["new_matches"], 1)
        self.assertEqual(result["lost_matches"], 1)
        self.assertAlmostEqual(result["jaccard_overlap"], 1 / 3)


class SemanticStatisticsTest(unittest.TestCase):
    def test_semantic_sequence_metrics(self):
        observations = [
            SemanticObservation(0, "car", 0.9),
            SemanticObservation(1, "car", 0.8),
            SemanticObservation(2, None),
            SemanticObservation(3, "truck", 0.7),
            SemanticObservation(4, "car", 0.6),
        ]
        result = semantic_stability_statistics(observations)
        self.assertEqual(result["observed_frames"], 5)
        self.assertEqual(result["camera_matched_frames"], 4)
        self.assertEqual(result["modal_semantic_class"], "car")
        self.assertEqual(result["modal_class_consistency_ratio"], 0.75)
        self.assertEqual(result["longest_consecutive_semantic_consistent_run"], 2)
        self.assertEqual(result["class_flips"], 1)
        self.assertEqual(result["matched_unmatched_transitions"], 2)

    def test_empty_sequence(self):
        result = semantic_stability_statistics([])
        self.assertEqual(result["camera_match_ratio"], 0.0)
        self.assertIsNone(result["modal_semantic_class"])


if __name__ == "__main__":
    unittest.main()
