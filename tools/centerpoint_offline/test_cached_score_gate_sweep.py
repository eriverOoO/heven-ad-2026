import copy
import unittest

import numpy as np

from cached_score_gate_sweep import (
    compare_rows,
    decode_s1,
    iou_nms,
    offline_postprocess,
    threshold_vector,
)
from centerpoint_raw_head import RawHeadSnapshot


def make_snapshot(heatmap: np.ndarray) -> RawHeadSnapshot:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    classes, height, width = heatmap.shape
    zeros = np.zeros((height, width), dtype=np.float32)
    return RawHeadSnapshot(
        metadata={
            "timestamp_ns": 1,
            "class_size": classes,
            "height": height,
            "width": width,
            "voxel_size_x": 1.0,
            "voxel_size_y": 1.0,
            "range_min_x": 0.0,
            "range_min_y": 0.0,
            "downsample_factor": 1,
            "distance_bin_upper_limits": [10.0],
            "score_thresholds": [0.35] * classes,
            "yaw_norm_thresholds": [0.3] * classes,
        },
        arrays={
            "heatmap": heatmap,
            "reg": np.stack((zeros, zeros)),
            "height": zeros[None],
            "dim": np.stack((zeros, zeros, zeros)),
            "rot": np.stack((zeros + 1.0, zeros)),
            "vel": np.stack((zeros, zeros)),
        },
    )


class CachedScoreGateSweepTest(unittest.TestCase):
    def test_lower_threshold_has_nondecreasing_s1_count(self):
        logits = np.asarray([[[-2.0, -1.0, 0.0, 1.0]]], dtype=np.float32)
        raw = make_snapshot(logits)
        counts = [len(decode_s1(raw, threshold_vector(1, value, "global"))) for value in (0.45, 0.35, 0.20)]
        self.assertEqual(counts, sorted(counts))

    def test_car_only_policy_keeps_other_classes_at_baseline(self):
        # Cell 0 is CAR winner and cell 1 is TRUCK winner, both score about .30.
        logits = np.asarray([[[-0.85, -3.0]], [[-3.0, -0.85]]], dtype=np.float32)
        raw = make_snapshot(logits)
        global_rows = decode_s1(raw, threshold_vector(2, 0.25, "global"))
        car_only_rows = decode_s1(raw, threshold_vector(2, 0.25, "car_only"))
        self.assertEqual([row["raw_label"] for row in global_rows], [0, 1])
        self.assertEqual([row["raw_label"] for row in car_only_rows], [0])

    def test_sweep_does_not_mutate_cached_threshold_metadata(self):
        raw = make_snapshot(np.asarray([[[0.0, 1.0]]], dtype=np.float32))
        before = copy.deepcopy(raw.metadata)
        offline_postprocess(raw, 0.20, "global")
        offline_postprocess(raw, 0.40, "global")
        self.assertEqual(raw.metadata, before)

    def test_threshold_replay_is_deterministic(self):
        raw = make_snapshot(np.asarray([[[0.0, 1.0, 2.0]]], dtype=np.float32))
        left = offline_postprocess(raw, 0.35, "global")
        right = offline_postprocess(raw, 0.35, "global")
        for stage in left:
            self.assertTrue(compare_rows(left[stage], right[stage])["equal"])

    def test_stage_counts_follow_pinned_flow(self):
        raw = make_snapshot(np.asarray([[[2.0, 2.0, -2.0]]], dtype=np.float32))
        stages = offline_postprocess(raw, 0.35, "global")
        self.assertEqual(len(stages["post_score"]), 2)
        self.assertEqual(len(stages["post_circle_nms"]), 2)
        self.assertEqual(stages["post_circle_nms"], stages["pre_iou"])
        self.assertGreaterEqual(len(stages["pre_iou"]), len(stages["post_iou"]))
        self.assertEqual(len(stages["post_iou"]), len(stages["final"]))

    def test_compare_rows_checks_geometry_score_and_class(self):
        raw = make_snapshot(np.asarray([[[2.0]]], dtype=np.float32))
        rows = offline_postprocess(raw, 0.35, "global")["final"]
        changed = [dict(rows[0], score=rows[0]["score"] - 0.1)]
        self.assertTrue(compare_rows(rows, rows)["equal"])
        self.assertFalse(compare_rows(rows, changed)["equal"])

    def test_iou_nms_compares_with_earlier_suppressed_input(self):
        def box(x: float, score: float) -> dict:
            return {
                "x": x, "y": 0.0, "length": 2.0, "width": 2.0, "yaw": 0.0,
                "score": score, "label": 1,
            }

        # A overlaps B; B overlaps C; A does not overlap C above the 0.1 IoU
        # threshold. Pinned Autoware still compares C with suppressed B.
        self.assertEqual(len(iou_nms([box(0.0, .9), box(1.5, .8), box(3.0, .7)])), 1)


if __name__ == "__main__":
    unittest.main()
