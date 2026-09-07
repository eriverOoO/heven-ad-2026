import unittest

from stability_metrics import candidate_counts, distance_bin, filter_score, one_to_one, temporal_residual_jitter


def box(x, y=0, score=0.5):
    return {"x": x, "y": y, "class_name": "vehicle", "class_label": "vehicle", "score": score}


class StabilityMetricsTest(unittest.TestCase):
    def test_one_gt_one_prediction_has_no_duplicate(self):
        self.assertEqual(candidate_counts([box(0)], [box(1)]), [1])

    def test_many_to_one_does_not_hide_duplicates(self):
        self.assertEqual(candidate_counts([box(0)], [box(0), box(1), box(2)]), [3])
        self.assertEqual(len(one_to_one([box(0)], [box(0), box(1), box(2)])), 1)

    def test_two_gt_one_to_one_differs_from_many_to_one(self):
        gt, pred = [box(0), box(4)], [box(2)]
        self.assertEqual(candidate_counts(gt, pred, 3.0), [1, 1])
        self.assertEqual(len(one_to_one(gt, pred, 3.0)), 1)

    def test_empty_and_score_filter_and_bin_boundary(self):
        self.assertEqual(candidate_counts([], [box(0)]), [])
        self.assertEqual(candidate_counts([box(0)], []), [0])
        self.assertEqual(len(filter_score([box(0, score=.1), box(1, score=.2)], .2)), 1)
        self.assertEqual(distance_bin(20), "20-40m")

    def test_temporal_residual_jitter(self):
        rows = [{"actor_id": 1, "frame_idx": 0, "gt_x": 0, "gt_y": 0, "pred_x": 1, "pred_y": 0, "score": .2, "yaw": 0, "length": 4},
                {"actor_id": 1, "frame_idx": 1, "gt_x": 1, "gt_y": 0, "pred_x": 3, "pred_y": 0, "score": .4, "yaw": .1, "length": 5}]
        self.assertEqual(temporal_residual_jitter(rows)["center_residual_delta_m"]["mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
