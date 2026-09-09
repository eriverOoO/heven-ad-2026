import unittest

from evaluate_av2_xyzirc_pair import (
    detections_from_ros_dict,
    evaluate,
    gated_hungarian_matches,
    select_gt_rows,
)


class Av2PairEvaluationTest(unittest.TestCase):
    def test_gated_matching_prefers_maximum_cardinality(self):
        gt = [{"x": 0.0, "y": 0.0}, {"x": 2.9, "y": 0.0}]
        detections = [
            {"x": 0.1, "y": 0.0, "score": 0.9},
            {"x": -2.8, "y": 0.0, "score": 0.8},
        ]
        matches = gated_hungarian_matches(gt, detections, gate_m=3.0)
        self.assertEqual(len(matches), 2)
        self.assertEqual(evaluate(gt, detections)["matched"], 2)

    def test_empty_inputs(self):
        self.assertEqual(gated_hungarian_matches([], [], 3.0), [])
        result = evaluate([], [])
        self.assertIsNone(result["recall"])
        self.assertEqual(result["false_positives"], 0)

    def test_exact_timestamp_and_roi_selection(self):
        columns = {
            "timestamp_ns": [7, 7, 8],
            "category": ["REGULAR_VEHICLE", "PEDESTRIAN", "REGULAR_VEHICLE"],
            "tx_m": [1.0, 2.0, 3.0],
            "ty_m": [1.0, 2.0, 3.0],
            "tz_m": [0.0, 0.0, 0.0],
        }
        self.assertEqual(len(select_gt_rows(columns, 7)), 1)
        self.assertEqual(select_gt_rows(columns, 9), [])

    def test_ros_detection_extraction_filters_label(self):
        def obj(label, x):
            return {
                "existence_probability": 0.75,
                "classification": [{"label": label, "probability": 1.0}],
                "shape": {"dimensions": {"x": 4.0, "y": 2.0, "z": 1.5}},
                "kinematics": {
                    "pose_with_covariance": {
                        "pose": {
                            "position": {"x": x, "y": 2.0, "z": 0.0},
                            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                        }
                    },
                    "twist_with_covariance": {
                        "twist": {"linear": {"x": 0.0, "y": 0.0, "z": 0.0}}
                    },
                },
            }

        rows = detections_from_ros_dict({"objects": [obj(1, 3.0), obj(7, 4.0)]})
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["x"], 3.0)


if __name__ == "__main__":
    unittest.main()
