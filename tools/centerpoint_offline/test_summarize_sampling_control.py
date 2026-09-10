import unittest

from summarize_sampling_control import aggregate, describe


class SamplingControlSummaryTest(unittest.TestCase):
    def test_describe_empty_and_values(self):
        self.assertEqual(describe([])["count"], 0)
        result = describe([1.0, 2.0, 3.0])
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["median"], 2.0)

    def test_aggregate_uses_fixed_per_frame_counts_and_gt_matches(self):
        frames = [{
            "gt": 2, "matched": 1, "vehicle_fp": 3, "input_points": 10,
            "roi_points": 9, "post_score_count": 8, "post_circle_nms_count": 4,
            "pre_iou_count": 4, "post_iou_count": 3, "final_count": 3,
        }]
        gt = [{
            "r0_car_score": 0.4, "r0_score_ge_035": True, "winner_center_xy_error_m": 1.0,
            "winner_peak_drift_m": 0.3, "winner_dimension_relative_error": 0.2,
            "winner_yaw_axis_error_rad": 0.1, "winner_z_error_m": 0.4, "points_in_gt": 5,
        }, {
            "r0_car_score": 0.2, "r0_score_ge_035": False, "winner_center_xy_error_m": 2.0,
            "winner_peak_drift_m": 0.5, "winner_dimension_relative_error": 0.4,
            "winner_yaw_axis_error_rad": 0.2, "winner_z_error_m": 0.8, "points_in_gt": 0,
        }]
        result = aggregate("control", frames, gt)
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["r0_ge_035"], 1)
        self.assertEqual(result["zero_point_fraction"], 0.5)
        self.assertEqual(result["s1_per_frame"]["mean"], 8.0)


if __name__ == "__main__":
    unittest.main()
