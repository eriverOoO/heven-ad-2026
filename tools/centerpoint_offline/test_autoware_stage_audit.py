import unittest

from autoware_stage_audit import STAGES, stage_metrics, validate_stage_record


def candidate(candidate_id, x, score=0.9):
    return {"candidate_id": candidate_id, "class_name": "vehicle", "score": score,
            "x": x, "y": 0.0, "z": 0.0, "length": 4.0, "width": 2.0,
            "height": 1.5, "yaw": 0.0}


def record():
    return {"schema": "heven.autoware_centerpoint_stage_dump.v1", "frame_idx": 4,
            "stages": {"decoded": [candidate(0, 0.0), candidate(1, 0.2), candidate(2, 8.0)],
                       "score": [candidate(0, 0.0), candidate(1, 0.2)],
                       "circle_nms": [candidate(0, 0.0)],
                       "iou_nms": [candidate(0, 0.0)], "final": [candidate(0, 0.0)]}}


class AutowareStageAuditTest(unittest.TestCase):
    def test_stage_counts_and_duplicate_reduction(self):
        summary = stage_metrics([record()], {4: [{"x": 0.0, "y": 0.0, "class_name": "vehicle"}]})
        self.assertEqual(tuple(summary), STAGES)
        self.assertEqual(summary["decoded"]["predictions_per_gt"]["mean"], 2.0)
        self.assertEqual(summary["score"]["predictions_per_gt"]["mean"], 2.0)
        self.assertEqual(summary["circle_nms"]["predictions_per_gt"]["mean"], 1.0)
        self.assertEqual(summary["final"]["recall"], 1.0)

    def test_final_provenance_must_equal_iou_stage(self):
        payload = record()
        payload["stages"]["final"] = []
        with self.assertRaisesRegex(ValueError, "final objects"):
            validate_stage_record(payload)

    def test_empty_gt_and_empty_predictions(self):
        payload = record()
        payload["stages"] = {stage: [] for stage in STAGES}
        summary = stage_metrics([payload], {4: []})
        self.assertIsNone(summary["final"]["recall"])
        self.assertEqual(summary["final"]["candidates_per_frame"], 0.0)


if __name__ == "__main__":
    unittest.main()
