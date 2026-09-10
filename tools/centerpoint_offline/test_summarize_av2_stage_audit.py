from pathlib import Path
import tempfile
import unittest

import numpy as np

from av2_vlp16_adapter import AdapterConfig
from summarize_av2_stage_audit import (
    assign_beam_candidates,
    box_axis_angle_difference,
    describe,
    outputs_equivalent,
    paired_gt_stability,
    source_ring_beam_audit,
    stage_rows,
    validate_stage_counts,
)


def detection(x, *, label=1, score=0.8):
    return {
        "x": x,
        "y": 0.0,
        "z": 0.0,
        "length": 4.0,
        "width": 2.0,
        "height": 1.5,
        "yaw": 0.0,
        "score": score,
        "label": label,
    }


class Av2StageSummaryTest(unittest.TestCase):
    def test_stage_count_consistency(self):
        validate_stage_counts(
            {"post_score": 5, "post_circle_nms": 2, "pre_iou": 2, "post_iou": 1, "final": 1}
        )
        with self.assertRaises(ValueError):
            validate_stage_counts(
                {"post_score": 2, "post_circle_nms": 3, "pre_iou": 3, "post_iou": 1, "final": 1}
            )

    def test_gt_conditioned_pairing_does_not_force_unmatched(self):
        gt = [
            {"track_uuid": "a", "x": 0.0, "y": 0.0},
            {"track_uuid": "b", "x": 10.0, "y": 0.0},
        ]
        rows = paired_gt_stability(gt, [detection(0.1), detection(10.1)], [detection(0.2)], 3.0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["track_uuid"], "a")

    def test_output_equivalence_is_order_independent_and_gated(self):
        left = [detection(0.0), detection(4.0)]
        right = [detection(4.001), detection(0.001)]
        self.assertTrue(outputs_equivalent(left, right))
        self.assertFalse(outputs_equivalent(left, [detection(0.001)]))

    def test_channel_elevation_assignment(self):
        elevations = np.radians(np.asarray([-15.0, 1.0, -12.0], dtype=np.float32))
        x = np.full(3, 10.0, dtype=np.float32)
        y = np.zeros(3, dtype=np.float32)
        z = np.tan(elevations) * x
        _, channels, keep = assign_beam_candidates(
            x, y, z, AdapterConfig(vertical_tolerance_deg=0.2)
        )
        self.assertEqual(channels[:2].tolist(), [0, 1])
        self.assertEqual(keep.tolist(), [True, True, False])

    def test_empty_stage_serialization_statistics(self):
        self.assertEqual(describe([])["count"], 0)
        self.assertIsNone(describe([])["mean"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = "header: {stamp: {sec: 0, nanosec: 0}, frame_id: test}\nobjects: []\n"
            for filename in (
                "post_score.yaml",
                "post_circle_nms.yaml",
                "pre_iou.yaml",
                "post_iou.yaml",
                "final_stage.yaml",
            ):
                (root / filename).write_text(payload)
            self.assertTrue(all(not rows for rows in stage_rows(root).values()))
            self.assertTrue(
                all(not rows for rows in stage_rows(root, expected_timestamp_ns=0).values())
            )
            with self.assertRaises(ValueError):
                stage_rows(root, expected_timestamp_ns=1)

    def test_overlay_stage_dump_defaults_off(self):
        patch = (
            Path(__file__).parent
            / "patches"
            / "autoware_lidar_centerpoint_stage_dump.patch"
        ).read_text()
        self.assertIn('enable_stage_dump" default="false"', patch)
        self.assertIn('declare_parameter<bool>("enable_stage_dump", false)', patch)

    def test_box_axis_treats_pi_flip_as_same_axis(self):
        self.assertAlmostEqual(box_axis_angle_difference(0.0, np.pi - 0.01), 0.01)

    def test_gt_conditioned_stability_records_both_errors(self):
        gt = [{"track_uuid": "a", "x": 0.0, "y": 0.0}]
        row = paired_gt_stability(gt, [detection(0.2)], [detection(0.5)], 3.0)[0]
        self.assertAlmostEqual(row["native_center_error_m"], 0.2)
        self.assertAlmostEqual(row["adapted_center_error_m"], 0.5)

    def test_source_ring_beam_audit_records_all_channels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lidar = root / "lidar"
            mode = "source_ring_vlp16_v2"
            lidar.mkdir()
            (root / mode).mkdir()
            paths = []
            for timestamp in (1, 2):
                path = lidar / f"{timestamp}.feather"
                path.touch()
                paths.append(path)
                np.savez(
                    root / mode / f"{timestamp}.npz",
                    x=np.zeros(16, np.float32),
                    y=np.zeros(16, np.float32),
                    z=np.zeros(16, np.float32),
                    intensity=np.zeros(16, np.uint8),
                    return_type=np.zeros(16, np.uint8),
                    channel=np.arange(16, dtype=np.uint16),
                    timestamp_ns=np.int64(timestamp),
                    source_log_id=np.str_("log"),
                    mode=np.str_(mode),
                    coordinate_frame=np.str_("av2_egovehicle"),
                )
            result = source_ring_beam_audit(paths, root, mode)
        self.assertEqual(result["points_per_channel"], [2] * 16)
        self.assertEqual(result["occupied_frames_per_channel"], [2] * 16)


if __name__ == "__main__":
    unittest.main()
