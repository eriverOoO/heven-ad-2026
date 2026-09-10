import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from centerpoint_raw_head import (
    RawHeadSnapshot,
    classify_native_hit_sparse_miss,
    decode_cell_geometry,
    gate_reason_summary,
    gt_neighborhood_evidence,
    project_xy_to_grid,
    reproduce_gate,
    sigmoid,
    top_k_rows,
)


def snapshot(heatmap, *, offset=None, rot=None, thresholds=None):
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
            "distance_bin_upper_limits": [2.0, 10.0],
            "score_thresholds": thresholds or [0.5] * (2 * classes),
            "yaw_norm_thresholds": [0.3] * classes,
        },
        arrays={
            "heatmap": heatmap,
            "reg": np.asarray(offset, dtype=np.float32)
            if offset is not None
            else np.stack((zeros, zeros)),
            "height": zeros[None],
            "dim": np.stack((zeros, zeros, zeros)),
            "rot": np.asarray(rot, dtype=np.float32)
            if rot is not None
            else np.stack((zeros + 1.0, zeros)),
            "vel": np.stack((zeros, zeros)),
        },
    )


class CenterPointRawHeadTest(unittest.TestCase):
    def test_sigmoid_is_stable(self):
        result = sigmoid(np.asarray([-1000.0, 0.0, 1000.0]))
        self.assertEqual(result.tolist(), [0.0, 0.5, 1.0])

    def test_gate_reason_accounting_and_threshold_semantics(self):
        # Scores: accepted, below score, outside distance bound.
        heatmap = np.asarray([[[2.0, -2.0, 2.0]]], dtype=np.float32)
        raw = snapshot(heatmap)
        raw.metadata["distance_bin_upper_limits"] = [1.0, 2.0]
        gate = reproduce_gate(raw)
        summary = gate_reason_summary(raw, gate)
        self.assertEqual(summary["total_cells"], 3)
        self.assertEqual(summary["accepted"], 1)
        self.assertEqual(summary["below_score"], 1)
        self.assertEqual(summary["outside_distance_bins"], 1)
        self.assertEqual(
            sum(
                summary[name]
                for name in (
                    "accepted",
                    "below_score",
                    "outside_distance_bins",
                    "invalid_yaw",
                )
            ),
            3,
        )

    def test_invalid_yaw_is_separate_from_score(self):
        heatmap = np.asarray([[[2.0]]], dtype=np.float32)
        rot = np.zeros((2, 1, 1), dtype=np.float32)
        gate = reproduce_gate(snapshot(heatmap, rot=rot))
        self.assertEqual(gate_reason_summary(snapshot(heatmap), gate)["invalid_yaw"], 1)

    def test_gt_projection_and_neighborhood_max(self):
        heatmap = np.full((1, 4, 4), -10.0, dtype=np.float32)
        heatmap[0, 2, 1] = 2.0
        raw = snapshot(heatmap)
        gate = reproduce_gate(raw)
        self.assertEqual(project_xy_to_grid(raw, 1.2, 2.2), (1, 2))
        evidence = gt_neighborhood_evidence(raw, gate, 1.2, 2.2, radius_cells=1)
        self.assertEqual((evidence["grid_x"], evidence["grid_y"]), (1, 2))
        self.assertGreater(evidence["threshold_margin"], 0.0)
        self.assertAlmostEqual(evidence["decoded_length"], 1.0)

    def test_cell_geometry_uses_pinned_dimension_and_yaw_decode(self):
        raw = snapshot(np.asarray([[[2.0]]], dtype=np.float32))
        raw.arrays["reg"][:, 0, 0] = (0.25, -0.5)
        raw.arrays["height"][0, 0, 0] = 1.5
        raw.arrays["dim"][:, 0, 0] = (math.log(2.0), math.log(4.0), math.log(1.0))
        raw.arrays["rot"][:, 0, 0] = (1.0, 0.0)
        decoded = decode_cell_geometry(raw, reproduce_gate(raw), 0, 0)
        self.assertAlmostEqual(decoded["x"], 0.25)
        self.assertAlmostEqual(decoded["y"], -0.5)
        self.assertAlmostEqual(decoded["z"], 1.5)
        self.assertAlmostEqual(decoded["length"], 4.0)
        self.assertAlmostEqual(decoded["width"], 2.0)
        self.assertAlmostEqual(decoded["height"], 1.0)
        self.assertAlmostEqual(decoded["raw_yaw"], math.pi / 2.0)
        self.assertAlmostEqual(decoded["ros_yaw"], -math.pi)

    def test_top_k_debug_output_is_bounded(self):
        raw = snapshot(np.asarray([[[3.0, 2.0], [1.0, 0.0]]], dtype=np.float32))
        rows = top_k_rows(raw, reproduce_gate(raw), top_k=2)
        self.assertLessEqual(len(rows), 4)  # one class, two views, two entries each
        self.assertEqual({row["kind"] for row in rows}, {"raw_class", "decoder_winner"})
        self.assertEqual(min(row["rank"] for row in rows), 1)

    def test_raw_binding_size_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = root / "7"
            metadata = {
                "timestamp_ns": 7,
                "dtype": "float32",
                "layout": "NCHW_without_batch",
                "class_size": 1,
                "height": 2,
                "width": 2,
                "bindings": {"heatmap": 1, "reg": 2, "height": 1, "dim": 3, "rot": 2, "vel": 2},
            }
            Path(f"{prefix}_meta.json").write_text(json.dumps(metadata))
            for name, channels in metadata["bindings"].items():
                suffix = "height" if name == "height" else name
                np.zeros(channels * 4, dtype=np.float32).tofile(f"{prefix}_{suffix}.f32")
            loaded = RawHeadSnapshot.load(root, 7)
            self.assertEqual(loaded.arrays["heatmap"].shape, (1, 2, 2))
            Path(f"{prefix}_heatmap.f32").write_bytes(b"bad")
            with self.assertRaises(ValueError):
                RawHeadSnapshot.load(root, 7)

    def test_instrumentation_is_default_off(self):
        patch = (
            Path(__file__).parent
            / "patches"
            / "autoware_lidar_centerpoint_raw_head_dump.patch"
        ).read_text()
        self.assertIn('declare_parameter<bool>("enable_raw_head_dump", false)', patch)
        self.assertNotIn("score_thresholds[", patch)
        self.assertIn(
            "   postProcess(det_boxes3d, post_score_boxes3d);\n"
            "+  if (raw_head_debug != nullptr)",
            patch,
        )

    def test_native_hit_sparse_miss_categories_do_not_force_pairing(self):
        common = {
            "corrected_threshold": 0.35,
            "native_margin": 0.2,
            "corrected_final_matched": False,
        }
        self.assertEqual(
            classify_native_hit_sparse_miss(
                corrected_local_score=0.20, corrected_s1_matched=False, **common
            ),
            "B1_peak_strongly_weakened",
        )
        self.assertEqual(
            classify_native_hit_sparse_miss(
                corrected_local_score=0.34, corrected_s1_matched=False, **common
            ),
            "B2_below_threshold",
        )
        self.assertEqual(
            classify_native_hit_sparse_miss(
                corrected_local_score=0.50, corrected_s1_matched=False, **common
            ),
            "B3_above_threshold_without_S1_match",
        )
        self.assertEqual(
            classify_native_hit_sparse_miss(
                corrected_local_score=0.50, corrected_s1_matched=True, **common
            ),
            "B4_lost_after_S1",
        )
        with self.assertRaises(ValueError):
            classify_native_hit_sparse_miss(
                corrected_local_score=0.50,
                corrected_s1_matched=True,
                corrected_final_matched=True,
                corrected_threshold=0.35,
                native_margin=0.2,
            )


if __name__ == "__main__":
    unittest.main()
