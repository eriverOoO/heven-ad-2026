#!/usr/bin/env python3

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from morai_dataset import MoraiHevenDatasetCore, collate_openpcdet_contract
from prediction_bridge import benchmark_centers, convert_openpcdet_prediction
from openpcdet_runtime import import_smoke_components


class CenterPointOfflineTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "points").mkdir()
        (self.root / "labels").mkdir()
        (self.root / "splits").mkdir()
        (self.root / "metadata.json").write_text(
            json.dumps({"dataset_version": "morai_heven_v1"}), encoding="utf-8"
        )
        (self.root / "splits" / "train.txt").write_text("scene__1\n", encoding="utf-8")
        np.asarray([[1.0, 2.0, 3.0, 0.5]], dtype="<f4").tofile(
            self.root / "points" / "scene__1.bin"
        )
        label = {
            "sample_id": "scene__1",
            "source": {"header_stamp_ns": 123},
            "points": {"path": "points/scene__1.bin", "finite_count": 1},
            "ground_truth": {
                "target_frame": "lidar_link",
                "boxes": [
                    {
                        "class_name": "vehicle",
                        "x": 10.0,
                        "y": -2.0,
                        "z": 0.5,
                        "length": 4.5,
                        "width": 1.8,
                        "height": 1.6,
                        "yaw": 0.25,
                        "num_lidar_points_inside_box": 7,
                    }
                ],
            },
        }
        (self.root / "labels" / "scene__1.json").write_text(
            json.dumps(label), encoding="utf-8"
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_loader_and_batch_contract(self):
        sample = MoraiHevenDatasetCore(
            self.root, expected_dataset_version="morai_heven_v1"
        )[0]
        self.assertEqual(sample["points"].shape, (1, 4))
        self.assertEqual(sample["gt_boxes"].shape, (1, 7))
        self.assertEqual(sample["gt_num_lidar_points"].tolist(), [7])
        batch = collate_openpcdet_contract([sample])
        self.assertEqual(batch["points"].shape, (1, 5))
        self.assertEqual(batch["gt_boxes"].shape, (1, 1, 8))
        self.assertEqual(float(batch["gt_boxes"][0, 0, 7]), 1.0)

    def test_prediction_bridge(self):
        frame = convert_openpcdet_prediction(
            sample_id="scene__1",
            source_header_stamp_ns=123,
            prediction={
                "pred_boxes": np.asarray([[1, 2, 3, 4, 5, 6, 0.1]]),
                "pred_scores": np.asarray([0.8]),
                "pred_labels": np.asarray([1]),
            },
            class_names=["vehicle", "pedestrian", "obstacle"],
        )
        self.assertEqual(benchmark_centers(frame), ((1.0, 2.0),))
        self.assertEqual(frame.detections[0].class_name, "vehicle")

    def test_minimal_smoke_import_skips_dataset_and_detector_initializers(self):
        fake_root = self.root / "OpenPCDet"
        for relative in (
            "pcdet/datasets",
            "pcdet/models/detectors",
        ):
            (fake_root / relative).mkdir(parents=True, exist_ok=True)
        (fake_root / "pcdet/__init__.py").write_text("", encoding="utf-8")
        (fake_root / "pcdet/datasets/__init__.py").write_text(
            "raise AssertionError('must not import pcdet.datasets')\n",
            encoding="utf-8",
        )
        (fake_root / "pcdet/datasets/dataset.py").write_text(
            "class DatasetTemplate: pass\n", encoding="utf-8"
        )
        (fake_root / "pcdet/models/__init__.py").write_text(
            "raise AssertionError('must not import pcdet.models')\n", encoding="utf-8"
        )
        (fake_root / "pcdet/models/detectors/__init__.py").write_text(
            "raise AssertionError('must not import detector registry')\n",
            encoding="utf-8",
        )
        (fake_root / "pcdet/models/detectors/centerpoint.py").write_text(
            "class CenterPoint: pass\n", encoding="utf-8"
        )

        saved_modules = {
            name: module
            for name, module in sys.modules.items()
            if name == "pcdet" or name.startswith("pcdet.")
        }
        try:
            for name in list(saved_modules):
                sys.modules.pop(name, None)
            with mock.patch.object(sys, "path", [str(fake_root), *sys.path]):
                dataset_template, centerpoint = import_smoke_components(fake_root)
            self.assertEqual(dataset_template.__name__, "DatasetTemplate")
            self.assertEqual(centerpoint.__name__, "CenterPoint")
            self.assertNotIn("pcdet.datasets.argo2", sys.modules)
        finally:
            for name in list(sys.modules):
                if name == "pcdet" or name.startswith("pcdet."):
                    sys.modules.pop(name, None)
            sys.modules.update(saved_modules)


if __name__ == "__main__":
    unittest.main()
