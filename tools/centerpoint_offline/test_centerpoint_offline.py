#!/usr/bin/env python3

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from morai_dataset import MoraiHevenDatasetCore, collate_openpcdet_contract
from prediction_bridge import (
    SCHEMA,
    benchmark_centers,
    benchmark_records,
    convert_openpcdet_prediction,
    read_jsonl,
)
from openpcdet_runtime import import_smoke_components
from infer_morai_centerpoint import filter_prediction
from train_morai_centerpoint import (
    checkpoint_paths,
    create_dataloader,
    load_yaml_config,
    parse_args,
)


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

    def test_training_arguments_and_base_config(self):
        args = parse_args(
            [
                "--dataset", str(self.root),
                "--openpcdet-root", str(self.root / "OpenPCDet"),
                "--epochs", "2",
                "--batch-size", "3",
                "--workers", "0",
                "--output-dir", str(self.root / "outputs"),
                "--seed", "7",
            ]
        )
        self.assertEqual((args.epochs, args.batch_size, args.seed), (2, 3, 7))
        config = load_yaml_config(
            Path(__file__).parent / "configs" / "morai_centerpoint_train.yaml"
        )
        self.assertEqual(config["MODEL"]["NAME"], "CenterPoint")
        self.assertEqual(config["OPTIMIZATION"]["OPTIMIZER"], "adam")

    def test_training_dataloader_creation(self):
        import torch

        class TinyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 2

            def __getitem__(self, index):
                return {"value": index}

            @staticmethod
            def collate_batch(samples):
                return {"values": [sample["value"] for sample in samples]}

        loader = create_dataloader(TinyDataset(), batch_size=2, workers=0, seed=5)
        self.assertEqual(sorted(next(iter(loader))["values"]), [0, 1])

    def test_checkpoint_paths(self):
        latest, epoch = checkpoint_paths(self.root / "run", 4)
        self.assertEqual(latest.name, "checkpoint_latest.pth")
        self.assertEqual(epoch.name, "checkpoint_epoch_004.pth")
        self.assertEqual(latest.parent, self.root / "run")

    def test_training_module_does_not_eagerly_import_openpcdet(self):
        self.assertNotIn("pcdet.datasets.argo2", sys.modules)

    def _prediction(self):
        return {
            "pred_boxes": np.asarray(
                [
                    [1, 2, 3, 4, 5, 6, 0.1],
                    [7, 8, 9, 4, 3, 2, -0.2],
                    [10, 11, 12, 1, 2, 3, 0.3],
                ],
                dtype=np.float32,
            ),
            "pred_scores": np.asarray([0.5, 0.9, 0.5], dtype=np.float32),
            "pred_labels": np.asarray([1, 2, 3], dtype=np.int64),
        }

    def test_jsonl_schema_timestamp_box_and_class_mapping(self):
        frame = convert_openpcdet_prediction(
            sample_id="scene__1",
            source_header_stamp_ns=123,
            prediction=self._prediction(),
            class_names=("vehicle", "pedestrian", "obstacle"),
            inference_time_ms=4.5,
        )
        path = self.root / "predictions.jsonl"
        path.write_text(json.dumps({
            "schema": frame.schema,
            "sample_id": frame.sample_id,
            "source_header_stamp_ns": frame.source_header_stamp_ns,
            "frame_id": frame.frame_id,
            "inference_time_ms": frame.inference_time_ms,
            "detections": [
                {"class_name": item.class_name, "score": item.score, "box_lidar": item.box_lidar}
                for item in frame.detections
            ],
        }) + "\n", encoding="utf-8")
        decoded = list(read_jsonl(path))[0]
        self.assertEqual(decoded.schema, SCHEMA)
        self.assertEqual(decoded.source_header_stamp_ns, 123)
        self.assertEqual(decoded.frame_id, "lidar_link")
        self.assertEqual(decoded.detections[0].class_name, "vehicle")
        self.assertEqual(decoded.detections[1].class_name, "pedestrian")
        self.assertEqual(decoded.detections[2].class_name, "obstacle")
        self.assertEqual(decoded.detections[0].box_lidar[:6], (1, 2, 3, 4, 5, 6))
        adapter = list(benchmark_records(path))[0]
        self.assertEqual(adapter["source_header_stamp_ns"], 123)
        self.assertEqual(adapter["centers_xy"][0], (1.0, 2.0))

    def test_empty_prediction_frame(self):
        frame = convert_openpcdet_prediction(
            sample_id="empty",
            source_header_stamp_ns=1,
            prediction={
                "pred_boxes": np.empty((0, 7)),
                "pred_scores": np.empty(0),
                "pred_labels": np.empty(0),
            },
            class_names=("vehicle", "pedestrian", "obstacle"),
        )
        self.assertEqual(frame.detections, ())

    def test_score_threshold_and_max_detections(self):
        filtered = filter_prediction(self._prediction(), 0.5, 2)
        self.assertEqual(filtered["pred_scores"].tolist(), [0.8999999761581421, 0.5])
        self.assertEqual(filtered["pred_labels"].tolist(), [2, 1])
        thresholded = filter_prediction(self._prediction(), 0.6, 10)
        self.assertEqual(thresholded["pred_labels"].tolist(), [2])

    def test_prediction_filter_order_is_deterministic(self):
        first = filter_prediction(self._prediction(), 0.0, 3)["pred_labels"].tolist()
        second = filter_prediction(self._prediction(), 0.0, 3)["pred_labels"].tolist()
        self.assertEqual(first, [2, 1, 3])
        self.assertEqual(first, second)

    def test_inference_module_does_not_eagerly_import_argoverse2(self):
        self.assertNotIn("pcdet.datasets.argo2", sys.modules)


if __name__ == "__main__":
    unittest.main()
