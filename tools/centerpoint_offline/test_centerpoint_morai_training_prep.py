#!/usr/bin/env python3
"""Preflight gate tests for CenterPoint MORAI training prep.

Exercises the dataset / split / model-contract / environment gates of
``preflight_morai_training.py`` without training anything. Real fixture
exports are built through the actual CenterPoint adapter.
"""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for extra in (REPO_ROOT / "ad_morai_bridge_dev", REPO_ROOT / "ad_morai_bridge_dev" / "test"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from test_centerpoint_adapter import SIX_GROUPS, make_source_dataset
from ad_morai_bridge_dev.dataset.centerpoint_adapter import (
    CenterPointExporter,
    load_adapter_config,
)

import preflight_morai_training as pf

CONFIG = HERE / "configs" / "centerpoint_morai_v1.yaml"
DATA_CFG = HERE / "configs" / "centerpoint_morai_v1_dataset.yaml"
MODEL_CFG = HERE / "configs" / "centerpoint_morai_v1_model.yaml"

PLAN = {
    "splits": {
        "train": [
            {"scenario": "lead_constant", "seeds": [0, 1]},
            {"scenario": "cut_in", "seeds": [0]},
        ],
        "val": [{"scenario": "dense_multi_object", "seeds": [0]}],
        "test": [{"scenario": "cut_in", "seeds": [1]}],
    }
}


def _build_export(root: Path, plan=PLAN, frames=3) -> Path:
    src = root / "src"
    out = root / "export"
    make_source_dataset(src, SIX_GROUPS, frames=frames)
    CenterPointExporter(src, out, load_adapter_config(None), "test").export(plan)
    return out


def _args(dataset: Path, output_dir: Path, **kw):
    ns = pf.argparse.Namespace(
        dataset=dataset,
        config=CONFIG,
        output_dir=output_dir,
        init_checkpoint=kw.get("init_checkpoint"),
        openpcdet_root=REPO_ROOT / "references" / "openpcdet",
        assert_real_dataset=kw.get("assert_real_dataset", False),
        attempt_loader_smoke=kw.get("attempt_loader_smoke", True),
        attempt_model_smoke=False,
        attempt_forward_smoke=False,
    )
    return ns


class PreflightDataGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.export = _build_export(self.root)
        self.out = self.root / "out"

    def tearDown(self):
        self.tmp.cleanup()

    def _report(self, **kw):
        return pf.build_report(_args(self.export, self.out, **kw))

    def test_valid_non_overlap_train_and_val_is_dataset_ready(self):
        report = self._report()
        self.assertEqual(report["split_checks"]["group_counts"]["val"], 1)
        self.assertTrue(report["split_checks"]["evaluation_ready"])
        self.assertTrue(report["independent_val_group_exists"])
        self.assertEqual(report["dataset_status"], "READY")
        # fixture -> still globally blocked until a real dataset is asserted
        self.assertFalse(report["real_dataset_available"])
        self.assertEqual(report["final_status"], pf.STATUS_BLOCKED_DATASET)

    def test_empty_val_blocks_evaluation(self):
        no_val = copy.deepcopy(PLAN)
        no_val["splits"]["val"] = []
        no_val["splits"]["train"].append({"scenario": "dense_multi_object", "seeds": [0]})
        export = _build_export(self.root / "novalp", no_val)
        report = pf.build_report(_args(export, self.out))
        self.assertEqual(report["split_checks"]["group_counts"]["val"], 0)
        self.assertFalse(report["split_checks"]["evaluation_ready"])
        self.assertFalse(report["independent_val_group_exists"])
        self.assertIn(
            "no independent validation group (val groups == 0): NOT evaluation ready",
            report["split_checks"]["blockers"],
        )

    def test_leaked_group_across_splits_is_a_blocker(self):
        export = _build_export(self.root / "leak")
        # force a group into two splits at the label level
        val_ids = (export / "splits" / "val.txt").read_text().split()
        train_txt = export / "splits" / "train.txt"
        moved = val_ids[0]
        train_txt.write_text(train_txt.read_text() + moved + "\n", encoding="utf-8")
        report = pf.build_report(_args(export, self.out))
        blob = json.dumps(report["dataset_checks"]["adapter_validator"])
        self.assertFalse(report["dataset_checks"]["adapter_validator"]["ok"])
        self.assertTrue(
            any("multiple splits" in e or "leakage" in e
                for e in report["dataset_checks"]["adapter_validator"]["errors"]),
            blob,
        )
        self.assertEqual(report["final_status"], pf.STATUS_BLOCKED_DATASET)

    def test_wrong_adapter_schema_is_a_blocker(self):
        export = _build_export(self.root / "schema")
        manifest = json.loads((export / "export_manifest.json").read_text())
        manifest["adapter_schema_version"] = "something_else_v9"
        (export / "export_manifest.json").write_text(json.dumps(manifest))
        report = pf.build_report(_args(export, self.out))
        self.assertTrue(
            any("adapter schema" in b for b in report["dataset_checks"]["blockers"])
        )

    def test_incomplete_export_missing_manifest_is_a_blocker(self):
        export = _build_export(self.root / "incomplete")
        (export / "export_manifest.json").unlink()
        report = pf.build_report(_args(export, self.out))
        self.assertTrue(
            any("export_manifest.json missing" in b
                for b in report["dataset_checks"]["blockers"])
        )
        self.assertEqual(report["final_status"], pf.STATUS_BLOCKED_DATASET)

    def test_temp_path_forces_fixture_even_when_real_asserted(self):
        report = self._report(assert_real_dataset=True)
        self.assertTrue(report["fixture_only"])
        self.assertFalse(report["real_dataset_available"])

    def test_loader_smoke_reads_positive_and_validation_samples(self):
        report = self._report()
        loader = report["loader_checks"]
        self.assertTrue(loader["success"])
        self.assertEqual(loader["positive_sample"]["gt_boxes_shape"][1], 7)
        self.assertEqual(loader["positive_sample"]["coordinate_frame"], "lidar_link")
        self.assertEqual(loader["pre_voxel_collate"]["vehicle_class_id"], 1.0)
        self.assertIn("val_samples", loader)

    def test_evaluator_detected_but_fixture_still_blocked_dataset(self):
        report = self._report()
        self.assertTrue(report["evaluation_metric_implemented"])
        ev = report["evaluator_checks"]
        self.assertEqual(ev["evaluator_schema_version"], "morai_centerpoint_eval_v1")
        self.assertTrue(ev["model_config_selects_evaluator"])
        # evaluator present, env may be ready, but a fixture is never trainable
        self.assertEqual(report["final_status"], pf.STATUS_BLOCKED_DATASET)


class PreflightConfigGateTests(unittest.TestCase):
    def setUp(self):
        self.data_cfg = pf._load_yaml(DATA_CFG)
        self.model_cfg = pf._load_yaml(MODEL_CFG)
        self.experiment = pf._load_yaml(CONFIG)
        self.dataset = {
            "metadata": {
                "class_names": ["vehicle", "pedestrian", "obstacle"],
                "point_feature_names": ["x", "y", "z", "intensity"],
                "box_fields": ["x", "y", "z", "length", "width", "height", "yaw"],
                "gt_frame": "lidar_link",
                "intensity_transform": "identity",
                "reported_point_cloud_range": [-4.0, -25.0, -3.0, 100.0, 25.0, 5.0],
                "point_range_cropping_applied": False,
                "box_range_filtering_applied": False,
            },
            "split_manifest": {"class_counts_by_split": {"train": {"vehicle": 5}}},
        }

    def _check(self):
        return pf.check_contracts(
            self.dataset, self.data_cfg, self.model_cfg, self.experiment
        )

    def test_valid_config_passes(self):
        result = self._check()
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["class_id_map"], {"vehicle": 1})
        self.assertEqual(result["grid"]["grid_size_xyz"], [832, 400, 40])
        self.assertEqual(result["grid"]["bev_feature_map_xy"], [104, 50])

    def test_wrong_class_name_fails(self):
        self.model_cfg["CLASS_NAMES"] = ["Car"]
        self.model_cfg["MODEL"]["DENSE_HEAD"]["CLASS_NAMES_EACH_HEAD"] = [["Car"]]
        result = self._check()
        self.assertFalse(result["passed"])
        self.assertTrue(any("not in the export" in e for e in result["errors"]))

    def test_extra_class_in_export_fails(self):
        self.dataset["split_manifest"]["class_counts_by_split"] = {
            "train": {"vehicle": 5, "pedestrian": 2}
        }
        result = self._check()
        self.assertFalse(result["passed"])
        self.assertTrue(any("pedestrian" in e for e in result["errors"]))

    def test_wrong_num_point_features_fails(self):
        self.data_cfg["POINT_FEATURE_ENCODING"]["used_feature_list"] = [
            "x", "y", "z", "intensity", "timestamp",
        ]
        result = self._check()
        self.assertFalse(result["passed"])
        self.assertTrue(any("used_feature_list" in e for e in result["errors"]))

    def test_invalid_point_cloud_range_fails(self):
        self.data_cfg["POINT_CLOUD_RANGE"] = [-4.0, -25.0, -3.0, 100.0, 25.0, 6.0]
        result = self._check()
        self.assertFalse(result["passed"])
        self.assertTrue(any("POINT_CLOUD_RANGE" in e for e in result["errors"]))

    def test_non_integral_voxel_grid_fails(self):
        for processor in self.data_cfg["DATA_PROCESSOR"]:
            if processor.get("NAME") == "transform_points_to_voxels":
                processor["VOXEL_SIZE"] = [0.13, 0.13, 0.2]
        result = self._check()
        self.assertFalse(result["passed"])
        self.assertTrue(any("integer multiple" in e for e in result["errors"]))

    def test_grid_not_divisible_by_feature_map_stride_fails(self):
        for processor in self.data_cfg["DATA_PROCESSOR"]:
            if processor.get("NAME") == "transform_points_to_voxels":
                processor["VOXEL_SIZE"] = [0.1, 0.1, 0.2]  # grid x=1040, /8=130 ok; use stride bump
        self.model_cfg["MODEL"]["DENSE_HEAD"]["TARGET_ASSIGNER_CONFIG"][
            "FEATURE_MAP_STRIDE"
        ] = 7
        result = self._check()
        self.assertFalse(result["passed"])
        self.assertTrue(any("FEATURE_MAP_STRIDE" in e for e in result["errors"]))

    def test_box_range_filtering_flag_fails(self):
        self.dataset["metadata"]["box_range_filtering_applied"] = True
        result = self._check()
        self.assertFalse(result["passed"])

    def test_config_fingerprint_is_stable(self):
        first = pf._canonical_hash(
            {"e": self.experiment, "d": self.data_cfg, "m": self.model_cfg}
        )
        second = pf._canonical_hash(
            {"e": self.experiment, "d": self.data_cfg, "m": self.model_cfg}
        )
        self.assertEqual(first, second)

    def test_base_config_inheritance_is_rejected(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write("_BASE_CONFIG_: other.yaml\nMODEL: {}\n")
            path = Path(handle.name)
        with self.assertRaises(ValueError):
            pf._load_yaml(path)
        path.unlink()


class PreflightEnvironmentGateTests(unittest.TestCase):
    def test_torch_missing_blocks_environment(self):
        real_import = __import__

        def fake_import(name, *args, **kw):
            if name == "torch" or name.startswith("spconv") or name == "pcdet":
                raise ImportError(f"no {name}")
            return real_import(name, *args, **kw)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            env = pf.audit_environment(REPO_ROOT / "references" / "openpcdet")
        self.assertFalse(env["torch_available"])
        self.assertFalse(env["environment_ready"])
        self.assertIn("torch not importable", env["blockers"])

    def test_dataset_ready_is_independent_of_environment(self):
        # a config-valid, val-populated export is "dataset ready" even if the
        # training environment is not.
        tmp = tempfile.TemporaryDirectory()
        try:
            export = _build_export(Path(tmp.name))
            real_import = __import__

            def fake_import(name, *args, **kw):
                if name == "torch" or name.startswith("spconv") or name == "pcdet":
                    raise ImportError(f"no {name}")
                return real_import(name, *args, **kw)

            with mock.patch("builtins.__import__", side_effect=fake_import):
                report = pf.build_report(_args(export, Path(tmp.name) / "o"))
            self.assertEqual(report["dataset_status"], "READY")
            self.assertEqual(report["env_status"], "BLOCKED")
            self.assertFalse(report["train_env_ready"])
            self.assertTrue(report["split_checks"]["evaluation_ready"])
        finally:
            tmp.cleanup()

    def test_output_dir_inside_repo_is_refused(self):
        code = pf.main(
            [
                "--dataset",
                str(REPO_ROOT),
                "--output-dir",
                str(REPO_ROOT / "tools" / "centerpoint_offline" / "x"),
            ]
        )
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
