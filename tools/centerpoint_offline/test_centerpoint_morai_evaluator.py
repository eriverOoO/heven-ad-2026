#!/usr/bin/env python3
"""Tests for the CenterPoint MORAI evaluator (morai_evaluator.py).

Geometry / AP / matching are exercised with explicit prediction arrays - no
torch, no OpenPCDet. Fixture datasets are built through the real Dataset
Factory writer + CenterPoint adapter; predictions are synthesised.
"""

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for extra in (
    REPO_ROOT / "ad_morai_bridge_dev",
    REPO_ROOT / "ad_morai_bridge_dev" / "test",
    REPO_ROOT / "ad_lidar_perception",
):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from test_centerpoint_adapter import SIX_GROUPS, make_source_dataset  # noqa: E402
from ad_morai_bridge_dev.dataset.centerpoint_adapter import (  # noqa: E402
    CenterPointExporter,
    load_adapter_config,
)

import morai_evaluator as ev  # noqa: E402
from morai_dataset import MoraiHevenDatasetCore  # noqa: E402

VEH = "vehicle"
BASE = [10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]


def _gt(*boxes, names=None):
    boxes = np.array(boxes, dtype=float).reshape(-1, 7)
    return ev.SampleGT(boxes, names or [VEH] * len(boxes))


def _pred(names, scores, *boxes):
    return ev.SamplePred(
        np.array(boxes, dtype=float).reshape(-1, 7),
        list(names),
        np.array(scores, dtype=float),
    )


def _run(gt, pred, **kw):
    return ev.evaluate_detections({"s0": gt}, {"s0": pred}, **kw)


# --------------------------------------------------------------------------
# oriented IoU
# --------------------------------------------------------------------------
def test_identical_boxes_iou_is_exactly_one():
    b = np.array(BASE)
    assert abs(ev.bev_iou(b, b) - 1.0) < 1e-9
    assert abs(ev.iou_3d(b, b) - 1.0) < 1e-9


def test_separated_boxes_iou_zero():
    a = np.array(BASE)
    far = np.array([100.0, 0, 0, 4, 2, 1.5, 0.0])
    assert ev.bev_iou(a, far) == 0.0
    assert ev.iou_3d(a, far) == 0.0


def test_touching_edge_iou_is_exactly_zero():
    a = np.array([0.0, 0, 0, 4, 2, 1, 0.0])
    b = np.array([4.0, 0, 0, 4, 2, 1, 0.0])  # shares the x=2 edge
    assert ev.bev_iou(a, b) == 0.0


def test_ninety_degree_rotation_unequal_lw():
    # 4x2 vs 2x4 same centre: intersection 2x2=4, union 8+8-4=12 -> 1/3
    a = np.array([0.0, 0, 0, 4, 2, 1, 0.0])
    b = np.array([0.0, 0, 0, 4, 2, 1, math.pi / 2])
    assert abs(ev.bev_iou(a, b) - 1.0 / 3.0) < 1e-6


def test_partial_translation_overlap():
    a = np.array([0.0, 0, 0, 4, 4, 2, 0.0])
    b = np.array([2.0, 0, 0, 4, 4, 2, 0.0])  # half overlap -> 8/(16+16-8)=1/3
    assert abs(ev.bev_iou(a, b) - 1.0 / 3.0) < 1e-6


def test_z_separated_bev_positive_3d_zero():
    a = np.array([0.0, 0, 0.0, 4, 2, 1, 0.0])
    b = np.array([0.0, 0, 5.0, 4, 2, 1, 0.0])
    assert ev.bev_iou(a, b) == 1.0
    assert ev.iou_3d(a, b) == 0.0


def test_vertical_partial_overlap():
    a = np.array([0.0, 0, 0.0, 4, 2, 2.0, 0.0])  # z in [-1, 1]
    b = np.array([0.0, 0, 1.0, 4, 2, 2.0, 0.0])  # z in [0, 2], overlap 1
    # inter vol = 8 * 1 = 8; union = 16 + 16 - 8 = 24 -> 1/3
    assert abs(ev.iou_3d(a, b) - 1.0 / 3.0) < 1e-6


def test_yaw_wrap_error():
    err = ev.wrapped_yaw_error(math.radians(179), math.radians(-179))
    assert abs(err - math.radians(2)) < 1e-6


# --------------------------------------------------------------------------
# AP / matching
# --------------------------------------------------------------------------
def test_perfect_prediction():
    r = _run(_gt(BASE), _pred([VEH], [0.9], BASE))
    m = r.metrics
    for t in (0.25, 0.50, 0.70):
        assert m[f"vehicle/bev_ap@{t:.2f}"] == 1.0
        assert m[f"vehicle/3d_ap@{t:.2f}"] == 1.0
        assert m[f"vehicle/3d_recall@{t:.2f}"] == 1.0
    assert m["vehicle/center_error_3d_m"] == 0.0
    assert m["vehicle/length_mae_m"] == 0.0
    assert m["vehicle/yaw_mae_rad"] == 0.0
    assert m["vehicle/fp@3d_0.50"] == 0


def test_missed_detection():
    r = _run(_gt(BASE), _pred([], []))
    m = r.metrics
    assert m["vehicle/3d_recall@0.50"] == 0.0
    assert m["vehicle/3d_ap@0.50"] == 0.0
    assert m["vehicle/fn@3d_0.50"] == 1
    assert m["vehicle/center_error_3d_m"] is None  # not fabricated as 0


def test_false_positive_on_negative_frame():
    r = ev.evaluate_detections(
        {"a": _gt(BASE), "b": ev.SampleGT(np.zeros((0, 7)), [])},
        {"b": _pred([VEH], [0.8], [3, 3, 0, 4, 2, 1.5, 0.0])},
    )
    m = r.metrics
    assert r.negative_frames == 1
    assert m["vehicle/fp@3d_0.50"] == 1
    assert m["vehicle/3d_precision@0.50"] == 0.0


def test_duplicate_prediction_one_tp_one_fp():
    r = _run(_gt(BASE), _pred([VEH, VEH], [0.9, 0.8], BASE, BASE))
    m = r.metrics
    assert m["vehicle/tp@3d_0.50"] == 1
    assert m["vehicle/fp@3d_0.50"] == 1


def test_confidence_order_changes_ap():
    fp_box = [3.0, 3.0, 0.0, 4, 2, 1.5, 0.0]
    high_fp = _run(_gt(BASE), _pred([VEH, VEH], [0.9, 0.5], fp_box, BASE))
    high_tp = _run(_gt(BASE), _pred([VEH, VEH], [0.9, 0.5], BASE, fp_box))
    assert high_fp.metrics["vehicle/3d_ap@0.50"] < high_tp.metrics["vehicle/3d_ap@0.50"]
    assert high_tp.metrics["vehicle/3d_ap@0.50"] == 1.0


def test_ap_threshold_monotonic():
    # a GT with a prediction whose IoU sits between 0.50 and 0.70
    gt = np.array([10.0, 0, 0, 4.0, 2.0, 1.5, 0.0])
    pred = np.array([10.6, 0, 0, 4.0, 2.0, 1.5, 0.0])  # ~0.7 BEV, lower 3D
    r = ev.evaluate_detections({"s0": ev.SampleGT(gt.reshape(1, 7), [VEH])},
                               {"s0": _pred([VEH], [0.9], pred.tolist())})
    m = r.metrics
    assert m["vehicle/bev_ap@0.25"] >= m["vehicle/bev_ap@0.50"] >= m["vehicle/bev_ap@0.70"]
    assert m["vehicle/3d_ap@0.25"] >= m["vehicle/3d_ap@0.50"] >= m["vehicle/3d_ap@0.70"]


def test_center_and_dimension_error():
    gt = [10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]
    pred = [11.0, 0.0, 0.0, 5.0, 2.5, 1.7, 0.0]
    # loosen the reference matcher so the diagnostic arithmetic is exercised
    r = _run(_gt(gt), _pred([VEH], [0.9], pred), reference_threshold=0.10)
    m = r.metrics
    assert abs(m["vehicle/center_error_bev_m"] - 1.0) < 1e-9
    assert abs(m["vehicle/length_mae_m"] - 1.0) < 1e-9
    assert abs(m["vehicle/width_mae_m"] - 0.5) < 1e-9
    assert abs(m["vehicle/length_bias_m"] - 1.0) < 1e-9  # pred - gt


def test_wrong_class_does_not_match_vehicle():
    r = _run(_gt(BASE), _pred(["pedestrian"], [0.9], BASE))
    m = r.metrics
    assert m["vehicle/pred_count"] == 0
    assert m["vehicle/3d_recall@0.50"] == 0.0
    assert m["vehicle/fn@3d_0.50"] == 1


def test_invalid_prediction_excluded_and_counted():
    good = BASE
    nan_box = [float("nan"), 0, 0, 4, 2, 1.5, 0.0]
    neg_dim = [10.0, 0, 0, -4, 2, 1.5, 0.0]
    r = _run(_gt(BASE), _pred([VEH, VEH, VEH], [0.9, 0.8, float("nan")], good, nan_box, neg_dim))
    assert r.invalid_prediction_boxes == 2
    assert r.pred_count == 1
    assert r.metrics["vehicle/tp@3d_0.50"] == 1


def test_invalid_gt_raises():
    bad = ev.SampleGT(np.array([[1, 2, 3, -1, 2, 1, 0]], dtype=float), [VEH])
    with pytest.raises(ev.MoraiEvaluatorError):
        ev.evaluate_detections({"s0": bad}, {})


def test_zero_gt_split_raises():
    with pytest.raises(ev.MoraiEvaluatorError):
        ev.evaluate_detections({"s0": ev.SampleGT(np.zeros((0, 7)), [])}, {})


def test_zero_predictions_is_valid_recall_zero():
    r = _run(_gt(BASE), _pred([], []))
    assert r.metrics["vehicle/3d_ap@0.50"] == 0.0
    assert r.metrics["vehicle/3d_recall@0.50"] == 0.0


def test_unsupported_class_raises():
    with pytest.raises(ev.MoraiEvaluatorError):
        _run(_gt(BASE), _pred([VEH], [0.9], BASE), class_name="pedestrian")


def test_determinism_and_frame_order_invariance():
    gt = {"a": _gt(BASE), "b": _gt([20.0, 5, 0, 4, 2, 1.5, 0.3])}
    pred = {
        "a": _pred([VEH], [0.7], [10.1, 0, 0, 4, 2, 1.5, 0.0]),
        "b": _pred([VEH], [0.9], [20.0, 5, 0, 4, 2, 1.5, 0.3]),
    }
    r1 = ev.evaluate_detections(gt, pred)
    r2 = ev.evaluate_detections(dict(reversed(list(gt.items()))),
                                dict(reversed(list(pred.items()))))
    assert r1.to_dict()["metrics"] == r2.to_dict()["metrics"]
    assert r1.result_string() == r2.result_string()


def test_range_binned_recall():
    near = [10.0, 0, 0, 4, 2, 1.5, 0.0]   # r=10 -> near
    mid = [30.0, 0, 0, 4, 2, 1.5, 0.0]    # r=30 -> mid
    far = [60.0, 0, 0, 4, 2, 1.5, 0.0]    # r=60 -> far
    r = ev.evaluate_detections(
        {"s0": _gt(near, mid, far)},
        {"s0": _pred([VEH, VEH], [0.9, 0.8], near, mid)},  # far missed
    )
    m = r.metrics
    assert m["range/near/gt_count"] == 1 and m["range/near/recall@3d_0.50"] == 1.0
    assert m["range/mid/gt_count"] == 1 and m["range/mid/recall@3d_0.50"] == 1.0
    assert m["range/far/gt_count"] == 1 and m["range/far/recall@3d_0.50"] == 0.0


def test_schema_version_constant():
    assert ev.MORAI_EVALUATOR_SCHEMA_VERSION == "morai_centerpoint_eval_v1"
    assert ev.PRIMARY_VALIDATION_METRIC == "vehicle/3d_ap@0.50"


# --------------------------------------------------------------------------
# port cross-check + fixture integration
# --------------------------------------------------------------------------
def test_bev_primitives_match_ab3dmot_geometry():
    geom = pytest.importorskip("ad_lidar_perception.ab3dmot_geometry", reason="ad_lidar_perception not importable")
    a = np.array([1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.6])
    b = np.array([2.0, 2.5, 0.0, 3.0, 2.5, 1.5, -0.2])
    ours = ev._polygon_area(ev._polygon_clip(ev._bev_corners(a), ev._bev_corners(b)))
    box_a = geom.Box3D(*[float(v) for v in (a[0], a[1], a[2], a[6], a[3], a[4], a[5])])
    box_b = geom.Box3D(*[float(v) for v in (b[0], b[1], b[2], b[6], b[3], b[4], b[5])])
    ref = geom.polygon_area(geom.polygon_clip(geom.bev_corners(box_a), geom.bev_corners(box_b)))
    assert abs(ours - ref) < 1e-9


def _build_fixture(root: Path) -> Path:
    src, out = root / "src", root / "export"
    make_source_dataset(src, SIX_GROUPS, frames=3)
    plan = {
        "splits": {
            "train": [{"scenario": "lead_constant", "seeds": [0, 1]}, {"scenario": "cut_in", "seeds": [0]}],
            "val": [{"scenario": "dense_multi_object", "seeds": [0]}],
            "test": [{"scenario": "cut_in", "seeds": [1]}],
        }
    }
    CenterPointExporter(src, out, load_adapter_config(None), "evaltest").export(plan)
    return out


def test_cli_evaluates_real_adapter_export_val_split():
    import evaluate_morai_predictions as cli

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        export = _build_fixture(root)
        core = MoraiHevenDatasetCore(export, split="val")
        lines = []
        for sid in core.sample_ids:
            label = core._load_label(sid)
            dets = [
                {
                    "class_name": b["class_name"],
                    "score": 0.9,
                    "box_lidar": [b["x"] + 0.25, b["y"], b["z"], b["length"], b["width"], b["height"], b["yaw"]],
                }
                for b in label["ground_truth"]["boxes"]
            ]
            lines.append(json.dumps({"sample_id": sid, "detections": dets}))
        preds = root / "preds.jsonl"
        preds.write_text("\n".join(lines) + "\n", encoding="utf-8")
        out = root / "eval.json"
        code = cli.main(["--dataset", str(export), "--split", "val",
                         "--predictions", str(preds), "--output", str(out)])
        assert code == 0
        payload = json.loads(out.read_text())
        assert payload["evaluator_schema_version"] == "morai_centerpoint_eval_v1"
        assert payload["metrics"]["vehicle/3d_recall@0.50"] == 1.0
        assert abs(payload["metrics"]["vehicle/center_error_bev_m"] - 0.25) < 1e-6


def test_cli_refuses_test_split_without_flag():
    import evaluate_morai_predictions as cli

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        export = _build_fixture(root)
        preds = root / "p.jsonl"
        preds.write_text("", encoding="utf-8")
        assert cli.main(["--dataset", str(export), "--split", "test",
                         "--predictions", str(preds)]) == 2
        assert cli.main(["--dataset", str(export), "--split", "train",
                         "--predictions", str(preds)]) == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
