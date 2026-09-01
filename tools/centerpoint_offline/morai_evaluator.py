#!/usr/bin/env python3
"""CenterPoint MORAI Evaluator v1 - deterministic, CPU-only, LiDAR-only.

Detector-level evaluation of CenterPoint predictions against the
geometric-centre ``lidar_link`` GT boxes exported by
``centerpoint_morai_adapter_v1``. Oriented BEV + 3D IoU matching, 101-point
interpolated AP, precision/recall, matched localisation / dimension / yaw
diagnostics, and BEV-range-binned recall.

**No tracking. No AB3DMOT / HOTA / IDSW / KalmanNet / planner. No training.**

Metrics from this evaluator are NOT automatically comparable to the historical
T-14 CenterPoint numbers (that experiment had 100% train/eval overlap and a
different protocol). A future comparison requires explicit protocol
reconciliation.

Pure NumPy: no torch, no OpenPCDet, no CUDA. The four oriented-rectangle
primitives are ported verbatim (with attribution) from
``ad_lidar_perception/ad_lidar_perception/ab3dmot_geometry.py`` - which in turn
ports them from ``AB3DMOT_libs/dist_metrics.py`` (Sutherland-Hodgman clip,
shoelace area) - so the evaluator stays a single self-contained, inspectable
file. KITTI's own R40 evaluator is camera-frame specific (KITTI "right x, down
y, front z", rotation about -Y) and is deliberately NOT reused here, the same
reasoning ``ab3dmot_geometry`` documents for the IoU math itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

MORAI_EVALUATOR_SCHEMA_VERSION = "morai_centerpoint_eval_v1"

# Reported IoU operating points (BEV and 3D), for AP + precision + recall.
IOU_THRESHOLDS: tuple[float, ...] = (0.25, 0.50, 0.70)

# Single matcher drives every matched diagnostic (centre / dim / yaw error,
# range-binned recall). Declared explicitly, not one matcher per diagnostic.
REFERENCE_MATCH_METRIC = "3d"
REFERENCE_MATCH_THRESHOLD = 0.50

# Engineering checkpoint-selection metric for MORAI v1 (NOT a universal
# CenterPoint standard). No checkpoint is selected by this module.
PRIMARY_VALIDATION_METRIC = "vehicle/3d_ap@0.50"

# Vehicle-only v1: the adapter's starter catalog only ground-truths vehicles.
SUPPORTED_CLASSES: tuple[str, ...] = ("vehicle",)

# BEV range of a GT centre = hypot(x, y). Bin edges match
# centerpoint_adapter.py (near <= 20 m, mid <= 45 m, far > 45 m) exactly - one
# canonical "range" definition across the factory, adapter and evaluator.
RANGE_BINS: tuple[tuple[str, float, float], ...] = (
    ("near", 0.0, 20.0),
    ("mid", 20.0, 45.0),
    ("far", 45.0, math.inf),
)

BOX_FIELDS = ("x", "y", "z", "length", "width", "height", "yaw")


class MoraiEvaluatorError(RuntimeError):
    """Raised when evaluation cannot produce a meaningful metric."""


# --------------------------------------------------------------------------
# oriented-rectangle primitives (ported, see module docstring)
# --------------------------------------------------------------------------
def _bev_corners(box: np.ndarray) -> np.ndarray:
    """4 BEV corners (CCW) of ``[x, y, z, l, w, h, yaw]``; length along local +x."""
    half_l, half_w = box[3] / 2.0, box[4] / 2.0
    local = np.array(
        [
            [half_l, half_w],
            [-half_l, half_w],
            [-half_l, -half_w],
            [half_l, -half_w],
        ]
    )
    cos_yaw, sin_yaw = math.cos(box[6]), math.sin(box[6])
    rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    return local @ rotation.T + np.array([box[0], box[1]])


def _polygon_area(points: np.ndarray) -> float:
    """Shoelace area of an ordered simple polygon (>=3 vertices)."""
    if points is None or len(points) < 3:
        return 0.0
    pts = np.asarray(points, dtype=float)
    rolled = np.roll(pts, -1, axis=0)
    return float(abs(np.sum(pts[:, 0] * rolled[:, 1] - pts[:, 1] * rolled[:, 0])) * 0.5)


def _polygon_clip(subject: np.ndarray, clip: np.ndarray) -> np.ndarray | None:
    """Sutherland-Hodgman clip of ``subject`` against convex CCW ``clip``.

    Returns ``None`` for an empty result. Degenerate (parallel) clip edges are
    treated as "no intersection" rather than raising.
    """

    def inside(p, cp1, cp2):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def intersection(cp1, cp2, s, e):
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        denom = dc[0] * dp[1] - dc[1] * dp[0]
        if denom == 0.0:
            return None
        n3 = 1.0 / denom
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    output_list = list(subject)
    cp1 = clip[-1]
    for clip_vertex in clip:
        cp2 = clip_vertex
        input_list = output_list
        output_list = []
        if not input_list:
            return None
        s = input_list[-1]
        for e in input_list:
            if inside(e, cp1, cp2):
                if not inside(s, cp1, cp2):
                    point = intersection(cp1, cp2, s, e)
                    if point is not None:
                        output_list.append(point)
                output_list.append(e)
            elif inside(s, cp1, cp2):
                point = intersection(cp1, cp2, s, e)
                if point is not None:
                    output_list.append(point)
            s = e
        cp1 = cp2
        if not output_list:
            return None
    return np.array(output_list)


def _bev_intersection_area(box_a: np.ndarray, box_b: np.ndarray) -> float:
    clipped = _polygon_clip(_bev_corners(box_a), _bev_corners(box_b))
    return _polygon_area(clipped)


def _height_overlap(box_a: np.ndarray, box_b: np.ndarray) -> float:
    a0, a1 = box_a[2] - box_a[5] / 2.0, box_a[2] + box_a[5] / 2.0
    b0, b1 = box_b[2] - box_b[5] / 2.0, box_b[2] + box_b[5] / 2.0
    return max(0.0, min(a1, b1) - max(a0, b0))


def bev_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Oriented BEV IoU (footprint only)."""
    inter = _bev_intersection_area(box_a, box_b)
    if inter <= 0.0:
        return 0.0
    area_a = float(box_a[3] * box_a[4])
    area_b = float(box_b[3] * box_b[4])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def iou_3d(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Oriented 3D IoU = (BEV intersection area x height overlap) / union volume."""
    inter_area = _bev_intersection_area(box_a, box_b)
    if inter_area <= 0.0:
        return 0.0
    overlap_h = _height_overlap(box_a, box_b)
    if overlap_h <= 0.0:
        return 0.0
    inter_vol = inter_area * overlap_h
    vol_a = float(box_a[3] * box_a[4] * box_a[5])
    vol_b = float(box_b[3] * box_b[4] * box_b[5])
    union = vol_a + vol_b - inter_vol
    return inter_vol / union if union > 0.0 else 0.0


def wrapped_yaw_error(pred_yaw: float, gt_yaw: float) -> float:
    """Shortest absolute angular difference in radians, in [0, pi]."""
    return abs(math.atan2(math.sin(pred_yaw - gt_yaw), math.cos(pred_yaw - gt_yaw)))


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
@dataclass
class SampleGT:
    boxes: np.ndarray  # (N, 7)
    names: list[str]


@dataclass
class SamplePred:
    boxes: np.ndarray  # (M, 7)
    names: list[str]
    scores: np.ndarray  # (M,)


def _coerce_boxes(boxes: Any) -> np.ndarray:
    arr = np.asarray(boxes, dtype=np.float64)
    if arr.size == 0:
        return arr.reshape(0, 7)
    if arr.ndim != 2 or arr.shape[1] < 7:
        raise MoraiEvaluatorError(f"boxes must be (N, >=7); got {arr.shape}")
    return arr[:, :7]


def _box_valid(box: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(box))) and float(min(box[3], box[4], box[5])) > 0.0


# --------------------------------------------------------------------------
# result
# --------------------------------------------------------------------------
@dataclass
class MoraiEvalResult:
    evaluator_schema_version: str = MORAI_EVALUATOR_SCHEMA_VERSION
    class_name: str = "vehicle"
    iou_thresholds: tuple[float, ...] = IOU_THRESHOLDS
    reference_metric: str = REFERENCE_MATCH_METRIC
    reference_threshold: float = REFERENCE_MATCH_THRESHOLD
    primary_validation_metric: str = PRIMARY_VALIDATION_METRIC
    gt_count: int = 0
    pred_count: int = 0
    invalid_gt_boxes: int = 0
    invalid_prediction_boxes: int = 0
    negative_frames: int = 0
    sample_count: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "evaluator_schema_version": self.evaluator_schema_version,
            "class_name": self.class_name,
            "iou_thresholds": list(self.iou_thresholds),
            "reference_metric": self.reference_metric,
            "reference_threshold": self.reference_threshold,
            "primary_validation_metric": self.primary_validation_metric,
            "gt_count": self.gt_count,
            "pred_count": self.pred_count,
            "invalid_gt_boxes": self.invalid_gt_boxes,
            "invalid_prediction_boxes": self.invalid_prediction_boxes,
            "negative_frames": self.negative_frames,
            "sample_count": self.sample_count,
            "metrics": dict(sorted(self.metrics.items())),
            "provenance": dict(sorted(self.provenance.items())),
        }
        return payload

    def result_dict(self) -> dict[str, float]:
        """OpenPCDet-style flat float dict (the ``ret_dict`` update payload)."""
        return {
            key: float(value)
            for key, value in self.metrics.items()
            if isinstance(value, (int, float)) and value is not None
        }

    def result_string(self) -> str:
        m = self.metrics
        lines = [
            f"MORAI CenterPoint evaluator [{self.evaluator_schema_version}]",
            f"  class={self.class_name}  samples={self.sample_count}  "
            f"negative_frames={self.negative_frames}",
            f"  gt_boxes={self.gt_count}  pred_boxes={self.pred_count}  "
            f"invalid_pred={self.invalid_prediction_boxes}",
        ]
        for t in self.iou_thresholds:
            lines.append(
                f"  BEV @{t:.2f}: AP={m.get(f'{self.class_name}/bev_ap@{t:.2f}'):.4f}  "
                f"P={m.get(f'{self.class_name}/bev_precision@{t:.2f}'):.4f}  "
                f"R={m.get(f'{self.class_name}/bev_recall@{t:.2f}'):.4f}"
            )
        for t in self.iou_thresholds:
            lines.append(
                f"  3D  @{t:.2f}: AP={m.get(f'{self.class_name}/3d_ap@{t:.2f}'):.4f}  "
                f"P={m.get(f'{self.class_name}/3d_precision@{t:.2f}'):.4f}  "
                f"R={m.get(f'{self.class_name}/3d_recall@{t:.2f}'):.4f}"
            )
        ref = f"{self.reference_metric}_{self.reference_threshold:.2f}"
        matched = m.get(f"{self.class_name}/matched_count@{ref}")
        lines.append(
            f"  reference match ({self.reference_metric} IoU >= "
            f"{self.reference_threshold:.2f}): matched={matched}  "
            f"TP={m.get(f'{self.class_name}/tp@{ref}')}  "
            f"FP={m.get(f'{self.class_name}/fp@{ref}')}  "
            f"FN={m.get(f'{self.class_name}/fn@{ref}')}"
        )

        def _fmt(key: str) -> str:
            value = m.get(key)
            return "n/a" if value is None else f"{value:.4f}"

        lines.append(
            f"  matched errors: centre_bev={_fmt(f'{self.class_name}/center_error_bev_m')} m  "
            f"centre_3d={_fmt(f'{self.class_name}/center_error_3d_m')} m  "
            f"yaw={_fmt(f'{self.class_name}/yaw_mae_rad')} rad"
        )
        lines.append(
            f"  matched dims MAE: L={_fmt(f'{self.class_name}/length_mae_m')}  "
            f"W={_fmt(f'{self.class_name}/width_mae_m')}  "
            f"H={_fmt(f'{self.class_name}/height_mae_m')} m   "
            f"length_bias={_fmt(f'{self.class_name}/length_bias_m')} m"
        )
        for name, _lo, _hi in RANGE_BINS:
            lines.append(
                f"  range {name:>4}: recall@{ref}="
                f"{_fmt(f'range/{name}/recall@{ref}')}  "
                f"gt={m.get(f'range/{name}/gt_count')}"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------
# AP protocol
# --------------------------------------------------------------------------
def _interpolated_ap_101(recall: np.ndarray, precision: np.ndarray) -> float:
    """101-point interpolated AP over recall in [0, 1].

    For each recall level r in linspace(0, 1, 101), take the maximum precision
    attained at any operating point with recall >= r (0 if none), then average
    over the 101 levels. No interpolation between operating points - the
    right-hand (higher-recall) precision is used, matching the standard
    interpolated-AP definition.
    """
    if recall.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    # monotone-decreasing precision envelope (right to left)
    for i in range(mpre.size - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    levels = np.linspace(0.0, 1.0, 101)
    snapped = np.empty(levels.size)
    for j, level in enumerate(levels):
        idx = int(np.searchsorted(mrec, level, side="left"))
        snapped[j] = mpre[idx] if idx < mpre.size else 0.0
    return float(np.mean(snapped))


def _match_one_threshold(
    ordered: list[tuple[str, int, float, np.ndarray]],
    gt_by_sample: dict[str, np.ndarray],
    iou_fn,
    threshold: float,
    total_gt: int,
) -> dict[str, Any]:
    matched: dict[str, set[int]] = {sid: set() for sid in gt_by_sample}
    tp = np.zeros(len(ordered), dtype=np.float64)
    fp = np.zeros(len(ordered), dtype=np.float64)
    match_pairs: list[tuple[int, str, int]] = []  # (pred_order_idx, sample_id, gt_idx)
    for order_idx, (sid, _pidx, _score, box) in enumerate(ordered):
        gts = gt_by_sample.get(sid)
        best_iou = 0.0
        best_gt = -1
        if gts is not None and len(gts):
            for gt_idx in range(len(gts)):
                if gt_idx in matched[sid]:
                    continue
                value = iou_fn(box, gts[gt_idx])
                # COCO rule: the threshold gates the CANDIDATE set. A below-
                # threshold "best match" is not a match and consumes no GT, so
                # cumulative TP is pointwise monotone in the threshold.
                if value >= threshold and value > best_iou:
                    best_iou = value
                    best_gt = gt_idx
        if best_gt >= 0:
            matched[sid].add(best_gt)
            tp[order_idx] = 1.0
            match_pairs.append((order_idx, sid, best_gt))
        else:
            fp[order_idx] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    denom = np.maximum(tp_cum + fp_cum, 1e-12)
    precision = tp_cum / denom
    recall = tp_cum / max(total_gt, 1)
    ap = _interpolated_ap_101(recall, precision)
    tp_total = int(tp_cum[-1]) if len(ordered) else 0
    fp_total = int(fp_cum[-1]) if len(ordered) else 0
    return {
        "ap": ap,
        "precision": float(precision[-1]) if len(ordered) else 0.0,
        "recall": tp_total / max(total_gt, 1),
        "tp": tp_total,
        "fp": fp_total,
        "fn": max(total_gt, 0) - tp_total,
        "match_pairs": match_pairs,
    }


# --------------------------------------------------------------------------
# core
# --------------------------------------------------------------------------
def evaluate_detections(
    gt_by_sample: dict[str, SampleGT],
    pred_by_sample: dict[str, SamplePred],
    *,
    class_name: str = "vehicle",
    iou_thresholds: Sequence[float] = IOU_THRESHOLDS,
    reference_metric: str = REFERENCE_MATCH_METRIC,
    reference_threshold: float = REFERENCE_MATCH_THRESHOLD,
    provenance: dict[str, Any] | None = None,
) -> MoraiEvalResult:
    if class_name not in SUPPORTED_CLASSES:
        raise MoraiEvaluatorError(
            f"class {class_name!r} not supported by evaluator v1 "
            f"(supported: {SUPPORTED_CLASSES})"
        )
    if reference_metric not in ("bev", "3d"):
        raise MoraiEvaluatorError("reference_metric must be 'bev' or '3d'")

    sample_ids = sorted(gt_by_sample)
    if not sample_ids:
        raise MoraiEvaluatorError("no evaluation samples")

    # ---- GT ---------------------------------------------------------
    class_gt: dict[str, np.ndarray] = {}
    total_gt = 0
    invalid_gt = 0
    negative_frames = 0
    gt_range_bins: list[str] = []  # per class GT, aligned with iteration below
    gt_index: list[tuple[str, int]] = []
    for sid in sample_ids:
        entry = gt_by_sample[sid]
        boxes = _coerce_boxes(entry.boxes)
        keep = []
        for local_idx, name in enumerate(entry.names):
            box = boxes[local_idx]
            if not _box_valid(box):
                invalid_gt += 1
                continue
            if str(name) != class_name:
                continue
            keep.append(box)
            gt_range_bins.append(_range_bin(box))
            gt_index.append((sid, len(keep) - 1))
        class_gt[sid] = np.asarray(keep, dtype=np.float64).reshape(-1, 7)
        total_gt += len(keep)
        if len(keep) == 0:
            negative_frames += 1
    if invalid_gt:
        raise MoraiEvaluatorError(
            f"{invalid_gt} invalid GT box(es) - the adapter validator should "
            "have rejected this export; refusing to evaluate"
        )
    if total_gt == 0:
        raise MoraiEvaluatorError(
            f"evaluation split has zero {class_name} GT boxes - INVALID "
            "(cannot produce a meaningful AP/recall)"
        )

    # ---- predictions ---------------------------------------------
    ordered: list[tuple[str, int, float, np.ndarray]] = []
    pred_count = 0
    invalid_pred = 0
    for sid in sample_ids:
        entry = pred_by_sample.get(sid)
        if entry is None:
            continue
        boxes = _coerce_boxes(entry.boxes)
        scores = np.asarray(entry.scores, dtype=np.float64).reshape(-1)
        for local_idx, name in enumerate(entry.names):
            if str(name) != class_name:
                continue
            box = boxes[local_idx]
            score = float(scores[local_idx]) if local_idx < scores.size else math.nan
            if not _box_valid(box) or not math.isfinite(score):
                invalid_pred += 1
                continue
            pred_count += 1
            ordered.append((sid, local_idx, score, box))
    # global deterministic order: score desc, then sample id, then local index
    ordered.sort(key=lambda item: (-item[2], item[0], item[1]))

    iou_fns = {"bev": bev_iou, "3d": iou_3d}
    metrics: dict[str, Any] = {}
    per_metric_pairs: dict[str, list[tuple[int, str, int]]] = {}
    for metric_name, iou_fn in iou_fns.items():
        for threshold in iou_thresholds:
            out = _match_one_threshold(
                ordered, class_gt, iou_fn, float(threshold), total_gt
            )
            tag = f"{class_name}/{metric_name}_"
            metrics[f"{tag}ap@{threshold:.2f}"] = out["ap"]
            metrics[f"{tag}precision@{threshold:.2f}"] = out["precision"]
            metrics[f"{tag}recall@{threshold:.2f}"] = out["recall"]
            per_metric_pairs[f"{metric_name}@{threshold:.2f}"] = out["match_pairs"]

    # ---- reference-matched diagnostics ---------------------------
    ref_key = f"{reference_metric}@{reference_threshold:.2f}"
    if ref_key not in per_metric_pairs:
        per_metric_pairs[ref_key] = _match_one_threshold(
            ordered, class_gt, iou_fns[reference_metric],
            float(reference_threshold), total_gt,
        )["match_pairs"]
    ref_pairs = per_metric_pairs[ref_key]
    ref_tag = f"{reference_metric}_{reference_threshold:.2f}"
    metrics[f"{class_name}/matched_count@{ref_tag}"] = len(ref_pairs)
    metrics[f"{class_name}/tp@{ref_tag}"] = len(ref_pairs)
    metrics[f"{class_name}/fp@{ref_tag}"] = pred_count - len(ref_pairs)
    metrics[f"{class_name}/fn@{ref_tag}"] = total_gt - len(ref_pairs)

    matched_gt_flat: set[tuple[str, int]] = set()
    centre_bev, centre_3d = [], []
    err_l, err_w, err_h = [], [], []
    bias_l, bias_w, bias_h = [], [], []
    err_yaw = []
    for order_idx, sid, gt_idx in ref_pairs:
        _s, _pidx, _score, pbox = ordered[order_idx]
        gbox = class_gt[sid][gt_idx]
        centre_bev.append(math.hypot(pbox[0] - gbox[0], pbox[1] - gbox[1]))
        centre_3d.append(
            math.sqrt(
                (pbox[0] - gbox[0]) ** 2
                + (pbox[1] - gbox[1]) ** 2
                + (pbox[2] - gbox[2]) ** 2
            )
        )
        err_l.append(abs(pbox[3] - gbox[3]))
        err_w.append(abs(pbox[4] - gbox[4]))
        err_h.append(abs(pbox[5] - gbox[5]))
        bias_l.append(pbox[3] - gbox[3])
        bias_w.append(pbox[4] - gbox[4])
        bias_h.append(pbox[5] - gbox[5])
        err_yaw.append(wrapped_yaw_error(pbox[6], gbox[6]))
        matched_gt_flat.add((sid, gt_idx))

    def _mean_or_none(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    metrics[f"{class_name}/center_error_bev_m"] = _mean_or_none(centre_bev)
    metrics[f"{class_name}/center_error_3d_m"] = _mean_or_none(centre_3d)
    metrics[f"{class_name}/length_mae_m"] = _mean_or_none(err_l)
    metrics[f"{class_name}/width_mae_m"] = _mean_or_none(err_w)
    metrics[f"{class_name}/height_mae_m"] = _mean_or_none(err_h)
    metrics[f"{class_name}/length_bias_m"] = _mean_or_none(bias_l)
    metrics[f"{class_name}/width_bias_m"] = _mean_or_none(bias_w)
    metrics[f"{class_name}/height_bias_m"] = _mean_or_none(bias_h)
    metrics[f"{class_name}/yaw_mae_rad"] = _mean_or_none(err_yaw)

    # ---- range-binned recall (reference matcher) ----------------
    bin_totals = {name: 0 for name, _lo, _hi in RANGE_BINS}
    bin_matched = {name: 0 for name, _lo, _hi in RANGE_BINS}
    for (sid, gt_idx), bin_name in zip(gt_index, gt_range_bins):
        bin_totals[bin_name] += 1
        if (sid, gt_idx) in matched_gt_flat:
            bin_matched[bin_name] += 1
    for name, _lo, _hi in RANGE_BINS:
        total = bin_totals[name]
        metrics[f"range/{name}/gt_count"] = total
        metrics[f"range/{name}/recall@{ref_tag}"] = (
            bin_matched[name] / total if total else None
        )

    metrics[f"{class_name}/gt_count"] = total_gt
    metrics[f"{class_name}/pred_count"] = pred_count

    result = MoraiEvalResult(
        class_name=class_name,
        iou_thresholds=tuple(float(t) for t in iou_thresholds),
        reference_metric=reference_metric,
        reference_threshold=float(reference_threshold),
        gt_count=total_gt,
        pred_count=pred_count,
        invalid_gt_boxes=invalid_gt,
        invalid_prediction_boxes=invalid_pred,
        negative_frames=negative_frames,
        sample_count=len(sample_ids),
        metrics=metrics,
        provenance=dict(provenance or {}),
    )
    return result


def _range_bin(box: np.ndarray) -> str:
    """BEV range hypot(x, y) -> near (<= 20 m) / mid (<= 45 m) / far (> 45 m)."""
    dist = math.hypot(float(box[0]), float(box[1]))
    if dist <= RANGE_BINS[0][2]:
        return RANGE_BINS[0][0]
    if dist <= RANGE_BINS[1][2]:
        return RANGE_BINS[1][0]
    return RANGE_BINS[2][0]


# --------------------------------------------------------------------------
# helpers to build core inputs from common shapes
# --------------------------------------------------------------------------
def sample_gt_from_label(label: dict[str, Any]) -> SampleGT:
    """Build a SampleGT from a centerpoint_morai_adapter_v1 label JSON."""
    boxes_raw = label["ground_truth"]["boxes"]
    boxes = np.asarray(
        [[float(b[f]) for f in BOX_FIELDS] for b in boxes_raw], dtype=np.float64
    ).reshape(-1, 7)
    names = [str(b["class_name"]) for b in boxes_raw]
    return SampleGT(boxes=boxes, names=names)


def sample_pred_from_arrays(
    names: Iterable[str], scores: Iterable[float], boxes: Any
) -> SamplePred:
    boxes_arr = _coerce_boxes(boxes) if np.asarray(boxes).size else np.zeros((0, 7))
    return SamplePred(
        boxes=boxes_arr,
        names=[str(n) for n in names],
        scores=np.asarray(list(scores), dtype=np.float64).reshape(-1),
    )
