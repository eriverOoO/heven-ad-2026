# CenterPoint MORAI Evaluator v1

**Evaluator implementation only. No training, no `optimizer.step()`, no weights
downloaded, no architecture / voxel / range change, no Dataset Factory or
adapter schema change, no tracking / planner change, no real-world
generalization claim, no historical T-14 comparison.**

## Why this exists

`MoraiHevenDataset.evaluation()` previously raised `NotImplementedError` and
`EVAL_METRIC` was `morai_not_implemented`. CenterPoint MORAI Training Prep v1
flagged this as a hard blocker: even with a real leakage-safe validation split,
there was no detector-level metric for checkpoint selection or validation
reporting. This task removes that blocker with a deterministic, versioned,
CPU-only LiDAR-only evaluator.

## Historical-overlap caveat (mandatory)

The historical T-14 CenterPoint experiment trained on 1764 frames with 0
independent validation, 0 independent test, and **100 % train/eval overlap**.
Its reported detector facts (Euclidean localisation ~1.176 m → ~0.368 m,
recall ~71.1 % → ~81.1 %, length bias ~-2.92 m → ~-0.34 m) are **not** an
independent generalisation result. **This evaluator's metrics are NOT
automatically comparable to those numbers.** A future comparison would require
independent protocol reconciliation (same match rule, same IoU definition,
same class/box convention, an independent split). The evaluator never emits
"T-14 reproduced".

## Evaluator schema version

`MORAI_EVALUATOR_SCHEMA_VERSION = "morai_centerpoint_eval_v1"` — a stable
exported constant in `tools/centerpoint_offline/morai_evaluator.py`. Every
machine-readable result carries it. Any protocol change requires a new version.
Training Prep reads this constant (not a source grep) to set
`evaluation_metric_implemented`.

## OpenPCDet evaluation contract

`references/openpcdet/tools/eval_utils/eval_utils.py` calls:

```python
result_str, result_dict = dataset.evaluation(
    det_annos, class_names,
    eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
    output_path=final_output_dir,
)
logger.info(result_str)
ret_dict.update(result_dict)   # -> tensorboard + checkpoint selection
```

- **Signature:** `evaluation(self, det_annos, class_names, **kwargs) -> (str, dict)`.
- **`det_annos`:** a `list`, one dict per evaluated sample, built by
  `DatasetTemplate.generate_prediction_dicts`:
  `name: np.ndarray[str] (N,)`, `score: np.ndarray[float] (N,)`,
  `boxes_lidar: np.ndarray (N, 7)` = `[x, y, z, length, width, height, yaw]`
  (LiDAR frame, geometric centre, CCW +x), `pred_labels: (N,)` 1-based,
  `frame_id` (scalar = the sample_id, usually `numpy.str_`).
- **`kwargs`:** `eval_metric`, `output_path` — **accepted and ignored** (this
  evaluator has one fixed protocol; running the smoke config's
  `morai_not_implemented` value does not error).
- **Return:** `result_dict` is a flat `{str: float}` folded into `ret_dict`;
  `result_str` is a short human table.

The MORAI wrapper recovers GT from `self.morai_core` (the split's
`MoraiHevenDatasetCore`), aligns predictions to GT by **`str(frame_id)` ==
`sample_id`** (never by list position), and calls the standalone core.
Duplicate `frame_id` → hard error. A `frame_id` not in the split → hard error.
A GT sample with no matching anno → treated as zero predictions for that
sample.

## Box / class contract

| property | value |
| --- | --- |
| GT box fields | `[x, y, z, length, width, height, yaw]` |
| GT frame | `lidar_link` |
| centre / z | geometric box centre; vertical interval `[z - h/2, z + h/2]` |
| yaw | CCW about +z (LiDAR z-up), radians |
| prediction box | identical 7-field convention (`boxes_lidar[:, :7]`) |
| prediction score | `float`; excluded (counted) if non-finite |
| classes evaluated | **`vehicle` only** in v1 — the adapter's starter catalog only ground-truths vehicles. A prediction whose class name is not `vehicle` never matches a vehicle GT; it is not silently remapped. |

No transform is applied inside the evaluator — GT and predictions are already
in the same `lidar_link` geometric-centre convention (adapter contract + the
Training Prep box audit).

## Oriented BEV IoU

`bev_iou(a, b)` = footprint intersection / footprint union, using the box's
oriented rectangle (length along local +x, rotated by yaw). Intersection is
Sutherland–Hodgman polygon clipping + the shoelace area; these two primitives
plus `_bev_corners` / `_height_overlap` are **ported verbatim (with
attribution) from `ad_lidar_perception/ad_lidar_perception/ab3dmot_geometry.py`**,
which in turn ports them from `AB3DMOT_libs/dist_metrics.py`. A test
(`test_bev_primitives_match_ab3dmot_geometry`) cross-checks the ported clip
against the live `ab3dmot_geometry` module. KITTI's own R40 evaluator is
camera-frame specific (KITTI "right x, down y, front z", rotation about -Y) and
is deliberately **not** reused — the same reasoning `ab3dmot_geometry`'s
docstring gives for the IoU math itself.

Verified cases: identical boxes → exactly 1.0; touching edge → exactly 0.0;
separated → 0.0; same centre + yaw π/2 with 4×2 dims → 1/3; half-translation
of a 4×4 → 1/3.

## 3D IoU

`iou_3d(a, b)` = `(BEV intersection area × height overlap) / union volume`,
where `height overlap = max(0, min(a.z+a.h/2, b.z+b.h/2) − max(a.z−a.h/2,
b.z−b.h/2))`. Verified: z-separated boxes → BEV IoU > 0 while 3D IoU = 0;
1-unit vertical overlap of two 2-tall boxes with equal footprint → 1/3.

## AP definition

Per class, per metric (`bev`, `3d`), per IoU threshold:

1. collect all valid predictions of that class across all samples;
2. sort **globally** by descending score, tie-break `(sample_id,
   prediction_local_index)` — repeat evaluation is byte-identical;
3. walk predictions in that order; among **unmatched** GT in the *same sample*
   with `IoU ≥ threshold`, take the highest-IoU one. **The threshold gates the
   candidate set** (COCO rule): a below-threshold "best" match is not a match
   and consumes no GT, so the cumulative TP curve is pointwise monotone in the
   threshold and `AP@0.25 ≥ AP@0.50 ≥ AP@0.70` holds by construction;
4. each GT is matched at most once; unmatched prediction → FP;
5. cumulative precision `= tp/(tp+fp)`, recall `= tp/total_gt`;
6. **101-point interpolated AP**: VOC-style monotone-decreasing precision
   envelope, then for each recall level `r ∈ linspace(0, 1, 101)` take the
   precision at the first operating point with recall ≥ `r` (0 if none), and
   average the 101 values.

An all-FP scene → AP 0.0. Confidence ordering matters: a high-score FP ahead of
a low-score TP gives AP 0.5 where the reverse order gives 1.0 (single-GT
fixture).

## IoU thresholds

`IOU_THRESHOLDS = (0.25, 0.50, 0.70)` for both BEV and 3D — an explicit module
constant, not hidden in code. The repo has no prior authoritative LiDAR AP
threshold set for this dataset (the OpenPCDet configs only carry
`RECALL_THRESH_LIST` for its internal ROI/RCNN recall counters, a different
quantity), so this set is declared here.

## Reference matcher for diagnostics

All matched diagnostics (centre / dimension / yaw error, range-binned recall)
are driven by **one** declared matcher: `REFERENCE_MATCH_METRIC = "3d"`,
`REFERENCE_MATCH_THRESHOLD = 0.50`. There is no second independent matcher per
diagnostic. If the reference matcher produces zero TP, every matched diagnostic
is `None` (JSON `null`) — never fabricated as 0.

## Matched diagnostics

- `vehicle/center_error_bev_m` = mean `sqrt(dx² + dy²)` over reference TP (m).
- `vehicle/center_error_3d_m` = mean `sqrt(dx² + dy² + dz²)` (m).
- `vehicle/length_mae_m`, `width_mae_m`, `height_mae_m` = mean `|pred − gt|` (m).
- `vehicle/length_bias_m`, `width_bias_m`, `height_bias_m` = mean `pred − gt`
  (m) — surfaces the historical CenterPoint length-underprediction issue.
- `vehicle/yaw_mae_rad` = mean `|atan2(sin(Δ), cos(Δ))|`, wrapped to `[0, π]`
  (radians; +179° vs −179° → ~0.035 rad, not ~6.25).

## Range-binned recall

GT is binned by **BEV range `hypot(x, y)`** — the same definition
`centerpoint_adapter.py` and the Dataset Factory summary use: `near ≤ 20 m`,
`mid ≤ 45 m`, `far > 45 m`. One canonical "range" across factory / adapter /
evaluator. For each bin: `recall@<reference> = (GT in bin matched by the
reference matcher) / (GT in bin)`; `None` if the bin has no GT. Predictions are
**not** binned independently for the denominator. Bins are diagnostic — not
labelled easy/moderate/hard.

## Result dictionary keys (stable)

```
vehicle/bev_ap@{0.25,0.50,0.70}      vehicle/3d_ap@{0.25,0.50,0.70}
vehicle/bev_precision@{...}          vehicle/3d_precision@{...}
vehicle/bev_recall@{...}             vehicle/3d_recall@{...}
vehicle/tp@3d_0.50  vehicle/fp@3d_0.50  vehicle/fn@3d_0.50  vehicle/matched_count@3d_0.50
vehicle/center_error_bev_m          vehicle/center_error_3d_m
vehicle/length_mae_m  vehicle/width_mae_m  vehicle/height_mae_m
vehicle/length_bias_m  vehicle/width_bias_m  vehicle/height_bias_m
vehicle/yaw_mae_rad
vehicle/gt_count  vehicle/pred_count
range/{near,mid,far}/gt_count       range/{near,mid,far}/recall@3d_0.50
```

`result_dict()` (the OpenPCDet `ret_dict` payload) drops `None` diagnostics;
the standalone `to_dict()` keeps them as `null`.

## Primary validation metric

`PRIMARY_VALIDATION_METRIC = "vehicle/3d_ap@0.50"` — the **engineering
checkpoint-selection metric for MORAI v1**, not a universal CenterPoint
standard. Recorded in `centerpoint_morai_v1.yaml` as `primary_validation_metric`.
**No checkpoint is selected in this PR** (or in Training Prep).

## SCORE_THRESH / NMS audit

`references/openpcdet/pcdet/models/detectors/detector3d_template.py::post_processing`
applies `class_agnostic_nms(..., score_thresh=POST_PROCESSING.SCORE_THRESH)`
**before** `generate_prediction_dicts`. So `det_annos` reaching `evaluation()`
are **post-NMS, post-score-threshold** (`SCORE_THRESH = 0.1`, `NMS_TYPE
nms_gpu`, `NMS_THRESH 0.7` in the v1 model config — unchanged by this task).
**AP is computed over that filtered detection set** — the precision/recall
curve has no sub-`SCORE_THRESH` tail, so it is not a full-score-sweep AP. This
is stated so results are not over-read; `SCORE_THRESH` and NMS are audited, not
retuned.

## Negative / degenerate handling

| condition | behaviour |
| --- | --- |
| frame with 0 GT | valid negative sample; predictions there are FP; counted in `negative_frames` |
| entire split with 0 vehicle GT | **`MoraiEvaluatorError`** — INVALID, never a misleading AP=1 |
| 0 predictions, non-empty GT | valid: recall 0, AP 0, no exception |
| invalid GT box (non-finite / dim ≤ 0) | **`MoraiEvaluatorError`** — the adapter validator should already prevent this; fail loud |
| invalid prediction box (non-finite / dim ≤ 0 / non-finite score) | excluded + counted in `invalid_prediction_boxes`; inference-output corruption stays observable |
| duplicate prediction for one GT | first (highest score) is TP, the rest FP |

## Standalone CLI

```
python3 tools/centerpoint_offline/evaluate_morai_predictions.py \
    --dataset <centerpoint_morai_adapter_v1 export> \
    --split val \
    --predictions <predictions.jsonl> \
    --output <evaluation.json>
```

Predictions: one JSON object per line — a bare
`{"sample_id", "detections": [{"class_name", "score", "box_lidar":[x,y,z,l,w,h,yaw]}]}`
or a `heven.offline_detection.v1` record (the format `infer_morai_centerpoint.py`
already writes). Default split `val`. Before evaluating it invokes
`ad_morai_dataset_validate_centerpoint` (subprocess) and **refuses to evaluate a
leaky export**.

### Split protection

- `--split val` is the default and the only training-validation path.
- `--split test` is refused unless `--allow-test` (reserved for a single final
  evaluation).
- `--split train` is refused unless `--allow-train` (diagnostic only, never
  validation) — the historical 100 %-overlap failure mode is not reachable
  through the normal UX.
- `DATA_SPLIT: {train: train, test: val}` in the v1 data config is unchanged;
  OpenPCDet's periodic validation reads `splits/val.txt`, and `splits/test.txt`
  stays structurally unreachable during tuning.

## Determinism / CPU

Pure NumPy. No torch, no OpenPCDet, no CUDA needed for the core, the geometry,
the AP protocol, or the standalone CLI. Repeated evaluation returns an
identical metric dict and identical result string. Aggregate metrics are
invariant to frame input order and (for unequal scores) preserve score
semantics; equal scores use the `(sample_id, index)` tie-break.

## Validation performed

**Schema-faithful temporary fixtures only — no real MORAI dataset exists.**
Fixtures are built through the real Dataset Factory writer + CenterPoint
adapter (6 `(scenario, seed)` groups, `--split-plan` → train 12 / val 4 /
test 4). Predictions are synthesised.

- 26 core tests (`test_centerpoint_morai_evaluator.py`): oriented IoU (identical
  → 1.0, touching → 0.0, 90°, translation, z-separation, vertical partial
  overlap, yaw wrap), perfect / missed / FP-on-negative / duplicate /
  confidence-order / threshold-monotonic AP, centre + dimension arithmetic,
  wrong-class, invalid-prediction exclude-and-count, invalid-GT raise,
  zero-GT-split raise, zero-predictions valid, unsupported-class raise,
  determinism + frame-order invariance, range-binned recall, the
  `ab3dmot_geometry` port cross-check, and CLI end-to-end on a real adapter
  export incl. split protection.
- `MoraiHevenDataset.evaluation()` integration smoke (venv: torch 2.1.2+cu118,
  OpenPCDet `0.6.0+233f849`): built via the real `DatasetTemplate`, fed
  synthetic `det_annos`, returned `(result_str, {37 float keys})`,
  deterministic, `eval_metric` kwarg accepted-and-ignored.
- Latency: fixture ~1 ms/frame; a 100-frame × 3–15-box synthetic benchmark
  ~7.5 ms/frame (CPU, single-thread, naive O(P×G) per frame — acceptable at
  validation scale).

## What remains blocked

- **No real MORAI dataset.** No `morai_tracking_dataset_v1` and no real
  `centerpoint_morai_adapter_v1` export exist (MORAI simulator / gRPC / ROS
  bridge unavailable in this environment — see the Dataset Collection Pilot
  attempt). This evaluator satisfies only the *software* readiness item.
- Full CenterPoint training is allowed only when a real leakage-safe dataset
  exists, the adapter + preflight pass on it, and this evaluator runs on its
  real validation split.

## Simulator → real caveat

MORAI LiDAR is not VLP-16. Any future metric from this evaluator is
simulator-domain detector validation — it does not prove real-sensor
generalisation. That gap is unaddressed by design.

## Files

`tools/centerpoint_offline/morai_evaluator.py` (core),
`tools/centerpoint_offline/evaluate_morai_predictions.py` (CLI),
`tools/centerpoint_offline/morai_dataset.py` (`evaluation()` wrapper only),
`tools/centerpoint_offline/preflight_morai_training.py` (evaluator-capability
audit), `tools/centerpoint_offline/configs/centerpoint_morai_v1.yaml`
(`primary_validation_metric`), `centerpoint_morai_v1_model.yaml` (`EVAL_METRIC`
→ `morai_centerpoint_eval_v1`),
`tools/centerpoint_offline/test_centerpoint_morai_evaluator.py`, this doc,
`docs/agent/STATUS.md`. `morai_heven_dataset.yaml` / `morai_centerpoint_smoke.yaml`
/ `morai_centerpoint_train.yaml` and the loader / runtime code are unchanged.
