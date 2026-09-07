# Autoware CenterPoint stability audit v1

## 1. Executive summary

This is separate from the historical local OpenPCDet T14 experiment. The two
pipelines use different models and post-processing, so the T14 result cannot
be reported as an Autoware measurement. This host cannot run the optional
Autoware detector: the pinned ONNX artifacts, TensorRT engines, and installed
`autoware_lidar_centerpoint_node` are absent. The only actor-GT MORAI export
is the training scene and its source bag payload is unavailable for a
TF-correct TensorRT replay.

No Autoware duplicate, recall, jitter, IDSW, fragmentation, or HOTA number is
reported. A strict stage-dump analyzer and bounded isolated capture procedure
are ready for the first runnable replay. **Retraining decision: NOT YET.**

## 2. Pipeline provenance

| Item | Historical experiment | HEVEN optional runtime |
| --- | --- | --- |
| Implementation | Local OpenPCDet CenterPoint | Autoware Universe `autoware_lidar_centerpoint` 0.51.0 |
| Weights | `centerpoint_t14_reproduction.pth` | Pinned external ONNX encoder/head; absent locally |
| Score gate | Global 0.10 | Per-class, per-distance-bin 0.35 |
| First NMS | Class-agnostic BEV IoU, 0.70 | Score-sorted GPU circle NMS, XY distance 0.5 m |
| Second NMS | None | CPU BEV IoU NMS, search distance 10 m, IoU > 0.1 suppresses |
| Runtime status | Evaluated previously | Not provisioned / not runnable |

The training corpus/provenance for the absent Autoware artifacts is
**unresolved**. The lock pins artifact hashes and release, not a training
dataset; no AV2-pretrained claim is made.

## 3. Actual HEVEN runtime graph

The production selection remains `euclidean_cluster`. CenterPoint is opt-in:

```text
/ad/perception/lidar/points_xyzirc
  -> autoware_lidar_centerpoint_node (TensorRT encoder + head)
  -> /ad/perception/objects/detected
  -> autoware_multi_object_tracker
  -> /ad/perception/objects/tracked
```

The model files define FP16 TensorRT, capacity 2,000,000, classes `CAR`,
`TRUCK`, `BUS`, `BICYCLE`, `PEDESTRIAN`, point feature size 4, range
`[-76.8,-76.8,-4,76.8,76.8,6]`, voxel `[0.32,0.32,10]`, and downsample 1
(`centerpoint`) or 2 (`centerpoint_tiny`). Neither was loaded in this audit.

`num_past_frames: 1` means current **plus one transformed past cloud**, not
current-only: the implementation sets cache size to `num_past_frames + 1`.

## 4. Code-level candidate flow

```text
TensorRT heatmaps / regression heads
  -> per-cell argmax, sigmoid, range, score and yaw-norm gates
  -> score-sorted candidates
  -> class-agnostic circle NMS (distance < 0.5 m)
  -> DetectedObject conversion
  -> BEV IoU NMS (IoU > 0.1, within 10 m)
  -> class remapper -> final DetectedObjects / tracker input
```

The score threshold is not global 0.10: the pinned ML-package config sets
0.35 for every listed class in the 0--50, 50--90, 90--121, and 121--200 m
bins. It is applied inside CUDA box generation. Circle NMS is class-agnostic;
IoU NMS preserves the score order and compares same-label or different
non-pedestrian pairs, but deliberately does not suppress pedestrian versus a
different label. Class remapping occurs after IoU NMS; no further suppression
is applied.

The existing node does not materialize a public pre-score stream: rejected
CUDA cells are set to score zero then immediately compacted. A final
`DetectedObjects` subscriber cannot truthfully reconstruct an earlier stage.

## 5. Stage instrumentation and metrics

`tools/centerpoint_offline/autoware_stage_audit.py` accepts only explicit
opt-in dumps with schema `heven.autoware_centerpoint_stage_dump.v1` and stages
`decoded`, `score`, `circle_nms`, `iou_nms`, `final`. It rejects invalid
candidate subset transitions, including a final set different from IoU-NMS.
It calculates candidate/frame, many-to-one predictions/GT, duplicate rates,
one-to-one recall, primary position error, and unmatched candidate/frame.

An upstream patch is required to emit stages 0--3. It must be opt-in
(`enable_centerpoint_stage_dump:=false` by default), give candidates immutable
IDs before compaction, stream JSONL to
`/home/didgang1203/datasets/centerpoint/autoware_stability_audit_v1/stage_dumps/`,
and be built in a separate overlay. No external tracked source was modified.

| Stage | candidates/frame | Pred/GT | duplicate >=2 | recall |
| --- | ---: | ---: | ---: | ---: |
| decoded | not measured | not measured | not measured | not measured |
| score | not measured | not measured | not measured | not measured |
| circle NMS | not measured | not measured | not measured | not measured |
| IoU NMS | not measured | not measured | not measured | not measured |
| final | not measured | not measured | not measured | not measured |

## 6. Baseline and historical comparison

No runnable Autoware baseline exists. Therefore whether actual HEVEN Autoware
CenterPoint outputs multiple boxes per GT is **unresolved**, not yes or no.
Historical OpenPCDet T14 did have severe duplicates (7.17 predictions/GT,
79.75% duplicate >=2), but its NMS 0.10 result cannot be copied numerically
or parametrically to Autoware's two-stage NMS.

## 7. Root-cause assessment

| Hypothesis | Assessment | Evidence |
| --- | --- | --- |
| Circle NMS | Unresolved | Semantics known; no candidate counts. |
| IoU NMS | Unresolved | Semantics known; no candidate counts. |
| Score calibration | Unresolved | Fixed class/distance gate known; no score data. |
| Model/domain mismatch | Unresolved | Artifact metadata and inference unavailable. |
| Preprocessing | Unresolved | Input/TF contract known; no replay. |
| Densification | Unresolved | Current + one past cloud confirmed; no ablation. |
| Tracker coupling | Unresolved | No Autoware replay reached the fixed tracker. |

## 8. Held-out data and capture procedure

`/home/didgang1203/datasets/morai_heven` has 1,764 actor-GT frames but only
the training scene `static_20260805_003151`; no sequence-disjoint split is
available. Its source bag has no payload, and `morai_cam4` lacks actor GT.

Record a different route, spawn pattern, or traffic density for 2--5 minutes:

```bash
export ROS_DOMAIN_ID=77
source /opt/ros/humble/setup.bash
ros2 bag record -o /safe/external/path/morai_autoware_audit_v1 \
  /ad/sensors/lidar/points /ad/dev/objects /ad/dev/vehicle/ego_status \
  /tf /tf_static /clock
ros2 bag info /safe/external/path/morai_autoware_audit_v1
```

First verify exact types using `ros2 topic info -v`. Export lidar-frame GT
with the existing timestamp/TF contract; never frame-random split. Replay in
`ROS_DOMAIN_ID=77` with a separately built instrumented overlay and verified
model artifacts. Store dumps outside git, then run:

```bash
PYTHONPATH=tools/centerpoint_offline python3 \
  tools/centerpoint_offline/autoware_stage_audit.py \
  --stage-dump /home/didgang1203/datasets/centerpoint/autoware_stability_audit_v1/stage_dumps/run.jsonl \
  --gt-json /safe/external/path/lidar_gt_by_frame.json \
  --output /home/didgang1203/datasets/centerpoint/autoware_stability_audit_v1/baseline/stage_summary.json
```

Only after a baseline proves duplicates should one NMS axis at a time be
screened, followed by fixed-tracker replay of baseline and Pareto candidates.

## 9. Recommendation

Do not tune Autoware NMS, alter the Euclidean default, or start MORAI/AV2
training. The highest-value next task is collection of a sequence-disjoint
MORAI actor-GT recording with TF and clock topics, while provisioning the
pinned Autoware model/executable in an isolated overlay.

