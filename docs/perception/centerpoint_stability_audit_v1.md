# CenterPoint stability audit v1

## 1. Executive summary

The historical T14 MORAI run has excessive final, post-NMS CenterPoint density: at score 0.10, the new many-to-one GT diagnostic finds **7.17 predictions per GT actor-frame** (p95 12, maximum 17), with >=2 candidates on 79.7% of GT actor-frames. This is not hidden by the ordinary one-to-one recall match (81.0%, mean centre error 0.370 m). The frozen downstream tracker had HOTA 0.0529, IDSW 443, and 5,916 IDs.

The recorded data establishes final-output density and tracker coupling, but cannot identify raw network heatmap instability versus NMS failure: the historical JSONL is already OpenPCDet post-NMS. A score-only sweep is not sufficient: 0.15 drops candidates/GT to 4.95 but costs 4.0 recall points; 0.30 reaches 1.85 but costs 19.6 points. The next action is a small raw→score→NMS instrumented replay and tracker replay of Pareto candidates, not training.

This checkpoint trained and evaluated on the same single 1,764-frame MORAI scene (100% overlap). Neither detector quality nor a future fine-tune generalizes until held-out scenes exist.

## 2. Environment

| item | value |
|---|---|
| branch / base / origin | `feat/centerpoint-stability-audit-v1` / `0f649c9b06f0cf34029a1fba921047d62a30039c` / same |
| timestamp | 2026-09-07 KST |
| Python / PyTorch / CUDA | `~/venvs/heven-centerpoint/bin/python` / 2.1.2+cu118 / 11.8 |
| GPU | NVIDIA GeForce RTX 4060 8 GiB |
| OpenPCDet | `/home/didgang1203/projects/OpenPCDet`, `233f849829b6ac19afb8af8837a0246890908755` |
| checkpoint | historical MORAI T14 reproduction, three classes, epoch 3 / iteration 5292 |
| evaluation | 1,764 frames, 2,844 valid vehicle actor-frames, train/eval overlap 100% |

KalmanNet PID 23728 was alive and MemAvailable was about 2.2 GiB. This audit used only streaming CPU-light analysis; no CenterPoint inference or training started.

## 3. Existing pipeline

`/ad/perception/lidar/cropped` (`lidar_link`, finite XYZI) → unchanged XYZI decode → OpenPCDet DatasetTemplate preprocessing → CenterPoint → OpenPCDet CenterHead NMS → ROS score/cap filter → `/ad/perception/objects/detected` → Autoware tracker.

CenterPoint is opt-in: `detector_backend=euclidean` and `enabled=false` in `config/detectors/centerpoint_ros.yaml`; production default remains Euclidean. Vehicle maps to CAR, pedestrian to PEDESTRIAN, and obstacle to UNKNOWN under the existing Autoware classification contract.

| setting | value |
|---|---|
| point range / voxel | `[-4,-25,-3,100,25,5]` / `[0.125,0.125,0.2]` |
| classes | vehicle, pedestrian, obstacle |
| ROS score/cap | 0.10 / 500 |
| model score/cap | 0.10 / 500 |
| NMS | OpenPCDet `nms_gpu` IoU NMS, threshold 0.70, pre 4096, post 500 |
| circle NMS/search distance | not configured; not used |

No further filter exists before publication. The stored `centerpoint_detections_raw.jsonl` is not pre-NMS raw proposals: it contains post-NMS model `pred_boxes` followed by the same score/cap filter.

## 4. Baseline and diagnostic definition

Official historical matching is GT-priority nearest unused prediction within 3.0 m. This audit reuses the class/gate but separately counts every compatible prediction within 3.0 m for each GT, so duplicate candidates are not concealed by one-to-one assignment. This is a proximity diagnostic, not a claim that every nearby box is definitely one physical actor.

| metric | score 0.10 |
|---|---:|
| frames / GT / detections | 1,764 / 2,844 / 29,056 |
| recall / mean centre error | 81.05% / 0.370 m |
| predictions per GT: mean / p95 / max | 7.17 / 12 / 17 |
| >=2 / >=3 / >=5 candidates | 79.75% / 78.09% / 75.14% |
| unmatched detections/frame | 15.16 |
| residual-centre jitter mean / p95 | 0.464 / 0.921 m |
| historical HOTA / IDSW / fragmentation | 0.05293 / 443 / 73 |

Matched-primary score mean is 0.280; duplicate-candidate mean 0.214; unmatched mean 0.206. The distributions overlap substantially (duplicate maximum 0.836), so no globally safe cutoff is evident. Consecutive-frame residual jitter removes GT motion: yaw delta median/p95 is 0.080/2.814 rad; length delta median/p95 is 0.131/2.179 m.

Results are external and untracked: `/home/didgang1203/datasets/centerpoint/stability_audit_v1/` contains baseline JSON plus per-threshold frame/object CSV. The historical forensic asset independently records a frame with 20 unmatched CenterPoint candidates within 5 m of one actor.

## 5. Score sweep

Only score was varied on saved post-NMS data. NMS cannot be reconstructed, so no invented circle-NMS or Cartesian sweep was run.

| score | recall | preds/GT | >=2 rate | unmatched/frame | jitter mean |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 81.05% | 7.17 | 79.75% | 15.16 | 0.464 m |
| 0.15 | 77.04% | 4.95 | 75.11% | 8.38 | 0.471 m |
| 0.20 | 72.54% | 3.53 | 68.99% | 5.30 | 0.472 m |
| 0.30 | 61.43% | 1.85 | 52.57% | 2.26 | 0.463 m |
| 0.40 | 43.85% | 0.82 | 24.79% | 0.74 | 0.453 m |
| 0.50 | 21.34% | 0.29 | 5.98% | 0.18 | 0.430 m |

`0.15` is a screening candidate only: it reduces density 31% for 4 pp recall loss, yet leaves extreme duplicates. There is no post-filter tracker rerun, so it is not a production recommendation and cannot be claimed to improve HOTA or IDSW.

## 6. Tracking, domain, and root cause

The frozen comparison uses the same 3 m Hungarian/linear-KF tracker: CenterPoint is 16.47 detections/frame versus Euclidean 9.15, 5,916 versus 1,749 IDs, and 443 versus 86 IDSW. Fragmentation is lower (73 vs 113), distinguishing identity churn from simply more gaps. This is strong evidence of tracker coupling, though it does not prove every duplicate alone causes an ID switch.

The runtime explicitly preserves XYZI, range and `lidar_link`; no direct coordinate/range/intensity contract violation was found. Ground removal is intentionally absent from CenterPoint's cropped→model path. The checkpoint is local MORAI T14, not AV2-pretrained; VLP-16/domain generalization is unverified because of the same-scene overlap.

| hypothesis | evidence strength | reason |
|---|---|---|
| NMS insufficient/misconfigured | Moderate | final post-NMS density is extreme; IoU 0.70 is permissive; pre-NMS unavailable |
| score calibration | Moderate | 0.10 admits density, but scores overlap and cutoff loses recall |
| model/domain mismatch | Moderate | competing high-score post-NMS boxes and residual outliers; no held-out scene |
| preprocessing mismatch | Weak | declared runtime contract matches historical config |
| tracker coupling | Strong | 5.2x IDSW and 3.4x IDs under fixed tracker with denser input |

## 7. Recommendation and training readiness

**NOT YET — measure/fix the postprocessing boundary first.** Add an opt-in wrapper that records per-frame decoded pre-NMS candidates, model NMS output and ROS score/cap output without changing semantics. On a bounded held-out MORAI replay (batch 1, workers 0), vary one confirmed NMS parameter at a time (e.g. IoU 0.3/0.5/0.7) and scores 0.10/0.15/0.20. Re-run the tracker only for Pareto candidates and require recall, candidates/GT, jitter, HOTA, IDSW, fragmentation and latency together.

If high-confidence competing centres survive reasonable NMS on held-out scenes, MORAI fine-tuning is likely useful. Do not automatically add AV2 first: there are no existing AV2-pretrained weights here; if chosen, AV2→MORAI must be its own controlled initialization experiment. MORAI labels already provide actor ID, class, lidar-frame centre, dimensions, yaw, timestamps and TF provenance; existing `ad_morai_dataset_export_centerpoint` / `centerpoint_morai_adapter_v1` define the conversion path. Schema readiness is adequate; data coverage is not. Collect disjoint scene/route/traffic sequences, split A/B/C train, D validation, E test at sequence level, validate transforms, then run a small RTX-4060 fine-tune only after the held-out split exists.

## 12. Stage instrumentation and provenance resolution

The apparent `0.70` versus `0.10` disagreement is resolved: they are **not the same CenterPoint pipeline**.

| pipeline | implementation / model | score | NMS | input / data |
|---|---|---:|---|---|
| Historical T14 train/evaluation | local OpenPCDet `CenterPoint`, T14 reproduction checkpoint | 0.10 in CenterHead and offline bridge | class-agnostic BEV `nms_gpu`, IoU suppression threshold 0.70, pre 4096/post 500; no circle NMS | cropped XYZI, range `[-4,-25,-3,100,25,5]`, voxel `[.125,.125,.2]`, 3 classes, single MORAI scene |
| Historical saved baseline generation | `infer_morai_centerpoint.py`, same checkpoint/config | 0.10 | the same model-internal 0.70 NMS, then identical score/cap bridge | 1,764 T14 train-split frames |
| Current HEVEN Autoware runtime | TensorRT/ONNX `autoware_lidar_centerpoint` (separate from local OpenPCDet) | model package controlled; not T14 checkpoint | circle distance 0.5, IoU NMS 0.1, 2-D search 10 m | capacity 2M; one past-frame densification; separate Autoware model package |
| Current HEVEN Tiny | TensorRT/ONNX `centerpoint_tiny` package | package controlled | same Autoware 0.5/0.1/10 m parameters | separate Tiny model package |

Historical baseline prediction stage is therefore **post-model-NMS, post-score-filter, pre/post ROS equivalently represented as offline JSONL**, not raw heatmap output. Evidence is `infer_summary.json` (checkpoint, 0.10, 29,056 outputs), `infer_morai_centerpoint.py` (calls model then `filter_prediction`), and OpenPCDet `CenterHead.generate_predicted_boxes` (decode score threshold then NMS). It is not an Autoware runtime artifact.

`dump_prediction_stages.py` adds an opt-in, in-process observation hook only. It leaves OpenPCDet tracked source and ROS production behavior unchanged, streams JSONL to the external audit root, and records decoded top-K candidates, score-filtered candidates, exact OpenPCDet NMS output, and the equivalent ROS-final output. “Raw” below means decoded, geometrically valid top-500 network candidates before score filtering—not the full dense heatmap tensor.

## 13. Raw vs post-NMS analysis

Thirty automatically selected frames (ten severe `predictions_per_GT >= 12`, ten typical median-near 8, ten clean 1) were replayed with batch 1/workers 0. The selection includes independent frame IDs and is diagnostic only, not held-out evaluation.

| stage | candidates/frame | predictions/GT mean | >=2 duplicate rate | recall | unmatched/frame |
|---|---:|---:|---:|---:|---:|
| decoded raw top-500 | 500.0 | 16.20 | 97.25% | 98.17% | 496.43 |
| score >=0.10 | 39.53 | 6.82 | 70.64% | 80.73% | 36.60 |
| OpenPCDet NMS 0.70 | 33.80 | 6.45 | 70.64% | 80.73% | 30.87 |
| ROS final | 33.80 | 6.45 | 70.64% | 80.73% | 30.87 |

NMS removes 5.73 candidates/frame after score filtering but only 0.37 candidate/GT, and does not change diagnostic duplicate rate or recall on this subset. ROS adds no further change: its score/cap is redundant for the already capped model output.

For 3,013 pairs of surviving vehicle candidates around the same GT, centre-distance median/p95 is 1.376/2.757 m, BEV-IoU median/p95 is 0.291/0.626 (maximum 0.69997), yaw difference median/p95 is 0.562/3.080 rad, and score-difference median is 0.092. Class-conflict fraction is 0: the historical three-class model's issue here is not CAR/TRUCK/BUS conflict. OpenPCDet uses **class-agnostic**, score-descending BEV-IoU NMS; `0.70` is the suppression IoU threshold (boxes at greater IoU are suppressed), not a keep score. Surviving boxes mostly have insufficient mutual IoU because centres/yaws scatter.

## 14. NMS sweep

Because stored historical data is already post-0.70-NMS, the full controlled sweep is explicitly a **secondary class-agnostic BEV-IoU NMS** over that fixed output. It does not claim to reproduce a fresh raw-model NMS sweep; it is a safe postprocessing candidate screen. Score stays 0.10 and tracker/association/estimator/lifecycle remain frozen.

| secondary IoU threshold | recall | mean centre error | predictions/GT | >=2 / >=5 rate | unmatched/frame |
|---:|---:|---:|---:|---:|---:|
| 0.70 baseline-equivalent | 81.05% | 0.370 m | 7.17 | 79.75% / 75.14% | 15.16 |
| 0.50 conservative | 80.94% | 0.501 m | 4.37 | 78.41% / 56.08% | 9.38 |
| 0.30 balanced | 80.84% | 0.695 m | 1.95 | 62.27% / 3.13% | 4.20 |
| 0.10 aggressive | 80.59% | 0.802 m | 0.96 | 13.82% / 0.00% | 1.57 |

The score-only result had a recall trade-off; by contrast, the secondary-NMS axis preserves recall within 0.46 points but worsens localization because it preferentially retains highest-score candidates rather than the nearest-to-GT candidate. Candidate A is 0.50, B is 0.30, and C is 0.10.

## 15. Tracker replay

All four candidates were replayed through the historical fixed AB3DMOT configuration: Euclidean 3 m, Hungarian, linear KF, detector yaw, `min_hits=1`, `max_age=2`; no tracker parameter changed. TrackEval uses the same GT and 2 m metric threshold as baseline.

| candidate | detections | IDSW | fragmentation | tracker IDs | HOTA |
|---|---:|---:|---:|---:|---:|
| baseline / 0.70 | 29,056 | 443 | 73 | 5,916 | 0.05293 |
| A / 0.50 | 18,841 | 506 | 116 | 4,358 | 0.06495 |
| B / 0.30 | 9,708 | 467 | 190 | 2,699 | 0.08435 |
| C / 0.10 | 5,069 | **273** | 235 | **1,676** | **0.12750** |

Candidate C cuts duplicate >=2 rate 82.7%, candidate count 82.6%, and IDSW 38.4%, while HOTA rises 141% and recall falls only 0.46 points. Fragmentation rises, so stricter NMS trades identity churn for missed/reacquired tracks. Candidate B is the less aggressive Pareto alternative when geometry error/fragmentation matters.

## 16. Held-out check

No held-out MORAI scene with both LiDAR and actor GT is available. Local labels contain only `static_20260805_003151`; `morai_cam4_20260813_163222` is a genuinely different replayable sequence but has no actor-GT topic. The metadata-only static bag is the same scene. Therefore all Stage-2 conclusions are **in-domain only**; no random-frame split or generalization claim was made.

## 17. Revised root-cause assessment

| hypothesis | Before Stage-2 | After Stage-2 | evidence |
|---|---|---|---|
| NMS | Moderate | **Strong** | 0.70 NMS barely alters GT duplicate counts; stricter secondary NMS improves HOTA/IDSW at near-constant recall. |
| Score calibration | Moderate | Moderate | score removes many candidates but loses recall; overlapping score distributions remain. |
| Model/domain mismatch | Moderate | Moderate | decoded candidates show competing scattered boxes/low mutual IoU; no held-out scene distinguishes learned output defect from general domain mismatch. |
| Preprocessing mismatch | Weak | Weak | XYZI/range/frame contract remains matched; no direct violation found. |
| Tracker coupling | Strong | **Strong** | candidate-density reduction materially lowers IDs/IDSW and raises HOTA under identical tracker settings. |

**Causal assessment: Duplicate detections are a major causal driver of tracking instability.** The fixed-tracker experiment establishes directionally strong evidence, while the fragmentation trade-off shows they are not the only tracking phenomenon.

## 18. Retraining decision

**NO — postprocessing appears sufficient for the observed in-domain duplicate/IDSW failure; do not start retraining now.**

1. Direct stage evidence shows score-filtered scattered candidates, not a ROS artifact.
2. Existing 0.70 NMS leaves nearly all diagnostic duplicate GT frames.
3. Stricter secondary NMS cuts IDSW 443→273 and raises HOTA 0.05293→0.12750 at 80.59% recall.
4. No held-out scene exists to justify a model-quality or domain-generalization training claim.
5. A production decision still requires fresh raw-model NMS 0.10 replay and a sequence-disjoint MORAI evaluation.

## 19. Fresh raw-model NMS experiment

`infer_fresh_nms.py` performs fresh CUDA inference from the same historical checkpoint and input for each requested threshold. It changes only the instantiated `model.dense_head.model_cfg.POST_PROCESSING.NMS_CONFIG.NMS_THRESH`; it does not write the YAML, OpenPCDet checkout, ROS node, or tracker configuration. The default requested threshold is the historical 0.70. Every run has score threshold 0.10, `nms_gpu`, pre-max 4096, post-max 500, batch 1, and workers 0. Per-run external metadata records the requested/actual in-model threshold.

The fresh 0.70 output reproduces the historical baseline exactly at reported precision (29,056 detections, 81.05% recall, 7.17 predictions/GT, IDSW 443, HOTA 0.05293), validating the default path. Each of the other settings is independently generated from a separate model forward over the same 1,764 frames, not chained from another NMS output.

| fresh NMS IoU | recall | mean position error | predictions/GT | >=2 / >=5 duplicates | unmatched/frame | residual-centre jitter |
|---:|---:|---:|---:|---:|---:|---:|
| 0.70 | 81.05% | 0.370 m | 7.17 | 79.75% / 75.14% | 15.16 | 0.464 m |
| 0.50 | 80.94% | 0.494 m | 4.45 | 78.52% / 57.03% | 9.55 | 0.573 m |
| 0.30 | 80.84% | 0.692 m | 1.97 | 62.31% / 3.20% | 4.23 | 0.682 m |
| 0.20 | 80.66% | 0.767 m | 1.38 | 42.37% / 0.07% | 2.82 | 0.650 m |
| 0.10 | 80.59% | 0.802 m | 0.96 | 13.82% / 0.00% | 1.58 | 0.583 m |

## 20. Secondary-vs-fresh NMS comparison

Fresh 0.10 and historical-0.70-plus-secondary-0.10 are close but **not identical**: fresh has 5,071 detections versus secondary 5,069; only 1,066/1,764 frames have exactly the same `(x,y,score)` candidate set, with 1,212 fresh-only and 1,210 secondary-only rounded candidates. This is expected from score-ordered greedy NMS: suppressing at 0.70 before a second 0.10 pass changes the available suppression graph.

| method at 0.10 | recall | predictions/GT | >=2 rate | IDSW | fragmentation | HOTA |
|---|---:|---:|---:|---:|---:|---:|
| historical 0.70 + secondary 0.10 | 80.59% | 0.96 | 13.82% | 273 | 235 | 0.12750 |
| fresh raw-model 0.10 | 80.59% | 0.96 | 13.82% | 273 | 235 | 0.12755 |

Thus the practical conclusion holds under actual model-time NMS, while the two procedures must not be described as mathematically equivalent.

## 21. Tracker replay and fragmentation analysis

The tracker remains fixed: 3 m Euclidean gate, Hungarian, linear KF, detector yaw, `min_hits=1`, `max_age=2`. TrackEval decomposition is available and was used.

| fresh NMS | IDSW | fragmentation | tracker IDs | HOTA | DetA | AssA | LocA |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.70 | 443 | 73 | 5,916 | 0.05293 | 0.04843 | 0.05817 | 0.80311 |
| 0.50 | 520 | 106 | 4,362 | 0.06412 | 0.06552 | 0.06307 | 0.77564 |
| 0.30 | 468 | 184 | 2,728 | 0.08430 | 0.10754 | 0.06640 | 0.74860 |
| 0.20 | 375 | 213 | 2,150 | 0.09949 | 0.13397 | 0.07435 | 0.74100 |
| 0.10 | **273** | **235** | **1,674** | **0.12755** | **0.17778** | **0.09218** | 0.73652 |

DetA and AssA increase monotonically, while LocA falls because stronger NMS retains the highest-score rather than GT-nearest candidate. Fragmentation increases monotonically, confirming the stated continuity trade-off.

For event inspection, all 13 actor-frames that were officially matched at 0.70 but missed at fresh 0.10 were enumerated (actors 48, 31, 18, 20; examples frame indices 435, 980, 1008, 1014–1024, 1298–1301). These are category A: a true match was removed by NMS. There are 539 actor-frames missed by both 0.70 and 0.10: score is fixed at 0.10, so they are pre-existing model/score-stage temporary misses, not new NMS suppressions. The 162-fragmentation increase cannot be explained by only 13 removed official matches; the 0.370→0.802 m localization degradation and retained-score behavior support category C (wrong duplicate retained) as the main remaining mechanism. Direct tracker assignment provenance was not logged, so no unsupported exact C/E percentage is claimed; tracker `max_age` is unchanged.

## 22. Pareto post-processing candidates

| role | candidate | reason |
|---|---|---|
| Conservative | fresh 0.50 | recall loss 0.11 pp, substantial candidate reduction; fragmentation +33 and IDSW worse, so it is conservative detection-side only. |
| Balanced | fresh 0.20 | duplicate >=2 drops 46.9 points, IDSW 443→375, HOTA nearly doubles; fragmentation +140. |
| Aggressive | fresh 0.10 | largest duplicate/IDSW/HOTA improvement; fragmentation 73→235 and localization degradation make continuity cost explicit. |

There is no candidate that simultaneously improves every metric. On this tracker/evaluation, 0.10 is the strongest practical duplicate/identity remedy; 0.20 is the appropriate balanced candidate for a continuity-sensitive follow-up.

## 23. Sequence-disjoint MORAI validation

No new actor-GT-qualified sequence was acquired: MORAI simulator capture is not safely unattended in the current environment, and local storage has no suitable second scene. `morai_cam4_20260813_163222` is sequence-disjoint and replayable but has LiDAR/camera/localization data without `/ad/dev/objects` actor truth; it cannot support recall, duplicate-per-GT, IDSW, or HOTA evaluation.

Capture a new route/traffic/spawn arrangement for 2–5 minutes with a distinct bag basename and record at minimum:

```bash
export ROS_DOMAIN_ID=<unused-id>
ros2 bag record -o <new_scene_id> \
  /ad/sensors/lidar/points /ad/dev/objects /ad/dev/vehicle/ego_status \
  /tf /tf_static /clock
```

Required types/contracts are: PointCloud2, `ad_morai_interfaces_dev/msg/ObjectStatusArray`, `ad_morai_interfaces/msg/EgoVehicleStatus`, and TFMessage for both TF topics. The exporter requires lidar frame `lidar_link`, map frame `map`, the documented `odom→base_link→rear_axle_link→lidar_link` transform chain, and 30 ms actor/ego/TF alignment. Add the bag as a whole-scene `test` entry with scenario class evidence in a new dataset config, then run the exporter and evaluate only 0.70, 0.20, and 0.10; never train on it.

## 24. Final retraining decision

**NO — defer retraining.** Actual fresh in-model NMS 0.10 retains recall (−0.46 pp), reduces duplicate >=2 by 82.7%, cuts IDSW by 38.4%, and improves HOTA 141%. The remaining fragmentation/localization trade-off requires postprocessing selection and sequence-disjoint validation, not ungrounded model training. Fine-tuning becomes justified only if the capture above shows that no fresh NMS candidate retains a reasonable recall/fragmentation trade-off on a disjoint scene.

OpenPCDet’s 0.10 result cannot be copied directly into Autoware: Autoware uses separate TensorRT models, circle and IoU stages, search-distance logic, and optional densification. A separate Autoware audit must first dump its own raw/post-circle/post-IoU/final candidates and reproduce the same GT metrics.
