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
