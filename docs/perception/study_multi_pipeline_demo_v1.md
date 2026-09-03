# Multi-pipeline study demo v1

This is an opt-in, same-bag visual study package. It does not change production
defaults, control, association, lifecycle, prediction, or occupancy algorithms.

---

## Tonight: run and record these three

Open one terminal, paste the **environment setup block** once, then run **A**,
record RViz with OBS / Windows screen recorder, `Ctrl-C`, and repeat for **B**
and **C**. Same bag, same rate (0.5x), same camera, same RViz layout every time.
**D is optional / exploratory** — run it only if you have spare time.

### Environment setup block (paste once per terminal)

```bash
source /opt/ros/humble/setup.bash
source /tmp/heven-study-demo-v1/install/setup.bash
source /home/didgang1203/venvs/heven-centerpoint/bin/activate
export AMENT_PREFIX_PATH=/home/didgang1203/ros-local-debs/extracted/opt/ros/humble:$AMENT_PREFIX_PATH
export PATH=/home/didgang1203/ros-local-debs/extracted/opt/ros/humble/bin:$PATH
export PYTHONPATH=/home/didgang1203/ros-local-debs/extracted/opt/ros/humble/local/lib/python3.10/dist-packages:$PYTHONPATH
export AD_DATA_DIR=/home/didgang1203/projects/heven-ad-2026/ad_data
export BAG=/home/didgang1203/projects/heven-ad-2026/morai_cam4_20260813_163222/morai_cam4_20260813_163222
export AB3DMOT_ROOT=/home/didgang1203/projects/heven-ad-2026/references/ab3dmot
export CP_CKPT=/home/didgang1203/heven_presentation_assets/end_to_end_detector_tracker/checkpoint/centerpoint_t14_reproduction.pth
export KN_CKPT=/home/didgang1203/heven_presentation_assets/kalmannet_training_stability/checkpoint/dense_kalmannet_v2.pt
export OPENPCDET_ROOT=/home/didgang1203/projects/OpenPCDet
```

Notes:
- `source /tmp/heven-study-demo-v1/install/setup.bash` is the isolated study
  overlay (it chains `/opt/ros/humble` and the main repo `install/`); it is the
  only overlay that carries `study_pipeline_rviz.launch.py`.
- `ros-humble-xacro` is not apt-installed on this box and `sudo` is unavailable,
  so the three `ros-local-debs` exports above put a locally-extracted `xacro` on
  the path. (Optional one-time cleanup for the user: `sudo apt install
  ros-humble-xacro`, after which the three `ros-local-debs` exports can be
  dropped.)
- The `heven-centerpoint` venv supplies `filterpy` (Linear KF, needed by A/C
  too) and the CUDA/torch/OpenPCDet stack (B/D).

### A — training-free (Euclidean + Hungarian + Linear KF)

```bash
ros2 launch ad_lidar_perception study_pipeline_rviz.launch.py \
  pipeline_variant:=training_free bag_path:=$BAG rate:=0.5 ab3dmot_root:=$AB3DMOT_ROOT
```

### B — learned detection (CenterPoint + Hungarian + Linear KF)

```bash
ros2 launch ad_lidar_perception study_pipeline_rviz.launch.py \
  pipeline_variant:=centerpoint_kf bag_path:=$BAG rate:=0.5 ab3dmot_root:=$AB3DMOT_ROOT \
  centerpoint_checkpoint:=$CP_CKPT openpcdet_root:=$OPENPCDET_ROOT centerpoint_device:=cuda:0
```

### C — learned estimation (Euclidean + Hungarian + KalmanNet)

```bash
ros2 launch ad_lidar_perception study_pipeline_rviz.launch.py \
  pipeline_variant:=euclidean_knet bag_path:=$BAG rate:=0.5 ab3dmot_root:=$AB3DMOT_ROOT \
  kalmannet_checkpoint:=$KN_CKPT kalmannet_device:=cpu
```

### D — CenterPoint + KalmanNet (OPTIONAL / exploratory)

```bash
ros2 launch ad_lidar_perception study_pipeline_rviz.launch.py \
  pipeline_variant:=centerpoint_knet bag_path:=$BAG rate:=0.5 ab3dmot_root:=$AB3DMOT_ROOT \
  centerpoint_checkpoint:=$CP_CKPT openpcdet_root:=$OPENPCDET_ROOT centerpoint_device:=cuda:0 \
  kalmannet_checkpoint:=$KN_CKPT kalmannet_device:=cpu
```

Each launch opens RViz automatically, replays the bag once (no loop) at 0.5x
with the front camera + localization + drivable mask enabled. Record ~30–60 s,
then `Ctrl-C`. The already-collected quantitative table (section 5) does **not**
need to be regenerated — use it as-is in the study session.

---

## 1. Pipeline table

| ID | Detector | Input protocol | Association / estimator | Status |
|---|---|---|---|---|
| A `training_free` | adaptive Euclidean | cropped → Patchwork++ nonground | BEV Euclidean 3.0 m, Hungarian / Linear KF | READY |
| B `centerpoint_kf` | historical T14 CenterPoint | cropped cloud directly; no ground removal | same gate, Hungarian / same Linear KF | READY (demonstration only) |
| C `euclidean_knet` | same as A | same as A | same gate, Hungarian / KalmanNet `[x,y,vx,vy]` | READY |
| D `centerpoint_knet` | same as B | same as B | same gate, Hungarian / same KalmanNet | OPTIONAL; runnable, not the main comparison |

The detector outputs the canonical `DetectedObjects` fields without invented
metadata: source stamp/frame, geometric centre, dimensions, yaw, score, and
class. B/D use the frozen 0.10 score threshold, 0.70 NMS threshold, and maximum
500 boxes from the historical model configuration; they were not tuned on this
bag. KalmanNet estimates already-associated planar state using actual message
timestamp `dt`; AB3DMOT still owns IDs, assignment, and lifecycle.

Audited artifacts:

- CenterPoint: `/home/didgang1203/heven_presentation_assets/end_to_end_detector_tracker/checkpoint/centerpoint_t14_reproduction.pth`, 93,474,618 bytes, SHA-256 `466c8181a377682e032bb32579c8ddb65807b5feebd8625a750b1d5538ddbc95`. T14 reproduction, seed 2026, epoch 3 / iteration 5292, 1764 training samples, val/test 0, 100% train/eval overlap. Three-class head (`vehicle`, `pedestrian`, `obstacle`), four point features `(x,y,z,intensity)`, range `[-4,-25,-3,100,25,5]`, OpenPCDet `233f849` / 0.6.0. Demonstration-compatible, not evidence of generalization.
- KalmanNet: `/home/didgang1203/heven_presentation_assets/kalmannet_training_stability/checkpoint/dense_kalmannet_v2.pt`, 31,436 bytes, SHA-256 `956604975e5204b2c584c3e8fe16a7ba9346097980566ecbbb22c2001fbf7d48`. Validation-selected dense v2, seed 1 / epoch 26, MORAI dense GT-identity sequences; 7,016-parameter residual-normalized FC-GRU-FC model, state `[x,y,vx,vy]`, measurement `[x,y]`, no dataset scaler, variable `dt`, historically Euclidean measurements. Runtime-compatible without changing the model.

## 2. Exact launch commands

See **"Tonight: run and record these three"** at the top of this document for
the copy-paste environment block and the A / B / C / D launch commands. Each
launch replays the same bag once from its beginning at 0.5x, enables the
recorded front camera + raw-input localization + drivable mask, opens the same
RViz file, and does not loop. Stop with `Ctrl-C` after recording.

`openpcdet_root=/home/didgang1203/projects/OpenPCDet` is an existing editable
checkout at the same pinned commit `233f849` as `references/openpcdet`; it is
used because its setup-generated `pcdet/version.py` is present. The submodule
itself remains the authoritative pinned source and was not modified.

## 3. What should appear in RViz

The shared `training_free_perception_camera.rviz` layout shows cropped 3D
LiDAR, detected/tracked/predicted boxes, track ID and velocity markers, future
prediction, Dynamic OGM, the front-camera Image panel, and a large A/B/C/D
pipeline label. The drivable mask is present as a disabled debug toggle. A
runtime smoke confirmed RViz stayed active and the camera publisher/subscriber
QoS matched Best-Effort/Volatile.

## 4. Suggested video filenames and recording

- `01_training_free_euclidean_kf.mp4`
- `02_centerpoint_kf.mp4`
- `03_euclidean_kalmannet.mp4`
- `04_centerpoint_kalmannet.mp4` (optional)

Use Windows OBS or the Windows screen recorder: 1920×1080 if available, 30 fps,
capture the whole RViz window, do not alter the layout, begin at the bag start,
record the same 60 source seconds, then Ctrl-C. No Linux capture setup is
required.

## 5. Same-bag GT-free quantitative table

Bag: `morai_cam4_20260813_163222`; rate 0.5; A/B/C exact source interval
`1786606342.500–1786606402.500` (60.0 s); RViz off only during collection.
Hz is based on source header stamps. D is a successful bounded exploratory run
starting 0.127 s earlier and is shown only as optional evidence, not part of the
locked causal table.

| Metric | A Euclid+KF | B CP+KF | C Euclid+KNet | D CP+KNet* |
|---|---:|---:|---:|---:|
| Detection messages | 517 | 508 | 522 | 502 |
| Detection Hz | 8.716 | 8.727 | 8.701 | 8.752 |
| Detections/frame mean | 5.489 | 7.419 | 5.446 | 7.474 |
| Candidate density B/A | 1.352× | — | — | — |
| Tracking Hz | 8.718 | 8.727 | 8.708 | 8.734 |
| Tracks/frame mean | 6.623 | 11.963 | 6.602 | 12.066 |
| Unique IDs | 447 | 2102 | 446 | 2108 |
| IDs created/min (lifecycle activity) | 447 | 2102 | 446 | 2108 |
| Motion jitter median / p90 (m) | 0.414 / 0.960 | 0.354 / 1.595 | 0.502 / 1.152 | 0.268 / 1.135 |
| Prediction Hz | 8.718 | 8.727 | 8.708 | 8.734 |
| Dynamic OGM non-empty | 48.6% | 36.6% | 48.7% | 36.3% |
| GPU peak | N/A | 273.7 MiB | N/A | 272.9 MiB* |

These are descriptive diagnostics without GT: jitter is not position error,
IDs created/deleted are not ID switches, and density is not asserted as the
sole cause of downstream behaviour. B CenterPoint model-forward latency over
601 samples was 89.184 ms median / 106.648 ms p95; complete callback boundary
(preprocess + forward + postprocess) was 123.115 / 143.915 ms. GPU figures are
PyTorch allocator peaks, not whole-device utilization.

## 6. Frozen historical GT-based results (separate experiment)

| Historical detector/tracker metric | Euclidean | CenterPoint |
|---|---:|---:|
| Localization error | 1.176 m | 0.368 m |
| Recall | 71.1% | 81.1% |
| Length bias | -2.92 m | -0.34 m |
| HOTA | 0.0630853 | 0.05293 |
| IDSW | 86 | 443 |
| Position jitter | 0.2884 m | 0.4632 m |

The historical CenterPoint evaluation had 100% training/evaluation overlap
(1764 train, val/test 0). It is not a generalization result.

| Historical estimator comparison | KalmanNet | Tuned Linear KF |
|---|---:|---:|
| Fair T9A position RMSE | ~0.854 m | ~0.855 m |
| Dense v2 position RMSE | 1.467 m | 1.456 m |
| Dense v2 velocity RMSE | 2.462 m/s | 2.416 m/s |

KalmanNet and tuned KF were effectively tied in T9A. Sequence construction was
important: fragmented T12 KalmanNet was 3.065 m; dense retraining restored it
to 1.476 m in the controlled ablation. Do not claim KalmanNet beats tuned KF.

## 7. Qualitative observation checklist

A vs B:

- [ ] distant object detection
- [ ] box dimensions and yaw quality
- [ ] split/merged detections and candidate density
- [ ] box jitter and prediction smoothness
- [ ] track births/deletions and ID continuity

A vs C:

- [ ] track-centre and velocity smoothness
- [ ] response to missed measurements
- [ ] recovery after measurement returns
- [ ] prediction-trajectory smoothness

Do not pre-fill these conclusions; score only what is visible in the recordings.

## 8. Five-line interpretation for presentation

1. Learned modules can improve a local metric without improving the whole tracking system.
2. Historical CenterPoint improved localization, recall, and dimensions, but worsened HOTA and IDSW in the frozen end-to-end experiment.
3. KalmanNet learned filtering behaviour, but did not clearly beat a fairly tuned Linear KF.
4. KalmanNet was sensitive to measurement and sequence construction.
5. A/B isolates learned detection and A/C isolates learned estimation; D is illustrative, not the main causal comparison.

## 9. Known limitations

No GT exists in the replay bag, so the new table contains no accuracy, recall,
HOTA, IDSW, or localization error. CenterPoint's checkpoint is same-scene and
overlap-contaminated. Its preprocessing differs historically from Euclidean
(cropped direct versus nonground), and that difference is documented rather
than silently changed. D's metrics are exploratory rather than exact-boundary.
Drivable-mask exact-stamp TF occasionally drops frames; Dynamic OGM remains
real and non-empty. Camera is a same-session 2D panel, not LiDAR-camera fusion.
CPU utilization and KalmanNet-only estimator latency were not separately
instrumented. Execution success is not performance validation.
