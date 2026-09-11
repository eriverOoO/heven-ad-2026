# HEVEN CenterPoint Competition Integration v1

## 1. Goal and boundary

This is an opt-in integration contract for the pinned Autoware Universe 0.51
CenterPoint runtime. It does not change the checked-in Euclidean production
default, detector thresholds, NMS, tracker parameters, or model weights.

The AV2 sparse-input audit is frozen as diagnostic evidence only: its
one-log source-ring proxy is not a MORAI or physical VLP-16 performance claim.

## 2. Backend selection and exclusion

`lidar_perception.launch.py` has three explicit detector choices:

| Launch value | Runtime | Output publisher |
| --- | --- | --- |
| `euclidean` (default) | Existing adaptive Euclidean clusterer | Existing detector node |
| `centerpoint` | Historical local OpenPCDet wrapper | `ad_centerpoint_detector` |
| `autoware_centerpoint` | Pinned Autoware 0.51 TensorRT CenterPoint | `autoware_lidar_centerpoint` |

`autoware_centerpoint` selects
`config/experiments/autoware_centerpoint_competition_candidate_v1.yaml`, whose
detector is exactly `centerpoint` and tracker is exactly `autoware`. The launch
branches are mutually exclusive: it includes `object_detection.launch.py` and
does not start either legacy detector. All detector choices publish the same
`/ad/perception/objects/detected` contract, so only one publisher is allowed.

## 3. Topic and frame contract

```text
/ad/sensors/lidar/points                 sensor_msgs/msg/PointCloud2
  -> /ad/perception/lidar/cropped         PointCloud2 (existing self crop)
  -> /ad/perception/lidar/points_xyzirc   PointCloud2 (Autoware input)
  -> /ad/perception/objects/detected      autoware_perception_msgs/msg/DetectedObjects
  -> /ad/perception/objects/tracked       autoware_perception_msgs/msg/TrackedObjects
  -> /ad/perception/objects/predicted     ad_interfaces/msg/PredictedObjectArray
```

The converter accepts only the existing MORAI XYZIRT source layout:
`x/y/z/intensity` `float32`, `ring` `uint16`, `time` `float32`, point step 22.
It preserves coordinates and header/frame/stamp, rejects malformed/non-finite
input, and emits exact XYZIRC: `x/y/z float32`, `intensity uint8`,
`return_type uint8`, `channel uint16`, point step 16. `return_type` is a
documented configured placeholder until a real MORAI bag validates an actual
return semantic. No source-cloud field assumption is silently accepted.

Autoware CenterPoint densification and the tracker use `odom`; prediction
rejects tracked input whose header frame is not `odom`. A live bag must provide
the TF chain required by the selected tracker and CenterPoint; coordinates are
never transformed merely to change a frame-id string.

## 4. Runtime and artifact contract

The candidate requires CUDA 11.8, local TensorRT 10.8.0.43 CUDA-11.8, pinned
full CenterPoint artifacts and prebuilt engines. The candidate helper validates
the two ONNX files, model/remapper YAML files, both engines, TensorRT libraries,
licensed-artifact acknowledgement, ROS overlay, and Autoware packages before
launching. It deliberately requires persistent overlay paths through
`HEVEN_ROS_OVERLAY` and `HEVEN_AUTOWARE_OVERLAY`; `/tmp` is not a deployment
default.

```bash
export HEVEN_ROS_OVERLAY=/absolute/persistent/heven/install
export HEVEN_AUTOWARE_OVERLAY=/absolute/persistent/autoware/install
export AD_AUTOWARE_MODEL_LICENSE_REVIEWED=1
export AD_DATA_DIR=/home/didgang1203
bash tools/centerpoint_runtime/run_heven_centerpoint_candidate.sh
```

The model root is `$AD_DATA_DIR/models/autoware/lidar_centerpoint`. Its
provenance lock and SHA checks remain authoritative.

## 5. Health and performance tools

`check_heven_centerpoint_runtime.py` checks exact topic types and publisher /
subscriber counts. Missing XYZIRC is reported as `WAITING_FOR_SENSOR`; a
downstream missing topic after input exists is `BLOCKED`, not a model claim.

```bash
python3 tools/centerpoint_runtime/check_heven_centerpoint_runtime.py
python3 tools/centerpoint_runtime/collect_runtime_telemetry.py \
  --duration-sec 60 --output /external/results/centerpoint_telemetry.jsonl
```

Telemetry is bounded to one sample per second and records GPU utilisation,
VRAM, available RAM, and swap. FPS and p95 end-to-end latency must be measured
on a real MORAI stream; no synthetic or AV2 value is a competition claim. The
existing `ad_record_detected_objects` records detector receipt timing; summarize
it after a replay rather than estimating FPS from launch logs:

```bash
ros2 run ad_lidar_perception ad_record_detected_objects \
  --backend centerpoint --count 100 --output /external/results/detected.jsonl
python3 tools/centerpoint_runtime/summarize_detection_runtime.py \
  /external/results/detected.jsonl
```

## 6. MORAI field-test and A/B workflow

First record a 30--60 s bag with LiDAR, actor GT, ego status, TF, TF static and
clock. Validate it before replay:

```bash
ros2 bag record \
  /ad/sensors/lidar/points /ad/dev/objects \
  /ad/dev/vehicle/ego_status /tf /tf_static /clock

python3 tools/centerpoint_offline/validate_morai_audit_bag.py BAG \
  --scene NAME --route NAME --traffic-configuration NAME \
  --spawn-configuration NAME --used-for-training no --deep
```

Then run the same validated bag once per backend. `run_morai_detector_ab.sh`
does not tune parameters and records the bag metadata requirement before launch.
It explicitly selects the checked-in `lidar_perception.yaml` legacy composition
for the Euclidean arm, so both arms retain the same Autoware tracker,
prediction, and occupancy selections; only the detector branch differs.

```bash
bash tools/centerpoint_runtime/run_morai_detector_ab.sh \
  --backend legacy --bag BAG --scene NAME --route NAME \
  --traffic NAME --spawn NAME

bash tools/centerpoint_runtime/run_morai_detector_ab.sh \
  --backend centerpoint --bag BAG --scene NAME --route NAME \
  --traffic NAME --spawn NAME
```

Compare detected/tracked/predicted topics and, only with actor GT, existing
detection metrics (recall, position error, Pred/GT, duplicates), tracker
metrics (HOTA, IDSW, fragmentation when evaluator support exists), and runtime
telemetry. The sequence is baseline, stage diagnosis, minimal calibration, then
tracker E2E; fine-tuning is not authorized by this document.

## 7. Current limitations and recovery

Runtime readiness is not a MORAI E2E validation. The remaining hard blocker is
a sequence-disjoint MORAI actor-GT bag with the field and TF contracts above.
On failure, use the candidate helper's fail-fast error first: overlay, TensorRT,
model/engine, or licence acknowledgement failures are distinct from a missing
sensor, unsupported XYZIRT schema, missing TF, tracker, or prediction topic.

For startup reliability, run bounded start -> health -> stop -> start cycles.
Confirm one detector publisher, no orphan process group, no engine rebuild, and
that VRAM returns after the process group exits. Do not change production
defaults while performing these checks.
