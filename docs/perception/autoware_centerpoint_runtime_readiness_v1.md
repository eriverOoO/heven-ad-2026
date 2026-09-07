# Autoware CenterPoint runtime readiness v1

## 1. Executive summary

**Runtime ready: PARTIAL.**  The audit worktree now has a strict held-out
MORAI bag validator, a read-only runtime readiness check, an experimental
CenterPoint selection, and a hash-verifying model provisioner.  No training,
NMS tuning, production selection change, or tracker change was made.

Actual TensorRT inference cannot yet start because this host has CUDA 11.8
compiler support but no discoverable TensorRT C++ libraries or `trtexec`,
and no installed `autoware_lidar_centerpoint_node`.  The official model
download is intentionally not executed: HEVEN's locked policy requires an
explicit license-review acknowledgement that this audit cannot self-assert.

## 2. Runtime dependencies

| Dependency | Required | Local status |
| --- | --- | --- |
| ROS | Humble | `/opt/ros/humble` present |
| Autoware source | Universe 0.51.0 / Autoware 1.8.0 contract | source at `/home/didgang1203/projects/autoware_tracker_ws/src/autoware_universe` |
| CenterPoint package | `autoware_lidar_centerpoint` 0.51.0 | source present; executable absent |
| CUDA compiler | CUDA 11.8 | present |
| NVIDIA driver/GPU | RTX 4060 | driver supports CUDA 12.6 |
| TensorRT | `nvinfer`, `nvonnxparser`, runtime builder | absent |
| Model | CenterPoint v3 ONNX set | absent; official URL and SHA locked |

The source package uses CMake `find_library(nvinfer)` and
`find_library(nvonnxparser)`; without both it installs launch/config only,
not the detector executable.  A full Autoware build is not required once the
TensorRT development runtime and its already-built dependencies are available,
but a minimal isolated build cannot proceed before that condition is met.

## 3. Model provisioning

The selected audit model is the full `centerpoint`, matching the explicit
experimental selection at
`ad_lidar_perception/config/experiments/autoware_centerpoint_audit_v1.yaml`.
It uses the Autoware 1.8.0 CenterPoint v3 artifacts:

| Artifact | SHA-256 |
| --- | --- |
| `pts_voxel_encoder_centerpoint.onnx` | `dc1a876580d86ee7a341d543f8ade2ede7f43bd032dc5b44155b1f0175405764` |
| `pts_backbone_neck_head_centerpoint.onnx` | `3fe7e128955646740c41a25be0c8f141d5a94594fe79d7405fe2a859e391542e` |
| `centerpoint_ml_package.param.yaml` | `9bbc16e521dd87c91cbadf1cb89c8b81393d1f8e1069af385aaba677576f0e27` |
| `detection_class_remapper.param.yaml` | `c711f8875ece9b527dfe31ffc75f8c0de2e77945ef67860a959a4e04c36772d5` |

After a human has reviewed the upstream artifact licence, run:

```bash
export AD_AUTOWARE_MODEL_LICENSE_REVIEWED=1
tools/centerpoint_offline/provision_autoware_centerpoint_model.sh
```

This writes only to
`/home/didgang1203/models/autoware/lidar_centerpoint/`, never overwrites a
hash-mismatched file, and downloads neither an engine nor a checkpoint.
Autoware's pinned 1.8.0 artifact task is the source for both URLs and hashes.

## 4. TensorRT engine and isolated overlay

When the compatible TensorRT development runtime is supplied, build only the
required package into this worktree; do not write the shared workspace:

```bash
export ROS_DOMAIN_ID=77
source /opt/ros/humble/setup.bash
source /home/didgang1203/projects/autoware_tracker_ws/install/setup.bash
colcon build --packages-select autoware_lidar_centerpoint \
  --base-paths /home/didgang1203/projects/autoware_tracker_ws/src/autoware_universe \
  --build-base /tmp/heven-worktrees/centerpoint-stability-audit-v1/.autoware_audit/build \
  --install-base /tmp/heven-worktrees/centerpoint-stability-audit-v1/.autoware_audit/install \
  --log-base /tmp/heven-worktrees/centerpoint-stability-audit-v1/.autoware_audit/log
```

Then source the isolated install and invoke the HEVEN experimental selection
with `AD_DATA_DIR=/home/didgang1203`.  `build_only:=true` is the official
node path that creates GPU-specific engines beside the verified ONNX files.
Record engine size, SHA-256, GPU, and TensorRT version externally; never commit
engines.

## 5. HEVEN integration and PointCloud contract

The learned-detector launch remaps:

```text
/ad/perception/lidar/points_xyzirc
  -> autoware_lidar_centerpoint
  -> /ad/perception/objects/detected
  -> autoware_multi_object_tracker
  -> /ad/perception/objects/tracked
```

The HEVEN layout adapter produces the strict compact `PointCloud2` layout:
`x,y,z` float32 at offsets 0,4,8; `intensity` uint8 at 12; `return_type`
uint8 at 13; and `channel` uint16 at 14, with 16-byte point stride. It
converts MORAI's strict XYZIRT cloud using ring as channel and a configured
return type. This is the contract to inspect live with
`ros2 topic echo --once /ad/perception/lidar/points_xyzirc` after startup.

## 6. Stage audit integration

`autoware_stage_audit.py` remains the analyzer for explicit candidate dumps.
The existing upstream node exposes only final `DetectedObjects`; raw,
score, circle-NMS and IoU-NMS candidates need a separate opt-in instrumentation
patch. Keep it off by default and build it only in the isolated overlay.

## 7. Held-out bag contract

`validate_morai_audit_bag.py` requires these exact topic/types:

| Topic | Type |
| --- | --- |
| `/ad/sensors/lidar/points` | `sensor_msgs/msg/PointCloud2` |
| `/ad/dev/objects` | `ad_morai_interfaces_dev/msg/ObjectStatusArray` |
| `/ad/dev/vehicle/ego_status` | `ad_morai_interfaces/msg/EgoVehicleStatus` |
| `/tf`, `/tf_static` | `tf2_msgs/msg/TFMessage` |
| `/clock` | `rosgraph_msgs/msg/Clock` |

It validates metadata topic/type/counts, duration, payload presence and held-out
metadata. `--deep` additionally deserializes payloads to check per-topic
timestamp monotonicity and non-empty actor arrays. The historic recording
correctly contains most topics but fails because its MCAP payload and `/clock`
are absent; it is also training data.

## 8. Exact bag recording and validation

```bash
export ROS_DOMAIN_ID=77
source /opt/ros/humble/setup.bash
ros2 bag record -o /safe/external/path/morai_autoware_audit_v1 \
  /ad/sensors/lidar/points /ad/dev/objects /ad/dev/vehicle/ego_status \
  /tf /tf_static /clock

PYTHONPATH=tools/centerpoint_offline python3 \
  tools/centerpoint_offline/validate_morai_audit_bag.py \
  /safe/external/path/morai_autoware_audit_v1 --deep \
  --scene '<scene>' --route '<route>' \
  --traffic-configuration '<traffic>' --spawn-configuration '<spawn>' \
  --used-for-training no \
  --output /home/didgang1203/datasets/centerpoint/autoware_stability_audit_v1/logs/bag_validation.json
```

## 9. Readiness and audit commands

Read-only environment diagnosis:

```bash
tools/centerpoint_offline/check_autoware_runtime.sh
```

The eventual audit command remains:

```bash
PYTHONPATH=tools/centerpoint_offline python3 tools/centerpoint_offline/autoware_stage_audit.py \
  --stage-dump /home/didgang1203/datasets/centerpoint/autoware_stability_audit_v1/stage_dumps/run.jsonl \
  --gt-json /safe/external/path/lidar_gt_by_frame.json \
  --output /home/didgang1203/datasets/centerpoint/autoware_stability_audit_v1/baseline/stage_summary.json
```

## 10. Remaining blockers

1. A reviewed, compatible TensorRT C++ development/runtime installation.
2. Explicit model-license acknowledgement before the official model download.
3. Isolated build of the detector executable after (1).
4. An opt-in stage-dump patch in that isolated source overlay.
5. A sequence-disjoint MORAI actor-GT bag meeting the validator contract.

No blocker is addressed by retraining, threshold tuning, or changing the
Euclidean production default.

