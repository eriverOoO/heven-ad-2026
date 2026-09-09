# Autoware CenterPoint runtime bring-up v1

## 1. Executive summary

**Status: BLOCKED on host dependencies and model licence acknowledgement.**
The isolated runtime path, exact input contract, model provisioner, held-out
bag validator, and a bounded synthetic XYZIRC publisher are ready. No model
was downloaded, no TensorRT engine was built, and no production launch or
detector selection changed.

The host exposes CUDA 11.8 compiler tooling but no discoverable TensorRT C++
libraries (`nvinfer`, `nvonnxparser`), `trtexec`, or installed
`autoware_lidar_centerpoint_node`. GPU/NVML was inaccessible to this audit
sandbox, so a GPU smoke result cannot be claimed.

## 2. Concurrent resource state

At bring-up start, MemAvailable was approximately 12 GiB, swap use was zero,
and root disk free space was approximately 898 GiB. GPU inspection returned
an OS-level NVML access denial. The CenterPoint branch was clean at
`d2fec2f109d1424f00e2199ab68c6c4e3fdc10c0` before this work.

No KalmanNet worktree, AV2 source data, shared ROS build/install/log tree, or
main checkout was written. The reserved KalmanNet worktrees are intentionally
not named as runtime dependencies.

## 3. Runtime dependency compatibility

| Item | Required contract | Observed status |
| --- | --- | --- |
| ROS | Humble | `/opt/ros/humble` available |
| Autoware source | Universe 0.51 / Autoware 1.8 contract | source package available |
| Detector source | `autoware_lidar_centerpoint` 0.51.0 | source only |
| CUDA compiler | CUDA 11.8 | available |
| TensorRT development/runtime | `nvinfer`, `nvonnxparser` | absent |
| Detector executable | `autoware_lidar_centerpoint_node` | absent |
| GPU status | NVML-visible RTX runtime | unresolved in sandbox |

The package's CMake checks for both TensorRT libraries before compiling the
detector executable. Therefore an isolated `colcon` build is deliberately not
started: it would be known to omit the executable, and installing TensorRT is
a user-approval/global-system boundary.

## 4. Model provisioning

The selected experimental model is full `centerpoint`, matching
`ad_lidar_perception/config/experiments/autoware_centerpoint_audit_v1.yaml`.
The existing provisioner pins the Autoware 1.8.0 CenterPoint v3 URLs and
SHA-256 digests, writes only below
`/home/didgang1203/models/autoware/lidar_centerpoint/`, and refuses a
hash-mismatched replacement.

It is intentionally gated:

```bash
export AD_AUTOWARE_MODEL_LICENSE_REVIEWED=1
tools/centerpoint_offline/provision_autoware_centerpoint_model.sh
```

Only run this after a human reviews the upstream model licence. This audit did
not set the acknowledgement and did not download artifacts.

## 5. Isolated build and smoke path

After compatible TensorRT is available, use the worktree-only roots below;
never the shared HEVEN ROS workspace roots:

```bash
export ROS_DOMAIN_ID=77
export CMAKE_BUILD_PARALLEL_LEVEL=2
source /opt/ros/humble/setup.bash
source /home/didgang1203/projects/autoware_tracker_ws/install/setup.bash
colcon build --packages-select autoware_lidar_centerpoint \
  --base-paths /home/didgang1203/projects/autoware_tracker_ws/src/autoware_universe \
  --build-base .autoware_runtime/build \
  --install-base .autoware_runtime/install \
  --log-base .autoware_runtime/log
```

`tools/centerpoint_offline/synthetic_xyzirc.py` is the bounded smoke
publisher. It publishes five deterministic, non-empty frames by default to
`/ad/perception/lidar/points_xyzirc`, with `lidar_link` as frame ID. It is not
a MORAI substitute and does not validate detection quality.

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=77
PYTHONPATH=tools/centerpoint_offline python3 \
  tools/centerpoint_offline/synthetic_xyzirc.py --count 5
```

The layout is exactly the HEVEN adapter contract: `x,y,z` float32 at offsets
0/4/8, `intensity` uint8 at 12, `return_type` uint8 at 13, `channel` uint16
at 14, and point stride 16 bytes.

## 6. HEVEN integration and stage audit

The intended experimental graph remains:

```text
/ad/perception/lidar/points_xyzirc
  -> autoware_lidar_centerpoint
  -> /ad/perception/objects/detected
  -> /ad/perception/objects/tracked
```

The node's upstream public interface exposes final `DetectedObjects`; it does
not expose decoded, score, circle-NMS or IoU-NMS candidates. The pre-existing
`autoware_stage_audit.py` can analyze an opt-in dump once an isolated overlay
has a minimal instrumentation patch. The patch must default off and cannot be
implemented or tested against a nonexistent executable.

## 7. MORAI held-out replay readiness

The exact acceptance check is already available:

```bash
PYTHONPATH=tools/centerpoint_offline python3 \
  tools/centerpoint_offline/validate_morai_audit_bag.py <bag> --deep \
  --scene '<scene>' --route '<route>' \
  --traffic-configuration '<traffic>' --spawn-configuration '<spawn>' \
  --used-for-training no
```

It requires LiDAR, actor GT, ego status, TF, TF static, and clock with the
specific ROS message types documented in
`autoware_centerpoint_runtime_readiness_v1.md`. The new sequence must be
route/traffic/spawn-disjoint from training.

## 8. Remaining blockers

1. A human review/acknowledgement of the pinned official model licence.
2. A compatible TensorRT C++ development/runtime installation (requires user
   approval before any system change).
3. GPU visibility for engine/model-load smoke testing.
4. A sequence-disjoint MORAI bag meeting the validator contract.

No blocker is addressed by CenterPoint retraining, AV2 training, tracker
tuning, or changing the Euclidean production default.
