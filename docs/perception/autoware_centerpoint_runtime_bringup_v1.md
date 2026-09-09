# Autoware CenterPoint runtime bring-up v1

## 1. Executive summary

**Status: PARTIAL; model provision is complete, but host dependencies block the
detector build.**
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

The user reviewed the licence and authorized the acknowledgement. The
provisioner found existing artifacts and verified them without replacement:

| Artifact | Size | SHA-256 verification |
| --- | ---: | --- |
| `pts_voxel_encoder_centerpoint.onnx` | 7.3 KiB | `dc1a876…05405764` |
| `pts_backbone_neck_head_centerpoint.onnx` | 20 MiB | `3fe7e128…e391542e` |
| `centerpoint_ml_package.param.yaml` | 1.1 KiB | `9bbc16e…7576f0e27` |
| `detection_class_remapper.param.yaml` | 2.7 KiB | `c711f887…c36772d5` |

## 5. Isolated build and smoke path

Autoware 1.8.0's x86_64 TensorRT role pins
`10.8.0.43-1+cuda12.8` and packages `libnvinfer10`,
`libnvinfer-plugin10`, `libnvonnxparsers10`, and their matching development
and header packages. The current host has CUDA 11.8 and no APT candidate for
these packages; its only NVIDIA APT source is the CUDA WSL repository. A
global TensorRT/CUDA 12.8 side-by-side installation requires separate user
approval and must not be improvised by this audit. That Ansible package pin is
not source-level proof that CenterPoint requires CUDA 12.8: the source uses
the TensorRT C++ API, and TensorRT 10.8 officially supports CUDA 11.8. It
does **not** justify a driver replacement, CUDA removal, ROS reinstall, or a
broad APT upgrade.

The preferred next path is an exact TensorRT 10.8.0.43 Linux x86_64 CUDA-11.8
tar distribution obtained from NVIDIA's authenticated official download flow.
The artifact filename/URL must be taken from that flow, not inferred. Unpack
it outside the repository, then run:

```bash
export TENSORRT_ROOT=/home/didgang1203/opt/tensorrt/10.8.0.43-cuda11.8
bash tools/centerpoint_offline/check_local_tensorrt.sh
bash tools/centerpoint_offline/build_autoware_centerpoint_isolated.sh
```

Both scripts scope `PATH`, `LD_LIBRARY_PATH`, CMake include/library lookup,
and all build products to the invoking process/worktree; neither edits
`.bashrc`, `/usr`, nor shared HEVEN ROS build directories.

Do not install CUDA 12.8 or a global TensorRT package in this phase. CUDA 12.8
is an escalation only if the exact CUDA-11.8 TensorRT artifact is unavailable
or the local compile/build smoke produces a documented CUDA-specific failure.

After the approved dependency installation, use the worktree-only build helper
below. It builds `--packages-up-to autoware_lidar_centerpoint`, its source
dependency closure rather than all of Universe, and never uses the shared
HEVEN ROS workspace roots:

```bash
export ROS_DOMAIN_ID=77
export CMAKE_BUILD_PARALLEL_LEVEL=2
bash tools/centerpoint_offline/build_autoware_centerpoint_isolated.sh
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

1. An exact TensorRT 10.8.0.43 CUDA-11.8 tar root from NVIDIA's official
   download flow, or evidence that NVIDIA no longer offers that artifact.
2. The isolated dependency-closure build after (1).
3. GPU engine/model-load smoke testing after (2).
4. A sequence-disjoint MORAI bag meeting the validator contract.

No blocker is addressed by CenterPoint retraining, AV2 training, tracker
tuning, or changing the Euclidean production default.
