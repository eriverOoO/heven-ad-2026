# STEP 05-A: CenterPoint offline environment and MORAI dataset integration

## Scope and decision

This step prepares only an offline execution boundary. It does not start
training, tune hyperparameters, compare performance, integrate ROS, modify the
Autoware tracker, or change the production perception pipeline.

The selected framework is the official OpenPCDet repository at immutable
commit `233f849829b6ac19afb8af8837a0246890908755` (its setup metadata reports
`0.6.0`). The commit is recorded in
`tools/centerpoint_offline/upstream.lock.yaml`. OpenPCDet is kept as an external
checkout and the MORAI class is registered at runtime; no upstream file is
patched or vendored. This choice provides the upstream CenterPoint model and a
documented unified LiDAR box convention while keeping repository ownership
clear.

## Audited host state (2026-08-15)

| item | observed state |
|---|---|
| OS / kernel | Ubuntu 22.04.5, Linux 6.8.0-136-generic x86_64 |
| Python | 3.10.12 |
| compiler | GCC/G++ 11.4.0, CMake 3.22.1 |
| visible display adapter | Intel Alder Lake-P integrated graphics only |
| NVIDIA package | `nvidia-driver-535` 535.288.01 installed |
| NVIDIA runtime | `nvidia-smi` cannot communicate with the driver; no NVIDIA kernel module/GPU is visible |
| CUDA toolkit | `nvcc` and `/usr/local/cuda*` absent |
| PyTorch | files exist in the user site, but import fails because `libtorch_global_deps.so` is missing |
| OpenPCDet/spconv | not installed in the active Python environment |

Therefore the data adapter and batch contract can be checked on this host, but
OpenPCDet model construction and CUDA forward cannot be claimed as passing.
CenterPoint's upstream `CenterHead` also creates class-mapping tensors with
`.cuda()` during construction, so a CPU-only model construction is not a valid
substitute without changing upstream code.

## Pinned target environment

Use a separate Python 3.10 environment with:

- an NVIDIA CUDA-capable GPU exposed to the process and a driver compatible
  with CUDA 11.8 (Linux driver 520.61.05 or newer);
- CUDA Toolkit 11.8 including `nvcc`, because OpenPCDet builds custom CUDA
  extensions;
- PyTorch `2.1.2+cu118`, torchvision `0.16.2+cu118`, and
  `spconv-cu118==2.3.6`;
- the exact remaining packages in `requirements-cu118.txt`.

No minimum VRAM is asserted here: it depends on the future batch size, voxel
population, model configuration, and training policy, none of which may be
tuned on the current static-only dataset.

Installation, from a directory outside the HEVEN repository:

```bash
python3 -m venv centerpoint-env
source centerpoint-env/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  -r /path/to/heven-ad-2026/tools/centerpoint_offline/requirements-cu118.txt

git clone https://github.com/open-mmlab/OpenPCDet.git OpenPCDet
git -C OpenPCDet checkout --detach 233f849829b6ac19afb8af8837a0246890908755
test "$(git -C OpenPCDet rev-parse HEAD)" = \
  233f849829b6ac19afb8af8837a0246890908755
python -m pip install -e ./OpenPCDet
```

Before smoke execution, all of these must succeed:

```bash
nvidia-smi
nvcc --version
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python -c 'import spconv.pytorch, pcdet; print("imports ok")'
```

## Dataset adapter and configuration

`MoraiHevenDatasetCore` reads both the established unversioned STEP 03 export
and `morai_heven_v1`. It validates sample IDs, finite XYZI, point counts,
classes, positive dimensions, and `lidar_link`. If present,
`num_lidar_points_inside_box` is retained as `gt_num_lidar_points`; legacy
labels use `-1` to mean “not exported”, never an inferred count.

`register_with_openpcdet()` creates a `DatasetTemplate` subclass and inserts it
into OpenPCDet's dataset registry at runtime. It passes raw points and labels to
the upstream `prepare_data()` path, which appends 1-based class IDs and applies
the configured range/voxel processors. `evaluation()` intentionally raises
`NotImplementedError`: STEP 05-A defines no MORAI performance metric.

The smoke dataset configuration uses the exporter ROI
`[-4,-25,-3,100,25,5]` metres and voxel size `[0.125,0.125,0.2]`, producing an
integer grid `[832,400,40]`. Shuffle and augmentation are disabled. These are
integration values, not tuned training hyperparameters. The model YAML contains
only an untrained CenterPoint architecture and has no optimizer or schedule.

### Coordinate and box proof

The exporter contract and OpenPCDet unified LiDAR contract are identical:

| property | `morai_heven` | OpenPCDet adapter |
|---|---|---|
| frame | `lidar_link` | LiDAR frame |
| axes | +x forward, +y left, +z up | +x forward, +y left, +z up |
| box | `[x,y,z,length,width,height,yaw]` | `[x,y,z,dx,dy,dz,heading]` |
| center | geometric box center | geometric box center |
| yaw | CCW from +x around +z | CCW from +x around +z |

Here `dx=length`, `dy=width`, and `dz=height`, so geometry conversion is the
identity. Class names map in the fixed order `vehicle=1`, `pedestrian=2`,
`obstacle=3`. The verification script reconstructs the seven values directly
from the source JSON and requires a zero maximum delta before batching.

## Prediction bridge to the STEP 02 metric

`prediction_bridge.py` defines `heven.offline_detection.v1`, one JSON object per
LiDAR source frame:

```json
{
  "schema": "heven.offline_detection.v1",
  "sample_id": "scene_stamp",
  "source_header_stamp_ns": 123,
  "frame_id": "lidar_link",
  "inference_time_ms": 12.3,
  "detections": [
    {"class_name": "vehicle", "score": 0.8,
     "box_lidar": [10.0, 1.0, 0.5, 4.5, 1.8, 1.6, 0.1]}
  ]
}
```

It converts OpenPCDet's 1-based `pred_labels`, validates scores and dimensions,
and preserves the seven box values except canonical yaw wrapping. Its
`benchmark_centers()` exposes the `(x,y)` centers consumed by the current STEP
02 distance metric. `source_header_stamp_ns` is deliberately the input LiDAR
header stamp, preserving exact frame identity. `inference_time_ms` is optional
offline inference duration and must not be confused with the STEP 02 ROS
publish-to-output latency. Feeding JSONL directly into the benchmark or
serializing it to `DetectedObjects` is future evaluation work, not performed in
this step.

## Verification commands and observed result

CPU-only unit and real-data checks:

```bash
cd /path/to/heven-ad-2026
python3 -m unittest discover \
  -s tools/centerpoint_offline -p 'test_*.py' -v
python3 tools/centerpoint_offline/verify_integration.py \
  --dataset ../../datasets/morai_heven \
  --batch-size 2
```

Both unit tests passed. On the real established dataset:

- split length: 1,764;
- sample `static_20260805_003151_1785857513201006723`:
  points `(14171,4)`, GT boxes `(1,7)`, class `obstacle`;
- two-sample batch: points `(28342,5)`, padded GT `(2,1,8)`, with point batch
  indices `{0,1}`;
- source JSON to adapter box maximum absolute delta: `0.0`;
- the same shapes and zero delta were reproduced with
  `/tmp/morai_heven_v1_repro`, proving versioned/unversioned read compatibility.

CUDA smoke command for the prepared target host:

```bash
python3 tools/centerpoint_offline/verify_integration.py \
  --dataset ../../datasets/morai_heven_v1 \
  --openpcdet-root /path/to/OpenPCDet \
  --attempt-model-smoke
```

On the audited host this was attempted and correctly reported failure at
PyTorch import (`libtorch_global_deps.so` missing). Model construction and a
forward pass remain blocked until the pinned CUDA environment is installed on
a visible NVIDIA GPU.

## Preconditions before real training

In addition to the requested extra bags and their scenario evidence:

1. bring up and re-run the CUDA smoke above on the actual training GPU;
2. validate the final merged `morai_heven_v1` and freeze its metadata/config
   digest and scene-level train/val/test allocation;
3. obtain enough populated validation and test scenes—current `val` and `test`
   are intentionally empty;
4. approve a training configuration (batch size, epochs, augmentation, class
   handling, thresholds) using the real dataset distribution and GPU memory;
5. define and implement MORAI evaluation metrics/class policy before any
   performance claim;
6. choose checkpoint/output storage and experiment provenance. Model weights,
   logs, and generated artifacts must remain outside Git.

## STEP 05-B: minimal CUDA training smoke

`train_morai_centerpoint.py` reuses the MORAI adapter and the minimal runtime
import path above. It builds the OpenPCDet dataset preprocessing path,
CenterPoint, Adam optimizer, and an epoch-based learning-rate scheduler. It
performs real CUDA forward/backward/optimizer steps and writes atomic
checkpoints, but intentionally performs no evaluation or metric calculation.
The training YAML inherits the existing smoke architecture and only adds
untuned optimizer defaults.

From the pinned Python environment, run a two-iteration smoke (outputs remain
outside the repository):

```bash
cd "$HOME/projects/heven-ad-2026"
"$HOME/venvs/heven-centerpoint/bin/python" \
  tools/centerpoint_offline/train_morai_centerpoint.py \
  --dataset "$HOME/datasets/morai_heven" \
  --openpcdet-root "$HOME/projects/OpenPCDet" \
  --epochs 1 \
  --batch-size 1 \
  --workers 0 \
  --output-dir /tmp/heven-centerpoint-smoke-step05b \
  --seed 2026 \
  --max-iterations 2
```

`--max-iterations` is an optional safety cap intended for smoke validation.
Without it, all batches in the requested epochs are processed. Each iteration
logs epoch, iteration, total loss, learning rate, current allocated GPU memory,
and peak allocated GPU memory. Each completed epoch produces
`checkpoint_epoch_XXX.pth` and atomically replaces `checkpoint_latest.pth`.
These checkpoints prove pipeline execution only; the current repeated static
scene cannot support a model-performance claim.

This exact command was verified on 2026-08-16 with an RTX 4060, PyTorch
2.1.2+cu118, and spconv-cu118 2.3.6. Both capped iterations completed forward,
backward, and optimizer steps. The observed losses were `57.568012` and
`104.720093`; peak memory allocated as reported by PyTorch was 329.1 MiB. It
created `checkpoint_epoch_001.pth` and `checkpoint_latest.pth` under the output
directory. These observations are execution evidence, not evaluation results.

Related validation commands are:

```bash
cd "$HOME/projects/heven-ad-2026"
"$HOME/venvs/heven-centerpoint/bin/python" -m unittest discover \
  -s tools/centerpoint_offline -p 'test_*.py' -v
"$HOME/venvs/heven-centerpoint/bin/python" -m compileall -q \
  tools/centerpoint_offline
```

### Short dry-run and resume verification

The following capped commands validate checkpoint continuation without turning
the repeated static scene into a performance experiment. The first process
runs three optimizer steps in epoch 1. The second, fresh process restores the
latest checkpoint and runs one more step in epoch 2:

```bash
cd "$HOME/projects/heven-ad-2026"
"$HOME/venvs/heven-centerpoint/bin/python" \
  tools/centerpoint_offline/train_morai_centerpoint.py \
  --dataset "$HOME/datasets/morai_heven" \
  --openpcdet-root "$HOME/projects/OpenPCDet" \
  --epochs 1 --batch-size 1 --workers 0 --seed 2026 \
  --max-iterations 3 \
  --output-dir /tmp/heven-centerpoint-dry-run-step05b2 \
  --summary-json /tmp/heven-centerpoint-dry-run-step05b2/dry_run_summary.json

"$HOME/venvs/heven-centerpoint/bin/python" \
  tools/centerpoint_offline/train_morai_centerpoint.py \
  --dataset "$HOME/datasets/morai_heven" \
  --openpcdet-root "$HOME/projects/OpenPCDet" \
  --epochs 2 --batch-size 1 --workers 0 --seed 2026 \
  --max-iterations 4 \
  --resume /tmp/heven-centerpoint-dry-run-step05b2/checkpoint_latest.pth \
  --output-dir /tmp/heven-centerpoint-dry-run-step05b2 \
  --summary-json /tmp/heven-centerpoint-dry-run-step05b2/dry_run_summary.json
```

Verified on 2026-08-17: the accumulated run completed two capped epochs and
four iterations with no non-finite loss. Resume restored model, optimizer,
scheduler, epoch 1, and iteration 3; the added step used the scheduled learning
rate `0.0001` and advanced the checkpoint to epoch 2 / iteration 4. PyTorch
reported 356.8 MiB peak allocated VRAM. The resulting
`dry_run_summary.json` is machine-readable execution evidence only. No
accuracy, recall, precision, mAP, or comparative evaluation was run.

## STEP 05-C: offline inference and benchmark interface

`infer_morai_centerpoint.py` constructs the same fixed CenterPoint architecture
and MORAI class order, then requires a strict checkpoint state-dictionary load.
It runs evaluation-mode CUDA inference and writes one
`heven.offline_detection.v1` JSON object per source frame. CUDA synchronization
brackets only the model forward call; dataset reads and JSON serialization are
outside `inference_time_ms`. For batches larger than one, the synchronized
batch time is reported as amortized per-frame time.

The verified ten-frame interface command was:

```bash
cd "$HOME/projects/heven-ad-2026"
"$HOME/venvs/heven-centerpoint/bin/python" \
  tools/centerpoint_offline/infer_morai_centerpoint.py \
  --dataset "$HOME/datasets/morai_heven" \
  --openpcdet-root "$HOME/projects/OpenPCDet" \
  --checkpoint /tmp/heven-centerpoint-dry-run-step05b2/checkpoint_latest.pth \
  --split train \
  --output /tmp/heven-centerpoint-inference-step05c/predictions.jsonl \
  --batch-size 1 --workers 0 \
  --score-threshold 0.1 --max-detections 500 \
  --max-frames 10 --seed 2026 \
  --summary /tmp/heven-centerpoint-inference-step05c/summary.json
```

The run strictly loaded epoch 2 / iteration 4 and processed ten frames. The
JSONL reader in `prediction_bridge.py` validates schema, frame, boxes, scores,
and fixed class mapping. Its small `benchmark_records()` adapter exposes source
timestamps, predicted XY centers, class names, and scores in input order. It
does not feed model inference time into the existing ROS/bag latency metric or
alter that metric's semantics.

Observed interface-validation output was 490 predictions (49.0/frame). CUDA
model-forward latency was mean 127.471 ms, p50 18.235 ms, p95 616.255 ms, and
maximum 1098.796 ms. The first measured call includes CUDA/model warm-up cost;
these values were retained as observed and are not a performance benchmark.
The dry-run checkpoint and single static scene cannot support any detector
quality or comparative claim.
