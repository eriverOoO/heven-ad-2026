# CenterPoint MORAI Training Prep v1

**Audit + config + preflight only. NO FULL TRAINING WAS RUN.** No optimizer
step, no backward pass, no epoch loop, no fine-tuning, no weights downloaded,
no OpenPCDet downloaded, no production detector default changed, no AB3DMOT /
tracker / planner change, no frozen T-2…T-16 conclusion touched, and no new
CenterPoint performance or generalization claim.

## Objective

Lock the complete interface between the leakage-safe
`centerpoint_morai_adapter_v1` dataset (PR #29) and the repository's **current**
CenterPoint / OpenPCDet model stack (`tools/centerpoint_offline/`), and provide
a reproducible preflight that refuses training/evaluation until a genuinely
non-overlapping validation split exists.

## Why this task exists — the historical overlap problem

The historical CenterPoint work (`docs/research/centerpoint_status.md`,
T‑14) trained on **1764 samples of one static scene**, with `val` and `test`
splits **empty** — `~/datasets/morai_heven/splits/` is `train=1764, val=0,
test=0`. Evaluation frames were a **100 % subset** of training frames. The
"0.368 m localization" figure from that era is therefore a same-scene,
in-distribution number, **not** a generalization result, and must never be
compared against a future MORAI-adapter result until an independent validation
split actually exists. This preflight makes that a hard gate.

## Current CenterPoint stack (audited on `main` @ PR #29)

| piece | current state |
| --- | --- |
| architecture | `CenterPoint` (OpenPCDet): `MeanVFE` → `VoxelResBackBone8x` → `HeightCompression` (256 BEV) → `BaseBEVBackbone` → `CenterHead` (`FEATURE_MAP_STRIDE 8`, `HEAD_ORDER [center, center_z, dim, rot]`) |
| model config | `tools/centerpoint_offline/configs/morai_centerpoint_smoke.yaml` (architecture only), `morai_centerpoint_train.yaml` (+ untuned Adam) |
| data config | `tools/centerpoint_offline/configs/morai_heven_dataset.yaml` |
| loader | `tools/centerpoint_offline/morai_dataset.py::MoraiHevenDatasetCore` (torch-free per-sample-JSON) + `make_openpcdet_dataset(DatasetTemplate)` adapter |
| runtime wrapper | `openpcdet_runtime.py` (imports only `DatasetTemplate` + `CenterPoint` from the pinned checkout) |
| training entry point | `train_morai_centerpoint.py` (CUDA forward/backward/optimizer, atomic checkpoints) — **not run here** |
| evaluation entry point | none — `MoraiHevenDataset.evaluation()` raises `NotImplementedError`; `EVAL_METRIC: morai_not_implemented` |
| expected point features | `[x, y, z, intensity]`, `NUM_POINT_FEATURES = 4` |
| expected GT box | `[x, y, z, length, width, height, yaw]`, geometric centre, `lidar_link`, yaw CCW +x, identity conversion |
| class list | historical: `[vehicle, pedestrian, obstacle]` (id = index + 1) |
| dependencies | `torch==2.1.2+cu118`, `spconv-cu118==2.3.6`, OpenPCDet @ `233f849` (`references/openpcdet` submodule) — see `requirements-cu118.txt` |
| checkpoint format | `torch.save` dict with `model_state` / `optimizer_state` / `scheduler_state` / `epoch` / `iteration` |

### Historical training config (labelled HISTORICAL — not adopted)

`morai_centerpoint_train.yaml` inherits `morai_centerpoint_smoke.yaml` and adds:
Adam, `LR 0.001`, `WEIGHT_DECAY 0.01`, `DECAY_STEP_LIST [1]`, `LR_DECAY 0.1`,
`LR_CLIP 1e-5`. No epoch count, no augmentation, no GT/database sampler, no
checkpoint init. The T‑14 reproduction used **seed 2026, 3 epochs, batch 1**
over the full 1764-sample overlapping split. These values are recorded for
provenance and are **not** silently reused as the v1 training config.

## Model-vs-data change discipline

v1 changes **data discipline only**. The architecture is held identical to the
existing smoke config with **one** exception forced by the adapter being
vehicle-only: the class list is reduced to `[vehicle]`, so the `CenterHead`
heatmap (`hm`) output channel is 1 instead of 3. Voxel size, point-cloud
range, backbone, BEV channels, `FEATURE_MAP_STRIDE`, NMS, and score threshold
are unchanged.

## Canonical input

Consume **only** a derived `centerpoint_morai_adapter_v1` export
(`ad_morai_dataset_export_centerpoint`). Never train from
`morai_tracking_dataset_v1` directly, from `~/datasets/morai_heven`, or from any
historical research folder. The new data config sets
`EXPECTED_DATASET_VERSION: morai_centerpoint_v1`, so `MoraiHevenDatasetCore`
**raises** if pointed at the `unversioned_step03` overlap dataset — a
loader-level guard, not a doc promise. `morai_heven_dataset.yaml` keeps its
`null` version and is left untouched.

## Contracts asserted by the preflight

| contract | adapter export | v1 config | preflight check |
| --- | --- | --- | --- |
| point features | `[x, y, z, intensity]` (`metadata.point_feature_names`) | `used_feature_list` / `src_feature_list` = same; `NUM_POINT_FEATURES = 4` | exact equality, FAIL otherwise |
| GT box order | `[x, y, z, length, width, height, yaw]` (`metadata.box_fields`) | same | exact, no dim swap |
| GT box origin/frame | geometric centre, `lidar_link`, no z shift | `lidar_link` | `metadata.gt_frame == "lidar_link"`, `point_range_cropping_applied == false`, `box_range_filtering_applied == false` |
| yaw | wrapped `[-π, π]`, CCW +x | same | validator asserts per-box `yaw ∈ [-π, π]` |
| classes | present set ⊆ `{vehicle}` (`split_manifest.class_counts_by_split`) | `CLASS_NAMES: [vehicle]`, `CLASS_NAMES_EACH_HEAD: [[vehicle]]` | present ⊆ config, config ⊆ loader-accepted, `vehicle → id 1` |
| point-cloud range | `[-4, -25, -3, 100, 25, 5]` (reported only; adapter never crops) | `POINT_CLOUD_RANGE` + head `POST_CENTER_LIMIT_RANGE` = same | 3-way equality |
| voxel grid | n/a | `VOXEL_SIZE [0.125, 0.125, 0.2]` | span/voxel is **exactly** integral → grid `[832, 400, 40]`; `grid_x`,`grid_y` % `FEATURE_MAP_STRIDE 8` == 0 → BEV map `[104, 50]`; divisible by `BACKBONE_2D LAYER_STRIDES [1, 2]` |

`NUM_BEV_FEATURES: 256` consistency is only provable by constructing the model
— covered by the optional model smoke below, not by static analysis.

## Split gate (mandatory)

Grouping is the adapter's `(scenario_id, requested_seed)` (fallback `run_id`).
The preflight reads `split_manifest.json` and enforces:

* `train` groups ≥ 1 and `train` frames ≥ 1, else **train empty** blocker;
* **`val` groups ≥ 1**, else `evaluation_ready = false` with the explicit
  reason *"no independent validation group (val groups == 0): NOT evaluation
  ready"*. The preflight never falls back to "evaluate on train";
* `val` groups below `minimum_val_groups + 1` → **warning** only ("small
  validation group count may yield unstable validation metrics"), not a block;
* `train` groups `< minimum_train_groups` (default 2) → warning;
* leakage: the adapter's own `check_split_leakage` (run inside
  `ad_morai_dataset_validate_centerpoint`) must be clean — any group/run
  spanning splits, or a duplicate sample id, is a blocker;
* `test` is reserved: `DATA_SPLIT: {train: train, test: val}` means OpenPCDet's
  `test` mode reads `splits/val.txt`; `splits/test.txt` is structurally
  unreachable through this config during tuning.

## Preflight result states

`final_status` precedence: `BLOCKED_CONFIG` → `BLOCKED_DATASET` →
`BLOCKED_EVALUATOR` → `BLOCKED_ENVIRONMENT` → `READY`.

* `dataset_status` (`READY`/`BLOCKED`) is **structural** readiness (contracts +
  val group + no leakage) and is reported **separately** from
  `real_dataset_available`.
* `real_dataset_available` is `false` unless the operator passes
  `--assert-real-dataset` **and** the path is not under a temp dir. A fixture
  therefore reports `dataset_status: READY` but `final_status: BLOCKED_DATASET`
  — a structurally valid dataset is not training evidence.
* `evaluation_metric_implemented` is `false` today (`evaluation()` raises), so
  `final_status` can never be `READY` yet even with a real val split — a
  second, independent blocker.
* `train_env_ready` is separate again: a valid dataset can be
  `dataset_status: READY` while `env_status: BLOCKED`.

## Environment status (audited)

| item | system `python3` | `~/venvs/heven-centerpoint` |
| --- | --- | --- |
| Python | 3.10.12 | 3.10.12 |
| torch | **missing** | `2.1.2+cu118` |
| CUDA available | n/a | **true** |
| torch CUDA | n/a | 11.8 |
| GPU | n/a | NVIDIA GeForce RTX 4060 |
| spconv.pytorch | missing | available |
| OpenPCDet (`pcdet`) | missing | `0.6.0+233f849`, resolved from `~/projects/OpenPCDet/pcdet` (a separate `-e` checkout at the same pinned commit as the `references/openpcdet` submodule) |
| `environment_ready` | **false** | **true** |

The dataset/config/split gates run under system `python3`; the model smoke
requires the venv. Both interpreters produce a valid `preflight_report.json`.

## Checkpoint audit

| field | value |
| --- | --- |
| candidate | `~/heven_presentation_assets/end_to_end_detector_tracker/checkpoint/centerpoint_t14_reproduction.pth` (via `HEVEN_CENTERPOINT_T14_CKPT` or `--init-checkpoint`) |
| SHA-256 | `466c8181a377682e032bb32579c8ddb65807b5feebd8625a750b1d5538ddbc95` |
| size | 93 474 618 bytes |
| provenance | T‑14 reproduction, epoch 3 / iter 5292, seed 2026, trained on the **overlapping** 1764/0/0 single-scene split |
| architecture | CenterPoint, 4 point features, **3-class** heatmap head (`hm.1.weight [3,64,3,3]`) |
| strict-load compatible with v1 | **no** — v1 is 1-class; strict `state_dict` load fails by construction |
| initialization policy | `train_from_scratch` (v1 default). A future selective backbone-only load is a separate, explicit decision, not part of training prep. |

This checkpoint may be reused as an **initialization artifact** but its
metrics carry **no evaluation validity** (train/eval overlap).

## Preflight loader / model smokes (fixture only)

Run against a factory-built fixture export (6 `(scenario, seed)` groups × 4
frames, `--split-plan` → train 12 / val 4 / test 4). **These prove software
compatibility, not training readiness.**

* **MoraiHevenDatasetCore (torch-free, always run):** positive sample
  `points (N,4)`, `gt_boxes (K,7)`, `coordinate_frame lidar_link`; negative
  (empty-GT) sample loads to `gt_boxes (0,7)` without error; pre-voxel
  `collate_openpcdet_contract` assigns `vehicle → class id 1`; `val` split
  loads.
* **DataProcessor + collate (venv):** `transform_points_to_voxels` produces
  `voxels (M,5,4)` / `voxel_coords (M,3)`; `collate_batch` yields
  `voxels` + `voxel_coords` + `points` + padded `gt_boxes (B,K,8)`.
* **Model construction (venv):** `CenterPoint`, **7 757 225** parameters, on
  `cuda`, from the 1-class config.
* **Forward smoke (venv, `--attempt-forward-smoke`):** one `torch.no_grad()`
  forward, `pred_boxes` shape `[N, 7]` — the only proof that the 3→1 head
  reduction yields correct output shapes. **No `.backward()`, no
  `optimizer.step()`, no metric.** Labelled `shape_smoke_test_only`.

## Experiment identity & fingerprints

* `experiment_id: centerpoint_morai_v1`
* `config_fingerprint` = SHA-256 over the sorted-JSON of the resolved
  `{experiment, data_config, model_config}` (fixture run:
  `b9451d6d6c88bfc90f466d96a87291c4428aaa0b9b7af1b2b3536d9ccf6abe4b`)
* a future training manifest records: `experiment_id`, `config_fingerprint`,
  `dataset content_fingerprint`, `source_dataset_manifest_sha256`,
  `repository_commit`, init-checkpoint SHA-256 (if any), seed, dependency
  versions, GPU info, preflight `final_status`, `training_started: false`
* `seed: 2026`; `determinism_level: seeded` — python/numpy/torch/cuda seeds are
  set, but `cudnn.deterministic` / `torch.use_deterministic_algorithms` are
  **not** and spconv voxelization is not bit-deterministic, so this is **not**
  bit-exact.

## CLI

```
python3 tools/centerpoint_offline/preflight_morai_training.py \
    --dataset  <centerpoint_morai_adapter_v1 export root> \
    --config   tools/centerpoint_offline/configs/centerpoint_morai_v1.yaml \
    --output-dir <a directory OUTSIDE the repository> \
    [--assert-real-dataset] [--init-checkpoint PATH] \
    [--attempt-loader-smoke] [--attempt-model-smoke] [--attempt-forward-smoke]
```

Writes `preflight_report.json` (atomic) to `--output-dir` and prints a human
summary. Refuses an `--output-dir` inside the repository. Exit 0 only when
`final_status == READY` (impossible today); exit 1 otherwise. **It never
trains.**

## Future training command (NOT RUN IN THIS PR)

```
python3 tools/centerpoint_offline/train_morai_centerpoint.py \
    --dataset      <REAL centerpoint_morai_adapter_v1 export> \
    --openpcdet-root references/openpcdet \
    --data-config  tools/centerpoint_offline/configs/centerpoint_morai_v1_dataset.yaml \
    --model-config tools/centerpoint_offline/configs/centerpoint_morai_v1_model.yaml \
    --epochs       <decide from real dataset size; historical smoke used 3> \
    --batch-size   1 \
    --workers      0 \
    --seed         2026 \
    --output-dir   <dir outside the repo>
```

Blocked on: (1) a real non-fixture export with `val` groups ≥ 1, (2) a MORAI
evaluation metric. Batch size, LR scaling, and epoch count are **not** selected
here — that is the next task's job, on the real dataset distribution and GPU.

## Future validation command (NOT RUN — additionally blocked on a metric)

Evaluate **only** `splits/val.txt` (`DATA_SPLIT.test → val`) during tuning;
keep `splits/test.txt` for a single final evaluation. `MoraiHevenDataset.
evaluation()` raises `NotImplementedError` — a MORAI detection metric (3D/BEV
AP, centre error, recall) must be defined first. **No tracking metric (HOTA,
IDF1, IDSW) belongs in detector training preflight** — that is a downstream
integration level.

## Future evaluation levels (documented, not performed)

1. detector validation metrics on an independent MORAI `val` split;
2. tracking replay with the frozen tracker on detector outputs;
3. real VLP-16 deployment / adaptation.

## Historical-comparison guard

Do **not** compare any future CenterPoint-MORAI number to the historical
`0.368 m` (or any T‑14 figure) until: an independent validation split exists,
the metric definition matches, the class/box convention matches, and the
evaluation protocol is comparable. Until then the two are not on the same
axis.

## Simulator → real domain gap

MORAI LiDAR is not VLP-16. Nothing in this task, and nothing a future MORAI
training run produces, is evidence about real-sensor performance. That gap is
unaddressed by design.

## Actual dataset availability

* Real `morai_tracking_dataset_v1` on disk: **NO** (no `dataset_manifest.json`
  anywhere; MORAI/`grpc` absent).
* Real `centerpoint_morai_adapter_v1` export on disk: **NO** (only pytest
  `tmp_path` fixtures).
* `fixture_only = true`, `real_dataset_available = false` →
  `final_status = BLOCKED_DATASET`, unconditionally.

## Why training remains blocked

Two independent blockers, plus a caveat:

1. **No real leakage-safe dataset.** No real adapter export exists; the only
   on-disk MORAI detection data is the historical 1764/0/0 overlap set.
2. **No evaluation metric.** `evaluation()` raises; a validation AP/recall
   number cannot be computed even with a real `val` split.
3. (caveat) The training **environment is ready** (`~/venvs/heven-centerpoint`,
   RTX 4060) — it is the one thing not blocking. That does not lift 1 or 2.

## Tests

`tools/centerpoint_offline/test_centerpoint_morai_training_prep.py` — 20 tests
(system `python3` + `pytest`): data gates (valid non-overlap → `dataset_status
READY`; empty `val` → evaluation blocked; leaked group → blocker; wrong adapter
schema → blocker; incomplete export → blocker; temp path forces fixture),
config gates (valid config passes; wrong class name; extra export class; wrong
point-feature count; bad range; non-integral voxel grid; grid not divisible by
`FEATURE_MAP_STRIDE`; box-range-filter flag; stable fingerprint;
`_BASE_CONFIG_` rejected), environment gates (torch missing → env blocked;
dataset-ready independent of environment; output-dir-inside-repo refused).
Regression: `test_centerpoint_adapter.py` 23, `test_dataset_factory.py` 27,
`tools/morai_dataset_exporter/test_exporter_core.py` 9, `test_centerpoint_
offline.py` 11/12 (`test_training_dataloader_creation` needs `torch` —
pre-existing env gap under system `python3`, unrelated). The venv lacks
`pytest`, so the new pytest module is exercised under system `python3`; the
model/forward smoke is verified by direct `preflight_morai_training.py` CLI
runs under the venv.

## Files

`tools/centerpoint_offline/preflight_morai_training.py`,
`tools/centerpoint_offline/configs/{centerpoint_morai_v1.yaml,
centerpoint_morai_v1_dataset.yaml, centerpoint_morai_v1_model.yaml}`,
`tools/centerpoint_offline/test_centerpoint_morai_training_prep.py`,
this doc, `docs/agent/STATUS.md`. The existing `morai_heven_dataset.yaml` /
`morai_centerpoint_smoke.yaml` / `morai_centerpoint_train.yaml` and
`train_morai_centerpoint.py` defaults are unchanged.
