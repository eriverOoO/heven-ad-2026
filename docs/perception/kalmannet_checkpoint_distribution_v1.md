# KalmanNet checkpoint distribution v1

Branch `feat/kalmannet-checkpoint-distribution-v1`, from `origin/main`
`5460027a` (merge of PR #61). **Checkpoint distribution / documentation
only. No KalmanNet architecture, training, or evaluation code changed.
No new research claim -- see "No MORAI claim" below.**

## What this is

`scripts/run_camera_lidar_tracking.sh` (mode 7: Euclidean detector +
KalmanNet estimator) has always expected a checkpoint at
`models/experimental/dense_kalmannet_v2.pt` with a hardcoded expected
SHA-256 -- but that file was never committed, so mode 7 failed with
"file not found" on any fresh clone. This task closes that gap: the
checkpoint the script already expected is now committed, plus the
AV2-pretrained family this project has repeatedly recommended as the
preferred candidate for further testing.

## Checkpoints provided (`models/experimental/`)

| file | family | recommended | size | SHA-256 |
|---|---|---|---|---|
| `dense_kalmannet_v2.pt` | historical MORAI-domain | existing script default | 31,436 B | `956604975e...fb7d48` |
| `kalmannet_av2_natural_10k_generic_robust_seed1.pt` | AV2-pretrained, NATURAL 10k GENERIC-ROBUST | **yes -- override `--kalmannet-checkpoint`** | 34,216 B | `a9a19353...98dc5ee5` |
| `kalmannet_av2_natural_10k_generic_robust_seed0.pt` | same family, alt seed | no (interchangeable) | 33,896 B | `30e0902e...73a778` |
| `kalmannet_av2_natural_10k_generic_robust_seed2.pt` | same family, alt seed | no (interchangeable) | 34,216 B | `a005efc5...8f3030c0bfe91e51` |

Full provenance (training dataset, split hashes, epoch, seed, evaluation
result, known limitations) for every file: `models/experimental/manifest.yaml`.

**Not provided here:** `centerpoint_t14_reproduction.pth`. CenterPoint
checkpoint distribution is handled on its own feature/experiment branch
and is never added to `main`, per this project's explicit branch policy.

## Why two different KalmanNet families

- **`dense_kalmannet_v2.pt`** was trained directly on a small, single-scene
  real MORAI GT capture (T-9A/T-12 series). It is the checkpoint the
  existing tracking-preset script already expected, and is already
  ROS-integration-proven (T-9B: loaded live through the production
  `ab3dmot_core.KalmanNetEstimator`). On its own (small) MORAI TEST split
  it **ties**, not beats, a classical Tuned Linear KF (1.467 m vs
  1.456 m position RMSE) -- not a demonstrated improvement.
- **`kalmannet_av2_natural_10k_generic_robust_seed1.pt`** was pretrained
  on 10,000 Argoverse 2 Motion Forecasting scenarios (public US driving
  data, not MORAI). Across three separate downstream ablations run
  against it (motion-composition sampling, physically-consistent
  variable-dt augmentation, and a fixed/thinned mixed-curriculum
  screening), this exact NATURAL 10k GENERIC-ROBUST family was
  **repeatedly reaffirmed as the preferred AV2-pretrained candidate**
  over every tested alternative. On AV2 held-out data it clearly beats
  both a tuned Linear KF and `dense_kalmannet_v2.pt` (evaluated
  cross-domain).

Neither is "the" answer for MORAI -- see the limitations section below.

## Why committed directly, not Git LFS or a GitHub Release

The repo already uses Git LFS for other large binaries (`.gitattributes`:
`ad_data/**`, `*.pcd`, etc.), and `ad_camera_perception/models/README.md`
documents a deliberate "do not commit weights directly" policy for the
~20 MB third-party YOLO COCO checkpoint used there. Neither condition
applies to these four files:

- They are **this project's own in-house-trained artifacts**, never a
  redistributed third-party weight -- no license question.
- They are **tiny** (31-34 KB; `KalmanNetGRU(hidden_size=32)`, 7,016
  trainable parameters each) -- no repo-bloat concern that would justify
  LFS or external hosting.
- `git-lfs` was found to be non-functional in the environment this task
  was done in (`git lfs version` fails), so LFS would also add a real
  setup dependency for zero benefit at this file size.

A `.gitignore` exception (`/models/*` + explicit `!` re-inclusions for
these five paths) makes this a deliberate, minimal carve-out rather than
a blanket change to the repo's binary-file policy -- `centerpoint_t14_reproduction.pth`
is *not* re-included by that exception and stays gitignored on `main`.

## Verification performed (this task)

Both checkpoint families were loaded through the **existing, unmodified**
production loader `ab3dmot_core.load_kalmannet_network()` (which infers
`hidden_size` from the checkpoint's own `output_fc.weight` shape and
validates the `STATE_DIM*MEAS_DIM` output dimension before constructing
`KalmanNetGRU` -- never trusts an external claim) and put through one
real `KalmanNetFilter.init_sequence()` + `.step()` smoke inference call:

```
DENSE_KALMANNET_v2   {'hidden_size': 32, 'n_trainable_params': 7016, 'checkpoint_sha256': '956604975e5204b2c584c3e8fe16a7ba9346097980566ecbbb22c2001fbf7d48'}
  smoke inference OK, x_post finite: [[0.4790, -0.0874, 3.5545, 2.2418]]
NATURAL_10k_seed0    {'hidden_size': 32, 'n_trainable_params': 7016, 'checkpoint_sha256': '30e0902ef91d2e8f189db163a2ec636d39a7ad9d4f5abc414b8aaa8ae873a778'}
  smoke inference OK, x_post finite: [[0.8806, 0.8112, 2.3998, 2.2249]]
NATURAL_10k_seed1    {'hidden_size': 32, 'n_trainable_params': 7016, 'checkpoint_sha256': 'a9a19353ddb272d3fc0ac3ad7c6754239ef3d8dd00acd3c52a59193d98dc5ee5'}
  smoke inference OK, x_post finite: [[0.8829, 0.8353, 2.6160, 2.5119]]
NATURAL_10k_seed2    {'hidden_size': 32, 'n_trainable_params': 7016, 'checkpoint_sha256': 'a005efc54b63ad92f4972a2731bc4ff1e5e1d9cc7c45e7e18f3030c0bfe91e51'}
  smoke inference OK, x_post finite: [[0.9430, 0.9392, 2.6411, 2.4895]]
```

All four SHA-256 values match exactly what every prior task in this
project already recorded for these files -- confirms byte-identical, not
a re-derived or re-trained artifact.

### Tests run this task

- `scripts/tests/test_camera_lidar_tracking_runner.py`: **6/6 pass**
  (asserts on the shell script's own source text -- unaffected by
  populating the checkpoint files, as designed).
- `ad_lidar_perception/test/test_kalmannet_core.py`: **16/16 pass**
  (via the torch-enabled `heven-centerpoint` venv). Unmodified from every
  prior KalmanNet task -- confirms this distribution task did not touch
  the architecture.
- `ad_lidar_perception/test/test_ab3dmot_kalmannet.py`: **19/19 pass**
  (checkpoint loading, config validation, multi-track hidden-state
  isolation, lifecycle). Unmodified.
- `py_compile`/`git diff --check`: clean.

## Third-party model note (out of scope, documented not redistributed)

`scripts/run_camera_lidar_tracking.sh` modes 8-10 additionally reference
Autoware's own official pretrained `autoware_lidar_centerpoint` model
(ONNX + TensorRT engine files under `$AD_DATA_DIR/models/autoware/
lidar_centerpoint/`, gated behind an explicit
`AD_AUTOWARE_MODEL_LICENSE_REVIEWED=1` environment variable the tooling
already enforces). Those `.engine` files are TensorRT-compiled for one
specific GPU/driver/TensorRT version and are **not portable** across
machines regardless of redistribution rights; the `.onnx` files are a
third-party (Autoware Foundation) artifact this task does **not**
redistribute. Obtain them via `autoware_lidar_centerpoint` 0.51.0's own
standard install/setup flow (see `docs/perception/current_pipeline_contract.md`
section 6), not from this repository.

## No MORAI claim

Committing these checkpoints changes what is *runnable*, not what has
been *validated*. This document does not add, and explicitly disclaims,
any of the following:

- "KalmanNet is validated on MORAI" -- not supported. The only
  independent MORAI GT evaluation of the AV2-pretrained family is n=3
  test sequences from one scene (`docs/perception/morai_estimator_eval_v2.md`).
- "The AV2-pretrained checkpoint is optimal for MORAI" -- not supported;
  it is the best AV2-*internal* candidate tested so far.
- Any conclusion contradicting `docs/perception/morai_estimator_eval_v2.md`,
  `kalmannet_av2_variable_dt_v1.md`, or any other frozen research doc in
  this repository.

## Running it

```bash
git clone https://github.com/eriverOoO/heven-ad-2026.git
cd heven-ad-2026
# checkpoints are already present -- verify before use:
sha256sum models/experimental/*.pt
# (compare against models/experimental/manifest.yaml)

./scripts/run_camera_lidar_tracking.sh --list-modes
./scripts/run_camera_lidar_tracking.sh --mode 7 --bag /absolute/path/to/bag
# or, to use the recommended AV2-pretrained checkpoint instead of the default:
./scripts/run_camera_lidar_tracking.sh --mode 7 --bag /absolute/path/to/bag \
  --kalmannet-checkpoint "$(pwd)/models/experimental/kalmannet_av2_natural_10k_generic_robust_seed1.pt"
```

No checkpoint path needs to be edited in source anywhere -- `--kalmannet-checkpoint`
/ `$HEVEN_KALMANNET_CHECKPOINT` are the only inputs, and the script's
default already resolves to a file that now exists in the clone.

## Limitations

- No new MORAI evaluation was run as part of this task (checkpoint
  distribution only).
- CenterPoint (modes 8-10) still requires a manual checkpoint copy on
  `main` by design (see branch policy above).
- The Autoware `lidar_centerpoint` third-party model (also modes 8-10)
  is documented, not redistributed.

## Files

`models/experimental/{manifest.yaml,README.md,dense_kalmannet_v2.pt,
kalmannet_av2_natural_10k_generic_robust_seed{0,1,2}.pt}` (new),
`.gitignore` (additive negation for exactly these 5 new paths),
`README.md`, `docs/morai/lidar-mcap-replay.md` (checkpoint-availability
wording updated), this file. No KalmanNet architecture, training script,
evaluation script, launch file, or production default changed.
