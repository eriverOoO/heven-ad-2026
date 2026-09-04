# Reproducible KalmanNet Training Entry Point v1 + AV2 Stage-0 Training Sanity Experiment

Status: **Stage-0 sanity only. No claim that AV2 pretraining improves
MORAI performance.** No runtime/ROS/AB3DMOT integration in this task —
this produces an offline checkpoint only. Depends on the merged AV2
Stage-0 sharded dataset (PR #45/#46,
`docs/perception/av2_kalmannet_adapter_v1.md`).

## Trainer entry point

`tools/kalmannet_training/`:

- `av2_split.py` — deterministic scenario-level train/val/test split.
- `kalmannet_sequences.py` — glue between the existing
  `av2_kalmannet_shard_loader.py` and the training loop. **Does not
  implement a second AV2 dataset loader** — every sequence comes from
  `SegmentArrays.to_kalmannet_sequence()`, unchanged.
- `trainer_core.py` — the ported historical training loop (see below).
- `checkpoint_utils.py` — versioned checkpoint + reproducibility manifest.
- `train_kalmannet.py` — training CLI.
- `evaluate_kalmannet.py` — held-out evaluation CLI (KNet vs. tuned
  LinearCVKF vs., if available, DENSE-KALMANNET-v2 as a cross-domain
  diagnostic).

Uses the **existing, unmodified** `KalmanNetGRU` / `KalmanNetFilter` /
`LinearCVKF` / `STATE_DIM` / `MEAS_DIM` from
`ad_lidar_perception/ad_lidar_perception/kalmannet_core.py`. No
architecture, analytical model, or AB3DMOT/CenterPoint/planner/
prediction/occupancy file is touched by this task.

## Historical training logic ported

Forensically ported from the real, still-present, never-committed
scripts that actually produced the frozen DENSE-KALMANNET-v2 checkpoint
(`~/heven_presentation_assets/kalmannet_training_stability/
instrumented_train.py` + `phase13_ten_seed.py`, inspected fresh from
disk for this task, not recalled from memory) — see `trainer_core.py`'s
own module docstring for the complete preserved/left-out breakdown.
Preserved: per-sequence `KalmanNetFilter` recursion (`batch_size=1`),
real variable `dt`, `z is None` → analytical predict-only step (no gain
call, hidden state untouched), Adam optimizer with the frozen dense-v2
hyperparameters, gradient clipping, validation-only best-checkpoint
selection with patience-based early stopping, decoupled init/order
seeds, catastrophic-run detection. Left out (documented why in
`trainer_core.py`): the exhaustive per-frame forensic diagnostic capture
T-12.3 used for its own gradient-explosion investigation (kept only a
per-epoch gradient-norm summary + non-finite/divergence counters,
sufficient for this task's own "no NaN/divergence" success criterion).

### Loss policy — audited precisely, not assumed

**Real finding, verified directly from the historical source code, not
memory:** `instrumented_train.py::run_sequence_instrumented` appends a
loss term whenever `dt is not None` — i.e. on every frame except the
very first, **regardless of whether that frame's own measurement was
present or missing**. This is option **B** in this task's own framing
("loss on ALL GT-valid timesteps, including prediction-only
missing-measurement steps"), **not** option A ("loss only on
measurement-update steps") — the task's own premise that history used
option A did not match the actual source. `loss_on_predict_only=True`
(the trainer's default) reproduces this exactly; `--no-loss-on-predict-only`
exposes the narrower option A as an explicit, non-default alternative,
per this task's own instruction to expose it as a config option without
silently changing the historical default.

## Stage-0 dataset

The already-downloaded/exported 120-scenario AV2 `train`-split Stage-0
sharded dataset (`~/datasets/av2/processed/av2_kalmannet_v1_sharded/`,
7,358 segments, `first_state_relative` coordinates — unchanged from
PR #46).

### A real, blocking bug found and fixed in this task

While building the trainer, a real sequence's `z_meas[0]` was observed
at `[721.16, 936.14]` instead of the expected `[0, 0]`. Root cause,
confirmed directly against a real shard: `apply_coordinate_transform`
(merged in PR #45/#46) transformed `gt_state` into first-state-relative
coordinates but **left `clean_position` in absolute, untransformed
coordinates** — so the exported `measurement`/`measurement_data` (built
from `clean_position`) silently ended up in a **different coordinate
frame** than `gt_state`/`x_true`. Confirmed on real data:
`state_data[0] == [0, 0, ...]` but `measurement_data[0] == origin_xy`,
not `[0, 0]`. The prior self-consistency test
(`measurement == clean_position`) could not catch this — both sides
shared the identical bug, so they were internally consistent with each
other while both being wrong relative to `gt_state`. **Fixed**:
`apply_coordinate_transform` now transforms `clean_position` by the
identical offset as `gt_state`. A new regression test
(`test_coordinate_transform_keeps_clean_position_and_gt_state_in_the_same_frame`)
checks against `gt_state[:, :2]` directly, not just internal
self-consistency, so this class of bug cannot silently reappear. The
Stage-0 shard dataset was **re-exported** after the fix (same 120
scenarios, same 7,358 segments, same content otherwise — only the
`measurement_data`/`clean_position` coordinate frame changed).

## Split

Scenario-level, never segment/track-level (`av2_split.py`):
`SplitConfig(train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=20260905)`.
Sorted-then-seeded-shuffle, deterministic regardless of input order
(confirmed by test — sorted/reversed input give identical assignment).
For the real 120-scenario Stage-0 set: **96 train / 12 val / 12 test**
scenarios (exactly 80/10/10). The split-*generation logic* (this module)
is committed; the *generated* manifest for this specific local download
(`~/datasets/av2/manifests/stage0_kalmannet_split.json`) is machine-local,
not committed — fully reproducible from the committed scenario manifest +
this module + the recorded seed. **Not** related to AV2's own official
`test` split — this is an internal split of the 120 scenarios already
pulled from AV2's public `train` split; AV2's own withheld test
answer-key files are never read anywhere in this project.

## Corruption conditions

Two Stage-0 conditions, both reusing the existing, unmodified
`CorruptionConfig`/`apply_corruption` (load-time transform, per PR #46 —
no new corruption types added):

- **A. CLEAN** — `CorruptionConfig(mode="none")`.
- **B. GENERIC-ROBUST** — `gaussian_noise_std_m=0.3`, `dropout_prob=0.1`,
  `dropout_burst_enabled=True` (`prob=0.02`, `min_len=2`, `max_len=5`),
  `seed=20260905`. **Explicitly not claimed MORAI-realistic** — a
  generic robustness sanity condition only, to verify the trainer
  supports deterministic corruption end-to-end, not a calibrated noise
  model.

## Baselines

Two distinct `LinearCVKF` baselines are reported, never conflated:

**1. `linear_kf_transferred_from_morai`** — `sigma_a=5.0`,
`r_std=0.2254682076528578`, `p0_scale=10.0`, the exact T-9A.1
validation-only-selected values
(`~/heven_presentation_assets/state_estimator_gt_comparison/
selected_kf_config.json`), **transferred to AV2 without retuning** —
never fit against any AV2 data, train, val, or test. Verified by a
dedicated test that these are literal module constants, not the output
of a fitting routine. This is a **cross-domain transferred** baseline,
not the fair AV2-native comparison point.

**2. `linear_kf_av2_tuned`** (added per an explicit mid-task request for
a fair classical baseline) — `tools/kalmannet_training/calibrate_kf.py`
measures an isotropic `sigma_z` directly from real AV2 TRAIN measurement
residuals, then does a bounded log-grid search over `sigma_a` (8 values)
x an `r_scale` factor on `sigma_z` (5 values, 40 candidates total),
selected purely by **VALIDATION** position RMSE. The held-out **TEST**
split is structurally unreachable by `calibrate_kf_on_av2()` — its own
function signature has no `test_seqs` parameter. Calibrated once on the
GENERIC-ROBUST condition's TRAIN+VAL (the only condition with real,
non-degenerate measurement noise to fit against — CLEAN's residuals are
exactly zero by construction), then evaluated on every held-out stream
(including CLEAN, where it is expected to be a *worse* fit, since it is
tuned for noise that isn't present there — see Results). Selected:
`sigma_a=5.0`, `r_std=0.3003765726729657` (= measured train
`sigma_z`, i.e. `r_scale=1.0`), `p0_scale=10.0`,
`val_position_rmse=0.3410` (40 candidates).

**DENSE-KALMANNET-v2 cross-domain diagnostic**: evaluated on the same
AV2 held-out stream since
`~/heven_presentation_assets/kalmannet_training_stability/checkpoint/
dense_kalmannet_v2.pt` is present locally on this machine — explicitly
labeled a cross-domain diagnostic only. **A MORAI-trained checkpoint
evaluated on AV2 data proves nothing about simulator superiority in
either direction** and is never spun as such.

## Reproducibility

Every checkpoint's `.manifest.json` records: git SHA, torch/CUDA
versions, GPU name (or `null` on CPU), model config, training config
(lr/grad_clip/epochs/patience/batch_size/optimizer/condition), seed info
(seed/init_seed/order_seed), corruption config, dataset + split manifest
SHA-256, coordinate mode, best epoch/val loss, and the checkpoint's own
SHA-256. Checkpoint family: `av2_stage0_kalmannet_v1` — a **distinct**
family from the historical `dense_kalmannet_v2`; never overwrites it.
Checkpoints are saved in the same wrapped `{"state_dict": ..., **manifest}`
shape `ab3dmot_core.load_kalmannet_network` already knows how to read, so
this family stays structurally loadable by the existing runtime loader
in principle — **this task does not wire it in** (see "No runtime
integration" below).

## Device / environment note

A real environment issue was found and fixed on this development
machine: the `av2`-package venv's resolved `torch` (2.14.0+cu130)
required a newer CUDA driver than the installed one (RTX 4060, driver
560.94, supports up to CUDA 12.6) — `torch.cuda.is_available()` silently
returned `False` despite a real GPU being present. Reinstalled
`torch==2.4.1+cu121` (matches the installed driver) into the same venv;
CUDA became available (`torch.cuda.get_device_name(0) ==
"NVIDIA GeForce RTX 4060"`).

### Formal device audit (performed after the CLEAN run, before GENERIC-ROBUST)

The CLEAN run itself was launched with an explicit `--device cpu` (a
deliberate choice, not an unavailable-CUDA fallback). Before starting
GENERIC-ROBUST, this was audited formally, item by item, in the exact
venv the trainer uses:

1. **`--device` value used for CLEAN**: `cpu`, explicit on the command line.
2. **`torch.cuda.is_available()`** in the same venv: `True` (re-confirmed
   independently of the CLEAN run).
3–5. **Device consistency, traced through one live training step on both
   `cpu` and `cuda`**: model parameters, `KalmanNetFilter.x_post`, the
   measurement tensor, and the GRU hidden state (`net.h`) all land on
   whichever device is requested — verified directly (`next(net.parameters()).device`,
   `kf.x_post.device`, the measurement tensor's `.device`, `net.h.device`
   all printed and matched the requested device on both runs).
6. **No silent CPU fallback found**: `kalmannet_core.py`'s
   `KalmanNetFilter.__init__(self, net, device="cpu")` has a *default*
   parameter of `"cpu"`, but every call site in `trainer_core.py`
   (`run_sequence`, every `torch.tensor(..., device=device)`) passes
   `device=` explicitly — the default is never silently triggered.
7. **`--device cuda` genuinely works end-to-end** — confirmed by the
   throughput comparison below, which ran the identical code path on
   `cuda` successfully.

**Small, fixed-subset throughput comparison** (the same 200 real Stage-0
train sequences, 10,303 total frames, identical order, both runs seeded
identically, 3-sequence warmup excluded from timing):

| device | seq/s | frames/s | elapsed (200 seqs) | mean loss after 200 steps | all finite |
|---|---|---|---|---|---|
| CPU | 25.73 | 1,325.6 | 7.77 s | 7.257158 | yes |
| CUDA | 5.68 | 292.7 | 35.19 s | 7.257171 | yes |

**CPU is ~4.5× faster than CUDA** on this exact workload (identical
subset, identical seed) — numerically sane on both (the two mean-loss
values agree to 4 significant figures; the tiny residual difference is
ordinary CPU/GPU floating-point non-associativity, not a bug). This
matches this project's own prior documented finding ("CUDA slower than
CPU at this tiny per-step scale (kernel-launch overhead dominates)",
T-9A), now reproduced with a controlled, apples-to-apples measurement
rather than an informal one. **Decision: CPU is used for GENERIC-ROBUST
and all subsequent Stage-0 training in this task** — CUDA is not
materially faster here because `batch_size=1`/Python-loop overhead
dominates, exactly the condition under which this task's own instruction
says to continue with the faster/simpler device rather than force GPU
usage.

GPU batching (processing multiple independent sequences per step, which
`KalmanNetGRU.forward()` already structurally supports via its `batch`
dimension, per the earlier read-only audit) is exactly the Stage-1
throughput work this finding motivates — not attempted here, per this
task's own explicit "do not redesign batching" instruction.

## Results — Stage-0 sanity experiment (seed 0, `--max-epochs 10 --patience 4`)

**Deviation from historical hyperparameters, recorded explicitly:**
`max_epochs=10` / `patience=4` (not the historical `60`/`10`) — a
deliberate Stage-0 sanity reduction given the observed convergence shape
(both runs' train loss was still decreasing but val loss had already
plateaued/started re-rising by epoch 6-9); flagged in every checkpoint
manifest via `training_config.deviation_from_historical_max_epochs: true`.
`hidden_size=32`, `lr=0.001`, `grad_clip=10.0` all kept at the historical
values, unchanged. Both runs: `hidden_size=32`, `batch_size=1`, `Adam`,
`device=cpu` (see the device audit above), seed 0 for model init + shuffle
order.

### CLEAN training (Run 1)

`n_train=5,769 n_val=865` (test held out: not touched during training).
10/10 epochs, best epoch **9**, best val loss **0.5693246408256769**,
`nonfinite_step_count=0`, not catastrophic, `train_time_s=2244.34`.
Checkpoint: `av2_stage0_clean_seed0.pt`
(SHA-256 `88277d45fd69985e21105511d7068f8d08bd2ca854ae6e68f85f17722a39a107`).

### GENERIC-ROBUST training (Run 2)

`n_train=5,768 n_val=865` (one fewer train sequence than CLEAN: a single
sequence's dropout corruption removed its own frame-0 measurement, so it
was excluded by the existing "sequences must have a first measurement"
precondition — expected, not a bug). 10/10 epochs, best epoch **6**, best
val loss **0.7202208805269417**, `nonfinite_step_count=0`, not
catastrophic, `train_time_s=1951.18`.
Checkpoint: `av2_stage0_generic_robust_seed0.pt`
(SHA-256 `39cb6ac71495f52a86c617fd3118a77d2f16a3a265f1b462d306fe99ee0789b5`).

Per-epoch train/val loss for both runs (full CSV history committed
nowhere — machine-local — but summarized here):

| epoch | CLEAN train | CLEAN val | GENERIC-ROBUST train | GENERIC-ROBUST val |
|---|---|---|---|---|
| 0 | 0.5458 | 0.6340 | 0.8694 | 0.7878 |
| 1 | 0.4380 | 0.6260 | 0.7702 | 0.7402 |
| 2 | 0.4307 | 0.5938 | 0.7588 | 0.7463 |
| 3 | 0.4247 | 0.6052 | 0.7487 | 0.7740 |
| 4 | 0.4174 | 0.5971 | 0.7416 | 0.7517 |
| 5 | 0.4164 | 0.6154 | 0.7398 | 0.7235 |
| 6 | 0.4135 | 0.5793 | 0.7306 | **0.7202** (best) |
| 7 | 0.4175 | 0.6059 | 0.7267 | 0.7410 |
| 8 | 0.4090 | 0.5722 | 0.7165 | 0.7360 |
| 9 | 0.4057 | **0.5693** (best) | 0.7133 | 0.7680 |

CLEAN's val loss is noisier than a smooth monotone decrease (it dips at
epochs 2, 6, and finally 8-9) despite zero measurement noise/dropout —
consistent with `batch_size=1`/small-hidden-size Adam training rather
than a sign of a problem (0 nonfinite steps, no catastrophic-run flag,
grad norms bounded throughout including the one early elevated
epoch-0 mean of 20.2, which settles by epoch 1).

### Held-out evaluation — GENERIC-ROBUST checkpoint (conditions A/B/C)

`evaluate_kalmannet.py --training-condition generic_robust` (n=724
held-out TEST sequences per condition; overall = `matched`+`missing`
combined, `gap::*` buckets only populated when corruption produces
missing frames):

**Overall position RMSE (m), lower is better:**

| condition | KNet (this task) | KF transferred (MORAI) | KF AV2-tuned | Dense-v2 (cross-domain) |
|---|---|---|---|---|
| A. CLEAN held-out | **0.0347** | 0.0563 | 0.0717 | 0.0456 |
| B. same corruption family | **0.2935** | 0.3415 | 0.3278 | 0.4377 |
| C. different corruption seed | **0.2903** | 0.3372 | 0.3246 | 0.4261 |

**Overall velocity RMSE (m/s):**

| condition | KNet | KF transferred | KF AV2-tuned | Dense-v2 |
|---|---|---|---|---|
| A | **0.7558** | 0.9017 | 0.8977 | 0.8680 |
| B | **0.9731** | 1.3585 | 1.2319 | 1.5975 |
| C | **0.9531** | 1.3436 | 1.2186 | 1.5376 |

**Interpretation (Stage-0 sanity only, single seed, no MORAI claim):**
this task's own GENERIC-ROBUST-trained KNet beats both KF baselines and
the cross-domain Dense-v2 diagnostic on every condition and both metrics
— expected, since it is the only estimator actually trained on this
exact AV2 GENERIC-ROBUST distribution. The two KF baselines cleanly
separate as designed: `linear_kf_transferred_from_morai` wins on
**A** (CLEAN — it is tuned for MORAI's own much smaller noise, closer to
zero, so it is the better fit when AV2 noise is also absent), while
`linear_kf_av2_tuned` wins on **B/C** (the noisy conditions it was
calibrated for) — a coherent, non-cherry-picked result confirming the
fair/transferred distinction is doing real work, not just bookkeeping.
Dense-v2 (a MORAI-domain checkpoint, never trained on AV2 at all) is
competitive with KNet on CLEAN (0.0456 vs 0.0347) but clearly worse under
both noisy conditions (0.44/0.43 vs 0.29/0.29) — consistent with it
never having seen this corruption distribution, not a claim about
MORAI-vs-AV2 domain superiority in either direction.

### Missing-measurement / gap-length metrics (condition B, GENERIC-ROBUST checkpoint)

| bucket | n | KNet pos RMSE | KF transferred pos RMSE | KF AV2-tuned pos RMSE |
|---|---|---|---|---|
| matched (no gap) | 32,979 | **0.2659** | 0.3006 | 0.2917 |
| missing (any gap) | 6,202 | **0.4101** | 0.5063 | 0.4755 |
| gap 1-2 frames | 4,927 | **0.3502** | 0.4206 | 0.3989 |
| gap 3-5 frames | 1,215 | **0.5867** | 0.7392 | 0.6883 |
| gap 6+ frames | 60 | **0.5920** | 0.9492 | 0.8412 |

KNet's advantage widens with gap length (matched: KNet 12% better than
transferred-KF; gap 6+: KNet 38% better) — consistent with the learned
gain adapting its trust in the prediction-only step, though this is a
single-seed Stage-0 observation, not a robustness claim across seeds.

### Class-stratified metrics (condition A, CLEAN held-out; GENERIC-ROBUST checkpoint)

| class | n | KNet pos RMSE | KF transferred pos RMSE |
|---|---|---|---|
| VEHICLE | 26,596 | 0.0384 | 0.0618 |
| PEDESTRIAN | 6,530 | 0.0208 | 0.0317 |
| OTHER | 6,213 | 0.0297 | 0.0513 |

Pedestrian trajectories have the lowest absolute error for every
estimator (smaller, slower motion, easier to fit under a CV model);
VEHICLE the highest (fastest, most maneuvering) — expected pattern, not
surprising.

### Divergence / non-finite counts

**0** divergences and **0** non-finite predictions across every
condition, every estimator, every bucket, in both the CLEAN-checkpoint
and GENERIC-ROBUST-checkpoint evaluations (`divergence_count`/
`nonfinite_count` fields, all zero) — the trainer/evaluator pipeline is
numerically stable end-to-end on real AV2 data.

## Training throughput / Stage-1 sizing

At-scale combined train+val throughput, measured from steady-state
per-epoch wall-clock deltas (epochs 1-9, excluding the epoch-0 warmup) on
CPU: **GENERIC-ROBUST ≈34.1 sequences/s** (6,633 seq / ≈194.3 s/epoch)
vs. **CLEAN ≈29.6 sequences/s** (6,634 seq / ≈224.2 s/epoch) — a real,
reproducible ~15% gap, not noise (every individual epoch in each run
falls in a tight band: GENERIC-ROBUST 192-197s, CLEAN 221-227s).
Plausible cause, consistent with `trainer_core.py`'s ported design: a
missing-measurement frame takes the cheaper analytical predict-only path
(no GRU forward/gain call), so GENERIC-ROBUST's ~15-20% dropped frames
skip the network entirely, more than offsetting the extra corruption
computation. AV2 Stage-0's 120 scenarios yielded 7,358 segments
(**≈61.3 segments/scenario**). Extrapolating linearly (same
segments/scenario ratio, same ~90% train+val fraction), using the
**slower CLEAN rate as the conservative estimate**:

| Stage-1 scenario count | extrapolated train+val sequences | extrapolated epoch time | 10 epochs, one condition |
|---|---|---|---|
| 5,000 (42x Stage-0) | ≈276,000 | ≈2.6 h | ≈26 h |
| 10,000 (83x Stage-0) | ≈552,000 | ≈5.2 h | ≈52 h (≈2.2 days) |
| 20,000 (167x Stage-0) | ≈1,104,000 | ≈10.4 h | ≈104 h (≈4.3 days) |

Two conditions (CLEAN + GENERIC-ROBUST) run sequentially roughly double
each figure (partially offset by GENERIC-ROBUST's own faster rate). This
is a **linear extrapolation from one measured data point per condition**,
not a benchmark sweep — reported as an order-of-magnitude estimate, not
a precise forecast.

**Recommendation: this task's own throughput measurement moderately
argues against the task's stated preference for 10,000 scenarios.**
10,000 scenarios extrapolates to ≈2.2 days of unattended CPU wall-clock
per condition per seed (≈4+ days for both conditions), which is feasible
as a background job but leaves little room for the "up to 3 seeds if
unstable" contingency or any hyperparameter follow-up within a
reasonable iteration cycle. **5,000 scenarios** (≈26 h/condition, ≈52 h
both conditions sequentially) is recommended instead as the more
practical Stage-1 size at the *current* CPU/`batch_size=1` throughput —
unless GPU batching (multiple independent sequences per step,
structurally already supported by `KalmanNetGRU.forward()`'s batch
dimension per the earlier read-only audit, not implemented in this task)
is built first, which would very plausibly make 10,000 or more
practical. This recommendation only; **no Stage-1 data was downloaded or
processed in this task.**

## Test results

`tools/kalmannet_training/` — 37 tests across 5 files, all pass:
`test_av2_split.py`, `test_kalmannet_sequences.py`, `test_trainer_core.py`
(9), `test_checkpoint_utils.py`, `test_calibrate_kf.py` (6, added for the
fair-AV2-tuned-KF baseline). `ad_morai_bridge_dev` regression (the
coordinate-transform bug fix): 161 pass, including the new
`test_coordinate_transform_keeps_clean_position_and_gt_state_in_the_same_frame`
regression test.

## No runtime integration

This task produces **offline checkpoints only**. `kalmannet_checkpoint`'s
default, `state_estimator`'s default, and every production launch file
are untouched. No AV2-derived checkpoint is wired into ROS/AB3DMOT.
Runtime integration is explicitly deferred to a future task, after a
meaningful Stage-1/Stage-2 result.

## No claim of MORAI improvement

Nothing in this task claims AV2 pretraining improves, or will improve,
MORAI/HEVEN tracking performance. The held-out metrics in this task are
an AV2-internal sanity check (does the trainer converge, is it
reproducible, does it not diverge) — not a MORAI evaluation, and not a
generalization claim.
