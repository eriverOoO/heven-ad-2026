# AV2 KalmanNet Full-Scale 10K Generic-Robust Pretraining v1

Status: **COMPLETE.** Trains ONE full-scale KalmanNet model (GENERIC-ROBUST
corruption, one seed) on the frozen fresh AV2 Scale-Up v2 10k dataset
(PR #54), evaluates it against the untouched 1,000-scenario official AV2
VAL set, and compares it against the Stage-1 pilot's 2k GENERIC-ROBUST
checkpoint on the identical VAL stream. **No MORAI evaluation is
available in this environment. This document makes no claim about MORAI
or competition performance anywhere -- AV2 official-VAL results only.**

Branch `exp/kalmannet-av2-10k-generic-v1`, from merged PR #54
(`data/av2-kalmannet-scaleup-v2`, merge commit `ba35b51`, verified via
`gh pr view 54` before branching). Built in a clean temporary worktree
(`/tmp/heven-worktrees/kalmannet-av2-10k-generic-v1`) because the primary
checkout carries ~1,900 pre-existing CRLF-only user modifications
(documented pattern from prior sessions' `docs/agent/STATUS.md`) -- per
this task's own instruction, those were left untouched.

## Goal

Does scaling AV2 motion pretraining from 2,000 -> 10,000 fresh scenarios
meaningfully improve AV2 estimator robustness?

**Headline answer: partially -- CASE B/C mixed.** The 10k model is
clearly better on CLEAN measurements (~37% lower position RMSE than the
2k model) but only marginally better under the GENERIC-ROBUST corruption
both models were trained for (~2% lower position RMSE). Full data below.

## 1. Dataset (frozen, unmodified)

Reused exactly as frozen by PR #54 -- no scenario added/removed, no
class-balancing, no speed-filtering, no coordinate-representation change,
no re-selection:

| | scenarios | segments | GT samples |
|---|---|---|---|
| TRAIN-source pool | 10,000 | 553,898 (553,810 `valid_for_kalmannet_gt`) | 28,572,250 |
| official AV2 VAL | 1,000 | 56,521 | 2,918,906 |

Known, unmodified dataset properties this task does not attempt to fix
(per its own explicit instruction): ~64% near-stationary tracks (speed
< 0.5 m/s), `dt` exactly 0.100 s at every percentile.

## 2. Internal TRAIN/VALIDATION split (new, this task)

The 1,000 official AV2 VAL scenarios are a physically separate root
(`~/datasets/av2/processed/kalmannet_scaleup_v2/official_val/`) and are
**never** read during training, early stopping, model selection, LR
decisions, corruption tuning, or seed selection.

A second, independent split subdivides the 10,000-scenario TRAIN-source
pool itself: **9,000 -> internal TRAIN, 1,000 -> internal VALIDATION**.
New `tools/kalmannet_training/build_scaleup_v2_internal_split.py`
(`assign_internal_split`/`build_internal_split_manifest`) -- deliberately
NOT `av2_split.py::assign_split` (that function enforces a mandatory
train/val/test 3-way carve-out with every bucket non-empty; this task
needs exactly two non-empty buckets, an empty `test` list). Deterministic
given (the *set* of scenario_ids, seed): sorts first, then a seeded
`random.Random` shuffle -- same discipline as every other split-generation
function in this project. **New fixed seed `20260930`**, distinct from
every seed already used in this project's AV2 history (dataset-selection
seeds 20260905/20260910/20260920/20260921; every prior split-generation
seed).

Real run against the committed
`scaleup_v2_{train,official_val}_scenarios.manifest.json`:

```
split_counts: {"train": 9000, "val": 1000, "test": 0}
verified_overlap_with_official_val: {"internal_train_vs_official_val": 0, "internal_val_vs_official_val": 0}
```

Zero overlap confirmed **programmatically** (`check_disjoint_from_official_val`,
called by `build_scaleup_v2_internal_split.py::main`, which refuses to
write the manifest at all -- `InternalSplitError` -- if either overlap is
nonzero). Manifest schema reused verbatim (`av2_kalmannet_stage0_split_v1`,
so `kalmannet_sequences.load_split_sequences` needs no change) with a new
`split_policy` value (`scenario_level_internal_train_val_of_scaleup_v2_train_pool`).
The generated manifest (scenario-id lists) is machine-local, not committed
(same convention as every prior split manifest in this project).

`internal_split_manifest_sha256`: `f4bb07b2c9a763f088d719c2722cc6f9be4a2255d87d012db7ea5024573983ad`
`train_scenario_manifest_sha256`: `12ae390ea1b01e714b6e9d3b8d4a06822547b035d8aa8e5746ebdc92526c4c59`
`official_val_scenario_manifest_sha256`: `bebbfd3bc1caca11ece6af6f8c78c88b2fbbd08b221b18824590d584348a66d0`

## 3. Model trained: ONE, GENERIC-ROBUST, ONE seed

Per this task's explicit scope: **only** GENERIC-ROBUST KalmanNet, seed 0.
CLEAN-10k, MORAI-calibrated-10k, 3-seed, mixed-corruption curriculum, and
class-specific KNet are explicitly out of scope (future experiments).

`KalmanNetGRU` architecture unchanged (`ad_lidar_perception/ad_lidar_perception/kalmannet_core.py`,
not touched by this task). State `[x, y, vx, vy]`, measurement `[x, y]`.

GENERIC-ROBUST corruption preset reused verbatim
(`train_kalmannet.CONDITION_PRESETS["generic_robust"]` --
`gaussian_noise_std_m=0.3, dropout_prob=0.1, dropout_burst_enabled=True,
dropout_burst_prob=0.02, dropout_burst_min_len=2, dropout_burst_max_len=5,
seed=20260905` -- the SAME generic robustness family the Stage-1 pilot
used, not the MORAI-calibrated family). Not claimed MORAI-realistic.

## 4. Frozen training config (unchanged, not retuned)

```
batch_size = 64
learning_rate = 0.004
max_epochs = 60
patience = 15
grad_clip = 10.0
loss_on_predict_only = True
hidden_size = 32 (historical default, unchanged)
use_length_bucketing = True, n_buckets = 8 (length_bucketed_v1 sampler, unchanged)
device = cpu
seed = 0 (init_seed = order_seed = 0)
```

## 5. Device: CPU

Per this task's own instruction and this project's own already-measured
finding (`docs/perception/kalmannet_batched_training_v1.md`): this
model/trainer is faster on CPU than the RTX 4060 even after batching.
No new CPU-vs-CUDA benchmark was run. No explicit thread/process pinning
was applied -- no such policy exists anywhere in this project's training
code (verified: no `torch.set_num_threads`/`OMP_NUM_THREADS` in any
existing trainer script), so this run used torch's own default CPU
thread behavior on a single process (no oversubscription). Note: the
checkpoint manifest's `environment.cuda_available: true` reflects that
this host *has* a GPU, not that it was used -- `--device cpu` was passed
explicitly and training ran entirely on CPU.

## 6. Resumability (NEW capability added by this task)

**Neither `batched_trainer.py` nor `trainer_core.py` had any resume
support before this task** (verified by reading both files fresh) -- a
crash at any point meant restarting the whole run from epoch 0. The
smallest correct capability was added:

- New `tools/kalmannet_training/resume_state.py`
  (`save_resume_state`/`load_resume_state`, atomic write via
  `.tmp` + `os.replace`): model `state_dict`, optimizer `state_dict`
  (Adam moment estimates, not just weights), the `numpy.random.RandomState`
  used for per-epoch shuffle order, `epoch_next`, `best_val`/`best_epoch`/
  `best_state_dict`/`epochs_since_improve` (exact early-stopping state),
  every completed epoch's `EpochRecord`, running `nonfinite_step_count`
  and `cumulative_train_time_s`, and a `validation_key` dict of
  everything that must be IDENTICAL across a resume. `load_resume_state`
  raises `ResumeValidationError` on ANY mismatch.
- `batched_trainer.train_one_run_batched` gained optional
  `resume_checkpoint_path`/`resume_validation_key`/`checkpoint_every_epochs`
  parameters (all default to the prior no-resume behavior --
  **byte-identical** when unused, confirmed by
  `test_no_resume_path_behaves_exactly_as_before`).
- `train_kalmannet_batched.py` CLI: `--resume-checkpoint` (default:
  `<output-checkpoint>.resume.pt`, auto-derived), `--no-resume`,
  `--checkpoint-every-epochs` (default 1, used for the real run below),
  `--log-file` (machine-local per-epoch log, see Section 8).

**Verified correct, not just implemented**, at two levels:
1. Library level (`test_batched_trainer.py::test_resume_from_checkpoint_reaches_the_same_final_result_as_uninterrupted_run`).
2. **CLI level, end-to-end, against a real (fixture-scenario) sharded AV2
   export** (`test_train_kalmannet_batched_cli_resume.py`): a real crash
   was simulated by monkeypatching `save_resume_state` to raise
   immediately after durably writing the epoch-2 resume checkpoint;
   restarting with the IDENTICAL CLI arguments resumed automatically and
   reached a **byte-identical final checkpoint SHA-256** to an
   uninterrupted 5-epoch reference run. A second test confirms a changed
   hyperparameter on restart raises `ResumeValidationError`.

**In the event, the real 10k run did NOT need to use this capability** --
it completed in one continuous process (6.45 hours, no crash). The
resume checkpoint (`*.pt.resume.pt`, 133,694 bytes) was still written
after every epoch as configured, confirming the capability is live and
correctly overwritten each epoch (not just theoretically present).

## 7. Training health / real-run notes

**A real, severe gradient-instability event occurred and is reported
honestly, not smoothed over.** From epoch 6 onward, every subsequent
epoch hit at least one batch with an `inf` (or, before that, extremely
large but technically finite, up to ~10^32) gradient norm. Gradient
clipping (`max_norm=10.0`) contained every one of these through epoch 15
-- `val_loss` stayed in a 0.745-1.031 band the whole time, well below the
`FAILURE_VAL_LOSS_THRESHOLD=100.0` catastrophic classification, and
`any_nan_train`/`any_nan_val` stayed `False`. **At epoch 16 the network's
weights went permanently non-finite** (`val_loss=nan`,
`any_nan_train=True`, `any_nan_val=True`) -- from epoch 17 onward *every*
batch produced zero valid loss terms (`n_loss_terms=0` universally,
consistent with a NaN-poisoned hidden state making an internal
finiteness-dependent mask resolve false everywhere), `grad_norm=0`,
`train_loss=nan`. **This is a real seed-instability failure, directly
analogous to this project's own prior finding (T-12.3,
`docs/agent/STATUS.md`) that some KalmanNet training seeds diverge
catastrophically while others don't** -- this 10k run's seed 0 appears to
be one of the unlucky ones, now demonstrated at 5x the prior scale. No
hyperparameter was changed in response; the run was allowed to continue
to its own natural early-stopping (per this task's explicit instruction
not to silently restart with different hyperparameters).

**Crucially, this did not corrupt the selected checkpoint.** The
trainer's own `math.isfinite(val_loss)` gate on checkpoint selection
correctly ignored every non-finite epoch; `best_epoch=6`
(`val_loss=0.7451300733912749`) was locked in before the collapse and
remained the selected checkpoint through to the end. `catastrophic=False`
in the final `TrainResult` (correct -- `best_val` itself, 0.745, is far
below the 100.0 threshold; `best_state_dict` is not `None`).

Full per-epoch record: `docs/perception/../` -- machine-local log at
`~/datasets/av2/logs/av2_scaleup_v2_10k_generic_robust_bs64_lr004_seed0_train.log`,
not committed. `nonfinite_step_count = 46,525` out of 170,654 total
optimizer steps across the run (27.3%) -- almost entirely from the
epoch-17-through-21 fully-collapsed tail (6 epochs x 7,757 steps =
46,542, matching almost exactly).

**Real environment constraint found while launching this run**: the host
has 16 GB RAM; loading all 553,806 internal-split sequences (496,434
internal-train + 57,372 internal-val) into memory (per this trainer's
existing, unmodified whole-dataset-resident design) used approximately
13 GB RSS during training, leaving roughly 2.4 GB free -- substantially
more than the Stage-1 pilot (2,000 scenarios, ~1/5 the sequence count)
needed. Memory was stable (not growing) once data loading completed.
During the later official-VAL evaluation phase (Section 12), which
re-loads the same internal split for KF calibration, swap briefly
engaged (~1 GB) under concurrent memory pressure -- slower, not fatal.
Documented as a real scale-driven resource constraint for any future
larger AV2 run on this host, not a code defect.

`train=496,434 val=57,372` sequences loaded under GENERIC-ROBUST
corruption (`load_split_sequences` + `_truncate_to_first_measurement`
drop a small fraction of the raw 553,810 segments that have no valid
leading measurement after dropout truncation -- expected, not a bug).

## 8. Per-epoch logging

One line per epoch, machine-local (`--log-file`, not committed):

```
epoch=<n> train_loss=<f> val_loss_mse_xy_vxvy=<f> grad_norm_mean=<f> grad_norm_max=<f>
optimizer_steps_this_epoch~=<n_batches> epoch_wall_time_s=<f> cumulative_wall_time_s=<f>
best_epoch=<n|None> patience_remaining=<n> any_nan_train=<bool> any_nan_val=<bool>
```

**Note on position/velocity RMSE per epoch**: the trainer's own
per-epoch validation metric is `val_loss` -- the mean per-frame MSE
across all four state dimensions `[x,y,vx,vy]` jointly -- not separately
split into a position-only and velocity-only RMSE (that split is
computed once, at the final official-VAL evaluation, via
`evaluate_estimator()`, Section 12-13).

## 9. Training health gates

Gradient clipping fixed at `max_norm=10.0` (unchanged) -- contained
every finite spike through epoch 15 (Section 7). NaN/Inf and
non-finite-step counts were tracked per epoch and are reported above.
No hyperparameter was changed mid-run in response to any transient or
sustained event. The run's own final state (`catastrophic=False`,
`best_state_dict is not None`, `best_val=0.745 << 100.0`) means the
early-stopping mechanism, not a manual STOP, correctly resolved this --
consistent with this task's own instruction to let the configured run
finish rather than intervene.

## 10. Model selection

Selected **purely from internal validation** -- the existing, unmodified
`train_one_run_batched` checkpoint-selection semantics (best finite
`val_loss`, `epsilon=1e-6` improvement threshold, patience-based early
stopping) were reused unchanged. **Best epoch: 6**
(`val_loss=0.7451300733912749`). Official AV2 VAL was never inspected to
choose an epoch or checkpoint.

## 11. Freeze manifest

New `tools/kalmannet_training/write_freeze_manifest_av2_10k_generic.py`
reuses the SAME `kalmannet_batch_calibration_freeze_v1` schema
(`freeze_manifest.py`, unmodified) every prior AV2 freeze manifest in
this project uses. Refuses to write if the checkpoint's own recorded
`split_manifest_sha256` does not match the internal split manifest
passed in. Written **before** any official-VAL evaluation (file mtime
18:00, first official-VAL evaluation process started after that) --
`evaluate_kalmannet_official_val.py` calls
`freeze_manifest.require_freeze_manifest_exists` first and refuses to
run if it does not exist -- verified by test.

Key frozen fields (`av2_scaleup_v2_10k_generic_robust_bs64_lr004_seed0_FREEZE.json`,
machine-local, not committed -- summarized here):

```
batch_size=64, learning_rate=0.004, max_epochs=60, patience=15, gradient_clip=10.0
loss_on_predict_only=true, hidden_size=32, seed=0 (init_seed=order_seed=0)
best_epoch=6, internal_validation_best_loss=0.7451300733912749
checkpoint_sha256=30e0902ef91d2e8f189db163a2ec636d39a7ad9d4f5abc414b8aaa8ae873a778
internal_split_manifest_sha256=f4bb07b2c9a763f088d719c2722cc6f9be4a2255d87d012db7ea5024573983ad
internal_split_seed=20260930, internal_split_counts={train:9000, val:1000, test:0}
train_scenario_manifest_sha256=12ae390ea1b01e714b6e9d3b8d4a06822547b035d8aa8e5746ebdc92526c4c59
official_val_scenario_manifest_sha256=bebbfd3bc1caca11ece6af6f8c78c88b2fbbd08b221b18824590d584348a66d0
dataset_manifest_sha256_recorded_in_checkpoint=ab0f9faa8318ecc13a9c0eea37c5414335540adca3b027194f5267da36baeb12
torch_version=2.4.1+cu121 (device used for training: cpu, see Section 5)
```

## 12. Official AV2 VAL evaluation (external, after freeze)

New `tools/kalmannet_training/evaluate_kalmannet_official_val.py` --
reuses `evaluate_kalmannet.py`'s own `run_kf`/`run_knet`/
`evaluate_estimator`/`GAP_BUCKETS`/`DENSE_V2_CHECKPOINT_PATH`/
`TUNED_HEVEN_KF_*` verbatim, and `calibrate_kf.calibrate_kf_on_av2` for
the fair AV2-tuned KF baseline. Adds a **physically separate calibration
root** (internal TRAIN/VAL, 9k/1k) from the **evaluation root** (official
VAL, 1k), the mandatory freeze-guard, and a motion-stratified breakdown.

**Anti-leakage guarantee verified by test**
(`test_kf_calibration_never_reads_official_val_sequences`,
`test_precomputed_kf_calibration_json_round_trips_and_skips_recalibration`):
the AV2-tuned KF calibration's own `load_split_sequences` call is made
exactly once, against the calibration (internal-split) root; every
subsequent call is against the official-VAL root.

**AV2-tuned KF, calibrated on internal TRAIN(9k)/VAL(1k) only**
(official VAL never touched): `sigma_a=2.0, r_std=0.150006, p0_scale=10.0`
(internal-val position RMSE at calibration time: 0.342m). This SAME
calibration was reused (via `--precomputed-kf-calibration-json`, a real
bug found and fixed during this task -- see Section 22) for every
baseline checkpoint's evaluation below, since KF calibration is
checkpoint-independent.

Four conditions evaluated per checkpoint:

- **A. CLEAN** -- `CorruptionConfig(mode="none")`.
- **B. GENERIC-ROBUST**, same corruption family/seed as training.
- **C. GENERIC-ROBUST**, different deterministic corruption seed (`999999`).
- **D. MORAI-CALIBRATED corruption** -- evaluation-only diagnostic (run
  for checkpoint A only), using the already-frozen
  `morai_calibration.build_morai_calibrated_corruption_config()`. **This
  is NOT MORAI evaluation** -- it never touches MORAI ground truth or a
  MORAI replay stream, it only tests tolerance to a harsher, anisotropic
  measurement-corruption family.

## 13. Full results table -- checkpoint A (new AV2 10k GENERIC-ROBUST KNet)

All N = number of GT frames evaluated (not sequences); n_sequences = 56,521
(56,348 under condition D, a small truncation difference under that
corruption profile). 0 divergence, 0 nonfinite across every condition.

| condition | N | KNet pos RMSE (m) | KNet vel RMSE (m/s) | KF-transferred pos RMSE (m) | KF-AV2-tuned pos RMSE (m) | dense-v2 pos RMSE (m) |
|---|---|---|---|---|---|---|
| A. CLEAN | 2,918,906 | **0.0435** | 1.258 | 0.0771 | 0.1076 | 0.0638 |
| B. GENERIC-ROBUST (same seed) | 2,908,189 | **0.3095** | 1.390 | 0.3556 | 0.3465 | 0.4467 |
| C. GENERIC-ROBUST (diff seed) | 2,908,098 | **0.3106** | 1.394 | 0.3567 | 0.3477 | 0.4468 |
| D. MORAI-calibrated (diagnostic) | 2,899,090 | **1.7155** | 2.063 | 2.0352 | 1.9662 | 2.9481 |

Checkpoint A's KNet is the best-or-tied estimator on every condition
except D-vs-KF-AV2-tuned is closer (1.72 vs 1.97, still KNet wins) --
i.e. this checkpoint beats both a transferred-from-MORAI KF, a KF
calibrated natively on this same 10k pool, and the older MORAI-trained
dense-v2 checkpoint, on every condition tested. **No MORAI performance
claim** -- these are all AV2-domain comparisons.

Matched/missing/gap breakdown (condition B, representative):

| bucket | N | pos RMSE (m) | vel RMSE (m/s) |
|---|---|---|---|
| matched (measurement present) | 2,456,380 | 0.268 | 1.381 |
| missing (any gap length) | 451,809 | 0.475 | 1.437 |
| gap 1-2 frames | 358,983 | 0.373 | 1.419 |
| gap 3-5 frames | 87,404 | 0.709 | 1.499 |
| gap 6+ frames | 5,422 | 1.221 | 1.596 |

Position RMSE degrades monotonically with gap length, as expected
(longer coast = more drift); condition D shows the same monotonic
pattern at a much larger scale (gap 6+ pos RMSE 4.12m vs. matched's
1.08m).

## 14. Class-stratified evaluation

`coarse_object_type` (VEHICLE/PEDESTRIAN/OTHER) is carried per segment
purely as evaluation metadata -- KalmanNet remains fully class-agnostic
(never fed class as a model input).

Checkpoint A, condition B (GENERIC-ROBUST):

| class | N | pos RMSE (m) | vel RMSE (m/s) |
|---|---|---|---|
| VEHICLE | 2,347,374 | 0.314 | 1.490 |
| PEDESTRIAN | 212,469 | 0.277 | 0.718 |
| OTHER | 348,346 | 0.300 | 0.920 |

Pedestrian velocity RMSE is consistently lowest across all classes/
conditions (pedestrians move slower, smaller absolute velocity errors) --
a descriptive pattern, not a claim about per-class model quality.

## 15. Motion-stratified evaluation (diagnostic only, never used for selection)

Checkpoint A, all conditions (GT speed `hypot(vx,vy)`, threshold 0.5 m/s):

| condition | near-stationary pos RMSE (m) | moving pos RMSE (m) |
|---|---|---|
| A. CLEAN | 0.0201 | 0.0680 |
| B. GENERIC-ROBUST | 0.257 | 0.388 |
| C. GENERIC-ROBUST (diff seed) | 0.258 | 0.389 |
| D. MORAI-calibrated | 1.279 | 2.308 |

Near-stationary tracks are consistently easier (lower RMSE) than moving
tracks across every condition, as expected -- purely descriptive, this
bucketing was never used to select the checkpoint, epoch, or any
hyperparameter.

## 16. Baselines evaluated on the SAME official-VAL streams

- **A.** New AV2 10k GENERIC-ROBUST KNet (this task, Section 13).
- **B.** AV2 Stage-1 pilot (2,000-scenario) GENERIC-ROBUST KNet --
  `av2_stage1_pilot_generic_robust_bs64_lr004_seed0.pt` (checkpoint
  SHA-256 `65fe0c2b13caa2948c8128aacb331a69e0b0f1087ba581be7df7d2e5944e0c17`,
  git SHA `63e83e1c60b5cf0035f0c11893ab6db3ceaa84bf` -- provenance verified
  present on disk with a matching manifest before use). **This IS the
  primary 2k-vs-10k comparison baseline, Section 17.**
- **C.** AV2 Stage-1 pilot MORAI-calibrated KNet (diagnostic only) --
  `av2_stage1_pilot_morai_calibrated_robust_bs64_lr004_seed0.pt`.
- **D.** DENSE-KALMANNET-v2 (MORAI-trained) -- cross-domain diagnostic,
  evaluated automatically alongside checkpoint A (Section 13 table).
- **E.** AV2-tuned `LinearCVKF`, Section 12/13 ("KF-AV2-tuned" column).

Baseline B (2k pilot GENERIC-ROBUST), same 4-column format:

| condition | N | KNet pos RMSE (m) | KNet vel RMSE (m/s) |
|---|---|---|---|
| A. CLEAN | 2,918,906 | 0.0691 | 1.271 |
| B. GENERIC-ROBUST (same seed) | 2,908,189 | 0.3157 | 1.400 |
| C. GENERIC-ROBUST (diff seed) | 2,908,098 | 0.3176 | 1.404 |

Baseline C (2k pilot MORAI-calibrated, its own native corruption family,
diagnostic only -- note this checkpoint's own "B/C" conditions are
MORAI-calibrated-based, not GENERIC-ROBUST, since that is its training
condition):

| condition | N | KNet pos RMSE (m) | KNet vel RMSE (m/s) |
|---|---|---|---|
| A. CLEAN | 2,918,906 | 0.1856 | 1.392 |
| B. MORAI-calibrated (native, same seed) | 2,899,090 | 1.4093 | 1.820 |
| C. MORAI-calibrated (native, diff seed) | 2,899,834 | 1.4024 | 1.826 |

Baseline C performs *worse* than both A and B on CLEAN (0.186m vs.
0.044m/0.069m) -- expected, since it was trained to expect biased,
harsher noise, not clean measurements. Baseline C performs *better* than
checkpoint A's own D-diagnostic result under MORAI-calibrated corruption
(1.41m vs. A's 1.72m) -- also expected, since C was actually trained on
this corruption family (in-distribution) while A was not
(out-of-distribution diagnostic only).

## 17. Primary scientific comparison: 2k vs. 10k

**Same architecture, optimizer family, corruption family, and evaluator
-- only AV2 scenario count differs (2,000 vs. 10,000).**

| condition | 2k (checkpoint B) pos RMSE (m) | 10k (checkpoint A) pos RMSE (m) | relative change |
|---|---|---|---|
| A. CLEAN | 0.0691 | 0.0435 | **-37.0%** (10k better) |
| B. GENERIC-ROBUST (same seed) | 0.3157 | 0.3095 | **-2.0%** (10k better) |
| C. GENERIC-ROBUST (diff seed) | 0.3176 | 0.3106 | **-2.2%** (10k better) |

Velocity RMSE shows the same direction, smaller magnitude:

| condition | 2k vel RMSE (m/s) | 10k vel RMSE (m/s) | relative change |
|---|---|---|---|
| A. CLEAN | 1.271 | 1.258 | -1.0% |
| B. GENERIC-ROBUST (same seed) | 1.400 | 1.390 | -0.7% |
| C. GENERIC-ROBUST (diff seed) | 1.404 | 1.394 | -0.7% |

**10k improves on 2k in every single condition tested (position and
velocity, all three corruption conditions) -- never worse.** The
improvement is large on clean measurements and small-but-consistent
under the corruption family both models were actually trained for.

## 18. Interpretation

**Classified as CASE C, leaning toward a mild CASE A** (using this
task's own four-outcome framing):

- Not CASE D (10k is never worse than 2k on any tested condition).
- Not a clean CASE A ("10k improves clean + robust official-VAL metrics"
  outright) because the robust-condition improvement (-2%) is small
  relative to the clean-condition improvement (-37%) -- the two gains
  are not of comparable magnitude.
- Not CASE B (10k does not hurt clean -- it dramatically helps clean).
- Best summary: **AV2 scale (2k -> 10k) clearly helps this model's
  ability to fit clean, well-measured trajectories (a ~37% error
  reduction, suggesting real undertraining/underfitting on 2k scenarios
  for the clean regime), but the corrupted-measurement regime shows
  diminishing returns from scale alone (~2% reduction)** -- consistent
  with GENERIC-ROBUST corruption (Gaussian noise + dropout) imposing an
  irreducible noise floor that more clean scenario diversity does not
  push through as effectively. This matches interpretation **C** ("motion-
  data scale is reaching diminishing returns for this architecture/config")
  for the robust regime specifically, while interpretation **A** ("strong
  evidence AV2 scale is beneficial in-domain") holds for the clean regime.

**No claim of MORAI or competition-performance improvement is made
anywhere in this document.** Whether either of these AV2-domain findings
transfers to MORAI or competition performance remains completely open
and untested -- that question is frozen until `MORAI_ESTIMATOR_EVAL_V2`
data can be collected.

## 19. No runtime integration

Confirmed: no change to the runtime KalmanNet checkpoint, AB3DMOT,
CenterPoint, ROS launch defaults, planner, prediction, or occupancy grid.
This remains strictly offline pretraining research.

## 20. Training time report

| metric | value |
|---|---|
| epoch count (`n_epochs_run`) | 22 (epochs 0-21) |
| early-stop epoch | 21 (patience exhausted, 15 epochs with no internal-val improvement past epoch 6) |
| best epoch | 6 |
| total wall time | 23,236.98 s = **6.455 hours** |
| mean epoch time, compute epochs (0-15) | 1,140.84 s = 19.01 min |
| mean epoch time, collapsed epochs (16-21) | 830.6 s = 13.84 min (faster -- zero loss terms computed) |
| sequences/sec (compute epoch) | ~435.1 seq/s |
| optimizer steps/epoch | 7,757 |
| total optimizer steps (22 epochs) | 170,654 |

**Real epoch time (19.0 min, compute epochs) vs. the PR #54 estimate
(54.2 min/epoch, linearly scaled from the 2k pilot's own measured
throughput) is ~2.85x faster than predicted** -- per-epoch cost did not
scale linearly with the 6.25x segment-count ratio; plausibly the 2k
pilot's own per-epoch measurement carried more fixed/startup overhead
proportionally, or length-bucketed batch composition differs at this
scale. **Total wall time (6.46 hours) vs. the PR #54 estimate (~28.9
hours for ~32 epochs) is dominated by both this faster-than-predicted
per-epoch cost AND fewer epochs actually run (22 vs. the assumed ~32,
because of the epoch-16 collapse cutting the run short).**

**3-seed cost estimate (NOT executed, estimate only)**: this single
seed's real cost (6.46 hours) included ~1.4 hours of post-collapse
epochs that did no useful computation. A more representative "healthy"
per-seed cost, if a seed did NOT collapse and ran a similar ~22-32
productive epochs, would be roughly 22-32 x 19.0 min ~= **7.0-10.1
hours/seed**. Three independent seeds (not all necessarily hitting this
same instability) would cost roughly **21-30 hours** total if run
sequentially -- a rough, real-data-informed range, not a guarantee (seed
0's own collapse demonstrates real per-seed variance in both epoch count
and outcome, exactly the kind of instability a 3-seed sweep would exist
to characterize). **Seeds 1/2 were NOT started, per this task's explicit
instruction.**

## 21. Disk usage

| | before this task | after this task |
|---|---|---|
| `~/datasets/av2/` total | 6.3 GB (PR #54, unchanged) | 6.3 GB (checkpoints/logs/reports add ~250 KB, negligible) |
| new checkpoint | -- | 33,896 bytes (`av2_scaleup_v2_10k_generic_robust_bs64_lr004_seed0.pt`) |
| resume checkpoint (final state) | -- | 133,694 bytes |
| checkpoint manifest | -- | 2,280 bytes |
| freeze manifest | -- | 3,438 bytes |
| training history CSV | -- | 1,530 bytes |
| official-VAL report (checkpoint A) | -- | 34,681 bytes |
| official-VAL report (baseline B) | -- | 20,401 bytes |
| official-VAL report (baseline C) | -- | 23,660 bytes |
| per-epoch training log (machine-local) | -- | ~5 KB |
| host free disk (`/`) | 900 GB | ~900 GB (unaffected) |

Nothing deleted.

## 22. Tests

New: `tools/kalmannet_training/{build_scaleup_v2_internal_split.py,
resume_state.py, evaluate_kalmannet_official_val.py,
write_freeze_manifest_av2_10k_generic.py}` + their test files
(`test_build_scaleup_v2_internal_split.py` 9, `test_resume_state.py` 5,
`test_evaluate_kalmannet_official_val.py` 5, plus 5 new resume tests in
`test_batched_trainer.py` and 2 new CLI-level resume tests in
`test_train_kalmannet_batched_cli_resume.py`). Modified (additive only,
default-preserving): `batched_trainer.py` (resume parameters),
`train_kalmannet_batched.py` (resume/log-file CLI args).

**A real bug was found and fixed during actual use of this pipeline (not
just in test fixtures)**: `evaluate_kalmannet_official_val.py`'s
`--precomputed-kf-calibration-json` path read the wrong dict keys
(`val_position_rmse`/`val_velocity_rmse` instead of the report's actual
`internal_val_position_rmse`/`internal_val_velocity_rmse`) -- raised a
`KeyError` on first real use when evaluating baseline B. Fixed, and
`test_precomputed_kf_calibration_json_round_trips_and_skips_recalibration`
added to lock the fix (also verifies the calibration/TRAIN root is never
touched when reusing a precomputed calibration).

Full `tools/kalmannet_training/` suite: **169/169 pass** (145
pre-existing unaffected + 24 new). AV2 adapter + scale-up dataset test
suites (`ad_morai_bridge_dev/test/test_av2_motion_forecasting_adapter.py`,
`test_av2_kalmannet_shard_loader.py`, `tools/av2_dataset_prep/`):
**102/102 pass**, unchanged. `pyflakes` clean on every new/modified file.
`py_compile` clean. `git diff --check` clean.

## 23. Files

New: `tools/kalmannet_training/{build_scaleup_v2_internal_split.py,
test_build_scaleup_v2_internal_split.py, resume_state.py,
test_resume_state.py, test_train_kalmannet_batched_cli_resume.py,
evaluate_kalmannet_official_val.py, test_evaluate_kalmannet_official_val.py,
write_freeze_manifest_av2_10k_generic.py}`, this file. Modified:
`tools/kalmannet_training/{batched_trainer.py, train_kalmannet_batched.py,
test_batched_trainer.py}` (additive, default-preserving). No repo
algorithm/config/launch/production file touched;
`ad_lidar_perception/kalmannet_core.py` untouched. No checkpoint, log,
dataset, or report file committed (all live under `~/datasets/av2/`,
machine-local per this project's established convention).

## 24. Limitations

Single seed -- and that seed suffered a real gradient-explosion collapse
at epoch 16 (Section 7), meaning the reported checkpoint reflects only
6 "healthy" epochs of training, not the full 22-epoch budget; a
non-collapsing seed might reach a materially different (better or worse)
result. CPU-only (no fresh CUDA benchmark at this scale, per instruction).
Per-epoch internal-val metric is joint MSE, not separately split RMSE.
AV2 official-VAL is not MORAI -- no domain-transfer or competition-
performance claim anywhere in this document. Memory footprint at 10k
scale (~13 GB RSS) is close to this host's 16 GB physical limit -- a
real scale-driven constraint, not a code defect. `run_knet`/`run_kf`
evaluation is the historical unbatched per-sequence path (same as every
prior AV2 evaluation in this project), making a full official-VAL sweep
slow (~1-1.5 hours per checkpoint per condition) -- not something this
task's own "smallest correct capability" scope justified batching.

## 25. Recommended next task

Given the real epoch-16 gradient-explosion collapse found here (Section
7, 9), the highest-value next step is **not** simply scaling further --
it is understanding and, if possible, hardening against this seed-
instability failure mode at scale (e.g. does a lower learning rate,
warmup, or stricter clipping threshold prevent it without sacrificing
the strong clean-data result?), likely via a bounded multi-seed sweep at
a cheaper scale (Stage-1 pilot, 2k) before committing another ~6-10 hour
10k run per seed. Per this task's own explicit instruction, no second
seed or training matrix was started here regardless of this finding.
