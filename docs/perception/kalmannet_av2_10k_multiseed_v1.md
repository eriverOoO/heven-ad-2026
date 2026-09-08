# AV2 10K GENERIC-ROBUST Multi-Seed Completion v1

Branch `exp/kalmannet-av2-10k-multiseed-v1`, from `origin/main` `754d6dc`
(merge of PR #58 `fix(kalmannet): handle aggregate gradient norm
overflow`, verified `MERGED` before branching). **Offline experiment
only. No `KalmanNetGRU`/`KalmanNetFilter`/AB3DMOT/CenterPoint/ROS/
prediction/planner/occupancy-grid file changed. No MORAI evaluation, no
MORAI/competition-performance claim -- `MORAI_ESTIMATOR_EVAL_V2` still
does not exist.**

## 1. Goal

PR #58 found that the finalized gradient-safety guard carries the exact
AV2 10k GENERIC-ROBUST seed-0 configuration through the entire 20-epoch
bounded window -- including epoch 16, the original run's own documented
permanent-collapse point -- without a single element-level non-finite
gradient. This establishes numerical *safety*, but not seed-level
*stability*: one seed cannot answer whether this is reliable across
initialization/data-order variation. This task trains seeds 1 and 2 of
the same frozen configuration to completion (not bounded to 20/12 epochs
-- the real `max_epochs=60`, `patience=15` budget) and compares all three
seeds on stability, internal validation, and official AV2 VAL
performance.

## 2. Precondition verification

`gh pr view 58 --json state,mergedAt` confirmed `MERGED` before any edit.
Worktree created at `/tmp/heven-worktrees/kalmannet-av2-10k-multiseed-v1`
(`git worktree add ... -b exp/kalmannet-av2-10k-multiseed-v1 origin/main`,
HEAD `754d6dc`), never the dirty main checkout.

## 3. Frozen dataset (unchanged, verified against seed 0's own freeze manifest)

```
train_scenario_manifest_sha256      = 12ae390ea1b01e714b6e9d3b8d4a06822547b035d8aa8e5746ebdc92526c4c59
official_val_scenario_manifest_sha256 = bebbfd3bc1caca11ece6af6f8c78c88b2fbbd08b221b18824590d584348a66d0
internal_split_manifest_sha256      = f4bb07b2c9a763f088d719c2722cc6f9be4a2255d87d012db7ea5024573983ad
  (9,000 TRAIN / 1,000 INTERNAL VAL, seed 20260930)
official_val: 1,000 scenarios, untouched
```

Same shard roots as PR #55/#57/#58:
`~/datasets/av2/processed/kalmannet_scaleup_v2/{train,official_val}`.
No split or scenario membership changed for this task.

## 4. Frozen training config (unchanged, not retuned)

```
KalmanNetGRU, hidden_size=32
batch_size=64, learning_rate=0.004, max_epochs=60, patience=15
grad_clip=10.0, loss_on_predict_only=True
corruption: GENERIC-ROBUST (gaussian_noise_std_m=0.3, dropout_prob=0.1,
            dropout_burst_enabled=True, dropout_burst_prob=0.02,
            dropout_burst_min_len=2, dropout_burst_max_len=5, seed=20260905)
device: cpu
use_length_bucketing=True, n_buckets=8
```

Exact CLI (seed 1, seed 2 identical except `--seed`):

```
python3 train_kalmannet_batched.py \
  --shard-root ~/datasets/av2/processed/kalmannet_scaleup_v2/train \
  --split-manifest ~/datasets/av2/manifests/scaleup_v2_internal_split_9k1k.json \
  --condition generic_robust --device cpu \
  --batch-size 64 --hidden-size 32 --lr 0.004 --grad-clip 10.0 \
  --max-epochs 60 --patience 15 --seed {1,2} \
  --output-checkpoint ~/datasets/av2/checkpoints/av2_scaleup_v2_10k_generic_robust_bs64_lr004_seed{1,2}.pt \
  --output-history ~/datasets/av2/checkpoints/av2_scaleup_v2_10k_generic_robust_bs64_lr004_seed{1,2}_history.csv \
  --checkpoint-every-epochs 1 \
  --log-file ~/datasets/av2/logs/av2_scaleup_v2_10k_generic_robust_bs64_lr004_seed{1,2}_train.log
```

## 5. Safety guard

Uses the finalized PR #58 `nonfinite_guard.safe_clip_and_step` unmodified
-- three-state classification (A healthy / B norm-overflow / C
element-nonfinite), STATE B skip-and-zero with a robust float64
diagnostic norm, STATE C skip-and-escalate per the existing PR #56
policy, immediate `catastrophic=True` on any parameter/optimizer-state
non-finite. No reversion to raw `clip_grad_norm_`.

## 6. A real gap found and fixed BEFORE launching seed 1 (not after)

`train_kalmannet_batched.py`'s existing summary/history output did not
record ANY of the PR #58 health counters
(`norm_overflow_count`/`per_element_nonfinite_gradient_count`/
`parameter_collapse_count`/`optimizer_state_collapse_count`/
`large_finite_gradient_count`/`training_unstable`) anywhere -- only the
old, pre-PR-58 combined `nonfinite_step_count`. Since these are cumulative
in-memory `TrainResult` fields never persisted, they would have been
**permanently lost** the moment the training process exited. Caught
~30 seconds into the first seed-1 launch attempt (before any real
progress was made) by cross-checking this task's own section 6
requirements against the actual script output -- the run was killed and
restarted after the fix, at negligible real cost (no epoch had completed
yet).

**Fixes applied** (`train_kalmannet_batched.py`, `trainer_core.py`,
`batched_trainer.py`, `resume_state.py`):
- All 9 cumulative health counters + `training_unstable`/
  `training_collapsed`/`abort_reason` added to the checkpoint's own
  `training_config` manifest section (an already-free-form dict, no
  change to `checkpoint_utils.build_checkpoint_manifest`'s own
  signature) and to the final printed summary JSON (including the
  catastrophic-abort early-return path, which previously printed almost
  nothing).
- `EpochRecord` (in `trainer_core.py`, shared by both trainers) gained
  `n_norm_overflow_batches`/`n_element_nonfinite_batches` -- a genuine
  **per-epoch** breakdown, not just a final cumulative total, addressing
  this task's own section 6 requirement directly. Threaded through both
  `train_one_run` and `train_one_run_batched`'s existing per-batch STATE-B/
  STATE-C branches (both already had the classification logic from PR #58;
  this only adds a local per-epoch counter and passes it into the two
  existing `EpochRecord(...)` construction sites). Surfaced in the
  per-epoch log line and the history CSV's own column list.
- **A second real gap, found while implementing the first fix**: adding
  these counters to the checkpoint manifest broke
  `test_cli_resume_reaches_the_same_result_as_an_uninterrupted_cli_run`
  (previously passing) -- `resume_state.py` never serialized/restored
  `grad_skip_count`/`norm_overflow_count`/etc. across a crash+resume
  (a pre-existing, previously-undocumented-as-a-bug limitation), so a
  resumed run's final counters only reflected its own post-resume
  epochs, silently undercounting the pre-crash portion -- and, once
  these counters became part of the persisted checkpoint manifest, this
  divergence changed the checkpoint file's own SHA-256, breaking a
  test that had never needed to care about this before. Fixed by adding
  a generic `extra_counters: dict | None = None` bag to
  `resume_state.save_resume_state`/its payload (avoids a long list of
  new named parameters for a module whose job is exactly "the state a
  resume needs"), threaded through `batched_trainer.py`'s save/load call
  sites. **7 new tests** verify both fixes (2 in `test_resume_state.py`
  for the round-trip + backward-compatible-default; per-epoch/resume
  correctness otherwise exercised via the existing, now-passing CLI
  resume test). Full suite **245/245 pass** after these fixes (216 from
  PR #58's own baseline + 29 new across this task).

## 7. Seed 0 reference -- a real, important nuance

Seed 0's checkpoint (`..._seed0.pt`, `best_epoch=6`,
`internal_validation_best_loss=0.7451300733912749`, git_sha `ba35b513`)
**predates this task**. Its own original training log
(`av2_10k_multiseed_results/seed0_train.log`) shows the REAL run
permanently collapsed to NaN starting at epoch 16
(`train_loss=9.78e35`, then `val_loss=nan grad_norm=nan` from epoch 16
through patience exhaustion at epoch 21 -- `any_nan_val=True` every
remaining epoch, never recovering). This predates PR #58's own
instrumentation, so no `n_norm_overflow_batches`/`n_element_nonfinite_batches`
columns exist for it. **The retained checkpoint is safe only because the
best-checkpoint mechanism saved epoch 6 before the later collapse -- it is
NOT evidence the PR #58 guard survived a real full run.** PR #58's own
"epoch 16 survived" claim refers to a SEPARATE 20-epoch bounded replay of
seed 0's exact config with the new guard active, not a re-run of this real
training. **Seeds 1 and 2 (this task) are therefore the first two
COMPLETE, guard-protected runs of this exact configuration** -- both
reached natural patience-based early stopping (never NaN-collapsed)
despite repeated STATE B/C events.

## 8. Seed 1 training result

23 epochs, `best_epoch=7`, `internal_validation_best_loss=0.7458607519010171`,
wall time 26,843.5 s (7.46 h). `catastrophic=false`, `training_unstable=false`,
`training_collapsed=false`, `abort_reason=null`. Health counters:
`norm_overflow_count=78` (all safely skipped), `per_element_nonfinite_gradient_count=1`
(safely skipped + escalated per PR #56 policy, no collapse), `parameter_collapse_count=0`,
`optimizer_state_collapse_count=0`, `large_finite_gradient_count=11989`
(batches where `grad_clip=10.0` legitimately clipped a large-but-finite
norm). Checkpoint SHA-256 `a9a19353ddb272d3fc0ac3ad7c6754239ef3d8dd00acd3c52a59193d98dc5ee5`.
Epoch-16 -- the exact epoch that permanently killed seed 0 -- passed with
**zero** norm-overflow/element-nonfinite events this seed
(`n_norm_overflow_batches=0 n_element_nonfinite_batches=0`), directly
confirming the guard carries this configuration through the historically
fatal point. One element-nonfinite event occurred later, at epoch 19
(`n_element_nonfinite_batches=1`, `grad_norm_mean=nan`) -- safely skipped,
training continued to natural completion.

## 9. Seed 2 training result

20 epochs, `best_epoch=4`, `internal_validation_best_loss=0.7468887660922348`,
wall time 23,410.2 s (6.50 h). `catastrophic=false`, `training_unstable=false`,
`training_collapsed=false`. `norm_overflow_count=36`,
`per_element_nonfinite_gradient_count=1` (epoch 16 -- the same historically
fatal epoch, safely skipped here too), `parameter_collapse_count=0`,
`optimizer_state_collapse_count=0`, `large_finite_gradient_count=9118`.
Checkpoint SHA-256 `a005efc54b63ad92f4972a2731bc4ff1e5e1d9cc7c45e7e18f3030c0bfe91e51`.

## 10. Numerical health across seeds 1/2 (seed 0 predates this instrumentation)

| | seed1 | seed2 |
|---|---|---|
| norm_overflow_count | 78 | 36 |
| per_element_nonfinite_gradient_count | 1 | 1 |
| parameter_collapse_count | 0 | 0 |
| optimizer_state_collapse_count | 0 | 0 |
| catastrophic | false | false |
| training_unstable | false | false |

Both seeds independently hit exactly one element-nonfinite event (the
same failure class that permanently killed seed 0's real run) and both
survived it without collapse -- this is the central positive result of
this task: the guard's tier-2/tier-3 escalation policy is not seed-0-specific
luck, it generalizes across at least two independent full runs.

## 11. Internal validation by seed

| seed | best_epoch | internal_val_loss | n_epochs_run | wall_s |
|---|---|---|---|---|
| 0 | 6 | 0.7451300733912749 | 22 | 23237.0 |
| 1 | 7 | 0.7458607519010171 | 23 | 26843.5 |
| 2 | 4 | 0.7468887660922348 | 20 | 23410.2 |

Mean 0.745960, std 0.000721 (0.097% relative) -- extremely tight. Checkpoint
selection was internal-validation-only throughout, before any official-VAL
read, per this task's own mandatory ordering (`write_freeze_manifest_av2_10k_generic.py`
+ `evaluate_kalmannet_official_val.py`'s own `require_freeze_manifest_exists` guard).

## 12. Official AV2 VAL by condition (mean / std / min / max across 3 seeds)

All 3 seeds evaluated on the identical, physically separate 1,000-scenario
official-VAL pool (0 overlap with internal TRAIN/VAL, reconfirmed per
seed's freeze manifest). 0 divergence, 0 non-finite evaluation outputs,
every seed, every condition.

| condition | position RMSE mean (std) [min, max] m | velocity RMSE mean (std) [min, max] m/s | n |
|---|---|---|---|
| A CLEAN | 0.042833 (0.002699) [0.039230, 0.045723] | 1.257158 (0.004979) [1.250579, 1.262621] | 2,918,906 |
| B GENERIC-ROBUST (same corruption family as training) | 0.308288 (0.001398) [0.306322, 0.309452] | 1.391835 (0.003285) [1.389182, 1.396464] | 2,908,189 |
| C GENERIC-ROBUST (different corruption seed, 999999) | 0.309574 (0.001384) [0.307617, 0.310574] | 1.395794 (0.003556) [1.392849, 1.400797] | 2,908,098 |
| D MORAI-calibrated corruption (**AV2 diagnostic only, NOT a MORAI evaluation**) | 1.719093 (0.005399) [1.715080, 1.726725] | 2.063006 (0.003178) [2.059335, 2.067087] | 2,899,090 |

Cross-seed relative std is <=0.45% for B/C/D and ~6.3% for A (still
0.0027 m absolute) -- checkpoint selection is robust to seed variation; no
outlier seed in any condition. B vs C (same vs. a different corruption
seed) differ by <0.5%, confirming the checkpoint is not overfit to the
specific training corruption-noise realization.

## 13. Baseline comparison (all reused, no baseline recalibrated)

| estimator | A CLEAN | B GENERIC | C diff-seed corruption | D MORAI-calib diagnostic |
|---|---|---|---|---|
| AV2-tuned Linear KF (calibrated once on internal TRAIN+VAL, reused via `--precomputed-kf-calibration-json`; `sigma_a=2.0, r_std=0.150, p0_scale=10.0`) | 0.1076 | 0.3465 | 0.3477 | 1.9662 |
| DENSE-KALMANNET-v2 (MORAI-domain-trained, evaluated cross-domain on AV2 -- diagnostic only) | 0.0638 | 0.4467 | 0.4468 | 2.9481 |
| Stage-1 2k GENERIC-ROBUST KNet (single seed, 88,875 train sequences, `internal_val_loss=0.7990`) | 0.0691 | 0.3157 | 0.3176 | not evaluated |
| **This task: 10k GENERIC-ROBUST KNet, 3-seed mean** (496,434 train sequences) | **0.0428** | **0.3083** | **0.3096** | **1.7191** |

10k beats the AV2-tuned KF and the MORAI-domain dense-v2 checkpoint on
every condition, every seed. 10k beats the single-seed 2k pilot on every
evaluated condition too, with **zero seeds regressing**.

## 14. 2k vs. 10k conclusion: **B (marginal improvement)**

Position-RMSE relative change, 2k -> 10k mean:
- A CLEAN: 0.0691 -> 0.0428 (**-38.0%**, large)
- B GENERIC-ROBUST: 0.3157 -> 0.3083 (**-2.4%**, marginal)
- C different-seed corruption: 0.3176 -> 0.3096 (**-2.5%**, marginal)

The CLEAN-condition improvement (-38%) and the internal-validation-loss
improvement (0.7990 -> 0.7460, -6.6%) both substantially **overstate** the
practically-relevant GENERIC-ROBUST improvement (~-2.4%). Per this task's
own instruction not to over-weight the CLEAN result: **classified B,
marginal** -- a real, consistent (3/3 seeds beat the 2k baseline), but
small improvement under the corruption regime that actually matters. This
is a genuine finding, not spun toward a stronger conclusion than the data
supports.

## 15. Real compute cost

| seed | epochs run | wall time | avg epoch time | optimizer steps/epoch (~) | total optimizer steps |
|---|---|---|---|---|---|
| 0 | 22 | 6.45 h | 1,056 s | 7,757 | ~170,654 |
| 1 | 23 | 7.46 h | 1,167 s | 7,757 | ~178,411 |
| 2 | 20 | 6.50 h | 1,171 s | 7,757 | ~155,140 |
| **3-seed total (incl. seed 0)** | 65 | **20.41 h** | -- | -- | ~504,205 |

All on CPU (`device=cpu`, per the task's frozen config), single-threaded
per run, run strictly sequentially (never concurrently), per this task's
own explicit instruction. `n_train_sequences=496,434 / n_val_sequences=57,372`
identical across all three seeds (same frozen split).

## 16. Training-stability classification across all 3 seeds

**B (norm-overflow-but-safe-recovery), with isolated C (element-nonfinite)
events that were also safely contained.** No seed reached **D (permanent
collapse)** under the finalized guard -- seed 0's real historical collapse
predates the guard entirely (section 7). Every STATE B/C event across
both guard-protected seeds (78+36 norm-overflow, 1+1 element-nonfinite)
resolved with `parameter_collapse_count=0` and `optimizer_state_collapse_count=0`
in both seeds -- the guard's containment property is confirmed across
independent runs, not a single-run artifact.

## 17. Checkpoint-selection robustness

Best-epoch varied more across seeds (4, 6, 7) than validation loss did
(std 0.00072, 0.097% relative) -- i.e. several different epochs represent
essentially the same validation quality, a flat-optimum property rather
than instability in the selection process itself. No outlier seed in any
official-VAL condition (section 12). Internal-validation-only selection
is confirmed robust to seed variation for this configuration.

## 18. No single "winner" seed picked

Per this task's own instruction, no seed is declared the winner using
official VAL. All 3 seeds' checkpoints, freeze manifests and official-VAL
reports are committed verbatim under
`tools/kalmannet_training/av2_10k_multiseed_results/`. If a future task
needs exactly one deployment checkpoint from this family, the
recommendation is to select using **internal validation only** (any of
the 3 is defensible; seed 0's best_epoch=6 val_loss is nominally lowest,
but the 0.097% spread is well within seed noise) -- never official VAL.

## 19. No MORAI claim (reaffirmed)

Every number in sections 12-15 is AV2-official-VAL or AV2-internal-VAL
only. Condition D applies a MORAI-*calibrated corruption profile* to
still-AV2 scenario data -- it is a diagnostic stress test of tolerance to
a more severe, anisotropic noise regime, **not** a MORAI evaluation.
`MORAI_ESTIMATOR_EVAL_V2` still does not exist. No claim is made that any
of these 3 checkpoints improves MORAI or competition-vehicle tracking
performance.

## 20. No runtime changes (reaffirmed)

No `kalmannet_core.py`/`KalmanNetFilter`/AB3DMOT/CenterPoint/ROS/prediction/
planner/occupancy-grid file was read for editing, let alone changed. This
task is training-and-evaluation-only, on an isolated worktree, on a branch
never merged.

## 21. Selected future training-policy decision

Of the 6 options this task posed, the evidence supports **(A) freeze this
10k checkpoint family and wait for a real MORAI-side evaluation/fine-tuning
bridge (`MORAI_ESTIMATOR_EVAL_V2`) before further AV2-only model or
hyperparameter work**, combined with a flag toward **(B) investigate
motion-composition/stationary-vs-moving sampling** as the most
evidence-grounded next AV2-side lever if pretraining work continues before
that bridge exists (section 22 rationale: AV2's own train-class
distribution mixes near-zero-velocity `STATIC`/`BACKGROUND` classes
(24,152 + 35,300 of 496,434 sequences) with fast `VEHICLE`/`BUS`/`MOTORCYCLIST`
classes in the same length-bucketed batches, a plausible per-batch gradient-
variance driver that this task's own scope explicitly forbade
investigating). **Not implemented** -- a decision, not an action, per this
task's explicit instruction.

## 22. Why this matters for the actual goal (MORAI competition, never a real vehicle)

Raised directly during this task and worth recording here: these AV2
checkpoints exist purely as large-scale, diverse **generic-robustness
pretraining** -- MORAI's own recorded KalmanNet data to date (a single
static scene, ~1,000-1,400 sequences, see `docs/perception/kalmannet_dense_baseline_v1.md`
and prior `docs/agent/STATUS.md` KalmanNet entries) is far too small/
homogeneous to train a data-hungry learned estimator from scratch. This
task's own 3-seed evidence (sections 12-17) says these checkpoints are
numerically safe and internally consistent -- it says nothing about
whether they help on the K-City competition map. **The concrete next step
for the actual goal is a real MORAI-recorded dataset** (GT actor
trajectories + ego GT + TF, via the existing `ad_morai_dataset_capture`/
`ad_morai_dataset_export_kalmannet`/`ad_morai_dataset_attach_kalmannet_measurements`
pipeline, ideally >=1 scenario+seed group with real detector-attached
measurements) used to build the still-missing `MORAI_ESTIMATOR_EVAL_V2`
and/or fine-tune one of these 3 frozen checkpoints -- not further AV2-only
seed/hyperparameter work.

## 23. Files (results artifacts)

`tools/kalmannet_training/av2_10k_multiseed_results/{summary.json,
seed{0,1,2}_freeze.json, seed{0,1,2}_official_val_report.json,
seed0_train.log, seed{1,2}_history.csv}` (148 KB total; no checkpoints, no
resume binaries, no AV2 data, no long per-batch logs committed, per this
task's explicit instruction).

## Files

`tools/kalmannet_training/{trainer_core.py,batched_trainer.py}`
(per-epoch STATE-B/C counters on `EpochRecord`),
`tools/kalmannet_training/resume_state.py` (generic `extra_counters` bag),
`tools/kalmannet_training/train_kalmannet_batched.py` (health counters
surfaced in checkpoint manifest / summary / per-epoch log / history CSV),
`tools/kalmannet_training/{seed_artifacts.py (new),multi_seed.py}`
(per-seed path generation + numerical-health aggregation),
`tools/kalmannet_training/test_{resume_state,seed_artifacts (new),
multi_seed,train_kalmannet_batched_cli_resume}.py`,
`tools/kalmannet_training/av2_10k_multiseed_results/` (new, committed
results: `summary.json`, per-seed freeze manifests, per-seed official-VAL
reports, seed 0's original train log, seed 1/2 history CSVs -- 148 KB
total), `docs/agent/STATUS.md`, this file. No `kalmannet_core.py`/
`KalmanNetFilter`/AB3DMOT/CenterPoint/ROS/prediction/planner/
occupancy-grid file changed. **Full test suite: 245/245 pass**
(`py_compile`/`pyflakes`/`git diff --check` all clean).
