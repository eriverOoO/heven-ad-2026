# KalmanNet Physically-Consistent Variable-dt Augmentation v1

Branch `exp/kalmannet-av2-variable-dt-v1`, from `origin/main` `61b03894`
(merge of PR #60 `exp(kalmannet): test AV2 motion-focused sampling`,
verified `MERGED` before branching -- PR #60 was NOT stacked into this
work, per instruction). **Offline AV2-only experiment. No
`kalmannet_core.py`/`KalmanNetFilter`/AB3DMOT/CenterPoint/ROS/prediction/
planner/occupancy-grid file changed. No KalmanNet architecture change --
`F_matrix(dt)`/`Q_matrix(dt)` already accept real variable dt, unchanged.
No MORAI evaluation, no MORAI/competition-performance/real-simulator-
transfer claim -- MORAI simulator access is currently unavailable.**

## 1. Motivation

The prior Motion-Composition Ablation v1 task (PR #60) concluded the
NATURAL 10k GENERIC-ROBUST KalmanNet remains the preferred AV2 pretrained
family. This task investigates a second, offline-only AV2/MORAI domain
gap: AV2 Motion Forecasting sequences are always exactly `dt=0.100 s`
(native 10 Hz), while historical MORAI-domain KalmanNet training data
(the small real-scene captures from the T-9A/T-12 series) has materially
variable dt (real ROS/replay timing jitter, not a clean fixed cadence).
This task tests whether training KalmanNet on physically-consistent
variable temporal intervals -- built ONLY from real AV2 samples via
temporal thinning, never interpolation -- improves robustness to
irregular sampling, without any architecture change.

## 2. Precondition / git

`gh pr view 60 --json state,mergedAt` confirmed `MERGED` before branching.
PR #60 explicitly does NOT need to be stacked (per instruction) -- this
branch starts from `origin/main` directly (`git fetch origin && git
checkout main && git pull --ff-only origin main`), which already contains
PR #60's merged content. Worktree created at
`/tmp/heven-worktrees/kalmannet-av2-variable-dt-v1` (isolated from the
main checkout's own unrelated dirty CRLF files, never touched).

## 3. Frozen preferred baseline (NOT retrained)

| seed | checkpoint SHA-256 | best_epoch | internal_val_loss |
|---|---|---|---|
| 0 | `30e0902ef91d2e8f189db163a2ec636d39a7ad9d4f5abc414b8aaa8ae873a778` | 6 | 0.7451300733912749 |
| 1 | `a9a19353ddb272d3fc0ac3ad7c6754239ef3d8dd00acd3c52a59193d98dc5ee5` | 7 | 0.7458607519010171 |
| 2 | `a005efc54b63ad92f4972a2731bc4ff1e5e1d9cc7c45e7e18f3030c0bfe91e51` | 4 | 0.7468887660922348 |

All 3 re-verified byte-identical on disk before this task started. This
is the **NATURAL 10k FIXED-DT** baseline for the eventual 10k comparison
(section 15-19) -- **not** the PR #60 motion-focused checkpoint, which is
diagnostic only and is never treated as a baseline anywhere in this task.

## 4. Fixed-dt assumption audit (section 5 of the task spec)

Searched the full offline training path (`kalmannet_sequences.py`,
`batched_kalmannet.py`, `batched_trainer.py`/`trainer_core.py`,
`evaluate_kalmannet.py`/`evaluate_kalmannet_official_val.py`,
`av2_motion_forecasting_adapter.py`, `av2_kalmannet_shard_loader.py`,
checkpoint manifest fields) for `0.1`/`10 Hz`/fixed-step-index literals.

**Result: no hardcoded `0.1`/`10 Hz`/fixed-cadence assumption found
anywhere in the training/loss/corruption/evaluation math.** `dt_s` is
already computed from real AV2 `timestamps_ns` deltas at export time
(`av2_motion_forecasting_adapter.py::clean_arrays`,
`dt_s[1:] = (stamps[1:] - stamps[:-1]) / 1e9`) -- it is exactly `0.1`
only because real AV2 scenarios are natively uniform 10 Hz, never because
the code assumes it. `KalmanNetFilter.F_matrix(dt)`/`Q_matrix(dt)`
(`kalmannet_core.py`, unmodified) already take the per-step `dt` value
from each sequence's own `dt` list. `batched_kalmannet.py`'s recursion
loops over `range(T)` using each row's own `dt[t]`, never a fixed step
count translated into elapsed time. The only place a fixed assumption
*could* have hidden was the historical `_truncate_to_first_measurement`
re-indexing (`kalmannet_sequences.py`) -- audited directly: it re-slices
`dt`/`frames`/`x_true`/`z_meas` generically by list position, never by an
assumed constant step, so it composes correctly with a thinned (variable-
dt) input sequence with zero change needed.

**One genuine, minor finding**: `checkpoint_utils.build_checkpoint_manifest`
and every existing freeze-manifest writer record `"sampler_version"` and
various dataset/model config fields, but had no field for a temporal-
augmentation policy before this task -- not a bug (nothing needed one
until now), addressed additively in section 17.

## 5. Temporal thinning design (physically consistent, no interpolation)

`variable_dt.py` (new, pure functions, no torch/ROS dependency):
`thin_sequence(seq, policy, augmentation_seed, segment_id)` selects a
STRICT SUBSET of a segment's own original frame indices (always
retaining index 0), recomputes `dt` for each retained transition as the
SUM of the original per-step `dt` values skipped in between (dt is
elapsed real time, so this sum is exact, not approximate -- matches the
task's own worked example: retaining `0,1,3,4,7` from a uniform
`dt=0.1` sequence gives real intervals `0.1, 0.2, 0.1, 0.3`, reproduced
exactly by `test_dt_recomputation_matches_task_example`). Every retained
state/measurement is the REAL AV2 sample at its original index -- `x_true`
and `z_meas` are indexed, never interpolated or synthesized.
`verify_thinned_sequence` (used both as a runtime assertion during
training-data construction and directly as a test) checks: strictly
increasing/no-duplicate retained indices, `dt>0` at every retained
transition, `dt` matches the resummed original deltas exactly, and every
retained state matches its claimed original index.

**Thinning vs. missing measurement (section 6) -- deliberately kept
distinct, never conflated.** A temporally thinned frame does not exist in
the output sequence at all -- `frames`/`dt`/`x_true`/`z_meas` are all
shorter by construction, and the removed original indices leave no trace
(no `None` placeholder, no `measurement_valid=False` flag). A missing
measurement (via GENERIC-ROBUST dropout/dropout-burst) keeps the timestep
-- the frame is still present, GT (`x_true`) still exists, only
`z_meas[t] is None`, driving the existing prediction-only transition.
**Corruption ordering (section 11)**: `segment_to_thinned_training_sequence`
enforces load CLEAN sequence (`segment.to_kalmannet_sequence(None)`) ->
`thin_sequence` -> `apply_corruption` on the THINNED array only (reusing
`apply_corruption` verbatim, unmodified -- it is a pure function of array
length + `segment_id`, agnostic to whether that length is the original or
a thinned one, so no second corruption implementation was written). This
guarantees a temporally-removed original frame can never later become a
"missing measurement" frame in the training sequence -- it was never a
candidate for dropout in the first place, since dropout is computed AFTER
thinning, over the thinned array's own indices only. Documented and
locked by `test_corruption_ordering`-class tests (section 21 below).

**Determinism (section 9)**: `thinning_seed(augmentation_seed, segment_id)`
uses the exact same SHA-256-truncation convention already established by
`av2_motion_forecasting_adapter._segment_corruption_seed` (first 4 digest
bytes mod `2**31-1`) -- never Python's `hash()`. Same
`(policy, augmentation_seed, segment_id)` always gives the same retained
indices/dt, independent of batch order, shard grouping, or worker
ordering (there is no batch-context input to the function at all).

**Data integrity (section 10)**: enforced by `verify_thinned_sequence`
(called on every non-fallback thinning during
`segment_to_thinned_training_sequence`) plus a deterministic fallback --
if the stochastic thinning process would retain fewer than
`policy.min_retained_frames` (default 5, mirrors the existing
`min_gt_samples` convention elsewhere), the ORIGINAL untouched sequence is
used for that specific segment instead (flagged `is_thinned=False`,
counted, never silently shipped as a too-short thinned sequence).
Canonical shards are never mutated -- augmentation happens entirely at
training load time, identical to how GENERIC-ROBUST corruption already
works.

## 6. MILD / STRONG dt distributions (declared BEFORE any screening/official-VAL run)

Declared as a generic, hand-chosen bounded distribution -- **not** derived
from the frozen MORAI TEST actors or any historical MORAI timing
statistic (task section 8). `skip_probs[k]` = probability of skipping `k`
original frames before the next retained one (k=0 -> dt stays at the
original per-step value).

| policy | skip_probs (k=0,1,2,3) | implied dt support (base dt=0.1s) |
|---|---|---|
| FIXED-DT | (1.0,) | 0.1s always (exact no-op) |
| MILD-VARIABLE-DT | (0.80, 0.15, 0.05) | 0.1s: 80% / 0.2s: 15% / 0.3s: 5% |
| STRONG-VARIABLE-DT | (0.55, 0.25, 0.15, 0.05) | 0.1s: 55% / 0.2s: 25% / 0.3s: 15% / 0.4s: 5% |

Matches the task's own qualitative description exactly ("0.1s dominant,
0.2s occasional, 0.3s rare" for MILD; "0.1/0.2/0.3/possibly 0.4s, more
aggressive" for STRONG). No interval larger than 0.4s in this first
experiment, per instruction.

## 7. 2k screening result (internal validation only)

Screened FIXED (baseline A, the frozen 2k pilot GENERIC-ROBUST checkpoint
`av2_stage1_pilot_generic_robust_bs64_lr004_seed0.pt`, re-evaluated, not
retrained) vs MILD (B) vs STRONG (C), all seed 1, on the Stage-1
2,000-scenario pilot dataset. Each checkpoint evaluated under all three dt
policies via `evaluate_with_dt_buckets` (`variable_dt.py`), **internal
validation split only** -- official AV2 VAL never read for selection
(task section 12/13).

| eval policy | A (FIXED-trained) | B (MILD-trained) | C (STRONG-trained) |
|---|---|---|---|
| fixed-dt  overall pos RMSE (m) | 0.3231 | **0.3195** | 0.3218 |
| mild-dt   overall pos RMSE (m) | 0.3602 | **0.3518** | 0.3550 |
| strong-dt overall pos RMSE (m) | 0.4440 | 0.4258 | **0.4248** |
| internal val loss (training)   | 0.7990 | **0.8288** | 0.8867 |
| norm-overflow events           | --     | 1      | 2      |
| thinning-applied fraction      | --     | 99.999% | 99.86% |

At 2k scale MILD improved every eval condition vs baseline A -- including
fixed-dt (-1.1%) -- and clearly beat STRONG on fixed+mild-dt eval and on
internal val loss, tying STRONG within 0.24% on strong-dt eval, with fewer
overflow events.

## 8. Selected policy: MILD-VARIABLE-DT (one full 10k run)

Per section 15's criteria: (1) variable-dt validation improvement -- yes
vs baseline; (2) fixed-0.1 preservation -- best of the three at 2k; (3)
numerical stability -- best (1 overflow); (4) data retention -- 99.999%.
STRONG rejected (weaker overall, higher internal loss). **STOP was not
warranted** -- MILD showed a real, consistent internal-val improvement in
the variable-dt regime at 2k. One full 10k MILD run launched (frozen
NATURAL 10k dataset + hyperparameters, only new axis = temporal thinning).

## 9. Full 10k MILD-VARIABLE-DT training run

`scaleup_v2` frozen 10k dataset (9k internal-TRAIN / 1k internal-VAL,
`dataset_manifest_sha256 ab0f9faa...`, `split_manifest_sha256 f4bb07b2...`
-- identical to the NATURAL 10k baseline family). One seed (1). Frozen
config: `hidden_size=32, batch_size=64, lr=0.004, max_epochs=60,
patience=15, grad_clip=10.0, loss_on_predict_only=True`, CPU,
GENERIC-ROBUST corruption. Thinning applied to 496,413 / 496,414 training
sequences (99.9998%; 1 too short to thin, 20 sequences n<2 dropped vs the
NATURAL baseline's 496,434).

Best epoch **15**, `val_loss_mse_xy_vxvy = 0.7973`, 31 epochs run
(early-stopped), 8.2 h wall time. Checkpoint SHA-256
`eb848a81fec1e71eb3907eefef4c48d4152f16f9f7d90d6d7d858e2645b72828`.
Freeze manifest (`av2_variable_dt_results/freeze_manifest_mild10k.json`)
written **before** any official AV2 VAL read.

## 10. Numerical health (task section 21)

| counter | 10k MILD-variable-dt | NATURAL 10k seed1 baseline |
|---|---|---|
| norm-overflow batches (all skipped) | **39** | 78 |
| per-element non-finite gradient batches | **0** | 1 |
| non-finite-gradient skips | 0 | 1 |
| parameter / optimizer-state collapse | 0 / 0 | 0 / 0 |
| large-but-finite gradient events | 9,432 | ~12,000 |
| training_collapsed / training_unstable | false / false | false / false |

9 epochs (11, 16, 17, 20, 22, 23, 26, 28, 30) hit at least one inf
gradient norm, every one safely skipped by the `nonfinite_guard`
three-state guard (`docs/perception/kalmannet_gradient_norm_overflow_v1.md`).
The best checkpoint (epoch 15) **predates** the worst instability window
(epochs 16-17, where val loss briefly rose to ~1.0 then recovered).
**Finding: temporal thinning at the MILD level did NOT increase numerical
instability** -- the MILD-variable-dt run had roughly half the
norm-overflow events and zero element-nonfinite events of the NATURAL
FIXED-DT baseline. Section 21's concern is not borne out for MILD.

## 11. Primary comparison -- internal validation dt-buckets (task section 14/19)

NATURAL 10k seed1 (FIXED-DT) vs 10k MILD-VARIABLE-DT, both evaluated
identically on the 1k internal-VAL split under each dt policy, overall +
per-transition-dt bucket position RMSE (m):

| eval condition | bucket | NATURAL | MILD-vardt | Δ pos | Δ vel |
|---|---|---|---|---|---|
| fixed  | overall | 0.3056 | 0.3087 | **+1.0%** | +1.2% |
| fixed  | 0.1s | 0.3028 | 0.3060 | +1.0% | +1.5% |
| mild   | overall | 0.3445 | 0.3439 | -0.2% | -1.0% |
| mild   | 0.1s | 0.3272 | 0.3276 | +0.1% | -0.6% |
| mild   | 0.2s | 0.3851 | 0.3820 | -0.8% | -1.9% |
| mild   | 0.3s | 0.4366 | 0.4302 | -1.5% | -5.2% |
| strong | overall | 0.4262 | 0.4176 | **-2.0%** | -4.4% |
| strong | 0.1s | 0.3815 | 0.3758 | -1.5% | -4.7% |
| strong | 0.2s | 0.4387 | 0.4298 | -2.0% | -5.4% |
| strong | 0.3s | 0.5041 | 0.4893 | -2.9% | -8.1% |
| strong | 0.4s+ | 0.5762 | 0.5576 | **-3.2%** | **-11.1%** |

**The improvement from temporal-thinning training scales monotonically
with transition dt** -- slightly worse at dt=0.1s (~+1%), progressively
better as dt grows, up to ~-3.2% position / ~-11% velocity RMSE at
dt >= 0.4s. Net effect is condition-dependent: -2.0% position under
strong-thinning eval, +1.0% under fixed-dt eval, a wash under
mild-thinning eval. Matched vs missing splits move together (both improve
~-2% under strong eval), so the effect is not confined to the
predict-only regime.

<!-- OFFICIAL_VAL_AND_KF_PLACEHOLDER -->

## Files

`tools/kalmannet_training/variable_dt.py` (new -- thinning core, dt-bucket
evaluator, `load_split_sequences_with_thinning`),
`tools/kalmannet_training/train_kalmannet_variable_dt.py` (new -- training
entry point with `--dt-policy`), `tools/kalmannet_training/test_variable_dt.py`
(new, 32 tests), `tools/kalmannet_training/av2_variable_dt_results/` (new,
small JSON/CSV summaries), this file, `docs/agent/STATUS.md`. No
`kalmannet_core.py`/`KalmanNetFilter`/AB3DMOT/CenterPoint/ROS/prediction/
planner/occupancy-grid file changed.
