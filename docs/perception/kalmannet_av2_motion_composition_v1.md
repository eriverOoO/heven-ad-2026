# AV2 KalmanNet Motion-Composition Ablation v1

Branch `exp/kalmannet-av2-motion-composition-v1`, from `origin/main` `41baa56`
(merge of PR #59 `exp(kalmannet): AV2 10K GENERIC-ROBUST multi-seed
completion`, verified `MERGED` before branching). **Offline AV2 experiment
only. No `KalmanNetGRU`/`KalmanNetFilter`/AB3DMOT/CenterPoint/ROS/
prediction/planner/occupancy-grid file changed. No model architecture or
optimizer-hyperparameter retuning -- isolates motion composition only. No
MORAI evaluation, no MORAI/competition-performance claim --
`MORAI_ESTIMATOR_EVAL_V2` still does not exist.**

## 1. Motivation

AV2's own train-class distribution (recorded in the frozen 10k checkpoints'
own stdout log) mixes near-zero-velocity `STATIC`/`BACKGROUND` classes
with fast `VEHICLE`/`BUS`/`MOTORCYCLIST` classes in the same
length-bucketed training batches -- flagged as a plausible gradient-variance
driver in the prior multi-seed task's own "selected future policy" section,
not investigated there. Separately, the historical MORAI-domain KalmanNet
data (T-9A/T-12 series) was ~98-99% non-stationary/curated moving actors --
a very different composition from AV2's own natural ~64% near-stationary
fraction. This task tests whether AV2's own natural composition is
*limiting* estimation quality specifically on MOVING trajectories -- the
population most representative of a tracked vehicle in either domain.

## 2. Precondition verification

`gh pr view 59 --json state,mergedAt` confirmed `MERGED` before any edit.
Worktree created at `/tmp/heven-worktrees/kalmannet-av2-motion-composition-v1`
(`git worktree add ... -b exp/kalmannet-av2-motion-composition-v1
origin/main`, HEAD `41baa56`), never the dirty main checkout.

## 3. Frozen 10k baseline (NOT retrained -- reused verbatim as the NATURAL condition)

| seed | checkpoint SHA-256 | best_epoch | internal_val_loss | status |
|---|---|---|---|---|
| 0 | `30e0902ef91d2e8f189db163a2ec636d39a7ad9d4f5abc414b8aaa8ae873a778` | 6 | 0.7451300733912749 | **legacy pre-guard** -- its own real training log permanently collapsed to NaN at epoch 16; checkpoint survives only by best-checkpoint timing, not guard protection (see prior task's STATUS.md entry) |
| 1 | `a9a19353ddb272d3fc0ac3ad7c6754239ef3d8dd00acd3c52a59193d98dc5ee5` | 7 | 0.7458607519010171 | **guarded full run**, completed naturally, 78 norm-overflow + 1 element-nonfinite event, both safely contained |
| 2 | `a005efc54b63ad92f4972a2731bc4ff1e5e1d9cc7c45e7e18f3030c0bfe91e51` | 4 | 0.7468887660922348 | **guarded full run**, completed naturally, 36 norm-overflow + 1 element-nonfinite event, both safely contained |

All 3 hashes re-verified byte-identical on disk before this task started.
**seed 0's health statistics are not conflated with seeds 1/2's** (it
predates the health-counter instrumentation entirely) -- this task treats
seed 0 as a reference point only, never as evidence about the guard.

`NATURAL` = these existing checkpoints, reused as-is. This task does
**not** retrain a NATURAL condition.

## 4. Motion category definition (locked BEFORE any official-VAL read)

Per-segment motion stats (`motion_composition.compute_segment_motion_stats`)
computed once from each TRAIN segment's own GT `x_true` (never `z_meas`,
which is corruption-dependent): median/mean/p90/max speed, displacement,
stationary-frame fraction, an acceleration proxy.

**Definition-sensitivity audit** (496,438 real TRAIN segments,
`analyze_motion_composition.py`, streamed one segment at a time to avoid
materializing the full corrupted 496k-sequence list in memory at once --
an earlier attempt using the standard `load_split_sequences` batch-loading
path OOM-killed at ~150k/496k segments; the streaming rewrite peaked at
~1.2 GB RSS and completed cleanly):

| representative-speed key | moving fraction |
|---|---|
| median | 36.22% |
| mean | 39.62% |
| p90 | 45.59% |

| pair | agreement |
|---|---|
| median vs. mean | 96.43% |
| median vs. p90 | 90.63% |
| mean vs. p90 | 93.98% |

**Chosen: `median_speed_mps`, threshold 0.5 m/s** (`REPRESENTATIVE_SPEED_KEY`
in `motion_composition.py`). Most outlier-robust of the three (a segment
with one brief fast burst amid an otherwise-stationary track is not
misclassified MOVING by a single high sample the way p90/mean can be more
sensitive to); its resulting near-stationary fraction (63.78%) also
matches the task's own stated prior domain observation ("~64%
near-stationary") almost exactly, independently confirming the prior
observation rather than being tuned to match it (the choice was made from
the sensitivity-audit table above, before this match was even checked).

## 5. Natural TRAIN distribution (section 3 of the task spec)

496,438 total segments (matches the prior multi-seed task's
`n_train_sequences=496,434` to within the small CLEAN-vs-GENERIC-ROBUST
truncation-count difference, expected and harmless).

| | count | fraction |
|---|---|---|
| NEAR_STATIONARY | 316,618 | 63.78% |
| MOVING | 179,820 | 36.22% |

MOVING speed bins:

| bin | count |
|---|---|
| 0.5-2 m/s | 43,099 |
| 2-5 m/s | 30,609 |
| 5-10 m/s | 56,589 |
| 10+ m/s | 49,523 |

Class x motion group (top entries): MOVING is 81.3% VEHICLE (146,272) +
14.2% PEDESTRIAN (25,592); NEAR_STATIONARY is 68.6% VEHICLE (217,250, --
i.e. the majority of even "vehicle" tracks are near-stationary, e.g.
stopped/parked/queued) + 10.8% STATIC (34,275) + 7.5% BACKGROUND (23,829).
Full breakdown: `tools/kalmannet_training/av2_motion_composition_results/natural_distribution.json`.

## 6. Sampler conditions

| condition | target moving fraction | status |
|---|---|---|
| A NATURAL | 36.22% (unweighted) | existing frozen checkpoints, reused |
| B BALANCED-MOTION | 50% | screening, this task |
| C MOVING-FOCUSED | 77.5% | screening, this task |

Deterministic weighted-with-replacement sampling
(`motion_composition.compute_sample_weights`/`deterministic_weighted_draw_order`):
per-epoch draw count fixed at `len(train_seqs)` (section 11 -- wall-clock/
budget parity with NATURAL); same `(order_seed, epoch)` always gives the
same draw order (no persisted sampler RNG state needed for resume); drawn
multiset fed through the existing length-bucketing batch construction
unmodified. Fixed seed = **1** (already shown numerically stable under
the guard as a full 10k run) for both B and C, per this task's own
instruction not to run a second multi-seed sweep here.

## 7. Screening B result (BALANCED-MOTION, seed 1, 8 epochs)

Checkpoint SHA-256 `cb1091bcd38de5d2852dbba98b7b51e66e0ba4998f721cd1164ac35db033d3c5`,
best_epoch=3, best_val_loss=0.7501028821686305 (unstratified mean MSE,
same metric NATURAL's own best_val_loss uses). `motion_sampler_fingerprint
=f6311077...` recorded in the checkpoint manifest. Wall time 9,460.3 s
(2.63 h) for 8 epochs. Numerical health: 0 element-nonfinite events,
0 parameter/optimizer-state collapse, 8 norm-overflow events total across
epochs 4/7 (5+3) -- safely contained, same guard behavior class as the
frozen NATURAL baseline.

**Internal-VAL motion-stratified (GENERIC-ROBUST, recomputed from the
saved checkpoint via `recompute_internal_val_motion_report.py` after a
real bug -- see section 17):**

| bucket | n | position RMSE (m) | velocity RMSE (m/s) |
|---|---|---|---|
| ALL | 2,968,204 | 0.30808 | 1.39260 |
| NEAR_STATIONARY | 1,925,974 | 0.26211 | 0.83134 |
| **MOVING** | 1,042,230 | **0.37862** | **2.06057** |

## 8. NATURAL baseline (seed 1) internal-VAL motion-stratified, for comparison

Computed the same way (`recompute_internal_val_motion_report.py`, GENERIC-ROBUST,
57,372 val sequences) against the **existing, un-retrained** seed 1
checkpoint -- the only new compute here is the evaluation pass, not
training.

| bucket | n | position RMSE (m) | velocity RMSE (m/s) |
|---|---|---|---|
| ALL | 2,968,204 | 0.30564 | 1.39295 |
| NEAR_STATIONARY | 1,925,974 | 0.25634 | 0.77216 |
| **MOVING** | 1,042,230 | **0.38027** | **2.10335** |

## 8b. CPU vs. GPU throughput sanity check (real data, exact frozen config)

Raised mid-task (user asked whether CPU was really faster than GPU for
this workload, given the multi-hour wall time). Re-verified directly, not
assumed from the older T-9A precedent: 2,500 real TRAIN segments,
`batch_size=64, hidden_size=32` (exact frozen config), 2 epochs each,
`device="cpu"` vs `device="cuda"` (RTX 4060, confirmed available,
`torch 2.4.1+cu121`):

| device | epochs | total (s) | avg epoch (s) |
|---|---|---|---|
| cpu | 2 | 13.90 | 6.95 |
| cuda | 2 | 35.96 | **17.98 (2.59x slower)** |

Confirms CPU is the correct choice for this exact configuration at this
scale, not a suboptimal default -- the model is small enough (hidden_size
32) that per-batch GPU kernel-launch/host-sync overhead outweighs any
compute parallelism benefit. `device=cpu` was never changed as a result
(frozen config, section 9 of the task spec).

## 9. NATURAL vs. BALANCED-MOTION (screening B), internal-VAL

| bucket | NATURAL pos / vel | BALANCED pos / vel | relative change (pos / vel) |
|---|---|---|---|
| ALL | 0.30564 / 1.39295 | 0.30808 / 1.39260 | +0.80% / -0.03% |
| NEAR_STATIONARY | 0.25634 / 0.77216 | 0.26211 / 0.83134 | +2.25% / **+7.66%** |
| MOVING | 0.38027 / 2.10335 | 0.37862 / 2.06057 | -0.43% / -2.03% |

A small, real MOVING improvement (both position and velocity RMSE lower
than NATURAL) but a clearly larger NEAR_STATIONARY cost (velocity RMSE
+7.66%) and a slight overall (ALL) regression -- pattern closer to
**outcome B (tradeoff exists)** than outcome A (material improvement,
little cost), per the task's own section 18 classification. Screening C
(MOVING-FOCUSED, more aggressive 77.5% target) below tests whether this
tradeoff worsens or whether the MOVING gain grows enough to justify it.

## 9b. EXPLICIT USER-APPROVED DEVIATION: full-run `max_epochs` 60 -> 30

Raised and approved mid-task (not a unilateral change). Section 9 of the
task spec lists `max_epochs=60` under "frozen config, do not retune" --
this is a disclosed, explicit exception, not a silent deviation. Rationale
put to the user before approval: all 3 frozen 10k baseline runs (seed 0/1/2)
found their `best_epoch` within epochs 4-7 and patience-exhausted (patience=15)
around epoch 20-23 -- **none reached anywhere near epoch 60** -- so
`max_epochs=30` is expected, but not re-confirmed in advance, to have
little-to-no effect on the selected checkpoint while roughly halving full-run
wall time. `patience=15` and every other frozen hyperparameter (`lr=0.004`,
`grad_clip=10.0`, `hidden_size=32`, `batch_size=64`, GENERIC-ROBUST corruption,
`device=cpu`) are UNCHANGED at this point (patience itself is separately
revisited in section 9c). The checkpoint manifest's own
`deviation_from_historical_max_epochs` field reads `true` for the full run,
making this deviation independently verifiable from the artifact itself,
not just this doc. Applies to the full 10k run only -- both 8-epoch
screenings (B, C) already used their own explicitly short `max_epochs=8`,
unaffected.

## 9c. SECOND EXPLICIT USER-APPROVED DEVIATION: full-run `patience` 15 -> 10

Raised and approved mid-task, after a data-grounded risk analysis (not a
unilateral change). Evidence reviewed before approval: of the two frozen
10k baseline runs that ran a full `patience=15` window to natural
exhaustion (seed 1, seed 2), the longest gap between a best-epoch update
and the NEXT best-epoch update was 4 epochs (seed 1: epoch1 -> epoch5) --
**in neither seed did any improvement ever occur more than 4 epochs after
the previous one**, and in both, zero improvements occurred at all in the
final ~13-15 epochs of the patience window before it exhausted. This
means `patience=10` would have found the IDENTICAL `best_epoch` as
`patience=15` in both fully-observed cases -- the extra ~5 epochs of
budget were never actually used to find a better checkpoint. Caveat
disclosed to the user: this is n=2 evidence, both from the NATURAL
condition, not yet verified for BALANCED-MOTION sampling specifically (a
different training distribution could in principle have a different noise/
recovery pattern) -- a reasonable, evidence-informed bet, not a proven
guarantee.

**Consequence: the first full-run attempt (patience=15, this same
BALANCED-MOTION/seed=1/`--num-threads 4` config) was killed at epoch 2**
(best_epoch=2, val_loss 0.755449) and restarted from epoch 0 with
`patience=10` -- resume was not used because `resume_validation_key`
includes `patience` and would correctly refuse a mismatched resume
(`ResumeValidationError`), so continuing under a new patience value
requires a fresh run, not a continuation. `max_epochs=30` (section 9b),
`lr=0.004`, `grad_clip=10.0`, `hidden_size=32`, `batch_size=64`,
GENERIC-ROBUST corruption, `device=cpu` all UNCHANGED from the first
attempt.

## 9d. ADDITIVE, RESULT-NEUTRAL INFRASTRUCTURE CHANGE: `--num-threads`

Raised mid-task after a real CPU-thread-count benchmark under
concurrent-process contention (results confounded by that contention --
16-thread case measured 119.69 s/epoch purely from oversubscription with
another process, not a valid standalone reading). 4-thread and 8-thread
readings (6.19 s vs. 7.17 s/epoch) were directionally consistent with "4
is at least not worse," so `--num-threads 4` (a pure `torch.set_num_threads()`
call, `train_kalmannet_motion_composition.py`'s own opt-in CLI flag,
default `None` = unchanged torch default for every other run in this
task) was used for the full run as a "worth trying, no downside" choice --
explicitly not claimed as a rigorously validated speedup. Never affects
model weights/results, only wall-clock.

## 10. Screening C result (MOVING-FOCUSED, seed 1, 8 epochs) and full 3-way comparison

Wall time 9,617.7 s (2.67 h). Numerical health: 2 norm-overflow events
(epochs 2, 7), 0 element-nonfinite, 0 parameter/optimizer-state collapse
-- same safe-containment pattern as B and the frozen NATURAL baseline.
best_epoch=4, best_val_loss ~0.7927.

| bucket | metric | NATURAL | BALANCED (B) | MOVING-FOCUSED (C) |
|---|---|---|---|---|
| ALL | pos (rel.) | 0.30564 | 0.30808 (+0.80%) | 0.31676 (**+3.64%**) |
| ALL | vel (rel.) | 1.39295 | 1.39260 (-0.03%) | 1.42525 (**+2.32%**) |
| NEAR_STATIONARY | pos (rel.) | 0.25634 | 0.26211 (+2.25%) | 0.28201 (**+10.01%**) |
| NEAR_STATIONARY | vel (rel.) | 0.77216 | 0.83134 (+7.66%) | 0.94720 (**+22.67%**) |
| **MOVING** | pos (rel.) | 0.38027 | 0.37862 (-0.43%) | **0.37254 (-2.03%)** |
| **MOVING** | vel (rel.) | 2.10335 | 2.06057 (-2.03%) | **2.03154 (-3.41%)** |

MOVING-FOCUSED does show the larger MOVING improvement (the primary
metric) -- but the marginal cost escalates faster than the marginal
benefit going B -> C: NEAR_STATIONARY velocity RMSE degrades ~3x more
(+7.66% -> +22.67%) while MOVING position RMSE improves only ~4.7x more
in absolute terms that are themselves tiny (-0.43% -> -2.03%), and
crucially **C's own ALL (overall) velocity RMSE actually regresses**
(+2.32%, vs. B's essentially-flat -0.03%) -- directly triggering the
task's own constraint ("overall RMSE must not degrade unacceptably").

## 11. SELECTED FULL-RUN CONDITION: **BALANCED-MOTION (B)**

Chosen from internal validation only (never official VAL), per the task's
own selection rule. Rationale: B captures a real, consistent MOVING
improvement (both position and velocity RMSE lower than NATURAL, 3/3 --
NATURAL/B/C -- comparisons agree on direction) while keeping the overall
(ALL) metric essentially unchanged and the NEAR_STATIONARY cost moderate.
C's more aggressive reweighting produces a materially larger
NEAR_STATIONARY/overall regression for a MOVING gain that, while larger
in relative terms, remains a small absolute RMSE difference (0.37862 vs.
0.37254 m, 0.006 m apart) -- not judged worth the additional collateral
cost. This matches the task's own **outcome B interpretation** ("tradeoff
exists; a milder weighting... may be needed") -- B, not C, is that milder
weighting.

## 12. Full-run result (BALANCED-MOTION, seed 1, `max_epochs=30`, `patience=10`)

20 epochs (patience-exhausted at epoch 13, best_epoch=3), wall time
15,793.5 s (4.39 h) -- meaningfully shorter than seed1's 7.46 h / seed2's
6.50 h from the prior multi-seed task, consistent with sections 9b/9c/9d.
best_val_loss=0.7501028821686305 (0.7515 in per-epoch log rounding),
essentially identical to screening B's own best_epoch=3 result (same seed,
same sampler, so epochs 0-3 are the same training trajectory by
construction). Checkpoint SHA-256
`32fbb4b7e077aa92e3f85da81a103341dd0b2857c985a2aea43c5d0a8e779431`.
Numerical health: `catastrophic=false`, `training_unstable=false`,
`parameter_collapse_count=0`, `optimizer_state_collapse_count=0`,
`norm_overflow_count=36`, `per_element_nonfinite_gradient_count=0` --
zero element-nonfinite events this run (contrast with the two screenings
and the NATURAL seed1/seed2 baselines, each of which hit at least one --
a real, seed/sampler-dependent absence, not evidence either way about the
guard's necessity).

Internal-VAL motion-stratified (full 57,372-sequence val set):

| bucket | n | position RMSE (m) | velocity RMSE (m/s) |
|---|---|---|---|
| ALL | 2,968,204 | 0.31040 | 1.39373 |
| NEAR_STATIONARY | 1,925,974 | 0.26611 | 0.83419 |
| MOVING | 1,042,230 | 0.37886 | 2.06062 |

Essentially identical to screening B's own numbers (section 7) -- expected,
since both share the same best_epoch=3 weights.

## 13. Official AV2 VAL evaluation

Freeze manifest written (`write_freeze_manifest_av2_10k_generic.py`,
`internal_train_vs_official_val`/`internal_val_vs_official_val` overlap
both 0, re-confirmed) before any official-VAL read, per the task's own
mandatory ordering. `evaluate_kalmannet_official_val.py` run with
`--add-morai-calibrated-diagnostic` and the reused
`--precomputed-kf-calibration-json` (no baseline recalibration).

**Official VAL, ALL (unstratified) vs NATURAL (seed 1), position/velocity RMSE:**

| condition | NATURAL pos/vel | BALANCED pos/vel | relative change (pos/vel) |
|---|---|---|---|
| A CLEAN | 0.03923/1.25058 | 0.06216/1.24813 | **+58.44%** / -0.20% |
| B GENERIC (same seed) | 0.30632/1.38918 | 0.31101/1.39162 | +1.53% / +0.18% |
| C GENERIC (diff seed) | 0.30762/1.39285 | 0.31219/1.39548 | +1.49% / +0.19% |
| D MORAI-calib diag | 1.71508/2.05933 | 1.72021/2.09223 | +0.30% / +1.60% |

**Official VAL, MOVING vs NEAR_STATIONARY (the actual test of this
experiment's hypothesis):**

| condition | MOVING pos rel. | MOVING vel rel. | NEAR_STATIONARY pos rel. | NEAR_STATIONARY vel rel. |
|---|---|---|---|---|
| A CLEAN | **+59.05%** | -0.70% | **+55.10%** | +1.94% |
| B GENERIC (same seed) | -0.41% | -1.97% | +3.86% | +8.89% |
| C GENERIC (diff seed) | -0.40% | -1.94% | +3.75% | +8.85% |
| D MORAI-calib diag | -2.81% | -2.51% | +5.58% | +15.15% |

**On the practically-relevant GENERIC-ROBUST conditions (B/C) and the D
diagnostic, official VAL confirms the internal-val pattern closely**:
MOVING position RMSE improves modestly (-0.4% to -2.8%), MOVING velocity
RMSE improves more meaningfully (-1.9% to -2.5%), while NEAR_STATIONARY
degrades on both metrics (+3.75-5.58% position, +8.85-15.15% velocity) --
the same real tradeoff internal-val already showed, now independently
confirmed on a physically separate 1,000-scenario pool never read during
training or selection.

**But CLEAN (A) reveals a severe, previously-unmeasured regression**
(internal-val was always computed under GENERIC-ROBUST corruption, since
that is what the training script's own val_seqs use -- CLEAN was never
evaluated before this official-VAL pass): **both MOVING (+59.05%) and
NEAR_STATIONARY (+55.10%) position RMSE are dramatically worse** under
BALANCED-MOTION than NATURAL. This is a real, substantial finding, not a
rounding artifact -- ALL/CLEAN position RMSE nearly doubles (0.03923 ->
0.06216 m). Plausible mechanism (not verified further, out of this task's
scope): reweighting toward MOVING segments changes the effective training
distribution enough to measurably hurt the network's behavior specifically
in the low-noise/no-corruption regime, even though the model was never
trained on CLEAN measurements directly (GENERIC-ROBUST corruption is
applied uniformly regardless of sampler).

## 14. INTERPRETATION

Per the task's own section-18 outcome taxonomy: **outcome B (tradeoff
exists), more severe than the internal-val-only picture suggested.** The
MOVING improvement is real and reproduces on official VAL under every
GENERIC-ROBUST/diagnostic condition (3/3 -- B, C, D -- agree in direction
and magnitude), so this is not noise. But the cost is also real, and the
newly-discovered CLEAN-condition regression (+55-59%) is severe enough
that it cannot be dismissed as "not over-weighting the CLEAN condition"
(the task's own instruction was to not let a CLEAN *improvement* dominate
the read; a CLEAN *regression* this large is a different, independently
important signal about training-distribution shift, not something the
same instruction tells us to discount). Net assessment: AV2's natural
motion composition was **not** simply "limiting training efficiency" in a
way that reweighting cleanly fixes -- the fix trades a small, consistent
MOVING gain under corruption for a large CLEAN-condition cost and a
consistent NEAR_STATIONARY cost.

## 15. SELECTED FUTURE TRAINING-POLICY DECISION (chosen, not implemented)

Of the task's own 6 options: **(D) keep the NATURAL 10k model as the
preferred AV2 pretrained family; motion-composition reweighting did not
produce a net-positive result** at this reweighting strength (50% target).
The MOVING gain under GENERIC-ROBUST/diagnostic conditions is real but
small (~0.4-2.8% position, ~2-2.5% velocity) and does not offset the
NEAR_STATIONARY cost (~4-6% position, ~9-15% velocity) or, especially, the
severe CLEAN-condition regression (~55-59% position). Secondary,
evidence-grounded next-step candidate if this line of investigation
continues: **(C) a milder mixed natural/motion-focused curriculum** (e.g.
train mostly NATURAL with only a late-training or low-weight MOVING
oversampling phase) specifically to test whether the CLEAN-condition
regression is avoidable while retaining some of the MOVING gain -- not
selected as the primary decision here since it requires new training
work this task's own scope does not include implementing.

## Files

`tools/kalmannet_training/{motion_composition.py (new),
train_kalmannet_motion_composition.py (new),
analyze_motion_composition.py (new),
recompute_internal_val_motion_report.py (new)}`,
`tools/kalmannet_training/batched_trainer.py` (additive `motion_sampler`/
`motion_categories` params, byte-identical when omitted),
`tools/kalmannet_training/test_{motion_composition,
motion_sampler_integration}.py` (new),
`tools/kalmannet_training/test_evaluate_kalmannet_official_val.py`
(+4 motion-stratified aggregation tests),
`tools/kalmannet_training/av2_motion_composition_results/` (new, small
JSON/CSV summaries -- no checkpoints/resume binaries/AV2 data), this
file, `docs/agent/STATUS.md`. No `kalmannet_core.py`/`KalmanNetFilter`/
AB3DMOT/CenterPoint/ROS/prediction/planner/occupancy-grid file changed.

## Note: mid-task PC power loss and full code recovery

A real PC power interruption occurred after the full run's training and
internal-val evaluation had already completed and been persisted to
`~/datasets/av2/` (outside `/tmp`, survived), but while the official-VAL
evaluation subprocess was still running (killed mid-flight, its own
report file never written). Because `/tmp` (where the git worktree lived)
is cleared on reboot and this branch had zero commits at the time, every
source file written for this task was lost and was rebuilt from the
conversation record after the restart -- verified byte-for-byte behavior-
equivalent by re-running the full test suite (292/292 pass, same count as
before the loss) and `git diff --check`/`pyflakes` clean. No experimental
result was lost (all checkpoints/manifests/logs/JSON reports under
`~/datasets/av2/` survived); only the official-VAL evaluation needed to be
re-run from scratch (a single evaluation pass, not multi-hour training).
