# KalmanNet Mixed Fixed / Variable-dt Curriculum v1

Branch `exp/kalmannet-av2-dt-curriculum-v1`, from `origin/main` `5460027a`
(merge of PR #61 `exp(kalmannet): test variable-dt AV2 training`, verified
`MERGED` before branching). **Offline AV2-only experiment. No
`kalmannet_core.py`/`KalmanNetFilter`/`F_matrix`/`Q_matrix`/AB3DMOT/
CenterPoint/ROS/prediction/planner/occupancy-grid file changed. No
KalmanNet architecture or optimizer-hyperparameter change. No MORAI
evaluation, no MORAI/competition/real-simulator claim -- MORAI simulator
access is unavailable.**

## 1. Baseline tradeoff (from PR #61, not re-derived here)

Variable-dt v1 found **OUTCOME B**: MILD physically-consistent temporal
thinning improves KalmanNet on heavier-thinned AV2 trajectories at a
small nominal-condition cost. Official AV2 VAL (1,000 held-out
`val`-split scenarios), overall position / velocity RMSE:

| eval condition | NATURAL 10k FIXED-DT | MILD var-dt 10k | Δ pos / Δ vel |
|---|---|---|---|
| A. fixed 0.1s    | 0.3063 / 1.389 | 0.3092 / 1.408 | **+0.9% / +1.4%** |
| B. MILD var-dt   | 0.3470 / 1.464 | 0.3468 / 1.453 | -0.1% / -0.8% |
| C. STRONG var-dt | 0.4227 / 1.620 | 0.4141 / 1.550 | **-2.0% / -4.3%** |

dt-bucket: the benefit grows monotonically with transition dt, up to
-3.2% pos / -11% vel at dt >= 0.4s. LinearCVKF degrades at essentially
the same rate as either KalmanNet across dt, so this is **not** a
KalmanNet learned-gain failure.

**Frozen baseline checkpoints (NOT retrained):**

| model | checkpoint SHA-256 | best epoch | internal val loss |
|---|---|---|---|
| A. NATURAL 10k FIXED-DT (seed1) | `a9a19353ddb272d3fc0ac3ad7c6754239ef3d8dd00acd3c52a59193d98dc5ee5` | 7 | 0.7459 |
| B. MILD var-dt 10k (seed1) | `eb848a81fec1e71eb3907eefef4c48d4152f16f9f7d90d6d7d858e2645b72828` | 15 | 0.7973 |

Both trained on `dataset_manifest_sha256 ab0f9faa…`,
`split_manifest_sha256 f4bb07b2…` (scaleup_v2 9k/1k internal split).
Their official-VAL + internal-VAL dt-bucket reports and the AV2-tuned
LinearCVKF dt-bucket report are reused verbatim from PR #61.

## 2. Mixed / curriculum policy

**MIXED-50** (the section-4 preferred primary candidate). For each
training segment a deterministic per-segment Bernoulli(0.5) draw
(`dt_curriculum.segment_is_thinned_in_mix`, SHA-256 over
`f"{policy_seed}:mix:{segment_id}"` with a distinct `:mix:` infix so it
is independent of the thinning-internal RNG and the corruption RNG)
decides whether that segment is used as its **original FIXED sequence**
or its **frozen MILD temporal thinning**. One static assignment, decided
once from stable identifiers -- resume-safe by construction, no per-epoch
RNG state. Each segment appears exactly once, in exactly one form; **no
physical duplicate dataset**.

`dt_curriculum.py` reuses `variable_dt.py` entirely --
`segment_to_thinned_training_sequence` is called with either `FIXED_DT`
(exact no-op) or the unchanged frozen `MILD_VARIABLE_DT`
(`skip_probs=(0.80, 0.15, 0.05)`). No second thinning implementation; no
change to thinning probabilities / dt calculation / retained-index
semantics.

**Curriculum (per-epoch ramp, the optional section-5 candidate): NOT
pursued.** `batched_trainer.train_one_run_batched` consumes a single
precomputed `train_seqs` list and resamples indices with replacement each
epoch -- it has no hook to re-thin sequence *content* per epoch. An
epoch-ramped schedule would require re-thinning every epoch (or carrying
both forms of every sequence in memory), neither of which is the minimal
change this task asks for. MIXED-50 already smooths the tradeoff at 2k (section 3), so per section 5
("If MIXED-50 already dominates the tradeoff, skip curriculum entirely")
no curriculum candidate was trained.

## 3. 2k screening result (internal validation only)

Stage-1 2,000-scenario pilot dataset, seed 1, frozen config
(`bs=64/lr=0.004/grad_clip=10.0/max_epochs=60/patience=15`,
GENERIC-ROBUST, CPU). MIXED-50 realised a 44,465-fixed / 44,410-thinned
train split (fraction 0.4997). Best epoch 21, `val_loss 0.8245`. Each
checkpoint evaluated on the internal-VAL split under all three dt
policies (`evaluate_with_dt_buckets`), overall position RMSE (m). The
FIXED / MILD / STRONG rows are the frozen PR #61 screening checkpoints,
re-evaluated, not retrained.

| eval | A FIXED baseline | B MILD | C STRONG | **D MIXED-50** |
|---|---|---|---|---|
| fixed  | 0.3231 | 0.3195 | 0.3218 | **0.3177** |
| mild   | 0.3602 | 0.3518 | 0.3550 | 0.3523 |
| strong | 0.4440 | 0.4258 | 0.4248 | 0.4297 |
| best internal val loss | 0.7990 | 0.8288 | 0.8867 | **0.8245** |
| norm-overflow events | -- | 1 | 2 | **0** |

**MIXED-50 improves the FIXED baseline on all three conditions**
(fixed -1.7%, mild -2.2%, strong -3.2%), has the **best** fixed-dt of any
variant (below even pure MILD), the **lowest** internal val loss of the
variable-dt family, and the fewest gradient-overflow events. It retains
~78% of pure MILD's strong-dt improvement over the FIXED baseline
(-3.2% vs MILD's -4.1%).

## 4. 10k run justified? YES

Section 7 criteria: (1) fixed-dt RMSE close to the NATURAL baseline --
MIXED-50 is *better* than the baseline at 2k, not merely close; (2)
strong-dt retains meaningful improvement -- yes, -3.2% vs baseline;
(3) not a mere average that loses on both endpoints -- MIXED-50 wins on
both. STOP (section 8) not warranted. One full 10k MIXED-50 run launched
(same frozen `scaleup_v2` 9k/1k split + hyperparameters as baselines A/B;
one seed).

## 5. 10k MIXED-50 full run

Same frozen hyperparameters as the NATURAL and MILD-vardt 10k baselines
(`bs=64, lr=0.004, grad_clip=10.0, max_epochs=60, patience=15,
generic_robust, CPU`), identical `dataset_manifest_sha256`/
`split_manifest_sha256` as both baselines, seed 1. Realised split:
248,316 fixed-form segments / 248,109 thinned-form segments (fraction
0.4998, matching the target 0.5 to within sampling noise).

**Result:** best_epoch=8, best val loss (MSE xy/vxvy) = 0.7783231921867783,
24 epochs run (early-stopped), checkpoint SHA-256
`0def32d5bf79ea1de76d6febb75b80bfd324889e1c2aa6cb5f90e590b9144550`.
Checkpoint selection used **internal validation only**, before any
official-VAL read (freeze manifest written first, per the same
discipline as every prior AV2 KalmanNet task).

**Numerical health:** `norm_overflow_count=67` (vs. MILD's 39, NATURAL's
78 -- all safely contained, STATE B), `per_element_nonfinite_gradient_count=0`,
`nonfinite_gradient_skip_count=0`, `parameter_collapse_count=0`,
`optimizer_state_collapse_count=0`, `large_finite_gradient_count=18786`
(vs. MILD's 9432), `training_collapsed=false`, `training_unstable=false`.
A real, disclosed rough patch: val loss transiently spiked to 298.16 at
epoch 21 (epochs 4/9/10/12/13/16/19/20/21/22 all logged
`any_nan_train=true` at the batch level, none escalating past the
STATE-B safe-skip guard) -- **the selected checkpoint (epoch 8) safely
predates this entire instability window**, so it does not affect the
frozen checkpoint. MIXED-50 shows a rougher mid-training instability
profile than either single-policy 10k run, though it never crosses the
established escalation thresholds.

## 6. Official AV2 VAL results (the actual test)

All four models evaluated on the identical official-VAL sample per
dt-policy (same `n` per policy across every model -- confirms directly
comparable evaluation sets, not independently resampled). Overall
position RMSE (m):

| policy | NATURAL 10k FIXED | MILD var-dt 10k | **MIXED-50 10k** | AV2-tuned LinearCVKF |
|---|---|---|---|---|
| fixed (n=2,908,189)  | **0.3063** | 0.3092 | 0.3112 | 0.3473 |
| mild (n=2,346,805)   | 0.3470 | **0.3468** | 0.3497 | 0.3932 |
| strong (n=1,749,000) | 0.4227 | **0.4141** | 0.4184 | 0.4739 |

**MIXED-50 is the WORST of the three KalmanNet variants on both fixed and
mild** (not the best, and not even a tie) -- it does not eliminate the
~0.9% fixed-dt regression MILD-vardt introduced; it makes it slightly
larger (NATURAL to MIXED-50: +1.6% on fixed, vs. NATURAL to MILD-vardt's
own +0.9%). On strong, MIXED-50 sits **between** the two single-policy
models (-1.0% vs. NATURAL, vs. MILD-vardt's own -2.0%) -- it retains only
about half of MILD-vardt's own strong-dt improvement, not "most" of it.
**Every KalmanNet variant clearly and decisively beats the AV2-tuned
LinearCVKF on every condition** -- this part of the original PR #61
finding is fully reconfirmed and unaffected by this task.

## 7. Matched / missing report (official VAL)

Position RMSE (m), matched (had a real detector measurement this frame)
vs. missing (predict-only frame), all three dt policies:

| policy | bucket | NATURAL | MILD var-dt | **MIXED-50** | KF |
|---|---|---|---|---|---|
| fixed  | matched (n=2,456,380) | **0.2649** | 0.2677 | 0.2703 | 0.2975 |
| fixed  | missing (n=451,809)   | **0.4716** | 0.4750 | 0.4756 | 0.5432 |
| mild   | matched (n=1,983,940) | 0.2840 | **0.2836** | 0.2871 | 0.3152 |
| mild   | missing (n=362,865)   | **0.5811** | 0.5815 | 0.5832 | 0.6757 |
| strong | matched (n=1,482,119) | 0.3147 | **0.3091** | 0.3131 | 0.3389 |
| strong | missing (n=266,881)   | 0.7879 | **0.7703** | 0.7761 | 0.9133 |

The same pattern holds in both buckets separately, not just in the
overall aggregate: MIXED-50 is worst-of-three on fixed and mild in
**both** matched and missing frames, and middle-of-three on strong in
**both** matched and missing frames. This rules out the possibility that
MIXED-50's overall regression is an artifact of one bucket dominating the
aggregate -- the underperformance is uniform across matched/missing.

## 8. Internal-VAL cross-check

The same relative ordering (NATURAL best on fixed, MILD-vardt best on
strong, MIXED-50 worst-of-three on fixed) reproduces on internal
validation (a structurally disjoint sample from official VAL, same
dataset/split manifests):

| policy | NATURAL | MILD var-dt | **MIXED-50** | KF |
|---|---|---|---|---|
| fixed (n=2,968,204)  | **0.3056** | 0.3087 | 0.3110 | 0.3474 |
| mild (n=2,395,775)   | **0.3445** | 0.3439 | 0.3468 | 0.3905 |
| strong (n=1,786,070) | 0.4262 | **0.4176** | 0.4215 | 0.4780 |

Internal-VAL matched/missing for MIXED-50, all three policies: fixed
matched 0.2696 / missing 0.4763; mild matched 0.2861 / missing 0.5752;
strong matched 0.3120 / missing 0.7905 -- the same worst-on-fixed-and-mild,
middle-on-strong pattern holds in both buckets here too.

**The ranking is fully robust across every split and every bucket
checked (official VAL, internal VAL, matched, missing, aggregate): on
fixed and mild NATURAL is best and MIXED-50 is worst-of-three; on strong
MILD var-dt is best and MIXED-50 sits between the two single-policy
models, closer to NATURAL than to MILD var-dt's own improvement.** This
is not a screening-vs-scale surprise unique to official VAL -- the
2k-internal-VAL screening signal that made MIXED-50 look like it
"dominates the tradeoff" (section 3) did **not** survive to 10k on
either the internal or official VAL sample, mirroring the same
screening-to-scale generalization gap already documented for MILD's own
2k signal in the variable-dt v1 task.

## 9. Accept / reject decision

**REJECT.** MIXED-50 does not achieve either of its two design goals:

1. **Did not eliminate the fixed-dt regression** -- it made it slightly
   larger than MILD-vardt's own regression (+1.6% vs. NATURAL, vs.
   MILD-vardt's +0.9%).
2. **Did not retain "most" of the strong-dt benefit** -- it retained
   roughly half of MILD-vardt's own strong-dt improvement over NATURAL
   (-1.0% vs. MILD-vardt's -2.0%).

MIXED-50 is dominated in the practically relevant sense: NATURAL is
better on fixed+mild, MILD-vardt is better on mild+strong, and MIXED-50
is never the best choice on any of the three conditions. The 2k screening
result that motivated this 10k run (section 3: "MIXED-50 improves the
FIXED baseline on all three conditions") did not transfer to 10k scale --
this is the same 2k-to-10k screening-signal-instability finding already
seen with MILD's own 2k result in the variable-dt v1 task, now confirmed
a second time with a different policy. **Recommendation carried forward
unchanged from PR #61: keep NATURAL 10k FIXED-DT (seed1) as the preferred
AV2-pretrained checkpoint for further downstream work**, per its clean
fixed-dt performance and the fact that MILD-vardt's own strong-dt gain
does not clearly outweigh its fixed-dt cost either (PR #61's own
conclusion, unchanged here). No further per-segment
fixed/thinned-mixture curriculum variant is recommended without first
understanding *why* 2k-scale internal-validation screening has now twice
failed to predict 10k-scale official-VAL ranking for this class of
policy.

## 10. Numerical health summary

All three 10k runs (NATURAL, MILD-vardt, MIXED-50) stayed within
STATE A/B (healthy / safely-contained aggregate norm-overflow) for their
entire training run -- zero STATE C (per-element non-finite gradient)
events, zero parameter or optimizer-state collapse, zero
`training_unstable`/`training_collapsed` flags, across all three. MIXED-50
had the roughest mid-training profile of the three (67 norm-overflow
events, 18,786 large-finite-gradient events, transient val-loss spike to
298.16 at epoch 21) but its selected checkpoint (epoch 8) predates that
window entirely and its own numerical-health counters never escalate
past the same safe thresholds the other two runs also stayed within.

## 11. Tests

- `test_dt_curriculum.py` (17 tests, pre-existing from this task's own
  earlier commits, all still passing): deterministic
  fixed-vs-thinned per-segment selection given a fixed policy seed,
  independence from the thinning-internal and corruption RNG streams
  (distinct `:mix:` infix), resume-safety (static assignment, no
  per-epoch state), no official-VAL leakage in the selection mechanism,
  temporal-thinning semantics unchanged (delegates to the frozen
  `variable_dt.py` unmodified), mixed-policy manifest
  serialization/round-trip.
- **Curriculum-probability-schedule tests: N/A.** Per section 2, a
  per-epoch curriculum ramp was never implemented (MIXED-50's 2k result
  made it unnecessary per the task's own stop condition) -- there is no
  such code path to test.
- Full `tools/kalmannet_training/` regression suite re-run: unaffected
  (no existing file modified beyond the new `dt_curriculum.py`/
  `train_kalmannet_dt_curriculum.py` additions from earlier in this
  task). `py_compile`/`pyflakes`/`git diff --check` clean.

## 12. Files

`tools/kalmannet_training/{dt_curriculum.py, train_kalmannet_dt_curriculum.py,
test_dt_curriculum.py}` (new, from earlier in this task),
`tools/kalmannet_training/av2_dt_curriculum_results/{screening_mixed50_2k_internal_val_dt.json,
screening_mixed50_2k_history.csv, freeze_manifest_mixed50_10k.json,
mixed50_10k_history.csv}` (new, small/machine-independent -- no
checkpoint/resume binaries/AV2 data committed), this file. No
`kalmannet_core.py`/`KalmanNetFilter`/`variable_dt.py`/AB3DMOT/
CenterPoint/ROS/prediction/planner/occupancy-grid file changed. No MORAI
claim anywhere -- MORAI simulator access is unavailable in this
environment.

**Recommended next task:** given two consecutive screening-to-scale
generalization failures (MILD's own 2k signal in the prior task, now
MIXED-50's), a task specifically investigating *why* 2k-scale internal
validation does not reliably predict 10k-scale official-VAL ranking for
this family of dt-augmentation policies (e.g. a 2k-vs-10k ranking-
correlation study across all three policies tested so far, or a larger
screening set) would be higher-value than a fourth dt-policy variant.
Not started here. **Do NOT start another experiment.**
