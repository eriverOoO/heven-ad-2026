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

<!-- FULL_RUN_PLACEHOLDER -->
