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
change this task asks for. <!-- CURRICULUM_DECISION -->

<!-- SCREENING_PLACEHOLDER -->
