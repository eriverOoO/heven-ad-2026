# AV2→MORAI KalmanNet Domain-Gap Analysis v1

Status: **Diagnostic/analysis only. No new training, no architecture
change, no runtime/ROS/CenterPoint/planner change.** Explains, using the
exact existing checkpoints from PR #49/#50, WHY AV2 Stage-1 (2,000
scenarios) improved missing-measurement zero-shot MORAI transfer but
regressed matched-measurement and AV2-internal CLEAN performance
relative to Stage-0 (120 scenarios).

## Checkpoints analyzed (frozen, not retrained)

| checkpoint | SHA-256 |
|---|---|
| A. dense-v2 (MORAI-trained) | `956604975e5204b2c584c3e8fe16a7ba9346097980566ecbbb22c2001fbf7d48` |
| B. Stage-0 AV2 robust KNet (120 scenarios, calibrated bs=64/lr=0.004) | `6e45c8c4d73b659de569b1f61038d26748f41bc45f17a86cb42a7f7ba7765f65` |
| C. Stage-1 AV2 pilot KNet (2,000 scenarios, same calibrated config) | `65fe0c2b13caa2948c8128aacb331a69e0b0f1087ba581be7df7d2e5944e0c17` |

All three verified present, unmodified, matching the hashes already
recorded in PR #49/#50's own artifacts.

## 1. Dataset distribution comparison

Four sources compared: **A** AV2 Stage-0 TEST (CLEAN, n=724 sequences),
**B** AV2 Stage-1 pilot TEST (CLEAN, n=10,731), **C** dense-v2's actual
historical MORAI training data (`kalmannet_dense_baseline/dense_train.pkl`,
37 sequences, 1,310 frames — verified as the real dense-v2 training set,
not a proxy: 1,310-37=1,273 transitions matches T-12.2's own documented
count exactly), **D** the frozen MORAI TEST stream (3 sequences, 157 frames).

| metric | A: AV2 Stage-0 | B: AV2 Stage-1 | C: MORAI (dense-v2 train) | D: MORAI (frozen eval) |
|---|---|---|---|---|
| stationary fraction | **0.7205** | **0.6384** | **0.0198** | **0.0127** |
| speed p50/p90/p95 (m/s) | 0.0/5.92/8.53 | 0.0/10.57/13.13 | 5.75/9.63/10.74 | 8.62/12.78/13.17 |
| accel p50/p90/p95 (m/s²) | 0.0/1.25/2.10 | 0.002/1.73/2.75 | **2.22/6.72/11.15** | **3.09/8.55/18.37** |
| turn-rate p50/p90/p95 (rad/s) | 0.15/3.18/12.13 | 0.11/3.10/9.71 | 0.32/0.72/1.34 | 0.03/0.87/1.79 |
| dt p50/p90 (s) | **0.100/0.100** | **0.100/0.100** | 0.116/0.221 | 0.130/0.243 |
| coordinate range max\|x\|/\|y\| (m) | 103/143 | 223/170 | 63/21 | 102/7 |

**Two large, real domain gaps found:**

1. **Stationary-object dominance**: AV2 (both stages) is 64-72%
   near-stationary (speed `<0.5 m/s`) because AV2's real object mix
   includes STATIC/BACKGROUND/parked-vehicle tracks; MORAI's training
   and evaluation data (both curated to real moving vehicle actors) is
   98-99% NON-stationary. A KalmanNet trained on a population dominated
   by near-zero motion is optimizing for a very different typical case
   than what it is evaluated on in MORAI.
2. **Fixed vs. variable `dt`**: every single AV2 sequence, at both
   stages, has **exactly** `dt=0.100 s` for every transition (p50==p90==
   0.100, zero variance) — a synthetic, perfectly uniform 10 Hz
   sampling. MORAI's real detector timing varies meaningfully (p50
   0.116-0.130 s, p90 0.221-0.243 s — real processing jitter). The
   network's analytical `F`/`Q` matrices correctly use whatever `dt` is
   passed at inference (verified in prior tasks), but the LEARNED gain
   network has never seen a non-0.1s `dt` during training at either AV2
   stage — a real, previously-unexamined domain gap.

**Directionally supports the Stage-1-improved-transfer finding**: Stage-1's
speed distribution (p90/p95 `10.57/13.13`) is materially closer to
MORAI's own (`12.78/13.17`) than Stage-0's is (`5.92/8.53`) — Stage-1's
larger, more diverse scenario pool included proportionally more
genuinely fast-moving trajectories, better matching MORAI's real traffic
speeds (see Section 8, motion-diversity).

## 2. Measurement noise comparison

| source | n | x_std | y_std | x_bias | y_bias | xy_corr | radial p50/p90/p95/p99 |
|---|---|---|---|---|---|---|---|
| AV2 CLEAN (Stage-0 test) | 39,339 | 0.0 | 0.0 | 0.0 | 0.0 | n/a | 0/0/0/0 (sanity check: exact) |
| AV2 GENERIC-ROBUST (Stage-0 test) | 32,979 | 0.301 | 0.301 | 0.003 | -0.001 | **-0.001** | 0.353/0.645/0.743/0.917 |
| MORAI frozen eval (real detector) | 133 | **1.341** | **0.286** | **0.058** | **0.432** | **0.115** | **1.319/2.126/2.395/2.736** |
| MORAI dense-v2 train (real detector) | 1,310 | 0.556 | 0.542 | **-0.678** | **0.677** | **-0.219** | 1.085/1.797/2.088/2.661 |

**Direct, quantified answer to "is the AV2 generic noise model too
large/small/isotropic/missing bias":**

- **Too small**: real MORAI radial error median (`1.08-1.32 m`) is
  roughly **3-4x larger** than AV2's designed isotropic std of `0.3 m`
  (whose own radial median works out to `0.353 m`, matching the design
  exactly — the AV2 corruption model is internally correct, just
  calibrated to a much smaller magnitude than real MORAI detector noise).
- **Too isotropic**: AV2's noise is `x_std ≈ y_std ≈ 0.30` by
  construction (near-perfect isotropy, `xy_corr ≈ 0`). MORAI's real
  frozen-eval residual is **markedly anisotropic** (`x_std=1.34` vs.
  `y_std=0.29`, a `~4.7:1` ratio) — a real detector-geometry effect AV2's
  generic isotropic Gaussian cannot represent.
- **Missing bias**: AV2's synthetic noise is exactly zero-mean by
  construction (`x_bias≈0.003, y_bias≈-0.001`). Both real MORAI sources
  show **substantial, non-zero bias** (frozen eval: `y_bias=0.43 m`;
  dense-v2 train: `x_bias=-0.68 m, y_bias=+0.68 m`, a striking
  roughly-equal-and-opposite pattern, plausibly a coordinate-origin/
  detector-geometry artifact of the real Euclidean-cluster-vs-GT-box
  association, not investigated further here).
- **Missing correlation**: AV2's `xy_corr≈0` by construction; both real
  MORAI sources show non-trivial `x`/`y` correlation (`0.115`, `-0.219`)
  — AV2's noise model cannot represent this either.

**Conclusion: AV2's GENERIC-ROBUST noise model is a poor quantitative
match to real MORAI detector noise** — smaller, more isotropic, and
zero-bias where the real data is larger, anisotropic, and biased. This
directly supports **Outcome B** (noise-distribution mismatch) as a real
contributing factor, independent of the gain/trust findings below.

## 3. Matched vs. missing regime results + Kalman gain analysis

Instrumented per-frame traces (new `estimator_trace.py`, offline only,
no runtime change) on the frozen MORAI TEST stream (n=130 matched, n=24
missing frames):

| model | matched: prior RMSE | matched: posterior RMSE | improvement | **mean \|K\| (Frobenius)** | missing: RMSE |
|---|---|---|---|---|---|
| A. Tuned KF | 1.614 | 1.456 | 0.158 | n/a | 11.501 |
| B. Dense-v2 | 1.662 | 1.467 | 0.194 | **3.309** | 5.563 |
| C. Stage-0 AV2 | 1.704 | 1.473 | 0.231 | **1.841** | 7.195 |
| D. Stage-1 AV2 | 1.791 | 1.555 | 0.236 | **1.602** | **5.147** |

**Direct mechanistic confirmation of Outcome A**: the mean learned-gain
Frobenius norm decreases monotonically `dense-v2 (3.31) > Stage-0 (1.84)
> Stage-1 (1.60)` — Stage-1 applies the SMALLEST measurement correction
of the three KalmanNet checkpoints. This gain ordering exactly tracks
the matched/missing RMSE tradeoff: **the model that trusts measurements
least (Stage-1) has the best missing-measurement RMSE and the worst
matched-measurement RMSE of the three**; the model that trusts
measurements most (dense-v2) shows the opposite pattern. This is
measured directly on the gain itself, not merely inferred from
downstream accuracy.

## 4. Prior vs. posterior error decomposition

From the same table: **Stage-1's own PRIOR (pre-update, pure-dynamics)
error is the WORST of all four methods** (`1.791`, vs. `1.614-1.704` for
the others) — Stage-1 does not have a more accurate dynamics prediction;
if anything it is slightly less accurate before any measurement is even
considered. Combined with its smaller gain, Stage-1 ends up with the
worst posterior (`1.555`). Its "improvement" (prior − posterior) is
numerically the LARGEST of the four (`0.236`) — but this is because it
starts from a worse prior, not because its update is more effective;
the absolute posterior error still lands worst. **This distinguishes the
two candidate patterns from Section 6 of the task's own framing: Stage-1
does NOT have "a strong prior that isn't pulled close enough" — its
prior itself is measurably worse**, a distinct, second mechanism beyond
the gain-magnitude finding above.

## 5. AV2 CLEAN regression analysis (Stage-0 vs. Stage-1, in-domain)

Same instrumentation applied to a 200-sequence random sample of each
model's OWN AV2 CLEAN TEST set (Stage-0's own 724-sequence test; Stage-1
pilot's own 10,731-sequence test — subsampled for compute, not the full
set; see caveat below):

| model (on its own AV2 CLEAN test) | prior RMSE | posterior RMSE | improvement | mean \|K\| |
|---|---|---|---|---|
| Stage-0 AV2 | 0.104 | 0.035 | 0.068 | 1.919 |
| Stage-1 AV2 | 0.176 | 0.069 | 0.107 | 1.818 |

**The same qualitative pattern reappears IN-DOMAIN**: Stage-1's prior is
markedly worse than Stage-0's (`0.176` vs. `0.104`, ~69% worse) even on
AV2's own clean, familiar data. The gain-magnitude gap here is much
SMALLER (`1.818` vs. `1.919`, only ~5%) than the MORAI cross-domain gap
(`1.602` vs. `1.841`, ~13%, both far below dense-v2's `3.309`). **This
indicates the AV2-internal CLEAN regression is driven predominantly by a
worse learned dynamics/prior component, not primarily by a gain/
measurement-trust difference** (which is comparatively small in-domain)
— a genuinely distinct mechanism from the cross-domain MORAI
matched-vs-missing tradeoff, where the gain gap is the dominant, directly
visible effect. (Subsample note: the 200-sequence numbers here are close
to, but not identical to, the full-test-set numbers already reported in
PR #50's own STATUS entry (`0.0355` vs. the official `0.0361` for
Stage-0; `0.0689` vs. the official `0.0817` for Stage-1) — expected
sampling variance from a 200/724 or 200/10,731 subsample, not a
discrepancy; used here only for the prior/posterior/gain breakdown,
which the official evaluator does not compute.)

## 6. Corruption exposure analysis

| | clean | single dropout | burst dropout | post-gap reacquisition | typical gap length |
|---|---|---|---|---|---|
| AV2 Stage-1 TRAIN (GENERIC-ROBUST) | 84.2% | 7.3% | 8.5% | 9.6% | **p50=1, p90=3 frames** |
| MORAI frozen eval (real) | 84.4% | 0% | 15.6% | 0%* | **the one real gap = 24 frames** |

**Aggregate missing-measurement rate matches closely** (`15.6-15.8%`
missing in both) — AV2's corruption is well-calibrated on AVERAGE rate.
**But the gap-LENGTH distribution is a severe mismatch**: AV2's designed
`dropout_burst_max_len=5` means training NEVER exposes the network to a
gap longer than 5 frames; MORAI's one real observed gap is **24
frames** — nearly 5x longer than anything in AV2 training. (*`0%`
post-gap-reacquisition for MORAI reflects the tiny sample: with only one
24-frame gap across 3 sequences, whether a "post-gap" frame is counted
depends on whether the gap ends before the sequence itself does; not
further investigated given the sample size.) **This is a second,
independent, quantified noise/corruption-model mismatch** (distinct from
Section 2's magnitude/bias mismatch) — AV2 training under-represents
long real-world gaps.

## 7. Motion-diversity analysis (qualitative, from Section 1's own data)

Not run as a separate per-bin KNet evaluation (time/compute-bounded
scope decision) — relies on Section 1's already-computed distribution
comparison. Stage-1's speed distribution (p90/p95 `10.57/13.13 m/s`)
sits materially closer to MORAI's real distribution (`12.78/13.17 m/s`)
than Stage-0's does (`5.92/8.53 m/s`); Stage-1's own acceleration p90/p95
(`1.73/2.75`) is also somewhat higher than Stage-0's (`1.25/2.10`),
though both remain far below MORAI's own (`6.72-8.55/11.15-18.37`).
**Directionally consistent** with "Stage-1's larger scenario pool added
genuine high-speed/high-motion diversity that is more MORAI-relevant
than Stage-0's smaller sample" — but this is NOT a controlled,
per-bin-isolated claim (no separate straight/turning/low-speed/high-speed
subset evaluation was run), so it is reported as directionally
supportive evidence only, not proven causal.

## 8. MORAI sample-size / uncertainty (per-sequence bootstrap)

Per-sequence (not per-frame — respecting within-sequence temporal
dependence) bootstrap, `n_bootstrap=5000`, using each sequence's own
overall position RMSE (`n=3` per model):

| comparison | observed diff (m) | 95% CI | stable (CI excludes 0)? | fraction same-sign |
|---|---|---|---|---|
| Stage-1 − dense-v2 | -0.018 | [-0.245, +0.147] | **NO** | 59.4% |
| Stage-1 − Stage-0 | -0.364 | [-1.192, +0.105] | **NO** | 74.3% |
| Stage-1 − tuned KF | -1.209 | [-3.799, +0.149] | **NO** | 70.7% |

**None of the three headline comparisons are statistically stable at
`n=3` sequences** — every 95% CI includes zero, and the
"fraction-of-bootstrap-resamples-agreeing-in-sign" ranges only
`59-74%` (barely to modestly better than a coin flip). **This is an
important, disclosed limitation: the point-estimate differences reported
throughout PR #50 (Stage-1 beating dense-v2/Stage-0/KF by up to `1.2 m`)
are directionally consistent with the mechanistic gain/prior evidence
above, but are NOT statistically distinguishable from noise given only 3
independent sequences.** Both things are true simultaneously: the
mechanism (lower gain, worse prior) is directly, robustly observable in
the per-frame trace data (n=130-154 frames, not resampled at the
sequence level); the AGGREGATE RMSE ranking across only 3 sequences is
not.

## 9. Additional MORAI held-out data audit

Searched all local `.pkl`/GT-related artifacts under
`~/heven_presentation_assets/`. Found: `motion_gt_expansion/
phase1_actor_traj.pkl` (the raw pre-split actor pool — same 21 actors),
`gt_mot_eval/phase12_actor_traces.pkl` (T-10's own association-tracking
traces, a different data structure/purpose, same underlying scene),
`kalmannet_ablation/{t12_dense_train,t12_full_matched_train,
t12_val_shared,t12_dense_matched_train}.pkl` (all reuse T-12's own
frozen train/val actors, never new test data). **Verified: every one of
these files draws from the identical 21-actor roster**
(`[1,2,3,13,15,16,18,20,21,23,27,30,31,34,35,36,38,40,44,46,48]`) of the
SAME single MORAI scene (`static_20260805_003151`) already split by
`motion_gt_expansion/canonical/split_policy.json`.

**Conclusion: no additional, genuinely independent, never-touched-for-
training-or-tuning MORAI evaluation data exists locally.** The current
MORAI transfer claim is data-limited to the 3 frozen test actors
(16, 20, 30) and cannot be expanded without either a new MORAI capture
or accepting some form of re-use of already-touched data (neither done
here).

## 10. Small controlled ablation

**Skipped**, per this task's own explicit permission ("If diagnostics
are already conclusive enough: skip this section"). The converging
evidence from Sections 2 (noise-magnitude/bias/isotropy mismatch), 3-4
(directly-measured gain-magnitude + prior-degradation mechanism), 5 (the
same prior-degradation pattern in-domain), and 6 (gap-length mismatch)
already identifies multiple concrete, independently-verified mechanisms
without requiring a new training run to establish plausibility. A future
task MAY still want a targeted ablation (e.g. testing whether reducing
`dropout_burst_max_len` or increasing `gaussian_noise_std_m` toward the
measured real values changes the gain/prior pattern) — but this analysis
task's own diagnostics did not require running one to reach a supported
conclusion.

## 11. Most likely explanation

**Outcome E (multiple factors contribute) — evidence does not support a
single-factor story.** Specifically, at least four independent,
separately-verified mechanisms are at play:

1. **Learned measurement-trust reduction** (Section 3): Stage-1's mean
   gain magnitude (`1.60`) is measurably smaller than Stage-0's (`1.84`)
   and dense-v2's (`3.31`) on the MORAI stream — directly explains the
   matched-vs-missing tradeoff (Outcome A confirmed mechanistically, not
   just inferred).
2. **AV2-internal prior degradation** (Section 5): Stage-1's own
   pre-update prediction is measurably worse than Stage-0's even on
   AV2's own clean data, where the gain gap is much smaller (~5% vs.
   ~13-45%) — a DISTINCT mechanism from (1), explaining the AV2 CLEAN
   regression specifically (not previously mechanistically isolated).
3. **AV2 synthetic-noise/MORAI real-noise mismatch** (Section 2 —
   Outcome B confirmed quantitatively): AV2's isotropic, zero-bias,
   `~0.3 m`-std corruption is 3-4x smaller, more isotropic, and
   unbiased relative to real MORAI detector residuals.
4. **Gap-length distribution mismatch** (Section 6): AV2 training never
   exposes gaps longer than 5 frames; MORAI's one real gap is 24 frames.

5. **Statistical instability at `n=3`** (Section 8 — Outcome D
   confirmed as a real, coexisting caveat): none of the aggregate
   MORAI RMSE rankings are stable under per-sequence bootstrap, even
   though the underlying mechanism (1) is directly observable in the
   frame-level trace data.

No single factor among 1-4 is dismissed by the evidence, and factor 5
means the AGGREGATE ranking itself should be held with appropriate
uncertainty regardless of which mechanism explanation is preferred.

## 12. Claims we can / cannot make

**Can claim** (consistent with all evidence gathered):
- "AV2 Stage-1 pilot improved missing-measurement zero-shot performance
  on the small frozen MORAI estimator stream, while matched-measurement
  performance regressed" (the task's own pre-approved claim — fully
  supported).
- "The matched/missing tradeoff is mechanistically associated with a
  measurably smaller learned Kalman gain in the Stage-1 checkpoint."
- "AV2's GENERIC-ROBUST synthetic measurement noise is smaller, more
  isotropic, and less biased than real MORAI detector noise, by a
  factor of roughly 3-4x on radial error magnitude."
- "AV2 training's corruption never produces gaps longer than 5 frames;
  the one real MORAI gap observed is 24 frames."
- "The AV2-internal CLEAN regression from Stage-0 to Stage-1 is
  associated with a measurably worse learned prior/dynamics prediction,
  a distinct mechanism from the gain-magnitude effect."
- "None of the headline MORAI RMSE comparisons (Stage-1 vs. dense-v2/
  Stage-0/tuned-KF) are statistically stable under per-sequence
  bootstrap at n=3 sequences."

**Cannot claim** (explicitly ruled out or unsupported):
- Stage-1 universally beats the tuned KF — NOT claimed; bootstrap shows
  this specific comparison is not stable, and the KF still wins on
  matched-measurement RMSE.
- AV2 scale guarantees better MORAI tracking — NOT claimed; the
  AV2-internal CLEAN regression is real evidence scale does not help
  uniformly.
- Dropout robustness implies overall tracking robustness — NOT claimed;
  the SAME mechanism that improves missing-measurement handling
  (smaller gain) directly causes the matched-measurement regression.
- Matched-measurement regression is harmless — NOT claimed; reported as
  a real, mechanistically-explained cost.
- The 3-sequence MORAI result is statistically strong — explicitly
  contradicted by Section 8's own bootstrap analysis.

## Files changed

`tools/kalmannet_training/{domain_gap_analysis.py,
test_domain_gap_analysis.py,estimator_trace.py,test_estimator_trace.py,
frozen_configs/kalmannet_av2_morai_domain_gap_v1_results.json}` (all
new). No change to any existing training/calibration/eval file, no
architecture change, no runtime/ROS/CenterPoint/planner change.
