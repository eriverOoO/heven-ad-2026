# MORAI-Calibrated AV2 Corruption v1

Offline KalmanNet experiment. Replaces the GENERIC-ROBUST AV2 corruption
model (isotropic zero-mean Gaussian + iid dropout + uniform-length burst
dropout) with a corruption distribution calibrated ONLY from MORAI
TRAIN/VALIDATION residual/gap statistics, then measures whether this
changes AV2 held-out performance, MORAI zero-shot performance, the
Stage-1 missing-measurement benefit, and the learned-gain behaviour.

**No KalmanNet architecture change. No AV2 dataset re-scale (still the
Stage-1 2,000-scenario pilot). No direct fine-tuning on MORAI state
trajectories.** Builds on merged PR #51 (`analysis/kalmannet-av2-morai-domain-gap-v1`,
merge commit `936e32a592697622f2d3dec021f6f375ca0fe5e3`), verified via
`gh pr view 51` before branching.

## 1. Calibration boundary (hard constraint, verified programmatically)

Calibration source: `~/heven_presentation_assets/state_estimator_gt_comparison/sequences.pkl`
(the ORIGINAL, non-dense-filtered MORAI GT-aligned measurement stream --
chosen over the dense-filtered `dense_train.pkl`/`dense_val.pkl` used by
the prior domain-gap task specifically because dense-filtering excludes
short/fragmented runs, which would bias the dropout/gap calibration in
Section 4 below).

- **TRAIN**: 22 sequences, actors `[1,13,15,18,35,36,38,40,44,46,48]`, 2053 frames.
- **VAL**: 4 sequences, actors `[21,23,27,31]`, 408 frames.
- **TEST (frozen, never touched for calibration)**: 3 sequences, actors `[16,20,30]`, 157 frames.

Split frozen before T-12, per
`~/heven_presentation_assets/motion_gt_expansion/canonical/split_policy.json`
(`"policy_frozen_before_t12": true`). `morai_calibration.py`'s
`calibrate_residual_stats()`/`calibrate_gap_stats()` both raise
`LeakedTestDataError` if handed any `role == "test"` row -- a
programmatic guard, not just a documentation promise (see
`test_morai_calibration.py`). The previously-observed frozen-TEST 24-frame
gap is an evaluation observation only and was never used as a training
parameter anywhere in this task.

## 2. MORAI TRAIN/VAL residual statistics

Pooled TRAIN+VAL (n=1890 present-measurement frames):

| statistic | TRAIN (n=1606) | VAL (n=284) | TRAIN+VAL (n=1890) |
|---|---|---|---|
| x_bias (m) | -0.6697 | -0.5705 | -0.6548 |
| y_bias (m) | +0.7171 | +0.3718 | +0.6652 |
| x_std / y_std (m) | 0.5845 / 0.5658 | 1.1922 / 0.6838 | 0.7108 / 0.5980 |
| xy correlation | -0.2402 | -0.0418 | -0.1812 |
| radial p50/p90/p95/p99 (m) | 1.101/1.868/2.285/2.768 | 1.177/2.565/2.725/2.857 | 1.105/1.984/2.445/2.793 |
| tail beyond 1m/2m/3m/5m | 0.572/0.077/0.0/0.0 | 0.592/0.222/0.0/0.0 | 0.575/0.099/0.0/0.0 |

y_bias is positive in EVERY per-actor split (see Section 4); x_bias is
negative in 14/15 actors (see Section 4 exception). tail_beyond_3m == 0.0
in both TRAIN and VAL independently -- no heavy outlier tail at this
sample size, evidence used in Section 3's model selection.

## 3. Model selection (independent Gaussian vs. full-covariance vs. bias+full-covariance)

Three candidate models were sampled at TRAIN+VAL's own pooled bias/
covariance and compared against the real radial-tail frequencies:

| model | tail>1m | tail>2m | tail>3m | radial p50 | radial p90 |
|---|---|---|---|---|---|
| **REAL (target)** | 0.575 | 0.099 | 0.0 | 1.105 | 1.984 |
| A. independent Gaussian x/y (same bias/marginal std, zero correlation) | 0.606 | 0.089 | 0.002 | 1.168 | 1.960 |
| B. full-covariance, **zero bias** | 0.302 | 0.010 | 0.0 | 0.752 | 1.362 |
| **C. bias + full-covariance (selected)** | 0.585 | 0.099 | 0.002 | 1.142 | 1.995 |

**Model B (no bias) is decisively wrong** -- dropping the bias collapses
the radial tail by roughly half and the median by ~32%; the bias term is
not optional. **Model A vs. C differ only marginally** (the -0.18
correlation is real and consistently signed across TRAIN and VAL
independently, but modest in magnitude); Model C is selected per the
task's own preferred formula (`z = GT + bias + L @ epsilon`,
`L L^T = covariance`) since it captures the correlation the data actually
shows without meaningfully complicating the model. **Selected: Model C
(bias + full-covariance Gaussian).**

## 4. Bias/covariance model (final)

```
bias_xy       = (-0.6548025429052758, 0.6652460172320311)
covariance_xy = [[ 0.5054367466769748, -0.07705166171251883],
                 [-0.07705166171251883, 0.3577342804083116]]
```

**Per-actor stability**: y_bias is **positive for all 15 TRAIN+VAL
actors** (range 0.084-1.336). x_bias is negative for 14/15 actors (range
-0.40 to -1.44) with **one exception, actor 31 (+0.226)** -- reported
honestly, not smoothed away. The pooled bias is judged usable because it
reflects the dominant, sign-consistent pattern across nearly every actor,
not a single actor's idiosyncrasy; actor 31 is a documented, known
exception, not evidence the bias direction is unstable.

**Speed-bin stability**: x_bias becomes more negative with speed
(-0.458 -> -0.601 -> -0.579 -> -1.021 across [0,3)/[3,6)/[6,9)/[9,100)
m/s bins); radial median error rises mildly with speed (0.797 -> 1.090 ->
1.153 -> 1.355 m). Real, but the task's own preferred model is a single
global bias/covariance, not a speed-conditioned one -- reported as a
secondary factor, not modeled.

## 5. Range dependence (measured, not modeled)

`range_m` was available for 100% of TRAIN+VAL frames (1890/1890).
Radial-median by range bin: `[0,10)`=0.644, `[10,20)`=1.050,
`[20,30)`=1.279, `[30,50)`=1.068, `[50,1000)`=1.465 (n=98/378/288/876/250).
A real increasing trend exists but is **not monotonic** (the `[30,50)`
bin dips below `[20,30)`), and the near-range bin has a comparatively
small sample (n=98). Per the task's own instruction ("if evidence is
weak: keep a single global covariance, do NOT invent a complex range
model"), **no range-conditioned covariance scale was implemented** --
this is documented as a caveat/limitation, not modeled.

## 6. Dropout/gap model

Pooled TRAIN+VAL gap statistics (n=2435 frames, dt-defined frames only):

| | TRAIN | VAL | TRAIN+VAL |
|---|---|---|---|
| availability_rate | 0.7799 | 0.6931 | 0.7655 |
| missing_rate | 0.2201 | 0.3069 | 0.2345 |
| n_gaps | 80 | 18 | 98 |
| n_single_frame_gaps | 27 | 7 | 34 |
| n_burst_gaps (len>=2) | 53 | 11 | 64 |
| gap_length p50/p90/max | 3.0/10.0/44 | 2.0/25.2/35 | 2.0/12.1/44 |

Final calibrated dropout model (pooled TRAIN+VAL):
- **single-frame miss probability**: `34/2435 = 0.013963`
- **burst-length pool** (empirical resampling, not a fitted parametric
  family -- the exact 64 observed TRAIN+VAL burst lengths):
  `[2]x16, [3]x8, [4]x5, [5]x6, [6]x3, [7]x4, [8]x2, [9]x6, [10]x4,
  17, 20, 21, 23, 24, 28, 29, 34, 35, 44`. Values up to 44 appear SOLELY
  because that is what TRAIN+VAL contains; the frozen-TEST-observed
  24-frame gap was never consulted to choose this pool or any parameter
  in it.
- **burst-start probability**: the naive rate (`64/2435 = 0.02628`)
  under-produced the target missing rate when replayed on real AV2
  sequences (measured 0.1694 vs. target 0.2345) -- a renewal-process
  artifact of `apply_corruption`'s sequential burst scan (each triggered
  burst consumes its own length in candidate start positions, so the
  realized burst-covered fraction is `p*E[L]/(p*E[L]+(1-p))`, not the
  naive product). Corrected via a small grid search **against AV2 TRAIN
  OUTPUT ONLY** (never any MORAI split) to `0.040` (measured missing rate
  0.2354 vs. target 0.2345).

## 7. Missing-measurement semantics (not re-derived this task)

MORAI's missing-measurement mechanism (per prior sessions' audit of
`build_dataset.py`/the Euclidean-detector association pipeline that
produced `sequences.pkl`) is **Euclidean-detection-and-association
miss**: a frame is missing when no clustered detection was successfully
matched to the GT actor at that timestep, conflating detector miss and
association miss into one observed outcome; track fragmentation is not
separately distinguishable from this stream alone. This task calibrates
the OBSERVED missing-frame process (whatever its cause) rather than
attempting to separately model detector-vs-association contributions --
consistent with the corruption model's own scope (it corrupts a clean AV2
measurement stream at the frame level, it does not simulate a detector).

## 8. Remaining gap: variable dt (not addressed this task)

AV2's own real, non-uniform `dt_s` sampling (10 Hz nominal, `dt_s[0]=nan`)
is unchanged. Per this task's explicit scope, dt/state sampling was left
untouched -- a real domain-gap factor (documented in the prior domain-gap
task) that this corruption calibration does not address.

## 9. Remaining gap: motion-composition (not addressed this task)

The Stage-1 pilot's 2,000-scenario/actor population (dominated by
`VEHICLE`/`PEDESTRIAN`/`STATIC`/`BACKGROUND` class mix, per the training
class-distribution log) is unchanged. MORAI's frozen TEST actors are a
small, hand-selected set of vehicle trajectories with their own
motion-regime composition (stationary fraction, turning content, etc.).
This task calibrates measurement CORRUPTION only, not which AV2 scenarios
are sampled or their motion composition -- a real, unaddressed remaining
domain-gap factor, documented and carried forward.

## 10. Corruption config v2 (implementation)

`ad_morai_bridge_dev/ad_morai_bridge_dev/dataset/av2_motion_forecasting_adapter.py`:
`CorruptionConfig` gains three opt-in, default-off fields --
`bias_xy: (float, float) = (0.0, 0.0)`, `covariance_xy: 2x2 tuple | None
= None` (supersedes `gaussian_noise_std_m` when set; Cholesky-sampled
correlated noise, `z = clean + bias + L @ epsilon`), and
`dropout_burst_length_samples: tuple[int, ...] = ()` (empirical
resampling pool; supersedes the uniform `dropout_burst_min_len`/
`dropout_burst_max_len` draw when non-empty). `validate()` checks
finite bias, symmetric + positive-semidefinite covariance (eigenvalue
check), and all-positive burst-length entries. Every existing field/mode
(`"none"`/`"gaussian"`/`"dropout"`/`"dropout_burst"`/`"custom"`) is
byte-for-byte unchanged when the new fields are left at their defaults --
verified by the full pre-existing `test_av2_motion_forecasting_adapter.py`
suite (47/47 unaffected) plus 10 new v2-specific tests (PSD rejection,
asymmetric-covariance rejection, non-finite-bias rejection, bias-only
application, deterministic correlated sampling, large-sample correlation
match, empirical-pool burst-length reachability, empty-pool fallback to
the exact pre-existing uniform path).

`tools/kalmannet_training/morai_calibration.py` (new): the frozen
calibration constants (Section 4/6 above) plus
`calibrate_residual_stats()`/`calibrate_gap_stats()` (both raise
`LeakedTestDataError` on any `role=="test"` input) and
`build_morai_calibrated_corruption_config()`. `train_kalmannet.py` gains
a new `CONDITION_PRESETS["morai_calibrated_robust"]` entry built from
this function -- `"clean"`/`"generic_robust"` unchanged.

## 11. Determinism

Reuses the existing per-segment `SHA-256(f"{seed}:{segment_id}")`-derived
`np.random.RandomState` stream unchanged (`_segment_corruption_seed`) --
neither the Cholesky-noise path nor the empirical-burst-length-pool path
introduces a second RNG source. `test_corruption_v2_full_covariance_noise_is_deterministic_given_seed`
and the existing `test_dropout_burst_deterministic_and_produces_contiguous_runs`
(reused unmodified for the fallback path) both lock this.

## 12. Pre-training distribution check (generated AV2 vs. MORAI TRAIN+VAL target)

Applied the final calibrated config to a real 3,000-sequence AV2 TRAIN
sample (Stage-1 pilot shard, 159,340 frames):

| statistic | GENERATED (AV2 TRAIN) | TARGET (MORAI TRAIN+VAL) |
|---|---|---|
| bias_xy | (-0.6539, 0.6634) | (-0.6548, 0.6652) |
| covariance_xy | [[0.5066,-0.0786],[-0.0786,0.3594]] | [[0.5054,-0.0771],[-0.0771,0.3577]] |
| radial p50/p90/p95/p99 | 1.131/1.972/2.221/2.696 | 1.105/1.984/2.445/2.793 |
| tail>1m/2m/3m/5m | 0.587/0.093/0.003/0.0 | 0.575/0.099/0.0/0.0 |
| missing_rate | 0.2338 | 0.2345 |
| gap p50/p90/max | 3.0/12.0/46 | 2.0/12.1/44 |

Close match across every reported statistic (bias/covariance essentially
exact, radial percentiles and missing rate within a few percent, gap
p90/max within 2 frames). The renewal-process burst-probability
correction (Section 6) was necessary to reach this match -- the
uncorrected naive rate produced missing_rate 0.1694, a ~28% relative
shortfall. No further implementation change was needed after the
correction.

## 13. Training result

One seed (seed 0), same Stage-1 pilot dataset/split/architecture/
optimizer as the frozen GENERIC-ROBUST checkpoint, frozen config
(`batch_size=64, lr=0.004, max_epochs=60, patience=15, grad_clip=10.0,
loss_on_predict_only=True`). **Best epoch: 6** (`val_loss=1.6461`).
Training ran 22 epochs total (0-21) before early-stopping (patience
exhausted, no new best after epoch 6), `catastrophic=false`,
`nonfinite_step_count=0`. **Checkpoint SHA-256**:
`125d02f84493b41413690a3c6c53e9618dbddf3a3b31843e61aa19046be8ae37`.

**A real, disclosed training-stability finding.** From epoch 9 onward,
`grad_norm_mean` (pre-clip diagnostic) repeatedly spiked to extreme values
(1181 at epoch 9; `inf` with `train_loss=5.5e24` at epoch 10; up to
`5.7e10` at epoch 15; up to `1.1e9` at epoch 20) while `val_loss` never
diverged (stayed in `[1.646, 2.134]` throughout, `nonfinite_step_count=0`
for the whole run) -- `grad_clip=10.0` evidently bounded every *applied*
parameter update even when the pre-clip norm was astronomically large.
The best checkpoint (epoch 6) predates every one of these spikes, so it
was never affected regardless. **Suspected cause**: this corruption
config's dropout bursts run far longer (up to 44 frames, vs.
GENERIC-ROBUST's uniform max of 5) -- occasional post-long-gap batches
plausibly produce numerically extreme KalmanNet gain computations before
clipping. This is the same general failure class as the project's
earlier documented KalmanNet training-instability investigation
(T-12.2/T-12.3, a different trigger point -- there it was epoch-0 seed
variance, here it is intermittent post-long-gap batches mid-training) --
not re-investigated further here, since it never affected the frozen
checkpoint.

## 14. Freeze

`tools/kalmannet_training/write_freeze_manifest_morai_calibrated.py`
writes `frozen_configs/kalmannet_morai_calibrated_corruption_v1_freeze.json`
from the checkpoint's own AV2-VAL-selected `best_epoch`/`best_val_loss`
plus the calibration provenance (Section 4/6 constants, TRAIN/VAL actor
IDs, frozen-TEST actor IDs explicitly recorded as never touched). Run
**before** any AV2 TEST or MORAI TEST evaluation of this checkpoint --
unlike the prior Stage-1 pilot task's disclosed procedural deviation
(freeze written after TEST eval), this task's ordering is correct:
freeze -> AV2 TEST eval -> MORAI zero-shot eval.

## 15. AV2 TEST result

Evaluated via `evaluate_kalmannet.py --training-condition
morai_calibrated_robust` on the held-out Stage-1 AV2 TEST split
(10,704-10,731 sequences depending on condition). **Caveat, stated up
front**: `B_same_corruption_family`/`C_different_corruption_seed` apply
the MORAI-calibrated corruption's OWN (larger, ~1.1m radial median)
measurement noise, vs. GENERIC-ROBUST's own (smaller, isotropic 0.3m std)
noise -- so absolute RMSE is **not** comparable across D's and E's own
`B` conditions; only the *within-condition* ranking (does KNet still beat
the KF baselines under its own noise) is informative there.

| condition | estimator | pos RMSE (m) | vel RMSE (m/s) |
|---|---|---|---|
| A_clean_held_out | E (this checkpoint) | 0.1931 | 1.4789 |
| A_clean_held_out | kf_transferred / kf_av2_tuned / dense_v2 | 0.0811 / 0.1032 / 0.0679 | 1.4747 / 1.4912 / 1.3628 |
| B_same_corruption_family | E | 1.4864 | 1.9225 |
| B_same_corruption_family | kf_transferred / kf_av2_tuned / dense_v2 | 2.1135 / 1.9723 / 3.5201 | 2.7814 / 2.4840 / 4.7645 |

Within its own `B` condition, **E still beats every KF baseline and
dense_v2** (same qualitative ranking GENERIC-ROBUST's own `D` showed:
KNet 0.320 vs. KF 0.351-0.358 vs. dense_v2 0.449 under D's own, smaller
noise). **On CLEAN held-out AV2 (no corruption at all), E is clearly
worse than D** (E: 0.1931 vs. D's own recorded clean-knet 0.0817) -- a
real, sensible tradeoff: training under substantially heavier noise/
dropout makes the model less sharp on the noise-free case. E's
`gap::gap_6_plus` bucket (n=58,590, pos RMSE 4.04m) is far larger and
much more populated than D's own `gap_6_plus` bucket under D's own
condition (n=1,057, pos RMSE 1.18m) -- expected, since D's own corruption
essentially never produces a gap beyond 5 frames, so this specific
bucket-vs-bucket number is not a fair "recovery quality" comparison
either (see Section 16 for the fair, fixed-gap-structure version).

## 16. MORAI zero-shot result

`evaluate_morai_frozen_stream.py`, frozen MORAI TEST (3 sequences,
actors 16/20/30, identical GT/measurements/dt/missing-pattern for every
condition -- the fair, fixed comparison):

| condition | overall pos RMSE (m) | overall vel RMSE (m/s) |
|---|---|---|
| A_tuned_kf | 3.0654 | 2.7960 |
| B_dense_v2 | 1.8837 | 2.3024 |
| C_stage0_av2 | 2.2262 | 2.4928 |
| D_stage1_generic_av2 | 1.8648 | 2.3714 |
| **E_stage1_morai_calibrated_av2 (this task)** | **1.8261** | 2.4727 |

E has the lowest overall position RMSE of all 5 conditions. This
headline number, however, is dominated by one sequence (seq0, the only
one with real missing frames) -- see Section 17.

## 17. Matched vs. missing comparison

Per-sequence pooled RMSE, D vs. E, computed identically
(`estimator_trace.py`, matches the historical D numbers exactly as a
cross-check: D matched=1.5551/missing=5.1472, reproducing
`kalmannet_av2_stage1_pilot_v1_results_summary.json`'s own recorded
`1.555124659289333`/`5.147171820744625` to full precision):

| | D (generic-robust) | E (morai-calibrated) | change |
|---|---|---|---|
| **matched** (n=130, pooled across all 3 seqs) | 1.5551 | 1.6668 | **+7.2% worse** |
| **missing** (n=24, seq0's real 24-frame gap only) | 5.1472 | 4.1619 | **-19.1% better** |

Per-sequence detail (pos RMSE): seq0 (has the missing gap) matched
1.531->1.264 (better), missing 5.147->4.162 (better) -- **E wins on
both fronts for the one sequence containing a real gap**. seq1
(all-matched, n=72) 1.744->2.000 (worse). seq2 (all-matched, n=19)
0.466->0.754 (worse, though both values are small in absolute terms).

**This is the opposite of the task's stated hoped-for pattern** (matched
improves while missing is preserved) -- here missing clearly improves
while matched pooled slightly regresses, driven entirely by the two
short, low-noise, all-matched sequences. Reported as measured, not
reframed to fit the hoped-for direction.

## 18. Gain / prior-posterior analysis

`estimator_trace.py`, matched-frame regime, D vs. E:

| | D | E |
|---|---|---|
| mean gain Frobenius norm | 1.6018 | **1.2159** |
| prior pos RMSE | 1.7911 | 1.9206 |
| posterior pos RMSE | 1.5551 | 1.6668 |
| prior->posterior improvement | 0.2360 | 0.2539 |

**E learns a smaller gain than D on matched frames** -- consistent with
the expected direction: E was trained against measurement noise with a
much larger radial spread (bias-driven, ~1.1m median vs. D's isotropic
0.3m std), so a well-calibrated network should trust each individual
measurement update less. This is a genuine, mechanistically-sensible
change in measurement-trust behavior, independent of whether it improved
raw RMSE (it did not, on the two low-noise sequences -- Section 17).
Missing-frame prior==posterior for both (pure prediction, no gain
applies) -- E's **prediction alone** does better after the real gap
(4.162 vs. 5.147), consistent with E's training exposure to real
long-burst dropouts (up to 44 frames) improving its recurrent
coasting/extrapolation behavior specifically, separate from its
(reduced) measurement trust.

## 19. Uncertainty / small-sample caveat

Per-sequence bootstrap (n=3, 2000 resamples) of the mean overall
position-RMSE difference D-E: observed diff `+0.0387`, 95% CI
`[-0.267, 0.638]` (crosses zero), `fraction_bootstrap_same_sign=0.716`,
`stable=False`. **D and E are not statistically distinguishable at this
sample size on the crude per-sequence-mean metric.** No
statistical-robustness claim is made anywhere in this document for any
MORAI-based comparison -- n=3 is far too small, and the entire
missing-frame finding (Section 17) rests on a single sequence's single
24-frame gap.

## 20. Accept/reject decision

**Closest to Outcome E (MORAI differences too unstable at this sample
size), with a genuine, mechanistically-explained but statistically-
unconfirmed direction reversal layered on top.** Neither A nor B applies
cleanly: matched did not improve (it regressed ~7%, pooled), and missing
did not regress (it improved ~19%) -- the **opposite pairing** from both
A's and B's hoped-for/feared directions. C ("neither improves") is also
wrong, since missing genuinely did improve. D does not apply either --
"AV2 improves" is itself ambiguous here (E beats its own KF baselines
under its own noise, same as D did, but is clearly worse than D on
CLEAN AV2, Section 15) so no clean "AV2 improves, MORAI doesn't" story
holds. The bootstrap CI crossing zero (Section 19) is the dominant fact:
**with only 3 MORAI TEST sequences, one of which single-handedly
supplies every missing-frame data point, this experiment cannot
distinguish "MORAI-calibrated corruption genuinely helps gap recovery
and genuinely hurts matched-frame precision" from "this specific
sequence happened to favor E and these two specific short sequences
happened to favor D."** The gain-direction finding (Section 18) is the
one result that IS robust to this sample-size concern (it is a
per-frame, n=130 statistic reflecting a mechanistic property of the
trained network, not a 3-sequence RMSE comparison), and it does confirm
the calibration changed measurement trust in the theoretically expected
direction.

## 21. Recommended next task

**Outcome E's own remedy: build/freeze a larger MORAI evaluation
dataset (Option E from the task's own next-step menu).** Every
conclusion in Sections 16-20 is bottlenecked by the same 3-sequence,
157-frame frozen MORAI TEST stream this project has reused since T-12 --
a single sequence supplies 100% of the missing-frame evidence here, and
the bootstrap CI is uninformatively wide as a direct, measured
consequence. Before drawing any further conclusion about whether
MORAI-calibrated corruption (or any future AV2-training variant) helps
or hurts real MORAI performance, a larger, still-leakage-safe MORAI
TEST split (more actors/sequences, ideally with more independent
long-gap examples so the missing-frame finding is not carried by a
single sequence) would let Sections 16-20's comparisons actually
distinguish signal from n=3 noise. Scaling the AV2 side (Option A,
10k scenarios) or adding a generic+MORAI corruption curriculum
(Option F) are both reasonable alternatives, but neither addresses the
actual bottleneck exposed by Section 19: the evaluation side, not the
training side, is what is currently starved of data. Not started this
session.

## Files changed

- `ad_morai_bridge_dev/ad_morai_bridge_dev/dataset/av2_motion_forecasting_adapter.py` (corruption_v2: bias/covariance/empirical-burst-pool, additive only)
- `ad_morai_bridge_dev/test/test_av2_motion_forecasting_adapter.py` (+10 tests)
- `tools/kalmannet_training/morai_calibration.py` (new)
- `tools/kalmannet_training/test_morai_calibration.py` (new)
- `tools/kalmannet_training/train_kalmannet.py` (+1 condition preset, reused unmodified otherwise)
- `tools/kalmannet_training/evaluate_kalmannet.py` (+`morai_calibrated_robust` training-condition choice)
- `tools/kalmannet_training/evaluate_morai_frozen_stream.py` (new)
- `tools/kalmannet_training/write_freeze_manifest_morai_calibrated.py` (new)
- `tools/kalmannet_training/frozen_configs/kalmannet_morai_calibrated_corruption_v1_freeze.json` (new, small/machine-independent)
- `tools/kalmannet_training/frozen_configs/kalmannet_morai_calibrated_corruption_v1_results.json` (new, small/machine-independent)
- `docs/perception/kalmannet_morai_calibrated_corruption_v1.md` (this file)

No `ad_lidar_perception/ad_lidar_perception/kalmannet_core.py` change. No
AV2/MORAI raw data, checkpoint, training log, or local path committed.
