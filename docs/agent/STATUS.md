# STATUS

## T-8A: Motion-regime evaluation dataset — PASS, CASE B (KalmanNet readiness)

Branch: `feat/ab3dmot-tracker`. Goal: build a reusable MORAI motion-regime
dataset (straight/turning/transition) so KF/EKF/IMM (and later KalmanNet)
can be evaluated on real sustained turning, since T-7A.5/T-7B found the
original 400-frame canonical recording's turning content modest. **No
KF/EKF/IMM algorithm code touched, per this task's explicit instruction —
dataset/evaluation-tooling only, all three estimators reused unchanged
from T-7A/T-7A.5/T-7B.** Full detail:
`~/heven_presentation_assets/motion_dataset/README.md` (14-section
report, figures/JSON/JSONL/CSV — outside repo, not committed).

**Canonical recording**: frames 200-999 (800 frames) of the same
1,764-frame `static_20260805_003151` MORAI export every prior session has
used, selected via a GT-label mining pass (never previously done) that
found real sustained-turn vehicle actors outside the previously-used
0-400 window (e.g. 6.24 rad yaw range / 401 m path over 430 obs).
800/800, 0 dup, 0 rollback, hash `f5b439e6...b72864b`, orientation
`UNAVAILABLE` for all 10,234 objects (re-confirmed, not assumed) — all
estimator runs used `yaw_measurement_mode="unobserved"`.

**Regime counts** (strict sliding-window classifier, two real jitter/
mis-association failure modes found and fixed with documented,
physically-motivated safeguards — not count-tuned): 61 straight / 19
left_turn / **2 right_turn** / 41 transition / 0 stationary (123 total
segments). `right_turn` shortfall reported honestly as a real property
of this left/counterclockwise-dominant route. Manual validation: 13/123
(10.6%) individually inspected and accepted; remaining 110 pass the
strict automatic classifier only (labeled as such, not misrepresented).

**Phase 13 (offline + live, all 3 conditions, Euclidean+Hungarian
association fixed, `yaw_measurement_mode=unobserved`)**: offline
diagnostics (`collect_offline.py`, direct-from-JSONL, no ROS) — 9,533 /
9,515 / 9,530 match records (linear_kf/ekf/imm), 0 NaN/Inf, deterministic.
Live ROS (fresh publisher/recorder per condition, publisher-isolation
verified via `ros2 topic info --verbose` + a full daemon restart to rule
out stale-cache false positives, exact-PID kill verified twice 2s apart
before each launch) — **800/800 messages for all three conditions**, 0
NaN/Inf. One real operational finding this session: `pkill -f "a\|b"`
silently matches nothing in this environment (pkill's ERE treats `\|` as
a literal pipe, not alternation) — every cleanup after this was switched
to `pgrep`/exact-PID `kill -9` plus double verification.

**Regime-specific residuals** (associated-measurement one-step prediction
residual, not ground-truth error; nearest-measurement-position matched to
manifest segments, ≤2.0 m tolerance): IMM has the lowest median residual
of the three estimators in every regime (e.g. straight: 0.364 m IMM vs.
0.404 m Linear KF vs. 0.429 m EKF). Full per-regime table in README §8-10.

**IMM mu_CV/mu_CTRV by regime — genuine, unresolved, not spun toward
IMM**: mu_CV is *higher*, not lower, during verified left/right turns
than during straight motion (mean mu_CV: 0.561 straight vs. 0.654
left_turn vs. 0.708 right_turn) — the opposite of naive expectation, and
not explained by this task (flagged as an open limitation, plausibly
related to T-7B's own finding that CV's flexibility fits jitter better
than CTRV detects real curvature, but not verified here).
Before/during/after a validated `transition` window shows no clean
dip-and-recover (mu_CV drifts gently upward 0.592 → 0.619 → 0.658 across
all three phases) — messier than T-7B's synthetic validation, reported as
such.

**Figures**: all 9 required figures present under
`~/heven_presentation_assets/motion_dataset/` (4 distribution figures
from before this checkpoint, reused unchanged; 5 new this session:
`estimator_residual_by_motion_regime.png`,
`imm_probability_by_motion_regime.png`, and 4
`representative_{straight,left_turn,right_turn,transition}.png` cases
built from real validated manifest segments — no jitter cherry-picked for
visual drama).

**Live ROS track-level summary** (secondary, no accuracy claim): unique
tracks 701/719/704 (linear_kf/ekf/imm), very-short fraction ~0.28-0.29
all three — nearly identical across conditions as expected, since
association is fixed to Euclidean+Hungarian for all three.

**Tests**: full directly-relevant suite (10 AB3DMOT test modules) —
**195/195 pass**. `git status --short` shows only the same pre-existing
unrelated dirty files from every prior session in this branch, plus this
`STATUS.md` update — no algorithm source file was touched.

**KalmanNet readiness: CASE B.** Useful turning exists (61 straight / 19
left_turn / 41 transition, IMM mechanism validated end-to-end on real
data with 0 NaN/Inf across all three live conditions) but coverage
remains limited — specifically `right_turn` (2 segments, far below a
usable per-regime evaluation split) and only 10.6% manual-validation
coverage. KalmanNet development can proceed for straight/left_turn/
transition, but additional motion data (more clockwise/right-turning
traffic, or a second scene) is recommended before any per-regime
KalmanNet comparison claims completeness. Based on measured coverage, not
chosen because IMM "worked" — §11's finding is in fact an unresolved,
not-obviously-IMM-favorable result.

**Recommended next task**: either (a) begin KalmanNet baseline
implementation scoped to straight/left_turn/transition regimes with the
right_turn limitation stated up front, or (b) investigate §11's
mu_CV-higher-during-turns finding directly (would need real-data
per-object curvature ground truth, not available in this GT-label-mined
dataset as currently used), or (c) capture a second MORAI route with more
right-turning traffic to close the CASE B gap. Not started this session,
per this task's explicit scope (dataset/tooling only).

Not committed/pushed, per this task's instruction; no checkpoint commit
has been made since `4d26e0e` (out of scope for this task, to be done
separately after this final report).

## T-8A result: **PASS**

---

## T-7B: IMM(CV + CTRV) state estimator — PASS, CASE A (mechanism) / CASE B persists (dataset)

Branch: `feat/ab3dmot-tracker`. Implemented an experimental 2-model IMM
(Model 1 = CV Linear KF, Model 2 = CTRV EKF) opt-in state estimator
(`state_estimator="imm"`) and ran a controlled Linear-KF-vs-CTRV-EKF-vs-IMM
comparison, association fixed to Euclidean+Hungarian, `yaw_measurement_mode
="unobserved"` throughout (T-7A.5). Full detail:
`~/heven_presentation_assets/imm/README.md` (`imm_design.md`,
`imm_estimator_compatibility_audit.md`, figures/JSON/JSONL — outside
repo, not committed).

**Architecture**: common 11-dim mixing state
`[x,y,z,yaw,l,w,h,vx,vy,vz,yaw_rate]`; Jacobian-based (not diagonal-copy)
covariance transforms both directions; standard IMM cycle (mixing ->
model predict -> model update -> innovation log-likelihood -> posterior
-> combined output with between-model spread term), not a heuristic.
Transition matrix `[[0.95,0.05],[0.05,0.95]]` (documented baseline, not
tuned); neutral 50/50 initial probabilities.

**Two real bugs found and fixed during implementation** (documented
in-code with the empirical evidence that found them): (1) an initial
CTRV-reseed design that re-derived heading via `atan2(vy,vx)` every
mixing cycle made `yaw_rate` wildly unstable on synthetic constant-turn
testing (±8 rad/s swings on a true 0.6 rad/s turn) — fixed by using the
common state's own already-correctly-circular-mixed `yaw` directly, plus
an isotropic low-speed covariance proxy instead of an arbitrary small
constant that had been collapsing the reseeded speed uncertainty from
~10,000-scale down to 10 and crippling the Kalman gain. (2) A latent
T-7A.5 bug: `LinearKFEstimator`'s own yaw-unobserved reduced update
called filterpy's `update()` with a 6-dim measurement against a
fixed-`dim_z=7` filter — **always raised `ValueError`**, never actually
exercised end-to-end before IMM called it directly (T-7A.5's real runs
only used Linear KF in `"detector"` mode). Fixed with a manual reduced
update mirroring EKF's already-correct math.

**Tests**: new `test_ab3dmot_imm.py` (31 tests: 8 state-transform + 7
probability + 3 mixing + 4 behavior + 9 tracker-integration). Full
related suite **207/207 pass** (176 pre-existing + 31 new); default
Linear KF confirmed unchanged.

**Synthetic validation** (`imm_synthetic_model_probabilities.png`): straight
sequence — mu_CV rises 0.5→0.95; turn sequence — mu_CTRV initially
declines then reverses and gains ground as evidence accumulates;
straight→turn→straight — mu_CV rises to 0.74, **falls to 0.49 during the
turn**, **recovers to 0.63** after — clean, smooth, mechanistically
correct regime-tracking (the strongest evidence the IMM math is right).

**Real-data result (400-frame canonical, hash `63198cd0...4a0c6`,
identical to every prior task)**: global mu_CV mean 0.432 / mu_CTRV mean
0.568 — **CTRV dominates overall**, the opposite of the naive
CASE-B-implies-CV-dominance expectation. Straight-vs-verified-turning
split (T-7A.5's strict classifier, 75 consistent turning runs): mean
mu_CV 0.691 (straight) vs. 0.645 (turning) — **correct direction, modest
effect size**. Interpretation: CTRV's overall dominance is likely driven
more by its greater flexibility fitting real Euclidean-clustering
position jitter (it can represent straight motion as ω≈0) than by
genuine curvature detection — illustrated concretely by a representative
near-stationary track (id 652) whose mu_CV swings 0.14↔0.76 purely from
jitter, honestly labeled as such, not misrepresented as a real turning
transition.

**Residuals**: Linear KF median 0.432m, CTRV EKF 0.495m, IMM 0.437m (IMM
close to Linear KF's own lowest value, consistent with blending toward
whichever model fits best). **Runtime**: IMM costs ~4.1x Linear KF's
total latency (4.83ms vs 1.16ms mean) — expected, both models run every
frame.

**Classification**: **CASE A for the mechanism itself** (synthetic tests
and the real-data straight/turning split both show correct-direction,
smooth regime discrimination) **combined with T-7A.5's CASE B persisting
for this dataset** (turning excitation is real but modest; much of the
real-data model-probability variance reflects detector jitter, not
genuine curvature) — not CASE B (CV does not dominate) or CASE C (the
mechanism does discriminate) or CASE D (no numerical/state-mixing
problems found).

**Limitations**: single static scene with modest turning excitation; no
GT/MOTA/HOTA/IDF1/ID-switch or accuracy claim anywhere; IMM's own
`velocity_covariance`/BEV innovation covariance are documented
approximations (mu-weighted combination of each sub-model's own value,
not rigorously re-derived).

**Recommended next task**: the IMM mechanism is validated and ready to
serve as a baseline for a later KalmanNet comparison, but a real turning-
motion dataset (or a scene with genuine sustained curvature) would make
that comparison far more informative than this canonical recording,
whose own turning content remains modest (T-7A.5/T-7B both independently
confirm this). Consider either (a) a KalmanNet baseline comparison on
this same dataset with the same honest framing, or (b) sourcing/
replaying a MORAI scene with genuine sustained turning before further
motion-model comparisons.

`git status --short` at the end of this task shows only the same
pre-existing unrelated dirty files from every prior session, the T-7B
source changes (`ab3dmot_core.py`, `ab3dmot_config.py`,
`ab3dmot_tracker_node.py`, `ab3dmot_tracker.launch.py`, `CMakeLists.txt`,
`test_ab3dmot_imm.py` new), plus this `STATUS.md` update. Not committed/
pushed, per this task's instruction.

## T-7B result: **PASS**

---

## T-7A.5: Heading observability audit and fix (pre-IMM correction) — PASS, CASE B

Branch: `feat/ab3dmot-tracker`. Audited and corrected heading/yaw
observability handling before implementing IMM, per T-7A's discovered
negative-speed artifact (48.9% of EKF matches had `v < -0.5`). Full
detail: `~/heven_presentation_assets/heading_observability/README.md`
(`heading_source_audit.md`, figures/JSON/JSONL — outside repo, not
committed).

**Root cause, source-confirmed**: `adaptive_euclidean_cluster_node.cpp`
sets the identity quaternion (yaw=0.0) as a structurally-required
placeholder and explicitly marks
`orientation_availability = UNAVAILABLE` on the very next line — per the
message contract's own documentation ("orientation is empty... direction
unknown"). `ab3dmot_ros.py::detected_objects_to_detections()` never reads
this flag and extracts yaw from the placeholder unconditionally. Directly
measured: 100.00% of 4,582 canonical-recording objects have `yaw==0.0`
and `orientation_availability==UNAVAILABLE`.

**Fix**: new `AB3DMOTConfig.yaw_measurement_mode: "detector" (default,
unchanged) | "unobserved"`. In unobserved mode, both estimators perform a
genuine reduced-dimension measurement update (6-dim, yaw row dropped from
`H`/`R` — not a fake yaw with inflated R); EKF additionally gets
one-time, threshold-gated (0.5m displacement) motion-heading
initialization via `atan2(dy,dx)` + `disp/dt` (magnitude/direction
computed separately, so `v >= 0` holds by construction at init, no
post-hoc pi-rotation hack needed).

**Tests**: new `test_ab3dmot_heading.py` (19 tests). Full related suite
**176/176 pass** (157 pre-existing + 19 new); default `detector` mode
confirmed byte-identical to T-7A's own behavior.

**Measured result**: heading initialized for 91.4% of tracks (median 1
observation). Negative-speed artifact shrinks ~5x: fraction `v < -0.5`
55.6% (T-7A detector-yaw) → **10.9%** (T-7A.5 unobserved-yaw); fraction
`v < 0` 68.5% → **23.9%**. Prediction residuals stay close across all
three conditions (Linear KF 0.432m / EKF+detector 0.471m /
EKF+unobserved 0.495m median) — unobserved-yaw's small residual increase
is the honest cost of not fabricating heading from one observation.

**Turning-motion discovery**: a naive single-step classifier initially
found "turning" in the majority of candidate segments but was found,
after direct inspection, to be dominated by detector-clustering jitter,
not real curvature — explicitly discredited, not used. A stricter
classifier (1.0m displacement floor on both adjacent steps, 0.5 rad/s
threshold, requiring ≥2 consecutive same-sign steps) finds **52 verified,
sustained turning segments** (one, track 83, directly inspected and
confirmed physically plausible) against a much larger straight/near-
stationary population.

**Classification: CASE B** — motion heading is genuinely observable, but
the canonical recording's own motion content remains overwhelmingly
straight; CTRV's curved-motion advantage cannot be strongly evaluated on
this specific dataset. IMM (CV+CTRV) may proceed in T-7B, but should
state this limitation explicitly rather than claim a demonstrated
turning-motion advantage from this scene.

**Limitations**: single static scene; 0.5m/1.0m/0.5 rad/s thresholds are
documented, untuned baselines; `v>=0` guaranteed only at initialization,
not every subsequent update (accepted plain-EKF limitation); no GT/MOTA/
HOTA/IDF1 claim anywhere.

**Recommended next task**: **T-7B — IMM estimator** (CV Linear KF + CTRV
EKF, both now heading-observability-corrected), explicitly scoped to
state the CASE B limitation (this dataset's sparse turning content) when
interpreting IMM's model-switching behavior. Not implemented this
session.

`git status --short` at the end of this task shows only the same
pre-existing unrelated dirty files from every prior session, the T-7A.5
source changes (`ab3dmot_core.py`, `ab3dmot_config.py`,
`ab3dmot_tracker_node.py`, `ab3dmot_tracker.launch.py`, `CMakeLists.txt`,
`test_ab3dmot_heading.py` new), plus this `STATUS.md` update. Not
committed/pushed, per this task's instruction.

## T-7A.5 result: **PASS**

---

## T-7A: Linear KF vs EKF (State Estimation chapter begins) — PASS

Branch: `feat/ab3dmot-tracker`. Refactored the experimental AB3DMOT
tracker so the state estimator is pluggable (`Track` now delegates to a
common `predict`/`update`/`position`/`velocity`/`yaw`/`dimensions`/
`*_covariance`/`predicted_bev_*` interface), added an opt-in planar CTRV
EKF, and ran a controlled Linear-KF-vs-EKF comparison with association
**fixed to Euclidean + Hungarian for both runs** (Mahalanobis
deliberately excluded — it consumes estimator covariance directly, which
would confound the comparison). Full detail:
`~/heven_presentation_assets/state_estimation/README.md`
(`linear_kf_audit.md`, `ekf_design.md`, figures/JSON/JSONL — outside
repo, not committed).

**Interface**: `LinearKFEstimator` (default, byte-for-byte the pre-T-7A
KF logic moved out of `Track` unchanged) and `EKFEstimator` (new). New
`AB3DMOTConfig.state_estimator: "linear_kf"|"ekf"` (default unchanged:
`"linear_kf"`). `Track` owns zero estimator-specific math; lifecycle
(`hits`/`time_since_update`/`is_confirmed`/`is_dead`) is untouched and
has zero dependency on which estimator is used (confirmed by audit).

**EKF design**: state `[x,y,z,yaw,l,w,h,v,yaw_rate,vz]` (mirrors Linear
KF's own index layout so both share an identical linear `H`); planar CTRV
transition with closed-form (non-finite-difference) Jacobian, numerically
stable straight-line limit for `|yaw_rate| <= 1e-4`; `P0`/`Q` directly
adapted from Linear KF's own numeric factors, `R` byte-for-byte identical
to Linear KF's own `R` — a defensible, untuned baseline per this task's
explicit scope. Full equations/Jacobian in `ekf_design.md`.

**Tests**: new `test_ab3dmot_ekf.py` (25 tests: 4 Linear KF regression +
11 EKF physics + 3 measurement + 7 tracker-integration). Full related
suite **157/157 pass** (132 pre-existing + 25 new); default Linear KF
behavior confirmed byte-identical to pre-refactor.

**Canonical input**: reused T-4/T-5A/T-5B/T-6's exact persisted 400-frame
recording (400/400, 0 dup, 0 rollback, hash `63198cd0...4a0c6`, identical
to T-6's own recorded hash). Association fixed to
`euclidean_gate_m=3.0`+`hungarian` for both runs, verified via config.

**Key finding — a real detector-input limitation, not an estimator bug**:
the canonical scene's Euclidean-clustering detector outputs **yaw = 0.0
for every single detected object** (verified directly). This means (a)
**zero turning segments exist** in this data (stated explicitly, not
fabricated) — CTRV's theoretical curved-motion advantage is not tested
here; and (b) EKF's `vx=v*cos(yaw)`/`vy=v*sin(yaw)` parametrization gets
pinned near the x-axis, forcing real `-x`-direction motion to appear as
**negative scalar speed** (48.9% of EKF matches have `v < -0.5`) rather
than the independent `vy` Linear KF can express directly.

**Prediction residuals** (associated-measurement vs. prior prediction,
NOT ground-truth error): Linear KF median 0.432m (p99 2.76m) vs. EKF
median 0.471m (p99 2.79m) — very close, Linear KF slightly lower at every
percentile. Yaw residual: Linear KF exactly 0.0 always (yaw state never
perturbed, no yaw-rate coupling); EKF's p99 reaches 0.41 rad, traced to
the Jacobian's position/yaw_rate covariance coupling.

**Track-level**: unique tracks 725 (Linear KF) vs. 736 (EKF) — nearly
identical, small real difference from predicted-position feedback into
Euclidean matching (documented, expected per this task's own note).
Speed distributions similarly close (EKF's Cartesian-converted speed
median 0.94 vs Linear KF's 1.28 m/s).

**Runtime**: EKF's own `predict()` costs ~47% more than Linear KF's
(closed-form Jacobian + matrix products vs. a single filterpy linear
step), but both stay sub-millisecond; per-frame totals are statistically
indistinguishable at this track-count scale (~1.2ms mean either way).

**Limitations**: single static scene with degenerate (always-zero)
detector yaw — materially limits what this comparison can show about
CTRV's actual motion-model benefit; no turning-track figure exists
because none exists in the data; EKF's `velocity_covariance` is not
Cartesian-shaped (documented, deferred — relevant if EKF is ever paired
with Mahalanobis association in a future task); no GT/MOTA/HOTA/IDF1/
ID-switch claim anywhere.

**Recommended next task**: **T-7B — IMM estimator** (not implemented this
session, per explicit scope). Suggested design direction: a 2-model IMM
mixing Linear KF (CV) and the new CTRV EKF, with model-probability
switching informed by innovation likelihood — reuses both estimators
built in T-7A unchanged; note that T-7B should also flag the same
detector-yaw limitation found here, since IMM's CTRV branch will inherit
it identically.

`git status --short` at the end of this task shows only the same
pre-existing unrelated dirty files from every prior session, the T-7A
source changes (`ab3dmot_core.py`, `ab3dmot_config.py`,
`ab3dmot_tracker_node.py`, `ab3dmot_tracker.launch.py`, `CMakeLists.txt`,
`test_ab3dmot_ekf.py` new), plus this `STATUS.md` update. Not committed/
pushed, per this task's instruction.

## T-7A result: **PASS**

---

## T-6: FINAL association-metric comparison (GIoU / Euclidean / Mahalanobis Hybrid 10m) — PASS — Association chapter closed

Branch: `feat/ab3dmot-tracker`. Final, presentation-ready three-way
association-metric comparison, replacing T-4's original pure-Mahalanobis
comparison (superseded by T-5A/T-5B's finding of covariance-driven
long-distance over-association). Association-layer only; no new code —
reuses the exact T-4/T-5B metric/gate implementations. Full detail:
`~/heven_presentation_assets/association_final/README.md` (figures/JSON/
JSONL — outside repo, not committed).

**Configuration** (all three, matcher=hungarian, same canonical
400-frame input, same KF/`Q`/`R`/`P0`/lifecycle/`min_hits=1`/`max_age=2`):
A. GIoU (`giou_gate=0.0`); B. Euclidean (`euclidean_gate_m=3.0`);
C. Mahalanobis Hybrid (`mahalanobis_gate=11.62`,
`mahalanobis_max_distance_m=10.0` — T-5B's recommended baseline).

**Key result — the corrected baseline no longer produces the pathological
tail**: large-jump counts `>10m`/`>20m`: GIoU 0/0, Euclidean 0/0,
**Mahalanobis Hybrid 0/0** (T-4's pure-Mahalanobis had 201/95). Speed
p99/max: GIoU 11.3/22.4, Euclidean 18.0/27.2, **Mahalanobis Hybrid
48.0/90.0 m/s** (T-4's pure-Mahalanobis had 217.7/330.5 m/s) — far more
physically reasonable, though still heavier-tailed than the other two,
reflecting jumps in the now-bounded 3-10 m range.

**Track population**: unique tracks GIoU 2,739 / Euclidean 725 /
Mahalanobis Hybrid **505** (vs. T-4's pure-Mahalanobis 446 — the 10m cap's
known continuity cost, already quantified in T-5B); very-short fraction
86.97% / 42.21% / **25.54%**. Mahalanobis Hybrid retains the lowest churn
of the three, at a reduced (vs. pure Mahalanobis) but still real margin.

**Runtime** confirms T-3/T-4's ordering: construction mean GIoU 20.6 ms >>
Mahalanobis Hybrid 7.7 ms >> Euclidean 0.67 ms; Hungarian solver
negligible (<0.04 ms) for all three; the hybrid distance check itself
adds <1 ms.

**Assignment disagreement**: all three metric pairs disagree on the
majority of the 400 frames (88.75% / 90.75% / 87.5%), consistent with
T-4's original finding — metric choice remains a first-order factor in
this tracker's behavior.

**Final takeaway (no accuracy winner claimed, no GT)**: GIoU is
geometry-rich but computationally expensive and highest-churn in this
scene; Euclidean is cheap and interpretable but uncertainty-blind;
Mahalanobis Hybrid is uncertainty-aware while using an absolute distance
cap to prevent the previously-observed covariance-driven long-distance
tail, at moderate computational cost and the lowest churn of the three.

**Limitations**: single static MORAI scene; canonical-recording hash is
stable-under-current-procedure, not byte-identical to T-4's original
(unrecoverable) hash — same documented caveat as T-5B; no MOTA/HOTA/IDF1/
ID-switch claim anywhere.

**This closes the Association chapter** (T-1 through T-6): metric choice
(GIoU/Euclidean/Mahalanobis), assignment solver (greedy/Hungarian), and
Mahalanobis over-association mitigation (hybrid gate) have each been
implemented, tested, and measured on a shared canonical input, with a
corrected, presentation-ready final comparison now in place.

**Recommended next task**: State-estimator experiments — Linear KF → EKF
→ IMM comparison, building on the now-finalized association layer. Not
started this session, per explicit scope.

`git status --short` at the end of this task shows only the same
pre-existing unrelated dirty files from every prior session (no new
source changes — T-6 reused T-4/T-5B's existing implementation
unmodified), plus this `STATUS.md` update. Not committed/pushed, per
this task's instruction.

## T-6 result: **PASS**

---

## T-5B: Mahalanobis hybrid (chi-square + absolute distance) gate — PASS

Branch: `feat/ab3dmot-tracker`. Implemented and evaluated a
covariance-aware hybrid association gate for the experimental Mahalanobis
path, motivated by T-5A's CASE C finding. **Association-layer only**: KF,
`Q`/`R`/`P0` initialization, lifecycle, `min_hits`/`max_age`, and Hungarian
are all unchanged. Full detail:
`~/heven_presentation_assets/hybrid_gate/README.md` (figures/JSON/JSONL —
outside repo, not committed).

**Implementation**: new `AB3DMOTConfig.mahalanobis_max_distance_m` field
(default `0.0` = disabled, exactly reproduces pure T-4/T-5A Mahalanobis
behavior — verified by unit test). When `> 0`, `_associate()`'s
Mahalanobis branch requires a pair to satisfy **both**
`d_M^2 <= mahalanobis_gate` **and** `d_E (BEV Euclidean) <=
mahalanobis_max_distance_m` (reusing the existing
`_euclidean_bev_distance_matrix`) to be valid. The Hungarian cost matrix
is always raw `d_M^2`, never modified by the cap — the cap is a validity
gate only. New `mahalanobis_max_distance_m` launch arg/node parameter.

**Tests**: new `test_ab3dmot_hybrid_gate.py` (13 tests, including a
regression test reproducing the T-5A failure mode directly — inflate a
track's `P` by 1e6, confirm the cap rejects a 40 m match that pure
Mahalanobis alone would accept). Full related suite **120/120 pass** (107
pre-existing + 13 new).

**Cap sweep** (no cap / 3 m / 5 m / 10 m, same canonical 400-frame input,
`mahalanobis_gate=11.62` fixed, Hungarian fixed): candidate pairs removed
by the cap: 0 / 25,682 / 14,824 / 7,739. Large-jump counts (`>3/5/10/20m`):
no cap 554/345/201/95; 3m gate **0/0/0/0**; 5m gate 159/0/0/0; 10m gate
343/157/0/0 — **every cap fully eliminates jumps beyond its own
threshold**, as designed. The previously-diagnosed ~75 m event (T-5A's
track id 33/85 family, `hits=1`, `d_M^2=5.18` well inside the chi-square
gate) is reproduced identically here and rejected by all three caps.

**Continuity/churn tradeoff** (real, substantial): unique tracks no cap
446 → 3m 853 → 5m 629 → 10m 505; very-short fraction 0.90% → 45.72% →
35.93% → 25.54%. Suppressing the over-association tail costs continuity —
a tight (3m) cap is *more* restrictive than either GIoU or plain
Euclidean alone (both AND-conditions must pass), producing more churn
than T-4's plain Euclidean metric (725 unique / 42.2% very-short).
Speed p99/max drop from 217.7/330.5 m/s (no cap, physically implausible)
to 18.0-48.0 / 27.2-90.0 m/s across the three capped conditions.

**Runtime**: the hybrid distance check itself is cheap (<1 ms mean, p99
under 5 ms) and remains negligible relative to Mahalanobis metric
construction (~9-10 ms mean), consistent with T-3/T-4/T-5A's finding that
metric construction dominates association latency.

**Recommended baseline for subsequent experiments (not "optimal")**:
**10 m** — fully suppresses the `>10m`/`>20m` tail (T-5A's most
implausible evidence) while retaining continuity closest to the no-cap
baseline (505 vs. 446 unique tracks) of the three capped options; 3m/5m
remain reasonable alternatives if a future task prioritizes maximum
suppression over continuity.

**Limitations**: single static MORAI scene; cap sweep is a sensitivity
sweep, not a tuned search; cross-condition track-identity comparison
(`matches_lost_vs_no_cap`) is only reliable in the earliest frames before
independently-run conditions' populations diverge — the within-condition
`hits=1`-fraction analysis is the more robust evidence; no GT/MOTA/HOTA/
IDF1/ID-switch claim anywhere.

**Recommended next task**: either (a) proceed to a controlled comparison
using the recommended 10m hybrid-gate baseline as the new Mahalanobis
reference point (e.g. re-running T-4-style GIoU/Euclidean/Mahalanobis-
hybrid comparison), or (b) begin the deferred KF/EKF/IMM comparison now
that the association-layer over-association tail is characterized and
mitigated; not started this session, per explicit scope.

`git status --short` at the end of this task shows only the same
pre-existing unrelated dirty files from every prior session, the T-5B
source changes (`ab3dmot_config.py`, `ab3dmot_core.py`,
`ab3dmot_tracker_node.py`, `ab3dmot_tracker.launch.py`, `CMakeLists.txt`,
`test_ab3dmot_hybrid_gate.py` new), plus this `STATUS.md` update. Not
committed/pushed, per this task's instruction.

## T-5B result: **PASS**

---

## T-5A: Mahalanobis gating characterization (diagnostic only) — PASS, CASE C

Branch: `feat/ab3dmot-tracker`. Diagnostic-only task (no algorithm/gate/KF/
Q/R/lifecycle/Hungarian change) characterizing *why* T-4's Mahalanobis
association reduced track churn so dramatically. Ran the exact frozen T-4
config (`association_metric=mahalanobis`, `matcher=hungarian`,
`mahalanobis_gate=11.62`, same canonical 400-frame input, same KF/lifecycle)
through an additive diagnostic subclass (`DiagnosticTracker`, snapshots
per-track `P`/`S`/gate-radius/age/hits before `_associate()`, never
touches production code). Full detail:
`~/heven_presentation_assets/mahalanobis_analysis/README.md` (figures/CSV/
JSON — outside repo, not committed).

**Key finding**: the Mahalanobis gate is never a fixed-meter radius —
major-axis gate radius ranges from median 6.05 m to p99 58.85 m (max
118 m) across 5,177 track-frame samples, directly explaining T-4's
candidate-count inflation (median 19% of valid pairs/frame come from the
top-10%-highest-uncertainty tracks alone). Of 4,136 accepted matches,
554 (13.4%) produced a KF position jump >3 m. **Root cause of the most
extreme jumps, verified by direct inspection**: of the top-15 largest
jumps, **15/15** occur on tracks with `hits=1` (never yet corrected by a
real measurement since birth) — the reference AB3DMOT KF's own initial
velocity-uncertainty scaling (`P[7:,7:] *= 1000`) propagates into a
50-115 m position gate after just 1-2 predict-only frames, before any
real evidence has constrained the track. This exactly reproduces and
explains T-4's originally observed ~40 m/~57 m single-step event (track
id 85, hex UUID `...55`) — traced precisely, not assumed.

**Classification: CASE C (both effects materially present).** 87% of
accepted matches have `d_M^2 < 2` (tight, legitimate) and the tracks
providing T-4's continuity benefit are specifically the *low*-covariance
population (long-lived top-20: mean P-trace 13.2 vs. high-covariance
top-20's much larger values); but a real, precisely-localized
over-association mechanism exists among never-corrected new tracks,
driving nearly all of the most extreme jumps while remaining short-lived
and almost entirely disjoint from the long-lived population (1/180
overlap) — i.e. the churn-reduction benefit and the over-association risk
are largely two different track populations, not the same tracks
achieving continuity by jumping.

**Recommended next step**: a covariance-dependent + absolute-distance
hybrid gate (association-layer only, no KF/Q/R change) — targets the
localized never-corrected-track failure mode directly without penalizing
the 87% of matches that are already tight, and without the larger
blast radius of a KF-level covariance cap or Q/R recalibration. Not
implemented, per this task's explicit scope.

`git status --short` at the end of this task shows only the same
pre-existing unrelated dirty files from every prior session, plus this
`STATUS.md` update — no tracking-algorithm source file was touched.
Not committed/pushed, per this task's instruction.

## T-5A result: **PASS**

---

## T-4: Association-metric comparison (GIoU vs Euclidean vs Mahalanobis) for experimental AB3DMOT — PASS

Branch: `feat/ab3dmot-tracker`. Added two opt-in association-metric
alternatives to the experimental AB3DMOT tracker (`AB3DMOTConfig.association_metric`:
`"giou_3d"` default, plus new `"euclidean"` and `"mahalanobis"`) and ran a
controlled GIoU-vs-Euclidean-vs-Mahalanobis comparison with the assignment
solver held fixed at Hungarian (T-3) for all three runs. Full detail:
`~/heven_presentation_assets/association_metrics/` (`README.md`,
`association_metric_audit.md`, `association_metric_input_validation.md`,
figures/JSON/JSONL — outside repo, not committed).

**Implementation**: `SUPPORTED_METRICS` extended to
`("giou_3d", "euclidean", "mahalanobis")`; new `euclidean_gate_m` (default
3.0 m, documented neutral baseline) and `mahalanobis_gate` (default
11.62, reused directly from this repo's own Autoware audit in
`tracking_architecture.md`: "99.6% confidence, chi-square 2 DOF") config
fields. `Track` gained two read-only accessors
(`predicted_bev_position`, `predicted_bev_innovation_covariance` — the
latter reuses the reference submodule's own
`KF.compute_innovation_matrix()`, `S = H P H^T + R`, rather than
re-deriving it). New free functions `_euclidean_bev_distance_matrix`
(BEV `[x,y]` center distance) and `_mahalanobis_bev_distance_matrix`
(`d_M^2 = y^T S^-1 y` via `np.linalg.solve`, large-finite-sentinel
handling for singular/ill-conditioned `S` instead of `np.inf`, which
would make `linear_sum_assignment` infeasible). `_associate()` now
dispatches metric construction and gate direction per
`config.association_metric`; GIoU's own code path, gate semantics, and
sign convention are byte-for-byte unchanged from T-3. Both new metrics
use only the BEV `[x, y]` measurement subspace (not yaw/size/z),
matching each other and the Autoware Mahalanobis-gate precedent's own
DOF choice. New `association_metric`/`euclidean_gate_m`/`mahalanobis_gate`
launch args on `ab3dmot_tracker.launch.py`; node parameter wiring added
in `ab3dmot_tracker_node.py`.

**Tests**: new `test_ab3dmot_association_metrics.py` (20 tests: 6
Euclidean + 8 Mahalanobis + 5 cross-metric, covering the 19 required
cases plus one extra size/yaw-blindness check) + full related suite
**107/107 pass** (87 pre-existing T-1/T-3 + 20 new), default
GIoU+greedy behavior confirmed unaffected by the metric-dispatch
refactor. `colcon build` clean.

**Canonical input**: T-3's original canonical recording/hash could not be
reused — it lived only in that session's now-cleaned scratchpad and was
never persisted. A fresh, procedurally-identical capture was made this
session (same Ground-ON Euclidean pipeline, same MORAI static scene, same
replay parameters): 400/400 messages, 0 duplicate timestamps, 0 clock
rollbacks, mean 11.455 objects/frame (matches T-3's recorded value to
three decimal places). Content hash differs from T-3's recorded value
because T-3's own hashing script is equally unrecoverable — reported
honestly rather than claimed as an exact match; full explanation in
`association_metric_input_validation.md`. All three T-4 runs (A/B/C
below) share this one capture, satisfying the controlled-comparison
requirement regardless.

**Controlled runs**: three independent `AB3DMOTTracker` instances (GIoU,
Euclidean, Mahalanobis; matcher fixed to Hungarian), each driven by a
fresh `ros2 launch ab3dmot_tracker.launch.py` process, replaying the
identical canonical pickle byte-for-byte into
`/ad/perception/objects/detected` (bypassing ground segmentation/
Euclidean re-run). Publisher isolation (`ros2 topic info --verbose`,
exactly 1 publisher) verified before each run; 400/400 output messages
in all three runs.

**Match-level findings**: Mahalanobis's uncertainty-normalized gate
admits far more gate-valid candidate pairs per frame than GIoU (6.7x) or
Euclidean (2.2x). Assignment sets differ frequently between every metric
pair (88.75% GIoU-vs-Euclidean, 87% Euclidean-vs-Mahalanobis, 91%
GIoU-vs-Mahalanobis, out of 400 frames). Track populations across the
three independent runs first diverge at frame 26 (much earlier than
T-3's greedy-vs-Hungarian divergence at frame 121, since metric choice
affects which detections spawn new tracks far more directly than matcher
choice). A concrete representative case (frame 37) shows the same raw
9.38 m BEV gap rejected by Euclidean's fixed 3 m gate but accepted by
Mahalanobis (d_M^2 = 9.65 < 11.62 gate) because that track's predicted
covariance was large enough to make the gap statistically unsurprising.

**Track population** (400 frames each): unique tracks GIoU 2,739 /
Euclidean 725 / Mahalanobis 446; very-short-track fraction (<=2 obs)
86.97% / 42.21% / 0.90%; longest continuous track 400 / 400 / 130 obs.
Euclidean and especially Mahalanobis produce far fewer, much
longer-lived tracks than GIoU — a direct, expected consequence of GIoU
requiring actual box overlap (broken by any detector jitter) vs. the
other two metrics' looser, shape-blind tolerance. Not claimed as "more
accurate" (no GT) anywhere.

**Velocity/state**: KF-reported speed distributions differ substantially
(median 0.00 / 1.28 / 3.41 m/s; p99 11.27 / 18.05 / 217.66 m/s).
Mahalanobis's extreme tail was inspected directly (not just tabulated):
one track's published position jumps ~40 m in a single 0.15 s step,
architecturally explained by the same uncertainty-normalized-gate
mechanism above — reported as an observed, explained behavioral
difference, not resolved as correct or incorrect without identity GT.

**Runtime**: Euclidean's metric construction is ~20-25x cheaper than
GIoU's (a single 2D norm vs. full 3D polygon geometry); Mahalanobis sits
in between (a 2x2 linear solve per pair, cheaper than GIoU but pricier
than raw Euclidean). The Hungarian solver itself remains negligible
relative to metric construction for all three metrics, consistent with
T-3's own finding. Answers this task's own research question directly:
yes, both alternative metrics substantially reduce metric-construction
cost relative to 3D GIoU.

**Limitations**: single static MORAI scene; canonical-recording hash
mismatch vs. T-3 (explained above, doesn't affect T-4's own internal
validity); `euclidean_gate_m`/`mahalanobis_gate` are documented
neutral/reused baselines, not tuned against this session's own tracking
outcomes; match-level diagnostics ran the tracker core directly
(`lidar_link` frame) while track-level/velocity/runtime tables use the
live ROS-published (`odom` frame) recordings — not claimed numerically
identical to each other; GIoU's own construction/solver split reuses
T-3's previously-measured baseline rather than being independently
re-measured in this session's exact format; no MOTA/HOTA/IDF1/ID-switch
claim anywhere.

**Recommended next task**: identity-ground-truth-free ways to
characterize the Mahalanobis speed-tail/uncertainty-growth behavior
further (e.g. bounding `P` growth during long coasts), or begin the
`giou_gate`/`euclidean_gate_m`/`mahalanobis_gate` real-data calibration
flagged as needed since T-1's integration-decisions doc (not started
this session, per explicit scope).

`git status --short` at the end of this task shows only the same
pre-existing unrelated dirty files from every prior session, the source
changes listed above (`ab3dmot_core.py`, `ab3dmot_config.py`,
`ab3dmot_tracker_node.py`, `ab3dmot_tracker.launch.py`, `CMakeLists.txt`,
`test_ab3dmot_association_metrics.py` new), plus this `STATUS.md` update.
Not committed/pushed, per this task's instruction.

## T-4 result: **PASS**

---

## T-3: Greedy vs Hungarian assignment for experimental AB3DMOT — PASS

Branch: `feat/ab3dmot-tracker`. Added an opt-in Hungarian-assignment
option to the experimental AB3DMOT tracker (`ab3dmot_core.py`) and ran a
controlled Greedy-vs-Hungarian comparison, holding detector/ground
segmentation/GIoU/gate/KF/lifecycle/ROS contract fixed — the only
experimental variable was the assignment strategy. Full detail:
`~/heven_presentation_assets/association/` (`association_audit.md`,
`association_input_validation.md`, `association_difference_summary.md`,
`README.md`, figures/CSVs/JSONLs — outside repo, not committed).

**Implementation**: `AB3DMOTConfig.matcher` gained `"hungarian"` alongside
the unchanged default `"greedy"` (`SUPPORTED_MATCHERS = ("greedy",
"hungarian")`). New `_hungarian_matching()` (`scipy.optimize.linear_sum_assignment`,
already available via the same `~/py-ab3dmot-deps` staged environment used
for `filterpy`/`scipy` since T-2). `_associate()` now builds
`cost_matrix = -giou_matrix` once and dispatches on `config.matcher`; the
post-hoc `giou_gate` filter applies identically to both matchers' raw
output, so a below-gate pair is always rejected regardless of matcher —
Hungarian cannot force an invalid pair through. Everything else (GIoU
construction, predict/update, birth/death, output filtering) untouched.
Also added an opt-in, additive-only `collect_diagnostics` flag on
`AB3DMOTTracker` (never used by production/ROS code) for Phase 11's
match-level instrumentation. New `matcher` launch arg on
`ab3dmot_tracker.launch.py` (default `"greedy"`, `matcher:=hungarian`
usage), YAML comment updated (value unchanged).

**Tests**: new `test_ab3dmot_association.py` (24 tests: the 10 required
correctness cases including #10's classic globally-suboptimal-greedy
synthetic matrix, verified Hungarian achieves strictly higher total
affinity; plus config validation and no-duplicate-assignment checks) +
9-test `HungarianRegressionParityTest` class re-running the existing
greedy-baseline scenarios (stable ID, two-track separation, coast/
reappear, max-age deletion, nearest-track association, m/s velocity, real-
dt, yaw wraparound) with `matcher="hungarian"`, verifying identical
invariants. **Full related suite: 87/87 pass** (63 pre-existing + 24 new),
default-greedy behavior confirmed byte-identical to before (regression
suite unchanged). `colcon build` clean.

**Canonical input**: one pre-captured Ground ON Euclidean `DetectedObjects`
recording (400 frames, 0 duplicate timestamps, 0 clock rollbacks, raw
CDR-serialized bytes saved) replayed byte-for-byte into two fresh,
isolated AB3DMOT instances (Greedy, then Hungarian) — publisher isolation
(`ros2 topic info --verbose`, exactly 1 publisher) verified before/after
each; 400/400 output messages each run.

**Match-level findings**: assignments differ in 133/400 frames (33.25%);
track populations first diverge at frame 121 (before that, matrices are
byte-identical between runs). On the 256 frames still structurally
comparable (identical GIoU matrix), Hungarian's **raw** (pre-gate) total
affinity is >= greedy's on **all 256, zero exceptions** — the optimality
guarantee verified on real scene data, not just the unit test. Post-gate,
Hungarian was lower on 14/256 — a verified, explained gate-interaction
artifact (the gate strips a different specific low-affinity pair per
matcher's differing raw solution), not a guarantee violation.

**Track population**: Greedy 2,722 unique tracks / Hungarian 2,741
(+0.7%); mean tracks/frame 18.40 vs 18.46; lifetime distributions
essentially identical (very-short-track fraction, defined as <=2
observations: 87.47% vs 86.98%). Small, real differences — not claimed as
"more accurate" (no GT).

**Latency**: mean 25.11ms (Greedy) vs 25.36ms (Hungarian), materially
unaffected by matcher choice. Separated instrumentation: GIoU-matrix
construction (~19-20ms mean) dominates total association cost by ~2
orders of magnitude over either solver (Hungarian solver mean 0.03ms,
actually faster than greedy's 0.28ms at this matrix scale).

**Limitations**: single static MORAI scene; Phase 11 diagnostics ran the
tracker core directly (no ROS, `lidar_link` frame) for the relative
greedy-vs-Hungarian comparison, not numerically identical to the
ROS/`odom`-frame Phase 8 runtime-comparison messages; `giou_gate=0.0`/
`min_hits=1`/`max_age=2` untuned baseline, not re-examined; no RViz
live-screenshot evidence (same documented no-screenshot-capability
limitation as prior sessions); no MOTA/HOTA/IDF1/ID-switch claim anywhere.

**Recommended next task**: T-4 or equivalent — Euclidean vs Mahalanobis
metric comparison, or begin addressing the `giou_gate` calibration flagged
as needing real-data tuning since T-1's integration-decisions doc (not
started this session, per explicit scope).

`git status --short` at the end of this task shows only the same
pre-existing unrelated dirty files from every prior session, the source
changes listed above (`ab3dmot_core.py`, `ab3dmot_config.py`,
`ab3dmot_tracker.launch.py`, `ab3dmot.yaml`, `CMakeLists.txt`,
`test_ab3dmot_association.py` new), plus this `STATUS.md` update. Not
committed/pushed, per this task's instruction.

## T-3 result: **PASS**

---

## T-2B: Resolve apparent contradiction in T-2 (byte-identical detections vs 5-7% tracking difference) — CASE A, T-2 causal claim corrected

Branch: `feat/ab3dmot-tracker`. T-2 reported Ground OFF vs ON Euclidean
output as "byte-identical" (40-frame sample) yet Autoware/AB3DMOT tracking
population/churn differed ~5-7% (400-frame sample). This task resolved the
apparent contradiction before any association/Hungarian work starts. Full
detail: `~/heven_presentation_assets/ground_segmentation/`
(`t2b_detection_identity_check.md`, `t2b_tracking_repeatability.md`/`.csv`,
outside repo, not committed). No algorithm changed, no tracker tuned, no
detector parameter changed, no Hungarian work started.

### A. Was "byte-identical" literally true?

No, and it was never claimed to be at the raw-bytes level — but this
session verified this explicitly rather than assuming. A fresh recorder
(`record_full_detected.py`) captured the full CDR-serialized
`DetectedObjects` bytes (`rclpy.serialization.serialize_message`) plus
every decoded field the message actually has: `existence_probability`,
full `classification[]`, `pose.position`/`orientation`/`covariance`,
`has_position_covariance`, `orientation_availability`, `twist` (linear/
angular/covariance), `has_twist`/`has_twist_covariance`, `shape.type`,
`shape.dimensions`, `shape.footprint`. Confirmed directly:
**`DetectedObject` has no UUID/`object_id` field** (only `TrackedObject`
does) — the "UUID if present" check is N/A for detections.

### B. Detection stream hashes / identity result

Two fresh, isolated 40-frame Ground OFF / ON captures (same replay
parameters as T-2's original Phase 4). Per-message and whole-recording
SHA-256 content hash (canonical JSON of every field above, **excluding**
`header.stamp`) were **identical for all 40/40 messages**, whole-recording
hash `9d1bce62...0264a1c` both conditions — a **strictly stronger**
confirmation than T-2's original claim (T-2 only checked
`dimensions`/`position`/`class_name`/`score`; orientation, covariance,
twist, existence_probability, and footprint were never checked before and
are now also confirmed identical for this sample). Raw serialized bytes
differ only because of `header.stamp` (genuinely different real
replay-wall-clock times between the two runs, not a bug).

**However — and this is the key finding — re-examining the actual 400-frame
detection streams that fed the tracking comparison (not the 40-frame
sample) found they were NOT identical**: 94/400 frames (23.5%) have
different `object_count` between OFF and ON, starting at frame 73/400
(first 40 frames genuinely match, consistent with the sample above).
Ground ON produces fewer detections in 91/94 differing frames, mean signed
difference -0.515 objects/frame (~4.3% of the OFF mean of 11.97) —
directionally and roughly magnitude-consistent with the previously
reported ~5-7% tracking-population reduction. Direct inspection of 4
differing frames (73, 74, 118, 368) shows the specific missing objects
under Ground ON are consistently the lowest-z clusters in each frame
(e.g. frame 73: OFF has an extra cluster at z=-0.97 absent from ON) —
the expected ground-adjacent-cluster-removal mechanism, verified directly,
not just inferred.

### C. Run-to-run tracker variability (identical-input replay)

The Ground-OFF 40-message canonical recording's raw serialized bytes
(Section A/B capture) were replayed directly into `/ad/perception/objects/detected`
(bypassing ground segmentation and Euclidean entirely) 3 times each for
Autoware and AB3DMOT, with **both trackers fully killed and relaunched
fresh before every run** (genuine independent replicates) and publisher
isolation confirmed before each. Stamps were rebased by a constant offset
to land near current wall-clock (exact relative deltas preserved) after an
initial attempt with stale stamps triggered Autoware's `InputManager`
"Resetting the latest measurement time" warning and dropped outputs — a
replay-methodology artifact (Autoware's tracker runs an internal
`publish_rate: 10.0` timer correlated with real elapsed time, not a 1:1
per-message publish), not tracker non-determinism; fixed before the
reported runs. **Result: every population/churn metric (mean/median/max
tracks-per-frame, unique tracks, births, disappearances, lifetime
distribution, longest track) was byte-for-byte identical across all 3 runs
for both trackers** — 0% run-to-run variability. AB3DMOT latency showed
only normal small scheduling jitter (mean 7.00/7.29/6.51 ms across the 3
runs), two orders of magnitude too small to explain a population-level
difference.

### D. Rollback/reset differences

None in either the original 400-frame OFF/ON recordings (1 reset segment
each, re-confirmed this session) or the 3 canonical-replay runs (1 segment
each). Not a factor.

### E. TF/message-loss differences

Re-checked the original 400-frame OFF/ON tracker logs: zero
warnings/errors/TF-failures/duplicate-timestamp-skips/reconstruction
events in either condition's final (clean) run. Message accounting:
0 missing, 0 extra, 0 duplicate `detected`→`autoware`/`ab3dmot` stamp
matches in both conditions (400/400/400 each); replay pacing statistically
identical (mean inter-arrival 164.32ms both conditions). No TF or
message-loss difference between OFF and ON was found. (One recorder-side,
not tracker-side, message-loss artifact was hit and fixed *during this
session's own new captures* — a CPU-heavy per-callback recorder lagged the
publisher's limited history depth and silently dropped early messages;
fixed by deferring all decoding/hashing out of the subscription callback.
This is a T-2B tooling artifact, not evidence about the original T-2 runs,
which used a different, already-lightweight recorder.)

### F. Final classification: **CASE A**

Detector streams were **not** actually identical over the full 400-frame
window that produced the tracking-population comparison (they were
identical only for a 40-frame sub-sample). The field difference
(`object_count`, driven by low-z/ground-adjacent clusters) plausibly and
directionally explains the previously observed tracking difference. Combined
with C's proof of 0% tracker-side run-to-run variability under genuinely
identical input, there is no remaining candidate explanation other than
the detector-input difference itself.

### G. T-2 claims that remain valid

- Ground-segmentation node build/launch/runtime validation (Phase A/B of
  T-2's original report).
- Bounded point-cloud validation (ground fraction ~53%, 0 NaN/Inf, the
  voxel-downsampling explanation for `ground+nonground < cropped`).
- The 40-frame Euclidean-identity claim itself, **as scoped to those 40
  frames** — it was correct, just not generalizable to the full 400-frame
  run as originally implied.
- All raw direct measurement numbers in the original tables (tracks/frame,
  unique tracks, births/deaths, speed distributions, latency) — these are
  real recorded values and remain accurate as measurements.

### H. T-2 claims withdrawn or softened

- **Withdrawn**: any implication that Ground OFF vs ON Euclidean output
  was identical over the *400-frame* tracking-comparison window — it was
  not.
- **Softened**: the ARCHITECTURAL INTERPRETATION section's "genuine,
  measured effect... real, modest (~5-7%) reduction... a modest
  stabilizing effect" framing, which treated the tracking difference as an
  unexplained but presumed-causal side-effect of ground segmentation on
  the *trackers*. T-2B replaces this with a direct, verified explanation:
  ground segmentation changes Euclidean's own detection output later in
  the replay (via removing low-z clusters), and *that* detection-count
  difference is what the trackers faithfully reproduce — not an effect
  "on the trackers" independent of their input.

### I. Safe to start T-3 (Greedy vs Hungarian)?

**Yes.** The contradiction is resolved with a verified causal mechanism
(Case A), tracker-side determinism is proven under identical input (C),
and no tracker/association code was touched by this investigation. T-3
can proceed on the existing, unmodified association/tracker code with the
corrected understanding that any future Ground OFF/ON-style comparison
must verify full-window detector-input identity before attributing
downstream differences to a specific upstream stage.

### J. Best corrected presentation interpretation

Ground segmentation **does** change Euclidean's detection output under the
current z-crop configuration, but only for a subset of frames (those where
a cluster sits near the `min_z_m: -1.0` crop boundary) — not uniformly, and
not because Euclidean's crop makes ground segmentation redundant (the
original T-2 40-frame-derived conclusion). The downstream tracking
population difference is a faithful, expected consequence of that detection
difference, not an independent or surprising tracker-level effect. See
`~/heven_presentation_assets/ground_segmentation/README.md`'s "T-2B
CORRECTION" section for the full writeup.

`git status --short` at the end of this task shows only the same
pre-existing unrelated dirty files from every prior session, plus this
`STATUS.md` update. Not committed/pushed, per this task's instruction.

---

## T-2 / T-2A: Ground-segmentation restoration — PASS

Branch: `feat/ab3dmot-tracker`. Goal: measure Euclidean/tracking behavior
with the repository's existing ground-segmentation stage restored
(Ground OFF vs Ground ON), holding detector/tracker/association/KF/replay
data fixed. Full detail: `~/heven_presentation_assets/ground_segmentation/`
(`ground_path_audit.md`, `ground_segmentation_blocker.md`, outside repo,
not committed).

**Source audit (T-2, unchanged)**: no code/launch change is needed. Ground
ON = launch `ground_segmentation.launch.py backend:=ransac` (input already
matches the MORAI replay topic/format) + `euclidean_clustering.launch.py`
**without** the `finite_input_topic` override every prior experiment used
(its own default already is `/ad/perception/lidar/nonground`). Ground OFF
= the already-established bypass, unchanged. 110/110 existing tests pass
unmodified.

**Resumed and completed this session** after the 8 consolidated system
dependencies (`ros-humble-autoware-lanelet2-extension`,
`ros-humble-autoware-lanelet2-utils`, `ros-humble-autoware-point-types`,
`ros-humble-autoware-sensing-msgs`, `ros-humble-autoware-vehicle-msgs`,
`ros-humble-cv-bridge`, `ros-humble-point-cloud-msg-wrapper`,
`librange-v3-dev`) were installed with root between sessions. Full A-M
report below; full detail/figures/raw data in
`~/heven_presentation_assets/ground_segmentation/` (`README.md`,
`ground_segmentation_summary.md`, `ground_point_counts.csv`, 8 PNGs — all
outside the repo, not committed).

### A. Remaining local package builds

`managed_transform_buffer` (empty/untracked dir in-repo — cloned fresh
from `dependencies.repos`' pinned commit `c77fe4e6b...` to a scratch dir
outside the repo, not into the tracked tree), `autoware_pcl_extensions`,
`autoware_utils_diagnostics` (split package, needed transitively via the
monolithic `autoware_utils`'s own installed header but not declared
directly in `autoware_pointcloud_preprocessor`'s package.xml — same class
of gap as `managed_transform_buffer`), `autoware_pointcloud_preprocessor`,
`autoware_ground_segmentation` — **all built successfully**, no upstream
source patched. Two build-time issues found and fixed, both local
wiring/resource issues, not new missing dependencies: (1)
`autoware_utils_diagnostics`'/`diagnostic_updater`'s headers weren't on
the compiler's search path despite being built and on
`AMENT_PREFIX_PATH`/`CMAKE_PREFIX_PATH` — fixed via an explicit
`-DCMAKE_CXX_FLAGS="-I<installed include dir>"` addition (same
already-established technique as the earlier Boost header-staging
precedent — adding an include path for an already-built/installed
dependency, not patching source); (2) full `-j16` parallel compilation of
these heavy Autoware C++ template files OOM-killed `cc1plus`
(`fatal error: Killed signal terminated program cc1plus`, 15GB RAM
system) — fixed via `MAKEFLAGS=-j2`/`CMAKE_BUILD_PARALLEL_LEVEL=2`, not a
dependency issue.

### B. Ground node launch — actually run, not just built

`ros2 pkg executables autoware_ground_segmentation` confirmed
`ransac_ground_filter_node` (plus `ray_ground_filter_node`,
`scan_ground_filter_node`, unused). Launched standalone (no replay) via
`ground_segmentation.launch.py backend:=ransac`: process stayed alive,
correct remaps confirmed (`input:=/ad/perception/lidar/cropped`,
`output:=/ad/perception/lidar/nonground`,
`debug/ground/pointcloud:=/ad/perception/lidar/ground`), `ros2 node list`
showed `/ad_ground_segmentation`, `ros2 topic list` showed all 3 expected
topics. One pure build-artifact gap found and fixed along the way (not
new source work): `heven_ros_ws/install/ad_description`'s
`local_setup.bash` was a broken symlink into a missing `build/ad_description`
directory (same class of issue as an earlier-documented stale-build-tree
fix); rebuilt `ad_description` directly from
`~/projects/heven-ad-2026/ad_description` into the existing
`heven_ros_ws` overlay — no source changed.

### C. Bounded 40-frame point-cloud validation

Ground fraction mean 0.530 (median 0.530, min 0.511, max 0.541), 0
NaN/Inf across all 40 frames on `cropped`/`ground`/`nonground`, single
publisher confirmed. `ground+nonground` undershoots `cropped` by ~5,373
pts/frame on average — explained directly by the shipped config's own
comment (`ransac_ground_filter.yaml`: the debug `ground` topic is
voxel-downsampled at 0.10m, not full-resolution) — not a defect. Full
detail: `ground_segmentation_summary.md`.

### D. Ground OFF vs ON — Euclidean detection (40-frame sample)

Objects/frame **byte-identical** OFF vs ON (mean 2.88, stdev 1.10,
per-frame counts and per-object position/dims identical across two
genuinely separate, distinctly-timestamped replays). Explained
architecturally: Euclidean's own `adaptive_euclidean_cluster.yaml` crop
(`min_z_m: -1.0, max_z_m: 3.0`) already excludes ground-plane points for
this LiDAR mount height independent of upstream ground segmentation, for
this scene/sample. Latency: OFF mean 16.95ms vs ON mean 20.48ms (ON adds
a pipeline hop; Euclidean actually receives *fewer* raw points under ON).

### E/F. Ground OFF vs ON — Autoware and AB3DMOT tracking (400-frame replay)

Zero clock rollbacks either run (1 reset segment each); publisher
isolation (exactly 1/topic) held throughout both runs, verified before
and after. One real integration gap found and fixed this session:
`filterpy`/`scipy`/compatible `numpy` were missing from the system Python
AB3DMOT's node uses (not present in earlier T-1B/T-1C sessions — a fresh
environment gap), and the static `odom->base_link->lidar_link` TF (the
same T-1C test-mode precedent) needed republishing fresh — both fixed
without touching repo/system state: `filterpy`+`scipy`+`numpy<2` staged
via non-root `pip install --target ~/py-ab3dmot-deps` (prebuilt wheels,
matching system Python's 3.10/x86_64 ABI) and exposed via `PYTHONPATH` at
launch; TF republished via a scratch `StaticTransformBroadcaster` script.
One real process-isolation bug caught and fixed mid-session: a stale
orphaned `ad_finite_point_filter_node` from an earlier aborted attempt
survived several `pkill` passes (wrong process-name pattern) and caused a
genuine 2x duplicate-publish on `/ad/perception/lidar/nonground_finite`
(and therefore Euclidean/Autoware) in the first Ground-ON tracking
attempt — caught via a stamp-uniqueness check (not assumed clean), fixed
by killing the exact stray PID and re-verifying single-publisher state
before re-running; AB3DMOT's own duplicate-timestamp-rejection design
silently absorbed the duplicates without producing bad data, but Autoware
did not, so the affected recording was discarded and redone cleanly
(final data below is from the clean rerun, 400/400/400, 0 duplicate
stamps on any of the 3 topics in either condition).

| metric | Autoware OFF | Autoware ON | AB3DMOT OFF | AB3DMOT ON |
|---|---|---|---|---|
| tracks/frame mean/median/max | 9.90/9.0/27 | 9.43/8.0/24 | 19.36/20.0/50 | 18.40/18.0/40 |
| segment-safe unique tracks | 488 | 456 | 2,895 | 2,721 |
| births / disappearances (mid-run) | 488/470 | 456/438 | 2,892/2,875 | 2,718/2,701 |
| lifetime mean/median (obs) | 8.12/4.0 | 8.27/4.0 | 2.67/2.0 | 2.70/2.0 |
| longest continuous track (obs) | 399 | 399 | 400 | 400 |

Real, modest (~5-7%) reduction in track population/churn under Ground ON
for both trackers over the full 400-frame replay — a genuine measured
effect distinct from D's byte-identical 40-frame Euclidean sample (longer
window, not the same sample). "Track churn" terminology only; no
MOTA/HOTA/IDF1/IDSW, no ground truth.

### G/H. Track churn / speed distribution changes

Covered in E/F table above and: AB3DMOT speed (reset-safe, n=7,743 OFF /
7,359 ON): median 0.00, p90 1.02/1.00, p95 3.61/3.47, p99 10.24/10.30,
max 18.91/22.27 m/s; fraction >1/>5/>10 m/s effectively unchanged
(0.100/0.033/0.011 vs 0.100/0.032/0.011) — Ground ON does not
meaningfully change AB3DMOT's speed distribution, only its track
population. Not a physical-accuracy claim (no GT).

### I. Latency / runtime

Autoware: mean 1.79ms (OFF) / 1.72ms (ON), materially unaffected.
AB3DMOT: mean 27.30ms (OFF) / 24.85ms (ON), tracking its own lower mean
track count under ON. AB3DMOT latency-vs-track-count Pearson r = 0.945
(OFF) / 0.946 (ON) — strong, consistent with `O(tracks × detections)`
GIoU-matrix cost already documented in T-1C, descriptive only.

### J. Clock rollback / reset segments

None occurred in either 400-frame recording (1 segment each) — stated
plainly; composite `(segment, uuid)` identity machinery was implemented
and exercised (segment assignment via true file/arrival order, never via
sort) but had nothing to merge-guard against this session.

### K. Best presentation figures

All 8 requested figures generated under
`~/heven_presentation_assets/ground_segmentation/`:
`ground_fraction_over_time.png`, `ground_detection_objects_per_frame.png`,
`ground_tracks_per_frame.png`, `ground_unique_tracks.png`,
`ground_track_lifetime_distribution.png`, `ground_births_deaths.png`,
`ground_ab3dmot_speed_distribution.png`,
`ground_ab3dmot_latency_vs_track_count.png`. RViz live visual (pixel-level)
confirmation was **not** repeated this session (same no-screenshot-capability
limitation already documented in the RViz Tracking Comparison session
above) — topic-level verification was used instead, consistent with that
session's own established precedent; not fabricated as a "verified visually"
claim anywhere in this report.

### L. Limitations

Single static MORAI scene (already-documented, unrelated to this task);
Euclidean OFF/ON comparison (40 frames, ~6s) and tracking OFF/ON
comparison (400 frames, ~60s) are not the same sample size — stated
explicitly rather than implied equal; no GT/MOTA/HOTA/IDF1/IDSW claim
anywhere; `filterpy`/`scipy`/`numpy` staging (see E/F) and the
`ad_description` build-artifact fix (see B) are both environment-repair,
not repo changes.

### M. T-2 result: **PASS**

Full experiment completed end-to-end in one continuous pass per this
task's explicit instruction not to stop at another planning checkpoint
after build success. Full interpretation (DIRECT MEASUREMENTS vs
ARCHITECTURAL INTERPRETATION) in
`~/heven_presentation_assets/ground_segmentation/README.md`. Not
committed/pushed, per this task's instruction; `git status --short`
confirms only the same pre-existing unrelated dirty files from every
prior session in this branch (list below), plus this `STATUS.md` update.

---

**[historical, T-2A build-blocker detail below retained for reference]**

**T-2A build progress**: the system libraries identified as the original
blocker (`libopencv-dev`, `libpcl-dev`, `ros-humble-pcl-ros`,
`ros-humble-pcl-conversions`, `libboost1.74-dev`) are now installed
(confirmed via `dpkg -s`, someone ran the recommended command with root
between sessions). Rebuilt in `~/projects/autoware_tracker_ws` (same
workspace/convention as `autoware_multi_object_tracker`):
`autoware_vehicle_info_utils` ✅, `autoware_utils` (monolithic package,
`src/autoware_utils/autoware_utils` — the real dependency
`autoware_ground_segmentation`'s package.xml needs; the earlier attempt
had only the split `autoware_utils_*` packages) ✅ built this session.

**Sophus/Ceres resolved**: `libceres-dev` and `ros-humble-sophus` were
installed with root between sessions; verified directly this session
(`dpkg -s` both "install ok installed", `SophusConfig.cmake` found under
`/opt/ros/humble/share/sophus/cmake/`) — not assumed.

**CGAL resolved**: `libcgal-dev` installed with root between sessions;
verified directly (`dpkg -s` "install ok installed",
`CGALConfig.cmake` found under `/usr/lib/x86_64-linux-gnu/cmake/CGAL/`).

**Targeted dependency check (this session, replaces one-by-one
discovery)**: `rosdep check`/`install --simulate` could not run (rosdep
itself needs a one-time root-only `sudo rosdep init`, unavailable).
Substituted a manual equivalent: read every `<depend>` in
`autoware_pointcloud_preprocessor/package.xml` and
`autoware_ground_segmentation/package.xml`, found one further required
local sibling (`autoware_pcl_extensions`, source present, not yet built),
read its package.xml too, and checked `dpkg -s` on every resulting
package name in one pass. Result: **8 remaining packages missing**, all
prebuilt/apt-installable, none requiring a further source build:
`ros-humble-autoware-lanelet2-extension`,
`ros-humble-autoware-lanelet2-utils`, `ros-humble-autoware-point-types`,
`ros-humble-autoware-sensing-msgs`, `ros-humble-autoware-vehicle-msgs`,
`ros-humble-cv-bridge`, `ros-humble-point-cloud-msg-wrapper`,
`librange-v3-dev`. Full table (incl. what's already satisfied) in
`ground_segmentation_blocker.md`. No root access this session; per this
task's explicit instruction, did **not** install these by hand or attempt
another partial build now that they're known to be missing.

**Phases 5 onward (verify node launches) through the full Ground OFF/ON
experiment remain BLOCKED** — not fabricated.

**Next recommended task**:
```
sudo apt install -y ros-humble-autoware-lanelet2-extension \
  ros-humble-autoware-lanelet2-utils ros-humble-autoware-point-types \
  ros-humble-autoware-sensing-msgs ros-humble-autoware-vehicle-msgs \
  ros-humble-cv-bridge ros-humble-point-cloud-msg-wrapper librange-v3-dev
```
then re-run the exact `colcon build` command in
`ground_segmentation_blocker.md` for `autoware_pcl_extensions` (new,
local, add to `--base-paths`) + `autoware_pointcloud_preprocessor` +
`autoware_ground_segmentation` + `managed_transform_buffer`. This list was
derived from an exhaustive package.xml read, not incremental
trial-and-error, so no further new dependency is expected — but if one
does appear, verify it the same way before installing. Once
`autoware_ground_segmentation` builds, verify
`ros2 pkg executables autoware_ground_segmentation` shows
`ransac_ground_filter_node`, then resume T-2 at Phase 6 (bounded
point-cloud validation) of the original spec.

## Autoware vs AB3DMOT: initial runtime comparison

Branch: `feat/ab3dmot-tracker`. Small, controlled runtime comparison —
**not** a TrackEval integration, no new tracking algorithm, no tuning of
either tracker, no RViz changes. Full detail, exact commands, result
table, and limitations in
`docs/research/autoware_vs_ab3dmot_initial_comparison.md`.

Both trackers ran simultaneously off the exact same
`/ad/perception/objects/detected` stream (same Euclidean detector, same
`ad_publish_morai_frames` replay of `~/datasets/morai_heven`/`train`
`static_20260805_003151`, 400 frames / ~65 s, same T-1C static-TF setup),
each unmodified and untuned (Autoware `tracking.launch.py` +
`autoware.yaml`; AB3DMOT `ab3dmot_tracker.launch.py enabled:=true` +
`ab3dmot.yaml`, both unchanged from T-1C).

Collected via a scratch (uncommitted) `rclpy` recorder writing JSONL for
`/ad/perception/objects/detected`, `/ad/perception/objects/tracked`, and
`/experiment/tracked/ab3dmot`: message counts, output frequency, objects/
frame, unique track-ID counts, track-lifetime distributions, longest
tracks, births/deletions, duplicate-ID-per-frame checks, and NaN/Inf/
invalid-quaternion/invalid-dimension checks — all zero for both trackers
(no data-integrity defect in either). Autoware: 398 output msgs, mean 9.92
objs/frame, 474 unique track IDs, longest track 64.8 s (near full-run).
AB3DMOT: 400 output msgs, mean 19.28 objs/frame, 1,720 unique track IDs,
longest track 59.5 s.

Latency measured for **both** via the same external receive-to-receive
probe (single recorder process, unmodified nodes) — a fair, symmetric
method, so both are reported (unlike T-1C, which measured AB3DMOT only):
Autoware mean 1.81 ms / p95 3.44 ms; AB3DMOT mean 26.9 ms / p95 60.1 ms
(excluding one single-sample recorder-side scheduling artifact, stated
explicitly in the doc, not hidden). Explicitly **not** claimed as a pure
algorithm-cost comparison — Autoware's tracker is compiled C++, AB3DMOT's
is Python/rclpy; the probe measures each complete node's receive→publish
time.

Visual/qualitative cases (identified from the same recorded message data,
not a GUI screenshot — same no-screenshot-capability limitation already
documented in the RViz Tracking Comparison session) are recorded in the
doc: a long persistent track for each tracker, concrete birth/deletion
frame ranges, a multi-simultaneous-track frame for each (Autoware 30
objects at t=52.9 s; AB3DMOT 50 objects at t=56.9 s), and one visible
disagreement — Autoware ramps up slowly at replay start (0→1→2→3 objects
over the first ~24 s) while AB3DMOT does not (3→9→7→7 over the same
window), a direct, expected consequence of the two trackers' already-
decided `min_hits` config difference, not an unexplained bug. No accuracy,
GT-based, or ID-switch-rate claim is made for either tracker — explicitly
stated in the doc per this task's instruction.

Not committed/pushed per this task's instruction.

## RViz Tracking Comparison

Branch: `feat/ab3dmot-tracker`. Visualization-only milestone extending the
existing `ad_viz` perception visualizer to show the Autoware tracker and
the experimental AB3DMOT tracker at the same time, off the same detection
stream, reusing the exact T-1C MORAI replay/TF setup. No tracking
algorithm, association, KF, Autoware tracker, detector, IMM/prediction, or
occupancy code was touched — only `ad_viz` (marker builder, node, CMake,
tests) and `ad_lidar_perception`'s visualization launch/RViz-config/tests.

### Approach

Reused the existing `perception_visualizer_node` executable by launching
it **twice** with different parameters, rather than creating a second
visualization package or node:
- Instance 1 (unchanged topics): `id_prefix:="A-"`, subscribes
  `/ad/perception/objects/tracked` (Autoware), publishes
  `/ad/visualization/tracked_objects`.
- Instance 2 (new): `id_prefix:="B-"`, subscribes
  `/experiment/tracked/ab3dmot`, publishes
  `/experiment/visualization/tracked_objects_ab3dmot`,
  `visualize_detections:=false` and `visualize_predictions:=false` (avoids
  duplicate detection markers and an unused prediction subscription).

`ad_lidar_perception/launch/perception_visualization.launch.py` now starts
both instances; `ad_lidar_perception/rviz/heven_perception.rviz` gained a
new "Tracked Objects (AB3DMOT)" `MarkerArray` display on the new topic,
and the old "Tracked Objects" display was renamed "Tracked Objects
(Autoware)" for clarity.

### Track visualization features (all in `ad_viz`)

- **ID prefix / disambiguation**: `ObjectMarkerConfig::id_prefix` is
  prepended to each track's on-screen label, so the two trackers never
  rely on color alone (`A-<id>` / `B-<id>`), per the task requirement.
- **Fixed a pre-existing label bug found while touching this code**: the
  ID suffix shown in the label was `uuid.substr(0, 8)` (first 8 hex
  chars). AB3DMOT's UUID encoding (`ab3dmot_ros.py::track_id_to_uuid`,
  from T-1B) packs its integer track id into the **low 8 bytes**
  (`uuid[8:16]`), so every AB3DMOT track showed an identical, useless
  `00000000` suffix. Fixed to take the **last 16 hex chars** (last 8
  bytes) — verified live: AB3DMOT labels now read e.g. `B-0000000000000001`,
  `B-00000000000000ce` (correct, distinct, matches the tracker's own small
  integer ids); Autoware labels remain distinct (its own UUIDs are
  effectively random in all 16 bytes), e.g. `A-7b4fb59b736c7e9f...`. Also
  exported `uuid_hex()` from `object_marker_builder` so the new trajectory
  code can key its per-track state identically without re-deriving the
  encoding.
- **Trajectory history**: new `ad_viz::perception::TrajectoryHistory`
  class (`trajectory_history.hpp`/`.cpp`) — a per-track key bounded
  `std::deque<Point>` (oldest dropped once `trajectory_max_points`, default
  30, is exceeded) plus time-based pruning (`prune_stale`, default 3.0 s
  timeout) so a track's history is fully removed once its tracker stops
  publishing it. Operates on plain `int64_t` nanosecond stamps (not
  `rclcpp::Time`), matching this repo's established convention for
  avoiding ROS clock-type pitfalls (`imm_predictor.hpp` etc.). Only ever
  stores already-observed positions — never predicts/extrapolates.
- **`build_trajectory_markers`**: renders one `LINE_STRIP` per track with
  >=2 points (namespace `trajectory/<id_prefix><uuid_hex>`); emits no
  `DELETEALL` of its own — `PerceptionVisualizerNode::on_tracks` appends
  its output into the *same* `MarkerArray`/topic as
  `build_tracked_markers` (which already emits one `DELETEALL` per
  publish), so stale trajectory markers are cleared in the same frame as
  stale boxes/labels — no orphaned markers, confirmed live (see below).
- New node params: `id_prefix` (string, default `""`), `visualize_detections`
  (bool, default `true`, gates the detection subscription/publisher so the
  second AB3DMOT-only instance doesn't republish detections twice),
  `trajectory_max_points` (int, default 30), `trajectory_stale_timeout_sec`
  (double, default 3.0).

### Tests

`colcon build --symlink-install --packages-up-to ad_viz` and
`--packages-select ad_lidar_perception`, both clean, after fixing one
build-environment issue unrelated to this task's code (the shared
`heven_ros_ws` build tree had a stale cached Python interpreter path from
an earlier venv-activated session, breaking `rosidl`/`ament` for several
packages incl. `ad_interfaces`; fixed by clearing those 4 packages' build
directories — pure build artifacts, not source — so CMake re-detected the
correct interpreter).

New/updated gtests, all passing (`ctest` in `ad_viz`, full suite: 7/7,
including the 2 new/changed ones plus the pre-existing 5): `ad_viz`
`test_trajectory_history` (new — 4 cases: rejects invalid construction,
accumulates points per track, bounds points per track with oldest-dropped,
prunes stale tracks by elapsed time) and `test_object_marker_builder`
(updated: fixed the now-stale first-8-hex-char UUID assertion to the
corrected last-16-hex-char format; added 2 new cases covering
`id_prefix` label prepending and `build_trajectory_markers`'s
short-history-skip / prefixed-namespace behavior).

`ad_lidar_perception`'s full `ctest -R "launch|rviz|visualiz"` (8/8,
including `test_perception_visualization_launch`, updated for the new
second `Node` instance, the renamed/added RViz `MarkerArray` displays, and
the `id_prefix`/`tracked_input_topic` launch-arg wiring) — all pass.

### Live RViz smoke test — actually run, not simulated

Reused the exact T-1C setup: same static `odom->base_link` (identity) +
`base_link->lidar_link` (`z=1.70`, identity) TF precedent (this time
correctly published via `tf2_ros.StaticTransformBroadcaster` on
`/tf_static`, not a periodically-republished dynamic `/tf` — an earlier
attempt using a repeating dynamic broadcaster hit a real "extrapolation
into the future" TF race against live detection timestamps; switching to
static, and restarting all TF-listening nodes fresh afterward so no node
retained a poisoned mixed static/dynamic TF cache, fixed it cleanly), same
`ad_publish_morai_frames` replay of `~/datasets/morai_heven` (`train`
split), same Euclidean detector feeding both trackers off one
`/ad/perception/objects/detected` stream. Launched: `ab3dmot_tracker.launch.py
enabled:=true`, `tracking.launch.py` (Autoware), `euclidean_clustering.launch.py`,
and `perception_visualization.launch.py start_rviz:=true` (both visualizer
instances + a real `rviz2` process, confirmed alive with a real OpenGL
context: `Stereo is NOT SUPPORTED` / `OpenGl version: 4.2` in its log).

**No pixel-level screenshot was possible in this environment** (no `sudo`
for `imagemagick`/`xwd`; `PIL.ImageGrab` against the available WSLg X
display failed with an X `BadMatch` error). Per this repo's own established
precedent for exactly this situation (`docs/research/centerpoint_status.md`:
"verified by direct topic echo, not a GUI screenshot"), verification was
instead done directly against the live `MarkerArray` content on the two
topics RViz displays, plus confirming RViz's own subscriptions:

1. **Both IDs visible with correct prefixes** — confirmed directly:
   Autoware labels `"UNKNOWN 1.00 A-7b4fb59b736c7e9f"` etc. (distinct
   16-hex-char suffixes per track); AB3DMOT labels
   `"UNKNOWN 1.00 B-0000000000000001"`, `"...B-00000000000000ce"` etc.
   (small integer ids, now fully shown thanks to the 16-char-suffix fix
   above).
2. **Boxes plausible / aligned** — sampled absolute box pose positions
   from a live AB3DMOT message: x in roughly [-3, 43] m, y in [-12, 19] m,
   z in [0.9, 2.9] m (odom frame) — consistent with the cropped point
   cloud's own configured range and the `z=1.70` m LiDAR mount height;
   not a pixel check, but geometrically plausible, not garbage/NaN.
3. **Velocity arrows render** — `Marker::ARROW` (`type: 0`) present per
   moving track in both trackers' live output.
4. **Trajectories update over time and are bounded** — sampled the same
   AB3DMOT/Autoware topics repeatedly over ~15 s: per-track `LINE_STRIP`
   point counts grew (2 -> 7 -> 10 -> ...) and were observed reaching
   exactly the configured cap of **30** on both trackers' longest-lived
   tracks — the bound holds in the live pipeline, not just in the unit
   test.
5. **Stale histories disappear** — resampled ~14 s apart: AB3DMOT
   trajectory-track count dropped from 80 (many short-lived, quickly
   replaced tracks — see below) to 20, and specific old track-id
   namespaces confirmed **absent** from the later message — stale
   pruning is working live, not just in `test_trajectory_history`.
6. **No marker-array accumulation** — sampled the AB3DMOT visualization
   topic's total marker count twice, 10 s apart, mid-replay: 128 both
   times (stable) — confirms the per-frame `DELETEALL` + bounded history
   design does not leak markers over a sustained run.
7. **RViz actually subscribed** — `ros2 topic info --verbose` on both
   `/ad/visualization/tracked_objects` and
   `/experiment/visualization/tracked_objects_ab3dmot` shows exactly one
   subscriber each: node `heven_perception_rviz` — the real RViz process
   launched above, confirming the `.rviz` config's two `MarkerArray`
   displays are wired to the right topics.
8. **Both trackers' output topics publish continuously off the same
   detection stream** during replay: `/ad/perception/objects/detected`
   ~8.6 Hz, `/experiment/tracked/ab3dmot` ~8.5 Hz, `/ad/perception/objects/tracked`
   ~8.2-8.4 Hz (`ros2 topic hz`, mid-replay).

### Qualitative observations (real, not manufactured)

- **Track-ID churn / fragmentation is real and frequent** in this
  baseline AB3DMOT config (`min_hits=1`, `giou_gate=0.0`, greedy
  matching): one sampled instant showed ~80 distinct AB3DMOT track-history
  namespaces alive at once, the large majority with only 2 trajectory
  points (i.e. created and about to be replaced almost immediately) and a
  handful with 7-10+ points (persistent, stable tracks). This is
  consistent with §3's already-documented, not-yet-calibrated
  `giou_gate` default (flagged in the AB3DMOT Integration Decisions as
  needing real-data calibration) — this session did not tune it, only
  observed and recorded the resulting visual behavior.
- **Persistent tracking**: multiple track ids (e.g. AB3DMOT's low integer
  ids `0x1..0x20` range, well as several Autoware UUIDs) remained present
  and updating across many consecutive messages during the replay window,
  with trajectories growing smoothly up to the 30-point cap — normal,
  stable tracking is visibly present alongside the high-churn tracks
  above.
- **Birth/deletion**: directly observed via the stale-pruning check above
  (§5) — dozens of AB3DMOT tracks born and pruned within a ~14 s window.
- **Crossing objects / ID switches**: not specifically isolated in this
  session (would require per-object trajectory-shape inspection across a
  longer window than the spot-checks performed here); not claimed either
  way beyond the general churn pattern already documented above. No
  accuracy claim is made from this RViz-only verification, per this
  task's own instruction.

### Files changed by this task (RViz Tracking Comparison; not yet
committed/pushed — see below)

```
ad_viz/CMakeLists.txt
ad_viz/include/ad_viz/perception/object_marker_builder.hpp
ad_viz/include/ad_viz/perception/perception_visualizer_node.hpp
ad_viz/include/ad_viz/perception/trajectory_history.hpp   (new)
ad_viz/src/perception/object_marker_builder.cpp
ad_viz/src/perception/perception_visualizer_node.cpp
ad_viz/src/perception/trajectory_history.cpp               (new)
ad_viz/test/test_object_marker_builder.cpp
ad_viz/test/test_trajectory_history.cpp                     (new)
ad_lidar_perception/launch/perception_visualization.launch.py
ad_lidar_perception/rviz/heven_perception.rviz
ad_lidar_perception/test/test_perception_visualization_launch.py
docs/agent/STATUS.md (this update)
```

**A. Pre-existing unrelated dirty files** (unchanged, not staged/touched
by this task — same 19 files as every prior task this session):

```
.claude/skills/bootstrap-repo/scripts/bootstrap-repo.sh
.claude/skills/bootstrap-repo/tests/test_bootstrap_repo.sh
.claude/skills/issue-worker/scripts/claim_issue.sh
.claude/skills/issue-worker/tests/test_claim_issue.sh
ops/runner/bin/agent-tick
ops/runner/bin/codex-pr-review.sh
ops/runner/bin/pr-set-in-review.sh
ops/runner/bin/render_prompt.sh
ops/runner/repo-context.sh
ops/runner/tests/test_env_precedence.sh
ops/runner/tests/test_pr_set_in_review.sh
scripts/apply_dependency_patches.sh
scripts/bootstrap_workspace.sh
scripts/check_autoware_perception.py
scripts/setup_dev_env.sh
scripts/test_python.sh
scripts/tests/test_verify_template_contract.sh
scripts/verify-template-contract.sh
scripts/verify_ad_data.py
```

### Blockers

None. Live comparison works end-to-end. Not committed/pushed per this
task's explicit instruction — left in the working tree for a future,
separate commit task.

## Previous: T-1C — Runtime verification of the AB3DMOT ROS2 tracker

Branch: `feat/ab3dmot-tracker`. **T-1C: PASS.** Actually launched the real
ROS2 graph (not just unit tests) and replayed real MORAI-exported LiDAR
frames through it. Used the currently-committed AB3DMOT configuration
unchanged (`ad_lidar_perception/config/tracking/ab3dmot.yaml`: `giou_3d`,
greedy, `min_hits=1`, `max_age=2`, `giou_gate=0.0`) — no parameters were
tuned. No RViz work started, no Autoware source modified.

### Replay/data source

`~/datasets/morai_heven` (`train` split, 1,764 exported frames of the
single `static_20260805_003151` scene, per prior sessions' audit — same
dataset the Euclidean/CenterPoint comparison tooling already uses). The
existing `ad_publish_morai_frames` executable (built in a prior T‑1B/CenterPoint
session) republished sequential samples onto `/ad/perception/lidar/cropped`
at ~6 Hz. No new replay system was created.

### TF prerequisite — reused an existing, explicit precedent

Live MORAI localization was not available in this environment (no running
MORAI bridge/simulator). Rather than inventing a workaround, this session
found that `ad_lidar_perception/test/test_autoware_pipeline_integration.py`
**already defines** exactly this "replay/test mode": it broadcasts a
synthetic `odom→base_link` (identity) + `base_link→lidar_link`
(translation `z=1.70`, identity rotation) TF chain for testing the Autoware
tracker pipeline without live localization. T‑1C reused that exact, already-
established transform (via a `StaticTransformBroadcaster`, run from the
session scratch directory only — not added to the repository) rather than
inventing a new one. `ros2 run tf2_ros tf2_echo odom lidar_link` confirmed
the chain resolved correctly before any detection traffic was sent.

### Exact commands (environment sourcing omitted for brevity — standard
ROS Humble + `autoware_perception_msgs` local overlay + `autoware_tracker_ws`
overlay + `heven_ros_ws` install + `heven-centerpoint` venv, same as prior
sessions)

```
ros2 launch ad_lidar_perception ab3dmot_tracker.launch.py enabled:=true
ros2 launch ad_lidar_perception tracking.launch.py
ros2 launch ad_lidar_perception euclidean_clustering.launch.py \
  finite_filter_enabled:=false finite_input_topic:=/ad/perception/lidar/cropped
ros2 run ad_lidar_perception ad_publish_morai_frames \
  --dataset ~/datasets/morai_heven --split train --count 80 \
  --topic /ad/perception/lidar/cropped --interval-sec 0.15
```

Both trackers subscribed the same `/ad/perception/objects/detected` topic
published by the Euclidean cluster node (fed, in turn, from the same
replayed MORAI frames) — satisfying "the two trackers must consume the
same detection stream." Output was captured with `ros2 topic echo
--full-length` into scratch YAML files and parsed with a small local
PyYAML script (not committed).

### 1. Detection input / 2. AB3DMOT output — both continuously published

- `/ad/perception/objects/detected`: ~5.98 Hz (`ros2 topic hz`, 77-sample
  window), matching the publisher's 0.15 s interval exactly. Source frame:
  `lidar_link`.
- `/experiment/tracked/ab3dmot`: ~5.98 Hz, **80/80** detection messages
  produced exactly 80 tracked-output messages (1:1, no drops, no gaps) —
  confirmed continuous, not just "sometimes publishes."
- Message publication alone was **not** treated as sufficient — see §4-8
  below for actual tracking-quality evidence.

### 3. Output frame — 100% odom

All 80 sampled `/experiment/tracked/ab3dmot` messages: `frame_id: odom`.
Zero frame changes, zero TF failures during the replay window (the TF
chain was up before any detections were sent).

### 4. Timestamp semantics — valid, monotonic, detection-derived

- Every output stamp equals its triggering detection's own header stamp
  (verified directly in the parsed data — this is also enforced by
  `ab3dmot_tracker_node.py`'s design, not just observed).
- Stamps strictly monotonic across all 80 frames (`all(stamps[i] <
  stamps[i+1])` = `True`); span ≈13.2 s over 80 frames at ~6 Hz.
- No wall-clock substitution: the publisher stamps each replayed frame
  with its own `now()` at publish time, and the tracked-object stamps
  match those exactly, not a later processing-time clock read.
- No duplicate stamps occurred in this run (replay interval 0.15 s ≫ ROS
  clock resolution), so the documented skip-duplicate path was not
  exercised live here — it remains covered by
  `test_ab3dmot_ros.py::ClassifyTimestampTest` and
  `test_ab3dmot_tracker_node.py::test_duplicate_timestamp_is_skipped_not_republished`.
- **Clock rollback did not occur naturally** in this replay (the publisher
  uses a monotonically increasing wall clock, and MORAI's own simulator
  was not live in this environment to produce a real sim-time reset).
  Per this task's own instruction, this is stated plainly rather than
  forced: rollback-reset behavior remains verified by
  `test_ab3dmot_tracker_node.py::test_clock_rollback_resets_tracker_state`
  (which explicitly asserts a fresh `AB3DMOTTracker` instance is
  constructed and no negative dt reaches `step()`), not re-demonstrated
  against live MORAI clock behavior in this session.

### 5. Persistent track IDs — multiple concrete, verified examples

At least 3 tracks persisted across many consecutive frames (IDs read
directly from each message's `object_id.uuid`, decoded via the same
`track_id_to_uuid` scheme the node uses — not inferred from position
similarity):

| track ID | first ts (ns, relative) | last ts (ns, relative) | consecutive obs | start pos (x,y,z) | end pos (x,y,z) |
|---|---|---|---|---|---|
| 2 | 0 | 13,202,307,199 | **80/80** (entire replay) | (-1.11, -0.00, 1.43) | (-1.11, 0.00, 1.43) |
| 3 | 0 | 4,368,447,013 | 27 | (-2.31, 5.33, 2.03) | (-3.43, 5.32, 2.17) |
| 1 | 0 | 4,200,644,691 | 26 | (-2.65, -4.34, 1.99) | (-3.39, -4.33, 2.01) |
| 4 | 1,200,053,125 (born frame 7, mid-run) | 3,703,748,803 | 16 | (84.06, 11.20, 3.18) | (83.98, 11.19, 3.18) |

Track 2 in particular is a clean, unambiguous example: the *same* AB3DMOT
ID (decoded from the message's real UUID field, not guessed) tracked one
near-stationary object across the full 80-frame, 13.2 s replay with a
sub-centimeter position range — this is real persistent tracking, not
just repeated publication.

### 6. Velocity — finite, m/s-scaled, plausible for this replay

- All 586 sampled velocity vectors across all tracks: finite (0 NaN/Inf).
- 94.4% of samples were near-zero (< 0.05 m/s) — consistent with this
  MORAI capture being a **static scene** (confirmed in prior sessions):
  most Euclidean clusters here are static-object/ground-remnant clusters,
  so near-zero velocity is the *correct*, plausible result, not a bug.
- Speed distribution: p50 ≈ 0, p90 ≈ 0.02 m/s, p95 ≈ 0.12 m/s, p99 ≈ 9.0
  m/s, max ≈ 17.75 m/s.
- **No meters-per-frame scaling bug**: cross-checked track 16 (7
  consecutive obs, frames 42-48, median speed 7.22 m/s) directly against
  its own raw position delta — it moved 7.28 m over 1.003 s of real
  elapsed time (its own header-stamp span), i.e. ≈7.26 m/s independently
  computed from position alone, matching the *published* velocity to
  within numerical noise. Same cross-check on track 157 (5.2 m real
  displacement / 0.999 s ⇒ ≈5.2 m/s, matching its published 5.6-6.2 m/s
  range). This directly demonstrates the T-1B real-dt fix is working
  correctly in the live ROS path, not just in T-1A's unit tests.
- **Flagged, explained, not silently ignored**: the small population of
  higher-speed short-lived tracks (tracks 16, 157, and similar, all
  ≤7 consecutive observations) are very likely Euclidean-detector-level
  clustering jitter, not AB3DMOT defects — this replay path deliberately
  bypasses ground segmentation (documented caveat, inherited from
  `docs/perception/centerpoint_vs_euclidean_comparison.md`), so a handful
  of unstable ground-adjacent clusters are expected. This is a data/
  detector-input characteristic of this specific verification setup, not
  a claim about AB3DMOT's or Euclidean's real-world quality.

### 7. Track lifecycle — creation, update, and deletion all observed live; coast not naturally exercised

- **Creation**: 214 of 217 distinct track IDs were born after frame 0
  (i.e., mid-replay, not just at startup) — continuous birth activity
  throughout the run, e.g. track 4 born at frame 7.
- **Update**: every persistent-track example above is itself repeated
  `update()` evidence (16-80 consecutive matched updates per track).
- **Deletion**: 7 tracks with ≥3 observations ended cleanly before the
  final frame and never reappeared under the same ID within the window
  (e.g., track 4: last seen frame 22 of 79, track 16: last seen frame 48
  of 79) — consistent with the **unmodified, committed** `max_age=2`
  policy. `min_hits`/`max_age`/`giou_gate` were not changed to produce
  this.
- **Coast (temporary miss then reappear under the same ID)**: **did not
  occur naturally** in this 80-frame window (0 tracks showed a 1-frame
  gap followed by re-observation). Stated plainly rather than forced —
  this exact behavior remains covered by
  `test_ab3dmot_core.py::LifecycleTest::test_temporary_missed_detection_keeps_same_track`
  and `test_ab3dmot_tracker_node.py`'s equivalent, not re-demonstrated
  live here.

### 8. Output validity — zero invalid values found

Across all 80 sampled frames / all objects (max 26 objects in a single
frame): **0** NaN/Inf values, **0** non-unit-norm quaternions (tolerance
1e-3), **0** non-positive box dimensions, **0** frames with a duplicate
track ID, **0** unexpected `frame_id` values, **0** invalid timestamps.

### 9. Runtime latency

Measured directly (input-topic receipt → matching-stamp output-topic
receipt, both timestamped on the same subscriber process's clock — a
dedicated scratch probe script, not committed) over **100 samples** from a
separate 100-frame replay pass (0.10 s interval, ~6 Hz) against the same
running node:

| stat | value |
|---|---|
| sample count | 100 |
| mean | 7.31 ms |
| median | 4.52 ms |
| p95 | 21.96 ms |
| max | 27.46 ms |

Latency trended upward over the run (from ~2-4 ms early to ~15-27 ms
later) — this tracks the growing live track count (up to several hundred
tracks accumulate over a long run against this noisy, ground-inclusive
Euclidean stream; GIoU-matrix cost is `O(tracks × detections)`), not
random jitter. Input/output frequency: ≈6 Hz in, ≈6 Hz out, matched 1:1.
This is execution/runtime evidence only — **no comparison to Autoware's
own latency is made or implied**, per AGENTS.md "execution success is not
performance validation" and this task's explicit instruction not to claim
performance superiority.

### 10. Parallel Autoware path — confirmed unaffected

- Same replay, same detection stream, `tracking.launch.py`'s
  `multi_object_tracker` running simultaneously: 80/80 detection messages
  → 80 `/ad/perception/objects/tracked` messages (79/80 non-empty), all
  `frame_id: odom`, zero errors in its log.
- Direct simultaneous-rate check (separate short replay, both `ros2 topic
  hz` running at once): Autoware ≈6.045 Hz, AB3DMOT ≈6.047 Hz — both
  tracking the same ~6 Hz input rate concurrently with no observable
  interference.
- `ad_lidar_perception/config/tracking/autoware.yaml`,
  `tracking.launch.py`, and the Autoware `multi_object_tracker` source
  were not modified or touched by this session. AB3DMOT publishes only to
  its own `/experiment/tracked/ab3dmot` topic; `/ad/perception/objects/tracked`
  was never written to by AB3DMOT.
- No accuracy or quality comparison between the two trackers is made.

### Runtime bugs found

**None.** No genuine T-1B/T-1C integration bug was encountered during this
verification pass — the node behaved exactly as designed on the first
real replay attempt (no code changes were made during T-1C).

### Limitations

- Single static MORAI scene (already-documented dataset-diversity
  limitation, unrelated to T-1C) — velocities are mostly near-zero because
  most tracked clusters are genuinely static in this data, and the higher-
  speed tracks are explained by detector-level (Euclidean, ground-seg-
  bypassed) clustering jitter, not evaluated against any ground truth.
- Clock-rollback and single-frame-coast behaviors were not naturally
  exercised in this specific replay window; both remain verified at the
  unit-test level only (already covered in T-1A/T-1B).
- Latency was measured via a receive-to-receive proxy on a third
  subscriber process (clean, standard technique) rather than in-node
  instrumentation, since the node does not currently self-log per-callback
  timing (CenterPoint's node does; AB3DMOT's does not yet) — flagged as a
  possible small future improvement, not a defect.

## T-1C: PASS

## T-1 overall: PASS

All required conditions are met: T-1A passed (committed `cceef3d`); T-1B
passed (ROS2 integration, still uncommitted pending a separate commit
task); this session observed **real** runtime `/experiment/tracked/ab3dmot`
messages (not just unit tests); **multiple track IDs demonstrably persist**
across many consecutive frames with concrete evidence (track 2: 80/80
frames); output `frame_id` is `odom` on every sampled message; timestamp
semantics are valid and monotonic with no wall-clock substitution;
velocity semantics are valid (finite, m/s, cross-checked against real
position deltas, no meters-per-frame artifact); zero NaN/Inf/invalid-
quaternion/invalid-dimension/duplicate-ID/TF/runtime failures occurred;
and the production Autoware path was verified to keep publishing,
unaltered, throughout.

## Previous: T-1B — HEVEN ROS2 integration for the AB3DMOT tracking core

Branch: `feat/ab3dmot-tracker`. **T-1B: PASS.** Implemented the opt-in ROS2

Branch: `feat/ab3dmot-tracker`. **T-1B: PASS.** Implemented the opt-in ROS2
node wrapping T-1A's `AB3DMOTTracker`, per the resolved design in
`docs/research/tracking_architecture.md` ("AB3DMOT Integration
Decisions") and this task's explicit interface/frame/timestamp/covariance
requirements. No algorithm redesign, no RViz change, no runtime
performance comparison started.

### ROS interface (as specified, unchanged from the request)

```
input:  /ad/perception/objects/detected   (autoware_perception_msgs/msg/DetectedObjects)
output: /experiment/tracked/ab3dmot        (autoware_perception_msgs/msg/TrackedObjects, frame_id=odom)
```

Production path untouched and verified unaffected: `tracking.launch.py`,
`ad_lidar_perception/config/tracking/autoware.yaml`, and the Autoware
`multi_object_tracker` source were not read or modified this session
beyond what was already known from the prior audit. The new node
(`ad_ab3dmot_tracker`) is disabled by default (`enabled:=false` node
parameter default; the new launch file flips it to `true` at the launch
level only, mirroring `centerpoint_detector.launch.py`'s own pattern) and
runs on a separate topic, so both trackers can run simultaneously off the
same detections.

### Files changed

- `ad_lidar_perception/ad_lidar_perception/ab3dmot_ros.py` **(new)** —
  pure message<->core adapter functions, message types always injected
  (same pattern as `centerpoint_ros.py`): `stamp_to_ns`,
  `select_classification`, `normalized_quaternion`/`yaw_from_quaternion`/
  `quaternion_from_yaw` (ports of `autoware_prediction_node.cpp`'s
  validated helpers), `quaternion_multiply`/`quaternion_rotate_vector`/
  `transform_pose_z_up` (full quaternion composition, not a naive yaw-add,
  so a transform with any roll/pitch is still handled correctly),
  `classify_timestamp`/`TimestampDecision` (pure first/duplicate/
  increasing/rollback classifier), `detected_objects_to_detections`
  (whole-message rejection on any malformed object, matching
  `AutowarePredictionNode`'s existing convention), `track_id_to_uuid`
  (deterministic mechanical int->16-byte encoding), `tracked_state_to_message`/
  `tracked_states_to_message` (the ROS-side half of the already-resolved
  field-mapping table).
- `ad_lidar_perception/ad_lidar_perception/ab3dmot_tracker_node.py`
  **(new)** — the `rclpy.Node`: declares `enabled`/`input_topic`/
  `output_topic`/`target_frame`/`ab3dmot_root` plus the T-1A config
  parameters; owns a `tf2_ros.Buffer`/`TransformListener`; on each
  `DetectedObjects` callback: validates/classifies the header stamp
  (`classify_timestamp`) -> looks up `lidar_link -> odom` at the
  message's own stamp (skipped entirely for an empty message, so an empty
  "heartbeat" frame never depends on TF availability) -> converts to
  `Detection`s -> drives `AB3DMOTTracker.step()` with the message's own
  timestamp (never wall-clock) -> publishes `TrackedObjects`. All
  malformed-input/TF-failure/duplicate-timestamp paths log a warning and
  skip that callback without crashing (never publish stale or
  wrong-frame data); a clock rollback logs and **reconstructs a fresh
  `AB3DMOTTracker` instance** (cleanest way to reset all experimental
  track state without touching T-1A's core reset-free design) rather than
  ever calling `step()` with a negative dt.
- `ad_lidar_perception/ad_lidar_perception/ab3dmot_core.py` **(modified,
  additive only)** — added `label`/`label_probability`/
  `existence_probability` passthrough fields to `Detection`, `Track`, and
  `TrackedState` (all with defaults, so every existing T-1A test still
  passes unchanged). These are never read by the KF/association/lifecycle
  math — they exist only to carry a detection's classification/score onto
  its matched track for the ROS mapping, explicitly modeled on AB3DMOT's
  own reference `Tracker.update()`, which carries an analogous per-match
  `info` payload the same way (updated only on a matched update, unchanged
  on coast frames). No algorithm logic changed.
- `ad_lidar_perception/launch/ab3dmot_tracker.launch.py` **(new)** — loads
  the existing `ad_lidar_perception/config/tracking/ab3dmot.yaml` as the
  node's parameters file (association_metric/matcher/min_hits/max_age/
  giou_gate all come from that YAML, unchanged from T-1A), plus launch
  args for `enabled`/topics/`target_frame`/`ab3dmot_root`.
- `ad_lidar_perception/CMakeLists.txt` **(modified)** — one new
  `install(PROGRAMS ... RENAME ad_ab3dmot_tracker)` entry (same pattern as
  `ad_centerpoint_detector`) and two new `ament_add_pytest_test` entries.
  `config`/`launch` directories are already installed wholesale by an
  existing rule, so the new YAML/launch files needed no new install rule.
- `ad_lidar_perception/test/test_ab3dmot_ros.py`,
  `ad_lidar_perception/test/test_ab3dmot_tracker_node.py` **(new)** — see
  Tests below.
- `docs/agent/STATUS.md` (this update).

No change to `package.xml` was needed: `tf2_ros`/`tf2_geometry_msgs`/
`geometry_msgs`/`rclpy` were already declared dependencies; the new code
never directly imports `unique_identifier_msgs` (the UUID field is
populated by assigning a numpy array to the already-auto-constructed
`object_id.uuid`, not by constructing a `UUID()` message).

### Submodule import from the installed environment — verified, not just source-tree pytest

Explicitly checked (this task's specific concern): after a real
`colcon build --symlink-install --packages-select ad_lidar_perception`,
`ad_lidar_perception.ab3dmot_core.__file__` resolves to a *symlink* into
the install tree, and `Path(__file__).resolve()` follows that symlink back
to the real source-tree file — so `_DEFAULT_AB3DMOT_ROOT` still correctly
locates `references/ab3dmot` from the installed package, with no code
change needed. Verified directly: `_load_ab3dmot_kf_class()` (and the full
test suite) both pass when run only after sourcing
`~/projects/heven_ros_ws/install/setup.bash` (not the source-tree
`PYTHONPATH` shortcut used for quick iteration). **Caveat worth recording**:
this specifically relies on HEVEN's established `--symlink-install`
convention; a plain (non-symlink) colcon install would copy the `.py` file
and break this path-inference, at which point `ab3dmot_root` would need to
be passed explicitly as a launch argument/parameter (already supported —
see `ab3dmot_tracker.launch.py`'s `ab3dmot_root` arg) rather than relying
on the default. Not a blocker today; flagged for whoever changes the build
convention later.

### Tests/builds run

`colcon build --packages-select ad_lidar_perception` (twice: after the
node/launch/CMake changes, and again after final edits) — both clean.

`python -m unittest test_ab3dmot_geometry test_ab3dmot_core test_ab3dmot_ros
test_ab3dmot_tracker_node test_centerpoint_ros test_morai_replay
test_detection_recording`, run twice: once via a source-tree `PYTHONPATH`
override, once via the fully-sourced **installed** environment
(`install/setup.bash`) to satisfy the installed-environment verification
requirement directly, not just by inference.

**Result: 75/75 pass both times** (63 AB3DMOT-related: 8 geometry + 14
core [unchanged from T-1A] + 31 new `ab3dmot_ros` + 10 new
`ab3dmot_tracker_node`, covering every category this task listed —
DetectedObjects conversion, odom-frame output, timestamp preservation,
real-dt propagation through the full adapter [node-level velocity
converges to the true m/s speed], first-frame behavior, duplicate-
timestamp rejection, non-positive/malformed-stamp rejection, clock-
rollback tracker reset, stable track identity [same UUID across frames,
distinct across track ids], velocity output in m/s, empty detections
[proven independent of TF availability], TF-unavailable/failure handling,
and malformed/unsupported objects [non-bounding-box shape, non-positive
dimensions, empty classification] — plus 12 pre-existing, unaffected).

Also live-smoke-tested via `ros2 launch ad_lidar_perception
ab3dmot_tracker.launch.py enabled:=true`: node starts cleanly, subscribes
`/ad/perception/objects/detected`, publishes on
`/experiment/tracked/ab3dmot`; a one-shot synthetic `DetectedObjects`
message was published against it with no TF broadcaster running, and the
node correctly logged a TF-unavailable warning and did not crash or
publish — confirms the same graceful-failure path the unit tests exercise
in isolation also works against the real ROS graph. Process cleaned up
afterward.

### Blockers

None. T-1B passes fully. The symlink-install caveat above is documented,
not blocking.

## Previous: T-1A — AB3DMOT tracking core + focused unit tests

**T-1A: PASS**, committed as `cceef3d` (2026-08-18). Standalone,
ROS2-independent tracking core (`ab3dmot_geometry.py`, `ab3dmot_config.py`,
`ab3dmot_core.py`) + 22 tests. Full detail in this file's prior revision
(git history) and `docs/research/tracking_architecture.md`.

## Previous: AB3DMOT integration decisions / CP-1

Resolved 2026-08-18: yaw convention, tracking frame (`odom`), initial
config, TrackedObjects mapping, ROS interface — full detail in
`docs/research/tracking_architecture.md` "AB3DMOT Integration Decisions".
CP-1 (CenterPoint) passed and merged to `main` in PR #2 / commit `aa5cacb`
(see `docs/research/centerpoint_status.md`).

## Git state at end of T-1B

**A. Pre-existing unrelated dirty files** (recorded before this task
started, unchanged, not staged/touched):

```
.claude/skills/bootstrap-repo/scripts/bootstrap-repo.sh
.claude/skills/bootstrap-repo/tests/test_bootstrap_repo.sh
.claude/skills/issue-worker/scripts/claim_issue.sh
.claude/skills/issue-worker/tests/test_claim_issue.sh
ops/runner/bin/agent-tick
ops/runner/bin/codex-pr-review.sh
ops/runner/bin/pr-set-in-review.sh
ops/runner/bin/render_prompt.sh
ops/runner/repo-context.sh
ops/runner/tests/test_env_precedence.sh
ops/runner/tests/test_pr_set_in_review.sh
scripts/apply_dependency_patches.sh
scripts/bootstrap_workspace.sh
scripts/check_autoware_perception.py
scripts/setup_dev_env.sh
scripts/test_python.sh
scripts/tests/test_verify_template_contract.sh
scripts/verify-template-contract.sh
scripts/verify_ad_data.py
```

**B. T-1B changes** (all AB3DMOT-related, verified against the above — no
overlap):

```
 M ad_lidar_perception/CMakeLists.txt
 M ad_lidar_perception/ad_lidar_perception/ab3dmot_core.py
?? ad_lidar_perception/ad_lidar_perception/ab3dmot_ros.py
?? ad_lidar_perception/ad_lidar_perception/ab3dmot_tracker_node.py
?? ad_lidar_perception/launch/ab3dmot_tracker.launch.py
?? ad_lidar_perception/test/test_ab3dmot_ros.py
?? ad_lidar_perception/test/test_ab3dmot_tracker_node.py
```
(plus this `docs/agent/STATUS.md` update, and T-1A's already-committed
files from `cceef3d`, unchanged except the additive `ab3dmot_core.py` edit
above).

No Autoware tracker, Euclidean detector, CenterPoint detector, HEVEN
IMM/prediction, occupancy, RViz, or reference-submodule files were
touched (`references/ab3dmot`/`simpletrack`/`trackeval` all still clean
and pinned to their recorded SHAs). Not committed or pushed, per this
task's instruction.

## Exact next task: T-1C

Per this task's instruction: stop after T-1B tests/build succeed; do not
start T-1C. T-1C, once explicitly requested, is **runtime verification**
against real data — replaying a real MORAI bag (or the existing
`ad_publish_morai_frames`/`ad_record_detected_objects` comparison tooling
from `docs/perception/centerpoint_vs_euclidean_comparison.md`) through the
actual TF tree (`lidar_link -> base_link -> odom`, needs a real
localization/TF source running, unlike this session's TF-unavailable
smoke test) to confirm: (a) the node produces non-empty, geometrically
plausible tracks against real detections, (b) `/experiment/tracked/ab3dmot`
and `/ad/perception/objects/tracked` can run side-by-side without
interfering, and (c) latency/runtime cost is measured (per AGENTS.md
"Measure runtime/latency for competition-critical modules") — still no
RViz wiring change and no accuracy/performance claim beyond execution
evidence, per AGENTS.md "execution success is not performance validation."
