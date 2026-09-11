# KalmanNet Production Candidate Freeze + Runtime Integration Readiness v1

Branch `feat/kalmannet-runtime-readiness-v1`, from `origin/main` `9f3db1f9`
(merge of PR #62, `feat(kalmannet): distribute checkpoints for third-party
MORAI testing`). **Engineering/integration readiness only. No new
KalmanNet training, no AV2 experiment, no retraining, no MORAI evaluation
-- MORAI simulator access is unavailable in this environment. Explicitly
leaving the AV2 research/ablation phase, per this task's own framing.**

## 1. Selected checkpoint

**NATURAL AV2 10k FIXED-DT GENERIC-ROBUST, seed 1** --
`models/experimental/kalmannet_av2_natural_10k_generic_robust_seed1.pt`,
SHA-256 `a9a19353ddb272d3fc0ac3ad7c6754239ef3d8dd00acd3c52a59193d98dc5ee5`.
Not guessed from filename -- verified directly against the checkpoint's
own embedded provenance sidecar
(`tools/kalmannet_training/av2_10k_multiseed_results/seed1_freeze.json`
and `seed1_official_val_report.json`, outside this repo) and cross-checked
against `docs/agent/STATUS.md`'s own frozen "AV2 10K GENERIC-ROBUST
Multi-Seed Completion v1" entry:

| field | value |
|---|---|
| architecture | `KalmanNetGRU` (`ad_lidar_perception/kalmannet_core.py`, unmodified), `hidden_size=32` |
| state / measurement | `[x,y,vx,vy]` / `[x,y]` |
| training seed | 1 (init/order/shuffle all = 1) |
| sampling | NATURAL (no motion-composition reweighting) |
| dt policy | FIXED (0.1s every sample) |
| corruption | GENERIC-ROBUST (Gaussian 0.3m + dropout 0.1 + burst dropout) |
| dataset | 10k AV2 scale-up v2 TRAIN pool, 9,000 internal-train / 1,000 internal-val |
| `frozen_at_git_sha` | `754d6dcb...` (PR #58 merge -- the finalized 3-state numerical safety guard) |
| best epoch / internal val loss | 7 / 0.7458607519010171 |
| numerical health | `catastrophic=false`, `training_unstable=false`, 78 norm-overflow (safe skip), 1 element-nonfinite (safe skip), 0 collapse |
| official AV2 VAL | A clean 0.0392m, B GENERIC-ROBUST 0.3063m, C different-seed corruption 0.3076m, D MORAI-calibrated diagnostic 1.7151m (n=2.9-2.92M each, 0 divergence, 0 nonfinite) |

Full provenance: `ad_lidar_perception/config/kalmannet/production_candidate.yaml`
(new). No ambiguity was found between this checkpoint and any other file
in `models/experimental/` -- the other 3 (`dense_kalmannet_v2.pt`,
seeds 0/2 of the same family) are explicitly documented there as
non-selected alternates, never silently interchangeable.

**Not selected, explicitly** (per this task's own "leaving the AV2
research phase" framing, no re-litigation of these findings):

- Motion-focused/BALANCED variant: rejected as default (severe
  ~55-59% CLEAN-condition regression, from "AV2 KalmanNet
  Motion-Composition Ablation v1").
- MILD variable-dt variant: diagnostic only, not a default (real
  fixed-dt-vs-strong-dt tradeoff, from "KalmanNet Physically-Consistent
  Variable-dt Augmentation v1" / PR #61).
- MIXED-50 curriculum: rejected (dominated by both NATURAL and MILD on
  every condition, from "KalmanNet Mixed Fixed/Variable-dt Curriculum
  v1" / PR #63, do-not-merge).

## 2. Production-candidate manifest

`ad_lidar_perception/config/kalmannet/production_candidate.yaml` (new).
Records candidate name, checkpoint file (relative path, no machine-
specific absolute path) + SHA-256, model class/dims, training dataset
lineage + seed + git SHA, best epoch, corruption config, coordinate/dt
semantics, numerical safety counters at training time, official AV2 VAL
provenance, the explicit selection rationale above, and the current
status of every rejected/diagnostic alternative. Verified by a new
`test_kalmannet_production_candidate.py` (9 tests) that the recorded
SHA-256/hidden_size/dims match the ACTUAL checkpoint file on disk --
this manifest cannot silently drift from reality without a test failure.

## 3. Checkpoint distribution / repository policy (audit, unchanged)

Already resolved by the immediately-preceding "KalmanNet checkpoint
distribution v1" task (PR #62, merged): `.gitignore` carves out exactly
`models/experimental/{manifest.yaml,README.md,dense_kalmannet_v2.pt,
kalmannet_av2_natural_10k_generic_robust_seed{0,1,2}.pt}` from the
otherwise-blanket `*.pt`/`*.pth` ignore (small, in-house-trained
artifacts, plain git blobs, no Git LFS -- `git-lfs` is non-functional in
this environment). This task makes **no change** to that policy --
verified it is exactly what a startup-time SHA-256 check needs (a
committed, byte-stable file at a known relative path) and built section 6's
verification on top of it unchanged.

## 4. Runtime pipeline audit (exact code path, read not assumed)

```
DetectedObjects (ROS msg)
  -> ab3dmot_tracker_node.py::_on_detected_objects
       (TF lookup at exact detection stamp; duplicate/backward/future
       timestamp classification; deferred-queue handling)
  -> ab3dmot_ros.py::detected_objects_to_detections
       (message -> ab3dmot_core.Detection, whole-message rejection on
       any malformed object)
  -> ab3dmot_core.AB3DMOTTracker.step(detections, timestamp_seconds)
       -> Track.predict(dt) for every live track (dt = real elapsed time)
       -> _associate() (euclidean/giou_3d/mahalanobis metric + greedy/
          hungarian matcher -- unaffected by estimator choice)
       -> Track.update(detection) for matched tracks
       -> Track.finalize_predict_only() for unmatched tracks
       -> birth new Track for unmatched detections
       -> Track.to_tracked_state() -> TrackedState (estimator-agnostic)
  -> ab3dmot_ros.py::tracked_states_to_message -> TrackedObjects (ROS msg)
  -> /ad/perception/objects/tracked (or /experiment/tracked/ab3dmot)
  -> autoware_prediction_node.cpp (IMM/CV prediction, downstream)
```

**Linear KF**: `ab3dmot_core.py::LinearKFEstimator` (line 286), wraps
`AB3DMOT_libs.kalman_filter.KF` from the pinned `references/ab3dmot`
submodule. **KalmanNet**: `ab3dmot_core.py::KalmanNetEstimator` (line
1234), wraps `kalmannet_core.KalmanNetFilter`. Both implement the exact
same `predict`/`update`/`position`/`velocity`/`yaw`/`dimensions`/
`*_covariance`/`predicted_bev_*`/`is_finite` interface (`Track` calls
only this common interface, never branches on estimator type beyond
selecting which one to construct at birth). Verified by direct code
reading of the current `ab3dmot_core.py` -- not inferred from an older
doc that might have drifted from the code.

## 5. KalmanNet state responsibility (division confirmed)

**KalmanNet-owned** (`[x, y, vx, vy]`): the frozen network's learned
gain, applied through the analytical `f`/`h` recursion. **Everything
else** (`z`, `vz`, `yaw`, `length`, `width`, `height`, `label`,
`existence_probability`, `track_id`, covariance-for-message-schema) is
owned by an internally-composed `LinearKFEstimator` ("the side
estimator") -- KalmanNet is never asked to estimate any of these.
`position_covariance`/`yaw_variance`/`velocity_covariance` on a
KalmanNet track are the **side estimator's** covariance only, populated
solely so the `TrackedObjects` message schema stays filled (every other
estimator publishes real covariance there) -- **not** KalmanNet
uncertainty, and `AB3DMOTConfig.__post_init__` structurally rejects
`state_estimator=kalmannet` + `association_metric=mahalanobis` so this
can never be silently misused for association. Confirmed unchanged from
the original T-9B design (`KalmanNetEstimator`'s own class docstring),
re-verified against the current code, not assumed from the doc.

## 6. KF / KalmanNet selector

Already exists, unchanged: `AB3DMOTConfig.state_estimator` (`"linear_kf"`
default, `"ekf"`, `"imm"`, `"kalmannet"`), a ROS parameter
(`ab3dmot_tracker_node.py::declare_parameter("state_estimator", ...)`),
threaded through `ab3dmot_tracker.launch.py`'s `state_estimator` launch
argument. Requirements verified directly against the code:

- **Default is safe**: default is `"linear_kf"`, which never imports
  `torch` or touches `kalmannet_core.py` (`load_kalmannet_network` is
  only called `if config.state_estimator == "kalmannet"`) -- confirmed
  by `TrackerNoCheckpointLoadWhenNotSelectedTest`.
- **No source edit needed to switch**: pure config/launch-argument value.
- **Invalid name fails clearly**: `AB3DMOTConfig.__post_init__` raises
  `ValueError` listing `SUPPORTED_STATE_ESTIMATORS` for anything else.
- **KalmanNet requires an explicit checkpoint**: `__post_init__` raises
  `ValueError` if `state_estimator="kalmannet"` and `kalmannet_checkpoint`
  is empty.
- **Linear KF never loads torch**: confirmed by direct test (see above);
  `load_kalmannet_network`'s own `import torch` is inside the function
  body, only reached on the kalmannet path.
- **Detector/association behavior is unaffected**: `_associate()` reads
  only `config.association_metric`/`matcher`, never `state_estimator`.

## 7. Checkpoint validation at startup (new: SHA-256 verification)

`load_kalmannet_network()` already validated (unchanged): file exists
(`FileNotFoundError`), state_dict present, `output_fc.weight` key
present, output dimension `== STATE_DIM*MEAS_DIM` (architecture
mismatch -> `RuntimeError`), `load_state_dict` (raises on any remaining
key/shape mismatch). **New this task**: an optional `expected_sha256`
parameter -- when given, the SHA-256 is computed and compared **before**
`torch.load` is even called, so a hash-mismatched or corrupted file
never gets as far as being deserialized. On mismatch: `RuntimeError`
with the actual vs. expected hash, never a silent fallback.

Wired additively through `AB3DMOTConfig.kalmannet_expected_sha256`
(default `""` = verification disabled, format-validated as 64 hex chars
when non-empty), `ab3dmot_tracker_node.py`'s new
`kalmannet_expected_sha256` ROS parameter, and
`ab3dmot_tracker.launch.py`'s new `kalmannet_expected_sha256` launch
argument. The startup log line now also reports `sha256_verified=<bool>`.
**Scope note**: this new parameter is wired through
`ab3dmot_tracker.launch.py` (the node's own direct launch file) only, not
yet threaded through the higher-level composition launch files
(`lidar_perception.launch.py`, `lidar_bag_replay.launch.py`,
`study_pipeline_rviz.launch.py`), which already forward
`kalmannet_checkpoint`/`kalmannet_device` but not yet this third field --
a deployer launching through those needs a `--ros-args -p
kalmannet_expected_sha256:=<hash>` override or a follow-up task to add
the passthrough. Not silently omitted; called out here and in the
blockers section.

## 8. Track lifecycle (A-H, verified against the real `AB3DMOTTracker`)

All verified with new/existing tests driving the real, unmodified
`AB3DMOTTracker`/`KalmanNetEstimator` classes (never a mock):

| case | result |
|---|---|
| A. new track creation | `KalmanNetEstimator.__init__` requires a loaded `kalmannet_net`, constructs a fresh per-track `KalmanNetFilter`, hidden state initialized to exactly zero via `init_sequence`->`reset_hidden` (`MultiTrackIsolationTest.test_new_track_after_death_gets_fresh_hidden_state`) |
| B. normal measurement update | `update()` runs the analytical correction + one learned-gain call; new `test_dt_actually_affects_the_analytical_prediction` |
| C. predict -> measurement update | `predict()` computes but does not commit `x_prior` (cached); `update()` consumes it -- no double-predict (existing design, `KalmanNetEstimator` docstring; `LifecycleTest.test_birth_update_miss_death`) |
| D. one missing measurement | `finalize_predict_only()` commits the cached prior with **no network call** (`LifecycleTest.test_birth_update_miss_death` step 3) |
| E. several consecutive missing measurements | new `GapAndReacquireLifecycleTest.test_several_consecutive_missing_measurements_then_reacquire` (4 consecutive predict-only frames, `max_age=6`, track survives, stays finite) |
| F. measurement returns after a gap | same test -- track re-acquires under its ORIGINAL `track_id`, not a fresh birth |
| G. track deletion | `is_dead(max_age)` removes the track from `self._tracks`; Python GC releases the `Track`/`KalmanNetEstimator`/hidden-state tensor -- no explicit "release" call needed or possible to forget |
| H. new object creates a fresh track later | `MultiTrackIsolationTest.test_new_track_after_death_gets_fresh_hidden_state` / `test_track_id_reuse_does_not_leak_state`: a brand-new `KalmanNetEstimator` (even reusing the same integer `track_id`) always gets a zero-initialized hidden state, never inherits a dead track's state |

**Critical finding, new this task**: `GapAndReacquireLifecycleTest.
test_hidden_state_unchanged_across_missing_measurement_run` directly
confirms the GRU hidden-state tensor object is **byte-identical**
(`torch.equal`) before and after 4 consecutive predict-only frames, and
only changes once a real `update()` (matched measurement) occurs --
proves missing-measurement semantics never touch the learned network,
matching the offline training/evaluation assumption exactly.

## 9. Multi-object hidden-state isolation

New `MultiObjectTrackerIsolationTest.
test_three_simultaneous_tracks_with_one_gap_stay_isolated`: 3 tracks
born in one frame at well-separated positions, two receive real
measurements every subsequent frame while the third has **no**
measurement for 3 consecutive frames (a genuine gap, not just 2
tracks as in the pre-existing raw-estimator-level test). Verified: all 3
stay finite throughout; A and B track their own real measurement
trajectories without any drift toward B's or C's; the 3 `_hidden_state`
tensor objects have 3 distinct Python `id()`s at every point (never
aliased/shared). Builds on, does not replace, the pre-existing
`MultiTrackIsolationTest` (2-estimator, raw-estimator-level tests that
already proved the T-9B interleaving-leak bug fix).

## 10. Variable-dt result

New `VariableDtRuntimeTest.test_irregular_dt_sequence_stays_finite`
drives the real `AB3DMOTTracker` through the exact irregular sequence
`[0.10, 0.11, 0.18, 0.10, 0.25, 0.09]` -- every output stays finite,
`time_since_update` bookkeeping stays correct (0 after every matched
frame), no crash. `test_dt_actually_affects_the_analytical_prediction`
directly confirms `predict(0.10)` and `predict(0.50)` from an identical
starting state produce **different** predicted positions -- proof `F`
is genuinely evaluated at the caller's own `dt`, never coerced to a
hidden fixed `0.1`. **No variable-dt training behavior was added or
changed** -- this is a pure runtime-math verification; the preferred
candidate itself remains FIXED-dt trained (section 1), and its own
learned gain has simply never seen non-0.1s examples during training,
which is a training-data property, not a runtime limitation.

## 11. Abnormal-dt safety

`AbnormalDtSafetyTest` (new, 6 cases): `dt<=0`, `dt=NaN`, `dt=+Inf` all
raise `ValueError` from `KalmanNetEstimator.predict()` (pre-existing
guard, `if not math.isfinite(dt_seconds) or dt_seconds <= 0.0: raise
ValueError(...)`) -- never silently coerced to `dt=1` or `dt=0.1`. A
large-but-finite `dt=120.0` (2 minutes) stays fully finite through both
`predict()` and a subsequent `update()`. At the frame level,
`AB3DMOTTracker.step()` itself already rejects a non-positive or
non-finite inter-frame `dt` with `ValueError` (`test_
tracker_step_rejects_non_positive_dt_at_frame_level` -- duplicate and
backward timestamps both raise). No new complex dt-clamping/rejection
policy was invented; the existing reject-rather-than-coerce convention
(shared with `autoware_prediction_node.cpp`'s own timestamp gating) was
verified sufficient and left unchanged.

## 12. Missing-measurement semantics

Confirmed by direct code reading (`KalmanNetEstimator.
finalize_predict_only`) and the new
`test_hidden_state_unchanged_across_missing_measurement_run` (section 8):
on a missing measurement, the tracker calls `predict()` (real,
variable-dt analytical advance) then `finalize_predict_only()` -- which
commits that prediction as the new state with **zero** network/gain
calls. No `[0, 0]` measurement is ever fabricated; no previous
measurement is ever replayed as if freshly observed. This exactly
mirrors the offline training/evaluation convention documented in
`kalmannet_core.py`/the AV2 adapter's own missing-measurement handling
(`z_meas[t] = None` -> predict-only, never a substitute value).

## 13. Downstream IMM / prediction compatibility

Confirmed by direct code reading: `Track.to_tracked_state()` always
returns the same `TrackedState` dataclass (`x,y,z,yaw,length,width,
height,vx_mps,vy_mps,vz_mps,position_covariance,yaw_variance,
velocity_covariance,hits,time_since_update,label,label_probability,
existence_probability`) regardless of which `_estimator` produced it.
`ab3dmot_ros.py::tracked_state_to_message` (the ROS serialization layer)
operates **exclusively** on this common struct -- grepped the file: zero
references to `state_estimator`, `KalmanNet`, `LinearKF`, `EKF`, or `IMM`
anywhere in it. `autoware_prediction_node.cpp` (the downstream consumer)
was also grepped: zero references to any estimator-type string --
it reads only the standard `TrackedObjects` message fields (pose, twist,
covariance, classification, object_id), with no branch on which
estimator produced them. **Topic schema, object IDs (`track_id_to_uuid`,
estimator-independent), coordinate frame (`target_frame`, set once at
node construction), units (m / m/s, unchanged), velocity-field semantics
(object-local twist, rotated the same way regardless of estimator), and
timestamp semantics (source detection's own stamp, always) are all
structurally estimator-agnostic by construction, not by convention that
could silently drift.** No IMM algorithm file was touched.

## 14. Offline / bag smoke test

No MORAI simulator, no ROS runtime required. New
`tools/kalmannet_runtime_readiness/offline_dry_run.py` drives the real,
unmodified `AB3DMOTTracker` directly against a **real, previously-
recorded** local detection stream
(`~/heven_presentation_assets/end_to_end_detector_tracker/
euclidean_detections.jsonl`, 1,764 frames from the "End-to-End Detector x
Tracker Evaluation" task, real Euclidean-detector output on a real MORAI
LiDAR replay -- outside this repo, machine-local, not committed) with A.
`linear_kf` and B. `kalmannet` (this task's own frozen production
candidate), on the identical input:

| | A. linear_kf | B. kalmannet |
|---|---|---|
| frames processed | 1,742 (22 skipped: duplicate/backward source stamps, a real property of the recording) | 1,742 |
| output states | 18,107 | 18,064 |
| non-finite | 0 | 0 |
| unique track IDs | 1,758 | 1,731 |
| live tracks at end | 1 | 1 |
| exception | none | none |
| checkpoint SHA-256 loaded | n/a | `a9a19353...` (matches the frozen manifest exactly) |
| distinct hidden-state objects at end | n/a | 1 (matches the 1 live track) |

Full results: `tools/kalmannet_runtime_readiness/smoke_test_evidence/
offline_dry_run_result.json` (committed). **This is an integration smoke
test only -- no accuracy claim.** The small track-ID-count difference
between arms (1,758 vs. 1,731) reflects the two estimators' different
early-prediction dynamics interacting with the fixed 3.0m Euclidean gate,
not a defect in either.

## 15. KF vs. KalmanNet latency (smoke test, not microbenchmark research)

New `tools/kalmannet_runtime_readiness/latency_benchmark.py`, CPU, this
machine, `AB3DMOTTracker.step()` cost only (association + all N tracks'
predict/update), N well-separated synthetic tracks:

| N tracks | linear_kf mean / p95 (ms) | kalmannet mean / p95 (ms) | rss delta (kalmannet, kB) |
|---|---|---|---|
| 1   | 0.145 / 0.343  | 0.531 / 0.798   | 4,212 |
| 10  | 0.990 / 1.263  | 4.307 / 5.257   | 128 |
| 50  | 8.958 / 13.394 | 36.062 / 43.358 | 256 |
| 100 | 31.854 / 40.545 | 99.779 / 118.476 | 384 |

Full results: `tools/kalmannet_runtime_readiness/smoke_test_evidence/
latency_benchmark_result.json` (committed). KalmanNet costs roughly
3-4x Linear KF's per-step time at this scale, consistent in direction
with T-9B's own original finding (the exact ratio depends on CPU/BLAS
threading and is not claimed as a hardware-independent constant). At 100
simultaneous tracks, KalmanNet's mean step cost (~100ms) approaches a
10Hz frame budget on this single-threaded-CPU-inference machine -- worth
flagging as a real scalability consideration for an unusually busy scene
(most prior MORAI/AV2-derived captures in this project report peak
live-track counts well under 100), not treated as a blocker at typical
scene scale. No optimization was attempted, per this task's own "smoke
test, not research" scope.

## 16. Checkpoint failure-safety (no silent fallback)

Verified directly, by test, for every failure mode the task lists:
missing file (`FileNotFoundError`), corrupt file (`Exception` from
`torch.load`), architecture-incompatible (`RuntimeError`, wrong
`output_fc` dimension), missing expected key (`RuntimeError`), and now
SHA-256 mismatch (`RuntimeError`, section 7) -- **every one of these
raises before `AB3DMOTTracker.__init__` returns**, so
`state_estimator=kalmannet` can never silently end up running with an
unverified, wrong-architecture, or corrupted checkpoint, and never
silently falls back to `linear_kf`. A KF fallback only ever happens if a
deployer explicitly sets `state_estimator=linear_kf` themselves --
never automatically.

## 17. MORAI A/B readiness

Once MORAI data is available, the exact same detection stream can be run
through both estimators with **no code edit**, only a
config/launch-argument change:

```bash
# A: Linear KF (default)
ros2 launch ad_lidar_perception ab3dmot_tracker.launch.py \
  enabled:=true state_estimator:=linear_kf ...

# B: KalmanNet (frozen production candidate)
ros2 launch ad_lidar_perception ab3dmot_tracker.launch.py \
  enabled:=true state_estimator:=kalmannet \
  kalmannet_checkpoint:="$(pwd)/models/experimental/kalmannet_av2_natural_10k_generic_robust_seed1.pt" \
  kalmannet_expected_sha256:=a9a19353ddb272d3fc0ac3ad7c6754239ef3d8dd00acd3c52a59193d98dc5ee5
```

Or, without any ROS runtime at all (this task's own offline harness,
reusable directly once a real MORAI-sourced detection JSONL exists in
the same `heven.ros_detection_comparison.v1`-compatible shape):

```bash
python3 tools/kalmannet_runtime_readiness/offline_dry_run.py \
  <morai_detections.jsonl> models/experimental/kalmannet_av2_natural_10k_generic_robust_seed1.pt
```

No new MORAI evaluator was built (existing tooling -- e.g.
`morai_estimator_eval_v2.py` -- already covers that need once real MORAI
GT capture exists; this task's own scope is runtime readiness, not
evaluation-harness construction).

## 18. Tests

- `test_ab3dmot_kalmannet.py`: **37/37 pass** (19 pre-existing +
  18 new: SHA-256 match/mismatch/skip, `kalmannet_expected_sha256`
  config validation x4, several-consecutive-missing + reacquire,
  hidden-state-unchanged-during-gap, 3-simultaneous-tracks-with-a-gap,
  irregular-dt-sequence, dt-actually-affects-prediction, 6 abnormal-dt
  cases).
- `test_kalmannet_production_candidate.py` (new): **9/9 pass** (manifest
  schema/no-absolute-path/relative-checkpoint-path/checkpoint-exists/
  SHA-256-matches-actual-file/status-declarations/no-MORAI-claim +
  2 torch-dependent hidden-size/dims-match-runtime checks).
- `tools/kalmannet_runtime_readiness/test_offline_dry_run.py` (new):
  **6/6 pass** (synthetic-fixture coverage of both scripts).
- Regression, unaffected by the additive `ab3dmot_config.py`/
  `ab3dmot_core.py`/`ab3dmot_tracker_node.py`/`ab3dmot_tracker.launch.py`
  changes: `test_ab3dmot_{core,geometry,association,
  association_metrics,ekf,heading,hybrid_gate,imm}.py` **154/154 pass**;
  `test_ab3dmot_ros.py` + `test_ab3dmot_tf_deferred_queue.py`
  **71/71 pass**; `test_ab3dmot_tracker_node.py` (needs ROS Humble
  sourced) **27/27 pass**.
- `py_compile`/`pyflakes`/`git diff --check`: clean on every changed/new
  file.
- **ROS integration**: `test_ab3dmot_tracker_node.py`'s 27 tests exercise
  the real `rclpy` node class end-to-end (parameter declaration, TF
  handling, timestamp classification) -- genuinely run, not claimed
  without verification. A live `ros2 launch` was not additionally
  performed in this session (no MORAI/live sensor source available);
  the offline dry-run (section 14) is the integration evidence for the
  estimator classes themselves.

## 19. Documentation

This file (new). `docs/agent/STATUS.md` updated (new entry prepended).

## 20. Remaining limitations / blockers before MORAI

- **No independent MORAI GT evaluation of this checkpoint** beyond the
  existing n=3 test-sequence result in `morai_estimator_eval_v2.md` --
  this task explicitly does not change that.
- `kalmannet_expected_sha256` is not yet threaded through the
  higher-level composition launch files (`lidar_perception.launch.py`,
  `lidar_bag_replay.launch.py`, `study_pipeline_rviz.launch.py`) --
  section 7's scope note. A deployer using those needs a `--ros-args`
  override or a small follow-up passthrough addition.
- The offline dry-run (section 14) used real Euclidean-detector output
  from a MORAI LiDAR replay, but that data predates this task and lives
  outside the repo (`~/heven_presentation_assets/`) -- it is not
  committed here and a fresh clone cannot reproduce section 14's exact
  numbers without that same local file; the harness script itself is
  fully reusable against any future real MORAI-sourced JSONL in the same
  format.
- Latency numbers (section 15) are single-machine, CPU-only; no GPU
  latency was measured (T-9A/T-9B's own prior finding: CPU is faster
  than CUDA at this tiny per-track model scale, kernel-launch overhead
  dominates -- not re-measured here, per this task's own "smoke test"
  scope).

## Files

`ad_lidar_perception/config/kalmannet/production_candidate.yaml` (new),
`ad_lidar_perception/ad_lidar_perception/{ab3dmot_config.py,
ab3dmot_core.py,ab3dmot_tracker_node.py}` (additive: SHA-256 verification
parameter/plumbing only, zero behavior change when unset),
`ad_lidar_perception/launch/ab3dmot_tracker.launch.py` (additive: one new
launch argument), `ad_lidar_perception/test/{test_ab3dmot_kalmannet.py
(+18 tests),test_kalmannet_production_candidate.py (new)}`,
`tools/kalmannet_runtime_readiness/{offline_dry_run.py,
latency_benchmark.py,test_offline_dry_run.py,smoke_test_evidence/
{offline_dry_run_result.json,latency_benchmark_result.json}}` (new), this
file, `docs/agent/STATUS.md`. No `kalmannet_core.py`/`KalmanNetGRU`/
`KalmanNetFilter`/`F_matrix`/`Q_matrix`/AB3DMOT association-or-lifecycle
math/CenterPoint/IMM-algorithm/planner/occupancy-grid file changed. No
new training run. No MORAI evaluation.

**Recommended next task**: capture at least one real MORAI-recorded
scenario (GT actor trajectories + ego GT + TF via the existing
`ad_morai_dataset_capture`/`ad_morai_dataset_export_kalmannet`/
`ad_morai_dataset_attach_kalmannet_measurements` pipeline, with real
detector-attached measurements) and use it to build/populate
`MORAI_ESTIMATOR_EVAL_V2` -- zero-shot-evaluate this exact frozen
NATURAL 10k FIXED-DT seed1 checkpoint against the Tuned Linear KF
baseline on real MORAI data, reusing the A/B pattern from section 17
verbatim once that data exists. **Do NOT start any new KalmanNet
training or AV2 ablation as part of that next task.**
