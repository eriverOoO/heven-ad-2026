# MORAI Frozen Estimator Evaluation Dataset v2

Evaluation-data task. **No model was trained or tuned in this task.**
Builds on merged PR #52 (`exp/kalmannet-morai-calibrated-corruption-v1`,
merge commit `764b4f82ebff57712ad25502982662617ce55a20`), verified via
`gh pr view 52` before branching.

## Why V1 is insufficient

MORAI_ESTIMATOR_EVAL_V1 (T-11/T-12 lineage, `morai_frozen_stream.py`) is
3 sequences / 154 frames, all drawn from actors 16, 20, 30 within one
single MORAI simulator run. Every comparison this project has run
against it (T-13, the domain-gap task, MORAI-Calibrated AV2 Corruption
v1) shows the same symptom: one sequence supplies essentially all
missing-measurement evidence, and a per-sequence bootstrap 95% CI over
just 3 units is wide enough to cross zero for nearly any real effect
size an estimator change could plausibly produce. 154 frames sliced from
3 trajectories is pseudo-replication, not 154 independent test cases --
the primary statistical unit this task is required to respect is the
independent actor trajectory (and, ideally, the independent simulator
run), not the frame.

## 1. Existing MORAI data inventory

`tools/kalmannet_training/morai_estimator_eval_v2.py::audit_source_runs()`
audits every locally-known MORAI-adjacent source (metadata/manifest
files only, no bag opened, no MORAI launched):

| source | run identity | usable? | why |
|---|---|---|---|
| `~/datasets/morai_heven/` (1764 GT+LiDAR frames, 21 actors) | `static_20260805_003151` | usable, but **not new** | the sole real MORAI run with actor GT ever captured; every actor already assigned TRAIN/VAL/TEST/excluded |
| `~/heven_presentation_assets/motion_gt_expansion/canonical/` (frozen GT+detection export derived from the same scene) | `static_20260805_003151` | usable, but **not new** | same underlying run as above, not an independent source |
| `bags/static_20260805_003151/` (git-tracked, metadata-only) | `static_20260805_003151` | **unusable** | `.mcap` payload absent from disk (confirmed: `ls` shows only `metadata.yaml`); its own metadata (`starting_time_ns=1785857511293849250`, `/ad/sensors/lidar/points` 1778 msgs) closely matches `morai_heven`'s first exported label's own source stamp (1785857513201006723, ~1.9s later) and message-count order of magnitude (1764 exported vs. 1778 recorded) -- confirmed to be **the same run**, not a second one, even setting aside that it is unreplayable |
| `morai_cam4_20260813_163222` (real, replayable `.mcap`, untracked local artifact) | different, moving-ego recording | **unusable for this task** | genuinely a different, independent run from `static_20260805_003151` -- but its topic list (`metadata.yaml`, checked directly) contains no `/ad/dev/objects` or any actor-GT topic. Real sensor data, zero actor ground truth. |

**Conclusion: exactly one MORAI simulator run with actor ground truth has
ever been captured in this project's history.** This matches every
prior session's own finding (T-11B's manual-capture protocol was never
executed; MORAI's simulator/gRPC bridge has never been available in any
sandboxed session of this project).

## 2. V2 data availability

`audit_actor_leakage()` enumerates all 21 actors in the sole captured
run against `motion_gt_expansion/canonical/split_policy.json` (T-11's
own frozen-before-T-12 split) plus this project's own subsequent
consumers:

| role | actor IDs | n | consumed by |
|---|---|---|---|
| train | 1, 13, 15, 18, 34, 35, 36, 38, 40, 44, 46, 48 | 12 | T-9A.1 KF tuning, T-12 KalmanNet training, dense-v2 training, this project's own MORAI-Calibrated AV2 Corruption v1 residual/gap calibration |
| val | 21, 23, 27, 31 | 4 | same consumers (model/hyperparameter selection) |
| **test (frozen V1, reused unchanged)** | **16, 20, 30** | **3** | MORAI_ESTIMATOR_EVAL_V1 -- **not counted as new** |
| excluded (GT quality, not leakage) | 2, 3 | 2 | none -- disqualified because both show vehicle-speed kinematics inconsistent with their `obstacle`/`pedestrian` class label (a real MORAI scenario labeling defect, per `split_policy.json`'s own `excluded_reason`) |

12 + 4 + 3 + 2 = 21 = every actor in the run. **Zero actors are eligible
for new V2 test data.** The excluded pair (2, 3) was explicitly
considered and rejected -- not because of leakage, but because their own
GT is not trustworthy enough to serve as evaluation ground truth either
(Section 11's allowed "physically impossible/unrecoverable source data"
exclusion criterion applies to them independently of this task).

## 3. New collection required? **YES**

Checked directly in this environment: `python3 -c "import grpc"` fails
(`ModuleNotFoundError`), no MORAI process is running, no simulator
installation is discoverable. Per this task's own Section 5 instruction,
**this task stops at a complete collection-ready implementation** (below)
rather than fabricating synthetic trajectories or reusing TRAIN/VAL
actors.

## 4. Dataset factory / alignment

`tools/kalmannet_training/morai_estimator_eval_v2.py` (new), reusing
this project's existing frozen-stream/eval conventions
(`morai_frozen_stream.py`'s actor-ID-guard pattern, `freeze_manifest.py`'s
save/load pattern, `domain_gap_analysis.py`'s bootstrap design) rather
than inventing a second unrelated pipeline:

- `SourceRunRecord`/`audit_source_runs()` -- Section 1's inventory, machine-independent (`~`-symbolic paths only, verified no absolute local path is embedded in committed output).
- `LeakageAuditRow`/`audit_actor_leakage()` -- Section 2's per-actor table, fails closed (an actor absent from the documented split is marked ineligible, never silently included).
- `build_v2_sequences(gt_frames, detection_frames, eligible_actor_ids, ...)` -- constructs one sequence per `(run_id, actor_id)` from **positionally joined** GT/detection frame lists (`gt_frames[i] <-> detection_frames[i]` by index), reproducing the exact, already-established alignment policy `motion_gt_expansion/canonical/provenance.json` documents: *"frame_index_join: ... same 1764-frame ordering throughout, positional join, not stamp-matched -- detection stamps are publish-time replay artifacts per established project convention."* Verified directly: the real detection replay's own `stamp_ns` (e.g. `1787286337875187707`) is on a completely different wall-clock epoch than the GT's own `header_stamp_ns` (e.g. `1785857513211291194`, ~23.8 minutes earlier) -- a real, measured confirmation that nearest-stamp matching would be meaningless here and positional join is the only correct policy, not an unjustified wide tolerance.
- `validate_v2_sequences()` -- the strict validator (Section 18).
- `build_v2_freeze_manifest()`/`save_v2_freeze_manifest()` -- the freeze machinery (Section 12).
- `paired_bootstrap()`/`run_level_macro_average()` -- the statistical design (Section 19).

CLI: `tools/kalmannet_training/build_morai_estimator_eval_v2.py`. Runs
the Section-1/2 audits **before** attempting any sequence construction
(validation-first, per Section 17's requirement) and writes
`morai_estimator_eval_v2_audit.json`. Given 0 eligible actors, this run
correctly stops after writing the audit -- it does not fabricate
sequences or silently fall back to a leaked actor. Deterministic and
restartable: re-running with the same inputs reproduces the same
report.

## 5. Coordinate / frame validation

Checked directly against a real `morai_heven` label file
(`static_20260805_003151_1785857513201006723.json`): GT boxes carry
`source_frame: "map"`, `target_frame: "lidar_link"`, and a recorded
`transform_alignment.chain = ["map", "odom", "base_link",
"rear_axle_link", "lidar_link"]` with the actual `map->odom` transform
(quaternion + translation) used to produce the stored `lidar_link`-frame
`x,y,yaw` -- so GT is genuinely ego-relative, not raw ENU map-frame,
despite MORAI's own native GT being ENU (`source_position_map`, also
recorded for provenance). The real frozen detector replay
(`motion_gt_expansion/canonical/detections/detections_full.jsonl`) is
independently confirmed `"frame_id": "lidar_link"` for every sampled
frame -- **the same frame as GT**, checked directly rather than assumed
plausible.

`morai_estimator_eval_v2.py` provides `assert_same_metric_frame()`
(raises `FrameMismatchError` on any GT/measurement frame_id mismatch,
called once per frame inside `build_v2_sequences`) and
`assert_valid_transform_chain()` (checks the expected 5-link
`map->odom->base_link->rear_axle_link->lidar_link` chain) as reusable,
tested assertions -- not a one-off manual check.

## 6. V2 inclusion / exclusion policy

**Inclusion** (would apply to any future new MORAI run): finite GT
position for every frame, monotonic per-actor timestamps, `dt>0`,
GT/detection frame_id match, actor not already consumed by any
TRAIN/VAL/calibration step. **Exclusion** (Section 11's allowed list
only): corrupted/non-finite GT, invalid timestamp, coordinate-transform
failure, or GT physically inconsistent with its own class label (the
precedent set by actors 2/3). **No model-dependent filtering anywhere**
-- these rules are defined in code (`validate_v2_sequences`,
`audit_actor_leakage`) before any estimator has been run against V2, and
there is currently no V2 estimator result to have filtered against
regardless.

## 7. Leakage audit

See Section 2's table above (the complete, real leakage table over all
21 actors). External AV2 overlap is correctly treated as irrelevant at
the actor level (per this task's own Section 13 note) -- Stage-0/Stage-1
AV2 KalmanNet checkpoints have zero MORAI training exposure; only the
Tuned Linear KF, dense-v2 KalmanNet, and this project's own MORAI
corruption calibration consumed MORAI TRAIN/VAL actors directly.

## 8. Frozen manifest

**No V2 freeze manifest was written for real data** -- there is no
eligible new sequence to freeze. `build_v2_freeze_manifest()`/
`save_v2_freeze_manifest()` are implemented and unit-tested (round-trip,
schema version, actor-id inclusion) against synthetic fixtures, ready
for a future session with real new MORAI captures. The one real,
committed artifact from this task is
`tools/kalmannet_training/frozen_configs/morai_estimator_eval_v2_audit.json`
(the Section 1/2 audit itself, run against real local data, machine-
independent).

## 9. Number of runs / sequences / frames

| | V1 (unchanged, reused) | V2 (real, this task) |
|---|---|---|
| independent simulator runs | 1 | 0 new |
| actor trajectories | 3 (actors 16, 20, 30) | 0 new |
| frames | 154 | 0 new |

## 10. Missing-measurement coverage

Not computed for V2 -- there are 0 new sequences. V1's own coverage
(154 frames, 24 missing = 15.3%, entirely from actor 16's single 24-frame
gap) is unchanged and already documented in the domain-gap and
corruption-calibration tasks' own reports.

## 11. Motion / class coverage

Not computed for V2, same reason. The sole run's full 21-actor
population is documented in `actor_manifest.csv` (already summarized in
prior tasks: predominantly `vehicle` class, median speed ~5.7-15.7 m/s
across actors, 2 actors flagged as class-mislabeled). If MORAI is
vehicle-only in every scenario captured so far -- stated plainly: yes,
19/21 actors are `vehicle`, the remaining 2 are the disqualified
`obstacle`/`pedestrian` pair whose kinematics don't match their label, so
there is effectively **no reliable non-vehicle class coverage** anywhere
in this project's MORAI GT to date.

## 12. Statistical evaluation design

Implemented and unit-tested (`paired_bootstrap`, `run_level_macro_average`):
frame-pooled RMSE, per-sequence RMSE distribution, macro-average across
sequences, and run-level macro-average are all supported; the paired
per-sequence bootstrap is the design's primary uncertainty estimate
(never frame-level bootstrap), with an explicit run-level variant ready
for whenever more than one independent run exists. With only one run
currently available, run-level bootstrap would trivially return `n=1`
and is not meaningful yet -- documented, not hidden.

## 13. Post-freeze estimator sanity result

**N/A.** No V2 freeze manifest with real sequences exists, so there is
nothing new to sanity-check. Re-running the existing frozen V1 stream
adds no information beyond what MORAI-Calibrated AV2 Corruption v1
already reported.

## Collection checklist (required before V2 can be populated)

MORAI is not available in this or any prior session of this project.
When a session WITH MORAI access runs this task's continuation:

1. **Environment**: confirm `python3 -c "import grpc"` succeeds and a
   MORAI simulator process (map `R_KR_PR_K-city_2025` or an equivalent
   scenario set) is reachable, matching this project's existing
   `ad_morai_bridge_dev.scenarios` reset/spawn tooling.
2. **Per run**: launch the existing training-free perception stack
   (`lidar_bag_replay.launch.py` precedent, or live capture) with the
   GT-logger path enabled (`/ad/dev/objects`, `/ad/dev/vehicle/ego_status`)
   alongside the normal runtime detector/tracker topics -- **the GT
   logger must be a visibly separate subscriber**, never feeding
   CenterPoint/association/KalmanNet/prediction/planner (Section 6's
   boundary; the runtime graph is unmodified by recording GT).
3. **Recommended count**: prefer several independent, shorter runs over
   one long one -- target >=5 runs / >=20 actor trajectories minimum,
   >=8-10 runs / 30-50 trajectories preferred, per this task's own
   collection targets. Vary scenario/traffic composition per run so
   actor encounters, starting conditions, and missing-measurement
   patterns are genuinely different, not the same scenario replayed.
4. **Per-run duration**: 60-120 s is enough to get several actor
   trajectories per run (the existing single run yielded 21 actors from
   ~360 s) -- shorter, more numerous runs are preferred over one very
   long capture, per this task's own instruction.
5. **Naming convention**: `<scenario_id>_<YYYYMMDD_HHMMSS>`, matching
   the existing `static_20260805_003151`-style convention already used
   throughout this project.
6. **Validate immediately after each run**:
   `python3 tools/kalmannet_training/build_morai_estimator_eval_v2.py
   --actor-manifest <new actor_manifest.csv> --split-policy <new
   split_policy.json> --output-dir <out> --split-role test` (once a
   per-run actor manifest / split policy exists for the new runs) --
   confirms 0 leakage against the existing TRAIN/VAL/TEST/excluded pool
   before the run is trusted as usable V2 data.
7. **After collecting enough runs**: use `build_v2_sequences()` +
   `validate_v2_sequences()` + `build_v2_freeze_manifest()` (all already
   implemented and tested in this task) to construct and freeze V2 --
   only then run the post-freeze sanity comparison (Section 22 of the
   task spec), never before.

## Tests

`test_morai_estimator_eval_v2.py`: 26 new (leakage audit incl.
fails-closed-on-unknown-actor; frame/transform-chain assertions;
sequence construction incl. missing-measurement-as-None, frame-mismatch
rejection, unequal-length rejection, deterministic sequence IDs;
validator incl. duplicate-ID/non-monotonic-timestamp/non-finite-GT
detection and short-trajectory/extreme-gap warnings; freeze-manifest
round-trip; content-hash determinism; paired-bootstrap determinism,
equal-length requirement, stability-when-separated; run-level macro
averaging and run-level bootstrap). Full `tools/kalmannet_training`
suite: **144/144 pass** (118 pre-existing unaffected + 26 new).
`py_compile`/`pyflakes`/`git diff --check` clean on every new file. No
absolute local path in the committed audit JSON (verified via grep
before commit).

## Files

- `tools/kalmannet_training/morai_estimator_eval_v2.py` (new)
- `tools/kalmannet_training/build_morai_estimator_eval_v2.py` (new)
- `tools/kalmannet_training/test_morai_estimator_eval_v2.py` (new)
- `tools/kalmannet_training/frozen_configs/morai_estimator_eval_v2_audit.json` (new, small/machine-independent -- the real Section 1/2 audit)
- `docs/perception/morai_estimator_eval_v2.md` (this file)
- `docs/agent/STATUS.md` (updated)

No `kalmannet_core.py`, CenterPoint, association, AB3DMOT runtime,
prediction, planner, or occupancy-grid file touched. No model trained or
tuned. No rosbag, raw GT log, large sequence dataset, checkpoint, or
local absolute path committed.

## Recommended next task

Execute the collection checklist above once MORAI access exists, then
resume this task's own remaining sections (7-24) to actually populate
and freeze MORAI_ESTIMATOR_EVAL_V2 and run the post-freeze sanity
comparison. Not started this session -- MORAI is unavailable here.
