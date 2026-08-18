# STATUS

## Current task: T-1A — AB3DMOT tracking core + focused unit tests

Branch: `feat/ab3dmot-tracker`. Implemented T-1A only: a standalone,
ROS2-independent HEVEN-compatible AB3DMOT tracking core, per the resolved
design in `docs/research/tracking_architecture.md` ("AB3DMOT Integration
Decisions"). No ROS2 live integration, no RViz changes, no redesign of the
already-resolved decisions.

### Files created

- `ad_lidar_perception/ad_lidar_perception/ab3dmot_geometry.py` — HEVEN
  z-up 3D GIoU (`Box3D`, `bev_corners`, `giou_3d`). Faithfully ports two
  frame-agnostic 2D primitives from `AB3DMOT_libs/dist_metrics.py`
  (`polygon_clip` Sutherland-Hodgman, `PolyArea2D` shoelace — verbatim
  algorithm, one deliberate safety deviation: a degenerate zero-denominator
  clip case returns "no intersection" instead of raising
  `ZeroDivisionError`). Does **not** reuse `box2corners3d_camcoord` or
  `dist_metrics.iou` (camera-frame-specific, per the already-resolved yaw
  decision) — recomputes the same GIoU_3D formula structure with z-up BEV
  corners and z-axis height range instead. Adds a small self-contained
  monotone-chain `convex_hull_2d` in place of AB3DMOT's
  `scipy.spatial.ConvexHull`, to avoid a new scipy dependency for this
  baseline.
- `ad_lidar_perception/ad_lidar_perception/ab3dmot_config.py` —
  `AB3DMOTConfig` dataclass holding the already-resolved baseline
  (`association_metric=giou_3d`, `matcher=greedy`, `min_hits=1`,
  `max_age=2`, `giou_gate=0.0`), validated, not hardcoded into the
  algorithm. Rejects any `matcher`/`association_metric` other than what's
  implemented (fails loudly rather than silently accepting an unsupported
  config value).
- `ad_lidar_perception/ad_lidar_perception/ab3dmot_core.py` — the tracking
  core: `Detection`, `TrackedState`, `Track`, `AB3DMOTTracker`. Imports
  `AB3DMOT_libs.kalman_filter.KF` directly from the pinned
  `references/ab3dmot` submodule (not forked) for the 10-state/
  7-measurement filter setup; `Track.predict(dt_seconds)` rebuilds the
  transition matrix with **real elapsed seconds** every step (mirrors
  SimpleTrack's documented `get_prediction()` fix, not AB3DMOT's implicit
  dt=1/frame) so the velocity state stays true m/s throughout, and raises
  `ValueError` on non-finite/non-positive dt instead of silently assuming
  dt=1. `_orientation_correction` and `_greedy_matching` are faithful ports
  of `AB3DMOT_libs/model.py::orientation_correction` and
  `AB3DMOT_libs/matching.py::greedy_matching` (ported rather than imported,
  to avoid `AB3DMOT_libs.matching`'s transitive `numba`/`scipy` deps, which
  exist there only for the camera-frame IoU path this module doesn't use).
  `AB3DMOTTracker.step()` implements predict → GIoU-cost association
  (greedy, gated by `giou_gate`) → update matched → birth unmatched → drop
  `time_since_update >= max_age` tracks → return confirmed tracks, matching
  AB3DMOT's `output()` condition exactly, including its early-sequence
  `frame_index <= min_hits` carve-out (preserved, not redesigned, per this
  task's instruction). Association is isolated in `AB3DMOTTracker._associate`
  so a Hungarian or other matcher can be added later without touching
  predict/update/lifecycle — not implemented now, per scope.
- `ad_lidar_perception/config/tracking/ab3dmot.yaml` — the same resolved
  baseline as a ROS-parameter-shaped YAML (not yet loaded by any node;
  documents the parameter surface for T-1B).
- `ad_lidar_perception/test/test_ab3dmot_geometry.py`,
  `ad_lidar_perception/test/test_ab3dmot_core.py` — see below.

### Files changed

- `ad_lidar_perception/CMakeLists.txt` — registered the two new test files
  via `ament_add_pytest_test` (same pattern as existing tests). No other
  change.

### New runtime dependency

`filterpy==1.4.5` (AB3DMOT's own pinned version, per its
`references/ab3dmot/requirements.txt`) installed into
`~/venvs/heven-centerpoint` — required to import
`AB3DMOT_libs.kalman_filter.KF`. Pure-Python, no CUDA/heavy deps. Not yet
recorded in a repo-tracked requirements file (T-1A has no ROS/build-system
wiring to hang one off yet); flag for T-1B to pin properly (e.g. alongside
`tools/centerpoint_offline/requirements-cu118.txt`'s pattern).

### Tests executed

`python -m unittest test_ab3dmot_geometry test_ab3dmot_core` (from
`ad_lidar_perception/test/`, both via a plain `PYTHONPATH` run and via the
colcon-installed `ad_lidar_perception` package after a successful
`colcon build --packages-select ad_lidar_perception`) plus the three
pre-existing test files (`test_centerpoint_ros`, `test_morai_replay`,
`test_detection_recording`) to confirm no regression.

**Result: 34/34 tests pass** (22 new: 8 geometry + 14 core covering all of
this task's required categories — KF predict/update with real dt, velocity
in m/s at two different frame rates, constant-velocity convergence, stable
track ID, two separated objects, greedy association [including a case
distinguishing "claims globally-cheapest pair" from "leaves no cutoff"],
temporary missed detection survives via coasting, `max_age` deletion +
new ID on reappearance, yaw wraparound via `_orientation_correction`'s
90-degree invariant and a live `Track.update` across the ±π discontinuity,
and z-up 3D GIoU geometry — plus 12 pre-existing, unaffected).

One test assertion was initially wrong (a hand-derived expectation for
`greedy_matching`'s conflict-resolution behavior assumed a cost cutoff
that doesn't exist inside that function — the gate is a separate step in
`_associate`); fixed the test after confirming the actual behavior was
correct per the ported reference algorithm, not a code bug.

### Blockers

None. No ambiguity in the documented design was exposed by the tests — the
resolved decisions in `docs/research/tracking_architecture.md` mapped onto
working code without needing a new algorithmic rule.
`docs/research/tracking_architecture.md` was **not** modified — no factual
contradiction was found.

### Git state at end of T-1A

Pre-existing dirty files (recorded before this task started, unchanged,
not staged/touched):

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

Files touched by T-1A (all AB3DMOT-related, verified against the above —
no overlap):

```
 M ad_lidar_perception/CMakeLists.txt
?? ad_lidar_perception/ad_lidar_perception/ab3dmot_config.py
?? ad_lidar_perception/ad_lidar_perception/ab3dmot_core.py
?? ad_lidar_perception/ad_lidar_perception/ab3dmot_geometry.py
?? ad_lidar_perception/config/tracking/ab3dmot.yaml
?? ad_lidar_perception/test/test_ab3dmot_core.py
?? ad_lidar_perception/test/test_ab3dmot_geometry.py
```

No Autoware tracker, Euclidean detector, CenterPoint detector, HEVEN
IMM/prediction, occupancy, RViz, or reference-submodule files were
touched (`references/ab3dmot`/`simpletrack`/`trackeval` all still clean
and pinned to their recorded SHAs). Not committed or pushed, per this
task's instruction.

## Previous: AB3DMOT integration decisions

Resolved 2026-08-18: yaw convention, tracking frame (transform to `odom`),
initial config (greedy/GIoU/min_hits=1/max_age=2), TrackedObjects
conversion mapping, ROS interface (`/experiment/tracked/ab3dmot`). Full
detail in `docs/research/tracking_architecture.md` "AB3DMOT Integration
Decisions" — this task (T-1A) implements exactly that design.

## Previous: CP-1 result

**CP-1: PASSED** (2026-08-18, merged to `main` in PR #2 / commit
`aa5cacb`). Full detail retained in `docs/research/centerpoint_status.md`.

## Remaining blocker

None for T-1A. Unrelated and unchanged: CenterPoint mAP/generalization
evaluation still blocked by dataset diversity (see
`docs/research/centerpoint_status.md`).

## Exact next task: T-1B

Per this task's instruction: stop after T-1A. T-1B, once explicitly
requested, is the ROS2 live integration: a new opt-in node (e.g.
`ad_ab3dmot_tracker`) that (a) subscribes
`/ad/perception/objects/detected` (`autoware_perception_msgs/msg/
DetectedObjects`), (b) transforms detections into `odom` via a `tf2_ros`
buffer (new code this node owns, per the tracking-frame decision), (c)
converts them to `ab3dmot_core.Detection` and drives `AB3DMOTTracker.step()`
each callback using the message header's real timestamp, (d) maps
`TrackedState` to `autoware_perception_msgs/msg/TrackedObjects` per the
already-resolved field table (position/orientation/dimensions direct;
velocity already in m/s; position/yaw/velocity covariance from the KF's
own `P`; untracked covariance dimensions left at safe defaults; track ID
encoded into a `unique_identifier_msgs/UUID`), and (e) publishes to
`/experiment/tracked/ab3dmot`, leaving `tracking.launch.py`, `autoware.yaml`,
and the production Autoware tracker completely untouched. RViz wiring
(reusing the existing generic `ad_viz` marker visualizer, same pattern as
the CenterPoint/Euclidean comparison tooling) comes after that, still
within T-1B or as a small follow-up — not started in T-1A.
