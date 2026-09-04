# Dynamic OGM Future Sweep v1 — swept predicted-object occupancy

**Scope:** one bounded, opt-in change in the Dynamic Occupancy Grid node
(`ad_lidar_perception/src/occupancy_grid/dynamic_occupancy_grid_node.cpp`) plus a
small pure helper in `dynamic_grid_builder.{hpp,cpp}`. **Not a motion predictor.**
No change to the motion-history yaw-rate estimator, the IMM, prediction
probabilities/horizons, AB3DMOT, KalmanNet, CenterPoint, the Hungarian matcher,
track lifecycle, the drivable-mask policy, `route_corridor.json`, or any planner
default.

## Problem

The dynamic layer rasterizes only each object's **current** footprint. The
node already receives the time-indexed predicted trajectory
(`ad_interfaces/PredictedObjectArray`, `states[]`) but iterates it for
validation only — the comment in `boxes_from_message()` said *"OccupancyGrid
has no time axis … keep only the current footprint here."* With Curve-Aware
Prediction v1 (PR #41) those predicted states now carry real curved per-state
yaw, so the swept region a moving object is about to occupy is available and
unused.

## What changed

`boxes_from_message()` gained an opt-in branch. When
`use_predicted_future_sweep: true` and `future_sweep_horizon_s > 0`:

1. keyframe 0 = the object's current footprint (unchanged path).
2. every predicted `state` whose `time_from_start ≤ future_sweep_horizon_s` is
   appended as a keyframe, transformed to the grid frame with **its own
   predicted orientation** (`state.pose.orientation`, the IMM's curved yaw).
   Horizons are strictly increasing, so the first out-of-window state ends the
   list.
3. the keyframe list is expanded by the **existing, unit-tested**
   `interpolate_dynamic_trajectory()` (via a thin `sweep_object_footprints()`
   wrapper that only picks the sample spacing). Spacing =
   `max(grid_resolution_m, 0.5 · min(length, width))` — half the smaller
   footprint edge guarantees consecutive rasterized boxes overlap (no grid-cell
   holes), never finer than one cell so the sample count stays bounded
   (`kMaximumSweepSamplesPerObject = 2048`, ~20× a realistic in-horizon
   prediction).
4. the resulting footprints are appended to the object list and rasterized by
   the **unchanged** `build_dynamic_grid()` — same SAT footprint test, same
   covariance inflation, same per-cell drivable-mask gate
   (`grid[i] = occupied_cost` only where `mask[i] == 0`), same 0/100 occupancy
   values.

Feature **off** (default) or `future_sweep_horizon_s: 0.0` → the object
contributes exactly its current footprint, byte-for-byte the prior behaviour.
A stationary object (identical keyframes) collapses to one footprint and never
sweeps a larger area. If the sweep can't be expanded into a bounded set
(degenerate spacing, sample-budget, interpolation overflow) that **one** object
falls back to its current footprint and the fallback is counted; the rest of
the frame is unaffected. Genuinely malformed message content
(non-finite pose/covariance/quaternion) still throws and still fail-safe-clears
the whole layer, exactly as before.

New `DYNAMIC_OGM_RUNTIME_SUMMARY` fields: `future_sweep=on|off`,
`future_sweep_objects` (objects with ≥1 in-horizon predicted state),
`future_sweep_skips` (per-object fallbacks), `max_future_sweep_horizon_ms`.

## Parameters (`config/occupancy_grid/dynamic.yaml`, both opt-in)

| param | default | meaning |
|---|---|---|
| `use_predicted_future_sweep` | `false` | off ⇒ unchanged current-footprint behaviour |
| `future_sweep_horizon_s` | `3.0` | forward horizon of the sweep; `0.0` also reproduces the legacy path; validated to `[0, 10]` |

Threaded through `dynamic_occupancy_grid.launch.py` →
`lidar_perception.launch.py` → `lidar_bag_replay.launch.py` →
`study_pipeline_rviz.launch.py` as `DeclareLaunchArgument` + dict forwarding at
each hop. Neither key is written to `dynamic.yaml` (see "Launch-plumbing bug
found" below), so the node default + launch dict are the only two sources. No
production launch default changed.

**Horizon default rationale.** `prediction.yaml`'s own horizon comment targets
"the 1.5 s DWA rollout, a complete 60 km/h emergency stop (~2.8 s), and the
planner's 0.5 s accepted prediction age." 3.0 s covers that. A full 6 s at
highway speed (~33 m/s ⇒ ~200 m) would blanket the 104 m-long grid, so it is
deliberately not the default. The A/B run below measured both 2.0 s and 3.0 s.

## Same-bag A/B (`morai_cam4_20260813_163222`, Pipeline A)

Method: one Pipeline-A run captured `/ad/perception/objects/predicted` +
`/ad/planning/drivable_mask` + `/tf[_static]` (520 / 516 prediction+mask
frames, ~120 s at `rate:=0.5`), then replayed verbatim into a standalone
`ad_dynamic_occupancy_grid` node three times (sweep off / on @ 2.0 s / on @
3.0 s) with `-p use_predicted_future_sweep:=…` — identical input every arm, so
this isolates **only** the OGM change. The predictions in this capture are
predominantly straight — `prediction_yaw_rate_source` does not reach the
prediction node through nested launches (a pre-existing PR #41 plumbing bug,
described under "Launch-plumbing bug found" below and **not** fixed in this
PR), so the IMM ran on the zero tracker yaw rate. The table below is therefore
a **straight-tail lower bound** on the sweep's effect; the curved-sweep
geometry is locked separately by unit tests C/D and by the curved-prediction
check below.

| metric | BEFORE (sweep off) | AFTER h=2.0 s | AFTER h=3.0 s |
|---|---|---|---|
| dynamic grids published | 512 | 510 | 512 |
| non-empty grids | 247 (48.2 %) | 256 (50.2 %) | 256 (50.0 %) |
| occupied cells / frame (non-empty): median | 1367 | 1546 | 1551 |
| occupied cells / frame (non-empty): p95 | 5639 | 5632 | 5658 |
| occupied cells / frame (non-empty): max | 11167 | 11801 | 12194 |
| occupied area m² / frame (non-empty median) | 13.67 | 15.46 | 15.51 |
| per-frame expansion ratio vs BEFORE: median / p95 | — | 1.05 / 1.79 | 1.05 / 1.85 |
| aggregate occupied-cell ratio vs BEFORE | — | 1.08 | 1.10 |
| objects with ≥1 in-horizon predicted state | 0 (off) | 3328 / 3328 (100 %) | 3319 / 3319 (100 %) |
| future-sweep whole-object fallbacks | — | 0 | 0 |
| max forward horizon rasterized | — | 2000 ms | 3000 ms |
| per-object oversized skips (PR #9 class) | 293 | 1593 | 2243 |
| frames with an oversized skip | 179 | 180 | 180 |
| step latency median / p95 / max (ms) | 0.55 / 1.45 / 3.04 | 1.05 / 3.29 / 7.46 | 1.26 / 3.99 / 8.51 |

**Reading.** Expansion is modest — aggregate **+10 %** occupied cells at
h=3.0 s, median frame **+5 %**, p95 essentially unchanged (5639 → 5658). The
sweep adds a thin tail per moving object, not a wall. Most future swept
cells fall **outside** the drivable mask and are dropped by the unchanged
road-gate, which is exactly the intended interaction (Section 7 of the task).
The large `max` per-frame ratio (11.75×) is a frame whose BEFORE occupancy was
a single tiny object — a large ratio on a near-zero base, not a flood: the
absolute max occupied cells/frame is 12194 of 208000 (**5.9 % of the grid**).

**Over-conservatism check.** No full-grid flood in any sampled frame.
Stationary objects do not expand (unit-tested; `interpolate` uses
`intervals = max(1, …)`). Fast objects are bounded to `horizon × speed`.
h=2.0 s vs h=3.0 s: near-identical on this bag (aggregate +8 % vs +10 %, p95 ratio 1.79 vs 1.85, median non-empty occupancy 1546 vs 1551 cells) — the extra second adds almost nothing because the road-gate clips the far tail, so 3.0 s is not over-conservative here and keeps margin for faster scenes.

**Runtime.** Step latency ~2.3× the median (0.55 → 1.26 ms) and ~2.7× the p95,
still ~1–4 ms — two orders of magnitude under the 0.5 s `prediction_timeout_sec`
and well under the ~230 ms inter-frame budget at 4.3 Hz. Publication rate
unchanged (512 grids either way). Throughput is not affected.

## Curved-track geometry check

**Unit level.** For a turning object the swept occupied cells follow the
**same** turn direction — the occupied-cell centroid is displaced laterally
toward the turn, and the opposite turn produces the mirror displacement,
symmetric magnitude / opposite sign (locked by
`test_dynamic_grid_builder.cpp::DynamicGridFutureSweep.{LeftBendingFutureSweepsCurvedRegion,RightBendingFutureIsTheMirrorOfLeftBending}`).
The sweep is piecewise-linear between predicted keyframes; at ~0.5 s keyframe
spacing the chord error against the arc is sub-cell.

**Real curved predictions.** `tracked_rec` (a captured
`/ad/perception/objects/tracked` stream) replayed through a standalone
`prediction.launch.py yaw_rate_source:=motion_history` produced 1 899
curved predicted object-frames (2 615 with non-zero fused yaw rate, IMM
coordinated-turn selected); fed into the sweep node: `future_sweep=on`,
**13 799 objects swept, 0 fallback skips**, step latency median 0.34 ms. The
curved trajectories are accepted and expanded without error. (Occupied-cell
counts from that particular run are not reported: the standalone replay had
no real localization TF, so the odom-frame predicted objects fell outside the
ego-relative grid extent — a harness limitation, not a code issue.)

**Full study pipeline.** `study_pipeline_rviz.launch.py` with
`use_predicted_future_sweep:=true`: 17 processes start, 0 crashes, the two new
params propagate through the whole nested launch chain
(`study_pipeline_rviz` → `lidar_bag_replay` → `lidar_perception` →
`dynamic_occupancy_grid`) — the node logs `future_sweep=on` with the requested
`future_sweep_horizon_s` reaching it (`max_future_sweep_horizon_ms` = 2000/3000
as set, `future_sweep_objects` in the thousands, `future_sweep_skips=0`),
**0 invalid grid cells**, maximum grid occupancy 5.7 % (no flood), Dynamic
Occupancy display present in the study RViz config. No screenshot tool in this
environment (documented precedent); verified from live message content. Curved
predictions through this same launch chain additionally require the PR #41
plumbing bug to be fixed (below) — out of scope here — so the curved-sweep
evidence in this PR is the unit tests and the standalone `pred_mh` run above.

## Launch-plumbing bug found — fixed separately (`fix/prediction-yaw-rate-launch-propagation`)

While wiring the two new params it surfaced that `prediction_yaw_rate_source`
(PR #41) was **silently inert through every nested launch** (`lidar_perception`
/ `lidar_bag_replay` / `study_pipeline_rviz`). Root cause: `prediction.launch.py`
passed `parameters=[prediction.yaml, {override}]` as two `--params-file`
arguments, and once `launch_ros`'s `SetParameter(use_sim_time=…)` (in
`lidar_bag_replay.launch.py`'s scoped group) emitted a leading `-p` argument,
`rcl` resolved `yaw_rate_source` against the node-name-scoped YAML section
(`ad_autoware_prediction:`) rather than the later wildcard (`/**:`) override
dict, keeping the file's `tracker`. Repro:
`study_pipeline_rviz.launch.py prediction_yaw_rate_source:=motion_history` left
`initial_twist.twist.angular.z == 0` on every predicted object; a direct
`prediction.launch.py yaw_rate_source:=motion_history` worked.

It was **out of scope for the future-sweep PR** (no prediction-path change), so
the two new sweep params sidestepped it structurally (comment-only in
`dynamic.yaml`). The bug itself is fixed in the follow-up branch
`fix/prediction-yaw-rate-launch-propagation`: `prediction.launch.py` now reads
`prediction.yaml` in Python and overlays the override before launch (one param
source, no two-file race — the same pattern `ad_planner`'s
`road_corridor_mask.launch.py` uses), with `test_prediction_launch.py` locking
the runtime parameter under the exact `SetParameter` + scoped-group trigger.

## Known limitations

- **Prediction-source dependency.** The curved per-state yaw comes from the
  IMM adapter (`map_imm_prediction`, sets `state.pose.orientation` from the
  predicted `yaw_rad`). If the node is ever switched to the non-IMM
  `map_prediction` path (which sets every state to the *current* orientation),
  the sweep degrades to a straight rectangle. It is deterministic geometry
  driven by whatever the predictor emits — motion-history / IMM based, **not**
  lane-aware, no maneuver-intent model, no future-object ground truth.
- **Not probabilistic.** Swept cells are hard `occupied_cost` (100), identical
  to the current footprint. This is geometry integration, not an occupancy
  *probability* field.
- **Bounded horizon.** Anything the object does after `future_sweep_horizon_s`
  is not represented in the dynamic layer (the planner still consumes the full
  time-indexed `PredictedObjectArray` directly).
- **`maximum_cells_per_object` is enforced per footprint, not per object.** A
  swept object becomes many footprints, so a high-covariance AB3DMOT track
  (already skipped as oversized in the current-footprint path, PR #9) is now
  skipped once per swept sample — the A/B `oversized_objects_skipped` counter
  rises from 293 to 2243 while `frames_with_oversized_skip` is unchanged
  (179 → 180). Effect: that one object's future tail has a hole; every other
  object still rasterizes and no frame is aborted. A per-object budget for
  swept groups is a follow-up, not done here.
- **Road-gated curved A/B not measured.** A second Pipeline-A capture with
  motion-history yaw rate forced on (via a direct `prediction.launch.py`, not
  the study chain) could not be replayed standalone: its recorded localization
  `/tf` is non-monotonic and continuously clears the node's TF buffer
  ("Detected jump back in time"), so every grid came back empty. The curved
  A/B would need an in-process pipeline run once the plumbing bug above is
  fixed.
- Single scene / single bag; message-count, area, throughput and
  TF-classification counters only — **no occupancy-accuracy or planner-safety
  claim.**
