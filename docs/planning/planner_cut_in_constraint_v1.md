# Planner Cut-in Speed Constraint v1

This is the first change that lets a cut-in actually alter production planner
longitudinal behaviour. It connects the existing `CutInResponse` request to the
one place the planner already accepts an external per-update speed limit: the
optional `target_speed_mps` argument of `PathTrackingController::update()`.

It adds **no** second longitudinal controller, publishes **no** additional
`CtrlCmd`, and touches **no** steering, path, lane, BehaviorTree, tracker,
prediction, occupancy, cut-in detection, or cut-in policy code. It is **opt-in
and off by default**; with `enable_cut_in_response_constraint: false` the planner
creates no subscription, reads nothing, and is behaviourally identical to a
build without this feature.

```text
/ad/planning/cut_in_response   (CutInResponse, from ad_cut_in_response)
              |
              v
   AdPlannerNode  (enable_cut_in_response_constraint = true)
    on_cut_in_response()  ->  latest message + steady receipt time
              |
   run_path_tracking():  cap = clamp(cut_in_response_speed_limit(...))
              |
   path_tracking_->update(pose, speed, dt, 0, gear, cap)   <-- existing arg
              |
              v
   one /ad/control/command   (unchanged single publisher)
```

## Where the constraint enters

`run_path_tracking()` (the `FollowGlobalPath` BehaviorTree leaf) already calls
`PathTrackingController::update(..., std::optional<double> target_speed_mps)`.
The Stanley / profile-Stanley backend resolves
`selected_target_speed = target_speed_mps.value_or(config_.target_speed_mps)`,
then applies its own `min` against the launch ramp and the route speed profile
(`stanley.cpp`). Passing a smaller `selected_target_speed` can therefore only
lower the governed target, never raise it, and never affects steering (proven by
`test_stanley.cpp::UsesPerUpdateTargetSpeedForSlowdown`). This is the narrowest
external-cap point that stays *inside* the existing longitudinal path — the
constraint enters before command generation, not after it.

The cap is `std::min(nominal_cruise_target, limit)`, and it is only passed when
it strictly lowers the target; otherwise `std::nullopt` is passed, which is a
byte-identical no-op. `nominal_cruise_target` is read once at construction from
the backend's own `*.target_speed_mps` parameter purely as the clamp ceiling.

### Scope: path-tracking only (Phase 9 blocker for local motion)

The `PerceptionMission` branch (`run_local_motion()`, DWA/Frenet/MPPI) is **not**
constrained in v1. Its speed budget is `VehicleConstraints.maximum_speed_mps`,
which the node validates with `positive_finite_parameter` (strictly `> 0`).
`ACTION_HOLD` maps to `0.0`, which cannot be expressed in that field without
breaking its established invariant, and lowering the dynamic-window speed cap
could also perturb DWA's `(v, ω)` trajectory selection — a lateral/path effect
this task forbids. The local-motion mission additionally runs its own
corridor-, occupancy-, and prediction-aware longitudinal response for objects
in or entering the corridor. Integrating the cut-in cap there needs a separate
representation and is deferred.

## Constraint semantics

Pure function `cut_in_response_speed_limit(CutInResponseConstraintInput)` →
`std::optional<double>` (`cut_in_speed_constraint.hpp`). It owns no thresholds,
no deceleration constants, and no cut-in equations.

| response | result | planner effect |
| --- | --- | --- |
| not received / stale / `active == false` / `ACTION_NONE` | `nullopt` | none — planner keeps its own desired speed |
| `ACTION_SLOWDOWN`, `requested_max_speed_valid`, finite `> 0` | `requested_max_speed_mps` | `final = min(existing target, requested_max_speed_mps)` |
| `ACTION_HOLD`, `requested_max_speed_valid`, value `== 0.0` | `0.0` | `final = min(existing target, 0.0) = 0.0` (prepare to hold, through the normal PID) |
| malformed / contradictory (invalid flag, non-finite, `SLOWDOWN` ≤ 0, `HOLD` ≠ 0, unknown action) | `nullopt` | none — a bad message is always a no-op, never a spurious brake and never a speed increase |

Guarantees: the return value is never negative and never NaN/Inf; the constraint
combines with every existing planner limit (route / curvature / launch ramp /
behaviour) by `min`, so it is order-independent and can only ever lower the
target; it never increases speed.

### Freshness

`on_cut_in_response()` stamps each message with `steady_now()`
(`RCL_STEADY_TIME`). The constraint is `fresh` when
`context_.steady_time_s - receipt <= cut_in_response_max_age_s` (default `0.5 s`,
the same order as the response publication and planner control periods).
`context_.steady_time_s` is set at the top of `tick()` before the BehaviorTree
runs, so `run_path_tracking()` reads a same-tick value. A missing or stale
response is treated exactly as `ACTION_NONE`.

## Configuration (`config/planner.yaml`)

| key | default | meaning |
| --- | --- | --- |
| `enable_cut_in_response_constraint` | `false` | master opt-in; off ⇒ no subscription, no behaviour change |
| `cut_in_response_max_age_s` | `0.5` | consumer-side freshness timeout |
| `topics.cut_in_response` | `/ad/planning/cut_in_response` | input (only subscribed when enabled) |
| `topics.cut_in_speed_limit` | `/ad/planner/cut_in_speed_limit` | `std_msgs/Float32` observability: active cap, or `-1.0` when no constraint (only published when enabled) |

No comfortable / maximum deceleration or cut-in threshold is repeated here; those
stay owned by `ad_cut_in_response`.

`planner.launch.py enable_cut_in_response_constraint:=true` sets the parameter
and also starts `ad_cut_in_response` (and `ad_cut_in_risk`), since the constraint
is useless without the response it consumes. The default is unset ⇒ config value
⇒ `false`.

## Single control-command publisher

`AdPlannerNode::publish_command()` → `PlannerRosInterfaces::publish_command()` on
`/ad/control/command` remains the sole intended production `CtrlCmd` publisher.
This change adds one `std_msgs/Float32` diagnostic publisher (enabled builds
only) and one `CutInResponse` subscription; it publishes no throttle, brake,
steering, gear, `cmd_vel`, or DBW message. `test_planner_cut_in_constraint.py`
asserts `count_publishers("/ad/control/command") == 1` with the feature enabled.

## Verification

* `test_cut_in_speed_constraint.cpp` — 22 pure-logic cases: NONE / SLOWDOWN /
  HOLD, stale, missing, inactive, invalid-flag, non-finite, `SLOWDOWN` ≤ 0,
  `HOLD` ≠ 0, unknown / negative action, non-negativity property sweep,
  monotonicity in requested speed, limit never exceeds request.
* `test_planner_cut_in_constraint.py` — live `ad_planner` in the
  `FollowGlobalPath` branch (`perception.enabled:=false`), straight fixture
  route, ego reporting `8.0 m/s`:
  * no response / `ACTION_NONE` → `/ad/planner/cut_in_speed_limit == -1.0`,
    forward throttle (`accel == 1.0`);
  * `ACTION_SLOWDOWN 3.0` → cap `3.0`, command flips to brake;
  * `ACTION_HOLD` → cap `0.0`, braking at least as strong;
  * stop publishing > `cut_in_response_max_age_s` → cap `-1.0`, baseline
    throttle returns;
  * exactly one `/ad/control/command` publisher throughout.
* `test_planner_launch.py` — `enable_cut_in_response_constraint` is a declared
  generic argument defaulting to config (`false`).
* Full `ad_planner` suite: 39 / 40 pass; the only failure is the pre-existing
  `test_mppi_nav2_launch` (host lacks `nav2_common` / `nav2_controller`),
  unrelated to this change.

## Not done in v1

No steering / lateral / path / lane / BehaviorTree change. No local-motion
(DWA/Frenet/MPPI) constraint. No emergency braking, roundabout gap acceptance,
highway merge gap, or generic TTC emergency policy. No tracker, prediction,
occupancy, or cut-in detection change. No direct actuator path.
