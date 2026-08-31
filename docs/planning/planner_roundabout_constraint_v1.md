# Planner Roundabout Response Constraint v1

This is the first roundabout change that may alter production planner longitudinal
behaviour. It connects the existing `RoundaboutGapResponse` advisory to the one
place the planner already accepts an external per-update speed limit: the optional
`target_speed_mps` argument of `PathTrackingController::update()` — the same
integration point the cut-in constraint (PR #16) uses.

It adds **no** second longitudinal controller, publishes **no** additional
`CtrlCmd`, and touches **no** steering, path, lane, BehaviorTree, tracker,
prediction, risk-estimation, or roundabout-response-policy code. It is **opt-in
and off by default**; with `enable_roundabout_response_constraint: false` the
planner creates no subscription, no observability publisher, reads nothing, and is
behaviourally identical to a build without this feature.

```text
/ad/planning/roundabout_gap_response   (RoundaboutGapResponse, from ad_roundabout_gap_response)
              |
              v
   AdPlannerNode  (enable_roundabout_response_constraint = true)
    on_roundabout_response()  ->  latest message + steady receipt time
              |
   run_path_tracking():
     cut_in_cap    = clamp(cut_in_response_speed_limit(...))
     roundabout_cap= clamp(roundabout_response_speed_limit(...))
     combined      = combine_speed_limits(cut_in_cap, roundabout_cap)   # min, order-independent
              |
   path_tracking_->update(pose, speed, dt, 0, gear, combined)   <-- existing 6th arg
              |
              v
   one /ad/control/command   (unchanged single publisher)
```

## Architecture: the response is the only consumer contract

```text
Roundabout Gap Risk  ->  Roundabout Gap Response  ->  planner longitudinal constraint
```

The planner consumes `RoundaboutGapResponse` **only**. It never reads
`RoundaboutGapRisk`, and it never touches the legacy first-interval
`occupancy_overlap` / `temporal_gap_s` fields — the all-interval safety summary
stays owned by the risk/response chain.

## Where the constraint enters

`run_path_tracking()` (the `FollowGlobalPath` BehaviorTree leaf) already calls
`PathTrackingController::update(..., std::optional<double> target_speed_mps)`.
Stanley / profile-Stanley resolves
`selected_target_speed = target_speed_mps.value_or(config_.target_speed_mps)`,
then applies its own `min` against the launch ramp and route speed profile
(`stanley.cpp`). `selected_target_speed` feeds only the longitudinal launch-ramp
and PID path (`stanley.cpp` lines ~292-297); the steering computation
(cross-track + heading, lines ~266-286) never reads it. Passing a smaller value
can therefore only lower the governed target, never raise it, and cannot affect
steering (`test_stanley.cpp::UsesPerUpdateTargetSpeedForSlowdown` locks the
longitudinal effect; `test_planner_roundabout_constraint.py` asserts the live
steering field is byte-identical across RELEASE / YIELD / HOLD).

This is the narrowest external-cap point that stays *inside* the existing
longitudinal path — the constraint enters before command generation, not after
it, so HOLD's `0.0` flows through the normal brake PID.

### Scope: path-tracking only (same limitation as the cut-in constraint)

The `PerceptionMission` branch (`run_local_motion()`, DWA / Frenet / MPPI) is
**not** constrained in v1, for the reasons already documented for the cut-in
constraint: its speed budget is `VehicleConstraints.maximum_speed_mps`, validated
as strictly `> 0`, so `HOLD = 0.0` cannot be expressed there without breaking the
field invariant, and lowering the dynamic-window speed cap could perturb DWA's
`(v, ω)` trajectory selection (a lateral effect this task forbids). Local motion
also runs its own corridor / occupancy / prediction longitudinal response. No
attempt is made to change `VehicleConstraints.maximum_speed_mps` here.

## Constraint semantics

Pure function `roundabout_response_speed_limit(RoundaboutResponseConstraintInput)`
→ `std::optional<double>` (`roundabout_speed_constraint.hpp`). It owns no
thresholds, no deceleration constants, and no roundabout equations.

| response | result | planner effect |
| --- | --- | --- |
| not received / stale / `active == false` / `ACTION_RELEASE` / unknown action | `nullopt` | none — planner keeps its own desired speed |
| `ACTION_HOLD` | `0.0` | `final = min(existing target, 0.0) = 0.0` (prepare to hold, through the normal PID) |
| `ACTION_YIELD`, consistent facts | comfortable-stop cap (below) | `final = min(existing target, cap)` |
| `ACTION_YIELD` with malformed facts (non-finite, `available_distance_m <= 0`, `comfortable_stop_distance_m <= 0`, negative ego speed) | `nullopt` | none — a bad frame is never a fabricated cap and never a speed increase |

Guarantees: the return value is never negative and never NaN/Inf; the constraint
combines with every other planner limit (route / curvature / launch ramp /
cut-in) by `min`, so it is order-independent and can only ever lower the target;
it never increases speed.

### YIELD speed-cap derivation and provenance

```text
cap = ego_speed_mps * sqrt(available_distance_m / comfortable_stop_distance_m)
```

The response publishes `comfortable_stop_distance_m = ego_speed_mps^2 /
(2 * a_comfortable)`, where `a_comfortable` is Roundabout Gap Response's own
canonical comfortable deceleration (`1.8 m/s^2`, owned by
`roundabout_gap_response.yaml`). Substituting:

```text
ego_speed_mps * sqrt(available_distance_m / comfortable_stop_distance_m)
  == sqrt(2 * a_comfortable * available_distance_m)
```

(the `ego_speed_mps` terms cancel exactly). This is the maximum speed from which
the ego can still stop comfortably over the remaining pre-entry distance under the
**same** constant-deceleration envelope the response already uses. As the ego
approaches, `available_distance_m` shrinks and the cap shrinks with it; once the
comfortable margin is consumed the response classifies HOLD and the cap becomes
`0`.

**Provenance:** the cap is derived entirely from `RoundaboutGapResponse`'s own
published policy facts (`ego_speed_mps`, `available_distance_m`,
`comfortable_stop_distance_m`). **No deceleration constant, standoff, or gap
threshold is duplicated in the planner.** There is no new
`roundabout_yield_speed` parameter.

### Freshness

`on_roundabout_response()` stamps each message with `steady_now()`
(`RCL_STEADY_TIME`). The constraint is `fresh` when
`context_.steady_time_s - receipt <= roundabout_response_max_age_s` (default
`0.5 s`, mirroring the response node's own `maximum_input_age_s`).
`context_.steady_time_s` is set at the top of `tick()`, so `run_path_tracking()`
reads a same-tick value. **The response header stamp is never consulted**, so a
future, backward, or duplicate stamp can neither extend freshness nor latch any
state. A missing or stale response is treated exactly as `ACTION_RELEASE` (no
constraint).

### Inactive / RELEASE / stale mapping

- **Inactive** (`active == false`, ego inside or past the conflict) → `nullopt`.
  The planner never HOLDs after entry.
- **RELEASE** → `nullopt`. RELEASE removes *only* the roundabout constraint. It
  never raises speed and never lifts a lower cut-in cap or any other planner
  limit.
- **Stale YIELD / HOLD / RELEASE** → after `roundabout_response_max_age_s` the
  roundabout source contributes `nullopt`. HOLD never latches; a stale RELEASE is
  never retained as "roundabout is clear" state. If a cut-in constraint is still
  active it continues to apply.

## External constraint composition

`combine_speed_limits(a, b)` (`external_speed_limit.hpp`) returns the smallest
present finite value, or `nullopt` when both are absent. It is symmetric:
`combine_speed_limits(a, b) == combine_speed_limits(b, a)` for every input, so the
planner never depends on the order it consults its constraint sources.

Each source is first clamped against the path-tracking nominal cruise target (a
value that does not lower it becomes `nullopt`), then the two clamped optionals
are combined:

```text
final_target_speed = min( nominal_planner_target,
                          cut_in_limit_if_active,
                          roundabout_limit_if_active )
```

`test_external_speed_limit.cpp` verifies the Phase-13 matrix directly:

| cut-in | roundabout | nominal | final |
| --- | --- | --- | --- |
| SLOWDOWN 6 | RELEASE | 10 | 6 |
| SLOWDOWN 6 | YIELD (cap > 6) | 10 | 6 |
| NONE | YIELD (cap) | 16.25 | cap |
| SLOWDOWN 4 | HOLD | 10 | 0 |
| HOLD | RELEASE | 10 | 0 |
| HOLD | HOLD | 12 | 0 |

No source can cancel another more restrictive source.

## Configuration (`config/planner.yaml`)

| key | default | meaning |
| --- | --- | --- |
| `enable_roundabout_response_constraint` | `false` | master opt-in; off ⇒ no subscription, no publisher, no behaviour change |
| `roundabout_response_max_age_s` | `0.5` | consumer-side freshness timeout (mirrors the response node's `maximum_input_age_s`) |
| `topics.roundabout_gap_response` | `/ad/planning/roundabout_gap_response` | input (only subscribed when enabled) |
| `topics.roundabout_speed_limit` | `/ad/planner/roundabout_speed_limit` | `std_msgs/Float32` observability: active cap, or `-1.0` when no constraint (only published when enabled) |

No comfortable / maximum deceleration, standoff, or gap threshold is repeated
here; those stay owned by `ad_roundabout_gap_response`.

`planner.launch.py enable_roundabout_response_constraint:=true` (or
`roundabout_gap_response:=true`) sets the parameter and also starts
`ad_roundabout_gap_response` and `ad_roundabout_gap_risk`, since the constraint is
useless without the response it consumes.

## Single control-command publisher / no actuator path

`AdPlannerNode::publish_command()` → `PlannerRosInterfaces::publish_command()` on
`/ad/control/command` remains the sole intended production `CtrlCmd` publisher.
This change adds one `std_msgs/Float32` diagnostic publisher (enabled builds
only) and one `RoundaboutGapResponse` subscription; it publishes no throttle,
brake, steering, gear, `cmd_vel`, or DBW message.
`test_planner_roundabout_constraint.py` asserts
`count_publishers("/ad/control/command") == 1` with the feature enabled.

## Verification

* `test_roundabout_speed_constraint.cpp` — 22 pure-logic cases: RELEASE / YIELD /
  HOLD, stale, missing, inactive, unknown / negative action, YIELD cap equals
  `sqrt(2 a d)`, cap independent of ego speed for the same envelope, cap shrinks
  and is monotonic in available distance, HOLD never weaker than YIELD, malformed
  YIELD facts (zero / negative / NaN / Inf distances, negative ego speed) →
  `nullopt`, non-negativity sweep, determinism.
* `test_external_speed_limit.cpp` — 14 cases: both-absent, single pass-through,
  most-restrictive-wins, order independence, zero honoured, non-finite ignored,
  plus the full cut-in × roundabout Phase-13 composition matrix and per-state
  order-independence.
* `test_planner_roundabout_constraint.py` — live `ad_planner` in the
  `FollowGlobalPath` branch (`perception.enabled:=false`), straight fixture
  route, ego reporting `8.0 m/s`:
  * no response / inactive → `/ad/planner/roundabout_speed_limit == -1.0`,
    governed target `16.25` (cruise);
  * RELEASE → limit `-1.0`, governed target `16.25` (byte-identical no-op);
  * YIELD (available 34 m, comfortable stop 17.78 m) → limit `11.063`, governed
    target `11.063` = `sqrt(2 * 1.8 * 34)`, still above the 8 m/s ego speed;
  * HOLD (available 8 m) → limit `0.0`, governed target `0.0`, command flips to
    brake;
  * stop publishing > `roundabout_response_max_age_s` → limit `-1.0`, cruise
    target returns;
  * no state raised the target above nominal; steering field byte-identical
    across all states; exactly one `/ad/control/command` publisher.
* `test_planner_launch.py` — `enable_roundabout_response_constraint` /
  `roundabout_gap_response` are declared generic arguments defaulting to config
  (`false`); enabling the constraint starts the response and (forced) risk nodes;
  the default planner node carries no override.
* Full `ad_planner` **sequential** ctest minus known-flaky launch-runtime
  tests: **40 / 40 pass** (33 gtest + 5 pytest + the two constraint launch
  tests), including the pre-existing cut-in constraint tests.
  `test_frenet_runtime_contract` passes in isolation (it only *errored* under
  the parallel `colcon test` run — a resource-contention artifact). The two
  constraint launch-runtime tests pass in isolation.
* Pre-existing host-environment failures, not caused by this change:
  * `test_mppi_nav2_launch` — host lacks `nav2_common` / `nav2_controller`
    (documented in the PR #17/#18/#19 STATUS entries).
  * `test_cut_in_response_runtime` — verified to fail identically on a
    changes-stashed clean `5c64c77` tree: `ad_dynamic_object_risk_node` (an
    `ad_lidar_perception` node, not rebuilt and not modified here) never
    publishes frame 0 and then hangs on shutdown; no `ad_planner` node
    participates in the failing assertion.
* Isolated `colcon build --packages-select ad_interfaces ad_planner
  --symlink-install` clean.

## Deterministic end-to-end sequence (composed)

The three-node chain `RoundaboutGapRisk -> RoundaboutGapResponse -> planner` was
not run in one process for this task. The planner's only input is the
`RoundaboutGapResponse` message, so the YIELD → HOLD planner-side sequence
follows by construction from two independently validated pieces:

* PR #19's canonical K-City replay: the response's validated approach sequence is
  YIELD near 40 m (`available ≈ 34 m`, comfortable stop `≈ 17.78 m`) then HOLD
  near 20 m (`available ≈ 14 m`).
* This task's `test_planner_roundabout_constraint.py`: `available = 34 m` maps to
  a governed target of `11.063 m/s` (`= sqrt(2 * 1.8 * 34)`); `available = 8 m`
  (HOLD) maps to `0.0`; steering is byte-identical across all states.

Composing the two: a real roundabout approach drives the planner's governed
longitudinal target smoothly down the comfortable-stop envelope during YIELD and
to `0.0` at HOLD, with no lateral change.

## Not done in v1

No steering / lateral / path / lane / BehaviorTree change. No local-motion
(DWA / Frenet / MPPI) constraint. No change to the RELEASE / YIELD / HOLD
classification rules or any roundabout policy parameter. No new
`RoundaboutGapResponse` message field. No hysteresis / release-confirmation. No
tracker, prediction, or risk-estimation change. No direct actuator path.

## Known limitations

- `FollowGlobalPath` (Stanley / profile-Stanley) only; `PerceptionMission` is
  unconstrained, as with the cut-in constraint.
- The YIELD → HOLD transition inherits the response's own discontinuity: at the
  YIELD/HOLD boundary the cap is `≈ ego_speed`, then drops to `0`. This reflects
  the response policy's split, not a planner artifact.
- Stopped ego cannot earn RELEASE (upstream ETA validity); the planner maps that
  frame's HOLD to a `0.0` cap, which is the conservative correct behaviour.
