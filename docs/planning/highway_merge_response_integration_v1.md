# Highway Merge Response Integration v1

This connects the advisory `HighwayMergeGapResponse` to the existing planner /
highway-mission boundary. It has two responsibilities and does **neither** of
them by commanding steering or a lane change:

1. **Longitudinal constraint** – `WAIT` / `HOLD` may lower the path-tracking
   target through the **existing** external speed-constraint path (the same 6th
   `target_speed_mps` argument of `PathTrackingController::update()` the cut-in
   (PR #16) and roundabout (PR #20) constraints already use).
2. **Merge authorization** – a fresh, active `ACTION_MERGE_READY` exposes a
   revocable `merge_authorized = true` fact to the mission layer
   (`PlannerContext::highway_merge_authorized`, the read-only `HighwayMergeReady`
   BehaviorTree condition, and a `/ad/planner/highway_merge_authorized`
   diagnostic). It does **not** initiate a merge.

It adds **no** second longitudinal controller, publishes **no** additional
`CtrlCmd`, and touches **no** steering, path, lane, route, trajectory, tracker,
prediction, risk-estimation, or merge-response-policy code. It is **opt-in and
off by default**; with `enable_highway_merge_response_integration: false` the
planner creates no subscription, no observability publisher, reads nothing,
never authorizes a merge, and is behaviourally identical to a build without this
feature.

```text
/ad/planning/highway_merge_gap_response   (HighwayMergeGapResponse, from ad_highway_merge_gap_response)
              |
              v
   AdPlannerNode  (enable_highway_merge_response_integration = true)
    on_highway_merge_response()  ->  latest message + steady receipt time
              |
   tick():   context_.highway_merge_authorized = compute_highway_merge_authorized()   # before the BT runs
              |
   run_path_tracking():
     cut_in_cap        = clamp(cut_in_response_speed_limit(...))
     roundabout_cap    = clamp(roundabout_response_speed_limit(...))
     highway_merge_cap = clamp(highway_merge_response_speed_limit(...))
     combined          = combine_speed_limits({cut_in_cap, roundabout_cap, highway_merge_cap})   # min, order-independent
              |
   path_tracking_->update(pose, speed, dt, 0, gear, combined)   <-- existing 6th arg
              |
              v
   one /ad/control/command   (unchanged single publisher)
```

## Architecture: the response is the only consumer contract

```text
Highway Merge Gap Risk  ->  Highway Merge Gap Response  ->  this integration
```

The integration consumes `HighwayMergeGapResponse` **only**. It never reads
`HighwayMergeGapRisk`, `DynamicObjectRisk`, or raw tracker output, and it never
re-derives the merge policy: the pair `(active, action)` is the whole input.
`reason` is diagnostic and is never branched on (an inactive frame carries
`REASON_NONE = 0`, the same enum-0 trap as `ACTION_MERGE_READY = 0`).

## Existing highway mission / merge boundary

**There is no executable highway-merge or lane-change mission path in the
codebase.** A full-repository audit found:

| area | finding |
| --- | --- |
| BehaviorTree (`behavior_trees/ad_planner.xml`) | `InputsReady` → (`CollisionRecovery` \| `TrafficStop` \| `PerceptionMission` \| `FollowGlobalPath`) → `FailSafeBrake`. No merge / lane-change / mission-readiness branch, condition, or placeholder. |
| `PlannerContext` / `PlannerRuntimeState` | no mission state, no merge flag, no blackboard of mission facts. |
| `frenet_lattice.cpp` | a `lane_change_weight` **cost term** for local trajectory scoring (`lane_index != primary_lane_index`). It is a local-motion cost, not a mission transition, and is not driven by any merge decision. |
| `dwa.cpp` / `mppi` | no merge concept. |
| route / global path | single fixed corridor; no route-change primitive. |

So this is Phase 21 of the task: **no mission seam is fabricated.** The
integration adds only (a) fresh merge-authorization state in a
planner/BT-accessible location, (b) a narrow read-only BT condition, and (c) the
longitudinal `WAIT` / `HOLD` constraint. **No production BehaviorTree transition
consumes `merge_authorized` yet.**

## Selected merge-authorization seam

`PlannerContext::highway_merge_authorized` (a plain `bool`, default `false`),
recomputed by `AdPlannerNode::tick()` at the top of every tick — right after
`context_.steady_time_s` is set and before `supervisor_->tick()` runs — so a
future mission condition reads this tick's value.

Exposed three ways:

* `PlannerContext::highway_merge_authorized` — the primary seam, directly
  readable by any BehaviorTree node through `BtNodeEnvironment::context`.
* `HighwayMergeReady` — a new `BT::SimpleCondition`, registered in
  `register_ad_bt_nodes` and listed in `ad_bt_node_ids()`. `SUCCESS` iff
  `context.highway_merge_authorized`, `FAILURE` otherwise. It publishes no
  control, changes no path/route/steering, and never internally latches
  `SUCCESS`. **It is registered but not referenced by the production tree** —
  the exact-XML assertion in `test_behavior_tree.cpp::HasExpectedPriorityAndCruises`
  is the standing proof that the tree is unchanged.
* `/ad/planner/highway_merge_authorized` (`std_msgs/Bool`) — a diagnostic mirror,
  published only when the integration is enabled. It is **not** a lane-change
  command and is not sufficient on its own to count as mission integration.

**No BT transition was changed. No lane-change planner, route change, or
steering command was added.**

## Enable flag, topics, freshness

| key (`config/planner.yaml`) | default | meaning |
| --- | --- | --- |
| `enable_highway_merge_response_integration` | `false` | master opt-in; off ⇒ no subscription, no publisher, no authorization, no behaviour change |
| `highway_merge_response_max_age_s` | `0.5` | consumer-side freshness timeout (mirrors the response node's `maximum_input_age_s`) |
| `expected_highway_merge_zone` | `kcity_highway_onramp` | required to match the response's `merge_zone_id`; a response for any other zone is ignored wholesale. An **empty** value accepts any zone. |
| `topics.highway_merge_gap_response` | `/ad/planning/highway_merge_gap_response` | input (only subscribed when enabled) |
| `topics.highway_merge_speed_limit` | `/ad/planner/highway_merge_speed_limit` | `std_msgs/Float32` observability: active cap, or `-1.0` when no constraint (only published when enabled) |
| `topics.highway_merge_authorized` | `/ad/planner/highway_merge_authorized` | `std_msgs/Bool` observability: the merge-authorization fact (only published when enabled) |

Freshness is measured from `steady_now()` at receipt vs `context_.steady_time_s`
(set at the top of `tick()`). **The response header stamp is never consulted**,
so a future, backward, or duplicate stamp can neither extend freshness nor latch
any state. `planner.launch.py enable_highway_merge_response_integration:=true`
also starts `ad_highway_merge_gap_response` and (forced) `ad_highway_merge_gap_risk`,
since the integration is useless without the response it consumes.

As with the cut-in and roundabout constraints, `ad_highway_merge_gap_response`
is auto-started only via the launch arguments (`highway_merge_gap_response:=true`
or `enable_highway_merge_response_integration:=true`); enabling the integration
by editing `planner.yaml` alone does **not** auto-start the response node (a
deployment that does that must launch the advisory chain itself).

No merge policy threshold (`minimum_front_time_headway_s`,
`minimum_rear_time_headway_s`, `minimum_rear_closing_time_s`,
`minimum_predicted_route_gap_m`, `merge_standoff_m`,
`comfortable_deceleration_mps2`) is duplicated here — all stay owned by
`ad_highway_merge_gap_response`.

## `merge_authorized` exact semantics

`highway_merge_response_merge_authorized(input)` returns `true` **only** when
**all** of:

* the integration is enabled;
* a `HighwayMergeGapResponse` has been received;
* it is fresh (steady receipt within `highway_merge_response_max_age_s`);
* `response.active == true`;
* `response.merge_zone_id == expected_highway_merge_zone` (or the expected zone
  is empty);
* `response.action == ACTION_MERGE_READY`.

Otherwise `merge_authorized = false`.

**`action == ACTION_MERGE_READY` alone is never sufficient.** `ACTION_MERGE_READY
= 0` is the enum default, and an **inactive** response also carries it. The
`active == true` check is what makes the fact meaningful. Locked by
`test_highway_merge_speed_constraint.cpp::InactiveMergeReadyEnumDoesNotAuthorize`
and the live `test_planner_highway_merge_constraint.py` inactive-response step.

## No `MERGE_READY` latch

`context_.highway_merge_authorized` is recomputed from the freshest response
every tick and is never stored as a persistent `true`. A fresh `MERGE_READY`
followed by `WAIT`, `HOLD`, a stale frame, a missing frame, an inactive frame,
or a wrong-zone frame makes `merge_authorized` immediately `false`. There is no
confirmation / hysteresis logic in v1.

## Longitudinal state mapping

`highway_merge_response_speed_limit(input)` → `std::optional<double>`
(`highway_merge_speed_constraint.hpp`, in `ad_planner_core`, no ROS-message
dependency):

| response | result | planner effect |
| --- | --- | --- |
| not received / stale / `active == false` / wrong zone / `ACTION_MERGE_READY` / unknown action | `nullopt` | none — planner keeps its own desired speed |
| `ACTION_HOLD` | `0.0` | `final = min(existing target, 0.0) = 0.0` through the normal brake PID |
| `ACTION_WAIT`, consistent facts | comfortable-stop cap (below) | `final = min(existing target, cap)` |
| `ACTION_WAIT` with malformed facts (non-finite, `available_distance_m <= 0`, `comfortable_stop_distance_m <= 0`, negative ego speed) | `nullopt` | none — a bad frame is never a fabricated cap and never a speed increase |

`ACTION_MERGE_READY` imposes **no** highway-merge-specific cap (`nullopt`). This
does not mean "increase speed" — other constraints remain authoritative; it only
means the merge advisory itself does not slow the ego.

The integration publishes no brake / throttle / steering / gear / `cmd_vel` /
DBW message. `HOLD`'s `0.0` flows through the existing `FollowGlobalPath`
controller, which decides how to reach zero.

### WAIT speed-cap derivation and provenance

```text
cap = ego_speed_mps * sqrt(available_distance_m / comfortable_stop_distance_m)
```

The response publishes `comfortable_stop_distance_m = ego_speed_mps^2 /
(2 * a_comfortable)` with `a_comfortable = 1.8 m/s^2` owned by
`highway_merge_gap_response.yaml`. Substituting:

```text
ego_speed_mps * sqrt(available_distance_m / comfortable_stop_distance_m)
  == sqrt(2 * a_comfortable * available_distance_m)
```

(the `ego_speed_mps` terms cancel exactly). This is the maximum speed from which
the ego can still stop comfortably over the remaining distance to the decision
boundary under the **same** constant-deceleration envelope the response already
uses. **No deceleration constant, standoff, or gap threshold is duplicated in
the planner, and there is no new `highway_merge_wait_speed` parameter.** This is
identical to the roundabout constraint's derivation.

The cap only lowers the target inside roughly `2·1.8·available < nominal²`, i.e.
`available < ~73 m` for a 16.25 m/s cruise target; farther out the cap is above
nominal and the constraint is a byte-identical no-op (never a speed increase).

## External constraint composition

`combine_speed_limits` gains an `std::initializer_list` fold overload alongside
the unchanged binary form. `min` is associative and the binary combiner is
symmetric, so:

```text
final_target_speed = min( nominal_planner_target,
                          cut_in_limit_if_active,
                          roundabout_limit_if_active,
                          highway_merge_limit_if_active )
```

is **order-independent, associative, and most-restrictive-wins**. Each source is
first clamped against the path-tracking nominal cruise target (a value that does
not lower it becomes `nullopt`), then all three are folded.
`test_external_speed_limit.cpp` verifies the Phase-16 matrix and full
permutation independence:

| # | cut-in | roundabout | highway merge | nominal | final |
| --- | --- | --- | --- | --- | --- |
| A | none | RELEASE | MERGE_READY | 10 | 10 (no cap) |
| B | SLOWDOWN 6 | RELEASE | MERGE_READY | 10 | 6 |
| C | none | YIELD ~7 | WAIT ~5 | 16.25 | 5 |
| D | SLOWDOWN 4 | RELEASE | WAIT ~6 | 16.25 | 4 |
| E | none | HOLD 0 | MERGE_READY | 10 | 0 |
| F | SLOWDOWN 4 | YIELD ~6 | HOLD 0 | 16.25 | 0 |
| G | HOLD 0 | HOLD 0 | HOLD 0 | 16.25 | 0 |

`MERGE_READY` never overrides a cut-in or roundabout cap; `WAIT` never raises
nominal speed; `HOLD` is always `0.0`.

## Scope: `FollowGlobalPath` longitudinal only

Same limitation as the cut-in and roundabout constraints. `PerceptionMission`
(`run_local_motion()`, DWA / Frenet / MPPI) is **not** constrained:
`VehicleConstraints.maximum_speed_mps` is validated strictly `> 0` so `HOLD`'s
`0.0` cannot be expressed there, and lowering the dynamic-window speed cap could
perturb DWA `(v, ω)` selection (a lateral effect this task forbids).
`VehicleConstraints.maximum_speed_mps` is **not** changed. Local motion also
runs its own corridor / occupancy / prediction longitudinal response.

## Single control-command publisher / no actuator path

`AdPlannerNode::publish_command()` → `PlannerRosInterfaces::publish_command()` on
`/ad/control/command` remains the sole intended production `CtrlCmd` publisher.
This change adds one `std_msgs/Float32` publisher, one `std_msgs/Bool`
publisher (both enabled builds only), and one `HighwayMergeGapResponse`
subscription; it publishes no throttle, brake, steering, gear, `cmd_vel`, DBW,
or lane-change message. `test_planner_highway_merge_constraint.py` asserts
`count_publishers("/ad/control/command") == 1` with the feature enabled, before
and after every advisory state.

## Verification

* `test_highway_merge_speed_constraint.cpp` — **26 pure cases**: disabled /
  missing / stale / inactive / wrong-zone → `nullopt` for every action;
  `MERGE_READY` → `nullopt`; `HOLD` → exactly `0.0`; `WAIT` → the comfortable-stop
  envelope (== `sqrt(2 a d)`); unknown action; malformed WAIT facts (NaN / Inf /
  negative ego speed / non-positive distances) → `nullopt`; WAIT cap
  non-negative / finite sweep; cap shrinks as the ego approaches; cap-above-nominal;
  determinism; authorization: fresh-active-matching-zone `MERGE_READY` → `true`;
  inactive `MERGE_READY` enum → `false`; stale / not-received / wrong-zone / `WAIT`
  (with generous facts) / `HOLD` / unknown action → `false`; `READY→WAIT` and
  `READY→stale` immediate revocation; determinism.
* `test_external_speed_limit.cpp` — the `initializer_list` fold + the Phase-16
  three-source matrix (A–G) + `AllThreeActiveMostRestrictiveWins` + full
  6-permutation order-independence over every `(cut-in, roundabout, merge)`
  state triple + `MERGE_READY` never overrides another source. The existing
  binary `combine_speed_limits` and the cut-in × roundabout tests are unchanged.
* `test_behavior_tree.cpp` — `registered_node_ids().size()` is now `7`;
  `HighwayMergeReadyConditionMirrorsAuthorizationFact` drives
  `context.highway_merge_authorized` false → true → false and asserts the
  condition's `FAILURE` / `SUCCESS` / `FAILURE` (no latch). The exact-XML
  production-tree assertion is unchanged and still passes.
* `test_planner_launch.py` — `enable_highway_merge_response_integration` is a
  declared generic argument defaulting to `""` (⇒ config `false`);
  `test_highway_merge_response_integration_is_opt_in_and_default_off` proves the
  default planner node carries no override, enabling it alone starts the response
  and forced-risk nodes and sets the override, and a bad value is rejected.
* `test_planner_highway_merge_constraint.py` — live `ad_planner` in
  `FollowGlobalPath` (`perception.enabled:=false`), straight fixture route, ego
  `8.0 m/s`, `highway_merge_response_max_age_s = 0.4`:
  * no response → limit `-1.0`, `merge_authorized == False`, governed target
    `16.25` (the integration-**disabled** baseline is not a second live launch;
    it is covered by `test_planner_launch.py`'s default-off assertions and the
    pure `not received / not enabled → nullopt / false` cases);
  * inactive `MERGE_READY` enum → limit `-1.0`, `merge_authorized == False`,
    target `16.25`;
  * `MERGE_READY` → limit `-1.0`, `merge_authorized == True`, governed target
    `16.25` (byte-identical to the no-response baseline);
  * `WAIT` (available 34 m, comfortable stop 17.78 m) → limit `11.063`,
    `merge_authorized == False`, governed target `11.063` = `sqrt(2·1.8·34)`,
    still above the 8 m/s ego speed;
  * `HOLD` (available 8 m) → limit `0.0`, `merge_authorized == False`, governed
    target `0.0`, command flips to brake (`throttle == -1.0`);
  * wrong-zone `MERGE_READY` → limit `-1.0`, `merge_authorized == False`, target
    `16.25` (ignored wholesale);
  * re-arm `MERGE_READY` (authorized), then stop publishing > `max_age` → limit
    `-1.0`, `merge_authorized == False`, target `16.25` (no latch);
  * no state raised the target above nominal; steering byte-identical (`0.0`)
    across every state; exactly one `/ad/control/command` publisher throughout.
* Full `ad_planner` gtest suite (36 targets) and `test_planner_launch.py`,
  `test_planner_cut_in_constraint.py`, `test_planner_roundabout_constraint.py`,
  the cut-in / roundabout / highway-merge risk & response launch tests, and the
  highway-merge gap-risk runtime + response policy replays all pass. Isolated
  `colcon build --packages-select ad_planner --symlink-install` clean.
* Pre-existing host-environment failures, not caused by this change:
  `test_mppi_nav2_launch` (host lacks `nav2_common` / `nav2_controller`),
  `test_cut_in_response_runtime` / `test_cut_in_risk_runtime`
  (`ad_dynamic_object_risk_node` OOM/SIGKILL under parallel load — a perception
  node, untouched), `test_frenet_runtime_contract` under parallel `colcon test`
  (resource contention; passes isolated).

## Downstream freshness assumption

A stale `HOLD` expires to **no** highway-merge constraint after
`highway_merge_response_max_age_s` — a single `HOLD` followed by silence lets the
ego resume cruising toward the merge once the frame ages out. This is inherited
from the cut-in / roundabout consumers and is what a stateless, non-latching
integration requires: authorization and constraints both come only from a
**fresh** response. A live integration must keep publishing the response at a
rate faster than `highway_merge_response_max_age_s`.

## Not done in v1

No lane-change planner, no route / corridor / target-lane change, no steering or
trajectory command, no lane-change request to the controller, no
DWA / Frenet / MPPI lateral change, no production BehaviorTree transition into a
merge branch. No change to any Highway Merge Gap Response policy threshold,
reason precedence, coverage requirement, or `WAIT` / `HOLD` classification — the
integration consumes the policy result and does not re-decide it. No
`ad_interfaces` message change. No `VehicleConstraints.maximum_speed_mps` change.

## Recommended next task

**Highway Merge Mission Transition v1** — gate the existing highway merge /
lane-selection mission transition with the fresh `merge_authorized` fact, while
leaving lateral trajectory generation and steering execution in the existing
planner implementation and preserving the single `CtrlCmd` publisher.

**However, this repository has no executable highway merge / lane-selection
mission path today** (see "Existing highway mission / merge boundary" above): the
BehaviorTree has no merge branch, there is no route-change primitive, and Frenet's
`lane_change_weight` is a local trajectory-scoring cost, not a mission
transition. Before any steering integration, the missing mission primitive —
a route/corridor change or a merge lane-selection state that the `HighwayMergeReady`
condition can gate — must be built. Recommend scoping that as the next task
rather than attempting to wire `merge_authorized` into a transition that does
not exist.
