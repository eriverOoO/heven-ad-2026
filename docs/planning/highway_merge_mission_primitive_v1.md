# Highway Merge Mission Primitive v1

The explicit mission-state layer between the revocable merge authorization
(`PlannerContext::highway_merge_authorized`, from the Highway Merge Response
Integration, PR #23) and a future lateral executor. It commands **nothing** — no
steering, no lane change, no `CtrlCmd`, no speed. It tracks approach / waiting /
authorization / commitment / completion from source-grounded route geometry so
that:

* authorization stays **revocable before the ego is physically committed**, and
* a transient authorization loss **after** commitment does not later cause a
  lateral executor to reverse a merge maneuver mid-way.

It is **opt-in and off by default** (`enable_highway_merge_mission: false`):
disabled ⇒ no merge geometry is loaded, the mission state is permanently
INACTIVE, no observability publisher is created, and planner behaviour is
identical to a build without this feature.

## Does the existing global path already encode the merge? — **CASE B**

**No.** The committed competition global path does not encode a ramp → mainline
merge maneuver — and, more precisely, **the ego is never on the acceleration
lane at all.**

Evidence (all asserted as a regression in
`test/test_highway_merge_mission_geometry.py` against
`ad_data/path/2026_molit_comp_global_path.txt` +
`ad_data/map/route_corridor.json`):

| what | finding |
| --- | --- |
| Global path vs `route:0` (mainline through lane), lateral offset at route_s 1120 / 1160 / 1200 / 1240 / 1286 m | **≈ 0.00 m at every station** — the global path *is* the `route:0` centerline through the entire merge region (sampled 1080–1320 m). |
| Global path vs `route:0:left:1` (the acceleration lane), lateral offset | **≈ 3.9 m at s 1120**, still ≈ 3.7 m at s 1200, closing to ≈ 0 only at s 1286 — i.e. `route:0:left:1` tapers *into* `route:0` from the side while the ego drives straight down the mainline. |
| Route checkpoints 10 (`61.99, 261.52`) and 11 (`66.48, 98.60`) — the ego route through the merge region | both sit on `route:0`, not on `route:0:left:1`. |
| `kcity-highway` actor preset (`ad_morai_bridge_dev/config/actor_presets.yaml`) | it is `heven-highway-npc` — an **NPC** vehicle spawned at `route:0:left:1` waypoint 0 (66.01, 256.53) at 50 m/s. It is a fast mainline/ramp object for gap-risk testing, **not the ego**. |
| Highway Merge Gap Risk v1 canonical replay | sweeps the ego "on the real `route:0` centerline" — the whole upstream chain projects the ego onto `route:0`. |

**Consequence.** There is no lateral transition in the global path to reuse, and
no lateral maneuver for the ego to perform on the committed route — the ego is
already established on the target corridor. Generating a ramp → mainline
reference path therefore has **no consumer scenario** until a route or spawn
places the ego on `route:0:left:1`. This mission primitive tracks mission
*state* and exposes a `source` / `target` corridor intent for a future lateral
executor; it generates no path. If a future route reroutes the ego via the
ramp, `test_highway_merge_mission_geometry.py` fails and this CASE A / CASE B
decision must be revisited.

## Mission states

`HighwayMergeMissionState` (`highway_merge_mission.hpp`), `std::uint8_t`-backed:

| value | state | meaning |
| --- | --- | --- |
| 0 | `INACTIVE` | ego outside the highway-merge approach window this traversal |
| 1 | `APPROACH` | ego inside the approach window, before the merge decision region |
| 2 | `WAITING` | the merge decision region is applicable but authorization is false |
| 3 | `AUTHORIZED` | fresh `merge_authorized == true` and the ego has **not** yet committed |
| 4 | `COMMITTED` | ego has crossed the source-grounded commit boundary **while authorized** |
| 5 | `COMPLETE` | ego has passed the merge-complete station (established on the target corridor) |

## Source-grounded stations (target corridor `route:0`)

Derived once at startup from `highway_merge.json` + the checksum-verified route
corridor. `approach_entry < zone_entry < commit < merge_complete <= exit`.

| station | value (kcity_highway_onramp) | provenance |
| --- | --- | --- |
| `approach_entry_route_s_m` | `zone_entry − mission_approach_window_m` = `1118.74 − 400` = **718.74 m** | `mission_approach_window_m` (400, config) — `configure_highway_merge_mission()` rejects a value `> 400` (the upstream `ad_highway_merge_gap_risk` `maximum_ego_approach_distance_m` default), soft-failing the mission to permanently INACTIVE, so APPROACH can never begin earlier than the upstream response can be active. |
| `zone_entry_route_s_m` | **1118.74 m** | first `route_s_m` of `route:0:left:1` (declarative in `highway_merge.json`, cross-checked at startup within 2 m against the source lane, the same margin `ad_highway_merge_gap_risk` uses). |
| `commit_route_s_m` | **≈ 1213.3 m** | **derived**: the first `route:0:left:1` station whose lateral separation from `route:0` (via `project_to_frenet`) has fallen below `commit_lateral_separation_m` = 3.5 m. See below. |
| `merge_complete_route_s_m` | **1286.15 m** | last `route_s_m` of `route:0:left:1` (`= merge_reference_route_s_m` upstream; the accel-lane lateral offset there is 0). |
| `exit_route_s_m` | `merge_complete + mission_exit_release_m` = `1286.15 + 20` = **1306.15 m** | `mission_exit_release_m` (20, config) — a short established-on-mainline release band. |

### Commit boundary derivation and provenance

`route:0:left:1`'s lateral separation from `route:0` is flat at ≈ 3.73–3.94 m
from its start (s 1118.7) through s ≈ 1200, then closes at ≈ 0.05 m/m to 0 at
s 1286.15. With `commit_lateral_separation_m = 3.5 m` the commit boundary is the
first station below that — **s ≈ 1213.3 m**, right at the taper onset. Past it
the acceleration-lane geometry itself is funneling a vehicle onto `route:0` and
the merge is physically committed. `3.5 m` is the first ≈ 0.44 m (> 10 %) of
taper below the ≈ 3.94 m nominal accel-lane offset, and is safely below the
≈ 3.73 m taper-plateau noise floor so a station on the plateau never trips it.

Candidates considered and **rejected**: a vehicle-half-width criterion
(separation < ≈ 1 m) lands at s ≈ 1264 m — only 22 m before merge complete, too
late to be a meaningful commit point; `merge_standoff_m = 6.0` is the response's
own WAIT/HOLD decision boundary and stays owned upstream — it is not reused
here.

The commit station is **computed from the corridor at load time**, not
hardcoded. If the source lane never crosses the threshold inside
`(zone_entry, merge_complete)` the mission soft-fails to permanently INACTIVE
(logged, never a throw at construction).

## State transitions (pure `step_highway_merge_mission`)

`s` = ego route station on `route:0`; `prev` = previous mission state.

```text
!enabled                                   -> INACTIVE
backward jump  s < prev_s - reset_jump_m    -> INACTIVE   (sim reset / route loop wrap)
!route_progress_valid                        -> prev       (hold; never advance)

prev == COMMITTED:
    s >= merge_complete                      -> COMPLETE
    else                                     -> COMMITTED   (post-commit auth loss is NOT honored)
prev == COMPLETE:
    s < approach_entry or s > exit            -> INACTIVE
    else                                     -> COMPLETE

prev in {INACTIVE, APPROACH, WAITING, AUTHORIZED}:
    s < approach_entry or s >= merge_complete -> INACTIVE
    s < zone_entry:
        merge_authorized                     -> AUTHORIZED
        else                                 -> APPROACH
    s in [zone_entry, merge_complete):
        merge_authorized and s >= commit      -> COMMITTED
        merge_authorized                     -> AUTHORIZED
        else                                 -> WAITING
```

* **Authorization revocable before commit**: yes. `AUTHORIZED → WAITING` the
  moment `merge_authorized` goes false while `s < commit` — no latch. Regained
  authorization returns `WAITING → AUTHORIZED`.
* **Authorization revocable after commit**: no (mission commitment ≠ current
  upstream authorization). Once `COMMITTED`, an upstream `WAIT` / `HOLD` / stale
  / missing frame drives `merge_authorized` false and the `authorized_now`
  output echoes that — but the mission state stays `COMMITTED` until `COMPLETE`.
  A future lateral executor must not reverse a merge mid-maneuver. The two
  facts are exposed separately (`committed` vs `authorized_now`).
* **Commit monotonicity within a traversal**: after `COMMITTED` the state can
  only be `COMMITTED` or `COMPLETE` (swept and locked by
  `NoBackwardTransitionAfterCommit`). The only way back is a traversal reset.
* **Crossing `commit` unauthorized never commits** — the state stays `WAITING`.
* **Stopped ego**: mission state is a function of route station and
  authorization only, never wall-clock — a stopped ego simply does not advance.

### Sim reset / route loop wrap

`route:0` is a 2184.6 m loop, so a large backward jump in ego route station is
either a MORAI sim-time reset or the loop wrapping. `mission_route_s_reset_jump_m`
(default 3.5) is ≈ 2 × the control period (0.05 s) × the 33.33 m/s merge-region
speed limit — the largest forward Δs one tick can physically explain, doubled.
A backward Δs beyond that resets the mission to `INACTIVE` (any carried
`COMMITTED` from an old run is dropped).

## Authorization and freshness ownership

The mission's only decision input is `PlannerContext::highway_merge_authorized`,
already computed by the Highway Merge Response Integration from a fresh,
zone-matched, revocable `HighwayMergeGapResponse`. The mission **never** reads
`HighwayMergeGapResponse`, `HighwayMergeGapRisk`, or `DynamicObjectRisk`, and
duplicates **no** freshness or gap-policy logic. Chain:

```text
Gap Risk -> Gap Response -> Response Integration -> Mission Primitive
```

The mission and response-integration feature flags are **independent**. With the
mission enabled but `enable_highway_merge_response_integration` **off**,
`highway_merge_authorized` is always false, so the mission never leaves
`APPROACH` / `WAITING` — the correct fail-safe. The mission flag deliberately
does **not** force-enable the response integration (that would change
longitudinal behaviour as a side effect of enabling a state tracker).

## Ego route progress

`update_highway_merge_mission()` (top of `tick()`, after the ego pose is
prepared and before the behavior tree) calls the existing
`project_primary_route(route_corridor_->corridor, ego_pose_in_map)` →
`route_s_m` on `route:0`. It is wrapped in try/catch: a projection failure (no
forward-facing segment, invalid pose, missing corridor, stale TF) is treated as
"no route progress this tick" → the mission holds its previous state, never
advancing. It does **not** reuse `context_.inputs.route_occupancy` (that is
gated on `perception.route_aligned_activation`, default false).

## BehaviorTree exposure

Mission state is recomputed in `AdPlannerNode::tick()` **before**
`supervisor_->tick()` and written to `PlannerContext::highway_merge_mission`
(`{state, active, committed, authorized_now}`, decoupled from the
Frenet-carrying mission header so the BT layer takes no route-geometry
dependency).

One new read-only `BT::SimpleCondition` is registered:

* **`HighwayMergeCommitted`** — `SUCCESS` iff
  `context.highway_merge_mission.committed` (state `COMMITTED` or `COMPLETE`).
  This is the one fact `HighwayMergeReady` (PR #23) cannot express: it stays
  `SUCCESS` through a transient post-commit authorization loss. It publishes no
  control, changes no path/route/steering, and never itself latches.

`HighwayMergeReady` (PR #23) is unchanged. `ad_bt_node_ids()` is now 8 entries;
`test_behavior_tree.cpp` expects `registered_node_ids().size() == 8`.

**Production BehaviorTree XML: unchanged.** No merge / lane-selection mission
path exists to gate (see CASE B above), so `HighwayMergeCommitted` is registered
but the production tree does not reference it — the exact-XML assertion in
`test_behavior_tree.cpp::HasExpectedPriorityAndCruises` is the standing proof.
`CollisionRecovery`, `TrafficStop`, `PerceptionMission`, `FollowGlobalPath`,
`FailSafeBrake` keep their exact priority and authority; `COMMITTED` disables
none of them, and every independent safety constraint (cut-in / roundabout
longitudinal caps, collision recovery, traffic stop, fail-safe brake) is
entirely separate.

## Configuration (`config/planner.yaml`)

| key | default | meaning |
| --- | --- | --- |
| `enable_highway_merge_mission` | `false` | master opt-in; off ⇒ no geometry loaded, state INACTIVE, no publisher |
| `mission_approach_window_m` | `400.0` | APPROACH begins this far before `zone_entry`; a value `> 400` is rejected at startup (soft-fail to INACTIVE) so it stays `<=` the risk `maximum_ego_approach_distance_m` default |
| `commit_lateral_separation_m` | `3.5` | commit-boundary derivation threshold (above) |
| `mission_exit_release_m` | `20.0` | COMPLETE ends this far past `merge_complete` |
| `mission_route_s_reset_jump_m` | `3.5` | one-tick backward-Δs beyond this ⇒ traversal reset |
| `highway_merge_mission_zone_id` | `kcity_highway_onramp` | zone in `merge_geometry_file` |
| `merge_geometry_file` | `""` | empty ⇒ `ad_planner/config/highway_merge.json` from the package share |
| `topics.highway_merge_mission_state` | `/ad/planner/highway_merge_mission_state` | `std_msgs/UInt8` observability: the state enum, published only when enabled |

No merge policy threshold (front/rear headway, rear-closing time, predicted
route gap, `merge_standoff_m`, comfortable deceleration) is repeated here — all
stay owned by `ad_highway_merge_gap_response`.

`planner.launch.py enable_highway_merge_mission:=true` sets the parameter. It
does **not** force-start any risk / response node (the mission is independent of
them; a deployment wanting a live authorization signal enables
`enable_highway_merge_response_integration` too, or launches the advisory
chain).

## Target-corridor intent (CASE B)

For a future lateral executor the mission exposes, via the startup log line and
`highway_merge.json`:

* **source corridor** `route:0:left:1` (the acceleration lane)
* **target corridor** `route:0` (the mainline through lane — which is also the
  ego's current global path, so a lateral executor tracking `route:0` is already
  correct on the committed route; the "merge" on this route is a longitudinal
  gap-acceptance problem, not a lateral one)

No controller subscriber is created for this intent in v1.

## Single control-command publisher / no actuator path

`AdPlannerNode::publish_command()` on `/ad/control/command` remains the sole
intended production `CtrlCmd` publisher. This change adds one `std_msgs/UInt8`
diagnostic publisher (enabled builds only) and reads the existing
`route_corridor_` / merge geometry; it publishes no throttle, brake, steering,
gear, `cmd_vel`, DBW, lane-change, or trajectory message, and touches no lateral
planner math (Stanley, DWA, Frenet, MPPI). `test_planner_highway_merge_mission.py`
asserts `count_publishers("/ad/control/command") == 1` and small steering
throughout.

## Verification

* `test_highway_merge_mission.cpp` — **27 pure gtests**: 4 geometry / commit-
  station-derivation cases (non-ordered / non-finite geometry rejected; first
  station below threshold; `nullopt` when the source lane never crosses the
  threshold / degenerate input) + 23 transition and sequence cases (disabled,
  far-before, approach, unauthorized→waiting, waiting+auth→authorized,
  revoke-before-commit→waiting, regain→authorized,
  cross-commit-authorized→committed, cross-commit-unauthorized→waiting,
  revoke-after-commit→committed, →complete, →inactive, sim-reset→inactive,
  small-noise-not-a-reset, wrong-zone→inactive, deterministic, no-backward-
  transition-after-commit sweep, hold-on-invalid-progress, output-flag /
  finiteness sweep, and the Phase 31-34 sequences: unsafe→authorized,
  authorized→unsafe-before-commit, authorized→committed→unsafe stays committed
  with `authorized_now == false`, commit→complete→inactive).
* `test_highway_merge_mission_geometry.py` — **3 data tests** locking CASE B:
  global path ≡ `route:0` (lateral < 0.5 m) and ≈ 3.9 m from `route:0:left:1` at
  the zone entry tapering to < 0.5 m by merge-complete; the derived commit
  boundary is source-grounded and inside `(zone_entry, merge_complete)` near
  s ≈ 1213 m with a > 60 m committed span; single configured zone.
* `test_behavior_tree.cpp` — `registered_node_ids().size() == 8`;
  `HighwayMergeCommittedConditionMirrorsMissionCommitFact` (false→true→true
  through an `authorized_now` drop). Production-tree exact-XML assertion
  unchanged.
* `test_planner_launch.py` — `enable_highway_merge_mission` declared generic arg
  (default `""` ⇒ config `false`);
  `test_highway_merge_mission_is_opt_in_and_default_off` proves the default
  planner node carries no override, opt-in sets it without composing extra
  nodes, and a bad value is rejected.
* `test_planner_highway_merge_mission.py` — live `ad_planner` on the **real**
  route corridor + global path + merge geometry, `FollowGlobalPath`,
  `enable_highway_merge_mission` + `enable_highway_merge_response_integration`
  true, ego teleported through `route:0` stations while
  `HighwayMergeGapResponse` authorization is toggled:
  `s 400 → INACTIVE`; `s 1000 → APPROACH`; `s 1150 unauth → WAITING`;
  `s 1150 MERGE_READY → AUTHORIZED`; `s 1160 WAIT → WAITING` (revoked, no latch);
  `s 1175 MERGE_READY → AUTHORIZED`; `s 1235 MERGE_READY → COMMITTED`
  (crossed ≈ 1213 m); `s 1255 WAIT → COMMITTED` (post-commit revocation NOT
  honored, `highway_merge_authorized == False`); `s 1295 WAIT → COMPLETE`;
  `s 1330 → INACTIVE`. Steering `< 0.2 rad` throughout; one `/ad/control/command`
  publisher; clean shutdown.
* Full `ad_planner` gtest suite **37/37**; `test_planner_cut_in_constraint.py`,
  `test_planner_roundabout_constraint.py`,
  `test_planner_highway_merge_constraint.py`, and the cut-in / roundabout /
  highway-merge risk & response launch + runtime tests all pass. Isolated
  `colcon build --packages-select ad_planner --symlink-install` clean.
* Pre-existing host failures, not caused by this change: `test_mppi_nav2_launch`
  (nav2 absent), `test_cut_in_response_runtime` / `test_cut_in_risk_runtime`
  (perception-node OOM under parallel load), `test_frenet_runtime_contract`
  (parallel contention; passes isolated).

## Not done in v1

No lateral path generation, no ramp-to-mainline reference path, no lane-change
spline / polynomial / Frenet merge trajectory, no Stanley target-path switch, no
steering / `CtrlCmd` / trajectory command, no route or target-lane change, no
DWA / Frenet / MPPI lateral change, no production BehaviorTree transition. No
change to any Highway Merge Gap Risk / Response policy. No `ad_interfaces`
change. No `VehicleConstraints.maximum_speed_mps` change.

## Recommended next task

**Highway Merge Lateral Path Primitive v1** — consume the committed
target-corridor intent and generate a source-grounded ramp-to-mainline
reference path for the existing lateral controller, without publishing steering
directly or creating a second `CtrlCmd` publisher.

**Prerequisite, stated plainly:** on the committed competition route the ego is
already on `route:0` and performs no lateral maneuver (CASE B above), so a
ramp-to-mainline reference path has no consumer scenario yet. Before building
the lateral primitive, a route or actor spawn must place the ego on
`route:0:left:1` through the merge zone — otherwise the lateral path would be
generated for a maneuver the route never asks for. Scope that route/spawn work
(or an explicit decision to model the ego-on-ramp scenario) as the immediate
next step.
