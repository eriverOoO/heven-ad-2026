# Highway Merge Lateral Reference Path Primitive v1

An **opt-in, default-off** online primitive that gives the **existing**
`FollowGlobalPath` lateral controller a source-grounded
`route:0:left:1` → `route:0` reference path while a genuine ramp ego is
merging, so the vehicle follows the acceleration lane instead of being pulled
prematurely toward the production mainline path.

It **selects a reference path**. It publishes no steering, adds no second
lateral controller, adds no second `CtrlCmd` publisher, and changes no
controller math or gains.

## Why an online primitive, when a validation fixture already exists

PR #25 shipped `ad_data/path/test_highway_merge_ego_ramp_path.txt` — a
validation *oracle* built offline as `route:0:left:1` + `route:0`. That file is
not a runtime input: production must not load it, and hard-coding a path defeats
the point. This primitive **derives** the same geometry online from the
checksum-verified `ReferenceCorridor` + `highway_merge.json`, and
`test_highway_merge_reference_path.py` proves the online output equals the
golden fixture (x/y) point-for-point.

## Why no quintic / spline / synthetic lane-change curve

PR #25 measured the real map: `route:0:left:1` already tapers from ~3.94 m off
`route:0` to **0.00 m** at merge completion, where its last point **coincides
with a `route:0` point** (0.00 m separation, ~0.50 m to the next point, ~0.0015
rad heading change). The acceleration-lane centerline *is* the merge curve. A
synthetic lateral offset would only add error against authoritative geometry, so
none is used. The builder rejects a build whose source→target join is not
continuous.

## Source and target corridors

| section | lane | content |
| --- | --- | --- |
| source | `route:0:left:1` | all 336 corridor points, verbatim (`route_s` 1118.7418 .. 1286.1546 m) |
| target | `route:0` | points with `route_s` strictly `> merge_complete`, up to `+ reference_path_target_continuation_m` (200 m → 400 points) |

`ReferencePoint` carries no `z`, so the built `ad_control::Route` has `z = 0`;
the Stanley / Profile-Stanley controller is 2-D and every comparison to the
golden fixture is x/y.

## Path construction algorithm (`build_highway_merge_reference_path`, pure)

1. Validate: both lanes ≥ 2 points; `merge_complete` finite; the source lane's
   last station within `station_consistency_margin_m` (2 m) of `merge_complete`.
2. **SOURCE section**: append every `route:0:left:1` point as `{x, y, 0}`;
   reject a non-finite or station-regressing point.
3. **TARGET section**: append `route:0` points with `route_s > merge_complete`
   (strict — the coincident point is contributed once, by the source), up to
   `merge_complete + target_continuation_m`; require ≥ 2.
4. **Join contract**: Euclidean gap ≤ `maximum_join_gap_m` (1 m), pre- and
   post-join heading delta ≤ `maximum_join_heading_delta_rad` (0.20 rad).
5. No consecutive duplicate points; bounded spacing.
6. Any failure throws → the feature is **inert** (no merge controller built,
   production path untouched).

Built once at node configuration. A **second** `PathTrackingController` is then
constructed from that route with the **same backend, gains, speed profile, and
PID** as the production controller (`make_path_tracking_controller(
path_tracking_backend_, merge_route, RosControllerParameterProvider(*this))`).

## Full path, not windowed

The merge route is the full source lane + 200 m of mainline (736 points). The
`RouteProgressTracker`'s first `update()` does a **global** nearest-segment
search, so no look-behind is needed; subsequent updates advance within
`forward_window` (the production controller's own value, unchanged). The 200 m
mainline continuation is the look-ahead so the tracker never hits a path-end
artifact while the ego finishes / just clears the merge.

## The seam into `FollowGlobalPath`

`AdPlannerNode::run_path_tracking()` (the `follow_global_path` BT leaf) now:

* computes the same combined longitudinal `target_speed_mps` override as before
  (cut-in ∧ roundabout ∧ highway-merge — unchanged);
* **always** updates the production controller (keeps its route progress / PID /
  launch-ramp state warm — see below);
* if the merge reference is selected this tick and its controller returns a
  valid result, `remember()`s that result; otherwise `remember()`s the
  production result.

`AdPlannerNode::publish_command()` on `/ad/control/command` is still the sole
`CtrlCmd` publisher.

### "Both controllers updated every tick" is deliberate — not a no-op

The production controller runs every tick even while the merge reference is
active. Its `pid_`, `launch_elapsed_s_`, `previous_steering_rad_` and route
progress therefore stay current, so the `COMPLETE` handoff back to it does
**not** re-trigger the launch ramp or a nearest-segment jump. During the merge
phase its (discarded) steering is large — it is tracking `route:0` from ~3.9 m
away — which is exactly why its output is discarded and the merge controller's
is used.

## Activation — NOT `COMMITTED`-only

`COMMITTED` is far too late: before commit, while the ego is physically on
`route:0:left:1`, the production global path is `route:0`, so Stanley would
already steer the ramp ego toward the mainline. The reference is active
throughout ramp participation.

`select_highway_merge_reference()` (per tick, deterministic):

```
feature off / build failed / mission geometry absent   -> production path, clear latch
mission state INACTIVE / APPROACH / COMPLETE            -> production path, clear latch
latched                                                 -> merge reference while state in {WAITING, AUTHORIZED, COMMITTED}
not latched:
    on_ramp AND state in {WAITING, AUTHORIZED, COMMITTED} -> latch, merge reference
    else                                                 -> production path
```

### Source-ramp proximity guard

The mission state alone does not prove the ego is on the ramp (PR #25: APPROACH
is represented on `route:0`). `on_ramp` requires
`|project_to_frenet(route:0:left:1, ego).d_m| ≤ reference_path_proximity_m`
**and** the source-lane projection station within `[zone_entry − 1, merge_complete + 1]`.

`reference_path_proximity_m` default **1.75 m** = `route:0:left:1`'s own
`left_width_m` / `right_width_m` in the route corridor. A `route:0` ego near the
zone entry is ~3.9 m off the ramp centerline → excluded. Near the taper end the
two centerlines converge (< 1.75 m); by then a real ramp ego is already latched,
and a `route:0` ego there is on geometry that is ~identical to the merge
reference anyway.

### Reference-selection latch (≠ authorization latch)

Once a genuine ramp ego is on the merge reference it stays there through
`COMMITTED` even if `on_ramp` transiently drops near the taper. The latch
releases at `COMPLETE` / `INACTIVE` / `APPROACH` / traversal reset. This is a
lateral-continuity latch; it does **not** latch merge *authorization* (which
stays revocable pre-commit, per PR #22–#24).

## Per-state reference behaviour

| mission state | lateral reference | note |
| --- | --- | --- |
| INACTIVE | production path | ego not on the ramp this traversal |
| APPROACH | production path | on `route:0` (no upstream ramp geometry — PR #25) |
| WAITING | **merge reference** | follow the ramp; longitudinal `WAIT`/`HOLD` owns whether the ego progresses |
| AUTHORIZED | **merge reference** | identical geometry to WAITING — no lateral change on authorization |
| WAITING (revoked pre-commit) | **merge reference** | no swap back to `route:0`; longitudinal constraint handles the revoked merge |
| COMMITTED | **merge reference** | monotone; no mid-merge reversal |
| COMMITTED + `authorized_now == false` | **merge reference** | matches PR #24 mission semantics |
| COMPLETE | production path | ego is on `route:0`; merge-reference tail **is** `route:0` → seamless |
| INACTIVE (after COMPLETE) | production path | latch cleared |

## Commit station vs geometric splice — different stations

* **Commit** ≈ 1213.28 m — mission *irreversibility* semantics only. The
  generated path still follows `route:0:left:1` there (lateral separation
  ~3.50 m); the reference is **not** spliced to `route:0` at commit.
* **Geometric splice** = merge complete ≈ **1286.15 m** — where source and
  target coincide (0.00 m). The source→target point handoff is here, and only
  here.

The live test confirms `highway_merge_reference_active` is `true` at the
`COMMITTED` pose (ego ~3.2 m off `route:0`), i.e. still on the acceleration
lane.

## Completion handoff

At `COMPLETE` the ego is on `route:0` and the merge-reference continuation past
`merge_complete` **is** `route:0` (same corridor points). Switching the selected
result back to the always-warm production controller is therefore geometric-
noise seamless: the live test records mean |steering| `0.003` rad at the
`COMPLETE` pose.

## Invalid geometry / invalid runtime path

* **Build-time**: any inconsistency → no merge controller, feature inert,
  production path and behaviour unchanged, `RCLCPP_ERROR` logged.
* **Runtime**: the merge reference is selected but its controller returns an
  invalid result (e.g. ego past the 200 m continuation) → fall back to the
  always-warm production controller for that tick, throttled `RCLCPP_WARN`. The
  independent longitudinal (`WAIT`/`HOLD`) and safety systems (collision
  recovery, traffic stop, fail-safe brake) still apply. This is the narrowest
  deterministic rule; a bounded "hold last merge reference" was considered and
  rejected as more state for no measured benefit (the built route always spans
  well past `COMPLETE`).

## Sim reset / route reset

`select_highway_merge_reference()` reads `context_.highway_merge_mission.state`,
which PR #24 already drives to `INACTIVE` on a backward `route_s` jump / sim
reset. `INACTIVE` clears the latch → production path. No merge reference state
survives a traversal.

## No policy / speed duplication

The reference-path code reads **only** mission state + `route:0:left:1`
geometry + ego pose. It never inspects front/rear gap, prediction coverage,
`MERGE_READY` reason, or closing time (owned by `ad_highway_merge_gap_response`),
and computes no `WAIT` / `HOLD` / merge / target speed (owned by the highway
merge response integration).

## Configuration (`config/planner.yaml`, all default-off / inert)

| key | default | meaning |
| --- | --- | --- |
| `enable_highway_merge_reference_path` | `false` | master opt-in; also requires `enable_highway_merge_mission` |
| `reference_path_proximity_m` | `1.75` | ramp-proximity tolerance (= `route:0:left:1` half-width) |
| `reference_path_target_continuation_m` | `200.0` | real `route:0` appended past the splice for look-ahead |
| `topics.highway_merge_reference_active` | `/ad/planner/highway_merge_reference_active` | `std_msgs/Bool`, published only when enabled |

`planner.launch.py enable_highway_merge_reference_path:=true` sets the param.
`highway_merge_ego_ramp_scenario.launch.py` enables it (with the mission +
response integration) on the validation route.

## Default-off contract / production lock

Disabled ⇒ no merge route built, no second controller, no publisher,
`run_path_tracking()` returns the production controller's result exactly as
before. `ad_data/path/2026_molit_comp_global_path.txt` is byte-unchanged
(SHA-256 `50658991…cc05`); `planner.yaml`'s default `path_file` is unchanged.
No Stanley / Profile-Stanley / PID gain, steering saturation, or wheelbase
change. No production BehaviorTree change, no new BT action. No `ad_interfaces`
change.

## Live validation

`test_planner_highway_merge_reference_path.py` — `ad_planner` with the
**production** route as the base path and the online reference override enabled,
ego teleported through real `route:0:left:1` samples:

| ego pose | mission | `reference_active` | mean \|steering\| |
| --- | --- | --- | --- |
| `route:0` s400 | INACTIVE | false | — |
| `route:0` s1000 (WAIT) | APPROACH | false | — |
| `route:0:left:1` s1120 | WAITING | **true** | 0.004 rad |
| `route:0:left:1` s1120 (MERGE_READY) | AUTHORIZED | **true** | 0.004 rad (unchanged) |
| `route:0:left:1` s1160 (WAIT) | WAITING | **true** | — |
| `route:0:left:1` s1220 (MERGE_READY) | COMMITTED | **true** | — |
| `route:0:left:1` s1250 (WAIT) | COMMITTED | **true** | 0.000 rad |
| `route:0` s1295 (WAIT) | COMPLETE | false | 0.003 rad (seamless) |
| `route:0` s1330 | INACTIVE | false | — |

The ego is **3.919 m** off production `route:0` at the zone-entry ramp pose;
PR #25's `test_planner_highway_merge_ego_ramp.py` recorded mean |steering|
~0.32–0.45 rad at the same poses with this feature **off** (Stanley pulling the
ego toward `route:0`). With the feature on it is ~0.004 rad — the vehicle
follows the ramp. Authorization does not change the lateral command (WAITING vs
AUTHORIZED at the identical pose: 0.0038 vs 0.0038 rad). Exactly one
`/ad/control/command` publisher; clean shutdown; 0 non-finite steering.

## Latency

The merge route + second controller are built once at configuration. Per tick
`select_highway_merge_reference()` is one `project_to_frenet` over the 336-point
source lane plus O(1) checks; `run_path_tracking()` calls one extra controller
`update()` (a nearest-segment search within `forward_window`). No 700-point path
is rebuilt per tick. Not separately instrumented; the live test's control loop
kept up with the 20 Hz input with no missed publishes.

## Not done in v1 / limitations

* No end-to-end MORAI run (no simulator / gRPC in this environment) —
  deterministic ROS replay only.
* The proximity guard cannot distinguish a ramp ego from a `route:0` ego once
  the centerlines converge (< 1.75 m, `route_s` ≳ 1250 m); the latch covers the
  real ramp case and the geometry is ~identical there regardless.
* The production controller runs an extra (discarded) `update()` per tick while
  the merge reference is active — deliberate (keeps handoff seamless), measured
  as negligible, but it is a real doubled call.
* No lateral behaviour when the mission is disabled (`enable_highway_merge_reference_path`
  requires `enable_highway_merge_mission`).

## Recommended next task

**Highway Merge End-to-End Execution Validation v1** — run the complete
Dynamic Object Risk → Merge Gap Risk → Merge Gap Response → longitudinal
integration → merge mission → online merge reference → existing
`FollowGlobalPath` controller chain in the ego-on-ramp scenario, and validate
`WAIT → AUTHORIZED → COMMITTED → COMPLETE` behaviour with deterministic traffic,
preferably in MORAI when available, before adding any further highway-merge
planning features.
