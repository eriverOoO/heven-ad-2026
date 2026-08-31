# Highway Merge Ego-on-Ramp Scenario v1

An **opt-in, source-grounded validation scenario** in which the ego begins on
the real K-City acceleration lane `route:0:left:1` and merges onto the mainline
`route:0`, so the existing highway-merge chain

```
Dynamic Object Risk -> Highway Merge Gap Risk -> Highway Merge Gap Response
                    -> Highway Merge Response Integration -> Highway Merge Mission Primitive
```

can be exercised with the ego **physically represented on the ramp**, and the
source-grounded commit / completion semantics of the mission primitive can be
checked against real geometry.

This task adds **no** lateral path generator, steering command, or second
`CtrlCmd` publisher. It is scenario / route-fixture infrastructure only.

## Why this scenario exists — the CASE B prerequisite

PR #24 (Highway Merge Mission Primitive v1) established **CASE B**: the committed
competition global path `ad_data/path/2026_molit_comp_global_path.txt` **is**
`route:0` (the mainline through lane) across the entire merge region, and the
production ego is never on the acceleration lane `route:0:left:1` at all
(`route:0:left:1` carries only the `heven-highway-npc` NPC). So
`HighwayMergeMissionState`, `HighwayMergeReady`, and `HighwayMergeCommitted`
exist but had **no ego-on-ramp scenario** in which a lateral merge primitive
could ever be meaningfully exercised.

The recommended next task after PR #24 — *Highway Merge Lateral Path Primitive
v1* — cannot proceed without such a scenario. This scenario is that
prerequisite.

## Source and target corridors

| role | lane | notes |
| --- | --- | --- |
| source | `route:0:left:1` | the real K-City acceleration lane; single source link `A2256W000409`; 336 corridor points; `route_s` 1118.7418 .. 1286.1546 m; lateral offset to `route:0` tapers 3.94 m -> 0.00 m |
| target | `route:0` | the mainline through lane / merge target (there is no separate post-merge centerline; see `highway_merge.json`) |

## Ego start pose — provenance

`route:0:left:1` point 0, **verbatim from `ad_data/map/route_corridor.json`**:

| field | value | source |
| --- | --- | --- |
| x, y, z (m, map) | `66.0109296059, 256.5335690919, 28.3102668207` | `route:0:left:1` point 0 |
| yaw (rad) | `-1.568021` (= `-89.8410` deg) | lane tangent at point 0 |
| start link | `A2256W000409`, ratio 0 | the `route:0:left:1` source link |
| `route:0`-projected station | ~ **1118.74 m** | `project_to_frenet(route:0, start)` |
| lateral offset to `route:0` | ~ **3.94 m** | acceleration-lane nominal offset |

**The ego cannot start further back on the ramp.** `route:0:left:1` does not
extend upstream of the merge zone entry on the K-City map — its first point
coincides with `route_s_zone_entry_m`. The `INACTIVE` (far) and `APPROACH`
states are therefore represented with the ego on `route:0` itself (the mainline
approach), and every `WAITING`..`COMPLETE` pose is a genuine `route:0:left:1`
sample. This is a direct consequence of CASE B, documented, not worked around.

## Route fixture — provenance and join semantics

`ad_data/path/test_highway_merge_ego_ramp_path.txt` (**validation-only, never a
production default**):

* **source section** — `route:0:left:1`, all 336 points, `route_s`
  1118.7418 .. 1286.1546 m, copied verbatim from `route_corridor.json`. The
  acceleration-lane centerline itself performs the ramp -> mainline lateral
  shift (offset 3.94 m -> 0.00 m); no lateral transition is synthesised.
* **target section** — `route:0`, points with `route_s` in
  `(1286.1546, ~1486.15]` m (400 points), copied verbatim.
* **join** — the last `route:0:left:1` point equals the `route:0` point at
  `route_s` 1286.1546 m **exactly** (0.00 m separation); the target section
  starts at the next `route:0` point (0.50 m nominal spacing, 0.0015 rad heading
  change). No discontinuous jump, no heading break, no invented Cartesian
  waypoint. `test_highway_merge_ego_ramp_scenario.py` regenerates the fixture
  from the corridor and asserts byte-equality plus continuity.

## Production global path — unchanged (mandatory)

`ad_data/path/2026_molit_comp_global_path.txt` is byte-for-byte unchanged
(SHA-256 `50658991…cc05`, matching `route_corridor.json`'s recorded
`source_sha256.global_path`). `planner.yaml` still selects it as the default
`path_file`. The scenario is reachable only through:

* `ros2 launch ad_planner highway_merge_ego_ramp_scenario.launch.py data_dir:=<abs ad_data>`
  (generates a fixture corridor at launch time — a byte copy of
  `route_corridor.json` with only `source_sha256.global_path` rewritten to the
  fixture digest, lane geometry identical), or
* the two launch tests below, which teleport the ego along the real ramp
  samples, or
* `ad_morai_scenario_reset` / `ad_morai_scenario_setup` with
  `scenario_file:=ad_data/scenarios/kcity_highway_ego_onramp_v1.json`.

## Primary-route projection behaviour (the critical compatibility check)

PR #24's mission primitive computes ego progress with
`project_primary_route(route:0, ego_pose)`, and `ad_highway_merge_gap_risk`
computes it with `project_to_frenet(route:0, ego_pose)`. Both were only ever
exercised with the ego on `route:0`.

With the ego on `route:0:left:1` (up to ~3.9 m laterally displaced), **both
projections stay fully monotonic** and track the source-lane station almost
exactly:

| projection | monotonic over all 336 ramp points | `projected_s − source_s` |
| --- | --- | --- |
| `project_primary_route` (pi/2 heading gate) | yes (0 regressions) | `[−0.10, +0.11] m` |
| `project_to_frenet` (nearest point) | yes (0 regressions) | `[−0.06, +0.11] m` |

`ego_route_speed_mps` (`s_dot` from the Frenet term `1/(1 − kappa·d)` with
`d ~ 3.9 m`) varies within `[7.66, 8.47]` m/s for a true 8 m/s ego — a <6 %
ETA effect from the small real curvature of `route:0` on this stretch, well
inside the gap-risk node's tolerances. **No mission-progress or gap-risk
architecture change is required.**

## Gap-risk / gap-response / integration compatibility

* **Gap Risk** already projects the ego onto `route:0` (the full target lane),
  not onto a ramp centerline. For a ramp ego the live pipeline test records
  `ego_route_s_m` monotonic and matching the source station within 0.4 m, and
  `ego_merge_timing_valid == true` throughout the sweep. No risk-policy change.
* **Gap Response** — actions remain driven by target-mainline traffic. The
  live pipeline test drives four deterministic cases (below). No threshold
  (`1.5` s front headway, `2.0` s rear headway, `3.0` s closing time, `6.0` m
  route gap) was retuned.
* **Response Integration** — `WAIT`/`HOLD` still lower the path-tracking target
  through the existing 6th `target_speed_mps` argument; a fresh active
  `MERGE_READY` still sets the revocable `PlannerContext::highway_merge_authorized`
  fact. No new speed-control logic.

## Mission commit — physical meaning for a ramp ego

The mission commit boundary (`derive_highway_merge_commit_station`, threshold
`commit_lateral_separation_m` = 3.5 m) resolves to source station
**~1213.28 m**. The live planner test confirms that at that pose the ego is
genuinely inside the acceleration-lane taper:

| check | value |
| --- | --- |
| commit source station | 1213.28 m |
| ego lateral separation to `route:0` at commit | **3.496 m** (just under the 3.5 m threshold) |
| committed span before merge completion | ~72.9 m |

So `COMMITTED` now corresponds to a real physical state: the ramp geometry is
funnelling the ego onto `route:0` and the merge cannot be cleanly abandoned.

## Mission completion — physical meaning

At `route_s_merge_complete_m` = 1286.1546 m the acceleration lane's last point
**coincides with `route:0`** (lateral separation `0.000 m`). The live planner
test confirms `COMPLETE` occurs there and the ego is established on the target
corridor.

## Traffic actors

Deterministic mainline NPCs on `route:0`
(`ad_data/scenarios/kcity_highway_ego_onramp_v1.json`):

| id | role | corridor | `route_s` | speed |
| --- | --- | --- | --- | --- |
| 101 | front mainline vehicle | `route:0` | ~1200 m | 25 m/s |
| 102 | rear mainline vehicle | `route:0` | ~1080 m | 25 m/s |

The stock `kcity-highway` actor preset (`heven-highway-npc`) spawns an NPC on
`route:0:left:1` — the ego's lane here — so it is **deliberately not reused**.
Ego / NPC separations at t=0 are 82 m (front) and 39 m (rear); no test NPC is
on the ego's acceleration lane.

The live pipeline test exercises four deterministic gap-response cases with the
ramp ego at `route:0:left:1` station ~1170 m:

| case | traffic | response |
| --- | --- | --- |
| no traffic | none | **MERGE_READY** (deterministic) |
| clear gap | front +80 m @ 26 m/s, rear −90 m @ 24 m/s | WAIT (mainline at 3x ego speed overtakes the slow ramp ego — a physically correct hold) |
| fast rear | rear −12 m @ 33 m/s | not MERGE_READY (WAIT) |
| insufficient prediction | rear object, 2 s prediction horizon | not MERGE_READY (WAIT) |

## Deterministic mission state sequence (live, ego on the real ramp)

`test_planner_highway_merge_ego_ramp.py` teleports the ego through real
`route:0:left:1` poses while toggling `HighwayMergeGapResponse`:

| step | ego pose | authorization | mission state |
| --- | --- | --- | --- |
| 1 | `route:0` s≈400 (mainline) | — | INACTIVE |
| 2 | `route:0` s≈1000 (mainline approach) | WAIT | APPROACH |
| 3 | `route:0:left:1` s≈1120 | none | WAITING |
| 4 | `route:0:left:1` s≈1150 | MERGE_READY | AUTHORIZED |
| 5 | `route:0:left:1` s≈1160 | WAIT | WAITING (revoked before commit, no latch) |
| 6 | `route:0:left:1` s≈1180 | MERGE_READY | AUTHORIZED |
| 7 | `route:0:left:1` s≈1220 | MERGE_READY | COMMITTED (crossed ~1213 m; ego 3.50 m off `route:0`) |
| 8 | `route:0:left:1` s≈1250 | WAIT | COMMITTED (post-commit revocation **not** honored) |
| 9 | `route:0` s≈1295 (merged) | WAIT | COMPLETE (ego 0.00 m off `route:0`) |
| 10 | `route:0` s≈1330 | — | INACTIVE |

Steering across the sweep is pure Stanley path-tracking (the ramp poses sit up
to ~3.9 m off the tracked production route by construction); it never increases
across the `AUTHORIZED -> COMMITTED` transition and stays far inside the
steering lock. Exactly one `/ad/control/command` publisher throughout.

## Validation type

**Not a live MORAI run** — no MORAI simulator / gRPC runtime is available in this
environment. Validation is:

* pure geometry / data tests against the real checksum-verified corridor
  (`test_highway_merge_ego_ramp_scenario.py`, 15 cases);
* a deterministic ROS pipeline replay: synthetic `DynamicObjectRiskArray` ->
  real `ad_highway_merge_gap_risk` -> real `ad_highway_merge_gap_response`, ego
  on real ramp poses (`test_highway_merge_ego_ramp_pipeline.py`);
* a deterministic live `ad_planner` replay with the response integration +
  mission primitive enabled, ego on real ramp poses
  (`test_planner_highway_merge_ego_ramp.py`).

The MORAI scenario JSON (`kcity_highway_ego_onramp_v1.json`) is authored to the
existing `ad_morai_bridge_dev.scenarios.reset.load_reset_plan` contract for a
future real run; its ego pose and actor placements are locked by the geometry
test.

## Not done in v1 (explicit)

**No lateral path generator.** No `HighwayMergeLateralPath`, spline / quintic
merge, Frenet lateral shift, Stanley target-path switch, steering command, or
second `CtrlCmd` publisher. No C++ change to `ad_planner` — the mission and
gap-risk projections already handle a ramp ego unchanged (see above). No
production global route, `planner.yaml` default, BehaviorTree, or safety-system
change. No `ad_interfaces` change.

## Files

* `ad_data/path/test_highway_merge_ego_ramp_path.txt` — validation route fixture
* `ad_data/scenarios/kcity_highway_ego_onramp_v1.json` — MORAI scenario (opt-in)
* `ad_data/scenarios/kcity_highway_ego_onramp_v1.manifest.yaml` — scenario manifest
* `ad_planner/launch/highway_merge_ego_ramp_scenario.launch.py` — opt-in launch
* `ad_planner/test/test_highway_merge_ego_ramp_scenario.py` — 15 pure-data tests
* `ad_planner/test/test_highway_merge_ego_ramp_pipeline.py` — live risk -> response replay
* `ad_planner/test/test_planner_highway_merge_ego_ramp.py` — live planner + mission replay
* `ad_planner/CMakeLists.txt` — test registration
* `docs/planning/highway_merge_ego_ramp_scenario_v1.md`, `docs/agent/STATUS.md`

## Recommended next task

**Highway Merge Lateral Path Primitive v1** — use this validated ego-on-ramp
scenario and the source-grounded `route:0:left:1` -> `route:0` corridor geometry
to generate a merge reference path for the existing lateral controller, gated by
the committed highway-merge mission state (`HighwayMergeCommitted`), without
publishing steering directly or creating a second `CtrlCmd` publisher.
