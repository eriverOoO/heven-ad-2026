# Highway Merge End-to-End Execution Validation v1

Validates the **complete existing** highway-merge planning-side chain as one
composed system. **No new algorithm, no policy retune, no new planner, no new
mission behaviour, no code fix** — the composed test found zero bugs.

## Test boundary

**Planning-side end-to-end**, not full perception-to-control. The canonical
input boundary is `DynamicObjectRiskArray`:

```
synthetic DynamicObjectRiskArray
  -> ad_highway_merge_gap_risk        (REAL node)
  -> ad_highway_merge_gap_response    (REAL node)
  -> ad_planner                       (REAL node; highway merge response
                                       integration + mission primitive + online
                                       reference path all enabled; PRODUCTION
                                       base path 2026_molit_comp_global_path.txt)
  -> existing FollowGlobalPath -> existing PathTrackingController(s)
  -> the single /ad/control/command publisher
```

Tracking → prediction → Dynamic Object Risk producer are **not** run — they have
their own runtime validation and the real `ad_dynamic_object_risk_node` OOMs on
this host (documented since the Cut-in Risk work). Per PR guidance the merge
chain's canonical boundary is `DynamicObjectRisk`; a real tracker is not
required here.

The chain is **connected, not mocked**: the real `ad_highway_merge_gap_response`
node publishes `/ad/planning/highway_merge_gap_response` and the real
`ad_planner` (integration on) subscribes it (asserted:
`count_subscribers >= 2`, `count_publishers >= 1`). The test never publishes
`HighwayMergeGapResponse` or `highway_merge_authorized` for the main
traffic → mission chain, and never recomputes a policy equation — it drives
input trajectories and asserts the composed outputs.

## Real vs synthetic components

| component | real? |
| --- | --- |
| `DynamicObjectRiskArray` | synthetic (deterministic script) |
| `ad_highway_merge_gap_risk` | **real node**, real corridor + shipped merge geometry |
| `ad_highway_merge_gap_response` | **real node** |
| `ad_planner` (integration + mission + reference path) | **real node**, production base path |
| route corridor / merge geometry / ramp poses | **real** (`route_corridor.json`, `highway_merge.json`, PR #25 `route:0:left:1` samples) |
| FollowGlobalPath / PathTrackingController / Stanley | **real, unchanged** |

**Real MORAI run: NO** — no MORAI simulator / gRPC runtime in this environment
(`import grpc` fails). Deterministic ROS replay only. The authored scenario JSON
`ad_data/scenarios/kcity_highway_ego_onramp_v1.json` (PR #25) is ready for a
future real run via `ad_morai_scenario_reset`.

## Complete chain diagram

```
 traffic (DynamicObjectRiskArray)
   │  per-object route-relative facts, predicted states
   ▼
 Highway Merge Gap Risk ──► HighwayMergeGapRiskArray (ego route_s, front/rear at merge, coverage, predicted min gap)
   ▼
 Highway Merge Gap Response ──► HighwayMergeGapResponse {action: MERGE_READY|WAIT|HOLD, reason, gaps, headways}
   ▼
 Highway Merge Response Integration (in ad_planner)
   ├─ WAIT/HOLD ──► longitudinal target cap via the existing 6th target_speed_mps arg
   └─ fresh active MERGE_READY ──► revocable PlannerContext::highway_merge_authorized
   ▼
 Highway Merge Mission Primitive (in ad_planner, before the BT tick)
   consumes highway_merge_authorized + ego route_s ──► state INACTIVE/APPROACH/WAITING/AUTHORIZED/COMMITTED/COMPLETE
   ▼
 Highway Merge Online Reference Path (in ad_planner)
   consumes mission state + route:0:left:1 proximity ──► selects merge-reference PathTrackingController vs production
   ▼
 FollowGlobalPath ──► selected PathTrackingController.update() ──► ONE ControllerResult ──► publish_command() ──► /ad/control/command
```

## Deterministic traffic scenario

Ego swept through real `route:0:left:1` acceleration-lane poses at 8 m/s.
Mainline traffic on `route:0`:

* **FAST_REAR**: a front vehicle at `route_s ~1200` @ 24 m/s + a rear vehicle
  ~12 m behind the ego's route projection @ 33 m/s (closing hard).
* **CLEAR**: zero relevant objects (Phase 8 — the deterministic positive
  `MERGE_READY` case; see "traffic-present MERGE_READY" below).

## Full causal traffic → response → authorization → mission trace

| step | ego (`route:0:left:1` s) | traffic | GapResponse | reason | `merge_authorized` | mission | `reference_active` | merge cap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A unsafe rear | 1120 | FAST_REAR | **WAIT** | REAR_CLOSING (4) | false | **WAITING** | true | −1 (none) |
| B gap opens | 1150 | CLEAR | **MERGE_READY** | CLEAR_GAP (1) | **true** | **AUTHORIZED** | true | −1 |
| B2 auth @ pose A | 1120 | CLEAR | MERGE_READY | CLEAR_GAP | true | AUTHORIZED | true | −1 |
| C pre-commit revoke | 1160 | FAST_REAR | **WAIT** | REAR_CLOSING (4) | **false** | **WAITING** | true | −1 |
| D reauthorize | 1180 | CLEAR | **MERGE_READY** | CLEAR_GAP | **true** | **AUTHORIZED** | true | −1 |
| E commit | 1220 | CLEAR | MERGE_READY | CLEAR_GAP | true | **COMMITTED** | true | −1 |
| F post-commit revoke | 1250 | FAST_REAR | **WAIT** | PREDICTED_ROUTE_CONFLICT (6) | **false** | **COMMITTED** (held) | true | **10.36 m/s** |
| F2 post-commit HOLD | 1283 | FAST_REAR | **HOLD** | ALONGSIDE (5) | false | **COMMITTED** (held) | true | **0.0 m/s** |
| G complete | `route:0` 1295 | CLEAR | inactive | NONE | false | **COMPLETE** | **false** | −1 |
| H release | `route:0` 1330 | CLEAR | inactive | NONE | false | **INACTIVE** | false | −1 |

The causal linkage is real: `FAST_REAR` → `REAR_CLOSING` → `WAIT` →
`highway_merge_authorized` false → mission `WAITING`; `CLEAR` → `CLEAR_GAP` →
`MERGE_READY` → `highway_merge_authorized` true → mission `AUTHORIZED`. Every
arrow is a separate real node.

## Full reference-active trace (steps A…J)

`[true, true, true, true, true, true, true, true, false, false, true, false]` —
active for every ramp-participation step (A–F2), `false` at `COMPLETE` (G) and
`INACTIVE` (H), and correctly `true` again for the cut-in composition step (I,
ramp WAITING) then `false` after the mid-merge reset (J).

## Full final-target / longitudinal trace

* `merge_speed_limit` observability (`-1` = no merge cap): `-1` while
  `MERGE_READY` or inactive; **10.36 m/s** at F (`WAIT`,
  `= sqrt(2 · 1.8 · available)` with `available ≈ 33 m`); **0.0** at F2
  (`HOLD`).
* At F2 the command shows braking (`brake > 0` / `accel ≤ 0`) — the `0.0`
  longitudinal cap flows through the **existing** external-constraint /
  `PathTrackingController` path; no brake is published by the merge code.

## Key question answers

1. **Unsafe traffic keeps WAIT/HOLD** — yes (A, C, F, F2).
2. **Valid gap → MERGE_READY** — yes (B, D; zero-relevant-object frame).
3. **Fresh MERGE_READY → `merge_authorized`** — yes (B, D: `authorized` true).
4. **Mission WAITING → AUTHORIZED → COMMITTED → COMPLETE** — yes (state trace
   `[2,3,3,2,3,4,4,4,5,0]` over A–H).
5. **Online ramp reference active throughout the ramp portion** — yes
   (`reference_active` true A–F2).
6. **Pre-commit auth loss revokes AUTHORIZED without changing the lateral
   reference** — yes (C: `AUTHORIZED → WAITING`, `reference_active` stays true;
   B2 vs A: mean |steering| 0.0038 vs 0.0038 at the identical pose).
7. **Post-commit auth loss preserves COMMITTED + merge reference** — yes (F, F2:
   mission `COMMITTED`, `authorized` false, `reference_active` true).
8. **COMPLETE handoff smooth** — yes (see below).
9. **Cut-in / roundabout / traffic-stop / collision keep higher-priority
   authority** — yes (step I: a cut-in `HOLD` during a merge `WAITING` frame
   still brakes; `test_behavior_tree` unchanged — `CollisionRecovery`,
   `TrafficStop`, `FailSafeBrake` priority above `FollowGlobalPath`).
10. **Exactly one CtrlCmd publisher** — yes
    (`count_publishers("/ad/control/command") == 1` throughout).

## COMPLETE handoff metrics

At `COMPLETE` the ego is on `route:0` and the merge-reference continuation past
merge-complete **is** `route:0` (same corridor points), so the switch from the
merge controller back to the (always-warm) production controller is
geometric-noise seamless:

| | just before (E, `route:0:left:1` s1220, merge controller) | at COMPLETE (G, `route:0` s1295, production controller) |
| --- | --- | --- |
| mean \|steering\| | 0.0084 rad | **0.0026 rad** |
| ego lateral offset to `route:0` | 3.23 m | **0.13 m** |

Empirical continuity threshold, from the observed controller noise on this
near-straight stretch: mean |steering| stays **< 0.02 rad** and the command is
valid every frame across the handoff. No aggressive universal threshold is
claimed.

## Second-controller architecture (PR #26) — audited, confirmed

With `enable_highway_merge_reference_path` on there are **two internal
`PathTrackingController` instances**: the production one (`route_`) and the
merge-reference one (`route:0:left:1` → `route:0`). Audit:

| property | result |
| --- | --- |
| duplicate subscriptions | **no** — the merge controller is a plain object, no ROS handles |
| duplicate publishers | **no** |
| duplicated PID / route-progress / launch-ramp state | **yes, by design** — the merge controller has its own |
| same backend / gains / speed profile / PID | **yes** — both from `make_path_tracking_controller(path_tracking_backend_, route, RosControllerParameterProvider(*this))` |
| production controller warm-updated every active tick | **yes** — `run_path_tracking()` always calls `path_tracking_->update(...)`; its result is discarded while the merge reference is selected, so its route progress / PID / launch-ramp never go stale and the COMPLETE handoff never re-triggers the launch ramp |

Correct claim: **one control publisher, one selected `ControllerResult` per
tick, with two internal controller instances while merge-reference mode is
enabled.** No blending or averaging of steering — `run_path_tracking()`
`remember()`s exactly one controller's result (the merge one iff selected and
valid, else the production one; a runtime-invalid merge result falls back to the
warm production controller for that tick with a throttled WARN).

## Timing (loopback, not internal compute cost)

25-sample loopback at a mainline pose (reference **inactive**) vs a ramp pose
(reference **active**):

| metric | reference inactive | reference active |
| --- | --- | --- |
| risk publish → gap-risk frame (ms) | median 6.15 / p95 7.01 / max 8.12 | median 6.15 / p95 6.61 / max 8.21 |
| risk publish → gap-response (ms) | median 6.23 / p95 7.10 / max 8.36 | median 6.25 / p95 6.70 / max 8.44 |
| publish → next command (ms) | median 42.95 / p95 44.04 / max 46.05 | median 43.73 / p95 45.43 / max 51.94 |

The `publish → command` figure is a **loopback** dominated by the test driver's
publish cadence + the planner's ~20 Hz tick, not internal compute. The
load-bearing result: **reference-active and reference-inactive are
statistically identical** — the deliberate second `update()` per tick (PR #26)
costs nothing measurable. No hard real-time claim.

**Planner loop frequency**: `/ad/control/command` published continuously
through WAITING / AUTHORIZED / COMMITTED with the reference active — ~1180
commands over the run, no missed cycles observed, sustained against the ~20 Hz
input.

**E2E `DynamicObjectRiskArray` → resulting-state command latency**: **not
measured** — the driver publishes risk + odometry + planner inputs together and
the planner ticks on its own timer, so a clean per-frame correlation across four
async nodes is ambiguous; not invented.

## Validation metrics (representative full run)

Frame counts are wall-clock-settle dependent (the driver holds each step for a
fixed dwell); they vary ~±1 % run to run. State/reference/authorization traces
and the qualitative counts (0 invalid, 0 NaN/Inf, 1 publisher) are
deterministic and asserted.

| metric | value |
| --- | --- |
| gap-risk output frames | ~2700 |
| gap-response output frames | ~2680 |
| planner command frames | ~1180 |
| MERGE_READY responses | ~1145 |
| WAIT responses | ~580 |
| HOLD responses | ~145 |
| authorization-true frames | ~465 |
| reference-active frames | ~735 |
| mission state trace (A–J) | `[2,3,3,2,3,4,4,4,5,0,2,0]` (asserted) |
| invalid command frames | 0 (asserted) |
| NaN / Inf / exceptions (accepted frames) | 0 (asserted) |
| `/ad/control/command` publishers | 1 (asserted) |

## Pre-commit revocation

At `AUTHORIZED → WAITING` (step C, still on the ramp before commit): the
`reference_active` stays `true`, there is no reference swap to `route:0`, and
mean |steering| stays small (0.009 rad). The longitudinal `WAIT` cap handles the
revoked merge; the lateral centerline does not change.

## Post-commit revocation

At `COMMITTED` + response `WAIT`/`HOLD` (F, F2): `merge_authorized` goes `false`,
the mission **stays `COMMITTED`**, `reference_active` stays `true` — the lateral
merge is not reversed. **Independent longitudinal safety still applies**: F
carries a `10.36 m/s` merge cap, F2 a `0.0` cap that brakes the vehicle, both
through the existing external-constraint path. This is the mandatory cross-layer
separation: mission commitment is lateral irreversibility; longitudinal
progression stays governed by the response.

## Stale response (documented, not changed)

Not re-exercised live in this PR (the driver publishes continuously); the
contract is inherited from the response integration (PR #23):

* **Pre-commit stale** → `merge_authorized` false → mission `WAITING`; the merge
  reference stays selected while the ego is physically on the ramp and the
  mission stays `WAITING`; the merge longitudinal cap **expires** (steady-
  receipt-time freshness, `highway_merge_response_max_age_s` 0.5 s).
* **Post-commit stale** → `merge_authorized` false → mission stays `COMMITTED`,
  reference stays active; the merge-specific longitudinal cap **expires**.
  **Known limitation (unchanged): a stale `HOLD` expires rather than
  latching** — a downstream assumption documented since PR #22. Not retuned
  here.

## Reset mid-merge

Step J teleports the ego backward (`route:0` s1330 → s400, a > 3.5 m route_s
rollback): mission → `INACTIVE`, the reference-selection latch clears, the
production path is selected, no stale `COMMITTED` / authorization / merge
reference.

## Invalid merge-controller result

The PR #26 fallback (merge reference selected but its controller returns
invalid) is covered by `test_planner_highway_merge_reference_path` /
`run_path_tracking()`'s throttled-WARN fallback to the warm production
controller; not re-exercised here (it needs an invasive hook to force, and the
built route always spans well past COMPLETE).

## Default-off / production lock

All highway-merge features remain opt-in, default off. `planner.yaml` defaults,
`2026_molit_comp_global_path.txt` (SHA-256 `50658991…cc05`), and the production
BehaviorTree XML are unchanged. No new BT action. Feature default-off →
`test_planner_launch::test_highway_merge_reference_path_is_opt_in_and_default_off`
proves no second controller / publisher is built; `test_behavior_tree`
`registered_node_ids().size() == 8` unchanged.

## Traffic-present MERGE_READY (Phase 9) — why zero-object is the realistic case

A ramp ego at 8 m/s against mainline traffic at 24–33 m/s: any mainline vehicle
inside the relevance window overtakes the slow ramp ego within the prediction
horizon, so the predicted-route-gap / rear-closing tiers correctly fire `WAIT`
(PR #25 found the same). A "safe" traffic-present frame would need mainline
speeds near 8 m/s (not a highway) or vehicles outside the relevance window
(= zero relevant objects). The **zero-relevant-object CLEAR_GAP frame is
therefore the realistic traffic-present-but-clear case**, and it is what the
positive steps use. No threshold was changed to force a traffic-present
`MERGE_READY`.

## Bugs found / fixed

**None.** The composed chain behaved exactly as the component designs (PR #21–
#26) specify on the first composed run. No code was changed.

## Files

* `ad_planner/test/test_highway_merge_end_to_end.py` (new) — the composed
  launch test
* `ad_planner/CMakeLists.txt` — test registration
* `docs/planning/highway_merge_end_to_end_validation_v1.md` (this file),
  `docs/agent/STATUS.md`

## Regression

`test_highway_merge_reference_path` (10) + `_golden` (6), `test_highway_merge_mission`
(27), `test_highway_merge_speed_constraint`, `test_external_speed_limit`,
`test_cut_in_speed_constraint`, `test_roundabout_speed_constraint`,
`test_behavior_tree`, `test_planner_launch` (30) — all pass. Highway-merge
launch-test regression (`ctest`): `test_planner_highway_merge_mission`,
`test_planner_highway_merge_reference_path`, `test_highway_merge_ego_ramp_pipeline`,
`test_highway_merge_gap_risk_runtime`, `test_highway_merge_end_to_end` — 5/5
pass. `ad_control` (Stanley / Profile-Stanley / PID) is **byte-unchanged**
(`git diff origin/main -- ad_control` empty; no `test_stanley` run needed — the
controller math is untouched by construction). Isolated `colcon build
--packages-select ad_planner --symlink-install` clean.

## Conclusion

**This closes the planning-side highway-merge implementation / validation
milestone** (pending real simulator / vehicle validation). The
planning-side chain — Dynamic Object Risk → Gap Risk → Gap Response → Response
Integration → Mission Primitive → Online Reference Path → existing
FollowGlobalPath — has been validated as one composed system on the
source-grounded ego-on-ramp scenario. Target-mainline traffic drives the facts,
the response converts them to `WAIT`/`HOLD`/`MERGE_READY`, fresh readiness drives
revocable authorization, the mission advances
`WAITING → AUTHORIZED → COMMITTED → COMPLETE`, the online
`route:0:left:1 → route:0` reference keeps the existing controller on the real
acceleration-lane geometry, and authorization loss before/after commit follows
the intended semantics. Longitudinal safety stays independent, one selected
controller result feeds exactly one CtrlCmd publisher.

**No further highway-merge planning feature work is recommended until real
MORAI / vehicle testing exposes a concrete issue.**

## Recommended next task

**MORAI Tracking Dataset Factory v1** — build a deterministic scenario / reset /
data-capture pipeline that records LiDAR, ego pose / TF, actor ground-truth
IDs / poses / 3D boxes, timestamps, scenario seed and manifest for repeated
tracking, CenterPoint and KalmanNet development, without changing the frozen
tracking research claims.
