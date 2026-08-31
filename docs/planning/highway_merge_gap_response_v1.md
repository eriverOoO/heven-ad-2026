# Highway Merge Gap Response v1

Highway Merge Gap Response v1 is a conservative, planner-facing **merge-decision
advisory**. It converts the policy-free `HighwayMergeGapRisk` conflict-geometry
facts into `MERGE_READY`, `WAIT`, or `HOLD`. It does not control the vehicle and
has **no production planner consumer** in v1.

```text
/ad/planning/highway_merge_gap_risks   (HighwayMergeGapRiskArray, map)
                         |
                         v
             ad_highway_merge_gap_response
                         |
                         v
/ad/planning/highway_merge_gap_response  (HighwayMergeGapResponse, map)
                         |
                         v
              future mission / planner integration
```

The standalone `highway_merge_gap_response.launch.py` is opt-in;
`planner.launch.py highway_merge_gap_response:=true` also starts the
`ad_highway_merge_gap_risk` node it consumes. Input diagnostics are published on
`/ad/planning/highway_merge_gap_response/diagnostics`. There is **NO lane-change
command, steering, path, trajectory, gear, brake, throttle, acceleration / speed
request, `CtrlCmd`, or production planner consumer** in this version.

## Action semantics

- **MERGE_READY**: the available merge-prediction evidence supports an
  acceptable front **and** rear target-lane gap under the explicit v1 policy. It
  does **not** mean "start steering now" -- a mission layer still owns the
  maneuver decision.
- **WAIT**: the merge cannot yet be justified, but more than the comfortable
  stopping distance remains before the decision boundary, so the ego can keep
  approaching and collecting evidence.
- **HOLD**: the merge cannot be justified and the comfortable stopping margin to
  the decision boundary has been consumed. **This is an advisory classification,
  not a brake command and not `speed = 0`.** Mapping HOLD to a longitudinal
  constraint belongs to a later planner/mission integration task.

## Applicability

The response is active while the ego is in the merge approach / merge zone for
the current forward traversal:

- `ego_route_distance_to_merge_m < 0` (past the merge reference station) ->
  `active = false`, non-restrictive `ACTION_MERGE_READY` enum, `REASON_NONE`.
- `ego_route_distance_to_zone_entry_m > maximum_approach_distance_m` (far before
  the zone) -> `active = false`. `maximum_approach_distance_m` defaults to
  **400.0 m** and must stay `<=` the risk node's own
  `maximum_ego_approach_distance_m` so a far ego is never reported active here.
- otherwise -> `active = true`.

A consumer must check `active`; the inactive enum value is not a merge
authorization.

## Exact MERGE_READY rule

`MERGE_READY` requires **all** of:

1. a fresh, strictly increasing, finite source frame in `map` for
   `kcity_highway_onramp`;
2. the response is applicable and `ego_merge_timing_valid == true` and the ego
   route speed is above `stopped_speed_threshold_mps`;
3. no relevant object was omitted over the policy budget;
4. every relevant object has `prediction_covers_merge_time == true` (the
   fallback constant-speed `delta_s_at_merge_m` never authorizes a merge --
   Phase 26 regression);
5. no relevant object is `is_alongside_at_merge`;
6. every relevant object with a valid `predicted_min_route_gap_m` satisfies
   `predicted_min_route_gap_m >= minimum_predicted_route_gap_m`;
7. no relevant object currently behind the ego (`delta_s_now_m < 0`) that is
   `longitudinal_gap_closing` has `time_to_route_coincidence_s <
   minimum_rear_closing_time_s` (an invalid coincidence time on a closing rear
   object also fails);
8. the nearest object **behind** the merge reference at the ego merge time (if
   any) has rear time headway `-delta_s_at_merge_m / max(object longitudinal
   speed, speed_epsilon_mps) >= minimum_rear_time_headway_s`;
9. the nearest object **ahead** of the merge reference at the ego merge time (if
   any) has front time headway `delta_s_at_merge_m / max(ego_speed_mps,
   speed_epsilon_mps) >= minimum_front_time_headway_s`.

A valid frame with **zero relevant objects** and valid ego timing is
`MERGE_READY` / `REASON_CLEAR_GAP`. Absence of a front object (or a rear object)
satisfies that side vacuously. No single total merge span (`merge_gap_m` in the
risk array) can override a failed front or rear condition -- `merge_gap_m` is
deliberately **not** copied into this message; a consumer reads the two signed
`front_gap_m` / `rear_gap_m` facts individually.

Front and rear are selected from the per-object `is_ahead_at_merge` /
`is_behind_at_merge` relations, never recomputed from raw Cartesian positions,
so a curved-geometry object that is ahead in route station while behind in
`base_link` x is still treated as a front gap.

## Front / rear / closing policy and threshold provenance

No existing planner following-headway parameter exists to reuse, so these are
**competition-v1 tuning values, not a universal safety guarantee**
(`config/highway_merge_gap_response.yaml`):

| parameter | value | boundary | provenance |
| --- | --- | --- | --- |
| `minimum_front_time_headway_s` | 1.5 | inclusive `>=` | standard lower-bound highway following headway; ~12 m ahead at the merge instant at the audited 8 m/s highway-entry approach |
| `minimum_rear_time_headway_s` | 2.0 | inclusive `>=` | larger than the front value: the mainline rear vehicle has right of way and may not decelerate for the merge |
| `minimum_rear_closing_time_s` | 3.0 | inclusive `>=` | route-coincidence time margin for a rear vehicle that is closing on the ego route station |
| `minimum_predicted_route_gap_m` | 6.0 | inclusive `>=` | ~one IONIQ 5 length (4.635 m) plus margin; also mirrors `merge_standoff_m` |
| `merge_standoff_m` | 6.0 | -- | cut-in response `longitudinal_standoff_m` / roundabout response `entry_standoff_m` (vehicle front extent plus margin); the WAIT/HOLD boundary sits `merge_standoff_m` before the merge reference station, never past merge completion |
| `comfortable_deceleration_mps2` | 1.8 | -- | `perception.braking_deceleration_mps2` in `config/planner.yaml` |
| `stopped_speed_threshold_mps` / `speed_epsilon_mps` | 0.5 | -- | mirrors the upstream Highway Merge Gap Risk ego-speed epsilon |

The rear-closing check is a **current-time** fact and is gated on `delta_s_now_m
< 0`. `longitudinal_gap_closing` is symmetric, so a slow front vehicle the ego
is overtaking never trips `REASON_REAR_CLOSING`; that case is caught (if at all)
by the predicted route-gap criterion, which sits above the front-gap tier.

Front / rear gap and headway, the limiting rear-closing object and its
coincidence time, and the smallest predicted minimum route gap are all reported
on every active frame regardless of the action, for transparency.

## WAIT / HOLD dynamics

For a non-`MERGE_READY` active frame:

```text
available_distance_m = max(0, ego_route_distance_to_merge_m - merge_standoff_m)
comfortable_stop_distance_m = ego_speed_mps^2 / (2 * comfortable_deceleration_mps2)

stopped ego                                          -> HOLD
available_distance_m > comfortable_stop_distance_m   -> WAIT
otherwise                                            -> HOLD
```

The strict `>` boundary preserves the full comfortable stopping distance. A
stopped ego (route speed below `stopped_speed_threshold_mps`, so
`ego_merge_timing_valid` is false) is always `HOLD` /
`REASON_INVALID_EGO_STATE`; it does not fabricate a merge ETA. A **moving** ego
with invalid timing (only possible on a misconfigured
`maximum_approach_distance_m` larger than the risk node's bound) never earns
`MERGE_READY` and takes `WAIT` / `HOLD` by the same distance rule --
`REASON_DECISION_BOUNDARY` was considered and folded into the `HOLD` action plus
the `available_distance_m` / `comfortable_stop_distance_m` fields rather than a
separate reason.

## Multi-object arbitration and reason selection

`MERGE_READY` is an all-object conjunction; gaps are never averaged. For a
non-ready frame one limiting object is chosen deterministically by reason
precedence, then the most restrictive metric within that reason, then the
lexicographically smallest UUID:

1. `REASON_ALONGSIDE` -- any relevant object alongside at merge;
2. `REASON_INSUFFICIENT_PREDICTION` -- coverage false, invalid
   `delta_s_at_merge`, or no predicted route gap;
3. `REASON_PREDICTED_ROUTE_CONFLICT` -- predicted minimum route gap below policy;
4. `REASON_REAR_CLOSING` -- a closing rear object inside the coincidence-time
   policy;
5. `REASON_REAR_GAP` -- rear time headway below policy;
6. `REASON_FRONT_GAP` -- front time headway below policy.

If there is no blocking object but the frame is still not ready (stopped /
invalid ego timing, or an over-budget relevant object) the reason is
`REASON_INVALID_EGO_STATE` or `REASON_INSUFFICIENT_PREDICTION` respectively and
`complete_prediction_coverage` is reported false.

## Hysteresis

**None.** The policy is stateless. Bounded threshold tests establish the exact
inclusive boundaries and an approaching ego progresses monotonically
`MERGE_READY -> WAIT -> HOLD`. The canonical replay showed no
`MERGE_READY <-> WAIT` flicker (it never reaches `MERGE_READY`). A later
integration may add a `merge_ready_confirmation_frames` delay on `MERGE_READY`
only (never on `WAIT` / `HOLD`) if vehicle replay demonstrates material flicker.

## Input rejection and expiry

Malformed / non-positive / duplicate / backward stamp (a backward jump beyond
`clock_rollback_threshold_ns` = 0.5 s is treated as a sim-time reset and clears
the latch), future stamp, stale stamp beyond `maximum_input_age_s`, wrong
`frame_id`, wrong `merge_zone_id`, non-finite ego facts, an inconsistent
`relevant_object_count`, and per-object inconsistencies (relation flags not
mutually exclusive; a relation flag set without a valid `delta_s_at_merge`;
`delta_s_at_merge_valid` or `prediction_covers_merge_time` true while
`ego_merge_timing_valid` is false; `longitudinal_gap_closing` with a
non-positive closing speed; `time_to_route_coincidence_valid` without
`longitudinal_gap_closing`; a negative predicted route gap) reject the whole
frame -- warn + skip, no publish, the prior stamp latch is not advanced, and no
`MERGE_READY` is ever reused. `delta_s_at_merge_valid == true` with
`prediction_covers_merge_time == false` is a **legitimate** state (the fallback
extrapolation) and is not rejected. Exactly one response is published per
accepted input; if the input stream stops, the output stops. A future consumer
must enforce its own freshness timeout (no greater than `maximum_input_age_s`,
default 0.5 s) and must not latch `MERGE_READY`.

## Configuration

`config/highway_merge_gap_response.yaml` -- topics, `expected_frame_id: map`,
`merge_zone_id: kcity_highway_onramp`, `maximum_input_age_s: 0.5`,
`maximum_approach_distance_m: 400.0`, the seven policy values above, and
`maximum_relevant_objects` / `runtime_summary_interval_frames`. There are no
scores, weights, actuator gains, requested speeds, or hysteresis parameters.

## Deterministic validation

`test_highway_merge_gap_response` -- **32 gtests** (pure policy + frame
builder): zero-object MERGE_READY; safe front / rear / both; front / rear gap
too small; rear closing too quick; alongside; insufficient coverage; fallback
extrapolation looks safe but coverage false -> not ready; predicted route gap
too small; large total span with unsafe rear -> not ready; multiple safe;
one-unsafe-among-many with a deterministic limiting UUID; reason precedence
(alongside > coverage > route); front / rear / rear-closing threshold
boundaries; approach WAIT->HOLD monotonicity; stopped ego; moving ego with
invalid timing; far ego inactive; ego past merge inactive; malformed / stale /
future / duplicate / backward rejection; large clock rollback clears the latch;
determinism + finiteness; gap-policy monotonicity; curved-geometry front
relation; slow front object the ego overtakes is not rear-closing; over-budget
forbids MERGE_READY; negative ego speed rejected.

`test_interface_contract.py` (12/12) -- declaration stable + conservative
defaults + actuator / speed / lane-change / CtrlCmd token disjoint set.

`test_highway_merge_gap_response_launch.py` -- config is small / physical / has
no actuator or speed request, `maximum_approach_distance_m <= 400`, standalone
launch starts only this node. `test_planner_launch.py` -- declared arg, opt-in
composition, force-start of the risk node, default off.

`test_highway_merge_gap_response_policy.py` -- live node driven by synthetic
`HighwayMergeGapRiskArray`: **23 inputs -> 22 responses, 1 rejected**
(stale/backward stamp); **5 MERGE_READY / 13 WAIT / 3 HOLD / 1 inactive**;
reason counts CLEAR_GAP 5, FRONT_GAP 3, REAR_GAP 7,
REAR_CLOSING 2, ALONGSIDE 1, PREDICTED_ROUTE_CONFLICT 1, INSUFFICIENT_PREDICTION
1, INVALID_EGO_STATE 1. Representative MERGE_READY: front gap 45.0 m / headway
5.625 s, rear gap 70.0 m / headway 7.0 s, predicted min route gap 25.0 m.
Representative WAIT (200 m to merge): available 194.0 m, comfortable stop
17.78 m. Representative HOLD (8 m to merge): available 2.0 m, comfortable stop
17.78 m. Insufficient-prediction example: fallback `delta_s_at_merge_m` = 200 m,
`complete_prediction_coverage` false, not ready. Rear-closing example: rear gap
60 m, coincidence time 2.0 s. The WAIT->HOLD approach sequence and the
unsafe-rear-closing -> rear-clears -> MERGE_READY sequence are both exercised.
0 NaN / Inf / exceptions.

`test_highway_merge_gap_risk_runtime.py` -- the existing canonical
`DynamicObjectRisk -> HighwayMergeGapRisk` replay now also runs
`-> HighwayMergeGapResponse`. 6 responses: **0 MERGE_READY / 5 WAIT / 1 inactive
/ 0 rejected**; reasons NONE (inactive) 1, PREDICTED_ROUTE_CONFLICT 3,
INSUFFICIENT_PREDICTION 2. The replay carries prediction-uncovered relevant
objects and two objects whose predictions cross the ego rollout, so it **never
earns MERGE_READY**, and the coverage policy is not weakened to force one
(Phase 39). All response fields finite.

Internal response callback latency (from
`HIGHWAY_MERGE_GAP_RESPONSE_RUNTIME_SUMMARY` in the chained replay) is
**0.0039 / 0.030 / 0.030 ms** median / p95 / max -- three orders of magnitude
below the ~16 ms Gap Risk geometry stage, since the policy is a single pass over
the relevant objects with no projection. The chained replay's publish-to-receive
latency (`~18 ms` median) is measured separately and is DDS + the 20 ms probe
poll, not compute. No hard-real-time claim is made.

## Limitations

Ego merge timing uses the upstream constant-current-speed ETA, so a stopped ego
cannot earn `MERGE_READY` in v1. `rear_time_headway_s` divides the at-merge rear
gap by `object_longitudinal_speed_mps`, which the risk layer resolves from the
object's **current** velocity, not its speed at the merge instant -- exact for a
steady mainline vehicle, an approximation under acceleration. The fixed merge
zone and single-lap forward-route contract are those of Highway Merge Gap Risk
v1. There is no hysteresis, no uncertainty-expanded gap policy, and no planner /
control integration -- `MERGE_READY` is a fact for a future mission layer, not a
maneuver trigger.

## Recommended next task

Highway Merge Response Integration v1 -- consume fresh `HighwayMergeGapResponse`
advisories at the existing highway mission / planner boundary: `WAIT` / `HOLD`
may constrain longitudinal progression while `MERGE_READY` exposes a
merge-authorization fact to the mission layer, without directly commanding
steering or creating a second `CtrlCmd` publisher.
