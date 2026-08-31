# Highway Merge Gap Risk v1

Policy-free highway-merge conflict-geometry facts. Consumes the canonical
`ad_interfaces/DynamicObjectRiskArray` (base_link, with discrete predicted
centroids) plus the ego route state and a fixed source-grounded merge zone, and
publishes `/ad/planning/highway_merge_gap_risks`
(`ad_interfaces/HighwayMergeGapRiskArray`).

**Facts only.** No `merge` / `wait` / `go` / `yield` / `hold` / `release` /
`lane_change` / `accelerate` / `decelerate` / `merge_allowed` / `merge_safe` /
`accepted_gap` / `safe_gap` / `requested_speed` / `brake` / `throttle` /
`steering` / `risk_score` field. No planner consumer, no CtrlCmd, no
BehaviorTree change. A response policy applies a gap threshold later.

## Merge geometry (source-grounded)

`ad_planner/config/highway_merge.json` names one merge zone,
`kcity_highway_onramp`:

* **Source lane** `route:0:left:1` — a left-adjacent acceleration lane in the
  checksum-verified `ad_data/map/route_corridor.json`. Its single source link
  `A2256W000409` is waypoint 0 of the `ad_morai_bridge_dev` **`kcity-highway`**
  actor preset (`actor_presets.yaml` / `actor_presets.provenance.yaml`, pinned
  `link_set_sha256 5ce0fd57…` — identical to the corridor's own
  `source_sha256["link_set.json"]`). Its lateral offset to primary route
  `route:0` falls from ~+3.94 m at its first station to 0.0 m at its last: it
  **tapers into `route:0`**. `route:0`'s posted speed limit steps
  `11.11 -> 33.33 m/s` at s ≈ 1119, the on-ramp / highway entry.
* **Target lane** `route:0` — the post-merge primary route **is** the mainline
  through lane; no separate physical target-lane centerline exists past the
  merge point (`route:0:left:2` is a further-left parallel lane). Ego and
  objects are both projected onto `route:0`; all longitudinal ordering / gaps
  are `route:0` stations. This choice is documented in the config `provenance`.
* `route_s_zone_entry_m = 1118.7418`, `route_s_merge_complete_m = 1286.1546` —
  the first / last `route_s_m` of `route:0:left:1` in the corridor cache. The
  node re-derives them from that lane at startup and **fails to start** on a
  mismatch beyond `merge_geometry_consistency_margin_m` (2 m), on an unknown
  lane id, on a target lane that does not span the zone, or on a degenerate
  station window.
* `merge_reference_route_s_m = route_s_merge_complete_m` — the station by which
  the acceleration lane ceases to exist and the merge is physically resolved
  (target-lane lateral offset there is 0.0). `HighwayMergeGapRisk.delta_s_at_merge_m`
  is measured against this station.

No provenance-verified MORAI highway-merge bag exists (the `kcity-highway`
preset is a pinned link set with no recorded actor state), so runtime
validation is a deterministic canonical ROS replay on the real geometry.

## Ego merge timing (a fact, not a trajectory)

Constant-current-speed arrival of the ego onto `merge_reference_route_s_m`
along the target corridor:
`ego_merge_time_s = max(0, merge_reference_s - ego_route_s) / ego_route_speed`
where `ego_route_s` / `ego_route_speed` are `project_to_frenet(target_lane, ego)`
`s_m` / `s_dot_mps`. `ego_merge_timing_valid` is false when the ego route speed
is below `ego_speed_epsilon_mps` (0.5), when the ego has passed
`merge_reference_route_s_m` on this lap (single-lap, forward-only), or when the
route distance to the zone entry exceeds `maximum_ego_approach_distance_m`
(400). `ego_in_merge_zone_now` = the ego station is between the zone entry and
`merge_reference_route_s_m`.

## Per-object facts

Projection uses the **existing** `project_to_frenet` (no second Frenet
implementation). The **ego** projects onto the full target lane (it drives the
whole route). **Objects** project onto a station-bounded **window** of `route:0`
around the merge zone (covering the ego approach bound plus the relevance
window); this keeps per-frame cost `O(objects · samples · window points)` and
independent of the 2184 m route-loop length, while one full-lane ego projection
per frame is negligible. An object that projects outside the window clamps to a
window end (large lateral offset) and is correctly reported
`relevant_to_merge = false`; a non-relevant object then carries only its
relevance flags, `predicted_corridor_entry_*`, `prediction_horizon_s`, and the
copied generic risk facts — every route-station / speed / gap field is zeroed
(the window-edge-clamped numbers would not be meaningful).

* `relevant_to_merge` — the object currently occupies the target corridor
  (within its lateral half-width + `target_corridor_lateral_margin_m`) **and**
  its route station is within
  `[zone_entry - relevant_rear_window_m, merge_complete + relevant_front_window_m]`
  (150 m / 120 m; generous "which objects to report" bounds, not a policy
  threshold — a fast closing vehicle can be well behind the zone over a merge
  horizon), **or** a predicted centroid does so. The longitudinal window test
  runs before any merge-time field is computed (loop-seam / far-object guard);
  a non-relevant object carries only projection context + generic facts.
* `object_route_s_m`, `object_lateral_offset_m` — current projection.
* `object_in_target_corridor_now`, `predicted_to_enter_target_corridor` +
  `predicted_corridor_entry_time_s` (first predicted centroid inside the
  corridor and window). The MORAI `kcity-highway` NPC — currently ~+3.9 m left
  on `route:0:left:1`, tapering to 0 — is `object_in_target_corridor_now =
  false` but `predicted_to_enter_target_corridor = true`.
* `delta_s_now_m` — `object_route_s - ego_route_s` on the target corridor.
* `object_longitudinal_speed_mps` — the object's **current** velocity
  (`vx_rel + ego_speed`, `vy_rel`) resolved onto the target-corridor tangent at
  the object's station (same CV reconstruction as Cut-in Risk). It is **not**
  differentiated from the prediction. `relative_longitudinal_speed_mps` =
  object − ego.
* `delta_s_at_merge_valid` / `delta_s_at_merge_m` — the object's predicted route
  station at `ego_merge_time_s`, minus `merge_reference_route_s_m`. The
  predicted station is a linear interpolation over
  `[(0, s_now), (t_k, s_k)…]` when the discrete prediction spans the ego merge
  time, otherwise a constant-current-longitudinal-speed extrapolation
  (`prediction_covers_merge_time` then false). Valid only when the ego merge
  timing is valid.
* `is_ahead_at_merge` / `is_behind_at_merge` / `is_alongside_at_merge` —
  mutually exclusive, from `delta_s_at_merge_m` vs
  `alongside_longitudinal_band_m` (5 m). Spatial-ordering facts, not a
  decision.
* `longitudinal_gap_closing` / `longitudinal_closing_speed_mps` —
  `-sign(delta_s_now) · relative_longitudinal_speed` (> 0 ⇒ the ego↔object
  route-station gap is shrinking). `time_to_route_coincidence_s` =
  `|delta_s_now| / closing_speed` when closing above
  `closing_speed_epsilon_mps` (0.1) — the merge-relevant time-to-coincidence,
  route-relative, not a raw TTC.
* `predicted_min_route_gap_valid` / `_m` / `_time_s` — minimum
  `|predicted_s_k − (ego_route_s + ego_route_speed · t_k)|` over the shared
  prediction horizon (the strongest fact; independent of the ego merge
  timing).
* `prediction_horizon_s`, `prediction_covers_merge_time`.
* `ttc_*` / `cpa_*` / `predicted_min_separation_*` copied unchanged.

## Array summary

`nearest_leading` / `nearest_trailing` — the relevant object with the smallest
`|delta_s_at_merge_m|` ahead of / behind `merge_reference_route_s_m` (ties
broken by the lexicographically smallest UUID). `merge_gap_m` =
`nearest_leading_delta_s_at_merge_m − nearest_trailing_delta_s_at_merge_m` (the
longitudinal span on the target corridor between them at the ego merge time),
valid only when both exist. A measured span, never an accepted or safe gap — a
consumer wanting the ego's own clearance reads the two signed deltas
individually (the ego sits at delta 0 by construction), not the span.

## Fail-safe

Whole-frame reject (warn + skip, never a stale publish): malformed / non-positive
/ duplicate / backward / future / stale risk stamp, wrong `risk_frame_id`
(default `base_link`), missing / stale odometry, ego route projection failure,
degenerate merge corridor. A large backward stamp jump resets the last-stamp
latch (sim-time reset). Individual malformed objects are skipped and counted;
the frame still publishes. An empty objects list with a valid header publishes
a valid empty array. No output field is ever NaN / Inf.

## Node / launch / config

`ad_highway_merge_gap_risk` (C++). Opt-in from `planner.launch.py`
`highway_merge_gap_risk:=true` (default **false**, zero behaviour change) or the
standalone `highway_merge_gap_risk.launch.py`. No planner node consumes the
output. Config `ad_planner/config/highway_merge_gap_risk.yaml` +
`highway_merge.json` — geometric parameters only, no accepted-gap / safe-gap /
merge / yield threshold. QoS: risk-in / facts-out
`rclcpp::QoS(1).reliable().durability_volatile()`, odometry-in
`rclcpp::QoS(10).reliable()`. Diagnostics on
`/ad/planning/highway_merge_gap_risks/diagnostics`;
`HIGHWAY_MERGE_GAP_RISK_RUNTIME_SUMMARY` log every N frames.

## Validation

* `test_highway_merge_gap_risk` — 34 gtests: ego timing (approach / in-zone /
  past / stopped / too-far); relevance + loop-seam window + a windowed-lane
  far-object clamp (relevant_to_merge false, no throw, context zeroed); ahead /
  behind / alongside at merge; relative & closing speed + coincidence time;
  predicted minimum route gap; prediction-coverage both branches; **curved
  geometry**
  (an object ahead along the route with negative base_link `x_rel` is still
  ordered correctly by route station); nearest-trailing arbitration (−15 wins
  over −55); merge-gap identity; malformed-object rejection; budget;
  determinism + finiteness; backend independence (no orientation input); node
  frame contract (wrong frame / missing odom / future / stale / duplicate /
  degenerate corridor / valid empty publishes).
* `test_interface_contract.py` — declaration stable + conservative defaults +
  policy-token disjoint set (`merge_allowed`, `lane_change`, `accepted_gap`,
  `safe_gap`, `go`/`wait`/`yield`/`hold`/`release`, …).
* `test_highway_merge_gap_risk_launch.py` — geometric config + canonical
  topics + merge geometry well-formed / map-frame / provenanced + cross-check
  against `route_corridor.json` source lane + pinned link set + standalone
  launch starts only this node + `planner.launch.py` opt-in default off.
* `test_highway_merge_gap_risk_runtime.py` — deterministic canonical replay:
  ego swept `route_s = 1040 → 1160` at 8 m/s on the real `route:0` centerline;
  4 mainline objects + one merging NPC per frame, plus one far-ego frame
  (`route:0` s ≈ 400). 6/6 frames published, 0 rejected, 25 relevant
  object-frames, 5 unique UUIDs, 20 ahead / 5 behind / 0 alongside, 10 closing
  with coincidence time, 25 predicted-min-route-gap valid, 15
  prediction-covers-merge-time (later ego stations) / 10 not (early stations),
  5 merging-predicted-enter, 5 merge-gap-valid; `merge_gap_m` median 110.8 m;
  the far-ego frame publishes with `ego_merge_timing_valid = false` and 0
  relevant objects (not a rejection); representative (ego s = 1130): ego merge
  ETA 19.5 s, `delta_s_at_merge` +69.9 m ahead / −28.0 m behind now, merge gap
  95.4 m. Internal callback latency median / p95 / max ~15.6 / 15.8 / 16.7 ms
  for a 5-object × 24-sample stress frame (the station-bounded object window
  keeps this independent of route length; well within the 0.5 s prediction
  timeout); loopback publish-to-receive ~17 ms (DDS + 20 ms probe poll, not
  compute). 0 NaN / Inf / exceptions.

## Recommended next task

Highway Merge Gap Response v1 — consume `HighwayMergeGapRisk` and produce a
planner-facing `MERGE_READY` versus `WAIT` / `HOLD` advisory using an explicit
gap / time-to-coincidence policy, requiring `prediction_covers_merge_time` for
every relevant object, without directly commanding lane change, steering,
acceleration, or CtrlCmd.
