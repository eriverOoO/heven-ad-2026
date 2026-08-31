# Roundabout Gap Risk v1

Roundabout Gap Risk v1 turns generic dynamic-object risk facts, the ego route
state, and a fixed shared roundabout conflict region into explicit per-object
**conflict-timing facts**: when the ego is expected to occupy the conflict
region, when each object is expected to occupy it, the signed arrival
difference, the non-overlap temporal gap, and whether the predicted occupancy
intervals overlap.

It is observational only. **There is NO GO / YIELD / STOP / HOLD / speed / brake
/ steering / BehaviorTree output** and no accepted-gap or safe-gap decision. A
Roundabout Gap Response node applies a temporal-gap policy later.

```text
/ad/planning/dynamic_object_risks   (DynamicObjectRiskArray, base_link)
/ad/localization/odometry           (Odometry)
checksum-verified primary ReferenceCorridor  (ad_data/map/route_corridor.json)
fixed roundabout conflict geometry  (ad_planner/config/roundabout_conflicts.json)
                  |
                  v
           ad_roundabout_gap_risk
                  |
                  v
/ad/planning/roundabout_gap_risks            (RoundaboutGapRiskArray, map)
/ad/planning/roundabout_gap_risks/diagnostics
```

The node is opt-in: `planner.launch.py roundabout_gap_risk:=true` starts it, or
`roundabout_gap_risk.launch.py` standalone. The default is `false`. **No planner
or BehaviorTree node consumes `/ad/planning/roundabout_gap_risks`.**

## Conflict region — source and provenance

The conflict region is an **annular sector** of the K-City roundabout,
materialised as an explicit map-frame polygon in
`ad_planner/config/roundabout_conflicts.json` (`conflict_zone_id:
kcity_roundabout`).

| element | value | source |
| --- | --- | --- |
| circulating-loop centroid | `(-101.717930, 343.622922)` m, map frame | mean of the four `kcity-roundabout-loop` link entry nodes in `ad_morai_bridge_dev/config/actor_presets.provenance.yaml` (pinned by `link_set_sha256 5ce0fd57…`) |
| circulating-lane centreline radius | `17.679` m | mean of the four corner radii (`17.39–18.02` m) |
| carriageway half-width | `5.0` m | single-lane K-City roundabout carriageway ≈ 10 m; 5 m each side of the lane centreline |
| inner / outer radius | `12.68` / `22.68` m | centreline radius ∓ carriageway half-width |
| angular span (from centroid) | `[-80°, -10°]` | the SE arc over which the primary route centreline lies within the annulus |
| polygon | 58-vertex arc approximation (outer arc + inner arc) | generated from the four items above |
| `route_s_enter_m` / `route_s_exit_m` | `890.1006` / `914.0039` m | primary-route stations where the route centreline enters / exits the sector (`ad_data/map/route_corridor.json`, sha `c66d978c…`) |

The primary ego route crosses the circulating carriageway once, along the SE
quadrant: it enters the sector from the south heading north, cuts the SE corner
where the circulating flow converges, and exits to the north-east. Over
`route s ≈ [890, 914]` (≈ 24 m, ≈ 3 s at 8 m/s) the ego and circulating traffic
share this arc. `route_corridor.json` was **not** modified; `roundabout_conflicts
.json` records its sha for auditability and the node cross-checks the two at
startup (below).

## Coordinate frames

| quantity | frame |
| --- | --- |
| `DynamicObjectRisk` input | `base_link` (ego body: +x forward, +y left) |
| ego odometry | pose in `map` (or transformed to `map` via TF at the risk stamp); `twist.twist.linear.x` is body-forward speed |
| conflict polygon and output | `map` |

Object current and predicted centroids are converted from the body-relative
`DynamicObjectRisk` form to `map` by
`p_map = ego_pose ⊕ R(ego_yaw)·(ego_speed·t + x_rel, y_rel)` — the same
constant-velocity-ego reconstruction Cut-in Risk uses, so a curved circulating
trajectory is followed by its discrete predicted samples rather than by
extrapolating the current velocity.

## Ego conflict ETA (constant-current-speed)

`ego_route_s_m` is the ego station on the primary route
(`project_to_frenet`). With `d_entry = max(0, route_s_enter_m − ego_route_s_m)`
and `d_exit = max(0, route_s_exit_m − ego_route_s_m)`:

```text
ego_entry_time_s = d_entry / ego_speed
ego_exit_time_s  = d_exit  / ego_speed
```

This is a **factual arrival estimate at the current speed**, not a planner
trajectory prediction and not the planner target speed.

`ego_entry_valid` / `ego_exit_valid` are `false` when:

- ego speed `< ego_speed_epsilon_mps` (default `0.5`) — a stopped ego has no
  arrival ETA (Phase 9);
- `route_s_exit_m − ego_route_s_m < −1 mm` — the ego has already passed the
  conflict exit on this lap (the route is a closed 2184 m loop; single-lap,
  forward-only, no next-lap ETA);
- `route_s_enter_m − ego_route_s_m > maximum_ego_approach_distance_m`
  (default `400`) — beyond a useful approach range, and a guard against a
  loop-seam projection artifact.

`ego_in_conflict_now` is the polygon containment of the current ego pose and can
be `true` while the timing is invalid (e.g. a stopped ego already inside).

## Object conflict entry / exit (discrete prediction)

Each `DynamicObjectRisk.predicted_states[]` sample is converted to `map` and
tested against the polygon, in increasing time order:

- object currently inside → `object_entry_valid = true`,
  `object_entry_time_s = 0`, `object_in_conflict_now = true`;
- otherwise the first sample inside sets `object_entry_valid` /
  `object_entry_time_s`;
- `object_exit_time_s` is the first sample **outside** at or after entry;
- `object_exit_valid = false` when the object is still inside at the last
  predicted sample (occupies the region through the horizon);
- only the **first contiguous** occupancy interval is reported: if a predicted
  path leaves the polygon and re-enters it later in the horizon, the re-entry is
  not represented in v1 and a consumer must not treat `object_exit_time_s` as
  "clear for all time" (locked by
  `test_roundabout_gap_risk.cpp::OnlyFirstContiguousOccupancyIntervalIsReported`);
- `relevant_to_conflict = object_in_conflict_now || object_entry_valid`. An
  object whose predicted centroids never enter the polygon is
  `relevant_to_conflict = false` regardless of how close it is; Euclidean
  proximity is never used to synthesise a conflict time.

There is no interpolation between samples and no extrapolation beyond the
prediction horizon.

## Arrival delta (sign)

```text
arrival_delta_s = object_entry_time_s − ego_entry_time_s
```

`< 0` object reaches the conflict region before the ego; `> 0` after the ego;
`≈ 0` together. Valid only when both `ego_entry_valid` and `object_entry_valid`.

## Occupancy interval and temporal gap

Ego interval `[E0, E1] = [ego_entry_time_s, ego_exit_time_s]`. Object interval
`[O0, O1]`, with `O1 = object_exit_time_s` when `object_exit_valid`, else `+∞`
(occupies through the horizon).

```text
occupancy_overlap = (O0 < E1) and (E0 < O1)          # strict

if occupancy_overlap:            temporal_gap_s = 0
elif O1 <= E0:                    temporal_gap_s = E0 − O1   # object clears first
else (E1 <= O0):                  temporal_gap_s = O0 − E1   # ego clears first
```

`temporal_gap_s` is always `≥ 0` and finite. **Touching boundaries are not an
overlap**: `O1 == E0` gives `occupancy_overlap = false`, `temporal_gap_s = 0`;
`E1 == O0` likewise. `temporal_gap_valid` requires both entries valid (ego exit
is always valid when ego entry is). No "safe gap" threshold is applied anywhere.

## Relevance definition

Geometric only: the object is currently inside the conflict polygon, or a
discrete predicted centroid enters it within the prediction horizon. TTC is not
required (TTC / CPA / predicted-minimum-separation are copied from
`DynamicObjectRisk` as related facts only). Object class is never a relevance
condition. No lane IDs (`circulating_lane_id`, `entry_lane_id`, …) are used or
invented — shared conflict-zone occupancy is the whole model in v1.

## Startup / invalid-data contract

The node **fails to start** on: a missing / malformed route corridor or
checksum mismatch; a missing / malformed conflict geometry file; an unknown
`conflict_zone_id`; a degenerate polygon (`< 3` vertices, non-finite vertex,
zero area); a non-increasing route span; or a conflict polygon that is **not
consistent with the loaded route** — the route centreline at
`route_s_enter_m`, `route_s_exit_m`, and their midpoint must lie inside the
polygon (or within `polygon_consistency_margin_m`, default `2 m`). This ties
`roundabout_conflicts.json` to `route_corridor.json` without a runtime file
hash.

A **risk frame is rejected** (no publish; WARN diagnostic) for a wrong risk
frame (`!= base_link`), malformed / non-positive / duplicate / backward stamp,
future or stale stamp, missing / stale odometry, or a failed route projection.
A large backward stamp jump (`> 0.5 s`) is treated as a simulated-clock reset
and clears only the stamp-ordering state. An individual non-finite / malformed
object is skipped and counted; other objects still publish. A valid zero-object
frame publishes a valid empty `RoundaboutGapRiskArray` with ego timing.

## Parameters (`ad_planner/config/roundabout_gap_risk.yaml`)

| parameter | default | meaning |
| --- | --- | --- |
| `conflict_geometry_file` | `""` → package `config/roundabout_conflicts.json` | conflict geometry |
| `conflict_zone_id` | `kcity_roundabout` | zone to load |
| `maximum_input_age_s` | 0.5 | risk staleness |
| `maximum_odometry_skew_s` | 0.5 | risk / odometry stamp skew |
| `maximum_future_skew_s` | 0.1 | tolerated future stamp |
| `transform_timeout_s` | 0.05 | route-frame TF lookup timeout |
| `maximum_objects` | 256 | per-frame compute bound |
| `ego_speed_epsilon_mps` | 0.5 | below this the ego has no arrival ETA |
| `maximum_ego_approach_distance_m` | 400.0 | approach-range / loop-seam guard |
| `polygon_consistency_margin_m` | 2.0 | startup route↔polygon cross-check tolerance |
| `runtime_summary_interval_frames` | 200 | diagnostic log interval; 0 disables |

No `safe_gap_s`, `yield_gap_s`, `go_gap_s`, `emergency_gap_s`, or risk weights.

## Backend independence

The core is a pure function on plain structs with no tracker / prediction /
detector branch. Equivalent `DynamicObjectRisk` predictions produce equivalent
gap facts regardless of whether the upstream tracker is Autoware or AB3DMOT
(`DynamicObjectRisk` is already backend-agnostic).

## Validation

No repository or user-local MORAI replay has provenance-verified roundabout
circulating-actor trajectories (only the `kcity-roundabout-loop` actor **preset**
exists, with pinned link IDs but no recorded actor-state bag). Runtime
validation is therefore a **deterministic canonical ROS replay** against the
real checksum-verified route corridor and the shipped conflict geometry, with
synthetic circulating-object trajectories.

`test_roundabout_gap_risk_runtime.py` sweeps the ego along the approach over 5
primary-route stations (`s = 850 … 886`, 8 m/s) with four circulating objects
each frame:

| # | counts |
| --- | --- |
| input risk messages / output gap messages | 5 / 5 |
| input objects | 20 |
| conflict-relevant object-frames | 15 |
| unique relevant UUIDs | 3 |
| valid ego-entry frames | 5 |
| valid object-entry / valid temporal-gap object-frames | 15 / 15 |
| occupancy-overlap object-frames | 8 |

Distributions (median / p10 / min / max):

| quantity | median | p10 | min | max |
| --- | --- | --- | --- | --- |
| ego entry ETA (s) | 2.512 | 0.500 | 0.500 | 4.999 |
| object entry ETA (s) | 2.500 | 0.500 | 0.500 | 7.000 |
| arrival delta (s) | 0.000 | -3.262 | -4.499 | 6.500 |
| non-overlap temporal gap (s) | 2.262 | 0.250 | 0.250 | — |
| publish→receive latency (ms), 5 samples | ≈1.2 | p95 ≈1.3 | — | ≈1.5 |

Representative frame (ego at `s = 870`, ≈ 20 m before the conflict entry):
ego occupancy `[2.512 s, 5.500 s]`.

| object | object interval | overlap | temporal gap | arrival delta |
| --- | --- | --- | --- | --- |
| clears before ego | `[0.5, 1.5] s` | false | `1.012 s` | `-2.012 s` |
| occupancy overlap | `[2.5, 5.0] s` | true | `0.0 s` | `-0.012 s` |
| arrives after ego | `[7.0, ∞) s` | false | `1.500 s` | `4.488 s` |
| nearby non-conflicting (centre island) | — | — | — | `relevant_to_conflict = false` |

0 NaN / Inf / exceptions across all frames and fields. End-to-end
publish→receive latency (probe-measured, publish-to-receive proxy over the
loopback DDS transport — **not** the node's internal compute cost, 5 samples)
was ≈1.2 ms median, ≈1.5 ms max (run-to-run ≈1.2–1.6 ms). The node's own internal
processing latency was not separately instrumented in this replay
(`runtime_summary_interval_frames` was set to 0); it is exposed per frame as the
`latency_ms` key of the `/ad/planning/roundabout_gap_risks/diagnostics` status
and aggregated (median / p95 / max) into `ROUNDABOUT_GAP_RISK_RUNTIME_SUMMARY`
every `runtime_summary_interval_frames` when enabled. `build_roundabout
_frame` is one polygon test per predicted sample plus a single `project_to
_frenet`; cost scales linearly with object and predicted-sample count under the
`maximum_objects` bound.

## Tests

- `test_roundabout_gap_risk.cpp` — 30 pure-core cases: point-in-polygon /
  distance, ego ETA (approaching / stopped / inside / past-exit / far),
  object entry-in-future / already-inside / never-enters / still-inside-at-
  horizon / first-contiguous-interval-only, the Phase 19 interval cases A/B/C
  plus touching-boundary and unbounded-object-exit, arrival-delta sign
  (before / after / simultaneous / invalid), multi-object one-record-each,
  zero objects, malformed-skipped, budget, degenerate polygon,
  non-increasing span, determinism, curved trajectory uses discrete prediction.
- `test_roundabout_gap_risk_launch.py` — config is geometric (no policy
  tokens), conflict geometry well-formed / map frame, standalone launch starts
  only the node, opt-in default off in `planner.launch.py`.
- `test_roundabout_gap_risk_runtime.py` — the live deterministic replay above.
- `test_interface_contract.py` — `RoundaboutGapRisk` / `RoundaboutGapRiskArray`
  declarations stable, no GO / YIELD / STOP / gap-accepted / safe-to-enter /
  risk-score field.
- Full `ad_planner` ctest **42 / 43** (only pre-existing `test_mppi_nav2_launch`
  fails — host lacks `nav2_common` / `nav2_controller`).
- `test_dynamic_object_risk` (upstream, `ad_lidar_perception`) unchanged and
  passing.

## Known limitations

- Object containment is **centroid-only** (no footprint) — a circulating
  vehicle whose body overlaps the SE arc but whose centroid is in an adjacent
  circulating lane is not counted.
- The conflict region is a fixed annular sector; the carriageway half-width
  (`5 m`) and angular span are documented engineering choices, not surveyed
  geometry.
- Object entry / exit resolution is bounded by the predictor sample spacing
  (no between-sample interpolation).
- The ego arrival estimate is constant-current-speed, not a planner
  trajectory; it is a fact, not a plan.
- Runtime validation is a deterministic canonical ROS replay with synthetic
  circulating trajectories, not an executed MORAI roundabout, because no
  provenance-verified circulating-actor bag is available.
- No vehicle-response policy consumes this interface yet.
