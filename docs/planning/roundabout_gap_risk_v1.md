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
- only the **first contiguous** occupancy interval is reported *in these
  fields*: if a predicted path leaves the polygon and re-enters it later in the
  horizon, the re-entry is not in `object_entry_time_s` / `object_exit_time_s` /
  `temporal_gap_s` / `occupancy_overlap` (locked by
  `test_roundabout_gap_risk.cpp::OnlyFirstContiguousOccupancyIntervalIsReported`).
  The **all-interval summary** below scans every predicted occupancy interval; a
  RELEASE / YIELD consumer must use it, never the first-interval fields alone;
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

## Multi-interval summary / policy-consumer contract

The first-interval fields (`object_entry_*`, `object_exit_*`, `arrival_delta_*`,
`temporal_gap_*`, `occupancy_overlap`) describe **only the first contiguous
predicted occupancy interval**. That is unsafe as the sole input to a
gap-response policy: a predicted path can leave the region and re-enter it
while the ego is still inside, so an early first exit does **not** mean the
conflict is clear.

**Unsafe first-interval counterexample.** Ego occupancy `[4, 6]` s; the object
is predicted to occupy the conflict region over `[1, 2]` s and again over
`[5, 7]` s. The first-interval fields report `object_exit_time_s = 2`,
`occupancy_overlap = false`, `temporal_gap_s = 2` — "clear". But the second
predicted interval `[5, 7]` overlaps the ego window. A response node reading
only the first-interval fields would incorrectly RELEASE.

The node therefore scans **every** discrete predicted centroid into contiguous
"inside the conflict region" runs (a currently-inside object contributes a run
starting at `0.0`; a run still inside at the last predicted sample is *open*
through the horizon) and adds these additive facts:

| field | meaning |
| --- | --- |
| `predicted_conflict_interval_count` | number of contiguous occupancy runs (`0` when the object never occupies the region) |
| `later_reentry_detected` | `predicted_conflict_interval_count > 1` — the object leaves and re-enters within the horizon |
| `any_occupancy_overlap` | **any** predicted occupancy interval strictly overlaps the ego occupancy interval (same strict / touching convention as `occupancy_overlap`). Valid only when ego entry+exit are valid; `false` otherwise |
| `minimum_temporal_gap_valid` / `minimum_temporal_gap_s` | minimum non-negative separation between the ego interval and **any** predicted occupancy interval. `0.0` when `any_occupancy_overlap` (and on a touching boundary). Valid only when ego entry+exit are valid and ≥ 1 predicted occupancy interval exists |
| `prediction_horizon_s` | latest discrete predicted-centroid time for this object (`0.0` when it carries none). Always finite |
| `prediction_covers_ego_exit` | `true` only when ego exit is valid **and** `prediction_horizon_s ≥ ego_exit_time_s` (within `1e-3 s`). When `false` the prediction does not span the whole ego conflict window, so the absence of a later overlap is unproven. **Data-coverage flag only — NOT a safe-to-enter decision** |

For the counterexample above: `predicted_conflict_interval_count = 2`,
`later_reentry_detected = true`, `any_occupancy_overlap = true`,
`minimum_temporal_gap_s = 0`.

**A future Roundabout Gap Response must** decide RELEASE / YIELD from
`any_occupancy_overlap`, `minimum_temporal_gap_s` and `later_reentry_detected`
(never the first-interval fields alone), and must require
`prediction_covers_ego_exit == true` for every relevant object before it can
justify a RELEASE — otherwise a late re-entry could be hidden by a
coverage-limited prediction. The `arrival_delta_s` / `temporal_gap_s` /
`occupancy_overlap` first-interval fields remain for diagnostics and backward
compatibility and are **not** redefined.

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
primary-route stations (`s = 850 … 886`, 8 m/s) with **five** circulating
objects each frame (clears before ego / occupancy overlap / arrives after ego /
nearby non-conflicting **with a deliberately short 4 s prediction** /
**leaves-then-re-enters the ego window**):

| # | counts |
| --- | --- |
| input risk messages / output gap messages | 5 / 5 |
| input objects | 25 |
| conflict-relevant object-frames | 20 |
| unique relevant UUIDs | 4 |
| valid ego-entry frames | 5 |
| valid object-entry / valid temporal-gap object-frames | 20 / 20 |
| first-interval occupancy-overlap object-frames | 10 |
| multi-interval object-frames / later-reentry object-frames | 5 / 5 |
| any-occupancy-overlap object-frames | 13 |
| minimum-temporal-gap-valid object-frames | 20 |
| prediction-covers-ego-exit object-frames | 21 |

`prediction_covers_ego_exit` is exercised in both states: the four
full-horizon objects (8 s) always cover; the short-prediction object (4 s)
covers only at the last station (`s = 886`, ego exit ETA 3.5 s) and is
`false` at the four earlier stations where the ego exit ETA is 4.25–8.0 s.

Distributions (median / p10 / min / max):

| quantity | median | p10 | min | max |
| --- | --- | --- | --- | --- |
| ego entry ETA (s) | 2.512 | 0.500 | 0.500 | 4.999 |
| object entry ETA (s) | 1.500 | 0.500 | 0.500 | 7.000 |
| arrival delta (s) | -0.387 | -4.499 | -4.499 | 6.500 |
| non-overlap temporal gap (s) | 2.262 | 0.250 | 0.250 | — |
| publish→receive latency (ms), 5 samples | ≈2.3 | p95 ≈2.5 | — | ≈3.0 |

Representative frame (ego at `s = 870`, ≈ 20 m before the conflict entry):
ego occupancy `[2.512 s, 5.500 s]`.

| object | first interval | first overlap | first gap | arrival delta | interval count | any_occupancy_overlap | min gap | covers ego exit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clears before ego | `[0.5, 1.5] s` | false | `1.012 s` | `-2.012 s` | 1 | false | `1.012 s` | true |
| occupancy overlap | `[2.5, 5.0] s` | true | `0.0 s` | `-0.012 s` | 1 | true | `0.0 s` | true |
| arrives after ego | `[7.0, ∞) s` | false | `1.500 s` | `4.488 s` | 1 | false | `1.500 s` | true |
| nearby non-conflicting (short 4 s prediction) | — | — | — | — | 0 | false | invalid | **false** (horizon 4.0 s < ego exit 5.5 s) |
| **leaves then re-enters** | `[0.5, 1.5] s` | **false** | **`1.012 s`** | `-2.012 s` | **2** | **true** | **`0.0 s`** | true |

The last row is the point of this revision: the first-interval fields alone
say "clear, gap 1.012 s", but the all-interval summary shows the re-entry
overlaps the ego window (`any_occupancy_overlap = true`,
`minimum_temporal_gap_s = 0`, `later_reentry_detected = true`). The
short-prediction object shows the opposite guard: its prediction stops at
4.0 s, before the ego exit ETA of 5.5 s, so `prediction_covers_ego_exit =
false` and a response consumer cannot RELEASE relative to it.

0 NaN / Inf / exceptions across all frames and fields. End-to-end
publish→receive latency (probe-measured, publish-to-receive proxy over the
loopback DDS transport — **not** the node's internal compute cost, 5 samples)
was ≈2.3 ms median, ≈3.0 ms max (run-to-run ≈1.5–3.5 ms; more objects than v1's
4). The node's own internal
processing latency was not separately instrumented in this replay
(`runtime_summary_interval_frames` was set to 0); it is exposed per frame as the
`latency_ms` key of the `/ad/planning/roundabout_gap_risks/diagnostics` status
and aggregated (median / p95 / max) into `ROUNDABOUT_GAP_RISK_RUNTIME_SUMMARY`
every `runtime_summary_interval_frames` when enabled. `build_roundabout_frame`
is still one polygon test per predicted sample plus a single `project_to_frenet`
per frame; the all-interval scan adds only an `O(intervals)` pass
(`intervals ≤ predicted samples`), so per-frame cost stays `O(N · M)` for `N`
objects and `M` predicted samples each, under the `maximum_objects` bound. No
all-pairs interval logic.

## Tests

- `test_roundabout_gap_risk.cpp` — **47** pure-core cases: point-in-polygon /
  distance, ego ETA (approaching / stopped / inside / past-exit / far),
  object entry-in-future / already-inside / never-enters / still-inside-at-
  horizon / first-contiguous-interval-only, the interval cases A/B/C
  plus touching-boundary and unbounded-object-exit, arrival-delta sign
  (before / after / simultaneous / invalid), multi-object one-record-each,
  zero objects, malformed-skipped, budget, degenerate polygon,
  non-increasing span, determinism, curved trajectory uses discrete prediction,
  plus **17 `RoundaboutMultiInterval` cases**: single-interval matches
  first-interval, first-clears-but-second-overlaps (the counterexample),
  two-separated-intervals minimum gap, intervals-on-both-sides-of-ego,
  first-overlaps-second-irrelevant, both-before / both-after ego, touching
  boundary, current-inside-exits-then-re-enters, open final interval,
  no-interval-no-summary, prediction horizon shorter / exactly / longer than
  ego exit, stopped-ego aggregates invalid, determinism, all outputs finite.
- `test_roundabout_gap_risk_launch.py` — config is geometric (no policy
  tokens), conflict geometry well-formed / map frame, standalone launch starts
  only the node, opt-in default off in `planner.launch.py`.
- `test_roundabout_gap_risk_runtime.py` — the live deterministic replay above,
  now with a leaves-then-re-enters object and a short-prediction object that
  exercises `prediction_covers_ego_exit == false`.
- `test_interface_contract.py` — `RoundaboutGapRisk` / `RoundaboutGapRiskArray`
  declarations stable (all-interval fields locked), all-interval defaults are
  the conservative "nothing proven" state, no GO / YIELD / RELEASE / STOP /
  gap-accepted / safe-to-enter / risk-score field.
- Full `ad_planner` ctest **42 / 43** (only pre-existing `test_mppi_nav2_launch`
  fails — host lacks `nav2_common` / `nav2_controller`).
- `test_dynamic_object_risk` (upstream, `ad_lidar_perception`) **38 / 38**,
  unchanged (the `ad_interfaces` change is purely additive).

## Known limitations

- Object containment is **centroid-only** (no footprint) — a circulating
  vehicle whose body overlaps the SE arc but whose centroid is in an adjacent
  circulating lane is not counted.
- The conflict region is a fixed annular sector; the carriageway half-width
  (`5 m`) and angular span are documented engineering choices, not surveyed
  geometry.
- Object entry / exit resolution is bounded by the predictor sample spacing
  (no between-sample interpolation) — this applies to the all-interval scan
  too: an entry / exit / re-entry between two discrete centroids is not seen.
- `prediction_covers_ego_exit` is a coverage flag, not proof of clearance: it
  says the prediction spans the ego window, not that the object stays out.
- The ego arrival estimate is constant-current-speed, not a planner
  trajectory; it is a fact, not a plan.
- Runtime validation is a deterministic canonical ROS replay with synthetic
  circulating trajectories, not an executed MORAI roundabout, because no
  provenance-verified circulating-actor bag is available.
- No vehicle-response policy consumes this interface yet.
