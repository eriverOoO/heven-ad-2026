# Cut-in Risk v1

Cut-in Risk v1 turns generic dynamic-object risk facts plus the active route
context into explicit per-object cut-in evidence. It is observational only:
no target-speed, brake, steering, path, lane-selection, emergency-stop, or
BehaviorTree behavior consumes this output in this change.

```text
/ad/planning/dynamic_object_risks  (DynamicObjectRiskArray, base_link)
/ad/planning/drivable_mask         (OccupancyGrid, base_link, exact stamp)
/ad/localization/odometry          (Odometry)
checksum-verified active ReferenceCorridor (map)
                  |
                  v
             ad_cut_in_risk
                  |
                  v
/ad/planning/cut_in_risks          (CutInRiskArray, map)
/ad/planning/cut_in_risks/diagnostics
```

The node is opt-in. `planner.launch.py cut_in_risk:=true` starts it; the
default is `false`. It can also be started with `cut_in_risk.launch.py`.

## Input and route contracts

| role | topic | type | frame and timestamp |
| --- | --- | --- | --- |
| object facts | `/ad/planning/dynamic_object_risks` | `ad_interfaces/msg/DynamicObjectRiskArray` | `base_link`; prediction source stamp |
| active route context | `/ad/planning/drivable_mask` | `nav_msgs/msg/OccupancyGrid` | `base_link`; generated at the triggering predicted-object or LiDAR stamp |
| ego state | `/ad/localization/odometry` | `nav_msgs/msg/Odometry` | pose transformed to route frame at the risk stamp; twist x is body-forward speed |

The canonical route is the existing `ReferenceCorridor` loaded from
`ad_data/map/route_corridor.json`. Its `source_sha256.global_path` must match
the selected global path. The same corridor drives
`ad_road_corridor_mask_node`, `future_road_risk`, and local planning. The
exact-stamp mask proves that current route geometry was available for the
risk frame; the cut-in node never silently reuses an older mask. Corridor
membership itself uses the corridor's primary-lane center points and its
per-point left/right widths, not occupancy-cell approximation.

The mask geometry contract remains the existing fixed base-link grid:
`x=[-4,100] m`, `y=[-10,10] m`, `0.1 m` resolution. Invalid dimensions,
origin, orientation, size, or cells reject the frame.

## Coordinates and corridor

Output uses the corridor frame (`map`). Each current and predicted object
centroid is projected with the existing `project_to_frenet` utility:

- `s`: progress on the active primary route;
- `d`: signed lateral offset, positive left and negative right;
- `route_s_rel_m = object_s - ego_s`.

The corridor at station `s` is the inclusive centroid interval
`[-right_width(s), +left_width(s)]`. Width is interpolated from the existing
route corridor; no generic 3.5 m lane width is introduced. `near_boundary`
means centroid distance to the nearest boundary is at most
`boundary_margin_m`. An outside centroid is in `adjacent_region` only when
its boundary gap is at most `maximum_lateral_gap_m`. Object footprints are
not mixed into this v1 geometry.

## Lateral approach

The object world velocity is reconstructed from generic relative velocity
plus ego forward velocity, then projected into route coordinates as
`d_dot`.

- outside left (`d > +left_width`): toward speed is `-d_dot`; approaching
  requires `d_dot < 0` with magnitude at least the configured minimum;
- outside right (`d < -right_width`): toward speed is `+d_dot`; approaching
  requires `d_dot > 0` with magnitude at least the configured minimum.

This is symmetric and direction-aware. `abs(vy_rel)` is never used as the
classification rule.

## Predicted entry

`DynamicObjectRisk.predicted_states[]` carries the source predictor's
discrete future centroids, expressed relative to a constant-velocity ego at
each horizon in the source-stamp base axes. This policy-free extension keeps
the core architecture `Prediction -> Dynamic Object Risk -> Cut-in Risk`;
the cut-in node does not subscribe to raw predictions or tracker output.

States are inspected in strictly increasing time order and projected onto
the active route. The first discrete state whose centroid is inside the
corridor sets:

- `predicted_entry_valid`, `predicted_entry_time_s`;
- `predicted_entry_route_s_rel_m`;
- `predicted_entry_lateral_offset_m`.

There is no interpolation and no extrapolation beyond the prediction
horizon. `predicted_entry_sustained` additionally requires every later
discrete state to remain inside and at least
`minimum_sustained_entry_time_s` of prediction to remain after first entry.
This factual persistence rejects a vehicle that merely crosses through the
road corridor.

## Candidate definition

Current longitudinal relevance is purely geometric:

```text
-rear_analysis_distance_m <= route_s_rel_m <= forward_analysis_distance_m
```

The deterministic v1 candidate is:

```text
adjacent_region
AND approaching_corridor
AND predicted_entry_valid
AND predicted_entry_sustained
AND longitudinally_relevant
```

TTC is not required. TTC, CPA, and predicted minimum separation are copied as
related facts only. There is no risk score or response threshold.

Stateless classification was stable in the 31-frame runtime replay: each
true-cut-in UUID had a single continuous candidate interval followed by one
release at corridor entry. Therefore v1 has no hysteresis or UUID state.

## Output

`/ad/planning/cut_in_risks` uses
`ad_interfaces/msg/CutInRiskArray`; its header stamp is the input risk stamp
and `frame_id` is the route corridor frame. Each `CutInRisk` contains:

- identity: UUID, classification, classification/existence probability;
- current route facts: relative `s`, signed `d`, left/right corridor widths,
  inside, near-boundary, adjacent-region, and side;
- motion facts: `d_dot`, velocity toward the boundary, approaching flag, and
  longitudinal relevance;
- entry facts: valid, time, entry relative `s`, entry `d`, and sustained;
- copied TTC, CPA, and predicted-minimum-separation facts;
- `cut_in_candidate`.

The interface has no `risk_score`, control request, GO/STOP/YIELD, or
BehaviorTree transition.

## Invalid data behavior

The whole frame is rejected and a warning diagnostic is emitted for a wrong
risk/mask frame, malformed/non-positive stamp, unequal risk/mask stamps,
duplicate/backward stamp, stale/future frame, missing/stale odometry, missing
transform, malformed mask, or invalid reference corridor. The checksum-
verified corridor must load at node startup; missing/invalid route data
prevents startup. A large simulated-clock rollback clears only timestamp
ordering state.

An individual non-finite/malformed object or predicted state is skipped and
counted, while other objects still publish. Objects over `maximum_objects`
are skipped and counted. A valid zero-object risk array plus valid route and
ego context publishes a valid empty `CutInRiskArray`.

## Parameters

`ad_planner/config/cut_in_risk.yaml` contains only geometry and interface
bounds:

| parameter | default | meaning |
| --- | --- | --- |
| `maximum_input_age_s` | 0.5 | risk and exact route-context age |
| `maximum_odometry_skew_s` | 0.5 | odometry/risk stamp difference |
| `maximum_future_skew_s` | 0.1 | tolerated future timestamp |
| `transform_timeout_s` | 0.05 | route-frame TF lookup timeout |
| `maximum_pending_frames` | 16 | exact-stamp pairing bound |
| `maximum_objects` | 256 | object compute bound |
| `forward_analysis_distance_m` | 80.0 | forward route interaction window |
| `rear_analysis_distance_m` | 5.0 | rear route interaction window |
| `minimum_lateral_approach_speed_mps` | 0.2 | minimum positive toward-boundary speed |
| `boundary_margin_m` | 0.25 | near-boundary centroid band |
| `maximum_lateral_gap_m` | 5.0 | adjacent-region boundary gap |
| `minimum_sustained_entry_time_s` | 0.5 | required predicted time remaining inside |
| `runtime_summary_interval_frames` | 200 | diagnostic log interval; 0 disables summary |

## Validation

Repository/user-local searches found no named `cutin`, `cut_in`, merge,
lane-change, crossing, adjacent, or dynamic-actor MORAI replay. Existing
MORAI actor tooling can spawn/route deterministic actors, but only the
roundabout/highway route presets have pinned link provenance; inventing
unverified cut-in link IDs would violate the repository contract. Runtime
validation therefore uses a deterministic canonical prediction replay on
the real checksum-verified route corridor and runs both production nodes.

The 31-frame replay contains mirrored left/right merging actors and three
negative controls: parallel adjacent, moving away, and a fast crossing that
does not remain in the corridor. Results:

- 31 DynamicObjectRiskArray processed, 31 CutInRiskArray published;
- 155 risk objects; 46 candidate object-frames; 2 unique candidate UUIDs;
- 72 predicted-entry-valid object-frames;
- first candidate at scenario time 0.0 s, actual right-object corridor entry
  at 2.2 s, detection lead time 2.2 s;
- false positives: parallel 0, moving-away 0, crossing 0;
- 0 NaN/Inf/exceptions;
- node processing latency median/p95/max
  `13.717712 / 14.537682 / 15.575071 ms` on this host, including exact route
  context validation and full-route projection; no hard-real-time claim.

At first detection the right actor reports side right, `s=15.025 m`,
`d=-3.901 m`, toward speed `1.000 m/s`, entry time `2.5 s`, TTC invalid,
CPA `4.0 s / 15.0 m`, predicted minimum separation `15.033 m`. The mirrored
left actor reports side left, `s=14.974 m`, `d=4.099 m`, toward speed
`1.000 m/s`, and the same entry/CPA/minimum-separation facts. Slight `s/d`
asymmetry is expected from projecting base-mirrored centroids onto the real
curved route; the straight-route unit pair locks exact left/right symmetry.

Focused tests cover right/left approach and moving-away cases, already
inside, stationary and parallel adjacent, far ahead/behind, crossing without
merge, stop before entry, actual predicted entry, inclusive/grazing boundary,
multiple/empty, malformed/non-finite, curved route, non-zero yaw,
determinism, backend equivalence, route/frame/staleness failures, interface
fields, config, opt-in and standalone launch, and the live two-node replay.

Final focused verification is **93 passed / 0 failed**: 20 cut-in core,
38 dynamic-object-risk core, 7 interface-contract, 23 planner-launch, 3
cut-in-launch, and 2 live-runtime tests. Isolated builds of `ad_interfaces`,
`ad_lidar_perception`, and `ad_planner` pass.

The broader selected-package regression reported **1007 passed, 28 failed,
7 collection errors, 2 skipped** out of 1044 tests. None of those failures
exercise the changed cut-in or dynamic-risk files: 7 collection errors and
the related tracker failures are caused by system Python lacking optional
`filterpy`; 7 planner failures are the existing optional Nav2 MPPI launch
tests with `nav2_common`/`nav2_controller` absent; the remaining unrelated
LiDAR launch assertions overlap pre-existing user-owned launch/config edits
in the dirty worktree. All directly affected tests pass.

## Known limitations

- Centroid membership does not account for object footprint; footprint-aware
  corridor overlap is a later geometry refinement.
- Discrete predictions bound entry timing to predictor sample resolution; no
  between-sample interpolation is claimed.
- The dynamic-risk ego rollout is constant velocity at source yaw; turning-
  ego future motion is not represented.
- Runtime validation is a deterministic canonical ROS replay, not an
  executed MORAI cut-in, because no provenance-verified cut-in actor route is
  currently available.
- No vehicle response policy consumes this interface yet.
