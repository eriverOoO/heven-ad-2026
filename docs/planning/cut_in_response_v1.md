# Cut-in Response v1

Cut-in Response v1 is the first planner-response policy layer. It turns the
factual `CutInRiskArray` frame plus the ego longitudinal speed into a single
planner-facing **longitudinal request**. It never commands an actuator, never
steers, and never plans a path.

```text
/ad/planning/cut_in_risks   (CutInRiskArray, route corridor frame)
/ad/localization/odometry   (Odometry, body-forward speed from twist.linear.x)
                  |
                  v
           ad_cut_in_response
                  |
                  v
/ad/planning/cut_in_response            (CutInResponse, source risk frame)
/ad/planning/cut_in_response/diagnostics
```

The node is opt-in. `planner.launch.py cut_in_response:=true` starts it (and
also starts `ad_cut_in_risk`, which it consumes); the default is `false`. It can
also be started standalone with `cut_in_response.launch.py`. **No production
planner node subscribes to `/ad/planning/cut_in_response` in this change** --
`planner_ros_interfaces.cpp` is untouched, so production driving behaviour is
unchanged.

## Input and output contracts

| role | topic | type | frame / timestamp |
| --- | --- | --- | --- |
| cut-in facts | `/ad/planning/cut_in_risks` | `ad_interfaces/msg/CutInRiskArray` | route corridor frame (`map`); source risk stamp |
| ego speed | `/ad/localization/odometry` | `nav_msgs/msg/Odometry` | `twist.twist.linear.x` is body-forward speed; negative (reverse) is clamped to 0 |
| output | `/ad/planning/cut_in_response` | `ad_interfaces/msg/CutInResponse` | echoes the input `header` (frame + stamp) |

`CutInResponse` carries: `action` (`ACTION_NONE` / `ACTION_SLOWDOWN` /
`ACTION_HOLD`), `active`, `reason`, `candidate_count`, `source_object_id` /
`source_side`, `requested_max_speed_valid` + `requested_max_speed_mps`, and the
transparent decision inputs for the limiting object (`ego_speed_mps`,
`required_deceleration_mps2`, `route_s_rel_m`, `predicted_entry_*`,
`lateral_velocity_toward_corridor_mps`, copied `ttc_*` / `cpa_*` /
`predicted_min_separation_*`). No field is ever NaN or Inf. There is no
`risk_score`, brake/throttle/steering value, gear, trajectory, or lane-change
field.

## Policy states

| state | meaning | `requested_max_speed_mps` |
| --- | --- | --- |
| `ACTION_NONE` | No planner restriction from any cut-in. | 0.0, `requested_max_speed_valid = false` |
| `ACTION_SLOWDOWN` | A cut-in is relevant and the comfortable-braking margin to its route station is eroding, but there is still room to reduce speed without a full stop request. | the speed from which the configured comfortable deceleration still stops the ego at the standoff point; in `[minimum_response_speed_mps, current ego speed)` |
| `ACTION_HOLD` | Continuing at the current speed would create a near-term longitudinal conflict; the planner should prepare to hold longitudinal progress. | 0.0, `requested_max_speed_valid = true` |

There is no `EMERGENCY_BRAKE` state. Emergency response is a separate safety
layer.

## Physical policy

Only candidates (`cut_in_candidate == true`) are considered. A non-candidate
object never contributes, even if its TTC or CPA happens to be small -- generic
collision response belongs to another layer.

Per candidate, with `v_ego` the clamped ego speed:

1. **Longitudinal headroom.** `station_ahead = min(route_s_rel_m,
   predicted_entry_route_s_rel_m)` -- both are signed route stations (same
   unit). 2-D separations (`cpa_distance_m`, `predicted_min_separation_m`) are
   **not** mixed in here: a completed cut-in always ends at a near-zero centroid
   separation, so thresholding them would collapse every candidate to a hold.
2. **Alongside / behind.** `station_ahead <= 0` => `ACTION_NONE`. A vehicle
   merging in beside or behind the ego is not a longitudinal-response case in
   v1; other planner safety layers still cover a genuine collision.
3. **Kinematic tier.** `available = station_ahead - longitudinal_standoff_m`.
   - `available <= 0` => `ACTION_HOLD`, `REASON_COLLISION_CONFLICT`.
   - else `a_req = v_ego^2 / (2 * available)` (the constant deceleration that
     brings the ego to rest at the standoff point) and
     `v_req = sqrt(2 * comfortable_deceleration_mps2 * available)`:
     - `a_req <= comfortable_deceleration_mps2` => `ACTION_NONE`;
     - `a_req <= maximum_deceleration_mps2` and
       `v_req >= minimum_response_speed_mps` => `ACTION_SLOWDOWN` at `v_req`,
       `REASON_APPROACHING_ENTRY`;
     - otherwise => `ACTION_HOLD`, `REASON_COLLISION_CONFLICT`.
4. **TTC hold.** If `ttc_valid` and `ttc_s <= v_ego / maximum_deceleration_mps2`
   (the collision arrives before the ego could stop under maximum braking) the
   action is raised to `ACTION_HOLD`. A valid TTC means the circumscribed
   circles actually make contact under constant relative velocity; a purely
   lateral merge that ends ahead of the ego without contact leaves `ttc_valid`
   false, so this never fires on a normal merge.

`required_deceleration_mps2` reports `a_req` from the stopping-distance tier;
when the object is already within the standoff distance it is reported as
`maximum_deceleration_mps2` (a floor). A hold driven by step 4 (TTC) can
therefore carry a `required_deceleration_mps2` below the comfortable value --
TTC, not stopping distance, set that action.

`REASON_SMALL_PREDICTED_CLEARANCE` is reserved for a future footprint-aware
refinement and is not emitted in v1.

### Monotonicity

Holding all other facts fixed: a smaller `route_s_rel_m` /
`predicted_entry_route_s_rel_m`, a smaller `ttc_s`, or a larger `v_ego` never
weakens the action and never raises `requested_max_speed_mps`.
`predicted_entry_time_s` is reported but does not drive the magnitude (entry
*position* does); decreasing it with all else fixed leaves the response
unchanged. Locked by `test_cut_in_response.cpp`.

## Multi-object arbitration

The aggregate action is the most restrictive over all candidates. Among the
candidates at that action, `requested_max_speed_mps` is the smallest and
`source_object_id` identifies that object; ties are broken by the
lexicographically smallest UUID. Safe and dangerous objects are never averaged.
`candidate_count` counts every candidate regardless of its individual action.

## Requested-speed semantics

`requested_max_speed_mps` is an **upper bound** the planner should apply to its
own desired speed, not a commanded velocity. For `ACTION_SLOWDOWN` it is always
`>= minimum_response_speed_mps` and strictly below the current ego speed (the
kinematic tier only reaches `SLOWDOWN` when `v_ego > v_req`), so it always
represents a real reduction and never exceeds the unconstrained reference
speed. For `ACTION_HOLD` it is `0.0`. It is never negative and never NaN/Inf.

## Release / hysteresis

Cut-in Risk v1 had one continuous candidate interval per true UUID, and the
response was likewise stable in the canonical replay (a single
`NONE -> HOLD -> NONE` block, no chattering), so v1 ships **stateless**: exactly
one response per accepted frame, never latched, no release or activation-delay
parameter. Release hysteresis (holding the last non-`NONE` response for a few
frames after it clears -- never an activation delay, so lead time is never
traded away) is the follow-up if a future scenario shows speed-request chatter.

## Stale / missing data

Exactly one response is published per accepted risk frame. A frame is rejected
(no publish, throttled `WARN` diagnostic) for: malformed / non-positive stamp,
empty `frame_id`, duplicate or backward stamp, stamp in the future beyond
`maximum_future_skew_s`, age beyond `maximum_input_age_s`, missing odometry, or
odometry stale beyond `maximum_odometry_skew_s` relative to the risk stamp. A
large simulated-clock rollback clears the timestamp-ordering state and the
release latch. An individual non-finite candidate field skips that object
(counted in `rejected_malformed`); the frame still publishes. A valid
zero-candidate frame publishes `ACTION_NONE`, `active = false`.

The response is never latched, so a consumer must apply its own freshness
timeout; the absence of a fresh response means "no cut-in longitudinal
constraint", which is the fail-safe direction (the planner keeps its own speed
and its own collision-avoidance layers).

## Parameters (`ad_planner/config/cut_in_response.yaml`)

| parameter | default | meaning |
| --- | --- | --- |
| `maximum_input_age_s` | 0.5 | risk frame age bound |
| `maximum_odometry_skew_s` | 0.5 | odometry / risk stamp difference |
| `maximum_future_skew_s` | 0.1 | tolerated future stamp |
| `longitudinal_standoff_m` | 6.0 | route-relative clearance kept to the cut-in station (~front bumper 3.845 m + margin) |
| `comfortable_deceleration_mps2` | 1.8 | mirrors `perception.braking_deceleration_mps2` |
| `maximum_deceleration_mps2` | 3.0 | within the measured IONIQ 5 brake envelope; also the TTC hold threshold divisor |
| `minimum_response_speed_mps` | 1.0 | a slowdown below this becomes a hold |
| `maximum_risks` | 256 | candidate compute bound |
| `runtime_summary_interval_frames` | 200 | diagnostic log interval; 0 disables the summary |

No `risk_low` / `risk_medium` / `risk_high` bands and no score weights.

## Validation

### Positive scenario (deterministic canonical replay)

Reused the Cut-in Risk v1 canonical scenario (mirrored left/right merging
actors plus parallel-adjacent, moving-away, and fast-crossing negative
controls) through all three production nodes -- `ad_dynamic_object_risk` ->
`ad_cut_in_risk` -> `ad_cut_in_response` -- with a prescribed ego speed ramping
`4.0 -> 10.0 m/s`. Ego pose is a fixed per-frame snapshot exactly as in the
Cut-in Risk v1 replay; only the reported body-forward speed varies.

- 31 `CutInRiskArray` frames -> 31 `CutInResponse` frames; every frame in the
  `map` frame; `candidate_count` matches the source frame's candidate count
  every frame.
- state sequence: `NONE` for scenario time `0.0-0.7 s`, `ACTION_HOLD` for
  `0.8-2.3 s`, `NONE` for `2.4-3.0 s` (16 hold frames, 15 none frames, 0
  slowdown frames in this ramp).
- first candidate at `0.0 s` (Cut-in Risk v1), true right-object corridor entry
  at `2.2 s`, first active response at `0.8 s` -> `1.4 s` response lead before
  entry. The response deliberately does **not** activate the instant the object
  is a candidate; at `4.0-5.4 m/s` the comfortable-braking margin to a 15 m
  cut-in is still fine, so the policy stays `NONE` until the ego is fast enough
  and the conflict close enough in time.
- limiting object alternates between the mirrored UUIDs 1 and 2; a negative
  control (3 / 4 / 5) is never the source and never produces an active
  response.
- 0 NaN / Inf / exceptions.
- node latency median / p95 / max `0.0036 / 0.0039 / 0.0041 ms` -- negligible
  relative to Cut-in Risk (`~14 ms`). No hard-real-time claim.

This ramp exercises `NONE -> HOLD -> NONE`. A closing ego on a 15 m cut-in is a
genuine hold case, so `SLOWDOWN` is exercised separately below.

### Policy state coverage (live node, synthetic risks)

`ad_cut_in_response` alone, driven by synthetic `CutInRiskArray` frames and a
fixed `8.0 m/s` ego:

- one right-side candidate swept from `route_s_rel_m = 45 m` to `7 m`:
  `NONE, NONE, NONE, SLOWDOWN, HOLD, HOLD, HOLD` -- all three states, urgency
  non-decreasing as the station shrinks. At `18 m` the request is
  `~6.57 m/s`; at `14 m` and closer it is a hold (`0.0`).
- non-candidate risk -> `ACTION_NONE`, `active = false`, `candidate_count = 0`.
- mirrored left / right candidates at the same station -> identical `action`,
  `requested_max_speed_mps`, and `required_deceleration_mps2`.
- two candidates (`40 m` left, `9 m` right) -> `ACTION_HOLD`, source is the
  `9 m` object, `candidate_count = 2`.
- a backward-stamped frame is not published.

### Negative controls

Parallel-adjacent, moving-away, and fast-crossing actors are never
`cut_in_candidate` (Cut-in Risk v1), so they never reach the response policy:
parallel-adjacent / moving-away / crossing false responses = `0 / 0 / 0`.

### Tests

`test_cut_in_response.cpp` (35 gtest cases): the five policy states, TTC-invalid
/ predicted-entry-near, large-CPA no-op, predicted-min-separation no-op,
alongside/behind, zero / low / high / negative ego speed, left/right symmetry,
2- and 3-candidate arbitration, UUID tie-break, malformed input, determinism,
parameter validation, and the three monotonicity locks (entry station, entry
time, TTC), plus node-level frame behaviour (non-candidate, empty, stale,
backward, malformed stamp, missing odometry, non-finite-field skip,
determinism).

`test_interface_contract.py`: `CutInResponse` declaration is stable and carries
no actuator / steering / gear / trajectory / score field.

`test_cut_in_response_launch.py`: config is physical-parameters-only,
deceleration ordering, standalone launch, and the opt-in `cut_in_response`
argument (default `false`).

`test_cut_in_response_runtime.py` and `test_cut_in_response_policy.py`: the two
live replays above.

Focused result: `ad_interfaces` `interface_contract` 8/8; `ad_planner`
**37 / 38** ctest (`test_cut_in_response` gtest, `test_cut_in_response_launch`,
`test_cut_in_response_runtime`, `test_cut_in_response_policy`,
`test_cut_in_risk*`, `test_planner_launch`, and all other planner tests). The
one failure, `test_mppi_nav2_launch`, is the pre-existing optional-Nav2 blocker
(`nav2_controller` / `nav2_common` absent on this host) and touches no file in
this change. `test_dynamic_object_risk` gtest unchanged and passing. Isolated
builds of `ad_interfaces` and `ad_planner` pass. `behaviortree_cpp_v3 3.8.7`
was built from source into a sibling workspace because the apt package
`ros-humble-behaviortree-cpp-v3` is not installed on this host.

## Known limitations

- The reused canonical replay is a deterministic ROS replay on the real route
  corridor, not an executed MORAI cut-in (no provenance-verified cut-in actor
  route exists), and the ego speed is a prescribed per-frame value, not an
  integrated trajectory.
- The kinematic tier uses the current route station; it does not model the
  ego's own advance toward the merge point during `predicted_entry_time_s`.
- `predicted_min_separation_m` and `cpa_distance_m` are reported but do not
  drive the action; footprint-aware clearance is a later refinement.
- No production planner consumes `/ad/planning/cut_in_response` yet.
