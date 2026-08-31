# Roundabout Gap Response v1

Roundabout Gap Response v1 is a conservative, planner-facing **entry advisory**.
It converts policy-free conflict timing facts into RELEASE, YIELD, or HOLD. It
does not control the vehicle and has no production planner consumer in v1.

```text
/ad/planning/roundabout_gap_risks  (RoundaboutGapRiskArray, map)
                         |
                         v
             ad_roundabout_gap_response
                         |
                         v
/ad/planning/roundabout_gap_response  (RoundaboutGapResponse, map)
                         |
                         v
              future planner integration
```

The standalone `roundabout_gap_response.launch.py` is opt-in. Input diagnostics
are published on `/ad/planning/roundabout_gap_response/diagnostics`. There is
**NO CtrlCmd, requested speed, brake, throttle, steering, path, or production
planner consumer** in this version.

## Applicability and actions

The response is active while ego is before the current forward traversal's
conflict exit and is not currently inside the conflict polygon.

- **RELEASE**: the valid source frame proves the configured all-object release
  conditions below. A valid zero-relevant-object frame releases.
- **YIELD**: release is not justified, but the remaining pre-entry distance is
  greater than the comfortable stopping distance.
- **HOLD**: release is not justified and the comfortable stopping margin has
  been consumed. This is an advisory classification, not a brake command.

If `ego_in_conflict_now` is true, or
`ego_route_distance_to_exit_m < 0`, the response is `active=false` with the
non-restrictive RELEASE enum value and `REASON_NONE`. This avoids advising a
stop inside the shared region and removes the gate after the maneuver. A
consumer must check `active`, not interpret the inactive enum as release
evidence.

## Exact RELEASE policy

RELEASE requires all of the following:

1. a fresh, strictly increasing, finite source frame in `map` for
   `kcity_roundabout`;
2. the response is applicable and both ego entry and exit timing are valid;
3. no relevant object was omitted or rejected;
4. every relevant object has `prediction_covers_ego_exit == true`;
5. every relevant object has `any_occupancy_overlap == false`;
6. every relevant object has `minimum_temporal_gap_valid == true`; and
7. every relevant object's `minimum_temporal_gap_s >= minimum_release_gap_s`.

The legacy `occupancy_overlap` and `temporal_gap_s` fields describe only the
first contiguous object interval and are never used for RELEASE. A later
re-entry therefore cannot be hidden. Prediction coverage is mandatory even
when currently visible intervals do not overlap.

`minimum_release_gap_s` defaults to **2.0 s**, with an inclusive `>=` release
boundary. A touching interval has minimum gap 0 and cannot release. The value
is competition-v1 policy tuning, not a universal safety guarantee: at the
audited 8 m/s roundabout approach speed it represents about 16 m of arrival
separation, and it aligns with the existing 2 s perception clear-release
duration while remaining subordinate to the stronger full-coverage and
all-interval no-overlap rules.

## YIELD / HOLD dynamics

For a non-release frame:

```text
available_distance_m = max(0, ego_route_distance_to_entry_m - entry_standoff_m)
comfortable_stop_distance_m = ego_speed_mps^2 /
                              (2 * comfortable_deceleration_mps2)

available_distance_m > comfortable_stop_distance_m  -> YIELD
otherwise                                             -> HOLD
```

The strict `>` boundary preserves the full comfortable stopping distance.
`comfortable_deceleration_mps2 = 1.8` reuses the canonical planner/perception
comfortable braking value. `entry_standoff_m = 6.0` reuses the existing cut-in
response clearance (vehicle reference/front extent plus margin) and places the
conceptual hold point before, not inside, the conflict polygon.

The upstream gap-risk ETA is invalid below 0.5 m/s. Response v1 mirrors that
threshold and makes a stopped ego before entry HOLD when timing is invalid; it
does not fabricate a conflict interval. A moving ego with invalid timing can
be YIELD or HOLD by the same distance rule but can never RELEASE.

## Multiple objects and reason selection

RELEASE is an all-object conjunction; gaps are never averaged. The limiting
object order is:

1. any all-interval occupancy overlap;
2. incomplete coverage or missing conservative gap evidence;
3. smallest below-threshold minimum temporal gap.

Within the same tier, the lexicographically smallest UUID wins, except the gap
tier first selects the smallest gap and then UUID. The output exposes whether
a source UUID and limiting gap are valid, aggregate overlap/coverage, ego
distance/speed, available distance, and comfortable stopping distance.

The policy is deliberately stateless: no hysteresis was added. Bounded
threshold tests establish the exact inclusive release boundary, and an unsafe
approach progresses monotonically YIELD to HOLD. A later integration may add
conservative release confirmation only if vehicle replay demonstrates material
flicker; unsafe evidence must take effect without delay.

## Input rejection and expiry

Empty/wrong frame or conflict zone, malformed time, stale/future/duplicate/
backward stamps, NaN/Inf, inconsistent relevant-object counts or interval
summaries, and object-budget overflow reject the entire frame. A rejected frame
publishes no response and never reuses a prior RELEASE. Exactly one response is
published for each accepted input. If input stops, output stops; any future
consumer must enforce its own freshness timeout (recommended no greater than
`maximum_input_age_s`, default 0.5 s) and must not latch RELEASE.

## Configuration

`config/roundabout_gap_response.yaml` contains only physical/context policy:

- `maximum_input_age_s: 0.5`
- `minimum_release_gap_s: 2.0`
- `entry_standoff_m: 6.0`
- `comfortable_deceleration_mps2: 1.8`
- `stopped_speed_threshold_mps: 0.5`
- `maximum_relevant_objects: 256`
- expected frame/zone, topics, and diagnostic summary interval

There are no scores, weights, actuator gains, requested speeds, or hysteresis
parameters.

## Deterministic validation

The existing canonical K-City replay now runs
`DynamicObjectRisk -> RoundaboutGapRisk -> RoundaboutGapResponse`. Five risk
frames (25 objects, 20 relevant object-frames) plus one separate valid
zero-object response frame produced 6 responses: 1 RELEASE, 2 YIELD, 3 HOLD,
0 rejected. All five canonical unsafe frames selected `REASON_OVERLAP`; the
zero-object frame selected `REASON_CLEAR_GAP`. At the 870 m representative
station the later-reentry object still has legacy `occupancy_overlap=false`
and first gap 1.012 s, but all-interval overlap is true and response is HOLD.
The 4.0 s short-prediction regression independently produces
`REASON_INSUFFICIENT_PREDICTION`, never RELEASE.

The deterministic approach sequence at 8 m/s changes from YIELD near 40 m
(available ≈34 m, comfortable stop ≈17.78 m) to HOLD near 20 m (available
≈14 m). The zero-object RELEASE example is 32 m before entry with complete
coverage vacuously true. There were no NaN/Inf values or exceptions.

On the recorded run, internal response callback processing latency was
0.003742 ms median / 0.010621 ms p95 / 0.010621 ms max (6 samples). End-to-end
risk-input publish to response receive latency was 2.266 / 2.665 / 7.325 ms
median/p95/max (5 canonical samples). The latter includes both nodes and DDS;
neither measurement is a hard-real-time guarantee.

## Known limitations

Ego occupancy uses the upstream constant-current-speed ETA, so stopped ego
cannot earn RELEASE in v1. The fixed conflict zone and single-lap forward route
contract remain those of Roundabout Gap Risk v1. There is no hysteresis, no
uncertainty-expanded interval policy, and no planner/control integration.
