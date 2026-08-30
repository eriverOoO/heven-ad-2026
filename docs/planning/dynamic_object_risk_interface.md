# Dynamic Object Risk Interface

A backend-agnostic ROS node that turns the canonical predicted-object stream
plus the ego state into planner-facing per-object relative-motion and risk
metrics. It produces **facts, not decisions**: no GO / STOP / YIELD / brake /
cut-in / merge / roundabout logic lives here. Those policies are later tasks
and consume this interface.

```text
/ad/perception/objects/predicted   (ad_interfaces/PredictedObjectArray, odom)
/ad/localization/odometry          (nav_msgs/Odometry, pose in odom)
        |
        v
  ad_dynamic_object_risk  (this node -- stateless geometry transform)
        |
        v
/ad/planning/dynamic_object_risks   (ad_interfaces/DynamicObjectRiskArray, base_link)
/ad/planning/dynamic_object_risks/diagnostics  (diagnostic_msgs/DiagnosticArray)
```

## Backend independence

The node consumes only the shared `PredictedObjectArray`. It contains no
`if backend == ab3dmot` / `if backend == autoware` path. The same node and the
same output result for the Autoware tracker -> Prediction path and the AB3DMOT
tracker -> Prediction path, because:

- object centroid, world-frame velocity, and dimensions are read from the
  canonical message the same way regardless of producer;
- **object orientation is never read.** AB3DMOT publishes an unavailable
  identity yaw; Autoware publishes a real yaw. Using either would make the
  same physical object produce different metrics, so the collision model is
  built from a rotation-invariant circumscribed circle instead (see TTC).

Locked by `test_dynamic_object_risk.cpp`
(`ObjectOrientationNeverAffectsRiskOutput` -- same physical state with an
AB3DMOT identity quaternion vs a real Autoware yaw, identical output) and
`test_lidar_perception_launch.py`
(`test_dynamic_object_risk_is_opt_in_and_backend_agnostic`).

## Inputs

| role | topic | type | frame |
| --- | --- | --- | --- |
| predictions | `/ad/perception/objects/predicted` | `ad_interfaces/msg/PredictedObjectArray` | `odom` |
| ego state | `/ad/localization/odometry` | `nav_msgs/msg/Odometry` | pose in `odom`, twist in `base_link` |

**Ego state.** `/ad/localization/odometry` is the canonical localization
output (the planner reads ego pose from it today). The node uses:

- ego position + yaw from `pose.pose` (odom);
- ego longitudinal speed from `twist.twist.linear.x` (body frame).

Ego **lateral** velocity and **yaw rate** are structurally unobserved by the
default `gnss_imu` localization backend (it only fills `twist.twist.linear.x`
from wheel speed) and are treated as **0** -- a non-holonomic assumption. Ego
velocity is never estimated from object tracks or by differencing TF. If the
ego position moves while the reported speed is ~0, the node counts an
`ego_twist_contradiction_frames` diagnostic (it does not reject the frame).

## Output

`ad_interfaces/msg/DynamicObjectRiskArray`:

- `header.frame_id = base_link` (the ego body frame, `+x` forward, `+y` left,
  REP-103).
- `header.stamp` = the source `PredictedObjectArray` stamp (one synchronous
  transform; there is no per-object timestamp).
- `objects[]` -- one `DynamicObjectRisk` per admitted predicted object.

An empty `objects` list with a valid header is a valid "no dynamic objects"
result. **No field is ever NaN or Inf**; when a metric is undefined its
`*_valid` flag is `false` and its value fields are `0.0` (never a sentinel
like `inf` or `1e9`).

**Consumer requirement.** A value field is meaningless unless its `*_valid`
flag is checked first. In particular `ttc_s == 0.0` is ambiguous on its own:
with `ttc_valid == true` it means **imminent or current contact** (footprints
already overlap); with `ttc_valid == false` it means **no modelled contact**.
Always branch on `ttc_valid` / `cpa_valid` / `predicted_min_separation_valid`
before reading the paired value.

### Coordinate and sign conventions

`r` and `v_rel` are computed in `odom` and rotated into `base_link` by the ego
yaw. `v_rel = v_object_world - v_ego_world`, with
`v_ego_world = (cos yaw, sin yaw) * longitudinal_speed`.

| field | meaning | example |
| --- | --- | --- |
| `x_rel_m` | object centroid, forward of ego | object directly ahead -> `> 0` |
| `y_rel_m` | object centroid, left of ego | object to ego's left -> `> 0` |
| `vx_rel_mps`, `vy_rel_mps` | relative velocity, forward / left axes | slower lead ahead -> `vx_rel < 0` |
| `range_rate_mps` | **radial closing speed** `-(r . v_rel)/|r|` | positive => straight-line range is decreasing. Sign-stable in every quadrant; this is the primary closing metric. |
| `longitudinal_closing_mps` | forward-gap closure `-sign(x_rel) * vx_rel` | lead ahead moving slower -> `> 0`; follower behind moving faster -> `> 0`; either separating -> `< 0`. |

### TTC definition

Constant-relative-velocity contact time between two **circumscribed
circles**:

- object radius `= max(0.5 * hypot(length, width), minimum_object_radius_m)`;
- ego radius `= hypot(ego_half_length_m, ego_half_width_m)`.

Solve `|r + v_rel * t|^2 = (R_ego + R_obj)^2` for the smallest `t >= 0`.

- footprints already overlapping (`|r| <= R_ego + R_obj`) -> `ttc_valid = true`,
  `ttc_s = 0`;
- no relative motion (`|v_rel| < closing_speed_epsilon_mps`) -> `ttc_valid = false`;
- no real non-negative root, or contact only in the past (diverging) ->
  `ttc_valid = false`;
- contact time beyond `ttc_horizon_s` -> `ttc_valid = false` (reported as
  invalid, not as a large number).

`distance / ego_speed` is deliberately **not** used -- it ignores object
motion. Object orientation is deliberately **not** used -- an oriented-box
TTC would depend on yaw quality and break backend independence; a
conservative circumscribed-circle contract is preferred over false
precision.

### CPA (closest point of approach) definition

Minimiser of `||r + v_rel * t||^2` over `t` in `[0, cpa_horizon_s]`:
`t_cpa = clamp(-(r . v_rel) / |v_rel|^2, 0, cpa_horizon_s)`,
`cpa_distance_m = |r + v_rel * t_cpa|`.

- `|v_rel| < closing_speed_epsilon_mps` -> `cpa_valid = false`;
- an object already separating yields `t_cpa = 0` and `cpa_distance_m` = the
  current range (closest approach is now).

CPA stays meaningful for crossing / near-miss geometry where TTC is
undefined.

### Prediction-horizon minimum separation

Uses the object's **discrete predicted `states[]`** (odom) against a
constant-velocity ego rollout: for each predicted point `(t_k, x_k, y_k)`
with `0 < t_k <= cpa_horizon_s`,
`sep_k = |(x_k, y_k) - (ego + v_ego_world * t_k)|`;
report the minimum and its time. `predicted_min_separation_valid = false`
when the object carries no predicted states. This is kept conceptually
separate from the kinematic CPA above and is intended for later cut-in /
gap-acceptance logic.

The same admitted discrete points are also exposed as
`DynamicObjectRiskState[] predicted_states`. Each state is relative to the
constant-velocity ego rollout at that horizon and expressed in the
source-stamp `base_link` axes. This preserves generic future centroid facts
for route-aware consumers without making them bypass this interface and
subscribe to raw predictions. Invalid/out-of-horizon points are omitted; no
new extrapolation is performed.

### Quality

`position_uncertainty_m` = sqrt of the larger eigenvalue of the
`initial_pose` xy covariance 2x2; `0.0` when the source reports none.

## Invalid / stale data contract

The frame is **rejected** (no publish; a `WARN` diagnostic carries the
reason) when:

| condition | reason string |
| --- | --- |
| stamp `sec < 0` or `nanosec` overflow | `prediction stamp is malformed` |
| stamp `<= 0` | `prediction stamp must be strictly positive` |
| `header.frame_id != odom` | `prediction frame_id is not 'odom'` |
| stamp `<=` last accepted stamp (small backward step) | `prediction is a duplicate or backward frame` |
| stamp `>` now + `maximum_future_skew_s` | `prediction stamp is in the future` |
| now - stamp `> maximum_prediction_age_s` | `prediction is stale` |
| no odometry received yet | `ego state is unavailable` |
| `|prediction stamp - ego stamp| > maximum_ego_age_s` | `ego state is stale relative to the prediction` |
| ego pose / speed non-finite | odometry is dropped on receipt; frame then sees `ego state is unavailable` |

A large backward stamp jump (`> clock_rollback_threshold_ns`, default 0.5 s)
is treated as a simulated-time reset: the node forgets the previous frame and
processes the new one.

An individual **malformed object** (non-finite centroid / velocity, or
non-positive dimensions) is **skipped and counted**
(`rejected_non_finite_state` / `rejected_non_finite_dimensions` in the
diagnostic), and the rest of the frame still publishes -- one bad predicted
object must not blind the planner to a real threat beside it. Objects past
`maximum_objects` are skipped and counted (`rejected_over_budget`).

## Parameters

`config/planning/dynamic_object_risk.yaml`:

| parameter | default | purpose |
| --- | --- | --- |
| `maximum_prediction_age_s` | 0.5 | prediction staleness (matches planner `prediction_timeout_sec`) |
| `maximum_ego_age_s` | 0.5 | ego / prediction stamp skew bound |
| `maximum_future_skew_s` | 0.10 | tolerated future stamp |
| `closing_speed_epsilon_mps` | 0.05 | "no relative motion" threshold for radial rate / TTC / CPA |
| `cpa_horizon_s` | 6.0 | CV rollout horizon for CPA and predicted-min-separation |
| `ttc_horizon_s` | 6.0 | modelled contact beyond this -> invalid |
| `ego_half_length_m` | 2.5 | ego circumscribed-circle half-extent (IONIQ 5 ~4.64 m body) |
| `ego_half_width_m` | 1.1 | ego circumscribed-circle half-extent (~1.89 m body) |
| `minimum_object_radius_m` | 0.30 | floor on the object circle, guards degenerate dims |
| `maximum_objects` | 256 | per-frame compute bound |

There are **no** scenario TTC thresholds here (`ttc_emergency`,
`ttc_warning`, etc.) -- those belong to later policy nodes.

## Launch integration

Standalone: `ros2 launch ad_lidar_perception dynamic_object_risk.launch.py`.

Opt-in from the top-level pipeline (default **off**, no behaviour change):

```bash
ros2 launch ad_lidar_perception lidar_perception.launch.py \
  dynamic_object_risk:=true                       # autoware -> prediction -> risk

ros2 launch ad_lidar_perception lidar_perception.launch.py \
  tracker_backend:=ab3dmot dynamic_object_risk:=true   # ab3dmot -> prediction -> risk
```

The node remains observational. The optional Cut-in Risk v1 fact extractor
consumes `/ad/planning/dynamic_object_risks`; no vehicle-response or
behaviour-tree node consumes it.

## Runtime validation

Bounded replay: 180 frames of `~/datasets/morai_heven` `static_20260805_003151`,
full AB3DMOT -> Prediction -> Dynamic Object Risk. That replay has no
`/ad/localization/odometry`, so a synthetic canonical odometry stream stands
in.

**Arm A -- stationary ego at the odom origin** (physically correct for this
static scene):

- 180 `PredictedObjectArray` in -> 180 `DynamicObjectRiskArray` out; 1334
  predicted objects in -> 1334 risk objects out; 0 rejected frames, 0
  rejected objects; every output frame `base_link`.
- **0 non-finite field values** across all 1334 objects x 14 numeric fields.
- TTC valid 215 (35 with a genuine `ttc_s > 0.05`, min/median/max
  0.06 / 2.50 / 5.96 s; the other 180 are near-origin clusters whose
  footprint overlaps the synthetic stationary-ego footprint -> `ttc_s = 0`).
- CPA valid 645, predicted-min-separation valid 1334.
- relative distance median/p95/max 26.2 / 84.8 / 100.4 m; relative speed
  median/p95/max 0.0 / 14.1 / 19.2 m/s (static scene -> most objects still,
  a few scripted actors moving).
- CPA min-separation median/p10/min 17.7 / 4.3 / 0.18 m;
  predicted-horizon min-separation median/p10/min 19.2 / 1.1 / 0.19 m.
- node latency median/p95/max **0.011 / 0.023 / 0.058 ms**.
- one publisher on every topic before and after; 0 NaN / Inf / exceptions /
  crashes; `ego_twist_contradiction_frames = 0`.

Representative arm-A objects (last frame):

| case | x_rel | y_rel | vx_rel | vy_rel | range_rate | long_closing | TTC | CPA t / d |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| closing lead-like | 4.2 | -2.7 | -6.76 | 3.6 | 7.64 | 6.76 | 0.26 | 0.65 / 0.42 |
| closing lead-like | 7.3 | -2.1 | -6.37 | -- | 7.43 | 6.37 | 0.73 | 0.90 / 2.72 |
| crossing | 12.7 | -3.1 | -3.86 | 5.85 | 5.13 | -- | invalid | 1.36 / 8.88 |
| object behind (parked) | -2.5 | -10.2 | -0.01 | 0.0 | ~0 | ~0 | invalid | 0.0 / 0.0 |

**Arm B -- synthetic ego at constant yaw 0.4 rad, 8 m/s forward** (mechanical
path check only, numbers NOT physical -- the ego drives ~150 m away from a
static scene): 164 frames in/out, 1275 objects in/out, 0 rejected, **0
non-finite**, base_link, one publisher. Every static object shows
`vx_rel = -8.0`, `range_rate ~ -8` (separating) -- confirming the base_link
rotation and the ego-velocity subtraction execute on real object data.
`ttc_valid = 0` (ego is leaving), `cpa_valid = 1275`.

`morai_cam4_20260813_163222` (the only moving-ego local bag) is not usable
here: the host lacks `rosbag2_storage_mcap`.

## Known limitations

- Ego lateral velocity and yaw rate are unobserved by the canonical
  localization output and assumed 0 (non-holonomic). A future yaw-rate-aware
  ego source would tighten crossing / turning geometry.
- `ego_twist_contradiction_frames` (a diagnostic counter for "ego pose moved
  at ~0 reported speed") lives in the odometry callback, outside the pure
  core and `build_risk_frame`; it is not covered by a unit test and neither
  replay arm triggers it (arm A speed 0 with a static pose, arm B a
  consistent moving pose). It is informational only and never rejects a
  frame.
- The bounded replay uses one static scene with a synthetic ego; the moving
  path (arm B) is a mechanical check, not physical validation. Turning-ego
  and moving-ego-vs-moving-object geometry is covered by unit tests, not by
  a live replay.
- TTC uses circumscribed circles -- conservative (over-triggers slightly for
  elongated vehicles passing abeam). An oriented-box model is deferred
  because it would couple the metric to object-yaw quality and break backend
  independence.
- No RViz markers (out of scope for this task).

## Tests

- `test_dynamic_object_risk.cpp` -- 38 cases: 20 geometry (stationary lead,
  slower/same/faster lead, oncoming, crossing both ways, behind
  separating/catching, zero relative velocity, zero objects, multiple
  objects, exact collision, near miss, large lateral, non-finite, ego-yaw
  rotation, determinism) plus radial-sign stability, predicted-min-separation,
  position uncertainty, overlap -> TTC 0, beyond-horizon -> invalid, object
  budget, object-at-origin, and 12 `build_risk_frame` invalid/stale
  contract cases.
- `test_dynamic_object_risk_launch.py` -- live node: finite base_link metrics,
  single publisher, physically sensible lead-object values.
- `test_interface_contract.py` -- 7/7; `DynamicObjectRisk` and its policy-free
  future-state extension are stable and contain no GO / STOP / YIELD / brake /
  risk_score / decision field.
- `test_lidar_perception_launch.py` -- opt-in include, backend-agnostic,
  ordered after prediction.
