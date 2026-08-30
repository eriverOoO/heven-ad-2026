# Competition Dynamic-Object Pipeline

This extends **Competition MOT Baseline v1**
(`docs/perception/competition_mot_baseline.md`) by connecting the merged,
model-free AB3DMOT tracker to the existing HEVEN prediction and dynamic
occupancy stages. Nothing new is invented: the AB3DMOT `TrackedObjects`
output is fed into the same `ad_autoware_prediction` node and the same
`ad_dynamic_occupancy_grid` / `ad_combined_occupancy_grid` nodes the
default Autoware path already uses.

Autoware remains the default tracker. No learned model, checkpoint, GPU,
or training step is required anywhere on this path.

## Architecture

```text
LiDAR
  -> self crop
  -> Patchwork++ ground segmentation
  -> finite-point filter
  -> Adaptive Euclidean clustering        -> /ad/perception/objects/detected  (DetectedObjects, lidar_link)
  -> AB3DMOT  (Euclidean BEV 3.0 m gate, Hungarian, Linear KF,
               yaw unobserved, min_hits=1, max_age=2)
                                          -> /ad/perception/objects/tracked   (TrackedObjects, odom)
  -> HEVEN stateful IMM prediction        -> /ad/perception/objects/predicted (PredictedObjectArray, odom)
  -> dynamic occupancy grid               -> /ad/perception/occupancy/dynamic (OccupancyGrid, base_link)
static occupancy + dynamic occupancy      -> /ad/perception/occupancy/combined (OccupancyGrid, base_link)
```

Both `tracker_backend` selections converge on the identical downstream
launches (`prediction.launch.py`, `dynamic_occupancy_grid.launch.py`,
`combined_occupancy_grid.launch.py`). There is no AB3DMOT-specific
prediction or occupancy node.

## Runtime backend switch

```bash
# Default and explicit Autoware are identical: Autoware tracker + prediction
# + dynamic/combined occupancy.
ros2 launch ad_lidar_perception lidar_perception.launch.py
ros2 launch ad_lidar_perception lidar_perception.launch.py tracker_backend:=autoware

# Opt-in model-free path: AB3DMOT replaces Autoware as the single canonical
# tracker, then feeds the same prediction and occupancy stack.
ros2 launch ad_lidar_perception lidar_perception.launch.py tracker_backend:=ab3dmot
```

`tracker_backend:=ab3dmot` still rejects a non-Euclidean detector. It
starts exactly one canonical publisher on `/ad/perception/objects/tracked`
(AB3DMOT, not Autoware) and one on `/ad/perception/objects/predicted`
(the shared prediction node).

## Topics, messages, frames, timestamps

| Topic | Message | Frame | Stamp |
| --- | --- | --- | --- |
| `/ad/perception/objects/detected` | `autoware_perception_msgs/DetectedObjects` | `lidar_link` | LiDAR scan stamp |
| `/ad/perception/objects/tracked` | `autoware_perception_msgs/TrackedObjects` | `odom` | detection stamp, reused unchanged |
| `/ad/perception/objects/predicted` | `ad_interfaces/PredictedObjectArray` | `odom` | tracked-array stamp, copied unchanged |
| `/ad/perception/objects/prediction_debug` | `diagnostic_msgs/DiagnosticArray` | `odom` | tracked-array stamp |
| `/ad/perception/occupancy/dynamic` | `nav_msgs/OccupancyGrid` | `base_link` | prediction stamp |
| `/ad/perception/occupancy/combined` | `nav_msgs/OccupancyGrid` | `base_link` | static-layer stamp, must equal dynamic stamp |

The prediction node requires the tracked frame to be exactly `odom`,
the stamp to be strictly positive and monotonically increasing, not in the
future, and at most `maximum_input_age_sec` (0.5 s) old. A backward stamp
resets the per-UUID IMM history (MORAI restarts simulated time with its
tracks). Dynamic occupancy requires the predicted frame to be exactly
`odom` and looks up `base_link <- odom` at the prediction stamp within
`transform_timeout_sec` (0.05 s); it clears the layer on any stale, out of
order, or invalid input rather than publishing wrong data.

## Yaw / orientation behavior

Adaptive Euclidean clustering never observes object heading. The AB3DMOT
baseline drops yaw from the KF measurement, keeps the latent yaw at zero,
and publishes `orientation_availability = UNAVAILABLE` with an identity
quaternion.

The prediction adapter needs **no compatibility change** for this:

- `convert_object` never rejects an object on the availability flag.
- It reads yaw from the pose quaternion; the identity quaternion yields
  yaw `0`, so rotating the object-local twist to world (`R(+yaw)`) and the
  serialization-side `R(-yaw)` are both the identity. World Cartesian
  motion is preserved exactly, and no heading is fabricated.
- The IMM consumes the (always zero, always consistent) yaw as a
  measurement with a bounded fallback variance, so the latent yaw stays at
  zero and never drives a spurious turn. In the bounded replay the IMM
  selected only `constant_velocity` and `stationary`; `coordinated_turn`
  was never selected.

New focused tests
(`test_autoware_prediction_adapter.cpp:Ab3dmotOrientationUnavailable*`)
assert the accept-as-Cartesian behavior and the no-forced-turn behavior.

## Velocity semantics

AB3DMOT stores Cartesian velocity and covariance in `odom`. The ROS
adapter rotates them into object-local axes by `R(-yaw)` for
`TrackedObject.twist` (matching Autoware). Prediction rotates the incoming
local twist back to world by `R(+yaw)`. The intended round trip is

```text
AB3DMOT world velocity -> local TrackedObjects twist -> prediction -> world velocity
```

**Live ROS check (bounded replay).** For every tracked object the AB3DMOT
node also published its pre-serialization world velocity on a diagnostic
audit topic (`velocity_audit_enabled`, on only for the `ab3dmot`
backend). Reconstructing the world velocity from the published local twist
and pose yaw and comparing to the audited value over 1318 object-frames
(651 with speed > 1 m/s, up to 20.4 m/s):

```text
max |Δvx| = 0.000e+00   max |Δvy| = 0.000e+00   max |Δv| = 0.000e+00
```

Because this baseline's yaw is identically zero, both rotations are the
identity, so this confirms the serialization *bookkeeping* end to end but
does not exercise a non-trivial rotation. The rotation math itself is
covered by the merged baseline's focused serialization tests. The IMM
`initial_twist` is the filtered `fused_state`, so it legitimately deviates
from the raw measurement (up to ~13 m/s at track birth in this replay);
that is filter behavior, reported separately, not round-trip error.

## Prediction behavior

Unchanged `ad_autoware_prediction`: stateful IMM (stationary /
constant-velocity / coordinated-turn, initial probabilities
0.20 / 0.60 / 0.20), 12 horizons from 0.5 s to 6.0 s, 1.0 s track
retention, 2.0 s maximum update interval. The predicted array header is
copied from the tracked array, so stamps and the `odom` frame pass
through unchanged.

## Dynamic OGM behavior

Unchanged `ad_dynamic_occupancy_grid`: 1040 x 200 cells, 0.1 m, origin
`(-4, -10)`, identity orientation, `base_link`. Each object's current
footprint is rasterized and inflated by the larger of `minimum_inflation_m`
(0.20 m) and the 2-sigma position covariance. Future states are validated
but not rasterized (the grid has no time axis). Invalid / stale / out of
order input clears the layer.

Checked-in `dynamic.yaml` keeps `road_gate.enabled: true`, so in
production the node pairs each prediction with an exact-stamp
`/ad/planning/drivable_mask` from the planning stack, exactly as on the
Autoware path. With no planning stack running, the node buffers and does
not publish - identical to the current Autoware behavior without planning.

## Combined OGM status

`ad_combined_occupancy_grid` pairs `/ad/perception/occupancy/static` and
`/ad/perception/occupancy/dynamic` by exact stamp and compatible geometry.
Because AB3DMOT preserves the detection stamp all the way to the dynamic
grid, the static and dynamic stamps coincide and the combined layer
publishes. This was validated in the bounded replay (171 / 171 combined
grids, geometry valid, no invalid cells).

## Validation dataset and window

`~/datasets/morai_heven`, source export `static_20260805_003151`, first
180 exported LiDAR frames (the same window as the Competition MOT Baseline
v1 smoke). Replayed from the committed XYZI export at ~6 Hz through:
self-crop-bypass -> Patchwork++ -> finite filter -> Adaptive Euclidean ->
AB3DMOT -> prediction -> dynamic occupancy, plus static and combined
occupancy. A replay-only static `odom -> base_link -> lidar_link` chain
was used. `road_gate.enabled` was set to `false` for static and dynamic
occupancy for this run because the points/labels export contains no
`/ad/planning/drivable_mask`; this is the node's own supported unmasked
mode, not a fabricated mask.

## Runtime metrics (bounded replay, execution evidence only)

| Stage | messages | median step | p95 step | max step |
| --- | --- | --- | --- | --- |
| AB3DMOT tracker | 172 `TrackedObjects` (172 unique monotonic stamps) | 0.822 ms | 2.323 ms | 3.356 ms |
| HEVEN prediction | 171 `PredictedObjectArray` (12 states each) | 0.125 ms | 0.299 ms | 0.539 ms |
| Dynamic occupancy | 171 unique-stamp grids | 0.410 ms | 0.898 ms | 1.456 ms |
| Combined occupancy | 171 unique-stamp grids | - | - | - |

- Predicted stamps are a subset of tracked stamps (exact-stamp pairing);
  all predicted / dynamic / combined output is `odom` / `odom` /
  `base_link` as contracted.
- Prediction rejected 1 array (`stamp is in the future`, a first-frame
  replay-clock transient); 0 rejections were orientation-related; every
  admitted object (1207+) reported `orientation_availability = UNAVAILABLE`
  and still produced a prediction.
- Occupied dynamic cells: median 4338, p95 24398, max 50026. Zero
  non-finite values, zero out-of-range cells, zero indexing exceptions,
  zero node crashes across all stages.
- Exactly one publisher on `/ad/perception/objects/tracked`,
  `/predicted`, `/occupancy/dynamic`, and `/occupancy/combined` before and
  after the replay.
- Component latencies only; detector inference is excluded and no
  end-to-end callback-to-publish figure was separately instrumented in
  this run.

## Known limitations

1. **Execution / interface evidence only** - one static-scene window, no
   HOTA / AssA / IDSW, no accuracy or obstacle-avoidance claim.
2. **Drivable-mask-gated occupancy not exercised against real planning.**
   No repository-local MORAI replay contains `/ad/planning/drivable_mask`,
   and `ros2 bag play` for the one `.mcap` bag is unavailable on this host
   (`rosbag2_storage_mcap` missing). The exact-stamp mask pairing,
   geometry-mismatch rejection, and stale-mask rejection remain covered by
   `test_occupancy_layer_launch.py` with a synthetic mask driver.
3. **Covariance-inflated footprints.** The Linear-KF AB3DMOT tracks carry
   large position covariance on young / coasting tracks; the 2-sigma
   inflation can push a coarse Euclidean AABB past
   `maximum_cells_per_object`, and the dynamic node then safely clears
   that frame (77 of 171 grids in the replay were empty, at least 13 from
   this path). With `road_gate.enabled` and a real drivable mask most of
   these off-road clusters would be masked out. Not tuned here (Phase 13);
   a covariance cap or occupancy inflation review is a separate task.
4. **From-scratch replay boundary loss.** Feeding Patchwork++ a cold 6 Hz
   stream dropped ~8 of 180 frames at the start / stop boundaries;
   Patchwork++ re-emitted the final buffered frame, and AB3DMOT correctly
   rejected the duplicate / backward stamps. The dedicated Competition MOT
   Baseline v1 replay harness reached 180 / 180.
5. **Boundary yaw.** The velocity round trip is numerically exact only
   because yaw is identically zero here; a detector that observes yaw
   would exercise the rotation, which is unit-tested but not replayed.

## Readiness

**Ready as an opt-in, model-free dynamic-object pipeline.** HEVEN can run
LiDAR -> Adaptive Euclidean -> AB3DMOT -> HEVEN prediction -> dynamic (and
combined) occupancy entirely through the canonical ROS interfaces, with
Autoware still the default tracker and no learned model required. The
chain is mechanically validated by bounded runtime replay. Accuracy
benchmarking and drivable-mask-gated occupancy against a real planning
stack are separate follow-up tasks.
