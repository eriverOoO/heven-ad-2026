# Competition MOT Baseline v1

Competition MOT Baseline v1 is an opt-in, model-free LiDAR tracking path.
It does not replace the default Autoware tracker.

```text
LiDAR
  -> self crop
  -> Patchwork++ ground segmentation
  -> finite-point filter
  -> Adaptive Euclidean clustering
  -> DetectedObjects
  -> AB3DMOT
  -> Euclidean BEV association (3.0 m)
  -> Hungarian assignment
  -> Linear KF
  -> TrackedObjects
```

## Exact configuration

| Setting | Value |
| --- | --- |
| Detector | Adaptive Euclidean clustering |
| Tracker | AB3DMOT |
| Association | Euclidean BEV center distance |
| Gate | 3.0 m |
| Matcher | Hungarian |
| Estimator | Linear constant-velocity KF |
| Yaw measurement | unobserved |
| `min_hits` | 1 |
| `max_age` | 2 frames |

The authoritative files are
`config/lidar_perception_competition_mot_baseline_v1.yaml` and
`config/tracking/competition_mot_baseline_v1.yaml`. Settings are explicit;
the baseline does not inherit historical AB3DMOT defaults.

## Launch and fallback behavior

The normal command and an explicit Autoware selection are equivalent:

```bash
ros2 launch ad_lidar_perception lidar_perception.launch.py

ros2 launch ad_lidar_perception lidar_perception.launch.py \
  tracker_backend:=autoware
```

Both start Autoware tracking, HEVEN prediction, and configured dynamic
occupancy. Neither starts AB3DMOT.

Select the competition baseline explicitly:

```bash
ros2 launch ad_lidar_perception lidar_perception.launch.py \
  tracker_backend:=ab3dmot
```

Alternatively, pass the baseline composition file as
`composition_config`. The AB3DMOT selection rejects a non-Euclidean detector.
It starts neither Autoware tracking nor Prediction/Dynamic OGM, preventing a
duplicate canonical publisher and keeping downstream integration out of v1.

## ROS contract

- Input: `/ad/perception/objects/detected`,
  `autoware_perception_msgs/msg/DetectedObjects`
- Output: `/ad/perception/objects/tracked`,
  `autoware_perception_msgs/msg/TrackedObjects`
- Detection frame: normally `lidar_link`
- Tracker/output frame: `odom`

For each nonempty input, the tracker requests the
`odom <- input_header.frame_id` TF at the input message's exact stamp. It
transforms detections before association and publishes `odom`. Empty inputs
do not require TF. The output reuses the input stamp object unchanged.

Input stamps must be strictly positive. Duplicate and backward stamps are
rejected without publication or tracker-state mutation. The KF transition
uses the actual positive elapsed seconds between successive accepted message
stamps, so velocity remains m/s rather than meters/frame.

## Yaw and velocity

Adaptive Euclidean clustering has no observed object heading. Its identity
quaternion is structural, so the baseline drops yaw from the KF measurement,
initializes the latent yaw to zero, and publishes
`orientation_availability=UNAVAILABLE`.

The yaw-yaw variance is serialized at `PoseWithCovariance.covariance` flat
index **35** (`5*6+5`), matching Autoware's convention and
`autoware_prediction_node.cpp` (which reads index 35). It previously went to
index 21 (roll-roll), so prediction read `0` there and fell back to its
`0.04 rad^2` default -- effectively trusting the always-zero placeholder yaw
to +/- ~11 deg. Since `yaw_measurement_mode=unobserved` never corrects the
latent yaw, the KF yaw variance `P[3,3]` is the birth prior plus process
noise (>= 10 rad^2, std > pi), so prediction now correctly treats the yaw as
unknown. `orientation_availability` stays `UNAVAILABLE`; index 21 stays `0`.
This changed no model selection in the bounded replay and introduced no
coordinated-turn behavior (turn selections 0 -> 0).

### Yaw-rate contract

The Linear KF has **no yaw-rate state**. No AB3DMOT estimator backend routes
a yaw rate into `TrackedState` or `TrackedObject.twist.angular`. The
serializer therefore publishes:

- `twist.twist.angular = 0` -- this is the constant-velocity model's actual
  output for a non-rotating track, not a fabricated measurement.
- `twist.covariance[35]` (`wz-wz`, `(rad/s)^2`) **left unset (0.0)**. A
  non-positive covariance entry is the ROS / `positive_variance(value,
  fallback)` encoding for "not provided". AB3DMOT genuinely has no yaw-rate
  covariance, so `0.0` is the correct encoding of that fact.

`autoware_prediction_node.cpp` then substitutes its shared
`positive_variance(twist.covariance[35], 0.04)` default -- **the same
`0.04 (rad/s)^2` every tracker (Autoware included) gets for an omitted
field.** This default is behaviorally inert for a zero-yaw-rate track:
coordinated-turn selection in `imm_predictor.cpp` is gated on the yaw-rate
*value* (`turn_evidence = |observed[kYawRate]|`, always `0` here, so the
`-1.0` log-likelihood penalty applies every frame), never on the yaw-rate
*variance*. A deterministic offline replay of the recorded 1257-object
TrackedObjects stream through a faithful `imm_predictor` port, sweeping
`twist.covariance[35]` across `{0.04, 0.25, 1.0, 3.0, 4.0, 9.0, 100.0}`
(`(rad/s)^2`), changed **0 coordinated-turn selections** (0 at every value),
introduced **0 curved predicted trajectories**, and left the fused yaw rate
at exactly `0` for every track at every horizon. The only measurable effect
of a larger variance was a sub-20% rise in the coordinated-turn *probability*
on a single track (never competitive with CV/stationary) and a mild
long-horizon position blend shift (6 s horizon, `<= ~0.2 m` mean) as the
`log(det S)` likelihood term slightly relaxes the stationary model's
advantage over CV -- both non-rotating models.

**Contract decision: no code change.** AB3DMOT publishes a truthful zero
yaw rate under an explicit non-rotating assumption, and the shared `0.04`
default is not misinterpreted as observed turning information because turn
detection is driven by the yaw-rate value, not its variance. The **rad^2**
yaw-*angle* variance (`pose.covariance[35]`, `>= 10` here) is deliberately
**not** copied into the **(rad/s)^2** yaw-*rate* slot. Regression tests
(`test_ab3dmot_ros.py::YawRateContractTest`,
`test_imm_predictor.cpp::ZeroYawRateNeverSelectsCoordinatedTurnRegardlessOfVariance`)
lock this.

**Known limitation (shared, not AB3DMOT-specific).** The `0.04` default
gives the stationary model a small `log(det S)` edge over CV via the
yaw-rate term (stationary's `variance[kYawRate]` is clamped to `0.02` in
`interact_and_predict`). Measured effect on the offline stream: two of 1257
object-frames flip `stationary <-> constant_velocity`; both models predict
straight lines. This is a shared-IMM likelihood detail; correcting the
`0.04` default belongs with the shared prediction node, not a per-tracker
`twist.covariance[35]` override (which would need an underivable constant
and would split the "field not provided" treatment between AB3DMOT and
Autoware).

AB3DMOT internally represents Cartesian velocity and covariance in `odom`.
Autoware `TrackedObject.twist` uses object-local axes. The ROS adapter rotates
velocity and its covariance by `R(-yaw)` during serialization, matching the
default Autoware tracker. With this baseline's unavailable zero yaw, the
rotation is numerically an identity. A downstream consumer applying
`R(yaw)` recovers the original world motion and covariance.

The published *position* covariance is additionally clipped so no direction's
reported standard deviation exceeds `maximum_position_std_m` (7.0 m for this
baseline). A born-then-lost single-hit track coasts one `predict()` step with
the reference AB3DMOT velocity prior (variance 10000), which `dt^2` propagates
into a ~360 m^2 (std ~19 m) position variance for the one frame it is
published at `time_since_update=1`. The clip is a PSD-preserving reporting
bound at the ROS boundary only -- the internal Kalman filter is unmodified
(proven byte-identical) -- so HEVEN prediction and the Dynamic OGM do not
inflate that one-frame reachability blob. See
`competition_dynamic_object_pipeline.md` "Position covariance contract".

## Lifecycle and observability

Tracks publish after the first hit (`min_hits=1`). An unmatched track coasts
for one frame and is removed on the second consecutive miss
(`max_age=2`, the existing frame-based AB3DMOT rule). Every 180 tracker steps,
the node logs births, deletions, live tracks, and median/p95/max step latency.
This instrumentation does not affect timestamp input or tracker state.

## Dependencies and model guarantee

Runtime requires ROS 2 Humble, NumPy, SciPy (Hungarian assignment), FilterPy,
the pinned `references/ab3dmot` submodule, and the normal HEVEN LiDAR
preprocessing dependencies. Patchwork++ is used by the checked-in default
ground configuration. Recursive submodule checkout must be complete.

No checkpoint, neural network, GPU, training step, or external model weight is
used by this path.

## Validated smoke result

The first 180 exported frames of the existing `static_20260805_003151` MORAI
LiDAR replay produced 180/180 DetectedObjects and 180/180 TrackedObjects
messages. Stamps were strictly monotonic and exactly preserved, frames were
`lidar_link -> odom`, and one canonical publisher was present. The run created
255 tracks, deleted 246, and ended with 9 live tracks. Tracker-step latency was
0.755203 ms median, 1.825568 ms p95, and 3.827874 ms maximum. No NaN, Inf,
exception, or executable-shebang failure occurred.

## Limitations and readiness

This is a runtime/interface smoke test, not an accuracy benchmark. The replay
used one existing static-scene window and cannot establish HOTA, AssA, IDSW,
or generalization. Euclidean detections do not observe yaw, and lifecycle is
frame-based. The temporary replay reconstructed ring/time fields because the
committed finite export retains XYZI only; no deskew claim is made.

Status: **ready as an opt-in Competition MOT baseline**. Autoware remains the
production default. Prediction and Dynamic OGM are now connected to this path
by a separate end-to-end integration; see
`docs/perception/competition_dynamic_object_pipeline.md`.
