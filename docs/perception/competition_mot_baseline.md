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

Input stamps must be strictly positive. Duplicate stamps are rejected without
publication. A backward stamp resets tracker state before processing the new
epoch. The KF transition uses the actual positive elapsed seconds between
successive accepted message stamps, so velocity remains m/s rather than
meters/frame.

## Yaw and velocity

Adaptive Euclidean clustering has no observed object heading. Its identity
quaternion is structural, so the baseline drops yaw from the KF measurement,
initializes the latent yaw to zero, and publishes
`orientation_availability=UNAVAILABLE`.

AB3DMOT internally represents Cartesian velocity and covariance in `odom`.
Autoware `TrackedObject.twist` uses object-local axes. The ROS adapter rotates
velocity and its covariance by `R(-yaw)` during serialization, matching the
default Autoware tracker. With this baseline's unavailable zero yaw, the
rotation is numerically an identity. A downstream consumer applying
`R(yaw)` recovers the original world motion and covariance.

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
production default. Prediction and Dynamic OGM are deliberately deferred even
though the twist basis is now consistent; enabling them requires a separate
end-to-end task and must not be inferred from this smoke result.
