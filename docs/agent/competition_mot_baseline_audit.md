# Competition MOT Baseline v1 audit

## Repository state and scope

Work resumed from `main` at
`2ff2301541cf0602912042bdcd87caf4e2b75f10`. The working tree already
reported 1,918 modified tracked files. An end-of-line-insensitive numstat
isolated substantive changes to 11 LiDAR-perception files; the other 1,907
tracked files were CRLF-only noise. The three untracked files were the two
new baseline configs and its focused test. No unknown substantive user edit
was found. Work continued on `feat/competition-mot-baseline-v1`; unrelated
line-ending changes were neither normalized nor staged.

The existing AB3DMOT integration was retained: the pinned upstream
`references/ab3dmot` implementation supplies the 10-state/7-measurement
linear KF, while HEVEN owns the ROS adapter, real-time `dt`, association,
and lifecycle integration. No Prediction, Dynamic OGM, IMM, occupancy,
detector, or planning algorithm was changed.

## Branch audit

The prior branch audit covered `main`,
`agent/lidar-replay-ground-segmentation`, `chore/tracking-references`,
`feat/ab3dmot-tracker`, `feat/centerpoint-milestone`,
`feat/kalmannet-tracker`, `feat/tracking-preset-replay`, and
`research/tracking-audit`. Only `feat/centerpoint-milestone` diverged, and
its MORAI replay/DetectedObjects recording utilities have newer equivalents
on `main`. Verdict: **SUPERSEDED — DO NOT PORT**.

## Main integration audit

The checked-in default composition still selects Adaptive Euclidean
clustering plus Autoware tracking. A new top-level `tracker_backend` launch
argument accepts a blank value (use the composition), `autoware`, or
`ab3dmot`. The AB3DMOT branch is accepted only with the Adaptive Euclidean
detector and explicitly supplies every Competition MOT setting:

- Euclidean BEV association, 3.0 m gate
- Hungarian assignment
- Linear KF
- yaw measurement unobserved
- `min_hits=1`, `max_age=2`

That branch publishes the canonical tracked topic but deliberately does not
start Autoware tracking, Prediction, Dynamic OGM, or combined occupancy.
Static occupancy remains independent. Launch-action tests and live graph
inspection prove one canonical publisher in all three selection cases.

## CRLF incident and containment

The first replay in the preceding session found that repository-wide CRLF
conversion made an installed Python executable request `python3\r`. Only the
required `ad_ab3dmot_tracker` executable was restored to an LF shebang, and
the isolated overlay build then succeeded. During this session the unrelated
`ad_publish_morai_frames` helper showed the same pre-existing shebang issue;
it was not modified. The bounded replay invoked a temporary interpreter-owned
publisher instead. No global normalization is part of this change.

## Velocity contract finding

AB3DMOT's pinned KF state stores Cartesian `vx, vy, vz` in the tracker/world
frame. Autoware's multi-object tracker stores world velocity internally but
rotates it and its covariance by `R(-yaw)` when serializing
`TrackedObject.twist`; its documentation calls this the vehicle-coordinate
output. HEVEN's prediction adapter performs the inverse `R(yaw)` operation,
treating incoming tracked twist as object-local before producing world-frame
motion. The previous direct AB3DMOT copy therefore violated the canonical
message contract when yaw was nonzero.

The smallest fix is at `ab3dmot_ros.py`: rotate the world-frame velocity and
3x3 covariance into object-local axes at serialization. The tracker core is
unchanged. Focused tests cover a 90-degree case, covariance rotation, and the
same local-to-world round trip used by prediction. In the Competition
baseline, yaw is unobserved, initialized to zero, and published with
orientation unavailable; its placeholder axes therefore coincide with
`odom`, so the conversion is numerically an identity while remaining
contract-correct.

## Bounded runtime smoke

Dataset: the first 180 exported frames from the existing
`static_20260805_003151` MORAI LiDAR replay, sample stamps
`1785857513201006723` through `1785857541916511060` ns. The temporary replay
publisher reconstructed the repository's strict XYZIRT input layout (zero
ring/time only because the committed export contains finite XYZI) and kept
the original source stamps. A replay-only static
`odom -> base_link -> lidar_link` chain used the checked-in MORAI mount.

The exact graph was crop -> Patchwork++ -> finite filter -> Adaptive
Euclidean clustering -> AB3DMOT. Results:

- 180 DetectedObjects messages, 870 detections, frame `lidar_link`
- 180 TrackedObjects messages, 1,142 published track states, frame `odom`
- tracked stamp sequence exactly equal to the detected stamp sequence;
  both strictly monotonic
- 255 tracks created, 246 deleted, 9 live at end
- 180 tracker steps; median/p95/max step latency
  0.755203/1.825568/3.827874 ms
- zero non-finite detected/tracked objects; no exception or shebang failure
- one detected publisher and one canonical tracked publisher
- observer and publisher exited with status 0; the bounded graph was stopped
  and no test process remained

This is execution and interface evidence, not HOTA/IDSW or accuracy
validation.

## Readiness

Competition MOT Baseline v1 is ready as an opt-in, model-free tracking
baseline. Autoware remains the default. Velocity ambiguity is resolved at the
message boundary, but Prediction and Dynamic OGM remain intentionally
disconnected: enabling them is a separate major-variable task requiring its
own orientation-unavailable and downstream occupancy runtime validation.
