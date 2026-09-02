# Training-Free Perception + Camera RViz Demo v1

One opt-in launch command brings up the HEVEN **model-free** perception stack
and shows it in RViz, with the front camera image in the same session.

Nothing in this demo needs CenterPoint, KalmanNet, a trained-weights file, a
MORAI training dataset, `torch`, CUDA, or OpenPCDet. It needs live **sensor
input** or an existing **rosbag** (a rosbag is runtime replay input, not a
training dataset).

## What runs

| stage | node | training-free choice |
| --- | --- | --- |
| preprocessing | self-crop + finite filter | geometric |
| ground segmentation | RANSAC (`ground_segmentation.launch.py backend:=ransac`) | classical |
| detection | adaptive Euclidean clustering | **no learned detector** |
| association | Euclidean BEV centre-distance gate (3.0 m) + Hungarian | `association_metric:=euclidean matcher:=hungarian` |
| estimation | Linear KF, AB3DMOT lifecycle (`min_hits`/`max_age`) | `state_estimator:=linear_kf`, `yaw_measurement_mode:=unobserved` |
| prediction | analytical CV / CT / IMM | `ad_autoware_prediction_node` (no learned model) |
| occupancy | dynamic + combined occupancy grids | `nav_msgs/OccupancyGrid` |
| planner-facing facts | Dynamic Object Risk (observational, opt-in on by default here) | no behaviour change |
| visualization | 2× `ad_viz/perception_visualizer_node` (read-only) + RViz | markers only |

The demo reuses `lidar_perception.launch.py` with `tracker_backend:=ab3dmot`
over the checked-in `config/lidar_perception.yaml` composition. **No production
default is changed.** The AB3DMOT arm publishes tracked objects on
`/ad/perception/objects/tracked` (same topic Autoware would use), so prediction,
the occupancy grids and the primary `A-` visualizer all consume the
training-free tracker directly.

## Topics

| topic | type | producer |
| --- | --- | --- |
| `/ad/sensors/lidar/points` | `sensor_msgs/PointCloud2` | sensor / bag (**required**) |
| `/ad/perception/lidar/cropped` | `sensor_msgs/PointCloud2` | self-crop |
| `/ad/perception/lidar/ground`, `/nonground` | `sensor_msgs/PointCloud2` | ground segmentation |
| `/ad/perception/objects/detected` | `autoware_perception_msgs/DetectedObjects` | Euclidean clustering (frame `lidar_link`) |
| `/ad/perception/objects/tracked` | `autoware_perception_msgs/TrackedObjects` | AB3DMOT (frame `odom`) |
| `/ad/perception/objects/predicted` | `autoware_perception_msgs/PredictedObjects` | prediction |
| `/ad/perception/occupancy/{dynamic,combined,static}` | `nav_msgs/OccupancyGrid` | occupancy grids |
| `/ad/planning/dynamic_object_risks` | `ad_interfaces/DynamicObjectRiskArray` | Dynamic Object Risk |
| `/ad/visualization/{detected,tracked,predicted}_objects` | `visualization_msgs/MarkerArray` | `ad_viz` (read-only) |
| `/ad/sensors/camera/front/compressed` | `sensor_msgs/CompressedImage` | sensor / bag (camera panel) |
| `/ad/localization/odometry` | `nav_msgs/Odometry` | localization / bag / `enable_localization:=true` (**required**: `odom` frame + OGM/risk) |
| `/tf`, `/tf_static` | `tf2_msgs/TFMessage` | localization + `ad_description` / bag |

## TF chain

```
odom -> base_link              (localization, or the bag)
base_link -> rear_axle_link -> lidar_link              (static, ad_description robot_state_publisher)
             rear_axle_link -> camera_front_link -> camera_front_optical_frame   (static)
```

RViz Fixed Frame is `odom`. In `input_mode:=replay`, `ad_description` and the
bag's `/tf` supply this chain. In `input_mode:=live` you must have a
localization stack publishing `odom -> base_link` and `/ad/localization/odometry`
(the demo does not own sensor drivers or localization).

## Camera: Level A only (2D image, not projected)

The camera image is shown in the **same RViz session** as the 3D LiDAR scene
(Level A). It is **not** geometrically projected into the 3D view (Level B).

Reason: there is **no `CameraInfo` publisher anywhere in the repository**. The
camera↔LiDAR extrinsics exist in TF (`camera_front_link` /
`camera_front_optical_frame` are static children of `rear_axle_link`, siblings
of `lidar_link`), but the intrinsics are only *reconstructed* (fx = fy = 640,
cx = 640, cy = 360, from HFOV 90° and 1280×720 in
`ad_description/config/sensor_mounts.yaml`). Projecting 3D boxes onto the image
or lifting 2D boxes to 3D from a reconstructed `K` would be a fabricated
geometric claim, so this demo does not do it. The camera and LiDAR are visible
together; they are not fused here, and camera does not affect tracking.

An optional 2D YOLO overlay (`enable_camera_perception:=true`) publishes
`/ad/viz/perception/camera/dynamic_obstacle` — a second RViz Image panel,
disabled by default. It needs `ultralytics`/`torch` and the separately-optional
`ad_camera_perception` package and is **not** part of the model-free demo.

## Commands

All commands assume the workspace overlay is sourced.

### A. Live LiDAR only

```
ros2 launch ad_lidar_perception training_free_perception_rviz.launch.py
```

Requires a running LiDAR driver on `/ad/sensors/lidar/points` and a
localization stack (`odom -> base_link`, `/ad/localization/odometry`).

### B. Rosbag LiDAR replay

```
ros2 launch ad_lidar_perception training_free_perception_rviz.launch.py \
    input_mode:=replay \
    bag_path:=/absolute/path/to/rosbag_dir \
    rate:=0.5
```

`bag_path` is an extracted ROS 2 **MCAP** bag directory containing
`metadata.yaml`. The bag must contain `/ad/sensors/lidar/points`, `/tf`,
`/tf_static` and `/ad/localization/odometry` — **or**, for a bag that has raw
MORAI ego sensors but no recorded odometry/dynamic TF, add
`enable_localization:=true` (see "Replaying a bag without recorded
localization" below).

### C. LiDAR + front camera (replay)

```
ros2 launch ad_lidar_perception training_free_perception_rviz.launch.py \
    input_mode:=replay \
    bag_path:=/absolute/path/to/rosbag_dir \
    enable_camera:=true
```

`enable_camera:=true` adds `/ad/sensors/camera/front/compressed` to the bag
playback. In `input_mode:=live` the camera panel is always present and is fed by
the camera driver if one is running.

### D. LiDAR only when no camera is available

Use command A or B. The RViz "Front Camera" panel simply stays blank; every
LiDAR display still works.

### E. Replay a bag that has raw ego sensors but no recorded localization

```
ros2 launch ad_lidar_perception training_free_perception_rviz.launch.py \
    input_mode:=replay \
    bag_path:=/absolute/path/to/rosbag_dir \
    rate:=0.5 \
    enable_camera:=true \
    enable_localization:=true
```

See "Replaying a bag without recorded localization" below.

### Optional flags

| flag | default | effect |
| --- | --- | --- |
| `enable_dynamic_object_risk` | `true` | start the observational Dynamic Object Risk node |
| `enable_camera_perception` | `false` | also start the YOLO 2D overlay (needs torch) |
| `enable_localization` | `false` | replay-only: also replay `/ad/sensors/gps/fix` + `/ad/vehicle/status` and start `ad_localization`'s `gnss_imu` backend, for a bag with raw ego sensors but no recorded `/ad/localization/odometry` |
| `start_rviz` | `true` | set `false` for a headless graph |
| `loop` | `true` | replay the bag repeatedly |
| `start_paused` | `false` | start the bag paused |
| `rviz_config` | demo config | absolute path to an alternate RViz config |

## Replaying a bag without recorded localization

Some bags (e.g. a MORAI camera/LiDAR capture that never had the localization
stack running) record raw ego sensors — `/ad/sensors/gps/fix`,
`/ad/sensors/imu/data`, `/ad/vehicle/status` — but no
`/ad/localization/odometry` and no dynamic `/tf`. Without those, AB3DMOT has
no `odom` transform for `lidar_link` and correctly rejects every detection
(tracked/predicted/occupancy stay empty) — this is not a bug, it is the
tracker refusing to build a track on unknown ego pose.

`enable_localization:=true` (replay only) additionally replays the bag's raw
`/ad/sensors/gps/fix` and `/ad/vehicle/status` topics and starts the
repository's own already-existing production `ad_localization` package
(`localization.launch.py`, default `gnss_imu` backend) inside the replay's
sim-time scope. That backend consumes real GNSS position and IMU orientation
and its `localization_manager_node` broadcasts a real dynamic
`odom -> base_link` TF via `tf2_ros::TransformBroadcaster`, plus a static
`map -> odom`. **No odometry or TF is fabricated by this demo** — it only
launches the existing, unmodified localization converter against the bag's
own recorded sensor data. `enable_localization` defaults to `false` and has
no effect outside `input_mode:=replay`; every other behaviour of this launch
is unchanged.

Because the detection path (self-crop -> ground-seg -> finite-filter ->
cluster) and the localization path (adapter -> `gnss_imu_localizer` ->
manager) are two independently-timed node chains advancing on the same
replay clock, the localization path's TF for a given recorded timestamp can
publish slightly after the detection path reaches that same timestamp in
real (wall-clock) time. AB3DMOT's TF lookup correctly refuses to extrapolate
into the future, so on a bag reconstructed this way, some non-empty
detection frames are rejected rather than tracked — see "Live validation
status" below for what was actually observed.

### Recording a bag that avoids this reconstruction

For a *future* capture, prefer recording localization directly (with the
localization stack running) so no replay-time reconstruction race is needed:

```
ros2 bag record --storage mcap -o <name> \
  /ad/sensors/lidar/points \
  /ad/sensors/camera/front/compressed \
  /ad/sensors/gps/fix \
  /ad/sensors/imu/data \
  /ad/vehicle/status \
  /ad/localization/odometry \
  /tf \
  /tf_static
```

Recording the raw ego sources (`gps/fix`, `imu/data`, `vehicle/status`) makes
the capture independently reproducible even if the localization output is
ever lost or needs re-deriving; recording the already-produced
`/ad/localization/odometry` and dynamic `/tf` directly avoids the
replay-time localization race described above for normal
visualization/replay use. For a bag that already contains a real, recorded
`/ad/localization/odometry` and dynamic `/tf`, leave `enable_localization`
at its default `false` — do not run a second, duplicate localization
producer against already-recorded localization output.

## RViz displays

`rviz/training_free_perception_camera.rviz` (Fixed Frame `odom`):

- **Grid**, **TF**
- **LiDAR (Cropped)** `/ad/perception/lidar/cropped` — enabled
- **LiDAR (Raw)** `/ad/sensors/lidar/points` — disabled (toggle on to compare)
- **Ground Points** `/ad/perception/lidar/ground` — disabled
- **Non-Ground Points** `/ad/perception/lidar/nonground` — disabled
- **Dynamic Occupancy** `/ad/perception/occupancy/dynamic` (Map) — enabled
- **Combined Occupancy** `/ad/perception/occupancy/combined` (Map) — disabled
- **Static Occupancy** `/ad/perception/occupancy/static` (Map) — disabled
- **Detected Objects** `/ad/visualization/detected_objects` (MarkerArray)
- **Tracked Objects** `/ad/visualization/tracked_objects` (MarkerArray) — training-free AB3DMOT
- **Predicted Objects** `/ad/visualization/predicted_objects` (MarkerArray)
- **Front Camera** `/ad/sensors/camera/front/compressed` (Image) — enabled
- **Camera Detections (2D)** `/ad/viz/perception/camera/dynamic_obstacle` (Image) — disabled

## Expected visible result

- A cropped LiDAR point cloud around the ego, updating at the sensor / replay rate.
- Wireframe **detection** boxes on raw clusters (no class, no orientation — the
  Euclidean detector marks orientation `UNAVAILABLE`).
- **Tracked** boxes with stable `A-<id>` labels and velocity arrows that persist
  across frames as objects move.
- **Predicted** trajectory poly-lines a short horizon ahead of each track.
- A translucent **dynamic occupancy** costmap under the scene.
- The **front camera** image in the docked panel (replay: only with
  `enable_camera:=true`).

## Readiness check

```
ros2 topic hz /ad/sensors/lidar/points
ros2 topic hz /ad/perception/objects/detected
ros2 topic hz /ad/perception/objects/tracked
ros2 topic hz /ad/perception/objects/predicted
ros2 topic hz /ad/perception/occupancy/dynamic
ros2 topic hz /ad/sensors/camera/front/compressed        # camera only
ros2 node list | grep -E 'euclidean|ab3dmot|prediction|occupancy|perception_visualizer'
```

There must be **no** `centerpoint` / `kalmannet` node in `ros2 node list`.

## Troubleshooting

| symptom | cause / fix |
| --- | --- |
| No points in RViz | LiDAR driver / bag not publishing `/ad/sensors/lidar/points`; check `ros2 topic hz`. |
| "Fixed Frame [odom] does not exist" | No `odom -> base_link`. Replay: the bag lacks `/tf` or `/ad/localization/odometry`. Live: no localization stack. |
| No detected boxes | Points are all cropped out, or ground segmentation removed everything — toggle **LiDAR (Cropped)** and **Non-Ground Points** on. |
| No tracked boxes | Detections exist but AB3DMOT has no `odom` transform for `lidar_link`, or fewer than `min_hits` consecutive detections. |
| Marker boxes stay frozen | `use_sim_time` mismatch — RViz on wall clock while data carries bag stamps. The demo passes `use_sim_time:=true` to RViz in replay mode; verify with `ros2 param get /heven_perception_rviz use_sim_time`. |
| Camera panel blank | Replay without `enable_camera:=true`; or no camera driver (live); or `compressed_image_transport` not installed. |
| Camera and LiDAR look out of sync | Different sensor rates + nearest-timestamp display; the demo does not time-align the two — expected, not a fusion. |
| CenterPoint / KalmanNet started | You passed `detector_backend:=centerpoint` or ran a different launch — this demo never sets them. |
| Occupancy grid missing | Two independent causes: (1) `/ad/localization/odometry` absent — the OGM nodes need ego pose; or (2) `road_gate.enabled: true` (the checked-in default for both `occupancy_grid/dynamic.yaml` and `.../static.yaml`) — the grids withhold publication until a timestamp-matched `/ad/planning/drivable_mask` arrives, and this demo does not include a planner/road-boundary node that produces one. Case (2) is expected, correct gating, not a bug — see "Live validation status". |

## Live validation status

The demo has since been brought up against a real bag,
`morai_cam4_20260813_163222` (360.170 s, 2,982 LiDAR + 7,038 front-camera
messages), across two sessions. **This is runtime/execution evidence, not a
detector or tracker accuracy evaluation, and not an OGM validation.**

**v1** (`fix/training-free-rviz-runtime-v1`): replayed the bag as-is. It has
one `/tf_static` message, zero `/tf`, and no `/ad/localization/odometry`, so
`odom -> base_link` was never populated; AB3DMOT correctly rejected every
detection for missing target-frame TF, and tracked/predicted/occupancy rates
were zero. Cropped LiDAR, raw/cropped/detected topics, and the front camera
were all confirmed live at their expected rates. One real launch bug was
found and fixed on that branch (`lidar_bag_replay.launch.py` was not passing
`start_visualization:=false`/`start_rviz:=false` into its nested perception
include, so the outer demo's `start_rviz:=true` leaked in and launched two
RViz processes).

**v2** (`fix/training-free-full-chain-runtime-v2`, this change): the same
bag has no recorded odometry/TF but does have raw
`/ad/sensors/gps/fix` + `/ad/sensors/imu/data` + `/ad/vehicle/status`.
`enable_localization:=true` (see above) reconstructed real, dynamic
`odom -> base_link` via the existing `ad_localization` `gnss_imu` backend —
**no fabricated ego motion**. Confirmed by direct `tf2_echo` at multiple
distinct real timestamps that the pose genuinely moves, and that the full
`odom -> base_link -> rear_axle_link -> {lidar_link, camera_front_link ->
camera_front_optical_frame}` chain resolves. With that TF present:

- cropped LiDAR flowed (~4.5 Hz at replay rate `0.5`)
- adaptive Euclidean detections flowed (~3.7-4.4 Hz)
- AB3DMOT produced **real, non-empty** `/ad/perception/objects/tracked`
  messages with finite odom-frame positions, KF-fused velocities, and
  covariances (one 90 s sample window: 35 messages, 12 non-empty, spanning
  **20 distinct track UUIDs**, 0 NaN/Inf) — tracking rate ~0.6-1.0 Hz
- `/ad/perception/objects/predicted` published concurrently, ~0.79-1.2 Hz
- the front camera image displayed in the same RViz session, ~9-10 Hz
- CenterPoint and KalmanNet were not running at any point
- freshly recomputed (not assumed from v1) nearest camera/LiDAR header-stamp
  offsets over all 2,982 LiDAR frames: median 19.995 ms, p95 39.139 ms, max
  49.693 ms — this characterizes *this recording's* timing only, it is not a
  calibrated synchronization claim

**Two limitations, both explained, neither fixed by weakening a safety
behaviour:**

1. **Dynamic/static/combined occupancy grids stayed empty.** Both
   `occupancy_grid/dynamic.yaml` and `.../static.yaml` ship
   `road_gate.enabled: true`, gating publication on a timestamp-matched
   `/ad/planning/drivable_mask`. No planner/road-boundary node is part of
   this opt-in demo graph, so the gate correctly withholds output by design
   — the same behaviour, and the same explanation, every other repo-local
   replay has already documented. **Not disabled in this change.**
2. **Tracking/prediction throughput was lower than detection throughput,
   not zero.** The detection path (4 processing hops) and the localization
   path (3 hops) are independently timed against the same replay clock;
   the localization path's TF for a given recorded timestamp can arrive
   slightly after the detection path reaches that timestamp in real
   wall-clock time, and AB3DMOT's TF lookup correctly refuses to
   extrapolate into the future rather than accept a possibly-wrong pose.
   Slowing the replay rate 10x (0.5 -> 0.05) did not change the accept
   ratio, ruling out simple throughput starvation — this is a fixed
   relative processing-latency skew between the two chains on this
   recording. **Not fixed** with an unsafe blocking TF timeout (the
   tracker's `rclpy.spin(node)` is single-threaded; blocking inside the
   detection callback while waiting for a `/tf` message would starve the
   same executor's own TF subscription) or with any relaxed/interpolated
   extrapolation.

Full detail, exact rates, and the launch-scoping bug fixed while wiring
`enable_localization` are in `docs/agent/STATUS.md`
("Training-Free Full-Chain RViz Validation v2").
