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
| `/ad/localization/odometry` | `nav_msgs/Odometry` | localization / bag (**required**: `odom` frame + OGM/risk) |
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
`/tf_static` and `/ad/localization/odometry`.

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

### Optional flags

| flag | default | effect |
| --- | --- | --- |
| `enable_dynamic_object_risk` | `true` | start the observational Dynamic Object Risk node |
| `enable_camera_perception` | `false` | also start the YOLO 2D overlay (needs torch) |
| `start_rviz` | `true` | set `false` for a headless graph |
| `loop` | `true` | replay the bag repeatedly |
| `start_paused` | `false` | start the bag paused |
| `rviz_config` | demo config | absolute path to an alternate RViz config |

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
| Occupancy grid missing | `/ad/localization/odometry` absent; the OGM nodes need ego pose. |

## Not live-tested here

This session's host lacks `rosbag2_storage_mcap` and
`compressed_image_transport`, and cannot run RViz or capture screenshots.
Validation was limited to launch-structure, RViz-config and unit tests
(`test_training_free_perception_rviz_launch.py` plus the affected
`test_perception_visualization_launch.py` / `test_lidar_bag_replay_launch.py` /
`test_lidar_perception_launch.py` / `test_selection_config.py`) and an
installed-launch import check. The graph was **not** brought up against a live
bag or live sensors in this session.
