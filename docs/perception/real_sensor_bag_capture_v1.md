# Real Sensor Bag Capture Readiness v1

One explicit, reproducible, source-grounded recording contract for
**LiDAR + camera + localization + TF** so a real-vehicle (or any non-MORAI
sensor rig) recording can be captured, checked immediately, replayed through
the existing training-free perception/RViz stack, and retained as useful real
sensor-domain data — **without MORAI and without any learned LiDAR model.**

**This is not a GT dataset task.** A real sensor bag captured this way has no
moving-object ground truth. It is directly useful for runtime replay,
sensor-domain debugging, timing/calibration checks, and (later) unlabeled
CenterPoint domain-adaptation work. It is **not** directly supervised
KalmanNet training data — see section G.

## A. Before recording

Confirm the recording machine has, and check free disk:

```
df -h ~/datasets
```

Storage root (never inside the repository, already `.gitignore`d like every
other `datasets/…`/`~/datasets/…` convention in this project):

```
~/datasets/heven_real_bags/<YYYYMMDD_HHMMSS>_<location_or_test>/
```

e.g. `~/datasets/heven_real_bags/20260903_143000_kcity_loop1/`.

Record the software commit you are about to validate against, before you
start:

```
git -C ~/projects/heven-ad-2026 rev-parse HEAD
```

## B. Record command

Canonical topic set (source-grounded — see "Canonical recording contract"
below for exactly where each name/type comes from):

```
source /opt/ros/humble/setup.bash
source <your_ws>/install/setup.bash

ros2 bag record \
  --storage mcap \
  -o ~/datasets/heven_real_bags/$(date +%Y%m%d_%H%M%S)_<location_or_test> \
  /ad/sensors/lidar/points \
  /ad/sensors/camera/front/compressed \
  /ad/sensors/gps/fix \
  /ad/sensors/imu/data \
  /ad/vehicle/status \
  /ad/localization/odometry \
  /tf \
  /tf_static
```

Do **not** use `ros2 bag record -a` for this workflow. `-a` (all topics) is
fine for a quick ad-hoc capture (see `README.md`'s own quickstart), but it
records an unbounded, unaudited topic set with no readiness guarantee. This
canonical list is the exact set the training-free replay pipeline actually
consumes (`ad_lidar_perception/launch/lidar_bag_replay.launch.py`'s own
`SOURCE_TOPICS` + `LOCALIZATION_SOURCE_TOPICS` + front-camera topic) — nothing
more, nothing less.

**Record both raw ego sensors (A) and resolved localization (B) together
when possible** — this is deliberate, not redundant:

- **A. Raw localization inputs** (`/ad/sensors/gps/fix`, `/ad/sensors/imu/
  data`, `/ad/vehicle/status`) make the capture independently reproducible:
  the existing `ad_localization` stack can re-derive odometry/TF from them
  later even if the original localization run is lost, or if a future
  localization algorithm needs reprocessing from scratch.
- **B. Resolved localization outputs** (`/ad/localization/odometry`, dynamic
  `/tf`) let the bag replay directly, without re-running any localization
  node — this is what makes a bag `READY_FULL_REPLAY` (see below) instead of
  `READY_REPLAY_WITH_LOCALIZATION_RECOMPUTE`.

Session manifest (write this immediately after recording, see section D):

```
tools/runtime/check_sensor_bag_ready.py --bag <new_bag> --deep --json \
  > ~/datasets/heven_real_bags/<bag_id>/preflight_report.json
```

## C. Immediately validate the recording

```
ros2 bag info ~/datasets/heven_real_bags/<bag_id>

python3 tools/runtime/check_sensor_bag_ready.py --bag ~/datasets/heven_real_bags/<bag_id> --deep
```

The workflow must fail **visibly**, not silently, if:

- LiDAR message count is 0 → `MISSING_LIDAR`
- the bag directory has a `metadata.yaml` but no underlying storage file
  (a broken/incomplete transfer) → `BROKEN`
- localization is unexpectedly absent (neither resolved odometry+TF nor a
  complete raw GPS+IMU+status set) → `MISSING_LOCALIZATION` /
  `PARTIAL_SENSOR_ONLY`

See "Preflight status categories" below for the full decision tree.

## D. Full replay

Only when the bag is `READY_FULL_REPLAY` (resolved `/ad/localization/
odometry` **and** dynamic `/tf` were actually recorded):

```
ros2 launch ad_lidar_perception training_free_perception_rviz.launch.py \
  input_mode:=replay \
  bag_path:=<bag> \
  enable_camera:=true \
  enable_localization:=false \
  enable_drivable_mask:=true \
  data_dir:=<abs path to heven-ad-2026/ad_data> \
  start_rviz:=true
```

`enable_localization:=false` here is deliberate: the bag already has real
odometry/TF, so no second localization producer should run against it (see
Phase 20/"E" below for why running one anyway would be wrong).

## E. Replay with localization recompute

For a bag with real raw ego sensors (GPS + IMU + vehicle status) but no
recorded `/ad/localization/odometry` / dynamic `/tf` — this is the category
`morai_cam4_20260813_163222` is already validated in:

```
ros2 launch ad_lidar_perception training_free_perception_rviz.launch.py \
  input_mode:=replay \
  bag_path:=<bag> \
  rate:=0.15 \
  enable_camera:=true \
  enable_localization:=true \
  enable_drivable_mask:=true \
  data_dir:=<abs path to heven-ad-2026/ad_data> \
  start_rviz:=true
```

`enable_localization:=true` starts the existing, unmodified
`ad_localization` `gnss_imu` backend against the bag's own replayed
`/ad/sensors/gps/fix` + `/ad/sensors/imu/data` + `/ad/vehicle/status` — real
GNSS/IMU fusion, no fabricated pose. **Never run this against a bag that
already has recorded odometry/TF** — that starts a second, duplicate
localization producer racing the recorded one.

## F. Expected RViz layers

Same layers `training_free_perception_camera.rviz` already defines
(`docs/perception/training_free_rviz_demo_v1.md`): cropped LiDAR, TF, front
camera, detected/tracked/predicted `MarkerArray`s, dynamic/combined/static
occupancy `Map` displays, and the disabled-by-default "Drivable Mask (debug)"
`Map` display (`enable_drivable_mask:=true`). Nothing about RViz changes for
a real bag versus a MORAI replay bag — the whole point of this contract is
that the training-free stack does not care which one it is.

## G. What this data can and cannot be used for

**Can:**

- runtime replay through the full training-free perception/RViz stack
- sensor-domain debugging (does the real LiDAR/camera/localization graph
  behave the same way as in simulation?)
- **unlabeled** CenterPoint domain-adaptation preparation: raw real VLP-16 (or
  whichever real LiDAR is used) point clouds, real timing, real ego pose, real
  camera images, for a future beam/range/intensity domain audit and
  pseudo-label preparation — **not** supervised training data by itself
- qualitative detector comparison, timing analysis, calibration-provenance
  verification, full-stack regression testing

**Cannot, without separately captured object ground truth:**

- **Supervised CenterPoint training or evaluation** — no 3D box labels exist
  in this data; only a future pseudo-labeling or manual-annotation step could
  change that, and neither is in scope here.
- **Supervised KalmanNet training** — KalmanNet needs a true object *state
  trajectory* (position/velocity over time) plus a detector measurement per
  timestep. A real bag alone provides neither; it has no moving-object
  identity or state at all. It is useful for **runtime validation** of the
  existing tracker (does AB3DMOT run, does it produce plausible-looking
  tracks) but that is a qualitative/execution check, not evidence toward any
  KalmanNet-vs-KF accuracy claim, and not a substitute for real object GT.

---

## Canonical recording contract

| topic | type | source in this repo |
| --- | --- | --- |
| `/ad/sensors/lidar/points` | `sensor_msgs/msg/PointCloud2` | `lidar_bag_replay.launch.py` `SOURCE_TOPICS` |
| `/ad/sensors/camera/front/compressed` | `sensor_msgs/msg/CompressedImage` | `lidar_bag_replay.launch.py` `FRONT_CAMERA_TOPIC` |
| `/ad/localization/odometry` | `nav_msgs/msg/Odometry` | `lidar_bag_replay.launch.py` `SOURCE_TOPICS`; published by `ad_localization`'s `localization_manager_node` |
| `/tf` (dynamic) | `tf2_msgs/msg/TFMessage` | `lidar_bag_replay.launch.py` `SOURCE_TOPICS`; `odom -> base_link` from `ad_localization` |
| `/tf_static` | `tf2_msgs/msg/TFMessage` | `lidar_bag_replay.launch.py` `SOURCE_TOPICS`; static vehicle/sensor edges also always supplied at replay time by `ad_description`'s `robot_state_publisher`, independent of what the bag itself has |
| `/ad/sensors/gps/fix` | `sensor_msgs/msg/NavSatFix` | `lidar_bag_replay.launch.py` `LOCALIZATION_SOURCE_TOPICS`; consumed by `ad_localization`'s adapter (`localization_node.cpp`) |
| `/ad/sensors/imu/data` | `sensor_msgs/msg/Imu` | `lidar_bag_replay.launch.py` `SOURCE_TOPICS` + `LOCALIZATION_SOURCE_TOPICS`; consumed by the adapter |
| `/ad/vehicle/status` | `ad_morai_interfaces/msg/EgoVehicleStatus` | `lidar_bag_replay.launch.py` `LOCALIZATION_SOURCE_TOPICS`; consumed unconditionally by the adapter (`localization_node.cpp`, regardless of backend) |

**Real-vehicle caveat on `/ad/vehicle/status`.** Its message type,
`ad_morai_interfaces/msg/EgoVehicleStatus`, is a MORAI-defined type. The
`ad_localization` adapter subscribes to it unconditionally at `on_activate`.
For a non-MORAI vehicle to use `enable_localization:=true` exactly as
validated here, a real-vehicle CAN/status bridge would need to publish this
same message type on this topic (repurposing it as a generic ego-status
schema) — there is no other adapter for a differently-typed vehicle-status
message in this repository today. This is a genuine, source-confirmed
limitation for a first real-vehicle capture, not a guess; it is why Phase 1's
"determine whether it is valuable to record raw localization inputs" gets a
qualified answer: GPS + IMU alone (both plain `sensor_msgs` types, no
MORAI-specific dependency) are enough to drive `gnss_imu`'s actual position/
orientation fusion — the adapter's `/ad/vehicle/status` subscription exists
but the `gnss_imu` backend's own fusion does not appear to depend on its
content (source-read, not independently re-validated end-to-end without it
in this task).

### Expected rates (from the one real bag currently available,
`morai_cam4_20260813_163222`, not a hard requirement — a real vehicle rig
will have its own real sensor rates)

| topic | measured native rate |
| --- | --- |
| LiDAR | ~8.3 Hz (2,982 msgs / 360.17 s) |
| camera (front) | ~19.5 Hz (7,038 msgs / 360.17 s) |
| GPS | ~18.8 Hz (6,770 msgs / 360.17 s) |
| IMU | ~39.7 Hz (14,280 msgs / 360.17 s) |
| vehicle status | ~39.3 Hz (14,141 msgs / 360.17 s) |
| dynamic `/tf` (`odom->base_link`) | not directly measured (0 recorded in this bag); inferred to track the localization backend's own output rate — the `gnss_imu` backend's fused-odometry rate has previously been measured at ~9.4-9.6 Hz replaying this same GPS stream at 0.5x, i.e. ~18-19 Hz native, matching the GPS rate it is bottlenecked on |

Use these as a **sanity range** to catch an obviously broken recording (e.g.
LiDAR at 0.1 Hz, or camera not incrementing at all) — not as a pass/fail
threshold; no part of this pipeline hard-codes a required rate.

### Approximate storage cost (measured directly, not estimated, from
`morai_cam4_20260813_163222`'s real per-topic byte totals)

| contract | approx. GB/min |
| --- | --- |
| LiDAR only | ~0.16 GB/min (970.7 MB / 6.0 min) |
| LiDAR + front camera | ~0.34 GB/min (2,023.9 MB / 6.0 min) |
| full canonical contract (LiDAR + front camera + GPS + IMU + vehicle status + `/tf`/`/tf_static`) | ~0.34 GB/min (small topics add a few MB total, negligible next to LiDAR/camera) |

This is capacity planning only; a real sensor rig's exact point density /
image resolution / codec will change these numbers.

## Preflight status categories

`tools/runtime/check_sensor_bag_ready.py` classifies a bag (or a live ROS
graph) into exactly one of:

| status | meaning |
| --- | --- |
| `BROKEN` | `metadata.yaml` missing/invalid, or it declares a storage file that does not exist on disk (a metadata-only bag) |
| `MISSING_LIDAR` | `/ad/sensors/lidar/points` absent or has 0 recorded messages |
| `MISSING_TF` | LiDAR present, `/ad/localization/odometry` recorded, but dynamic `/tf` is absent/empty — cannot do exact-stamp TF lookups from this bag alone |
| `READY_FULL_REPLAY` | LiDAR present, and both `/ad/localization/odometry` and dynamic `/tf` were recorded with real messages — replays directly, `enable_localization:=false` |
| `READY_REPLAY_WITH_LOCALIZATION_RECOMPUTE` | LiDAR present, no usable recorded odometry/TF, but GPS + IMU + vehicle status are all recorded — `enable_localization:=true` lets `ad_localization` regenerate odometry/TF |
| `PARTIAL_SENSOR_ONLY` | LiDAR present, some but not all of GPS/IMU/vehicle-status recorded — not enough for either replay path |
| `MISSING_LOCALIZATION` | LiDAR (and maybe camera) present, but zero localization-related topics of any kind — useful for raw LiDAR/camera domain work only, tracking replay is not possible |

Camera is never part of the readiness decision (it only affects whether the
RViz camera panel is populated) — a `READY_FULL_REPLAY` or
`READY_REPLAY_WITH_LOCALIZATION_RECOMPUTE` bag without a recorded camera
topic is still exactly that status, with a note in `reasons`.

### Deep diagnostics (`--deep`, needs a sourced ROS environment)

- Header-stamp audit per topic: nonzero, monotonic, duplicate count,
  backward-jump count (reads real message content, no bag rewriting).
- Nearest-stamp `|dt|` between LiDAR and camera, and between LiDAR and
  odometry (median/p95/max) — timing characterization only, **not** a
  hardware-synchronization claim unless separately proven for the specific
  sensors involved.

### Live mode (`--live`, needs a sourced ROS environment against a running
graph)

Checks the same 8 canonical topics for presence/type on the current ROS
graph, plus whether `odom -> base_link -> rear_axle_link -> lidar_link` and
`rear_axle_link -> camera_front_optical_frame` all resolve via `tf2_ros` -
read-only, subscribes only, never publishes.

## Existing-bag classification (regression, this task)

| bag | classification | why |
| --- | --- | --- |
| `morai_cam4_20260813_163222` | `READY_REPLAY_WITH_LOCALIZATION_RECOMPUTE` | LiDAR (2,982) + camera (7,038) + GPS (6,770) + IMU (14,280) + vehicle status (14,141) all recorded; `/ad/localization/odometry` absent; `/tf` present but 0 messages |
| `bags/static_20260805_003151` | `BROKEN` | `metadata.yaml` declares `static_20260805_003151_0.mcap`, which does not exist on disk in this checkout - a metadata-only remnant, not a usable bag |

Live-mode regression (same bag, `enable_localization:=true
enable_drivable_mask:=true`): all 8 canonical topics present with the exact
expected types, and all 4 TF chains (`odom->base_link`,
`base_link->rear_axle_link`, `rear_axle_link->lidar_link`,
`rear_axle_link->camera_front_optical_frame`) resolved `OK`. Detection and
drivable-mask topics continued publishing at the same rates already
documented in `training_free_rviz_demo_v1.md` / `docs/agent/STATUS.md` - no
new performance analysis was re-run here.

`--deep` diagnostics on this bag reproduce the previously-independently-
measured camera/LiDAR timing exactly: median 19.995 ms, p95 39.139 ms, max
49.693 ms nearest-stamp offset over all 2,982 LiDAR frames — cross-validating
both this new tool and the prior finding.

## Session manifest

Write a small sidecar JSON next to each bag
(`~/datasets/heven_real_bags/<bag_id>/session_manifest.json`) — never inside
the rosbag itself:

```json
{
  "bag_id": "20260903_143000_kcity_loop1",
  "date_time_local": "2026-09-03T14:30:00+09:00",
  "vehicle_or_platform": "2023 Hyundai Ioniq5 (competition vehicle) | test rig name",
  "sensor_setup": "VLP-16 + front camera + GNSS + IMU, as configured in ad_description/config/sensor_mounts.yaml",
  "lidar_topic": "/ad/sensors/lidar/points",
  "camera_topic": "/ad/sensors/camera/front/compressed",
  "localization_mode": "raw_recompute | recorded_full",
  "software_commit": "<git rev-parse HEAD, recorded at capture time>",
  "record_command": "<the exact ros2 bag record invocation used>",
  "sensor_mounts_config": {
    "path": "ad_description/config/sensor_mounts.yaml",
    "sha256": "<sha256sum of that file at capture time>",
    "active_profile": "current_front_sensor_mounts"
  },
  "preflight_status": "READY_FULL_REPLAY | READY_REPLAY_WITH_LOCALIZATION_RECOMPUTE | ...",
  "notes": "free-text: route, weather, anything unusual"
}
```

`software_commit` is critical for future replay reproducibility (Phase 25):
the exact detector/tracker/prediction/OGM/localization code that produced a
given replay result must be traceable back to the commit that was checked
out when the bag was captured and when it was later replayed. `sensor_mounts_
config.sha256` is calibration **provenance**, not a calibration **accuracy**
claim — no `CameraInfo` publisher exists anywhere in this repository, so no
authoritative camera intrinsics claim is made here either (same limitation
`training_free_rviz_demo_v1.md` already documents for Level A camera
display).

No personally identifying information is required in the manifest.
