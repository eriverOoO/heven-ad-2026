# CenterPoint ROS detector interface preparation

## Scope

This optional adapter prepares a ROS boundary for the OpenPCDet checkpoint
format validated offline. It does not select CenterPoint by default, modify the
Euclidean clusterer, tracker, IMM, occupancy nodes, or Autoware, and the current
dry-run checkpoint is not a production model.

## Existing production contract

- Raw input is `sensor_msgs/msg/PointCloud2` on
  `/ad/sensors/lidar/points`.
- With the current MORAI classical composition, self-cropped points feed ground
  segmentation and the finite nonground cloud reaches the Euclidean detector
  on `/ad/perception/lidar/nonground_finite`.
- `ad_adaptive_euclidean_cluster_node` subscribes with
  `rclcpp::SensorDataQoS()` and publishes
  `autoware_perception_msgs/msg/DetectedObjects` with depth 10 on
  `/ad/perception/objects/detected`.
- `autoware_multi_object_tracker` consumes exactly that topic as
  `input/detection01/objects`, channel `lidar_clustering`, and publishes
  `/ad/perception/objects/tracked`.
- The detector copies the complete input header, so output stamp and frame are
  the source cloud stamp and `lidar_link`. No wall-clock replacement is made.

## Optional CenterPoint graph and message mapping

Selecting `detector_backend:=centerpoint` replaces only the detector leaf:

```text
/ad/sensors/lidar/points -> existing preprocessing -> cropped PointCloud2
  -> ad_centerpoint_detector -> /ad/perception/objects/detected
  -> autoware_multi_object_tracker -> existing prediction/IMM/occupancy
```

The adapter reads scalar FLOAT32 `x`, `y`, `z`, and `intensity`, removes only
non-finite points, and passes `[x,y,z,intensity]` without axis conversion. It
requires `lidar_link`: +x forward, +y left, +z up. Boxes remain
`[x,y,z,length,width,height,yaw]`, with CCW yaw about +z.

| MORAI class | Autoware classification |
|---|---|
| vehicle | `CAR` |
| pedestrian | `PEDESTRIAN` |
| obstacle | `UNKNOWN` |

`obstacle` deliberately follows the existing Euclidean detector's `UNKNOWN`
convention. Detection score populates both existence probability and the sole
classification probability. Position is the geometric box center, dimensions
map length/width/height to shape x/y/z, and yaw becomes a z-axis quaternion.
Orientation is `AVAILABLE`. The detector invents no velocity:
`has_twist=false`, `has_twist_covariance=false`, with default-zero twist data.

## Selection and parameters

The top-level default remains:

```bash
ros2 launch ad_lidar_perception lidar_perception.launch.py \
  detector_backend:=euclidean
```

Interface-only mock mode requires no checkpoint and publishes an empty,
stamp-preserving `DetectedObjects` message for each valid input cloud:

```bash
ros2 launch ad_lidar_perception lidar_perception.launch.py \
  detector_backend:=centerpoint \
  centerpoint_mock_mode:=true
```

Real activation additionally requires `checkpoint_path`, `openpcdet_root`,
`device`, `score_threshold`, `max_detections`, and the fixed
`point_cloud_range`. `centerpoint_enabled` is the launch-facing form of the
node's `enabled` parameter. The dry-run checkpoint must not be deployed as a
production model.

## Latency instrumentation

Every successful callback logs four independent durations in milliseconds:

- `preprocessing`: PointCloud2 decode and finite filtering;
- `model_forward`: only model forward, bracketed by CUDA synchronization;
- `postprocessing`: filtering and DetectedObjects construction;
- `total`: complete detector callback through publish.

These values are detector-local instrumentation. They are not written into or
combined with existing bag/ROS pipeline latency metrics.

## Runtime validation on WSL2 (2026-08-17)

The resolved build root is `/home/didgang1203/projects/heven_ros_ws`; the
repository remains a source base at
`/home/didgang1203/projects/heven-ad-2026`. ROS Humble supplies `rclpy`,
`sensor_msgs`, `geometry_msgs`, and `std_msgs`. The exact locked
`autoware_perception_msgs` 1.13.0 apt artifact was extracted, without a full
Autoware installation, under
`/home/didgang1203/ros-local-autoware-msgs/opt/ros/humble` and its package
`local_setup.bash` is sourced before build and runtime.

The isolated CenterPoint venv must use two different path policies:

- CMake build only: prepend `/usr/lib/python3/dist-packages` so ROS build tools
  can import Ubuntu `catkin_pkg`;
- runtime: do **not** prepend that directory, because it would shadow pinned
  NumPy 1.26.4 with Ubuntu NumPy 1.21.5. ROS setup files already expose rclpy.

The build command was:

```bash
source /opt/ros/humble/setup.bash
source "$HOME/ros-local-autoware-msgs/opt/ros/humble/share/autoware_perception_msgs/local_setup.bash"
source "$HOME/venvs/heven-centerpoint/bin/activate"
export PYTHONPATH="/usr/lib/python3/dist-packages:${PYTHONPATH:-}"
colcon --log-base "$HOME/projects/heven_ros_ws/log" build \
  --base-paths "$HOME/projects/heven-ad-2026" \
  --build-base "$HOME/projects/heven_ros_ws/build" \
  --install-base "$HOME/projects/heven_ros_ws/install" \
  --symlink-install --packages-up-to ad_lidar_perception \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_PREFIX_PATH="$HOME/ros-local-autoware-msgs/opt/ros/humble" \
  -DPython3_EXECUTABLE="$HOME/venvs/heven-centerpoint/bin/python"
```

The build completed `ad_description`, `ad_interfaces`, and
`ad_lidar_perception`. A top-level mock launch with a minimal composition
preserved the exact input stamp and `lidar_link`, published zero detections,
and exposed exactly one `/ad/perception/objects/detected` publisher; no
Euclidean node was present.

The dry-run checkpoint was then loaded strictly for interface validation on an
RTX 4060. Six checked output frames each contained 49 untrained diagnostic
predictions. Every output preserved stamp/frame, had finite box/quaternion
values, positive dimensions, an allowed class mapping, normalized orientation,
and no twist flags. The first CUDA model-forward callback was 1136.965 ms.
After warm-up, a 15-callback sample averaged 27.872 ms preprocessing, 19.763 ms
model forward, 1.795 ms postprocessing, and 53.080 ms total. Model-forward
samples ranged from 16.358 to 28.584 ms. PyTorch reported 146.4 MiB allocated
and 271.8 MiB peak allocated during callbacks. These are runtime-interface
observations on repeated static data, not detector performance results.

## Minimal Autoware tracker overlay

The tracker runtime is built separately at
`$HOME/projects/autoware_tracker_ws`; no Autoware source is vendored into this
repository. HEVEN's `dependencies.repos` pins `autoware_universe` commit
`d4d260983d357e1b2b34291d91933f9f4b53bf94`, whose
`autoware_multi_object_tracker/package.xml` is version 0.51.0. The matching
Autoware 1.8.0 manifest pins `autoware_cmake` 1.2.0, `autoware_utils` 1.7.2,
`autoware_core` 1.8.0, `agnocast` 2.3.3, `autoware_internal_msgs` 1.12.1, and
muSSP commit `c79e98fd5e658f4f90c06d93472faa977bc873b9`. HEVEN's existing lock
continues to pin the installed `autoware_perception_msgs` interface to 1.13.0.
Exact resolved commits are recorded in
`$HOME/projects/autoware_tracker_ws/tracker-dependencies.lock.yaml`.

The computed and actually built package closure is:

```text
autoware_lint_common (build metadata only)
mussp
autoware_cmake
autoware_common_msgs
autoware_internal_debug_msgs
autoware_internal_metric_msgs
autoware_planning_msgs
autoware_internal_perception_msgs
autoware_internal_planning_msgs
autoware_internal_msgs
autoware_utils_math
autoware_utils_rclcpp
autoware_utils_system
autoware_agnocast_wrapper
autoware_utils_debug
autoware_utils_geometry
autoware_multi_object_tracker
```

Only `diagnostic_updater` 4.0.7 and Boost 1.74 development headers were added
as locally extracted Ubuntu packages under
`$HOME/ros-local-tracker-deps`; the system ROS/Python/CUDA installations were
not changed. Although the official `agnocast` source is pinned in the external
manifest, the computed closure builds only `autoware_agnocast_wrapper`, not the
agnocast package itself.

Source overlays in this order for runtime:

```bash
source /opt/ros/humble/setup.bash
source "$HOME/ros-local-autoware-msgs/opt/ros/humble/share/autoware_perception_msgs/local_setup.bash"
source "$HOME/ros-local-tracker-deps/opt/ros/humble/share/diagnostic_updater/local_setup.bash"
source "$HOME/projects/autoware_tracker_ws/install/setup.bash"
source "$HOME/projects/heven_ros_ws/install/setup.bash"
source "$HOME/venvs/heven-centerpoint/bin/activate"
export CPLUS_INCLUDE_PATH="$HOME/ros-local-tracker-deps/usr/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
```

Packages were built topologically into the one isolated install prefix with
`colcon build --symlink-install`, `BUILD_TESTING=OFF`, and
`CMAKE_BUILD_TYPE=RelWithDebInfo`. Building one exact package base at a time
avoids selecting unrelated Autoware packages, for example:

```bash
colcon --log-base "$HOME/projects/autoware_tracker_ws/log-one" build \
  --symlink-install \
  --base-paths "$HOME/projects/autoware_tracker_ws/src/autoware_universe/perception/autoware_multi_object_tracker" \
  --build-base "$HOME/projects/autoware_tracker_ws/build-one" \
  --install-base "$HOME/projects/autoware_tracker_ws/install" \
  --cmake-args -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_PREFIX_PATH="$HOME/projects/autoware_tracker_ws/install;$HOME/ros-local-autoware-msgs/opt/ros/humble;$HOME/ros-local-tracker-deps/opt/ros/humble"
```

The installed tracker takes reliable, volatile
`autoware_perception_msgs/msg/DetectedObjects` on
`/ad/perception/objects/detected`, transforms `lidar_link` measurements through
`base_link` into configured world frame `odom`, and publishes reliable,
volatile `autoware_perception_msgs/msg/TrackedObjects` on
`/ad/perception/objects/tracked`. HEVEN retains channel `lidar_clustering` and
the checked-in tracker parameters; no tracker semantics were changed.

Standalone validation accepted both empty and non-empty synthetic detection
arrays without a type, frame, or QoS error. Ten inputs produced ten tracked
arrays; after tracker initialization, outputs contained one tracked object.
Output stamps matched their source inputs and the output frame was `odom`.

The integrated smoke launched exactly one CenterPoint detected-object
publisher, the tracker, and HEVEN's existing IMM/prediction adapter. A
controlled one-publish-per-stamp run exercised eight unique MORAI PointCloud2
stamps: the detector delivered 8/8 messages with 49 dry-run-checkpoint
diagnostics per frame, and the tracker delivered 8/8 `odom` messages, seven of
which were non-empty after initialization. A separate repeated-input run also
confirmed that the prediction adapter published non-empty `odom` output.
Topic introspection showed one publisher and one matching subscriber at both
detected and tracked boundaries, all reliable/volatile. A WSL wall-clock jump
of about ten seconds occurred during the run, causing one TF extrapolation
warning and repeated-stamp rejection warnings in the prediction adapter; all
three processes remained alive and shut down cleanly. This is a host clock
stability issue to eliminate before deterministic bag replay, not detector or
tracker performance evidence.

In the controlled run, prediction processing was delayed until its 0.5-second
freshness bound had elapsed and it correctly rejected the stale tracker array;
therefore downstream compatibility is verified, but deterministic downstream
delivery timing remains blocked by host clock/executor timing stability.
