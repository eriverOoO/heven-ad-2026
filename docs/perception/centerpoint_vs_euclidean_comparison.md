# Euclidean vs CenterPoint: same-run qualitative comparison

## Scope and decision

This adds the one missing piece identified in
`docs/research/centerpoint_status.md`: a way to run Euclidean and
CenterPoint against the exact same MORAI frames and compare their output.
It does not change the Euclidean detector, the Autoware tracker, IMM, or
occupancy nodes, and it does not compute or imply any accuracy/mAP number.

## Why a rosbag isn't required

The only local rosbag for this scene
(`bags/static_20260805_003151/metadata.yaml`) has no `.mcap` payload file
(gitignored, and none is present on this host) — replaying it through
`lidar_bag_replay.launch.py` is not currently possible. Both detectors'
existing default input topics accept a plain FLOAT32 XYZI
`sensor_msgs/PointCloud2` (`/ad/perception/lidar/cropped` for CenterPoint;
`/ad/perception/lidar/nonground_finite` for Euclidean's cluster node), which
is exactly the layout the exported MORAI frames are already stored in. Two
new tools publish/record at that boundary instead of requiring a bag.

## New tools (`ad_lidar_perception` package)

- `ad_publish_morai_frames` — reads exported MORAI samples via the existing
  `MoraiHevenDatasetCore` (`tools/centerpoint_offline/morai_dataset.py`,
  unmodified) and publishes them as `PointCloud2` on a given topic, one
  sample at a time, `frame_id=lidar_link`.
- `ad_record_detected_objects` — subscribes `DetectedObjects`, tags each
  message with a `--backend` label, and writes JSONL records
  (`heven.ros_detection_comparison.v1`: object count, class/score/box per
  object, and publish→receipt latency).
- Pure encode/decode and summarization logic lives in
  `ad_lidar_perception/morai_replay.py` and
  `ad_lidar_perception/detection_recording.py`, unit-tested in
  `test/test_morai_replay.py` and `test/test_detection_recording.py`
  (round-trips against the existing `pointcloud2_to_xyzi` decoder).

## Known comparison caveat: ground segmentation is bypassed

Publishing directly onto `/ad/perception/lidar/cropped` /
`/ad/perception/lidar/nonground_finite` skips self-crop and ground
segmentation for both backends. CenterPoint's production ROS input is
already this same "cropped" stage (see
`docs/perception/centerpoint_ros_interface.md`), so this is a faithful
replay for CenterPoint. Euclidean's production input is normally
*post*-ground-segmentation; feeding it the un-ground-segmented cloud means
its clusters can include ground points. This is a deliberate, documented
simplification (ground segmentation's exact upstream `patchwork` message
contract was not reverse-engineered to avoid inventing an unverified
interface, per AGENTS.md) — not a claim that Euclidean's normal accuracy is
represented here.

## Exact commands

Build once (adds the two new executables and the `ad_viz` visualizer used
below):

```bash
source /opt/ros/humble/setup.bash
source "$HOME/ros-local-autoware-msgs/opt/ros/humble/share/autoware_perception_msgs/local_setup.bash"
source "$HOME/venvs/heven-centerpoint/bin/activate"
export PYTHONPATH="/usr/lib/python3/dist-packages:${PYTHONPATH:-}"
colcon build --symlink-install --packages-up-to ad_lidar_perception ad_viz \
  --base-paths "$HOME/projects/heven-ad-2026" \
  --build-base "$HOME/projects/heven_ros_ws/build" \
  --install-base "$HOME/projects/heven_ros_ws/install" \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_PREFIX_PATH="$HOME/ros-local-autoware-msgs/opt/ros/humble" \
  -DPython3_EXECUTABLE="$HOME/venvs/heven-centerpoint/bin/python"
```

Runtime overlay (one shell per process):

```bash
source /opt/ros/humble/setup.bash
source "$HOME/ros-local-autoware-msgs/opt/ros/humble/share/autoware_perception_msgs/local_setup.bash"
source "$HOME/projects/heven_ros_ws/install/setup.bash"
source "$HOME/venvs/heven-centerpoint/bin/activate"
```

**Euclidean run**, publishing straight onto its cluster input (bypasses
ground segmentation, see caveat above):

```bash
ros2 launch ad_lidar_perception euclidean_clustering.launch.py \
  finite_filter_enabled:=false \
  finite_input_topic:=/ad/perception/lidar/cropped
ros2 run ad_viz perception_visualizer_node   # reused RViz marker bridge, optional
ros2 run ad_lidar_perception ad_record_detected_objects \
  --backend euclidean --count 2 --output /tmp/euclidean_detections.jsonl
ros2 run ad_lidar_perception ad_publish_morai_frames \
  --dataset "$HOME/datasets/morai_heven" --split train --count 2 \
  --topic /ad/perception/lidar/cropped --interval-sec 1.5
```

**CenterPoint run**, same publisher/recorder pattern, its own detector
process on its default input topic:

```bash
ros2 launch ad_lidar_perception centerpoint_detector.launch.py \
  detector_backend:=centerpoint enabled:=true mock_mode:=false \
  checkpoint_path:=<checkpoint.pth> openpcdet_root:="$HOME/projects/OpenPCDet" \
  score_threshold:=0.1 input_topic:=/ad/perception/lidar/cropped
ros2 run ad_lidar_perception ad_record_detected_objects \
  --backend centerpoint --count 2 --output /tmp/centerpoint_detections.jsonl
ros2 run ad_lidar_perception ad_publish_morai_frames \
  --dataset "$HOME/datasets/morai_heven" --split train --count 2 \
  --topic /ad/perception/lidar/cropped --interval-sec 1.5
```

`ros2 run ad_viz perception_visualizer_node` (already-existing, unmodified
node) republishes either backend's `DetectedObjects` as `MarkerArray` on
`/ad/visualization/detected_objects`, which `ad_lidar_perception/rviz/
heven_perception.rviz` (unmodified, reused) already displays — running
`ros2 launch ad_lidar_perception perception_visualization.launch.py
start_rviz:=true` on top of either backend shows its boxes in RViz.

## Observed results (2026-08-18, RTX 4060, same 2 replayed frames)

| backend | checkpoint / config | objects/frame | latency (publish→receipt) |
|---|---|---|---|
| euclidean | `adaptive_euclidean_cluster.yaml` defaults | 3 | mean 16.7 ms |
| centerpoint | 2-iteration dry-run smoke checkpoint | up to 50 (capped by `max_detections`) | mean high, dominated by first-call CUDA warm-up |
| centerpoint | 5-iteration smoke checkpoint (this session) | **0** at `score_threshold=0.0` | mean 295 ms (warm-up dominated) |

The 5-iteration checkpoint's `CenterHead` produced zero candidate boxes
even with the score filter disabled — a real, observed effect of a few
extra untrained gradient steps, not a filtering bug (`model_forward`
latency in `ad_centerpoint_detector`'s own log confirms the model ran:
28–39 ms after warm-up). The 2-iteration checkpoint produced dense
diagnostic boxes instead. Both behaviors are internally consistent with
`docs/perception/centerpoint_offline_environment.md`'s existing statement
that "the dry-run checkpoint... cannot support a model-performance claim" —
box presence/absence at this training stage reflects near-random head
initialization, not detection quality, and is not being read as evidence of
anything beyond wiring.

Euclidean's 3 objects/frame include ground-plane clusters, consistent with
the documented ground-segmentation-bypass caveat above.

**This is wiring/diagnostic evidence only: no accuracy, precision, recall,
or mAP comparison is made or implied.**
