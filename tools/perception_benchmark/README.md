# Perception detection benchmark

This offline harness evaluates any detector that publishes the STEP 01
`DetectedObjects` contract. It reads recorded output; it does not launch or
modify the production detector or tracker.

## Run

From the workspace root:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 src/heven_ad_2026/tools/perception_benchmark/evaluate_detection.py \
  --bag bags/static_20260805_003151 \
  --config src/heven_ad_2026/tools/perception_benchmark/configs/adaptive_euclidean_aabb.yaml \
  --output-dir src/heven_ad_2026/results/perception
```

The command creates `<experiment>.csv` and `<experiment>.md`. The CSV is a
machine-readable one-row summary. The Markdown report includes alignment,
interpretation, and git/bag/config/environment metadata.

`results/` is intentionally git-ignored by repository policy. Reports remain
local runtime artifacts unless a maintainer deliberately changes that policy.

## Contract and alignment

- Source: `sensor_msgs/msg/PointCloud2` on `/ad/sensors/lidar/points`.
- Detection: `autoware_perception_msgs/msg/DetectedObjects` on
  `/ad/perception/objects/detected`.
- Source and detections are paired only by exact `header.stamp`; no
  interpolation is performed. `frames_compared` includes only detection frames
  with that exact recorded source plus valid actor and TF alignment.
- Actor GT and dynamic TF both carry source stamps. They use nearest source
  stamp with explicit maximum deltas from the config. Static TF is timeless.
- The supplied MORAI bag config declares the observed `map == odom` identity
  assumption. It is not a global evaluator assumption.

Distance bins are half-open (`0 <= r < 20`, etc.), except the final `60m+`
bin. Matching is greedy shortest-center-distance, one-to-one, and is evaluated
independently at 1.5, 2.5, and 4.0 m. The main recall proxy uses 2.5 m.

Latency uses two different timestamps deliberately:

- Pairing key: source `PointCloud2.header.stamp` equals detection
  `DetectedObjects.header.stamp` exactly.
- Start: MCAP recorder receive timestamp of the source PointCloud2 record.
- End: MCAP recorder receive timestamp of the DetectedObjects record.
- Value: `(detection_record_timestamp - source_record_timestamp)` in
  milliseconds.

It includes pipeline scheduling, DDS delivery, and recording overhead; it is
not isolated detector CPU time. Negative recorder-order samples, if any, are
reported and excluded because cross-topic recorder callback order cannot
represent a physical negative pipeline latency.

## Audited bag count difference

For `static_20260805_003151`, header-stamp set comparison found:

- recorded source LiDAR: 1,778 messages / 1,778 unique stamps;
- recorded detections: 1,796 messages / 1,796 unique stamps;
- exact source/detection intersection: 1,761 stamps;
- detection stamps absent from the recorded source topic: 35;
- source stamps without a recorded detection: 17.

Thus the net topic-count difference is `35 - 17 = 18`; the former
`frames_compared=1,788` was not a raw/detection comparison count. It counted
detection frames having actor and TF alignment even when the bag lacked the
source record. The evaluator now excludes those frames.

The metadata records `/ad/sensors/lidar/points` as best-effort and
`/ad/perception/objects/detected` as reliable. Each rosbag topic is received
independently, so the bag can contain a downstream output even when its source
sample was not captured by the recorder. Conversely, a source-only stamp can
mean either no downstream output or an output not captured. The bag alone
cannot distinguish those causes; no interpolation or synthetic pairing is
used.

## Dependencies

Python 3, PyYAML, ROS 2 `rosbag2_py`, `rclpy`, and the message packages recorded
in the bag must be available. Source both ROS 2 and this workspace before use.
