# Ground segmentation benchmark: Patchwork, Patchwork++, and RANSAC

Date: 2026-08-12 (Asia/Seoul)

## Executive conclusion

On the available flat-road, mostly static MORAI recording, the checked-in
Patchwork++ configuration with an 80 m segmentation range did not improve the
simulated-actor recall proxy over classic Patchwork. It preserved every finite
input point and produced more nonground points, but those additional points
increased detections without increasing matched actors. It also added about
1.6 ms to median ground-segmentation latency.

An experimental Patchwork++ configuration with a 105 m range recovered three
additional actor-frame instances at the 2.5 m matching threshold, but also
produced more detections and remained slower than classic Patchwork. This is a
candidate for evaluation on moving, sloped, and tunnel-transition recordings;
one static recording is insufficient evidence for adopting it as the default.

Autoware RANSAC recovered the most actor-frame instances in this recording,
but was about 2.7 times slower at the segmentation stage and its debug ground
output assigned stamp zero to every frame. Its nonground output can feed the
current clustering path, but its ground output does not satisfy the existing
timestamp contract.

The conservative choice for the present flat-road stack is therefore classic
Patchwork. Keep Patchwork++ as an experimental backend until a scenario-diverse
dataset demonstrates a repeatable gain. RANSAC is useful as an ablation and
possible recall-oriented candidate after fixing the ground-message contract.

## Scope and questions

The experiment was designed to answer five separate questions:

1. Does each algorithm keep up with the recorded LiDAR stream without dropping
   frames?
2. How much of the finite, self-cropped cloud is classified as ground and
   nonground?
3. What latency does ground segmentation add?
4. How does the choice propagate into Euclidean clustering and detected-object
   counts?
5. Does an increased detection count correspond to more MORAI actor matches,
   or merely more unmatched clusters?

This is not a full safety validation. It covers one recording and one vehicle,
sensor, clustering, and parameter profile.

## Software and hardware

- Host: Ubuntu 22.04, Linux 6.8.0-136-generic.
- CPU: Intel Core i5-1240P, 12 physical cores / 16 logical CPUs.
- ROS: ROS 2 Humble. The interactive shell also contained ROS 1 Noetic paths;
  runtime trials used the built Humble workspace but the mixed shell warning
  should be removed in future controlled benchmarks.
- HEVEN repository commit before local edits:
  `45cb3610f3c0e09c85296f406e940e15462e766c`.
- Patchwork++ repository commit:
  `3e6903a1d5537a4cc2ace897b0bbb98a92d6014c` (upstream v1.4.1 release commit).
- Patchwork ROS package manifest version: 0.1.0, GPL-3.0.

## Dataset

Source bag:
`bags/static_20260805_003151`

- Storage: MCAP.
- Size: 2.6 GiB.
- Duration: 242.322 s.
- LiDAR topic: `/ad/sensors/lidar/points`.
- LiDAR messages: 1,778, averaging approximately 7.34 Hz over the complete bag.
- Sensor profile: MORAI VLP-16 at `lidar_link`.
- Configured sensor height supplied by launch: 1.7685 m.
- A raw cloud contains 28,800 records. In the measured window the median raw
  finite-point count was about 14,844; invalid/no-return records account for
  the difference.
- The final comparison uses approximately the first 41 seconds. Depending on
  process startup, 261 to 264 input frames were captured. Metrics that compare
  algorithms directly use common header timestamps where applicable.

The bag is named `static` and vehicle speed is approximately zero in the
inspected interval. It is useful for ground/clutter consistency, but cannot
establish performance during strong ego motion, scan distortion, steep grades,
banked roads, potholes, or tunnel transitions.

## Pipeline under test

The classic Patchwork and Patchwork++ production-path trials used:

```text
recorded raw LiDAR
  -> self-crop
  -> ground segmentation
  -> finite XYZ filter
  -> adaptive Euclidean clustering
  -> DetectedObjects
```

MORAI deskew remained disabled, matching the existing instantaneous-scan
profile. Gravity leveling remained disabled, matching the current default.

RANSAC was fed the recorded `/ad/perception/lidar/cropped` topic and then the
same finite filter and clustering stages. This gives RANSAC the same pointcloud
content at its segmentation input, but its detection latency is measured from
`cropped` rather than raw LiDAR and therefore must not be interpreted as a
strict end-to-end comparison.

Every trial ran alone in a distinct ROS domain. Playback rate was 1.0. Outputs
were recorded to separate MCAP files. Running algorithms sequentially avoided
CPU contention between candidates.

## Configurations

### Classic Patchwork

- `algorithm: patchwork`
- `sensor_height: 1.7685` supplied from the mount profile
- `num_iter: 3`
- `num_lpr: 20`
- `num_min_pts: 5`
- `th_seeds: 0.35`
- `th_dist: 0.18`
- `min_range: 3.0 m`
- `max_range: 80.0 m`
- `uprightness_thr: 0.707`

Classic Patchwork places only points inside its configured radial range into
the concentric-zone model. Points outside that range are not included in either
output by the local classic implementation. This explains its partition
completeness below 100%.

### Patchwork++ 80 m

The current checked-in candidate uses the shared parameters above plus:

- `algorithm: patchworkpp`
- `th_seeds_v: 0.25`
- `th_dist_v: 0.1`
- `max_range: 80.0 m`

Unlike the local classic implementation, Patchwork++ places points outside its
segmentation range into nonground. Its output therefore partitions 100% of the
finite self-cropped input.

Patchwork++ additionally applies its adaptive elevation/flatness logic and
temporal ground-revert processing. The ROS wrapper disables reflected-noise
removal because intensity is not supported by that wrapper.

### Patchwork++ 105 m experiment

This changes only `max_range` from 80 to 105 m. The motivation was a discovered
range mismatch: segmentation ended at 80 m while clustering accepted points up
to 100 m forward. The extra 5 m beyond the clustering boundary covers lateral
range near the ROI corners. This configuration was evaluated but was not
written into the production YAML.

### Autoware RANSAC

- Axis: z.
- Maximum iterations: 200.
- Minimum trials: 1,000.
- Minimum points: 500.
- Plane outlier threshold: 0.15 m.
- Plane slope threshold: 12 degrees.
- Voxel size: 0.10 m on each axis.
- Height threshold: 0.18 m.
- Debug ground publication: enabled.

## Metric definitions

### Frame and rate metrics

- `matched frames`: frames for which all required topics shared an identical
  source header timestamp.
- `output rate`: `(N - 1) / (last header stamp - first header stamp)`.
- A 1:1 input/output frame count means the algorithm did not drop a frame in
  the captured window. It does not prove unlimited throughput.

### Point metrics

- All ratios use finite XYZ points after self-crop. NaN/no-return records are
  excluded.
- `finite ground ratio = finite ground points / finite cropped points`.
- `finite nonground ratio = finite nonground points / finite cropped points`.
- `partition completeness = (ground + nonground) / cropped finite points`.
- Median point counts describe a typical frame; means are used for ratios to
  avoid constructing a ratio from unrelated medians.

### Latency metrics

Latency is computed from MCAP recorder receive timestamps for messages carrying
the same source header stamp:

- Segmentation latency: later of ground/nonground receive time minus cropped
  receive time. For RANSAC, only nonground is usable because ground stamp is
  zero.
- Detection latency: detected-object receive time minus raw-cloud receive time.
  RANSAC uses cropped-to-detected time instead.

These are pipeline latency proxies, not isolated CPU execution times. They
include ROS scheduling, serialization, DDS delivery, and recorder timing.

### Detection consistency metrics

- `objects/frame` counts all cluster-derived detections in each frame.
- `object-count delta` is the mean absolute difference in object count between
  adjacent frames. Lower values suggest count stability, but do not prove
  stable identities or correct tracking.
- `total detections` is the sum across frames, not the number of unique objects.

### MORAI ground-truth matching proxy

The bag's `/ad/dev/objects` actor positions are in map coordinates. For each
detection timestamp, the nearest actor and `odom -> base_link` transform
messages were selected. `map -> odom` is identity in this recording. Actor
centers were transformed into `lidar_link` using the configured 1.15 m forward
LiDAR offset, then filtered by the clustering ROI:

```text
x: -4 .. 100 m
y: -25 .. 25 m
```

Detection centers and actor centers were greedily matched by smallest 2D
distance, one-to-one, using thresholds of 1.5, 2.5, and 4.0 m. The main table
uses 2.5 m.

This produces a recall and precision *proxy*, not standard perception AP:

- It uses center distance rather than 3D IoU.
- It ignores class because the Euclidean detector emits generic/unknown
  objects.
- It does not account for LiDAR visibility or occlusion.
- The MORAI actor topic omits static infrastructure such as walls, poles, and
  curbs, while the LiDAR detector intentionally reports some of these as
  obstacles. Such detections reduce the proxy precision even when useful for
  collision avoidance.
- The recording contains 104 actor-frame instances inside the ROI across the
  common evaluation frames, not 104 unique actors.

Consequently, recall proxy is more informative here than precision proxy.

## Ground-segmentation results

| Configuration | Matched frames | Output rate | Ground | Nonground | Partition | Nonground p50 | Segmentation latency p50 / p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classic Patchwork 80 m | 264 | 6.376 Hz | 81.75% | 15.22% | 96.98% | 1,516.5 | 3.42 / 4.31 ms |
| Patchwork++ 80 m | 261 | 6.359 Hz | 80.60% | 19.40% | 100.00% | 2,106 | 5.04 / 6.63 ms |
| Patchwork++ 105 m | 264 | 6.376 Hz | 82.47% | 17.53% | 100.00% | 1,851 | 5.12 / 6.71 ms |
| Autoware RANSAC | 261 | 6.456 Hz | unavailable | 20.11% | unavailable | 2,547 | 9.25 / 18.83 ms |

Observations:

- Each production-path Patchwork trial produced one output at every recorded
  input timestamp. Its output rate equals its captured input-section rate.
- Patchwork++ 80 m classified about 4.18 percentage points more finite input as
  nonground than classic. Much of the raw-cloud difference seen before
  self-crop was NaN/no-return or ego-body data; the production-path comparison
  above is the relevant one.
- Classic omitted about 3.02% of finite cropped points from both outputs due to
  radial filtering. Patchwork++ explicitly returns out-of-range points as
  nonground.
- Increasing Patchwork++ range to 105 m moved some 80--105 m points from
  automatic nonground into fitted ground. Nonground therefore fell from 19.40%
  to 17.53% even though the range increased.
- RANSAC had the largest and most variable latency. Its median was 2.70 times
  classic and its p95 was 4.37 times classic.
- RANSAC published every debug ground cloud with timestamp zero. It cannot be
  exact-time paired with its nonground output without a wrapper fix.

### Ground-label agreement

For identical timestamps and exact float XYZ coordinates:

| Pair | Common frames | Ground Jaccard | Classic ground retained | Candidate ground also classic |
|---|---:|---:|---:|---:|
| Classic vs Patchwork++ 80 m | 261 | 95.56% | 97.08% | 98.39% |
| Classic vs Patchwork++ 105 m | 264 | 92.37% | 96.45% | 95.63% |

The 80 m variants agree strongly. The 105 m variant changes more labels because
it actually estimates ground in a region that the 80 m variants treat
differently or omit.

## Downstream clustering and detection

| Configuration | Cluster points p50 | Objects/frame mean | p50 | p95 | max | Adjacent count delta | Detection latency p50 / p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classic Patchwork 80 m | 428.5 | 6.09 | 5 | 14 | 20 | 1.04 | 7.53 / 64.88 ms raw-to-detected |
| Patchwork++ 80 m | 450 | 6.56 | 6 | 15 | 20 | 1.16 | 9.19 / 68.68 ms raw-to-detected |
| Patchwork++ 105 m | 450 | 6.72 | 5 | 15 | 20 | 1.21 | 9.90 / 70.13 ms raw-to-detected |
| Autoware RANSAC | 450 | 6.68 | 6 | 16 | 19 | not measured | 10.96 / 77.43 ms cropped-to-detected |

On the 261 timestamps common to classic and Patchwork++ 80 m:

- Patchwork++ produced 0.487 more detections per frame on average.
- The two algorithms produced the same object count in 59.0% of frames.
- The additional detections did not produce additional 2.5 m actor matches.

On the 264 timestamps common to classic and Patchwork++ 105 m:

- Patchwork++ produced 0.625 more detections per frame on average.
- The algorithms produced the same object count in 52.3% of frames.
- Patchwork++ recovered three additional actor-frame matches.

The large p95 detection latency relative to the median affects every
configuration and is therefore likely dominated by full-pipeline scheduling or
recording rather than ground segmentation alone.

## MORAI actor matching

### Main 2.5 m threshold

| Configuration | Common frames | Actor-frame instances | Total detections | Matches | Recall proxy | Precision proxy | Mean matched distance |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classic vs Patchwork++ 80 comparison | 261 | 104 | 1,585 | 31 | 29.81% | 1.96% | 0.623 m |
| Patchwork++ 80 m | 261 | 104 | 1,712 | 31 | 29.81% | 1.81% | 0.613 m |
| Classic vs Patchwork++ 105 comparison | 264 | 104 | 1,609 | 31 | 29.81% | 1.93% | 0.623 m |
| Patchwork++ 105 m | 264 | 104 | 1,774 | 34 | 32.69% | 1.92% | 0.612 m |
| Classic on RANSAC-common frames | 254 | 104 | 1,535 | 31 | 29.81% | 2.02% | 0.623 m |
| Autoware RANSAC | 254 | 104 | 1,722 | 41 | 39.42% | 2.38% | 0.501 m |

The rows use pairwise-common frames, which is why classic total detections vary
slightly among comparisons. The matched classic result remains 31 in every
case.

### Threshold sensitivity

| Configuration | Matches at 1.5 m | Recall at 1.5 m | Matches at 2.5 m | Recall at 2.5 m | Matches at 4.0 m |
|---|---:|---:|---:|---:|---:|
| Classic Patchwork | 27 | 25.96% | 31 | 29.81% | 31 |
| Patchwork++ 80 m | 27 | 25.96% | 31 | 29.81% | 31 |
| Patchwork++ 105 m | 30 | 28.85% | 34 | 32.69% | 34 |
| Autoware RANSAC | 38 | 36.54% | 41 | 39.42% | 41 |

No configuration gains another match between 2.5 and 4.0 m. The ranking is
therefore not caused by choosing a narrowly favorable 2.5 m threshold.

## Interpretation by algorithm

### Classic Patchwork

Strengths:

- Lowest median and p95 segmentation latency.
- Same actor recall proxy as Patchwork++ 80 m.
- Fewer unmatched detections and slightly more stable adjacent frame counts.
- Fully compatible ground and nonground timestamps.

Risks:

- Does not partition about 3% of finite cropped points because points outside
  its radial range are omitted.
- The available recording does not exercise the slopes and uneven terrain
  where Patchwork++ is intended to help.

### Patchwork++ 80 m

Strengths:

- Complete finite-point partition.
- High ground-label agreement with classic.
- Latency remains well below a 100 ms VLP-16 scan period.

Risks:

- No actor recall improvement in this recording.
- More nonground points and detections without more matches.
- About 47% higher median segmentation latency than classic.
- The 80 m segmentation range conflicts with a clustering ROI extending to
  100 m forward.

### Patchwork++ 105 m

Strengths:

- Resolves the segmentation/clustering range mismatch.
- Recovers three more actor-frame instances than classic in this window.
- Preserves every finite input point.

Risks:

- Only a 2.88 percentage-point absolute recall gain on 104 actor-frame
  instances.
- About 10.3% more total detections than classic on common frames.
- About 49% higher median segmentation latency than classic.
- Lower ground-label agreement with classic because the additional radial area
  is actively classified.

### Autoware RANSAC

Strengths:

- Highest simulated-actor recall proxy and shortest mean matched-center
  distance in this recording.
- Nonground output successfully passes through the current finite filter and
  Euclidean detector when fed the production self-cropped cloud.

Risks:

- Highest and most variable segmentation latency.
- All debug ground messages have stamp zero.
- Ground output is voxel-downsampled debug data, not a lossless complement of
  nonground; partition completeness cannot be compared fairly.
- A single dominant-plane model is less attractive for slopes and multiple
  road surfaces despite its result on this flat recording.

## Decision matrix

| Goal | Preferred candidate | Reason |
|---|---|---|
| Lowest latency on current flat road | Classic Patchwork | 3.42 ms median, same recall as Patchwork++ 80 m |
| Preserve every finite point | Patchwork++ | 100% partition completeness |
| Best actor recall in this one static bag | RANSAC | 41/104 matches, but with integration and latency costs |
| Patchwork++ tuning candidate | Patchwork++ 105 m | Better-aligned range and modest recall gain |
| Production default today | Classic Patchwork | Best-supported tradeoff from current evidence |
| Future uneven-road candidate | Patchwork++ 105 m | Requires scenario-diverse confirmation |

## Recommended acceptance gates

Do not choose a production default from this bag alone. Collect or generate
recordings for at least:

- flat open road with and without traffic;
- uphill, downhill, and crest transitions;
- banked turns;
- curbs, speed bumps, ramps, and potholes;
- tunnel entrance/exit and strong pitch changes;
- stationary and moving ego vehicle;
- near pedestrians/bicycles and long-range vehicles;
- rain/noise or deliberately injected invalid returns if available.

For every scenario, require:

- 1:1 input-to-nonground frame delivery at the target LiDAR rate;
- segmentation p95 comfortably below the scan period;
- no malformed PointCloud2 fields, sizes, or zero timestamps;
- actor recall reported separately by distance band and object type;
- static-obstacle labels or manual review so poles/walls are not incorrectly
  counted as false positives;
- end-to-end tracking continuity, not only per-frame detection count;
- repeated runs to report median and variance rather than one execution.

A reasonable promotion rule would require Patchwork++ to improve recall or
ground IoU on challenging scenarios without materially increasing unmatched
clusters, while maintaining p95 segmentation latency below the real-time
budget. RANSAC should additionally be blocked on correcting its ground header.

## Reproduction notes

The key production launch was invoked at playback rate 1.0 with the ground
configuration replaced per trial. Separate ROS domain IDs were used. The
important topics recorded were:

```text
/ad/sensors/lidar/points
/ad/perception/lidar/cropped
/ad/perception/lidar/ground
/ad/perception/lidar/nonground
/ad/perception/lidar/nonground_finite
/ad/perception/lidar/clusters
/ad/perception/objects/detected
```

The evaluation generated MCAP artifacts under `/tmp/heven_*`, totaling roughly
1.7 GiB. Those files are temporary and are not part of the repository. The
machine-readable summary accompanying this report is
`ground-segmentation-benchmark-2026-08-12.csv`.

## Known limitations

- One static simulation bag; no real LiDAR recording.
- No manually annotated ground/non-ground point labels.
- No bounding-box IoU or standard AP computation.
- Sparse actor ground truth inside the ROI: 104 actor-frame instances.
- Startup timing gives slightly different total captured frames; pairwise
  comparisons use common source timestamps.
- Recorder receive-time latency includes middleware and scheduling.
- CPU frequency, thermal state, and system load were not pinned.
- RANSAC did not run through the identical raw-to-crop launch process in its
  final trial, so only its segmentation-stage latency is directly comparable.
- No repeat-run variance or confidence intervals were measured.

These limitations mean the report supports an engineering decision about what
to test next and what not to promote yet; it does not establish general
algorithm superiority.
