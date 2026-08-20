# Autoware vs AB3DMOT: initial same-scene runtime comparison

**This is an initial same-scene runtime comparison, not a tracking-accuracy
benchmark.** No ground truth, TrackEval, or any GT-based metric was used.
Both trackers ran once, unmodified and untuned, against the same replayed
detection stream. Do not read differing object/track counts below as
accuracy differences — see Limitations.

## Scope

- Not a TrackEval integration. No new tracking algorithm was implemented.
- Neither tracker's parameters were changed or tuned.
- RViz was not modified. This comparison reuses the RViz Tracking
  Comparison milestone's topics for the qualitative cases below, but the
  cases themselves were identified from the same recorded message data,
  not a GUI screenshot (no screenshot capability in this environment,
  same limitation already documented in the RViz Tracking Comparison
  session).
- Both trackers subscribed the exact same `/ad/perception/objects/detected`
  stream, published by the same Euclidean cluster node, fed by the same
  replayed MORAI frames — satisfying "same input for both."

## Exact replay source and configuration

- Dataset: `~/datasets/morai_heven`, `train` split, `static_20260805_003151`
  scene (same dataset used by T-1C and the CenterPoint-vs-Euclidean
  comparison).
- Replay: `ad_publish_morai_frames --dataset ~/datasets/morai_heven --split
  train --count 400 --topic /ad/perception/lidar/cropped --interval-sec
  0.15` (single pass, no repeat) — 400 frames, ~65 s wall time.
- Detector: `euclidean_clustering.launch.py finite_filter_enabled:=false
  finite_input_topic:=/ad/perception/lidar/cropped` — same Euclidean
  cluster node used throughout this branch's work, ground-segmentation
  bypassed (same already-documented caveat as
  `docs/perception/centerpoint_vs_euclidean_comparison.md`).
- TF: the same `odom->base_link` (identity) + `base_link->lidar_link`
  (`z=1.70`, identity) static chain established in T-1C, published via
  `tf2_ros.StaticTransformBroadcaster` (the fix verified in the RViz
  Tracking Comparison session).
- Tracker A: `ros2 launch ad_lidar_perception tracking.launch.py` —
  production Autoware `multi_object_tracker`, unmodified, config
  `ad_lidar_perception/config/tracking/autoware.yaml`, unchanged.
- Tracker B: `ros2 launch ad_lidar_perception ab3dmot_tracker.launch.py
  enabled:=true` — experimental AB3DMOT node, unmodified, config
  `ad_lidar_perception/config/tracking/ab3dmot.yaml` (`giou_3d`, greedy,
  `min_hits=1`, `max_age=2`, `giou_gate=0.0`), same values as T-1C — not
  tuned for this comparison.
- Both processes ran simultaneously for the entire replay, each consuming
  the same `/ad/perception/objects/detected` topic independently.
- Recording: a scratch-only `rclpy` subscriber (not part of the committed
  package; single process, three subscriptions) wrote one JSONL record per
  message from `/ad/perception/objects/detected`,
  `/ad/perception/objects/tracked` (Autoware), and
  `/experiment/tracked/ab3dmot` (AB3DMOT) — header stamp, object count,
  per-object UUID/pose/twist/dimensions/classification, and this
  recorder's own wall-clock receipt time (used only for the latency probe
  below). Not committed — regenerable via the commands above plus this
  recorder script.

## Result table

| metric | Autoware (`/ad/perception/objects/tracked`) | AB3DMOT (`/experiment/tracked/ab3dmot`) |
|---|---|---|
| input messages consumed | 400 | 400 |
| output messages | 398 | 400 |
| output frequency | 6.11 Hz | 6.14 Hz |
| mean objects/frame | 9.92 | 19.28 |
| median objects/frame | 9.0 | 20.0 |
| max objects in one frame | 30 (frame 323, t=52.9 s) | 50 (frame 350, t=56.9 s) |
| unique track IDs over the run | 474 | 1,720 |
| track lifetime — mean | 1.24 s | 14.62 s |
| track lifetime — median | 0.50 s | 0.17 s |
| track lifetime — p90 | 2.96 s | 43.17 s |
| single-frame-only tracks | 41 (8.6% of tracks) | 11 (0.6% of tracks) |
| tracks born after frame 0 | 474 / 474 (100%) | 1,717 / 1,720 (99.8%) |
| tracks not seen in the last 2 frames | 456 / 474 (96%) | 1,700 / 1,720 (99%) |
| duplicate track IDs within one frame | 0 | 0 |
| NaN/Inf in position/velocity/orientation/dims | 0 | 0 |
| non-unit-norm quaternions (tol 1e-2) | 0 | 0 |
| non-positive box dimensions | 0 | 0 |
| receive-to-receive latency (external probe) — mean/median/p95/max | 1.81 / 1.64 / 3.44 / 6.64 ms (n=398) | 26.9 / 23.5 / 60.1 / 112.9 ms (n=399, see note) | 

Longest-persisting tracks (uuid/id, lifetime, observation count, first→last
frame):

- Autoware: `...db829223e2b32a73` (id suffix `0500…`) — 64.80 s, 397/398
  frames (essentially the entire run, frame 1→397).
- AB3DMOT: track id `9` — 59.50 s, 80 observations, frame 33→399; track id
  `2` — 53.75 s, 140 observations, frame 0→331; track id `1` — 53.05 s,
  222 observations, frame 0→326.

**Latency note**: measured via an external, single-process `rclpy`
subscriber timing its own wall-clock receipt of the matching-stamp
`detected` message vs. each tracker's output message — the same technique,
applied identically and without modifying either node, satisfying "fair
and comparable... without modifying their algorithms." One AB3DMOT sample
(1 of 400) showed an impossible negative delta (-550 ms); traced to the
single recorder process's own callback-scheduling order across three
topics, not tracker behavior, and excluded from the AB3DMOT statistics
above (stated explicitly rather than silently smoothed over). This
measures each **complete node's** receive→publish latency, including
implementation language (Autoware's tracker is compiled C++; AB3DMOT's is
Python/rclpy) — it is not a pure algorithm-cost comparison, and no claim
is made that AB3DMOT's association/KF math is inherently slower.

## Notable qualitative observations

1. **Autoware ramps up slowly at the start of the replay; AB3DMOT does
   not** — sampled per-frame object counts: frame 0 → Autoware 0 objects,
   AB3DMOT 3; frame 50 (t=8.3 s) → Autoware 1, AB3DMOT 9; frame 150
   (t=24.4 s) → Autoware 3, AB3DMOT 7. This is a direct, expected
   consequence of the two trackers' already-documented config difference
   (Autoware's own confirmation-count gating vs. AB3DMOT's `min_hits=1`,
   unchanged from T-1A's resolved decision) — not a claim that either
   tracker's object count is more "correct."
2. **AB3DMOT accumulates roughly 3.6x more unique track IDs than Autoware
   over the same window** (1,720 vs 474), while still showing a
   proportionally *smaller* fraction of single-frame-only tracks (0.6% vs
   8.6%). Both trackers show heavy churn (>95% of all tracks end before
   the replay's last 2 frames) in this un-ground-segmented, Euclidean-fed
   replay — a data/detector-input characteristic already flagged in
   `docs/perception/centerpoint_vs_euclidean_comparison.md`, not a claim
   about either tracker's real-world ID-switch rate.
3. **AB3DMOT's track-lifetime distribution is strongly bimodal**: a
   median of 0.17 s (many short-lived churn tracks) alongside a mean of
   14.6 s and a p90 of 43.2 s, driven by a handful of very persistent
   tracks (ids 1-9, alive most of the 65 s run). Autoware's distribution
   is comparatively tighter (median 0.50 s, p90 2.96 s) with one dominant
   near-full-run track. This qualitative shape difference is recorded as
   an observation of this specific run, not a general claim.
4. **Both output streams were fully valid** across the entire run: zero
   NaN/Inf values, zero non-unit-norm quaternions, zero non-positive box
   dimensions, zero duplicate track IDs within any single frame, for both
   trackers — no data-integrity defect was found in either.

## Limitations

- Single MORAI scene, single 65 s replay pass, no repeated trials — these
  are point-in-time counts, not statistically averaged results.
- No ground truth is available for this scene; object/track counts,
  lifetimes, and ID totals are **not** accuracy, precision, recall,
  ID-switch-rate, or MOTA/HOTA metrics. Higher or lower counts are not
  interpreted as better or worse tracking here.
- Ground segmentation is bypassed for both trackers' shared detection
  input (pre-existing, already-documented Euclidean-replay caveat) — the
  detection stream itself may include ground-adjacent clusters, which
  both trackers process identically but which is not representative of
  Euclidean's normal production input.
- The latency probe measures each complete ROS node's receive→publish
  time via one external subscriber process, not each algorithm's inner
  compute cost in isolation, and is influenced by implementation language
  (C++ vs Python) as noted above.
- "Visible disagreement" (observation 1) reflects a known, already-decided
  configuration difference between the two trackers' baseline configs, not
  an unexplained bug.
- No TrackEval, MOTA, HOTA, or any GT-based metric was computed. No claim
  of higher tracking accuracy, better generalization, or a lower ID-switch
  rate is made for either tracker.
