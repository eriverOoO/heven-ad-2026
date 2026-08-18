# STATUS

## RViz Tracking Comparison

Branch: `feat/ab3dmot-tracker`. Visualization-only milestone extending the
existing `ad_viz` perception visualizer to show the Autoware tracker and
the experimental AB3DMOT tracker at the same time, off the same detection
stream, reusing the exact T-1C MORAI replay/TF setup. No tracking
algorithm, association, KF, Autoware tracker, detector, IMM/prediction, or
occupancy code was touched — only `ad_viz` (marker builder, node, CMake,
tests) and `ad_lidar_perception`'s visualization launch/RViz-config/tests.

### Approach

Reused the existing `perception_visualizer_node` executable by launching
it **twice** with different parameters, rather than creating a second
visualization package or node:
- Instance 1 (unchanged topics): `id_prefix:="A-"`, subscribes
  `/ad/perception/objects/tracked` (Autoware), publishes
  `/ad/visualization/tracked_objects`.
- Instance 2 (new): `id_prefix:="B-"`, subscribes
  `/experiment/tracked/ab3dmot`, publishes
  `/experiment/visualization/tracked_objects_ab3dmot`,
  `visualize_detections:=false` and `visualize_predictions:=false` (avoids
  duplicate detection markers and an unused prediction subscription).

`ad_lidar_perception/launch/perception_visualization.launch.py` now starts
both instances; `ad_lidar_perception/rviz/heven_perception.rviz` gained a
new "Tracked Objects (AB3DMOT)" `MarkerArray` display on the new topic,
and the old "Tracked Objects" display was renamed "Tracked Objects
(Autoware)" for clarity.

### Track visualization features (all in `ad_viz`)

- **ID prefix / disambiguation**: `ObjectMarkerConfig::id_prefix` is
  prepended to each track's on-screen label, so the two trackers never
  rely on color alone (`A-<id>` / `B-<id>`), per the task requirement.
- **Fixed a pre-existing label bug found while touching this code**: the
  ID suffix shown in the label was `uuid.substr(0, 8)` (first 8 hex
  chars). AB3DMOT's UUID encoding (`ab3dmot_ros.py::track_id_to_uuid`,
  from T-1B) packs its integer track id into the **low 8 bytes**
  (`uuid[8:16]`), so every AB3DMOT track showed an identical, useless
  `00000000` suffix. Fixed to take the **last 16 hex chars** (last 8
  bytes) — verified live: AB3DMOT labels now read e.g. `B-0000000000000001`,
  `B-00000000000000ce` (correct, distinct, matches the tracker's own small
  integer ids); Autoware labels remain distinct (its own UUIDs are
  effectively random in all 16 bytes), e.g. `A-7b4fb59b736c7e9f...`. Also
  exported `uuid_hex()` from `object_marker_builder` so the new trajectory
  code can key its per-track state identically without re-deriving the
  encoding.
- **Trajectory history**: new `ad_viz::perception::TrajectoryHistory`
  class (`trajectory_history.hpp`/`.cpp`) — a per-track key bounded
  `std::deque<Point>` (oldest dropped once `trajectory_max_points`, default
  30, is exceeded) plus time-based pruning (`prune_stale`, default 3.0 s
  timeout) so a track's history is fully removed once its tracker stops
  publishing it. Operates on plain `int64_t` nanosecond stamps (not
  `rclcpp::Time`), matching this repo's established convention for
  avoiding ROS clock-type pitfalls (`imm_predictor.hpp` etc.). Only ever
  stores already-observed positions — never predicts/extrapolates.
- **`build_trajectory_markers`**: renders one `LINE_STRIP` per track with
  >=2 points (namespace `trajectory/<id_prefix><uuid_hex>`); emits no
  `DELETEALL` of its own — `PerceptionVisualizerNode::on_tracks` appends
  its output into the *same* `MarkerArray`/topic as
  `build_tracked_markers` (which already emits one `DELETEALL` per
  publish), so stale trajectory markers are cleared in the same frame as
  stale boxes/labels — no orphaned markers, confirmed live (see below).
- New node params: `id_prefix` (string, default `""`), `visualize_detections`
  (bool, default `true`, gates the detection subscription/publisher so the
  second AB3DMOT-only instance doesn't republish detections twice),
  `trajectory_max_points` (int, default 30), `trajectory_stale_timeout_sec`
  (double, default 3.0).

### Tests

`colcon build --symlink-install --packages-up-to ad_viz` and
`--packages-select ad_lidar_perception`, both clean, after fixing one
build-environment issue unrelated to this task's code (the shared
`heven_ros_ws` build tree had a stale cached Python interpreter path from
an earlier venv-activated session, breaking `rosidl`/`ament` for several
packages incl. `ad_interfaces`; fixed by clearing those 4 packages' build
directories — pure build artifacts, not source — so CMake re-detected the
correct interpreter).

New/updated gtests, all passing (`ctest` in `ad_viz`, full suite: 7/7,
including the 2 new/changed ones plus the pre-existing 5): `ad_viz`
`test_trajectory_history` (new — 4 cases: rejects invalid construction,
accumulates points per track, bounds points per track with oldest-dropped,
prunes stale tracks by elapsed time) and `test_object_marker_builder`
(updated: fixed the now-stale first-8-hex-char UUID assertion to the
corrected last-16-hex-char format; added 2 new cases covering
`id_prefix` label prepending and `build_trajectory_markers`'s
short-history-skip / prefixed-namespace behavior).

`ad_lidar_perception`'s full `ctest -R "launch|rviz|visualiz"` (8/8,
including `test_perception_visualization_launch`, updated for the new
second `Node` instance, the renamed/added RViz `MarkerArray` displays, and
the `id_prefix`/`tracked_input_topic` launch-arg wiring) — all pass.

### Live RViz smoke test — actually run, not simulated

Reused the exact T-1C setup: same static `odom->base_link` (identity) +
`base_link->lidar_link` (`z=1.70`, identity) TF precedent (this time
correctly published via `tf2_ros.StaticTransformBroadcaster` on
`/tf_static`, not a periodically-republished dynamic `/tf` — an earlier
attempt using a repeating dynamic broadcaster hit a real "extrapolation
into the future" TF race against live detection timestamps; switching to
static, and restarting all TF-listening nodes fresh afterward so no node
retained a poisoned mixed static/dynamic TF cache, fixed it cleanly), same
`ad_publish_morai_frames` replay of `~/datasets/morai_heven` (`train`
split), same Euclidean detector feeding both trackers off one
`/ad/perception/objects/detected` stream. Launched: `ab3dmot_tracker.launch.py
enabled:=true`, `tracking.launch.py` (Autoware), `euclidean_clustering.launch.py`,
and `perception_visualization.launch.py start_rviz:=true` (both visualizer
instances + a real `rviz2` process, confirmed alive with a real OpenGL
context: `Stereo is NOT SUPPORTED` / `OpenGl version: 4.2` in its log).

**No pixel-level screenshot was possible in this environment** (no `sudo`
for `imagemagick`/`xwd`; `PIL.ImageGrab` against the available WSLg X
display failed with an X `BadMatch` error). Per this repo's own established
precedent for exactly this situation (`docs/research/centerpoint_status.md`:
"verified by direct topic echo, not a GUI screenshot"), verification was
instead done directly against the live `MarkerArray` content on the two
topics RViz displays, plus confirming RViz's own subscriptions:

1. **Both IDs visible with correct prefixes** — confirmed directly:
   Autoware labels `"UNKNOWN 1.00 A-7b4fb59b736c7e9f"` etc. (distinct
   16-hex-char suffixes per track); AB3DMOT labels
   `"UNKNOWN 1.00 B-0000000000000001"`, `"...B-00000000000000ce"` etc.
   (small integer ids, now fully shown thanks to the 16-char-suffix fix
   above).
2. **Boxes plausible / aligned** — sampled absolute box pose positions
   from a live AB3DMOT message: x in roughly [-3, 43] m, y in [-12, 19] m,
   z in [0.9, 2.9] m (odom frame) — consistent with the cropped point
   cloud's own configured range and the `z=1.70` m LiDAR mount height;
   not a pixel check, but geometrically plausible, not garbage/NaN.
3. **Velocity arrows render** — `Marker::ARROW` (`type: 0`) present per
   moving track in both trackers' live output.
4. **Trajectories update over time and are bounded** — sampled the same
   AB3DMOT/Autoware topics repeatedly over ~15 s: per-track `LINE_STRIP`
   point counts grew (2 -> 7 -> 10 -> ...) and were observed reaching
   exactly the configured cap of **30** on both trackers' longest-lived
   tracks — the bound holds in the live pipeline, not just in the unit
   test.
5. **Stale histories disappear** — resampled ~14 s apart: AB3DMOT
   trajectory-track count dropped from 80 (many short-lived, quickly
   replaced tracks — see below) to 20, and specific old track-id
   namespaces confirmed **absent** from the later message — stale
   pruning is working live, not just in `test_trajectory_history`.
6. **No marker-array accumulation** — sampled the AB3DMOT visualization
   topic's total marker count twice, 10 s apart, mid-replay: 128 both
   times (stable) — confirms the per-frame `DELETEALL` + bounded history
   design does not leak markers over a sustained run.
7. **RViz actually subscribed** — `ros2 topic info --verbose` on both
   `/ad/visualization/tracked_objects` and
   `/experiment/visualization/tracked_objects_ab3dmot` shows exactly one
   subscriber each: node `heven_perception_rviz` — the real RViz process
   launched above, confirming the `.rviz` config's two `MarkerArray`
   displays are wired to the right topics.
8. **Both trackers' output topics publish continuously off the same
   detection stream** during replay: `/ad/perception/objects/detected`
   ~8.6 Hz, `/experiment/tracked/ab3dmot` ~8.5 Hz, `/ad/perception/objects/tracked`
   ~8.2-8.4 Hz (`ros2 topic hz`, mid-replay).

### Qualitative observations (real, not manufactured)

- **Track-ID churn / fragmentation is real and frequent** in this
  baseline AB3DMOT config (`min_hits=1`, `giou_gate=0.0`, greedy
  matching): one sampled instant showed ~80 distinct AB3DMOT track-history
  namespaces alive at once, the large majority with only 2 trajectory
  points (i.e. created and about to be replaced almost immediately) and a
  handful with 7-10+ points (persistent, stable tracks). This is
  consistent with §3's already-documented, not-yet-calibrated
  `giou_gate` default (flagged in the AB3DMOT Integration Decisions as
  needing real-data calibration) — this session did not tune it, only
  observed and recorded the resulting visual behavior.
- **Persistent tracking**: multiple track ids (e.g. AB3DMOT's low integer
  ids `0x1..0x20` range, well as several Autoware UUIDs) remained present
  and updating across many consecutive messages during the replay window,
  with trajectories growing smoothly up to the 30-point cap — normal,
  stable tracking is visibly present alongside the high-churn tracks
  above.
- **Birth/deletion**: directly observed via the stale-pruning check above
  (§5) — dozens of AB3DMOT tracks born and pruned within a ~14 s window.
- **Crossing objects / ID switches**: not specifically isolated in this
  session (would require per-object trajectory-shape inspection across a
  longer window than the spot-checks performed here); not claimed either
  way beyond the general churn pattern already documented above. No
  accuracy claim is made from this RViz-only verification, per this
  task's own instruction.

### Files changed by this task (RViz Tracking Comparison; not yet
committed/pushed — see below)

```
ad_viz/CMakeLists.txt
ad_viz/include/ad_viz/perception/object_marker_builder.hpp
ad_viz/include/ad_viz/perception/perception_visualizer_node.hpp
ad_viz/include/ad_viz/perception/trajectory_history.hpp   (new)
ad_viz/src/perception/object_marker_builder.cpp
ad_viz/src/perception/perception_visualizer_node.cpp
ad_viz/src/perception/trajectory_history.cpp               (new)
ad_viz/test/test_object_marker_builder.cpp
ad_viz/test/test_trajectory_history.cpp                     (new)
ad_lidar_perception/launch/perception_visualization.launch.py
ad_lidar_perception/rviz/heven_perception.rviz
ad_lidar_perception/test/test_perception_visualization_launch.py
docs/agent/STATUS.md (this update)
```

**A. Pre-existing unrelated dirty files** (unchanged, not staged/touched
by this task — same 19 files as every prior task this session):

```
.claude/skills/bootstrap-repo/scripts/bootstrap-repo.sh
.claude/skills/bootstrap-repo/tests/test_bootstrap_repo.sh
.claude/skills/issue-worker/scripts/claim_issue.sh
.claude/skills/issue-worker/tests/test_claim_issue.sh
ops/runner/bin/agent-tick
ops/runner/bin/codex-pr-review.sh
ops/runner/bin/pr-set-in-review.sh
ops/runner/bin/render_prompt.sh
ops/runner/repo-context.sh
ops/runner/tests/test_env_precedence.sh
ops/runner/tests/test_pr_set_in_review.sh
scripts/apply_dependency_patches.sh
scripts/bootstrap_workspace.sh
scripts/check_autoware_perception.py
scripts/setup_dev_env.sh
scripts/test_python.sh
scripts/tests/test_verify_template_contract.sh
scripts/verify-template-contract.sh
scripts/verify_ad_data.py
```

### Blockers

None. Live comparison works end-to-end. Not committed/pushed per this
task's explicit instruction — left in the working tree for a future,
separate commit task.

## Previous: T-1C — Runtime verification of the AB3DMOT ROS2 tracker

Branch: `feat/ab3dmot-tracker`. **T-1C: PASS.** Actually launched the real
ROS2 graph (not just unit tests) and replayed real MORAI-exported LiDAR
frames through it. Used the currently-committed AB3DMOT configuration
unchanged (`ad_lidar_perception/config/tracking/ab3dmot.yaml`: `giou_3d`,
greedy, `min_hits=1`, `max_age=2`, `giou_gate=0.0`) — no parameters were
tuned. No RViz work started, no Autoware source modified.

### Replay/data source

`~/datasets/morai_heven` (`train` split, 1,764 exported frames of the
single `static_20260805_003151` scene, per prior sessions' audit — same
dataset the Euclidean/CenterPoint comparison tooling already uses). The
existing `ad_publish_morai_frames` executable (built in a prior T‑1B/CenterPoint
session) republished sequential samples onto `/ad/perception/lidar/cropped`
at ~6 Hz. No new replay system was created.

### TF prerequisite — reused an existing, explicit precedent

Live MORAI localization was not available in this environment (no running
MORAI bridge/simulator). Rather than inventing a workaround, this session
found that `ad_lidar_perception/test/test_autoware_pipeline_integration.py`
**already defines** exactly this "replay/test mode": it broadcasts a
synthetic `odom→base_link` (identity) + `base_link→lidar_link`
(translation `z=1.70`, identity rotation) TF chain for testing the Autoware
tracker pipeline without live localization. T‑1C reused that exact, already-
established transform (via a `StaticTransformBroadcaster`, run from the
session scratch directory only — not added to the repository) rather than
inventing a new one. `ros2 run tf2_ros tf2_echo odom lidar_link` confirmed
the chain resolved correctly before any detection traffic was sent.

### Exact commands (environment sourcing omitted for brevity — standard
ROS Humble + `autoware_perception_msgs` local overlay + `autoware_tracker_ws`
overlay + `heven_ros_ws` install + `heven-centerpoint` venv, same as prior
sessions)

```
ros2 launch ad_lidar_perception ab3dmot_tracker.launch.py enabled:=true
ros2 launch ad_lidar_perception tracking.launch.py
ros2 launch ad_lidar_perception euclidean_clustering.launch.py \
  finite_filter_enabled:=false finite_input_topic:=/ad/perception/lidar/cropped
ros2 run ad_lidar_perception ad_publish_morai_frames \
  --dataset ~/datasets/morai_heven --split train --count 80 \
  --topic /ad/perception/lidar/cropped --interval-sec 0.15
```

Both trackers subscribed the same `/ad/perception/objects/detected` topic
published by the Euclidean cluster node (fed, in turn, from the same
replayed MORAI frames) — satisfying "the two trackers must consume the
same detection stream." Output was captured with `ros2 topic echo
--full-length` into scratch YAML files and parsed with a small local
PyYAML script (not committed).

### 1. Detection input / 2. AB3DMOT output — both continuously published

- `/ad/perception/objects/detected`: ~5.98 Hz (`ros2 topic hz`, 77-sample
  window), matching the publisher's 0.15 s interval exactly. Source frame:
  `lidar_link`.
- `/experiment/tracked/ab3dmot`: ~5.98 Hz, **80/80** detection messages
  produced exactly 80 tracked-output messages (1:1, no drops, no gaps) —
  confirmed continuous, not just "sometimes publishes."
- Message publication alone was **not** treated as sufficient — see §4-8
  below for actual tracking-quality evidence.

### 3. Output frame — 100% odom

All 80 sampled `/experiment/tracked/ab3dmot` messages: `frame_id: odom`.
Zero frame changes, zero TF failures during the replay window (the TF
chain was up before any detections were sent).

### 4. Timestamp semantics — valid, monotonic, detection-derived

- Every output stamp equals its triggering detection's own header stamp
  (verified directly in the parsed data — this is also enforced by
  `ab3dmot_tracker_node.py`'s design, not just observed).
- Stamps strictly monotonic across all 80 frames (`all(stamps[i] <
  stamps[i+1])` = `True`); span ≈13.2 s over 80 frames at ~6 Hz.
- No wall-clock substitution: the publisher stamps each replayed frame
  with its own `now()` at publish time, and the tracked-object stamps
  match those exactly, not a later processing-time clock read.
- No duplicate stamps occurred in this run (replay interval 0.15 s ≫ ROS
  clock resolution), so the documented skip-duplicate path was not
  exercised live here — it remains covered by
  `test_ab3dmot_ros.py::ClassifyTimestampTest` and
  `test_ab3dmot_tracker_node.py::test_duplicate_timestamp_is_skipped_not_republished`.
- **Clock rollback did not occur naturally** in this replay (the publisher
  uses a monotonically increasing wall clock, and MORAI's own simulator
  was not live in this environment to produce a real sim-time reset).
  Per this task's own instruction, this is stated plainly rather than
  forced: rollback-reset behavior remains verified by
  `test_ab3dmot_tracker_node.py::test_clock_rollback_resets_tracker_state`
  (which explicitly asserts a fresh `AB3DMOTTracker` instance is
  constructed and no negative dt reaches `step()`), not re-demonstrated
  against live MORAI clock behavior in this session.

### 5. Persistent track IDs — multiple concrete, verified examples

At least 3 tracks persisted across many consecutive frames (IDs read
directly from each message's `object_id.uuid`, decoded via the same
`track_id_to_uuid` scheme the node uses — not inferred from position
similarity):

| track ID | first ts (ns, relative) | last ts (ns, relative) | consecutive obs | start pos (x,y,z) | end pos (x,y,z) |
|---|---|---|---|---|---|
| 2 | 0 | 13,202,307,199 | **80/80** (entire replay) | (-1.11, -0.00, 1.43) | (-1.11, 0.00, 1.43) |
| 3 | 0 | 4,368,447,013 | 27 | (-2.31, 5.33, 2.03) | (-3.43, 5.32, 2.17) |
| 1 | 0 | 4,200,644,691 | 26 | (-2.65, -4.34, 1.99) | (-3.39, -4.33, 2.01) |
| 4 | 1,200,053,125 (born frame 7, mid-run) | 3,703,748,803 | 16 | (84.06, 11.20, 3.18) | (83.98, 11.19, 3.18) |

Track 2 in particular is a clean, unambiguous example: the *same* AB3DMOT
ID (decoded from the message's real UUID field, not guessed) tracked one
near-stationary object across the full 80-frame, 13.2 s replay with a
sub-centimeter position range — this is real persistent tracking, not
just repeated publication.

### 6. Velocity — finite, m/s-scaled, plausible for this replay

- All 586 sampled velocity vectors across all tracks: finite (0 NaN/Inf).
- 94.4% of samples were near-zero (< 0.05 m/s) — consistent with this
  MORAI capture being a **static scene** (confirmed in prior sessions):
  most Euclidean clusters here are static-object/ground-remnant clusters,
  so near-zero velocity is the *correct*, plausible result, not a bug.
- Speed distribution: p50 ≈ 0, p90 ≈ 0.02 m/s, p95 ≈ 0.12 m/s, p99 ≈ 9.0
  m/s, max ≈ 17.75 m/s.
- **No meters-per-frame scaling bug**: cross-checked track 16 (7
  consecutive obs, frames 42-48, median speed 7.22 m/s) directly against
  its own raw position delta — it moved 7.28 m over 1.003 s of real
  elapsed time (its own header-stamp span), i.e. ≈7.26 m/s independently
  computed from position alone, matching the *published* velocity to
  within numerical noise. Same cross-check on track 157 (5.2 m real
  displacement / 0.999 s ⇒ ≈5.2 m/s, matching its published 5.6-6.2 m/s
  range). This directly demonstrates the T-1B real-dt fix is working
  correctly in the live ROS path, not just in T-1A's unit tests.
- **Flagged, explained, not silently ignored**: the small population of
  higher-speed short-lived tracks (tracks 16, 157, and similar, all
  ≤7 consecutive observations) are very likely Euclidean-detector-level
  clustering jitter, not AB3DMOT defects — this replay path deliberately
  bypasses ground segmentation (documented caveat, inherited from
  `docs/perception/centerpoint_vs_euclidean_comparison.md`), so a handful
  of unstable ground-adjacent clusters are expected. This is a data/
  detector-input characteristic of this specific verification setup, not
  a claim about AB3DMOT's or Euclidean's real-world quality.

### 7. Track lifecycle — creation, update, and deletion all observed live; coast not naturally exercised

- **Creation**: 214 of 217 distinct track IDs were born after frame 0
  (i.e., mid-replay, not just at startup) — continuous birth activity
  throughout the run, e.g. track 4 born at frame 7.
- **Update**: every persistent-track example above is itself repeated
  `update()` evidence (16-80 consecutive matched updates per track).
- **Deletion**: 7 tracks with ≥3 observations ended cleanly before the
  final frame and never reappeared under the same ID within the window
  (e.g., track 4: last seen frame 22 of 79, track 16: last seen frame 48
  of 79) — consistent with the **unmodified, committed** `max_age=2`
  policy. `min_hits`/`max_age`/`giou_gate` were not changed to produce
  this.
- **Coast (temporary miss then reappear under the same ID)**: **did not
  occur naturally** in this 80-frame window (0 tracks showed a 1-frame
  gap followed by re-observation). Stated plainly rather than forced —
  this exact behavior remains covered by
  `test_ab3dmot_core.py::LifecycleTest::test_temporary_missed_detection_keeps_same_track`
  and `test_ab3dmot_tracker_node.py`'s equivalent, not re-demonstrated
  live here.

### 8. Output validity — zero invalid values found

Across all 80 sampled frames / all objects (max 26 objects in a single
frame): **0** NaN/Inf values, **0** non-unit-norm quaternions (tolerance
1e-3), **0** non-positive box dimensions, **0** frames with a duplicate
track ID, **0** unexpected `frame_id` values, **0** invalid timestamps.

### 9. Runtime latency

Measured directly (input-topic receipt → matching-stamp output-topic
receipt, both timestamped on the same subscriber process's clock — a
dedicated scratch probe script, not committed) over **100 samples** from a
separate 100-frame replay pass (0.10 s interval, ~6 Hz) against the same
running node:

| stat | value |
|---|---|
| sample count | 100 |
| mean | 7.31 ms |
| median | 4.52 ms |
| p95 | 21.96 ms |
| max | 27.46 ms |

Latency trended upward over the run (from ~2-4 ms early to ~15-27 ms
later) — this tracks the growing live track count (up to several hundred
tracks accumulate over a long run against this noisy, ground-inclusive
Euclidean stream; GIoU-matrix cost is `O(tracks × detections)`), not
random jitter. Input/output frequency: ≈6 Hz in, ≈6 Hz out, matched 1:1.
This is execution/runtime evidence only — **no comparison to Autoware's
own latency is made or implied**, per AGENTS.md "execution success is not
performance validation" and this task's explicit instruction not to claim
performance superiority.

### 10. Parallel Autoware path — confirmed unaffected

- Same replay, same detection stream, `tracking.launch.py`'s
  `multi_object_tracker` running simultaneously: 80/80 detection messages
  → 80 `/ad/perception/objects/tracked` messages (79/80 non-empty), all
  `frame_id: odom`, zero errors in its log.
- Direct simultaneous-rate check (separate short replay, both `ros2 topic
  hz` running at once): Autoware ≈6.045 Hz, AB3DMOT ≈6.047 Hz — both
  tracking the same ~6 Hz input rate concurrently with no observable
  interference.
- `ad_lidar_perception/config/tracking/autoware.yaml`,
  `tracking.launch.py`, and the Autoware `multi_object_tracker` source
  were not modified or touched by this session. AB3DMOT publishes only to
  its own `/experiment/tracked/ab3dmot` topic; `/ad/perception/objects/tracked`
  was never written to by AB3DMOT.
- No accuracy or quality comparison between the two trackers is made.

### Runtime bugs found

**None.** No genuine T-1B/T-1C integration bug was encountered during this
verification pass — the node behaved exactly as designed on the first
real replay attempt (no code changes were made during T-1C).

### Limitations

- Single static MORAI scene (already-documented dataset-diversity
  limitation, unrelated to T-1C) — velocities are mostly near-zero because
  most tracked clusters are genuinely static in this data, and the higher-
  speed tracks are explained by detector-level (Euclidean, ground-seg-
  bypassed) clustering jitter, not evaluated against any ground truth.
- Clock-rollback and single-frame-coast behaviors were not naturally
  exercised in this specific replay window; both remain verified at the
  unit-test level only (already covered in T-1A/T-1B).
- Latency was measured via a receive-to-receive proxy on a third
  subscriber process (clean, standard technique) rather than in-node
  instrumentation, since the node does not currently self-log per-callback
  timing (CenterPoint's node does; AB3DMOT's does not yet) — flagged as a
  possible small future improvement, not a defect.

## T-1C: PASS

## T-1 overall: PASS

All required conditions are met: T-1A passed (committed `cceef3d`); T-1B
passed (ROS2 integration, still uncommitted pending a separate commit
task); this session observed **real** runtime `/experiment/tracked/ab3dmot`
messages (not just unit tests); **multiple track IDs demonstrably persist**
across many consecutive frames with concrete evidence (track 2: 80/80
frames); output `frame_id` is `odom` on every sampled message; timestamp
semantics are valid and monotonic with no wall-clock substitution;
velocity semantics are valid (finite, m/s, cross-checked against real
position deltas, no meters-per-frame artifact); zero NaN/Inf/invalid-
quaternion/invalid-dimension/duplicate-ID/TF/runtime failures occurred;
and the production Autoware path was verified to keep publishing,
unaltered, throughout.

## Previous: T-1B — HEVEN ROS2 integration for the AB3DMOT tracking core

Branch: `feat/ab3dmot-tracker`. **T-1B: PASS.** Implemented the opt-in ROS2

Branch: `feat/ab3dmot-tracker`. **T-1B: PASS.** Implemented the opt-in ROS2
node wrapping T-1A's `AB3DMOTTracker`, per the resolved design in
`docs/research/tracking_architecture.md` ("AB3DMOT Integration
Decisions") and this task's explicit interface/frame/timestamp/covariance
requirements. No algorithm redesign, no RViz change, no runtime
performance comparison started.

### ROS interface (as specified, unchanged from the request)

```
input:  /ad/perception/objects/detected   (autoware_perception_msgs/msg/DetectedObjects)
output: /experiment/tracked/ab3dmot        (autoware_perception_msgs/msg/TrackedObjects, frame_id=odom)
```

Production path untouched and verified unaffected: `tracking.launch.py`,
`ad_lidar_perception/config/tracking/autoware.yaml`, and the Autoware
`multi_object_tracker` source were not read or modified this session
beyond what was already known from the prior audit. The new node
(`ad_ab3dmot_tracker`) is disabled by default (`enabled:=false` node
parameter default; the new launch file flips it to `true` at the launch
level only, mirroring `centerpoint_detector.launch.py`'s own pattern) and
runs on a separate topic, so both trackers can run simultaneously off the
same detections.

### Files changed

- `ad_lidar_perception/ad_lidar_perception/ab3dmot_ros.py` **(new)** —
  pure message<->core adapter functions, message types always injected
  (same pattern as `centerpoint_ros.py`): `stamp_to_ns`,
  `select_classification`, `normalized_quaternion`/`yaw_from_quaternion`/
  `quaternion_from_yaw` (ports of `autoware_prediction_node.cpp`'s
  validated helpers), `quaternion_multiply`/`quaternion_rotate_vector`/
  `transform_pose_z_up` (full quaternion composition, not a naive yaw-add,
  so a transform with any roll/pitch is still handled correctly),
  `classify_timestamp`/`TimestampDecision` (pure first/duplicate/
  increasing/rollback classifier), `detected_objects_to_detections`
  (whole-message rejection on any malformed object, matching
  `AutowarePredictionNode`'s existing convention), `track_id_to_uuid`
  (deterministic mechanical int->16-byte encoding), `tracked_state_to_message`/
  `tracked_states_to_message` (the ROS-side half of the already-resolved
  field-mapping table).
- `ad_lidar_perception/ad_lidar_perception/ab3dmot_tracker_node.py`
  **(new)** — the `rclpy.Node`: declares `enabled`/`input_topic`/
  `output_topic`/`target_frame`/`ab3dmot_root` plus the T-1A config
  parameters; owns a `tf2_ros.Buffer`/`TransformListener`; on each
  `DetectedObjects` callback: validates/classifies the header stamp
  (`classify_timestamp`) -> looks up `lidar_link -> odom` at the
  message's own stamp (skipped entirely for an empty message, so an empty
  "heartbeat" frame never depends on TF availability) -> converts to
  `Detection`s -> drives `AB3DMOTTracker.step()` with the message's own
  timestamp (never wall-clock) -> publishes `TrackedObjects`. All
  malformed-input/TF-failure/duplicate-timestamp paths log a warning and
  skip that callback without crashing (never publish stale or
  wrong-frame data); a clock rollback logs and **reconstructs a fresh
  `AB3DMOTTracker` instance** (cleanest way to reset all experimental
  track state without touching T-1A's core reset-free design) rather than
  ever calling `step()` with a negative dt.
- `ad_lidar_perception/ad_lidar_perception/ab3dmot_core.py` **(modified,
  additive only)** — added `label`/`label_probability`/
  `existence_probability` passthrough fields to `Detection`, `Track`, and
  `TrackedState` (all with defaults, so every existing T-1A test still
  passes unchanged). These are never read by the KF/association/lifecycle
  math — they exist only to carry a detection's classification/score onto
  its matched track for the ROS mapping, explicitly modeled on AB3DMOT's
  own reference `Tracker.update()`, which carries an analogous per-match
  `info` payload the same way (updated only on a matched update, unchanged
  on coast frames). No algorithm logic changed.
- `ad_lidar_perception/launch/ab3dmot_tracker.launch.py` **(new)** — loads
  the existing `ad_lidar_perception/config/tracking/ab3dmot.yaml` as the
  node's parameters file (association_metric/matcher/min_hits/max_age/
  giou_gate all come from that YAML, unchanged from T-1A), plus launch
  args for `enabled`/topics/`target_frame`/`ab3dmot_root`.
- `ad_lidar_perception/CMakeLists.txt` **(modified)** — one new
  `install(PROGRAMS ... RENAME ad_ab3dmot_tracker)` entry (same pattern as
  `ad_centerpoint_detector`) and two new `ament_add_pytest_test` entries.
  `config`/`launch` directories are already installed wholesale by an
  existing rule, so the new YAML/launch files needed no new install rule.
- `ad_lidar_perception/test/test_ab3dmot_ros.py`,
  `ad_lidar_perception/test/test_ab3dmot_tracker_node.py` **(new)** — see
  Tests below.
- `docs/agent/STATUS.md` (this update).

No change to `package.xml` was needed: `tf2_ros`/`tf2_geometry_msgs`/
`geometry_msgs`/`rclpy` were already declared dependencies; the new code
never directly imports `unique_identifier_msgs` (the UUID field is
populated by assigning a numpy array to the already-auto-constructed
`object_id.uuid`, not by constructing a `UUID()` message).

### Submodule import from the installed environment — verified, not just source-tree pytest

Explicitly checked (this task's specific concern): after a real
`colcon build --symlink-install --packages-select ad_lidar_perception`,
`ad_lidar_perception.ab3dmot_core.__file__` resolves to a *symlink* into
the install tree, and `Path(__file__).resolve()` follows that symlink back
to the real source-tree file — so `_DEFAULT_AB3DMOT_ROOT` still correctly
locates `references/ab3dmot` from the installed package, with no code
change needed. Verified directly: `_load_ab3dmot_kf_class()` (and the full
test suite) both pass when run only after sourcing
`~/projects/heven_ros_ws/install/setup.bash` (not the source-tree
`PYTHONPATH` shortcut used for quick iteration). **Caveat worth recording**:
this specifically relies on HEVEN's established `--symlink-install`
convention; a plain (non-symlink) colcon install would copy the `.py` file
and break this path-inference, at which point `ab3dmot_root` would need to
be passed explicitly as a launch argument/parameter (already supported —
see `ab3dmot_tracker.launch.py`'s `ab3dmot_root` arg) rather than relying
on the default. Not a blocker today; flagged for whoever changes the build
convention later.

### Tests/builds run

`colcon build --packages-select ad_lidar_perception` (twice: after the
node/launch/CMake changes, and again after final edits) — both clean.

`python -m unittest test_ab3dmot_geometry test_ab3dmot_core test_ab3dmot_ros
test_ab3dmot_tracker_node test_centerpoint_ros test_morai_replay
test_detection_recording`, run twice: once via a source-tree `PYTHONPATH`
override, once via the fully-sourced **installed** environment
(`install/setup.bash`) to satisfy the installed-environment verification
requirement directly, not just by inference.

**Result: 75/75 pass both times** (63 AB3DMOT-related: 8 geometry + 14
core [unchanged from T-1A] + 31 new `ab3dmot_ros` + 10 new
`ab3dmot_tracker_node`, covering every category this task listed —
DetectedObjects conversion, odom-frame output, timestamp preservation,
real-dt propagation through the full adapter [node-level velocity
converges to the true m/s speed], first-frame behavior, duplicate-
timestamp rejection, non-positive/malformed-stamp rejection, clock-
rollback tracker reset, stable track identity [same UUID across frames,
distinct across track ids], velocity output in m/s, empty detections
[proven independent of TF availability], TF-unavailable/failure handling,
and malformed/unsupported objects [non-bounding-box shape, non-positive
dimensions, empty classification] — plus 12 pre-existing, unaffected).

Also live-smoke-tested via `ros2 launch ad_lidar_perception
ab3dmot_tracker.launch.py enabled:=true`: node starts cleanly, subscribes
`/ad/perception/objects/detected`, publishes on
`/experiment/tracked/ab3dmot`; a one-shot synthetic `DetectedObjects`
message was published against it with no TF broadcaster running, and the
node correctly logged a TF-unavailable warning and did not crash or
publish — confirms the same graceful-failure path the unit tests exercise
in isolation also works against the real ROS graph. Process cleaned up
afterward.

### Blockers

None. T-1B passes fully. The symlink-install caveat above is documented,
not blocking.

## Previous: T-1A — AB3DMOT tracking core + focused unit tests

**T-1A: PASS**, committed as `cceef3d` (2026-08-18). Standalone,
ROS2-independent tracking core (`ab3dmot_geometry.py`, `ab3dmot_config.py`,
`ab3dmot_core.py`) + 22 tests. Full detail in this file's prior revision
(git history) and `docs/research/tracking_architecture.md`.

## Previous: AB3DMOT integration decisions / CP-1

Resolved 2026-08-18: yaw convention, tracking frame (`odom`), initial
config, TrackedObjects mapping, ROS interface — full detail in
`docs/research/tracking_architecture.md` "AB3DMOT Integration Decisions".
CP-1 (CenterPoint) passed and merged to `main` in PR #2 / commit `aa5cacb`
(see `docs/research/centerpoint_status.md`).

## Git state at end of T-1B

**A. Pre-existing unrelated dirty files** (recorded before this task
started, unchanged, not staged/touched):

```
.claude/skills/bootstrap-repo/scripts/bootstrap-repo.sh
.claude/skills/bootstrap-repo/tests/test_bootstrap_repo.sh
.claude/skills/issue-worker/scripts/claim_issue.sh
.claude/skills/issue-worker/tests/test_claim_issue.sh
ops/runner/bin/agent-tick
ops/runner/bin/codex-pr-review.sh
ops/runner/bin/pr-set-in-review.sh
ops/runner/bin/render_prompt.sh
ops/runner/repo-context.sh
ops/runner/tests/test_env_precedence.sh
ops/runner/tests/test_pr_set_in_review.sh
scripts/apply_dependency_patches.sh
scripts/bootstrap_workspace.sh
scripts/check_autoware_perception.py
scripts/setup_dev_env.sh
scripts/test_python.sh
scripts/tests/test_verify_template_contract.sh
scripts/verify-template-contract.sh
scripts/verify_ad_data.py
```

**B. T-1B changes** (all AB3DMOT-related, verified against the above — no
overlap):

```
 M ad_lidar_perception/CMakeLists.txt
 M ad_lidar_perception/ad_lidar_perception/ab3dmot_core.py
?? ad_lidar_perception/ad_lidar_perception/ab3dmot_ros.py
?? ad_lidar_perception/ad_lidar_perception/ab3dmot_tracker_node.py
?? ad_lidar_perception/launch/ab3dmot_tracker.launch.py
?? ad_lidar_perception/test/test_ab3dmot_ros.py
?? ad_lidar_perception/test/test_ab3dmot_tracker_node.py
```
(plus this `docs/agent/STATUS.md` update, and T-1A's already-committed
files from `cceef3d`, unchanged except the additive `ab3dmot_core.py` edit
above).

No Autoware tracker, Euclidean detector, CenterPoint detector, HEVEN
IMM/prediction, occupancy, RViz, or reference-submodule files were
touched (`references/ab3dmot`/`simpletrack`/`trackeval` all still clean
and pinned to their recorded SHAs). Not committed or pushed, per this
task's instruction.

## Exact next task: T-1C

Per this task's instruction: stop after T-1B tests/build succeed; do not
start T-1C. T-1C, once explicitly requested, is **runtime verification**
against real data — replaying a real MORAI bag (or the existing
`ad_publish_morai_frames`/`ad_record_detected_objects` comparison tooling
from `docs/perception/centerpoint_vs_euclidean_comparison.md`) through the
actual TF tree (`lidar_link -> base_link -> odom`, needs a real
localization/TF source running, unlike this session's TF-unavailable
smoke test) to confirm: (a) the node produces non-empty, geometrically
plausible tracks against real detections, (b) `/experiment/tracked/ab3dmot`
and `/ad/perception/objects/tracked` can run side-by-side without
interfering, and (c) latency/runtime cost is measured (per AGENTS.md
"Measure runtime/latency for competition-critical modules") — still no
RViz wiring change and no accuracy/performance claim beyond execution
evidence, per AGENTS.md "execution success is not performance validation."
