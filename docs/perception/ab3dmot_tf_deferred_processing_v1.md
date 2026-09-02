# AB3DMOT Exact-Stamp TF Deferred Processing v1

Fixes a real replay-scheduling race in the experimental AB3DMOT ROS tracker
(`ad_ab3dmot_tracker`, opt-in, `tracker_backend:=ab3dmot`) that was silently
discarding detection frames whose exact-stamp `odom <- lidar_link` transform
had not arrived *yet*, even though it would become available a short time
later. **No association/Hungarian, Linear KF, AB3DMOT state model, lifecycle
threshold, ground segmentation, Euclidean detector, prediction math, or
planner code changed.** Autoware remains the default production tracker.

## The observed race

On `morai_cam4_20260813_163222` (`enable_localization:=true`, since this bag
has raw GPS/IMU/vehicle-status but no recorded odometry/TF -- see "Replaying
a bag without recorded localization" in
`docs/perception/training_free_rviz_demo_v1.md`), detection throughput
(~4.2-4.5 Hz) was consistently far above tracked-object throughput
(~0.6-1.0 Hz), previously documented (unresolved) in that same doc's "Live
validation status" section. Replaying 10x slower did not materially change
the acceptance ratio, ruling out simple CPU/throughput starvation.

Directly reproduced this session: `ad_ab3dmot_tracker`'s detection callback
(`ab3dmot_tracker_node.py::_on_detected_objects`) requests
`self._tf_buffer.lookup_transform(target_frame, msg.header.frame_id,
msg.header.stamp)` -- the transform **at the detection's own exact source
stamp `t`**, never `Time()`/latest. Detection (self-crop -> ground-seg ->
finite-filter -> cluster, 4 hops) and localization (adapter ->
`gnss_imu_localizer` -> manager, 3 hops) are two independently-scheduled
chains racing against the same replay clock. At the instant the detection
callback for stamp `t` runs, tf2 frequently has only the localization sample
*before* `t`; the sample *after* `t` (needed for tf2's own linear
interpolation) has not been received yet. tf2 correctly raises
`ExtrapolationException` ("... extrapolation into the future ...") for this
-- refusing to extrapolate is safety-correct -- but the pre-existing code
treated that exactly like every other TF failure: log a warning and drop the
frame **permanently**, even though the missing sample typically arrives
within tens of milliseconds.

## The fix: hold, don't extrapolate

New module `ab3dmot_tf_deferred_queue.py` (`DeferredDetectionQueue`, pure,
no `rclpy`/`tf2_ros` import -- unit-testable with plain fakes, mirroring
`ab3dmot_ros.py`'s injection pattern). `ab3dmot_tracker_node.py` wires it in:

- **Classification, not blanket deferral.** Every `can_transform(...,
  return_debug_tuple=True)` failure is classified
  (`classify_transform_availability`): only a debug message containing
  `"extrapolation into the future"` is `FUTURE_DATA_PENDING` (transient --
  the sample will arrive); everything else (unknown/disconnected frame,
  `"extrapolation into the past"` -- the earlier sample has already been
  evicted from tf2's fixed-size buffer, or any other reason) is
  `PERMANENT_FAILURE` and fails fast exactly like the pre-existing
  unconditional-drop behaviour. Measured directly against a real
  `tf2_ros.Buffer` on this repo's ROS Humble install (see the module
  docstring for the exact three message strings).
- **Never extrapolates, never fabricates.** The queue only ever calls
  `can_transform` (a non-blocking availability check) itself; it never
  returns a transform and never calls `lookup_transform`. The node performs
  the actual (unchanged) `lookup_transform` call once the queue reports
  "ready", so tf2's own interpolation reconstructs the pose **at the
  original stamp `t`** using both the before- and after- samples. The
  tracker never consumes a "future" sample's pose directly -- it is only
  used, after the fact, to interpolate a transform *at* the earlier,
  already-past detection stamp.
- **FIFO, order-preserving.** A detection is released only after every
  detection ahead of it in the queue has itself been released (processed or
  dropped). A later detection whose own transform happens to already be
  ready may **never** overtake an earlier one still waiting -- it is
  enqueued behind it unconditionally. `_last_admitted_stamp_ns` (the
  duplicate/rollback reference point `classify_timestamp` compares against)
  now advances at **admission** time (as soon as a detection is accepted
  into the pipeline, whether processed immediately or deferred), not at
  final-processing time -- otherwise a duplicate of a still-pending stamp
  could be silently admitted a second time while the first copy was still
  parked in the queue.
- **Exactly-once processing.** `offer()` either processes a detection
  immediately (queue was empty, transform ready -- never enqueued at all)
  or enqueues it for exactly one later release via `drain()`. Both paths
  converge on the single existing `_process_detection()` implementation (no
  copy/paste tracker-update branch); `drain()` never re-releases an entry it
  has already released.
- **Bounded.** `max_pending_detections` (FIFO depth; oldest unresolved entry
  dropped deterministically on overflow, counted as `dropped_queue_overflow`
  -- never a random entry) and `max_tf_wait_ms` (a detection stuck in
  `FUTURE_DATA_PENDING` beyond this wall/steady-clock duration is dropped as
  `dropped_tf_timeout`, and a subsequent detection's callback discovers and
  clears an already-expired backlog entry the first time it runs, so a long
  localization outage never later floods the tracker with a burst of stale
  frames once it resumes).
- **Non-blocking, single-threaded-executor-safe.** No `sleep`, no blocking
  `waitForTransform`. `_on_detected_objects` calls `_drain_deferred_queue()`
  at the top of every callback (event-driven: any new detection arriving is
  an opportunity to release whatever became resolvable since the last
  callback) -- this pipeline's detections arrive continuously (~4 Hz) even
  during a slow replay, so no periodic ROS-time timer is used. A ROS-time
  timer would not fire while `/clock` is paused (`start_paused:=true`); an
  event-driven drain has no such dependency and needed no such timer.
- **Source timestamp never modified.** The queue stores and returns the
  detection's own original `header.stamp`; `_process_detection` always
  drives `AB3DMOTTracker.step()` and the published `TrackedObjects.header`
  with that same original stamp, regardless of how long it sat in the
  queue. Prediction downstream sees the identical source timeline as
  before -- only the *processing* of an already-admitted detection is
  deferred, never its logical timestamp.

## Parameters

| parameter | default | meaning |
| --- | --- | --- |
| `defer_until_tf_ready` | `true` | Opt-out reproduces the pre-fix behaviour byte-for-byte: any TF failure (including a future-pending one) drops the frame immediately, no queue is even constructed. |
| `max_tf_wait_ms` | `500` | Bounded wall/steady-clock wait before a `FUTURE_DATA_PENDING` detection is dropped as `tf_timeout`. See "Measured TF-availability lag" below for the evidence behind this default -- it is a conservative engineering margin, not a scientifically derived bound, and stays fully configurable per launch/bag. |
| `max_pending_detections` | `8` | Bounded FIFO depth. At ~4.2 Hz detection input and a 500 ms wait ceiling, at most ~2 detections would realistically ever be pending at once on this recording; 8 leaves generous headroom without allowing unbounded growth. |

All three are threaded through `ab3dmot_tracker.launch.py` and, as
`ab3dmot_defer_until_tf_ready` / `ab3dmot_max_tf_wait_ms` /
`ab3dmot_max_pending_detections`, through `lidar_perception.launch.py`,
`lidar_bag_replay.launch.py`, and `training_free_perception_rviz.launch.py`
(`tracker_backend:=ab3dmot` path only; inert otherwise).

## Measured TF-availability lag (Phase 2)

From the BEFORE run's (`defer_until_tf_ready:=false`, byte-identical to the
pre-fix behaviour) own `ExtrapolationException` messages ("Requested time X
but the latest data is at time Y", both in bag/source time; n=730 rejections
over a ~92.5 s source-time / 185 s wall-clock window, `rate:=0.5`,
`enable_localization:=true`), the deficit `X - Y` at the moment of rejection
is a direct measurement of how far behind the localization TF was, in
source time, when a since-permanently-lost detection was rejected:

| percentile | source-time gap | wall-clock equivalent (`gap / rate`, `rate=0.5`) |
| --- | --- | --- |
| p50 | 11.1 ms | 22.2 ms |
| p90 | 32.8 ms | 65.6 ms |
| p95 | 65.9 ms | 131.8 ms |
| p99 | 88.5 ms | 177.0 ms |
| max | 120.8 ms | 241.6 ms |

`max_tf_wait_ms:=500` was chosen with ~2x the observed maximum as margin,
not as a tightly-fitted scientific threshold; it stays fully overridable per
launch. (Converting to a wall-clock estimate divides by the replay rate,
since a replay running slower than real time -- `rate < 1` -- takes
proportionally *longer* wall-clock time for the same amount of missing
source/bag time to actually arrive.)

## Production default decision (Phase 37)

`defer_until_tf_ready` defaults to `true`. Justification:

- **Geometry is unchanged.** The transform actually applied to a deferred
  detection is `lookup_transform` at the *same* original stamp `t` it would
  have used if TF had already been ready -- interpolated from the same two
  real samples either way.
- **Source timestamps are unchanged.** A deferred detection's published
  `header.stamp` and the timestamp driven into `AB3DMOTTracker.step()` are
  identical to the immediate-processing case.
- **Safety is equal or stronger.** No extrapolation is ever introduced;
  every non-transient TF failure (unknown frame, evicted/past-side gap,
  disconnected tree) still fails exactly as fast as before. The only
  behavioural difference from the pre-fix code is that a detection whose
  transform genuinely becomes available a short time later is now used
  instead of discarded.
- **Memory and wait are bounded.** `max_pending_detections` and
  `max_tf_wait_ms` cap both dimensions explicitly; a long localization
  outage still degrades to dropped frames, never unbounded queue growth or
  a stale-frame burst on recovery.
- **Ordering is preserved.** No detection can ever reach the tracker out of
  its own source-timestamp order as a result of this change.

Since this only ever widens the set of frames the strictly-experimental
`ad_ab3dmot_tracker` node accepts, under identical geometry/timestamp/safety
guarantees, it is the default for this opt-in node. Production Autoware
tracking is untouched by this change.

## Live A/B replay validation (Phase 29-34)

Same bag, same launch, same window, same config on both arms -- only
`ab3dmot_defer_until_tf_ready` differs (`false` = byte-identical to the
pre-fix behaviour, `true` = this fix, the new default). `rate:=0.5`,
`loop:=false`, `enable_localization:=true`, `enable_camera:=false`,
`enable_drivable_mask:=false`, 185 s wall-clock counting window (after a 20 s
startup buffer), source-time span ~92.5 s (matches the "approximately 90 s
source time" target). Process hygiene: exact install-path-pattern kill +
`ros2 daemon` restart before each run, verified clean via `ps aux` before
launch.

| metric | BEFORE (`defer_until_tf_ready:=false`) | AFTER (`:=true`, default) |
| --- | --- | --- |
| detected messages (total / non-empty) | 786 / 756 | 786 / 756 (identical -- detection stream is unaffected by this change) |
| tracked messages (total / non-empty) | 107 / 82 | 785 / 762 |
| predicted messages (total / non-empty) | 107 / 82 | 785 / 762 |
| tracked throughput (wall) | 107/185 s = 0.578 Hz | 785/185 s = 4.243 Hz |
| detected throughput (wall) | 786/185 s = 4.249 Hz | 786/185 s = 4.249 Hz |
| **acceptance ratio (tracked/detected)** | **13.6 %** | **99.9 %** |
| prediction throughput | tracks 1:1 with tracked (as before/after both show) | tracks 1:1 with tracked |
| `TF unavailable` rejections logged | 730 | 0 |
| deferred-queue drops (`tf_timeout`/`permanent`/`overflow`) | n/a (queue not constructed) | 0 / 0 / 0 (final periodic summary, frames=720: `deferred_tf_dropped_timeout=0 deferred_tf_dropped_permanent=0 deferred_tf_dropped_overflow=0`) |
| processed immediately vs. deferred | n/a | 110 / 610 (frames=720: 15.3 % immediate, 84.7 % needed deferral -- confirms the race is frequent on this recording, not a rare edge case) |
| max observed deferred-queue depth | n/a | 2 (`max_pending_detections:=8` has ample headroom) |
| errors / exceptions / crashes | 0 | 0 |
| tracks created / deleted (cumulative, last periodic summary) | n/a (never reached one 180-frame summary interval -- itself a symptom of the pre-fix throughput problem) | 721 / 703 (frames=720; `live_tracks=18` at that point) -- ordinary continuous birth/death activity, not a stall or runaway growth |

**Prediction and Dynamic Object Risk both track the AB3DMOT throughput
increase directly** (`PREDICTION_RUNTIME_SUMMARY frames=720
objects_in=5436 objects_out=5436 ... rejected=0`,
`DYNAMIC_OBJECT_RISK_RUNTIME_SUMMARY frames=720 published=720 rejected=0` --
both AFTER-run log lines): once AB3DMOT accepts a frame, every downstream
stage that was already correctly implemented consumes it with 0 rejections.

**Dynamic OGM (secondary, `enable_drivable_mask:=true`, Phase 34, this fix
enabled, shorter 100 s wall-clock window):**
`/ad/perception/occupancy/dynamic` published 428 grids, **232 non-empty
(54.2%)** -- regularly non-empty now that far more tracked/predicted frames
survive to reach it, confirming the throughput increase propagates all the
way to the road-gated occupancy layer. `DYNAMIC_OGM_RUNTIME_SUMMARY frames=180
predicted_objects=784 empty_grids=158 nonempty_grids=22
oversized_objects_skipped=111 ...` -- the periodic per-object
`oversized_objects_skipped` warnings ("inflated footprint exceeds
maximum_cells_per_object") are the same pre-existing, already-documented
covariance-driven skip behaviour from PR #9
(`docs/agent/STATUS.md` "Dynamic OGM robust to oversized predicted-object
uncertainty"), untouched by this change. **OGM code was not modified.**

**Interpretation.** Detection throughput is unchanged (identical 786/786,
756/756 non-empty on both arms -- confirms the fix touches only the
tracker's own admission logic, not anything upstream). Tracking and
prediction throughput moved from 13.6% to 99.9% of detection throughput --
materially closer to detection throughput, exactly the direction Phase 31
anticipated, without reaching a hardcoded target and without any residual
rejection reason remaining unexplained (both timeout and permanent-failure
counters are 0 for the full run). No detection frame was lost to the
deferred-processing mechanism itself failing; the handful of frames that
still do not reach a tracked output (786 detected vs. 785 tracked, one
frame) is attributable to the ordinary startup window before the TF tree
first connects (`ConnectivityException` "not part of the same tree",
observed only in the first ~1-2 s of the log, a permanent-at-that-instant,
correctly-fast-failing condition -- not something this fix defers).

**No tracking-accuracy claim is made anywhere in this document.** All
numbers above are message counts, throughput, and TF-classification
counters -- execution/runtime evidence, per this repo's established
convention (see AGENTS.md "execution success is not performance
validation").

## Limitations

- Single bag, single scene (`morai_cam4_20260813_163222`), single replay
  rate regime tested end-to-end; the measured lag distribution characterizes
  this recording's detection-vs-localization processing skew, not a general
  property of the pipeline.
- `max_tf_wait_ms`/`max_pending_detections` are conservative engineering
  defaults derived from one bag's observed lag, not a formally derived
  bound; a much slower localization backend or a much higher detection rate
  could need different values (both remain fully configurable).
- This does not address any *other* reason a detection frame might still be
  rejected (malformed stamp, permanently invalid frame, a genuinely
  disconnected TF tree at startup before localization publishes its first
  sample) -- those still fail exactly as before.
