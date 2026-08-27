# Tracking architecture audit: DetectedObjects → Autoware tracker → TrackedObjects → HEVEN IMM/prediction

Audited 2026-08-18. Scope: the current HEVEN tracking I/O contract, the
Autoware `multi_object_tracker` implementation actually built at
`~/projects/autoware_tracker_ws/src/autoware_universe/perception/
autoware_multi_object_tracker` (not vendored in this repo, but present on
this host and read directly rather than guessed), and the two pinned
reference repos under `references/`. No repository-wide scan was performed;
only the files cited below were read. **No implementation was done.**

> Branch note: this audit's branch (`research/tracking-audit`) does not
> currently contain the `references/` submodule commit
> (`chore: add tracking research references`, `4d26e0e`, present on
> `chore/tracking-references` / `origin/chore/tracking-references`). The
> three reference checkouts physically exist on disk at their pinned SHAs
> (verified against `references/README.md`'s pins) and were read directly;
> this doc doesn't depend on which branch eventually carries the submodule
> commit, but the two should be reconciled (merge/rebase) before further
> Tracking work lands here.

## 1. HEVEN tracking I/O

### Topics and message types

| stage | topic | message type |
|---|---|---|
| detector → tracker input | `/ad/perception/objects/detected` | `autoware_perception_msgs/msg/DetectedObjects` |
| tracker → HEVEN prediction input | `/ad/perception/objects/tracked` | `autoware_perception_msgs/msg/TrackedObjects` |
| HEVEN prediction output | `/ad/perception/objects/predicted` | `ad_interfaces/msg/PredictedObjectArray` |
| HEVEN prediction diagnostics | `/ad/perception/objects/prediction_debug` | `diagnostic_msgs/msg/DiagnosticArray` |

Source: `ad_lidar_perception/launch/tracking.launch.py`
(`input/detection01/objects` → `/ad/perception/objects/detected`, channel
`lidar_clustering`, `output/objects` → `/ad/perception/objects/tracked`),
`ad_lidar_perception/src/tracking/autoware_prediction_node.cpp` (default
parameter values for the prediction node's own topics).

### Frames

- Detector output (`DetectedObjects`) is in the sensor/vehicle-local frame
  the detector was configured for — `lidar_link` for both Euclidean and
  CenterPoint per `docs/perception/centerpoint_ros_interface.md`.
- The Autoware tracker's `world_frame_id` parameter is `odom`
  (`ad_lidar_perception/config/tracking/autoware.yaml`), `ego_frame_id` is
  `base_link`. Internally it holds a `tf2_ros::Buffer` and transforms
  incoming detections through `base_link` into `odom` before
  associating/updating tracks (`multi_object_tracker_core.cpp`: TF buffer
  construction; tracked/tentative output headers are stamped
  `params.world_frame_id`; the optional merged-objects output is stamped
  `params.ego_frame_id` instead).
- `TrackedObjects` therefore arrive at HEVEN's prediction node in `odom`.
  `AutowarePredictionNode` enforces this: `config_.expected_frame_id` is
  hardcoded to `"odom"` and any other frame is rejected
  (`autoware_prediction_node.cpp::adapt_tracked_objects` /
  `Impl::adapt_with_diagnostics`, "tracked objects are not in the odom
  frame").
- `PredictedObjectArray` output re-uses the same header (frame `odom`) as
  its `TrackedObjects` input.

### Timestamp handling

- The tracker publishes at a fixed `publish_rate` (10.0 Hz,
  `autoware.yaml`); `enable_delay_compensation` is `false`, so it does not
  extrapolate to a future publish time — the published stamp reflects the
  last processed measurement time (`multi_object_tracker_node.cpp::publish`
  uses `state_.last_tracker_time`).
- HEVEN's `AutowarePredictionNode` treats the tracker's output stamp as
  authoritative and gates on it strictly:
  - frame must be `odom` (above);
  - stamp must be strictly positive and well-formed
    (`stamp_to_ns` throws on negative/overflowed fields);
  - stamp must be **strictly newer** than the last successfully processed
    stamp (rejects non-monotonic/duplicate input);
  - stamp must not be in the future relative to `now()`;
  - input is rejected as **stale** if `now() - stamp > maximum_input_age_sec`
    (default 0.5 s, `autoware_prediction_node.cpp`).
- A **clock-rollback** special case exists for MORAI: if a new stamp is
  *older* than the last successfully processed one, the node does not just
  reject it — it resets the entire per-track IMM history
  (`ImmUpdateReason::kClockRollback`) because "MORAI resets simulated time
  together with object tracks" (comment in
  `AutowarePredictionNode::on_tracked_objects`).
- IMM per-track state additionally has its own **retention window**
  (`imm.track_retention_sec`, default 1.0 s): a track absent from the input
  for longer than that is dropped from IMM history and re-initialized
  (`ImmUpdateReason::kRetentionExpired`) rather than silently continuing
  with stale state.
- Rejections are never silent: every failure path publishes a
  `diagnostic_msgs/DiagnosticArray` entry categorizing the rejection reason
  (`rejection_reason()` maps exception messages to
  `rejected_frame_gate` / `rejected_monotonic_stamp_gate` /
  `rejected_future_stamp_gate` / `rejected_stale_array_gate` /
  `rejected_imm_update` / `rejected_object_validation`).

### QoS

| link | QoS |
|---|---|
| detector → tracker input | `rclcpp::QoS{1}` (tracker side, `multi_object_tracker_node.cpp`) — reliable, volatile by rclcpp default depth-constructor semantics |
| tracker → tracked output | `rclcpp::QoS{1}` (tracker side, same file) |
| tracked input → HEVEN prediction | `rclcpp::QoS(KeepLast(1)).reliable().durability_volatile()` (`AutowarePredictionNode` constructor) |
| HEVEN prediction output | `prediction_output_qos()` = `KeepLast(1)`, reliable, volatile |
| HEVEN prediction diagnostics | `KeepLast(10)`, reliable, volatile |

No best-effort/sensor-data QoS appears anywhere in this chain (unlike the
raw point cloud path) — everything from `DetectedObjects` onward is
reliable/volatile, depth 1 except diagnostics.

## 2. Current Autoware tracker (`autoware_multi_object_tracker`, pinned via `autoware_universe` commit `d4d260983d357e1b2b34291d91933f9f4b53bf94`, package version 0.51.0 per `docs/perception/centerpoint_ros_interface.md`)

Read directly from
`~/projects/autoware_tracker_ws/src/autoware_universe/perception/autoware_multi_object_tracker`
(built separately per HEVEN's existing overlay process; not vendored here).

### Association

- Default associator is **BEV** (`bev_association.cpp`); a **polar**
  associator also exists (`polar_association.hpp`) but is only selected
  per-input-channel via `association.<channel>.associator_type`, not used
  by HEVEN's single `lidar_clustering` channel.
- Gating is spatial first: each tracker is inserted into an R-tree
  (`boost::geometry::index`) keyed by BEV position; each measurement only
  queries trackers within its class's configured `max_dist` (squared)
  bounding box (`BevAssociation::processMeasurement`).
- Cost/score (`calculateBevAssignmentScore`,
  `lib/association/scoring/bev_assignment_scoring.cpp`):
  - **unknown ↔ unknown**: pure 2D generalized IoU (GIoU), gated by
    `unknown_association_giou_threshold`, rescaled to `[0, 1]`.
  - otherwise: squared-distance gate (`max_dist_sq`) → for non-vehicle
    tracker types, an area gate (`min_area`/`max_area`) and a **Mahalanobis
    distance** gate using the tracked object's inverse position covariance
    (empirical threshold `11.62`, "99.6% confidence, chi-square 2 DOF") →
    an IoU/GIoU score (1D IoU for pedestrian trackers, 3D GIoU if both
    sides "trust extension", else 2D GIoU), gated by `min_iou`, rescaled to
    `[0, 1]`.
  - A `has_significant_shape_change` flag is raised when a vehicle
    tracker's matched IoU is below `CHECK_GIOU_THRESHOLD` and its BEV area
    changed by more than `AREA_RATIO_THRESHOLD` — used later to allow a
    stronger "conditioned" pose update (see State estimation).
- Assignment solver: **not Hungarian** — `gnn_solver::MuSSP`
  (`BevAssociation::BevAssociation` constructs
  `std::make_unique<gnn_solver::MuSSP>()`), a min-cost-flow "successive
  shortest paths" solver (`association/solver/mu_ssp.hpp`,
  `gnn_solver.hpp`; the `muSSP` dependency HEVEN already pins per
  `docs/perception/centerpoint_ros_interface.md`'s tracker dependency
  closure). A separate score threshold (`0.01`) discards any assigned pair
  below it after solving (`BevAssociation::assign`).

### State estimation

- Per-class tracker type mapping (`ad_lidar_perception/config/tracking/
  autoware.yaml` → `params_.tracker_type_map`, consumed in
  `TrackerProcessor::createNewTracker`, `processor.cpp`):
  - `car`/`truck`/`bus`/`trailer` → `multi_vehicle_tracker`
    (`MultipleVehicleTracker`)
  - `pedestrian`/`bicycle`/`motorcycle` → `pedestrian_and_bicycle_tracker`
    (`PedestrianAndBicycleTracker`)
  - (other selectable-but-unused-by-HEVEN types exist:
    `general_vehicle`/`normal_vehicle`/`bicycle`/`big_vehicle` single
    `VehicleTracker` variants, `PolygonTracker`, `PassThroughTracker`.)
- `VehicleTracker` (used, via `MultipleVehicleTracker`, for car/truck/
  bus/trailer) wraps a **`BicycleMotionModel`**, a 6-state model
  (`X1,Y1,X2,Y2,U,V` — rear-wheel position, front-wheel position,
  longitudinal velocity, lateral velocity) with a nonlinear
  `predictStateStep(dt, KalmanFilter & ekf)` — i.e. this is an **EKF**, not
  a linear KF (`bicycle_motion_model.hpp`). Process noise is parameterized
  per-axis (longitudinal/lateral accel std, yaw-rate std, slip-rate std,
  slip-angle limit) rather than a single scalar `Q`.
- `PedestrianTracker` wraps a **`CTRVMotionModel`**, a 5-state model
  (`X,Y,YAW,VEL,WZ` — coordinated-turn rate model), also nonlinear
  (`ctrv_motion_model.hpp`) — EKF as well.
- `PedestrianAndBicycleTracker` (used for HEVEN's pedestrian/bicycle/
  motorcycle classes) is **not** a single motion model — it privately holds
  *both* a `PedestrianTracker` (CTRV) and a `VehicleTracker` constructed
  with the `bicycle` object model, and its own `predict`/`measure`/
  `getTrackedObject` delegate to whichever is currently authoritative
  (`pedestrian_and_bicycle_tracker.hpp`). The exact model-selection/
  hand-off logic between the two internal sub-trackers was **not traced
  further** in this pass (its `.cpp` was not read) — flagged here rather
  than guessed.
- `PolygonTracker`/`PassThroughTracker` (unknown-class fallback, not
  reached by HEVEN's configured labels) were not read in detail — out of
  scope for the car/pedestrian-class path this task asked about.
- Conditioned/"weak" updates: `VehicleTracker` supports
  `FRONT_WHEEL_UPDATE` / `REAR_WHEEL_UPDATE` / `WEAK_UPDATE` strategies
  (`vehicle_tracker.hpp`) — used when `has_significant_shape_change` is
  true, to avoid corrupting the state with a measurement whose extent
  jumped; the exact selection rule was not traced further here.

### Lifecycle

Entirely inside `Tracker` (base class,
`lib/tracker/model/tracker_base.cpp`) — **not** a simple hit/miss counter
like AB3DMOT/SimpleTrack, but a per-channel **existence-probability**
model plus **covariance-size** checks:

- **Spawn**: `TrackerProcessor::spawn` creates a new tracker for every
  measurement the associator left unmatched, provided the input channel's
  `is_spawn_enabled` flag is set (`processor.cpp`); initial existence
  probability comes from the measurement's own `existence_probability` if
  the channel is configured to trust it, else a fixed default.
- **Update**: matched trackers get `updateWithMeasurement` (resets
  `no_measurement_count_`, increments `total_measurement_count_`, updates
  per-channel and total existence probability via `updateProbability`);
  unmatched trackers get `updateWithoutMeasurement` (increments
  `no_measurement_count_`/`total_no_measurement_count_`, **decays** all
  existence probabilities by elapsed time via `decayProbability`).
- **Confident (i.e., published) test** (`Tracker::isConfident`,
  `tracker_base.cpp:538`): requires `total_measurement_count_ >= 2`, then
  either (a) predicted position covariance's major eigenvalue is below a
  fixed `0.28` threshold, or (b) total existence probability `> 0.50` *and*
  the covariance is below an **adaptive** threshold that grows with the
  tracker's BEV area and distance from ego
  (`computeAdaptiveThreshold`, base `1.6`, fallback `2.6` when no ego pose
  is available). Unconfident tracks are still tracked internally and
  optionally published as "tentative" objects
  (`publish_tentative_objects`, default `false` in HEVEN's config).
- **Expiry** (`Tracker::isExpired`, `tracker_base.cpp:580`): true if (a)
  elapsed time since last update exceeds `1.0 s`, or (b) total existence
  probability drops below `0.015`, or (c) elapsed time exceeds `0.18 s`
  *and* existence probability is below `0.3` *and* the position covariance
  (major or minor eigenvalue, both via the same adaptive-threshold
  mechanism) exceeds its threshold.
- **Pruning/merge**: `TrackerProcessor::prune` (throttled to run at most
  every 2 ms) removes expired trackers, then runs
  `TrackerOverlapManager::merge`, which removes/merges trackers whose
  bounding boxes overlap beyond class-specific IoU/distance thresholds
  (`min_known_object_removal_iou`, `min_unknown_object_removal_iou`,
  `pruning_generalized_iou_thresholds`, `pruning_distance_thresholds` — all
  present in HEVEN's `autoware.yaml`); this file's internals were not
  traced further here.

### Class-specific tracker behavior

Summarized above (per-class tracker-type map, distinct motion models per
class, and per-class association thresholds
`association.max_dist/max_area/min_area/min_iou.<label>.<tracker_type>` +
per-class pruning thresholds — all declared as ROS parameters in
`multi_object_tracker_node.cpp` and populated for HEVEN in
`ad_lidar_perception/config/tracking/autoware.yaml`, though that file was
already read in a prior session and is not re-quoted in full here).

## 3. AB3DMOT (`references/ab3dmot`, pinned `61f3bd72574093e367916c757b4747ca445f978c`)

Read: `AB3DMOT_libs/kalman_filter.py`, `AB3DMOT_libs/matching.py`,
`AB3DMOT_libs/model.py` (partial — birth/update/output/orientation
sections).

- **State vector** (`kalman_filter.py::KF`): 10-dim
  `[x, y, z, theta, l, w, h, dx, dy, dz]` — constant-velocity model over
  position only; size and orientation are carried but not differentiated.
- **Measurement vector**: 7-dim `[x, y, z, theta, l, w, h]`; `H` selects
  the first 7 state dimensions unchanged.
- **KF prediction/update**: a **linear** `filterpy.kalman.KalmanFilter`
  (`dim_x=10, dim_z=7`) — explicitly commented "no need to use EKF here as
  the measurement and state are in the same space with linear
  relationship". `F` is the identity plus a `dt=1` (implicit, one frame per
  step) constant-velocity block for `x,y,z`. Defaults: `P[7:,7:] *= 1000`
  (very uncertain initial velocity), `P *= 10` overall, `Q[7:,7:] *= 0.01`
  (fairly certain constant-velocity process model), `R` left at
  `filterpy`'s identity default (a commented-out `R *= 10.` line exists but
  is inactive).
- **Association cost** (`matching.py::compute_affinity` +
  `dist_metrics.py`, not fully read but referenced): selectable per run —
  IoU/GIoU (2D or 3D), Euclidean/Mahalanobis distance (`m_dis`, using the
  KF's own innovation covariance `H P Hᵀ + R`), or ground-plane distance.
  **Per-class, per-dataset** algorithm/metric/threshold/`min_hits`/
  `max_age` tuples are hardcoded in `model.py` (e.g. KITTI 3D: Car →
  Hungarian + `giou_3d` + threshold `-0.2` + `min_hits=3` + `max_age=2`;
  Pedestrian → greedy + `giou_3d` + `-0.4` + `min_hits=1` + `max_age=4`).
- **Hungarian assignment**: `scipy.optimize.linear_sum_assignment` on the
  *negated* affinity matrix (cost = `-affinity`) when `algm == 'hungar'`;
  a hand-rolled **greedy** matcher (sort all pairwise costs ascending,
  first-come-first-served, ties broken by sort order) is used for classes
  configured with `algm == 'greedy'` — this is the same "sort-and-greedily-
  claim" pattern SimpleTrack also implements independently.
- **Yaw handling** (`model.py::orientation_correction`, called before every
  KF update): wraps both the propagated-track and the incoming
  measurement's `theta` into `[-π, π]`
  (`within_range`), then if their difference is obtuse (between 90° and
  270°) rotates the *track's* angle by π and re-wraps, and if still ≥270°
  apart, adds/subtracts a full 2π turn — i.e. it resolves KITTI's
  front/back bounding-box ambiguity by snapping the filter's prior heading
  to the closest of the two 180°-symmetric options before filtering, not
  by changing the KF's own angular process/measurement model.
- **Lifecycle**: birth on every unmatched detection (new `KF` instance,
  monotonically increasing integer ID); each processed frame calls
  `kf.predict()` for every existing tracker (incrementing
  `time_since_update`) then `kf.update()` for matched ones (resetting
  `time_since_update`, incrementing `hits`); `output()` publishes a track
  iff `time_since_update < max_age` **and** (`hits >= min_hits` **or**
  `frame_count <= min_hits`) — the latter clause lets tracks publish
  immediately during the first `min_hits` frames of the whole sequence,
  matching an offline/batch KITTI-evaluation assumption more than a
  continuously-running online service; tracks with `time_since_update >=
  max_age` are deleted from the list entirely.

## 4. SimpleTrack decomposition (`references/simpletrack`, pinned `05c96bb7ed98fc179856f327544612a66c839b5e`)

Read: `mot_3d/mot.py`, `mot_3d/association.py`,
`mot_3d/motion_model/kalman_filter.py`, `mot_3d/life/hit_manager.py`,
`mot_3d/preprocessing/nms.py` + `bbox_coarse_hash.py` (headers),
`mot_3d/redundancy/redundancy.py` (header/dispatch only).

- **Detection preprocessing** (`mot_3d/preprocessing/`): a **coarse spatial
  hash** (`BBoxCoarseFilter`, splits the scene into a `grid_size` grid so
  boxes in non-overlapping cells can be skipped cheaply) feeding a
  score-sorted greedy **NMS** (`nms()`, thresholds on IoU-band
  `[threshold_low, threshold_high]` and yaw difference
  `threshold_yaw`) that also discards `weird_bbox` (non-positive
  length/width/height) detections. `mot.py::forward_step_trk` additionally
  applies a plain score threshold (`configs['running']['score_threshold']`)
  before this.
- **Association** (`mot_3d/association.py::associate_dets_to_tracks`):
  configurable metric (`iou`, `giou`, `m_dis` Mahalanobis using each
  track's own KF innovation matrix, or `euler` L2) and configurable solver
  — `bipartite` (Hungarian, `scipy.optimize.linear_sum_assignment` on a
  *distance* matrix, i.e. minimized directly rather than negated-affinity)
  or `greedy` (same sort-and-claim pattern as AB3DMOT, independently
  reimplemented). A single `dist_threshold` cutoff (not per-class) discards
  low-quality matches after solving.
- **Motion model** (`mot_3d/motion_model/kalman_filter.py`): explicitly
  commented "borrowed from AB3DMOT" — the **same** 10-state/7-measurement
  linear constant-velocity KF, `P[7:,7:] *= 1000` / `P *= 10` defaults
  (SimpleTrack's `Q`/`R` scaling lines are present but commented out,
  unlike AB3DMOT which leaves `Q[7:,7:]` active). Genuine difference from
  AB3DMOT: `get_prediction()` rebuilds `F` with the **actual elapsed
  `time_lag`** between frames in the velocity block, instead of AB3DMOT's
  implicit `dt=1`-per-frame assumption — relevant for variable-rate or
  frame-dropped input. Yaw handling in `update()` performs the identical
  wrap-to-`[-π,π]` + closest-180°-flip correction as AB3DMOT's
  `orientation_correction`, just inlined rather than factored out.
- **Lifecycle** (`mot_3d/life/hit_manager.py::HitManager`): an explicit
  finite-state machine — `birth → alive → dead` (a `no_asso` flag exists in
  the state model/docstring for a fourth "about to die" state, but
  `valid_output` only ever checks `state == 'alive' and not no_asso`, and
  `no_asso` is never set to `True` anywhere in this file, so it is
  effectively dead code in the reviewed version — flagged rather than
  described as active). Birth transitions to `alive` once `hits >=
  min_hits` (or unconditionally within the first `min_hits` frames of the
  whole run, same early-sequence carve-out as AB3DMOT); an `alive` track
  transitions to `dead` once `time_since_update >= max_age`. `mot.py`
  additionally runs a **`RedundancyModule`** for tracks that failed primary
  association: rather than coasting on the motion-model prediction alone,
  it re-associates that specific unmatched track against *lower-score*
  detections that didn't clear the main `score_threshold`
  (`motion_model_redundancy`, gated by its own
  `redundancy.det_score_threshold`/`det_dist_threshold`) — a second-chance
  association pass this audit did not trace to its exact matching call in
  full, but whose dispatch and intent (recover tracks that a stricter
  detector confidence gate would otherwise coast/kill) are clear from
  `RedundancyModule.infer`/`motion_model_redundancy`'s opening logic.

## 5. Where KF, EKF, IMM, and KalmanNet could later be compared

All four are candidate **motion-model/state-estimation** components — they
would slot into the same place in the pipeline (the per-tracker
predict/update step), not change the surrounding association or lifecycle
machinery by themselves:

- **KF (linear)**: matches AB3DMOT's and SimpleTrack's existing 10-state
  constant-velocity model directly — the smallest possible comparison
  point, since both reference repos already implement it and it requires
  no nonlinear Jacobian work.
- **EKF (nonlinear)**: what Autoware already runs today (bicycle model for
  vehicles, CTRV for pedestrians) — any KF/EKF comparison should include
  Autoware's own EKF as the current-production baseline, not just
  AB3DMOT's linear KF.
- **IMM**: HEVEN already runs one, but **only in the downstream prediction
  node** (`ad_autoware_prediction`'s 3-model stationary/CV/coordinated-turn
  IMM, see §6) — it is not currently part of tracking state estimation
  (the Autoware tracker) at all. A KF/EKF/IMM comparison *within tracking*
  would mean moving (or duplicating) an IMM-style multi-model estimator
  into the state-estimation stage — a materially different, larger change
  than swapping a single motion model, and one AGENTS.md's "Existing
  IMM/prediction... remain unchanged unless explicitly requested" already
  flags as requiring explicit go-ahead.
- **KalmanNet**: a learned filter that replaces the KF gain computation
  with a small neural network while keeping the same predict/update
  control flow — architecturally, the natural integration point is again
  the single-tracker predict/update step (same seam as swapping AB3DMOT's
  `filterpy` KF for something else), but it would require its own
  train/inference pipeline analogous to CenterPoint's (real GPU training,
  bounded smoke run, no accuracy claims without real val/test data) rather
  than a pure config change like KF→EKF.
- A fair comparison harness would need to hold association and lifecycle
  fixed (so only the motion model varies — "change one major experimental
  variable at a time" per AGENTS.md) and would need per-class results,
  since Autoware, AB3DMOT, and SimpleTrack all already treat vehicle vs.
  pedestrian motion differently.

## 6. Tracking state estimation vs. future trajectory prediction

These are two distinct pipeline stages in HEVEN today, and this audit's
scope (§§2-5) only covers the **tracking** half:

| | tracking (state estimation) | prediction (future trajectory) |
|---|---|---|
| HEVEN component | Autoware `multi_object_tracker` (external, built separately) | `ad_autoware_prediction` node, `ad_lidar_perception/src/tracking/autoware_prediction_node.cpp` (HEVEN's own code) |
| input | `DetectedObjects` (per-frame, current instant) | `TrackedObjects` (the tracker's current-instant state estimate) |
| output | `TrackedObjects`: one filtered pose/twist **at the current timestamp**, with covariance | `PredictedObjectArray`: **multiple future timestamps** per object (`horizons_s`, default `{0.5, 1.0}` in the adapter, up to `{0.5..6.0}` in `ImmConfig`'s own default), each a projected pose |
| "predict" step meaning | one filter time-update to the *next incoming measurement's* timestamp (sub-second, data-driven cadence) | an explicit multi-second-ahead rollout from the *current fused state*, independent of when the next measurement arrives |
| model | Autoware: EKF (bicycle/CTRV per class, §2). AB3DMOT/SimpleTrack: linear KF, constant-velocity (§§3-4) | HEVEN: 3-model IMM (stationary / constant-velocity / coordinated-turn, Markov-switching, `imm_predictor.hpp`) |
| role of AB3DMOT/SimpleTrack in this comparison | **both are tracking-stage systems** — their own "predict" methods are the same single-step time-update as Autoware's, not multi-horizon prediction | neither reference repo implements anything resembling HEVEN's downstream prediction stage |

AGENTS.md's research order lists "Tracking → Association → KF/EKF/IMM/
KalmanNet → Prediction" as separate stages for exactly this reason: KF/
EKF/IMM/KalmanNet experiments (§5) are about the **tracking** stage's
motion model, and are logically prior to (and separate from) any change to
the **prediction** stage's own IMM, which AGENTS.md separately protects
("Existing IMM/prediction... remain unchanged unless explicitly
requested").

## 7. Proposed smallest HEVEN-compatible AB3DMOT adapter (design only, not implemented)

Goal: let AB3DMOT's tracking loop run against real HEVEN `DetectedObjects`
input, opt-in, without touching the production Autoware tracker path, so it
can eventually be compared against Autoware's tracker under §5's
"comparison harness" idea. Sketch only — every open question below must be
resolved (per AGENTS.md, by reading further source / checking real data,
not by guessing) before any code is written:

1. **New, separate ROS2 node** (e.g. `ad_ab3dmot_tracker`, analogous to how
   `ad_centerpoint_detector` sits opt-in alongside the default Euclidean
   detector), not a modification of `tracking.launch.py`'s existing
   Autoware path. Subscribes the same `/ad/perception/objects/detected`
   contract the Autoware tracker already consumes, so it can run **in
   parallel** off the same detector output for comparison, rather than
   replacing it.
2. **Message adapter, `DetectedObjects` → AB3DMOT's 7-dim box array**
   `[x, y, z, theta, l, w, h]`: geometric center and dimensions map
   directly (same convention Euclidean/CenterPoint already use, per
   `docs/perception/centerpoint_ros_interface.md`), but **the yaw/theta
   convention needs explicit verification, not assumption** — AB3DMOT was
   built for KITTI's camera-frame convention; HEVEN's `lidar_link` is
   +x forward/+y left/+z up with CCW yaw about +z (already verified
   elsewhere in this repo for the CenterPoint path). Whether AB3DMOT's
   `theta` axis/sign matches this without transformation is an **open
   question to verify by reading `AB3DMOT_libs/box.py`'s `Box3D` class**
   before writing the adapter, not to guess.
3. **Per-tracker instantiation policy**: AB3DMOT's `min_hits`/`max_age`/
   association metric/algorithm are hardcoded per `(dataset, class)` pair
   in `model.py` for KITTI/nuScenes categories that don't exist in MORAI's
   `vehicle`/`pedestrian`/`obstacle` label set (§ from
   `docs/research/centerpoint_status.md`). These would need HEVEN-specific
   values chosen deliberately (config, not hardcoded per AGENTS.md
   "Algorithm code and experiment config must remain separate") — not
   reused from KITTI defaults, since AGENTS.md forbids inventing
   unexplained association/gating values.
4. **Frame/TF responsibility**: AB3DMOT itself has no TF/ego-motion
   awareness beyond its own KITTI-`oxts`-file-based
   `ego_motion_compensation` helper, which doesn't apply to a live ROS
   system. The adapter would need its own explicit decision — track in the
   detector's native `lidar_link` frame (simplest, but not directly
   comparable to Autoware's `odom`-frame `TrackedObjects` without an extra
   transform step at comparison time) vs. transform detections into `odom`
   before feeding AB3DMOT (mirrors Autoware's behavior, reuses the existing
   TF buffer pattern, but is new code this adapter would own) — a decision
   to make explicitly, not silently default.
5. **Output message shape**: AB3DMOT's native output (7-dim box + ID +
   `info` passthrough) does not carry a covariance in the
   `autoware_perception_msgs/TrackedObjects` sense (position/twist
   covariance matrices) — republishing as real `TrackedObjects` would mean
   either inventing a covariance from the KF's own `P` (defensible, since
   the KF does track one, unlike AB3DMOT's plain point-estimate `info`
   passthrough) or publishing a HEVEN-specific comparison-only message type
   instead, similar in spirit to the existing
   `heven.ros_detection_comparison.v1` JSONL format used for the
   Euclidean/CenterPoint comparison tool
   (`docs/perception/centerpoint_vs_euclidean_comparison.md`) but for
   tracks instead of detections.
6. **Reuse boundary**: per AGENTS.md ("Prefer adapters outside submodules"),
   the adapter node would import/wrap `AB3DMOT_libs.kalman_filter.KF` and
   `AB3DMOT_libs.matching` directly from the pinned submodule (no forking
   or copying its association/motion-model code into HEVEN), with only the
   ROS message conversion, config loading, and topic wiring living in
   HEVEN's own tree — mirroring exactly how `openpcdet_runtime.py` imports
   OpenPCDet's `CenterPoint` class today rather than reimplementing it.

This is the smallest adapter that would let AB3DMOT's actual tracking loop
run on real HEVEN detections; it is *not* a recommendation to build it yet,
and items 2-5 above are explicitly open questions requiring verification
against source/data before any code is written.

## AB3DMOT Integration Decisions

Audited 2026-08-18, same session as this file's original content. Resolves
the open questions from §7 by reading `references/ab3dmot` source directly
(and, for §3 below, executing its own documented formula numerically as a
verification step — not guessing). **Still no code written.**

### 1. Yaw convention — RESOLVED

**Finding: AB3DMOT's `theta`/`ry` is a KITTI camera-frame angle, not a
z-up yaw. It is not identical to HEVEN's `lidar_link`/`odom` yaw, and the
non-identity extends beyond the state's `theta` field into AB3DMOT's own
geometry (corner/IoU) functions.**

- `AB3DMOT_libs/box.py::box2corners3d_camcoord` docstring is explicit: "the
  box is in the camera coordinate with right x, down y, front z" and
  `ry = roty(bbox.ry)` is "rotation around the yaw axis" implemented as
  `kitti_oxts.py::roty` — a rotation **about the Y axis**, which in this
  frame points **down**. HEVEN's `lidar_link` is +x forward, +y left, +z
  up, with yaw defined as CCW rotation **about +z** (verified for the
  CenterPoint path in `docs/perception/centerpoint_ros_interface.md`).
  Different rotation axis, different frame handedness convention — not the
  same angle, and not related by a trivial sign flip alone.
- The same function's corner assignment is also camera-frame-specific, not
  just the angle: `x_corners` is built from `l` (length), `z_corners` from
  `w` (width), and `y_corners` from `h` (height, along the camera's **down**
  axis, negated) — i.e., AB3DMOT's own IoU/corner geometry code
  (`box2corners3d_camcoord`, and anything in `dist_metrics.py` that calls
  it) assumes **height lives on the camera's Y axis**, not HEVEN's Z axis.
  Feeding it HEVEN `[x,y,z,yaw,l,w,h]` values verbatim would silently
  compute wrong 3D overlaps (it would treat HEVEN's up-axis extent as a
  camera-down displacement), not just a wrong angle.
- `AB3DMOT_libs/kitti_oxts.py` separately defines `rotz` (rotation about
  Z) alongside `roty`, and `AB3DMOT_libs/nuScenes_utils.py::create_nuScenes_box`
  builds a composite quaternion
  `Quaternion(axis=(0,1,0), angle=obj.ry) * Quaternion(axis=(1,0,0), angle=π/2)`
  to re-express a KITTI-camera-style `Object_3D` in the `nuscenes-devkit`
  `Box` container — confirming the codebase treats "camera-frame `ry`" as
  its one true internal representation across datasets, with adapters
  converting *into* it, not proof that any of those adapters produce a
  pure z-up yaw compatible with HEVEN.
- To numerically pin down what that composite formula actually does
  (rather than guess), it was executed directly (both a hand-written,
  self-verified Hamilton-quaternion implementation and, cross-checked, the
  real `pyquaternion` library installed in a throwaway venv — outputs
  matched exactly). Result: the box's local "up" axis maps to a **constant**
  direction in the output frame regardless of `ry`, which is inconsistent
  with a simple z-up ego frame unless further calibration transforms
  (`nuScenes_lidar2world`'s `cs_record`/`pose_record` rotations, which
  depend on `nuscenes-devkit` data not present in this pinned submodule)
  are also applied. In other words: **`create_nuScenes_box` converts a
  KITTI-camera-frame object into an intermediate, still-not-yet-ego-frame
  representation for later dataset-specific calibration chaining — it is
  not itself the "camera-ry → z-up-yaw" formula HEVEN would need**, and
  asserting a specific closed-form scalar mapping (e.g. "`ry = -yaw`" or
  similar) from this file alone would be guessing exactly what AGENTS.md
  and this task prohibit.
- **Recommendation (the actionable resolution)**: do not attempt to reuse
  AB3DMOT's angle/geometry functions in their camera-frame form at all.
  Treat AB3DMOT's `filterpy`-based `KF`/`matching.py` numerical core as
  **frame-agnostic** — it only ever treats `theta` as an abstract periodic
  scalar with wrap-around handling (`within_range`,
  `orientation_correction`), never as something tied to a specific physical
  axis — and feed it HEVEN's own z-up yaw directly, self-consistently
  (`theta_ab3dmot := HEVEN_yaw_rad`, no conversion applied, because none of
  the reused code cares which physical axis `theta` represents as long as
  it is used consistently). What must **not** be reused verbatim is
  anything that calls `box2corners3d_camcoord` for cost computation (e.g.
  IoU/GIoU association metrics from `dist_metrics.py`) — those must be
  replaced with a HEVEN-side z-up IoU/GIoU implementation (conceptually
  the same kind of overlap computation already used in
  `shapes::get3dGeneralizedIoU` on the Autoware side, just not literally
  reused across the license boundary) rather than AB3DMOT's camera-frame
  corner code. This keeps the adapter reference-faithful for the
  motion-model/lifecycle logic while avoiding a silently-wrong geometric
  cost function — an explicit, source-grounded design constraint, not a
  guess.

### 2. Tracking frame — RESOLVED

**Recommendation: transform detections to `odom` before tracking (mirror
Autoware's own approach), not track directly in `lidar_link`.**

Justification against each consideration:

- **Ego motion (primary, correctness reason)**: `lidar_link` is a
  body-fixed, non-inertial frame. AB3DMOT's (and SimpleTrack's) motion
  model is a constant-*position*-velocity model in whatever frame it's
  given — it has no ego-motion compensation term of its own beyond its
  KITTI-`oxts`-file-specific `ego_motion_compensation` helper (which reads
  offline calibration files, not applicable to a live ROS system, per §7
  item 4). If fed raw `lidar_link` detections, a *stationary* real-world
  object would appear to translate every time the ego vehicle moves,
  directly violating the constant-velocity assumption the KF relies on.
  Autoware's own tracker already transforms into `odom` for exactly this
  reason (§2 above, `multi_object_tracker_core.cpp`'s TF buffer +
  `world_frame_id=odom`) — this is existing, working precedent in this
  exact repo's dependency chain, not a novel idea.
- **Comparison with Autoware**: per AGENTS.md ("change one major
  experimental variable at a time"), comparing AB3DMOT against Autoware's
  tracker is only a fair comparison of *association/motion-model/lifecycle*
  if both consume the same-frame input. Tracking AB3DMOT in `lidar_link`
  while Autoware tracks in `odom` would confound the comparison with a
  frame difference neither algorithm chose.
- **Downstream HEVEN IMM requirement**: `AutowarePredictionNode`
  hard-rejects any `TrackedObjects` input whose `frame_id != "odom"`
  (`autoware_prediction_node.cpp`, verified in §1 above). If AB3DMOT's
  output is ever meant to be consumable — even experimentally — by that
  same node, `odom` is not just preferred, it is the only frame that node
  will accept without its own modification (out of scope here).
- **RViz comparison**: overlaying AB3DMOT tracks, Autoware tracks, and the
  raw point cloud is possible in either frame as long as the necessary TF
  is being broadcast (RViz can resolve cross-frame display via TF), but a
  single common frame avoids depending on TF-lookup timing/extrapolation
  matching between two independently-clocked ROS graphs during a live
  side-by-side comparison, and is simpler to reason about.
- **Cost of this recommendation**: the AB3DMOT adapter node must own its
  own `tf2_ros::Buffer` lookup from `lidar_link` (or whatever frame the
  active detector publishes in) into `odom`, mirroring the TF pattern
  already used elsewhere in `ad_lidar_perception` — new code this adapter
  would need, not a reused component, but a small and well-precedented one.

### 3. Initial AB3DMOT configuration — RESOLVED

Per "prefer the smallest reference-faithful baseline" and "do not tune for
performance," every choice below is either the most common value already
present across AB3DMOT's *own* per-class/per-dataset configs in
`AB3DMOT_libs/model.py`, or explicitly left uncertain/configurable where no
such defensible default exists — not a HEVEN-specific tuning pass.

- **Association metric**: **3D GIoU (`giou_3d`)**. It is the metric used
  by nearly every class/dataset combination in `model.py` (KITTI 2D, KITTI
  3D, and the nuScenes-style block all default to it) — the one clearly
  dominant, reference-faithful choice, not a HEVEN-specific pick.
- **Hungarian vs. greedy**: **greedy**, uniformly across classes for the
  initial baseline. AB3DMOT itself splits this per class/dataset (e.g.
  Hungarian for KITTI-2D "Car" only), but **greedy is the majority choice**
  across the full set of configs read (used for Pedestrian in every block,
  and for most other classes including "Car" in two of the three blocks).
  A single uniform algorithm is also the smaller configuration surface
  ("smallest baseline"); Hungarian remains available as a config option
  (per AGENTS.md "parameter variations use configs, not branches") for a
  later, explicitly-requested comparison — not implemented now.
- **Gating threshold (GIoU)**: **left configurable, no committed numeric
  default** — this is the one sub-item where the reference genuinely gives
  no single defensible value: AB3DMOT's own per-class GIoU thresholds span
  `-0.2` to `-0.8` and visibly track dataset/class-specific tuning, exactly
  what AGENTS.md warns against copying uncritically. Per this task's own
  instruction ("If a value cannot be justified from the reference or
  current runtime cadence, leave it configurable and document the
  uncertainty"): the config schema should mirror AB3DMOT's own per-class
  threshold structure, defaulting to `0.0` (the loosest defensible
  starting point — GIoU ≥ 0 already implies actual geometric overlap, so it
  rejects nothing AB3DMOT's own tuned values wouldn't already reject, while
  not asserting a borrowed dataset-specific number) until measured against
  real HEVEN detections.
- **`min_hits`**: **1**. The single most common value across every
  class/dataset combination read (Pedestrian uses it in all three blocks;
  most other classes use it in at least one) — also the simplest possible
  lifecycle policy (a track is immediately eligible for output on its
  first update rather than requiring a multi-frame probation window),
  matching "smallest reference-faithful baseline."
- **`max_age`**: **2** (frames, not seconds — flag below). The most common
  value across nearly all classes/datasets read (Car, Truck, Bus, Trailer,
  Motorcycle, Bicycle all use `2` in the blocks read; only Pedestrian uses
  a higher value, `4`, in two of the three). **Uncertainty flagged as
  instructed**: `max_age` in AB3DMOT counts *frames* since the last
  update, not a time duration — its effective real-world tolerance depends
  on the detector's actual publish cadence, which AB3DMOT's own defaults
  assume is roughly KITTI's/nuScenes' native rate. This repo's own bag
  metadata (`bags/static_20260805_003151/metadata.yaml`:
  `/ad/sensors/lidar/points` message_count `1778` over `242.322234383` s
  duration) measures HEVEN's actual LiDAR cadence at **≈7.34 Hz**, the same
  order of magnitude as KITTI's ~10 Hz, so carrying over `2` frames
  (≈0.27 s of tolerance at this rate) is a defensible starting point rather
  than an arbitrary one — but it should be re-verified against each
  detector backend's *actual* live publish rate (Euclidean vs. CenterPoint
  may differ) once the adapter is running, not assumed to stay exactly
  `2` forever.

### 4. TrackedObjects conversion — RESOLVED

| AB3DMOT KF state | `autoware_perception_msgs/msg/TrackedObjects` field | mapping |
|---|---|---|
| `x[0:3]` (`x,y,z`) | `kinematics.pose_with_covariance.pose.position` | direct, after the frame transform from §2 |
| `x[3]` (`theta`) | `kinematics.pose_with_covariance.pose.orientation` | `orientation_from_yaw`-style conversion (`z=sin(θ/2), w=cos(θ/2)`) — **reuse the exact existing pattern** already in `centerpoint_ros.py::yaw_to_quaternion` / `autoware_prediction_node.cpp::orientation_from_yaw`, not a new formula, per §1's resolution that `theta` is used as HEVEN's own yaw directly |
| `x[4:7]` (`l,w,h`) | `shape.dimensions.{x,y,z}` | direct, **provided** the adapter always constructs AB3DMOT's measurement vector as `[x,y,z,yaw,length,width,height]` using HEVEN's own axis semantics (`length`=x-extent, `width`=y-extent, `height`=z-extent) — must **not** use AB3DMOT's own `Box3D.bbox2array_raw` (`[h,w,l,x,y,z,theta]` order, camera semantics), which would silently swap axes |
| `x[7:10]` (`dx,dy,dz`) | `kinematics.twist_with_covariance.twist.linear` | **not a direct copy** — flagged finding below |
| KF `id` (int) | `object_id.uuid` | a deterministic, purely mechanical int→16-byte encoding (e.g. zero-padded big-endian bytes); no research ambiguity, just an implementation detail to fix when coding |
| `P[0:3,0:3]` | `pose_with_covariance.covariance` (position block) | direct — a real filter quantity, not invented |
| `P[3,3]` | `pose_with_covariance.covariance[35]` (yaw variance) | direct — matches the existing convention already used for the yaw-variance slot in `autoware_prediction_node.cpp` |
| `P[7:10,7:10]` | `twist_with_covariance.covariance` (velocity block) | derivable, **but must be scaled consistently with the velocity fix below** |
| everything else (z-velocity coupling, roll/pitch, off-diagonal terms AB3DMOT never tracks) | remaining covariance entries | **leave at safe defaults (zero/untouched), explicitly documented as untracked** — per this task's explicit "do not invent covariance values," and mirroring `autoware_prediction_node.cpp`'s own existing pattern of only populating validated fields and leaving the rest at message-default zero |

**Flagged finding — velocity units require a fix, not a direct copy**:
AB3DMOT's `kalman_filter.py::F` hardcodes coefficient `1` (not `dt`) in the
position↔velocity coupling (`F[0,7]=1`, etc. — already quoted in full in
§3 above), meaning `x[7:10]` is in units of **meters per KF step
(= per frame)**, not meters/second, because AB3DMOT always calls
`kf.predict()` exactly once per incoming frame with no explicit time
argument. Publishing this directly into `twist.linear` (which
`autoware_perception_msgs` expects in m/s) would be **numerically wrong**
whenever the detector's frame rate isn't exactly 1 Hz. §4 of this file
already documents that **SimpleTrack's own fork of this same KF**
(`mot_3d/motion_model/kalman_filter.py`) fixes exactly this by rebuilding
`F` with the real elapsed `time_lag` before each `predict()`/
`get_prediction()` call. The adapter should do the same — scale the
velocity state (and, for internal consistency, its variance in
`P[7:10,7:10]` by `dt²`) by the real measured inter-detection interval
before publishing `twist`, rather than using AB3DMOT's literal per-frame
`F` unmodified. This is a source-grounded, already-precedented fix (visible
in the second pinned reference repo), not a guess.

### 5. ROS interface — RESOLVED

```
input:  /ad/perception/objects/detected   (autoware_perception_msgs/msg/DetectedObjects, unchanged contract)
output: /experiment/tracked/ab3dmot        (autoware_perception_msgs/msg/TrackedObjects, frame_id=odom per §2)
```

- Matches the template given in this task exactly. The adapter subscribes
  the same `/ad/perception/objects/detected` topic the production Autoware
  tracker already consumes (§1 above) — no change to the detector side.
- Publishes to a new, `/experiment/`-namespaced topic, not
  `/ad/perception/objects/tracked` — the production Autoware tracker's
  output topic, launch file (`tracking.launch.py`), and config
  (`autoware.yaml`) are **untouched**, consistent with AGENTS.md
  ("Default tracker = Autoware," "Experimental algorithms must be
  opt-in"). Both trackers can run simultaneously off the same detections
  for direct comparison, exactly the same pattern already established by
  the existing Euclidean/CenterPoint comparison tooling
  (`docs/perception/centerpoint_vs_euclidean_comparison.md`).
- Frame: `odom`, per §2's resolution — required for the topic's messages
  to be meaningfully comparable to `/ad/perception/objects/tracked` and,
  if ever desired, consumable by the existing `AutowarePredictionNode`
  (which would need its own separate instance/config pointed at this topic
  to do so — not implemented or requested here).
