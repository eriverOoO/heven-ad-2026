# KalmanNet Detector Measurement Attachment v1

Attaches sample-aligned **real detector outputs** to a
`kalmannet_morai_trajectory_v1` export, producing a
`kalmannet_morai_measurement_v1` dataset: the same GT trajectories, byte-for-
byte unchanged, with the measurement slots populated **only** where a real
detection was supervised-matched to the GT actor.

## Purpose

KalmanNet training needs a true state `x_k` and a measurement `z_k`. The
trajectory adapter provides `x_k` and intentionally exports
`measurement_valid = false` everywhere. This layer fills `z_k` — and only
`z_k` — from actual detector outputs supplied as input. It never fabricates
an observation.

**Not in scope:** no KalmanNet training, no `optimizer.step()`, no synthetic
Gaussian noise, no simulated dropout, no filled / interpolated / forward-
filled measurements, no detector inference, no KNet architecture / state dim /
measurement dim change, no Linear KF / CenterPoint / AB3DMOT / production
tracker / planner change, no frozen KNet/KF conclusion touched.

Frozen research context preserved (data only, restated nowhere as output):
historical T-9A KalmanNet ~0.854 m vs tuned Linear KF ~0.855 m (tied); dense
v2 ~1.467 m vs ~1.456 m position, ~2.462 vs ~2.416 m/s velocity; long
controlled dropout favours the tuned KF (~1.23 m) over KalmanNet (~28.92 m).

## Three concepts — only #2 is handled

1. **GT presence** — an actor has valid simulator ground truth.
2. **detector measurement presence** — a real detection can be associated to
   the GT actor. This layer builds #2 **for supervised dataset construction**.
3. **runtime tracker association** — a live multi-object tracker deciding
   which detection belongs to which track. **Not implemented or evaluated
   here.** The runtime system has no ground truth and cannot reproduce the
   association this layer performs.

## Source trajectory schema

`kalmannet_morai_trajectory_v1` — time-major `[T, ...]` per-trajectory NPZ,
GT state `[x, y, vx, vy]` in the MORAI `map` frame, `dt_s[0] = nan`, empty
measurement slots. See `kalmannet_morai_trajectory_adapter_v1.md`.

## Detector-record schema (`kalmannet_detector_measurements_v1`)

Detector-neutral, built by a **source adapter** from an already-serialized
detector output (no inference is run):

| field | notes |
| --- | --- |
| `detector_source` | `centerpoint` / `euclidean` / … |
| `detector_config_id` | opaque provenance string |
| `class_semantics` | `class_aware` / `class_agnostic` |
| `detector_manifest_sha256` | SHA-256 of the exact input file bytes |
| `canonical_dataset_id` | optional; checked against the canonical dataset |
| `completeness` | `{expected_frames, processed_frames, failed_frames, complete}` |
| `frames[sample_id]` | `{sample_id, lidar_stamp_ns, detector_frame_present, detections}` |
| `detection` | `{index, class_name, center_lidar[3], dims_lwh[3], yaw, yaw_available, score}` |

### CenterPoint source adapter

`centerpoint_records_from_heven_offline(...)` reads the existing
`heven.offline_detection.v1` JSONL (produced by
`tools/centerpoint_offline/prediction_bridge.py` — the same contract the
CenterPoint evaluator consumes). Boxes are `lidar_link` geometric-centre
`[x, y, z, l, w, h, yaw]`, `class_aware`, `yaw_available = True`. Alignment
key: the record's own `sample_id`.

### Euclidean source adapter

`euclidean_records_from_ros_comparison(...)` reads the existing
`heven.ros_detection_comparison.v1` JSONL (produced by
`ad_lidar_perception/record_detected_objects.py`). That record has a LiDAR
header stamp and `frame_id` but **no `sample_id` and no yaw** (the adaptive
Euclidean cluster node emits an axis-aligned box centre with
`orientation_availability = UNAVAILABLE` and `classification = UNKNOWN`). So
it is `class_agnostic`, `yaw_available = False`, and alignment is by **exact
LiDAR anchor stamp scoped to a caller-supplied `--run-id`**; two frames of
that run at the same stamp are a hard error.

## Historical observation vector

Verified against the current `kalmannet_core.py`: `MEAS_DIM = 2`,
`MEAS_FIELDS = [x, y]`. Unchanged. `z_k` here is a 2-D position measurement.

## Coordinate frame

The KNet GT state is `map`-frame `[x, y, vx, vy]`, so `z_k` is stored as
**map-frame `[x, y]`**. Both detector sources emit `lidar_link` boxes, so
each detection centre is transformed `lidar_link -> map` at the exact source
sample timestamp before matching and storage.

### `lidar_link -> map` transform

Recomposed offline from the canonical dataset's own
`runs/<scenario>/<run_id>/tf/<frame>.json` — exactly the pieces the factory
recorded (`map_to_odom`, `odom_to_base`, `static_edges`) and the same
composition as `transforms.build_map_to_lidar`:
`map_to_lidar = base_to_lidar^-1 . odom_to_base^-1 . map_to_odom`, then
inverted. **No live / latest TF is queried.** A test recomposes the transform
and verifies that applying it to a frame's GT `map_frame.center` reproduces
the factory's independently-recorded `lidar_frame.center` for the same box.

A canonical frame that lacks the `map -> lidar` chain (TF not matched — a
tracking-valid frame does not require TF) yields `measurement_valid = false`
with reason `no_map_transform`, distinct from a genuine detector miss.

## Sample alignment

Primary key: source `sample_id`. Fallback (Euclidean only, whose record has
no `sample_id`): exact `lidar_header_stamp_ns`, scoped to `--run-id`. Never
nearest wall time, frame counter, or file order. A detector run whose
canonical `dataset_manifest.json` SHA-256 does not match the one the
trajectory export was derived from is refused (`cross-dataset pairing
refused`); a mismatched `dataset_id` or `canonical_dataset_id` is also
refused.

## Supervised GT association

Frame-local, one frame at a time; no future GT, no past/future detections, no
track continuity. Every export records
`assignment_method: gt_supervised_bev_center_distance_hungarian`.

- **candidate** = (optionally) class-compatible AND BEV centre distance
  `<= max_distance_m`.
- **metric**: `bev_center_distance` (the historical observation is a 2-D
  position; the Euclidean detector has no yaw).
- **assignment**: pure-NumPy deterministic Hungarian (`hungarian_min_cost`),
  minimum total candidate distance. No scipy; **no silent greedy fallback**.
  Tested optimal against brute force on every matrix up to 4×4.
- **one-to-one**: per frame, one detection ↔ at most one GT actor, and one GT
  actor ↔ at most one detection. A non-candidate can never be forced through.
- **gate**: `max_distance_m` default **2.5 m**, derived from the MORAI
  vehicle footprint (~4.6 × 1.9 m, half-diagonal ≈ 2.5 m). It is a
  **label-construction** gate — explicitly **not** the AB3DMOT runtime
  `euclidean_gate_m` (3.0 m) and **not** the CenterPoint evaluator's IoU
  threshold (0.50). Configurable; the derivation is recorded in the manifest.
- **class compatibility**: `vehicle <-> {vehicle, car, truck, bus}`.
  Automatically disabled for a `class_agnostic` detector source; the
  effective value is recorded in `export_manifest.json -> association`.
- **IoU diagnostic**: oriented BEV IoU in the map frame, computed only when
  the detector supplies yaw (`yaw_available`). `measurement_match_iou` is
  `nan` for a yaw-less detector. Assignment never uses IoU.

## Measurement vector

For a matched frame: `measurement[t] = [det_center_map_x, det_center_map_y]`
— the **detector's** transformed centre, never the GT position.
`measurement_state[t]` carries the same `[x, y]` in columns 0–1; the velocity
columns stay `nan` (the detector supplies no velocity and KNet `z_k` is
position-only; no finite differencing).

## Missing-measurement semantics

`measurement_valid = false`, `measurement = nan`, and the GT timestep is
**kept**. The per-trajectory index row's `miss_reason_counts` distinguishes:

| reason | meaning |
| --- | --- |
| `detector_frame_missing` | no detector record for this canonical sample |
| `zero_detections` | detector record present, 0 detections |
| `no_map_transform` | canonical frame lacks the `map -> lidar` chain |
| `gated_out` | a detection existed but no candidate within gate / class |
| `assigned_elsewhere` | a candidate detection went to another GT actor |

No forward-fill, no interpolation, no KF pseudo-measurement. A mask pattern
`[True, False, True]` stays exactly that, with the middle `measurement` `nan`.

## False positives

A detector box with no GT candidate within the gate is an **unmatched
detection**, counted at frame / run / dataset level. It never creates a GT
trajectory. A detector-only appearance where the GT actor is absent is not a
supervised sample.

## Natural miss runs

`longest_missing_run_frames` and `longest_missing_run_s` (per trajectory) and
`longest_natural_missing_run_frames` (dataset) are **dataset characterization
only** — computed from naturally-missing detector observations, never a
designed removal. This is **not** a KalmanNet robustness claim and no
T-12-style gap is synthesised.

## Training-ready

`kalmannet_training_ready` (per trajectory) =
`valid_for_kalmannet_gt` AND `measurement_count > 0` AND all valid
measurements finite AND `measurement` is 2-D. There is **no coverage
threshold** — a trajectory with one real measurement satisfies the schema.
The index reports `measurement_count` / `measurement_coverage` /
`longest_missing_run_*` so a later Training Prep can choose *recommended*
sequence eligibility separately from *schema* readiness.

## Split preservation

`splits/{train,val,test}.txt` are copied **verbatim** from the trajectory
export. No resplit. Attaching Euclidean vs CenterPoint outputs to the same
trajectory export yields identical train/val/test membership; the validator
(`--trajectory-root`) checks this.

## Output layout

```
<root>/
  .kalmannet_morai_measurement_dataset
  export_manifest.json
  metadata.json
  trajectory_index.jsonl
  split_manifest.json
  splits/{train,val,test}.txt
  trajectories/<trajectory_id>.npz    # every GT array verbatim + populated measurement arrays
```

Trajectory ids are inherited from the GT export. Each NPZ carries every
`kalmannet_morai_trajectory_v1` GT array **numerically identical** (locked by
a test and by the validator) plus the (re)written measurement arrays.

## Provenance / fingerprints

`export_manifest.json` records: measurement schema version, source trajectory
`content_fingerprint` + commit, canonical `dataset_id` + manifest SHA-256,
detector source / config id / class semantics / manifest SHA-256 /
completeness / provenance, association config, adapter commit, aggregate
coverage / false-positive / miss-reason / match-distance / IoU stats.
`content_fingerprint` = SHA-256 over measurement schema + trajectory
fingerprint + detector manifest SHA-256 + association config + sorted
per-artifact hashes. A repeat export of the same inputs is byte-identical;
record order in the detector JSONL does not change any trajectory NPZ or
index row (only the input-bytes SHA legitimately differs).

## Real data status

No `morai_tracking_dataset_v1` exists on disk in this environment, so no real
detector records exist and **no real measurement-attached trajectories were
produced**. All validation is fixture-only: sources are built through the
real Dataset Factory + real trajectory adapter; the detector-prediction
records are synthesised for unit testing (`fixture_detector_records`) and are
**not** detector-performance evidence.

## Tests

`ad_morai_bridge_dev/test/test_kalmannet_measurement_attachment.py` (27):
Hungarian vs brute force + deterministic tie-break; positive match uses the
detection not GT; physical gate; class mismatch / class-agnostic; one-det/
two-GT and two-det/one-GT one-to-one; false positive unmatched; `lidar->map`
recomposition vs recorded `lidar_frame.center`; TF-absent → `no_map_transform`
masked + counted; end-to-end export + validator; measurement value is the
detector centre; frame-present-zero-detections vs detector-frame-missing;
false positive creates no trajectory; GT arrays immutable; natural miss-run
stats; sample-order invariance; deterministic repeat export; training-ready
criterion; split preserved verbatim + detector-agnostic; cross-dataset
pairing refused; incomplete detector refused; Euclidean source adapter
(class-agnostic, no IoU); Euclidean stamp-ambiguity hard error. Regression:
`test_kalmannet_trajectory_adapter.py` 27, `test_dataset_factory.py` 27,
`test_centerpoint_adapter.py` 22 pass unchanged. pyflakes + `py_compile`
clean.

## Recommended next task

Conditional. **If** a real `morai_tracking_dataset_v1` exists **and** real
detector measurements from at least one detector source have been attached
**and** train / val independent groups exist: **KalmanNet MORAI Training Prep
v1** — audit the real measurement-attached trajectories against the current
KalmanNet model / state / observation contract, define sequence batching,
masking, normalization, loss / evaluation protocol and leakage-safe train /
validation manifests, and run data / model / environment preflight without
starting full KalmanNet training. **Else**: **MORAI Dataset Collection Pilot
v1** — run the already-defined real MORAI pilot on a machine / session with
the simulator, gRPC runtime and ROS bridge active, then export the same
canonical capture through the CenterPoint adapter and the KalmanNet
trajectory adapter, generate real detector outputs, and attach those
measurements before any learned-model training.
