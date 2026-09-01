# KalmanNet MORAI Trajectory Adapter v1

Deterministic, versioned converter from a canonical
`morai_tracking_dataset_v1` (MORAI Tracking Dataset Factory output) to a
per-trajectory ground-truth dataset for later Linear-KF replay and
KalmanNet training / validation.

**This adapter trains nothing.** It builds only the GT trajectory side plus
an explicit measurement-attachment contract whose values are entirely
absent in v1. It does not import or modify the Linear KF, the production
tracker default, AB3DMOT, CenterPoint, the planner, or `kalmannet_core.py`.
No synthetic noise is written as canonical data. No estimator claim is made
anywhere in the output.

Frozen KalmanNet / Linear-KF research context this adapter must not
contradict (it produces data only, so it restates none of these):

| context | KalmanNet | tuned Linear KF |
| --- | --- | --- |
| historical T-9A position RMSE | ~0.854 m | ~0.855 m (effectively tied) |
| dense-trained v2 position RMSE | ~1.467 m | ~1.456 m |
| dense-trained v2 velocity RMSE | ~2.462 m/s | ~2.416 m/s |
| long controlled measurement dropout | ~28.92 m | ~1.23 m |

## Schema

`ADAPTER_SCHEMA_VERSION = "kalmannet_morai_trajectory_v1"`,
`MEASUREMENT_CONTRACT_VERSION = "kalmannet_morai_measurement_contract_v1"`.

State / measurement dimensions are mirrored from
`ad_lidar_perception/ad_lidar_perception/kalmannet_core.py`
(`STATE_DIM = 4` -> `[x, y, vx, vy]`; `MEAS_DIM = 2` -> `[x, y]`); a parity
test scans that file for the literals.

GT kinematics are in the **MORAI `map` frame** (absolute, planar). This is
the frame `valid_for_tracking_gt` guarantees on a canonical box (pose +
dims + `map_frame` present), the same frame the historical KalmanNet
trajectory study used, and it needs no TF chain.

Position is `box.map_frame.center[:2]` (the **box centre** - a future
detector measurement is also a box centre, so no constant rear-axle offset
is learned as bias). Velocity is `box.source.velocity_map[:2]` - the
**native MORAI source velocity**, never a finite difference. A diagnostic
`finite_diff_velocity` array (central difference of the map-frame position)
is stored alongside but is never the state.

## Trajectory unit and segmentation

One trajectory = one persistent MORAI `unique_id` within one capture run,
optionally split into segments. Trajectory id:
`<run_id>__actor_<unique_id>__segment_<nnn>` where `run_id` is itself
`<scenario_id>__seed_<seed>__run_<nnn>` - so scenario, run, actor and
segment are all encoded and no trajectory ever spans a run.

Frame eligibility (both gates, counted separately in the manifest):

1. the `frames.jsonl` row has `valid_for_tracking_gt == true` (which
   already implies `actor_gt.status == "matched"`), and
2. the per-actor box in `gt/<frame>.json` has `valid_for_tracking_gt ==
   true` and a finite `map_frame` + `source.velocity_map` +
   `source.position_map`.

Within a `(run, actor)`, samples are ordered by `lidar_header_stamp_ns`
(exact-duplicate stamps dropped and counted; backward stamps dropped and
counted), then a new segment starts when either

- the gap to the previous sample exceeds `max_gt_gap_s` (default **1.0 s**
  = 10 x the nominal 0.1 s / 10 Hz LiDAR anchor period; the observed
  per-export `dt` distribution is recorded in
  `export_manifest.json -> observed_dt_s` for retuning), or
- the implied step speed exceeds `max_teleport_speed_mps` (default
  **60.0 m/s** - actor respawn under a reused `unique_id`).

No fabricated zero states, no interpolation, no forward-fill: only
actor-present samples are stored, each with its source frame index and
`lidar_header_stamp_ns`.

## Output layout

```
<derived_root>/
  .kalmannet_morai_trajectory_adapter    # marker
  metadata.json                          # schema + state/measurement contract
  export_manifest.json                   # adapter schema, source hash, counts, fingerprint
  trajectory_index.jsonl                 # one row per trajectory: provenance + stats + flags
  split_manifest.json                    # grouping key, group + trajectory counts, leakage guarantee
  splits/{train,val,test}.txt
  trajectories/<trajectory_id>.npz
```

Each `trajectories/<id>.npz` is a deterministic (uncompressed, fixed
1980-01-01 member stamps, sorted) archive of time-major `[T, ...]` arrays:

| array | shape | notes |
| --- | --- | --- |
| `timestamps_ns` | `[T]` int64 | strictly increasing |
| `dt_s` | `[T]` float64 | `dt_s[0]` is `nan`; `dt_s[1:] > 0` |
| `gt_state` | `[T, 4]` float64 | `[x, y, vx, vy]` - the KalmanNet state |
| `gt_position_map` | `[T, 3]` float64 | map-frame box centre incl. z (aux) |
| `gt_velocity_map` | `[T, 3]` float64 | native source velocity (aux; state uses `[:2]`) |
| `gt_yaw` | `[T]` float64 | wrapped map-frame yaw (aux) |
| `source_position_map` | `[T, 3]` float64 | raw `source.position_map` (provenance) |
| `finite_diff_velocity` | `[T, 2]` float64 | diagnostic only |
| `source_frame_indices` | `[T]` int64 | |
| `source_sample_ids` | `[T]` `<U128` | |
| `actor_gt_skew_ns` | `[T]` int64 | actor-GT match skew per frame |
| `measurement` | `[T, 2]` float64 | **all `nan` in v1** |
| `measurement_state` | `[T, 4]` float64 | **all `nan` in v1** |
| `measurement_valid` | `[T]` bool | **all `False` in v1** |
| `measurement_score` | `[T]` float64 | all `nan` in v1 |
| `measurement_match_distance_m` | `[T]` float64 | all `nan` in v1 |
| `measurement_match_iou` | `[T]` float64 | all `nan` in v1 |
| `measurement_box_lidar` | `[T, 7]` float64 | all `nan` in v1 |
| `measurement_source` | `[T]` `<U64` | `""` in v1 |
| `measurement_class` | `[T]` `<U32` | `""` in v1 |

## Measurement-attachment contract (interface only in v1)

`build_measurement_arrays(trajectory, per_frame_measurements, *,
assignment_method, source)` is the pure function a future Euclidean /
CenterPoint measurement pass must satisfy. It overlays the measurement
arrays for the frames it is given (keyed by `source_frame_index`) and
leaves every other frame masked out (`measurement_valid = False`,
`measurement = nan`). No forward-fill, no interpolation, no detector
inference, no synthetic noise: a frame is either genuinely paired to a
real detector output or it is masked. v1 exports never call it (the slots
ship empty). The mask semantics (`measurement` finite iff
`measurement_valid`) hold in v1 and after any future attachment, and the
validator checks them.

## Splits and leakage

Grouping key `(scenario_id, requested_seed)` with `run_id` fallback,
assigned at group level - never per trajectory - so all trajectories of one
run stay in one split. The split machinery (`plan_split`, `auto_split`,
`check_split_leakage`) is **imported verbatim from
`centerpoint_adapter.py`**: `auto_split` hashes with the shared adapter
salt, so a `(scenario, seed)` group lands in the **same** split for the
KalmanNet and CenterPoint derived datasets. This is deliberate shared
provenance - export both from one canonical capture and their train / val /
test partitions agree. `< 3` groups -> all `train` + a warning (never a
faked val/test split). `check_split_leakage` runs inside `export` and in
the standalone validator and fails on any group or run spanning splits.

A source with zero tracking-valid trajectories raises rather than writing
an empty-looking manifest.

## GT-ready vs training-ready

- `valid_for_kalmannet_gt` (per trajectory): `sample_count >=
  min_gt_samples` (default 5) and all `dt_s[1:]` finite positive and all
  `gt_state` finite. A shorter trajectory is still exported, flagged false.
- `kalmannet_training_ready` / manifest `training_ready`: **always `false`
  in v1** - training needs at least one real detector measurement, and v1
  attaches none.

## Content fingerprint

SHA-256 over the adapter schema, the source `dataset_manifest.json` SHA-256,
the canonical adapter config, the split manifest, and the sorted per-artifact
hashes (deterministic NPZ bytes + index-row bytes). `created_at` is not an
input; a repeat export of the same source is byte-identical (tested).

## Entry points

- `ad_morai_dataset_export_kalmannet <source_root> --output <derived_root>
  [--split-plan plan.yaml] [--config trajectory_config.yaml] [--dry-run]
  [--validate-only] [--overwrite]`
- `ad_morai_dataset_validate_kalmannet <derived_root> [--json]`

`ad_morai_bridge_dev/dataset/kalmannet_trajectory_loader.py` is a
torch-free, ROS-free reader: `load_trajectory(path)` /
`iter_split(root, split)` return time-major arrays;
`TrajectoryArrays.to_kalmannet_sequence()` returns a dict matching
`kalmannet_core.Sequence`'s field contract (`dt[0] is None`, `z_meas[k] is
None` wherever the frame is unmeasured - every frame, in v1).

## Validation

No real `morai_tracking_dataset_v1` exists on disk; every test builds its
source through the actual factory `RunWriter` / `CaptureSession`. 27 tests
(`ad_morai_bridge_dev/test/test_kalmannet_trajectory_adapter.py`): contract
parity vs `kalmannet_core.py`, single / multi-actor extraction, run-reset
same-id independence, native-velocity-not-finite-difference (deliberate
mismatch), time-gap / teleport / variable-dt segmentation, eligibility
accounting, full export + validator, zero-trajectory rejection,
deterministic repeat export, `min_gt_samples` flag, explicit split plan,
multi-actor run never frame-split, injected-leakage detection, `< 3` groups
all-train, absent / double-listed split-plan group errors,
CenterPoint-shared auto-split partition, empty v1 measurement slots,
`build_measurement_arrays` mask semantics + unknown-frame rejection, loader
tensor-layout parity, source-never-modified. Regression: `test_dataset_factory.py`
27 + `test_centerpoint_adapter.py` 22 pass unchanged. A venv cross-check
runs one exported trajectory's GT through the real `kalmannet_core.LinearCVKF`
(`STATE_DIM = 4`, `MEAS_DIM = 2`, `dt[0] = None`, time-major) - layout
parity confirmed at runtime.

## Recommended next task

Conditional. **If** a real `morai_tracking_dataset_v1` exists **and** real
detector measurements have been produced from the same canonical capture:
KalmanNet MORAI Training Prep v1 (audit this trajectory dataset + the
attached measurements against `kalmannet_core.py`, define a leakage-safe
train/val experiment, preflight without training). **Else**: MORAI Dataset
Collection Pilot v1 - run the existing real MORAI pilot on a machine /
session with the simulator + gRPC + ROS bridge active, then export both the
CenterPoint dataset and KalmanNet GT trajectories from the same canonical
capture before any learned-model training.
