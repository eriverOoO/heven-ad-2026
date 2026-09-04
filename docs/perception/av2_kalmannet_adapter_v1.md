# AV2 Motion Forecasting -> HEVEN KalmanNet Dataset Adapter v1 (Stage 0)

Status: **adapter correctness only, real data verified. No training.**
Follows the read-only audit (see git history for the "KalmanNet / AV2
audit" conversation). This document, `av2_motion_forecasting_adapter.py`,
`av2_motion_forecasting_adapter_cli.py`,
`av2_motion_forecasting_adapter_validate.py`, and
`tools/av2_dataset_prep/fetch_av2_stage0.py` are the only artifacts this
task produced. **No KalmanNet architecture file was touched. No training
was run.**

## Why Argoverse 2 Motion Forecasting

HEVEN's own KalmanNet training data has always come from one MORAI static
scene (`morai_heven`, per `docs/agent/STATUS.md`'s T-9A/T-12 family of
experiments). Argoverse 2 (AV2) Motion Forecasting offers 250,000 real,
independently-GT'd driving scenarios across 6+ cities as free, public,
motion-only ground truth (position + native velocity, no LiDAR/camera
needed) -- a much larger and more diverse source of clean `[x, y, vx, vy]`
trajectories to pretrain the *existing* KalmanNet gain network on, before
fine-tuning on real HEVEN/MORAI data. This adapter builds the smallest
possible bridge from AV2's real schema to HEVEN's existing training
convention. It does not train anything and does not claim any benefit --
that is explicitly the next task (see "Next task" below and
`docs/agent/STATUS.md`).

## Exact mapping

| HEVEN KalmanNet field (`kalmannet_core.py`) | AV2 source (`av2.datasets.motion_forecasting.data_schema`) |
|---|---|
| `gt_state[t] = [x, y, vx, vy]` | `ObjectState.position` (x, y) + `ObjectState.velocity` (vx, vy) -- AV2 provides velocity natively, never finite-differenced, matching HEVEN's own existing convention |
| `timestamps_ns[t]` | `ArgoverseScenario.timestamps_ns[t]` |
| `measurement[t] = [x_meas, y_meas]` | a corrupted copy of `position` (§ Corruption v1) -- AV2 itself has no independent detector |
| `measurement_valid[t]` | produced entirely by this adapter's corruption layer (§ below); AV2's own `object_states` are internally gap-free (see "Real AV2 schema" below) |
| segment boundary | one AV2 `Track` (one `(scenario_id, track_id)`) segmented on a large gap / implausible teleport, using the exact same policy and defaults as `kalmannet_trajectory_adapter.SegmentConfig` |

`KNET_STATE_DIM` / `KNET_MEAS_DIM` are **imported** from
`kalmannet_trajectory_adapter.py`, never redefined. No KalmanNet
architecture (`kalmannet_core.py`, `kalmannet_arch2_core.py`) was touched.

## Real AV2 schema, confirmed against 120 real downloaded `train` scenarios

Not assumed from documentation -- measured directly (`av2` pip package
0.3.6, live against 120 real scenario parquet files, 7,358 tracks total):

- 110 timesteps per scenario, always, at a real uniform 100 ms step (10 Hz)
  -- confirmed on all 120/120 scenarios.
- `FOCAL_TRACK`/`SCORED_TRACK` (full 110-step) tracks always have exactly
  50 `observed=True` states followed by 60 `observed=False` states (the
  "50 observed + 60 future" convention) -- confirmed on all 1,425 such
  tracks in the sample, not just one example scenario.
- Every track's `object_states` list is internally **contiguous** over its
  own appearance window (`[min(timestep), max(timestep)]`, no internal
  gaps) -- confirmed 0/7,358 non-contiguous tracks. A "missing
  measurement" in the sense this adapter's corruption layer introduces is
  therefore a **synthetic construct this adapter adds**, not something raw
  AV2 itself contains.
- 0/7,358 tracks contained any non-finite `position`/`velocity`/`heading`.
- Object type distribution (real, this sample): `VEHICLE` 5,088, `PEDESTRIAN`
  996, `STATIC` 498, `BACKGROUND` 337, `RIDERLESS_BICYCLE` 153,
  `CONSTRUCTION` 96, `CYCLIST` 83, `BUS` 80, `MOTORCYCLIST` 17, `UNKNOWN` 10
  -- confirms the audit's expected class imbalance.
- Real S3 bucket layout (resolved, not guessed -- the two official
  documentation pages disagreed on a hyphen vs. underscore):
  `s3://argoverse/datasets/av2/motion-forecasting/{train,val,test}/<scenario_id>/`,
  containing `scenario_<id>.parquet` (needed) and `log_map_archive_<id>.json`
  (map file, **never downloaded** by this adapter's tooling -- KalmanNet v1
  needs no map geometry, confirmed: the real loader
  `load_argoverse_scenario_parquet(scenario_path)` takes only the parquet
  path).

## No KalmanNet architecture change

`kalmannet_core.py` and `kalmannet_arch2_core.py` are unmodified.
KalmanNet's existing, already-implemented real-`dt` support
(`KalmanNetFilter.step(z, dt)` -> analytical `F_matrix(dt)`/`Q_matrix(dt)`)
is reused unchanged; `dt` is **not** added as a neural-network feature
here or anywhere. The existing missing-measurement behavior (prediction-
only step, no learned-gain update, for a short/in-sequence miss; sequence
split for a large gap) is reused unchanged -- this adapter's
`measurement_valid[t] = False` / `measurement[t] = [NaN, NaN]`
representation (never `[0, 0]`, never a repeated previous value) is
exactly the input the existing `kalmannet_trajectory_loader.
TrajectoryArrays.to_kalmannet_sequence()` conversion (`z_meas[t] = None`
wherever invalid) already expects, so it feeds the existing prediction-
only behaviour with zero architecture change.

## Coordinate preprocessing: first-state-relative (versioned)

`coordinate_mode = "first_state_relative"`. For each segment,
`x'_t = x_t - x_0`, `y'_t = y_t - y_0`; velocity is untouched (translation-
invariant). The origin is always the segment's **own first sample** --
never a mean/centroid over the whole segment, so no future information
leaks into any timestep, including the first. `origin_x`/`origin_y` are
stored per segment (`trajectory_index.jsonl` row + `origin_xy` array in
the NPZ) so the transform is always exactly reversible.

**Structurally validated, not just asserted**, via a focused translation-
equivariance test (`test_translation_equivariance_linear_kf`,
`test_translation_equivariance_kalmannet_gru_gain` in
`test_av2_motion_forecasting_adapter.py`):

1. The classical `LinearCVKF` (from `kalmannet_core.py`, unmodified) run
   on a synthetic trajectory near the origin and on the identical
   trajectory translated by a large (~53 km) constant offset, with
   consistently-shifted initialization, produces **identical error
   dynamics** (`x_post - x_true` at every step, `atol=1e-8`) -- expected,
   since `F`/`H`/`Q`/`R` contain no absolute-position term.
2. `KalmanNetGRU` (the learned-gain network) never receives absolute
   position as an input feature at all -- only four *difference* vectors
   (`obs_diff`, `obs_innov_diff`, `fw_evol_diff`, `fw_update_diff`), which
   cancel any constant offset by construction. Confirmed numerically: an
   untrained, fixed-seed network fed the identical relative-difference
   features produces a byte-identical gain regardless of the underlying
   trajectory's absolute position.

Both tests pass. First-state-relative preprocessing is therefore
structurally safe for the existing model -- had either test failed, this
document would report that and a different coordinate representation
would have been chosen instead (per the task's own instruction), but that
did not happen here.

No rotation into an actor heading frame, no normalization by future
trajectory statistics -- both explicitly out of scope for v1.

## Segment / gap policy

Reuses the exact same philosophy and defaults as
`kalmannet_trajectory_adapter.TrajectoryConfig`
(`max_gt_gap_s=1.0`, `max_teleport_speed_mps=60.0`, `min_gt_samples=5`),
not a second segmentation system. A track/segment is never rejected
because of its AV2 `TrackCategory` alone (including `TRACK_FRAGMENT`,
which makes up 5,933/7,358 = 80.6% of tracks in the Stage-0 sample) --
only actual state finiteness/continuity and `min_gt_samples` gate
inclusion; a short segment is still exported, flagged
`valid_for_kalmannet_gt = false`. No interpolation, no fabricated
positions, ever -- a non-finite frame is dropped and counted
(`frames_dropped_nonfinite`), never replaced.

## Class metadata (never fed to KalmanNet)

Both the exact, untouched `av2_object_type` (one of AV2's real 10 enum
members) and a proposed `coarse_object_type` (`VEHICLE`/`PEDESTRIAN`/
`OTHER`, per the project's own stated future taxonomy target) are stored
in `trajectory_index.jsonl`. **Neither is a network input** --
`export_manifest.json`/`metadata.json` explicitly declare
`class_fed_to_kalmannet: false`, and the validator (§ below) fails closed
if that declaration is missing or false-negative. The coarse mapping
(`COARSE_CLASS_MAP` in `av2_motion_forecasting_adapter.py`) is a
**proposed v1 adapter-metadata convention**, not a learned class label:
`VEHICLE`+`BUS` -> `VEHICLE`; `PEDESTRIAN` -> `PEDESTRIAN`; everything else
(`MOTORCYCLIST`, `CYCLIST`, `STATIC`, `BACKGROUND`, `CONSTRUCTION`,
`RIDERLESS_BICYCLE`, `UNKNOWN`) -> `OTHER`, under the project's own fixed
3-bucket constraint -- not a claim that e.g. a cyclist is motion-equivalent
to a static obstacle.

## Corruption v1 (deliberately minimal)

Every knob is independently composable and default-off
(`CorruptionConfig`, deterministic given `seed` via a per-segment SHA-256-
derived RNG seed so results never depend on processing order):

- **A. no corruption** (the default) -- `measurement == clean_position`
  exactly, `measurement_valid = True` everywhere.
- **B. zero-mean Gaussian position noise** -- `gaussian_noise_std_m`.
- **C. independent per-frame dropout** -- `dropout_prob` (Bernoulli per
  frame).
- **D. short dropout bursts** -- `dropout_burst_enabled` +
  `dropout_burst_prob` + `dropout_burst_min_len`/`dropout_burst_max_len`.

Explicitly **not** implemented in v1 (deferred, per the audit's own
staged corruption policy): range-dependent noise, heavy-tail tuning, true
ID-switch injection, class-conditioned corruption, MORAI-calibrated
noise, complex outliers. A future MORAI-calibrated corruption pass should
estimate these from real detector-vs-GT residuals before this config is
extended -- not guessed.

## Split / leakage policy

AV2's own `train`/`val`/`test` are disjoint S3 prefixes, already
leakage-free by construction -- this adapter **never re-splits AV2
scenarios**; `--split` selects which of AV2's own three splits to process,
verbatim. `check_scenario_split_leakage` still guards against a bug that
would otherwise let one `scenario_id` land in two different output splits
within one export. `split_manifest.json` records `"split_source":
"av2_native (this adapter never re-splits AV2 scenarios)"` explicitly.

## Storage policy

Raw parquet and processed NPZ shards live **outside the git repository**,
under `~/datasets/av2/{raw_staging,processed}/` -- never committed. Only
a small (48 KB), deterministic scenario-id manifest (scenario IDs, S3
keys, split, selection rule + seed, SHA-256 per downloaded file -- no
absolute machine paths) is committed, at
`ad_morai_bridge_dev/config/dataset_factory/av2_motion_forecasting_adapter/stage0_scenarios.manifest.json`.
Download and preprocessing are deliberately separate tools
(`tools/av2_dataset_prep/fetch_av2_stage0.py` vs.
`av2_motion_forecasting_adapter_cli.py`).

## Competition compliance boundary

**AV2 trajectory labels (position, velocity, heading, object type,
track category) are OFFLINE training data only.** They exist solely to
pretrain the KalmanNet gain network before any MORAI fine-tuning. **No
AV2 data, and no MORAI simulator ground truth, is ever a runtime
inference input.** The AV2 adapter produces files on disk consumed only
by a future offline training script (not built in this task -- see "Next
task"); nothing in this task subscribes to, queries, or wires any AV2 or
MORAI GT source into any ROS node, launch file, or the AB3DMOT/Autoware
runtime graph. The permitted competition runtime inputs (VLP-16, camera,
GPS, IMU, Competition Vehicle Status, and other explicitly-permitted UDP
data) are entirely untouched by this task.

## Real Stage-0 run (this task, on this machine)

120 real `train` scenarios (deterministic sample: pool of 2,000 listed
scenario ids, seed `20260905`, uniform sample without replacement --
recorded in `stage0_scenarios.manifest.json`), 20 MB raw, downloaded via
plain unsigned HTTPS `GET` (no `s5cmd`/`boto3`/AWS account needed -- the
bucket permits anonymous reads). Adapter export (no-corruption config):
7,358 tracks -> 7,358 segments (0 needed splitting -- AV2 tracks are
already contiguous), 0 rejected, 382,867 total GT samples, ~66 MB
processed (one NPZ per segment, uncompressed/deterministic, matching the
existing `kalmannet_morai_trajectory_v1` container convention -- larger
than the compressed-columnar raw parquet, an *expansion*, not a
compression, dominated by per-file container overhead at this small
per-segment scale), ~10 s wall time. A second real export with a Gaussian
+ dropout + dropout-burst example config produced the identical segment
count with corrupted measurements. Both real exports pass the validator
with zero errors.

## Next task (explicitly NOT started here)

Bring a reproducible KalmanNet training entry point into the repository
(none currently exists -- every historical training run
lived in ad-hoc, uncommitted scripts under `~/heven_presentation_assets/`
on one development machine) and run the first small AV2-pretraining
sanity experiment, comparing against the existing Tuned Linear KF and
DENSE-KALMANNET-v2 baselines on the existing frozen MORAI evaluation
data. Not implemented in this task.
