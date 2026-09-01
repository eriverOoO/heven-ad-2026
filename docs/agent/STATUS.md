# STATUS

## KalmanNet MORAI Trajectory Adapter v1 — COMPLETE (GT trajectory data only)

Branch `feat/kalmannet-morai-trajectory-adapter-v1`, from merged PR #31 main
`7ccf9de` (`feat(perception): add CenterPoint MORAI evaluator (#31)`).
**Data adaptation only. No KalmanNet trained, no KNet architecture /
hyperparameter change, no Linear KF / production tracker default / AB3DMOT /
CenterPoint / planner change, no frozen T-9 / T-12 conclusion touched, no
synthetic noise written as canonical data, no KNet-vs-KF claim.**

**What it is:** deterministically converts the *tracking-valid* GT of a
canonical `morai_tracking_dataset_v1` into a per-trajectory dataset for
later Linear-KF replay and KalmanNet training / validation. Builds the GT
trajectory side + an explicit measurement-attachment contract whose values
are **entirely absent in v1**. New modules
`ad_morai_bridge_dev/ad_morai_bridge_dev/dataset/{kalmannet_trajectory_adapter.py,
kalmannet_trajectory_adapter_cli.py,kalmannet_trajectory_adapter_validate.py,
kalmannet_trajectory_loader.py}`. Entry points
`ad_morai_dataset_export_kalmannet`, `ad_morai_dataset_validate_kalmannet`.

**Schema** `kalmannet_morai_trajectory_v1` /
`kalmannet_morai_measurement_contract_v1`. State / measurement dims mirrored
from `ad_lidar_perception/.../kalmannet_core.py` (`STATE_DIM = 4` ->
`[x, y, vx, vy]`, `MEAS_DIM = 2` -> `[x, y]`; parity test scans the
literals). GT in the MORAI **`map` frame** (what `valid_for_tracking_gt`
guarantees; no TF chain). Position = `box.map_frame.center[:2]` (box centre
- a future detector measurement is also a box centre, so no rear-axle
offset is learned as bias). Velocity = `box.source.velocity_map[:2]`
(**native**, never finite difference; `finite_diff_velocity` stored as a
diagnostic only).

**Trajectory unit:** one persistent MORAI `unique_id` within one capture
run, optionally segmented. Id `<run_id>__actor_<uid>__segment_<nnn>`
(`run_id` = `<scenario>__seed_<seed>__run_<nnn>`), so no trajectory spans a
run. Frame eligibility = frame-row `valid_for_tracking_gt` **and** per-box
`valid_for_tracking_gt` + finite `map_frame`/`velocity_map`/`position_map`;
box-level vs frame-level rejections counted separately. Within `(run,
actor)`: order by `lidar_header_stamp_ns` (dup stamps dropped+counted,
backward dropped+counted); new segment when `dt > max_gt_gap_s` (default
**1.0 s** = 10 x nominal 0.1 s / 10 Hz; observed dt distribution recorded in
`export_manifest.json -> observed_dt_s`) or implied step speed >
`max_teleport_speed_mps` (default **60 m/s**). No fabricated zero states, no
interpolation, no forward-fill.

**Output:** `<root>/{.kalmannet_morai_trajectory_adapter, export_manifest.json,
metadata.json, trajectory_index.jsonl, split_manifest.json,
splits/{train,val,test}.txt, trajectories/<id>.npz}`. Each `.npz` is a
**deterministic** archive (uncompressed, fixed 1980-01-01 member stamps,
sorted) of time-major `[T, ...]` arrays: `timestamps_ns` (int64, strictly
increasing), `dt_s` (`dt_s[0]=nan`, `dt_s[1:]>0`), `gt_state [T,4]`,
`gt_position_map [T,3]`, `gt_velocity_map [T,3]`, `gt_yaw`,
`source_position_map`, `finite_diff_velocity` (diagnostic), `source_frame_indices`,
`source_sample_ids` (`<U128`), `actor_gt_skew_ns`, and the measurement-contract
slots (`measurement [T,2]`, `measurement_state [T,4]`, `measurement_valid [T]`,
`measurement_score`, `measurement_match_distance_m`, `measurement_match_iou`,
`measurement_box_lidar [T,7]`, `measurement_source`, `measurement_class`) -
**all `nan` / `False` / `""` in v1**.

**Measurement contract:** `build_measurement_arrays(trajectory,
per_frame_measurements, *, assignment_method, source)` - pure function a
future Euclidean / CenterPoint measurement pass must satisfy; overlays only
the frames it is given (keyed by `source_frame_index`), leaves the rest
masked. No forward-fill / interpolation / detector inference / synthetic
noise. Mask invariant (`measurement` finite iff `measurement_valid`) holds
in v1 and after attachment; validator checks it. v1 exports never call it.

**Splits:** `plan_split` / `auto_split` / `check_split_leakage` **imported
verbatim from `centerpoint_adapter.py`** (shared `AdapterError` too).
Grouping `(scenario_id, requested_seed)` fallback `run_id`, group-level
assignment - a run never frame-split across splits. `auto_split` uses the
shared adapter salt, so a `(scenario, seed)` group lands in the **same**
split for the KalmanNet and CenterPoint derived datasets (deliberate shared
provenance). `<3` groups -> all-train + warning. Zero tracking-valid
trajectories -> hard error (no empty manifest).

**GT-ready vs training-ready:** `valid_for_kalmannet_gt` per trajectory
(`sample_count >= min_gt_samples` default 5, all `dt` finite positive, all
`gt_state` finite); a shorter trajectory is still exported, flagged false.
`kalmannet_training_ready` / manifest `training_ready` = **always false in
v1** (no measurements attached). Content fingerprint SHA-256 over schema +
source manifest sha + canonical config + split manifest + sorted per-artifact
hashes (`created_at` excluded); repeat export byte-identical.

**Real source dataset: NO** (no `morai_tracking_dataset_v1` on disk).
Validated by **27 tests** (`test_kalmannet_trajectory_adapter.py`), sources
built through the real factory `RunWriter`/`CaptureSession`: contract parity
vs `kalmannet_core.py`, single/multi-actor extraction, run-reset same-id
independence, native-velocity-not-finite-diff (deliberate 7 vs 10 m/s
mismatch), time-gap / teleport / variable-dt segmentation, eligibility
accounting, full export+validator, zero-trajectory rejection, deterministic
repeat export, `min_gt_samples` flag, explicit split plan, multi-actor run
never frame-split, injected-leakage detection, `<3` groups all-train,
absent/double-listed split-plan errors, CenterPoint-shared auto-split
partition, empty v1 measurement slots, `build_measurement_arrays` mask +
unknown-frame rejection, loader tensor-layout parity, source-never-modified.
Regression: `test_dataset_factory.py` 27, `test_centerpoint_adapter.py` 22
pass unchanged; pyflakes clean; CLI `--validate-only` / `--dry-run` /
`--overwrite` export / standalone validator end-to-end; venv cross-check
runs an exported trajectory's GT through the real `kalmannet_core.LinearCVKF`
(`STATE_DIM=4`, `MEAS_DIM=2`, `dt[0]=None`, time-major - runtime parity).
`colcon build --packages-select ad_morai_bridge_dev` still can't complete
here (deps not built - pre-existing); `find_packages` + `py_compile` +
entry-point import + config parse verified.

**Files:** `ad_morai_bridge_dev/ad_morai_bridge_dev/dataset/{kalmannet_trajectory_adapter.py,
kalmannet_trajectory_adapter_cli.py,kalmannet_trajectory_adapter_validate.py,
kalmannet_trajectory_loader.py}`;
`ad_morai_bridge_dev/config/dataset_factory/kalmannet_trajectory_adapter/{trajectory_config.yaml,
split_plan.example.yaml}`;
`ad_morai_bridge_dev/test/test_kalmannet_trajectory_adapter.py`;
`ad_morai_bridge_dev/setup.py` (+2 entry points, +config data_files);
`docs/perception/kalmannet_morai_trajectory_adapter_v1.md`, this file. No
`kalmannet_core.py` / Linear KF / AB3DMOT / CenterPoint / planner change.

**Recommended next task:** CONDITIONAL. **If** a real `morai_tracking_dataset_v1`
exists **and** real detector measurements have been produced: KalmanNet
MORAI Training Prep v1 - audit this trajectory dataset + the attached
measurements against `kalmannet_core.py`, define a leakage-safe train/val
experiment and preflight dataset/model compatibility without training.
**Else:** MORAI Dataset Collection Pilot v1 - run the existing real MORAI
pilot on a machine/session with the simulator + gRPC + ROS bridge active,
then export both the CenterPoint dataset and KalmanNet GT trajectories from
the same canonical capture before any learned-model training.

---

## CenterPoint MORAI Evaluator v1 — COMPLETE (evaluator implementation only)

Branch `feat/centerpoint-morai-evaluator-v1`, from merged PR #30 main
`ddb0b61211493f419aee38822bcfd1bd89fcdab9` (`feat(perception): prepare
CenterPoint MORAI training (#30)`). **No training, no `optimizer.step()`, no
weights downloaded, no architecture / voxel / range change, no Dataset Factory
or adapter schema change, no tracking / planner change, no real-world
generalization claim, no historical T-14 comparison.**

**What it is:** replaces `MoraiHevenDataset.evaluation()` (was
`NotImplementedError`) with a deterministic, versioned, **CPU-only, torch-free**
LiDAR-only detector evaluator: oriented BEV + 3D IoU AP / precision / recall at
`{0.25, 0.50, 0.70}`, matched centre / dimension / yaw diagnostics, and
BEV-range-binned recall. No AB3DMOT / HOTA / IDSW / KalmanNet / planner.

**PR #30 verification (squash-aware):** `gh pr view 30` → state `MERGED`, merge
commit `ddb0b61`; `origin/main` contains `preflight_morai_training.py`, the
`centerpoint_morai_v1*` configs and the training-prep doc.

**MORAI Dataset Collection Pilot still blocked** (separate task): no MORAI
simulator / gRPC / ROS bridge in this environment → no real
`morai_tracking_dataset_v1`, no real adapter export. This task removes the
*second* blocker (the missing evaluator) at the software level only.

**OpenPCDet contract (audited, not guessed):**
`references/openpcdet/tools/eval_utils/eval_utils.py` calls
`dataset.evaluation(det_annos, class_names, eval_metric=..., output_path=...)`
→ `(result_str: str, result_dict: {str: float})`; `result_dict` is folded into
`ret_dict` for tensorboard + checkpoint selection. `det_annos` = list, one dict
per sample (`name` `(N,)` str, `score` `(N,)`, `boxes_lidar` `(N,7)` =
`[x,y,z,l,w,h,yaw]` lidar-frame geometric centre, `frame_id` scalar
`numpy.str_`). The wrapper recovers GT from `self.morai_core`, aligns by
`str(frame_id) == sample_id` (never list position), and ignores the
`eval_metric` / `output_path` kwargs.

**Core** `tools/centerpoint_offline/morai_evaluator.py` (pure NumPy):
`MORAI_EVALUATOR_SCHEMA_VERSION = "morai_centerpoint_eval_v1"` (exported
constant Training Prep reads), `IOU_THRESHOLDS = (0.25, 0.50, 0.70)`,
`REFERENCE_MATCH_METRIC/THRESHOLD = "3d" / 0.50`,
`PRIMARY_VALIDATION_METRIC = "vehicle/3d_ap@0.50"` (**engineering**
checkpoint-selection metric for MORAI v1, not a universal standard; **no
checkpoint selected**). Vehicle-only. Range bins `hypot(x,y)` ≤ 20 / ≤ 45 / >
45 m — identical to `centerpoint_adapter.py`.

**Geometry:** oriented BEV IoU + 3D IoU built on `_bev_corners` /
`_polygon_clip` (Sutherland-Hodgman) / `_polygon_area` (shoelace) /
`_height_overlap` — **ported verbatim with attribution from
`ad_lidar_perception/ad_lidar_perception/ab3dmot_geometry.py`** (itself a port
of `AB3DMOT_libs/dist_metrics.py`), so the evaluator is one self-contained
file. A test cross-checks the ported clip against the live `ab3dmot_geometry`.
KITTI's R40 evaluator is camera-frame-specific and deliberately not reused.
Verified: identical → exactly 1.0, touching edge → exactly 0.0, yaw π/2 (4×2
dims) → 1/3, z-separated → BEV > 0 & 3D = 0, yaw wrap +179°/−179° → ~0.035 rad.

**AP:** global score-desc sort, tie-break `(sample_id, pred_index)`; **COCO
rule — the IoU threshold gates the candidate set** (a below-threshold "best"
consumes no GT), so `AP@0.25 ≥ AP@0.50 ≥ AP@0.70` holds by construction; each
GT matched once; **101-point interpolated AP** (VOC monotone envelope). All-FP
→ AP 0; confidence order matters (high-score FP before low-score TP → AP 0.5
vs 1.0).

**Negative / degenerate:** frame with 0 GT = valid negative (predictions there
are FP, counted); **entire split with 0 vehicle GT → `MoraiEvaluatorError`**
(never a misleading AP=1); 0 predictions + GT = valid (recall 0, AP 0);
**invalid GT box → raise** (adapter validator should prevent it); invalid
prediction box → excluded + counted (`invalid_prediction_boxes`); duplicate
prediction on one GT → first TP, rest FP. Diagnostics with 0 reference-TP →
`None` (never fabricated 0).

**`SCORE_THRESH` / NMS audit:** `detector3d_template.post_processing` applies
`score_thresh=SCORE_THRESH` (0.1) inside NMS **before**
`generate_prediction_dicts`, so AP is over post-NMS post-threshold detections —
not a full score sweep. Documented; `SCORE_THRESH` / NMS unchanged.

**Standalone CLI** `tools/centerpoint_offline/evaluate_morai_predictions.py`
(`--dataset --split val --predictions <jsonl> --output`): reads
`heven.offline_detection.v1`-compatible JSONL, invokes
`ad_morai_dataset_validate_centerpoint` and **refuses a leaky export**, default
split `val`, `--split test` needs `--allow-test`, `--split train` needs
`--allow-train`. `DATA_SPLIT: {train: train, test: val}` unchanged.

**Config:** `centerpoint_morai_v1_model.yaml` `EVAL_METRIC`
`morai_not_implemented` → `morai_centerpoint_eval_v1`; `centerpoint_morai_v1.yaml`
+`primary_validation_metric: vehicle/3d_ap@0.50`. `morai_heven_dataset.yaml` /
`morai_centerpoint_{smoke,train}.yaml` untouched.

**Preflight:** new `audit_evaluator()` — `evaluation_metric_implemented` is now
`True` when the evaluator module exports `morai_centerpoint_eval_v1` **and**
the model config's `EVAL_METRIC` selects it. On a fixture:
`EVALUATOR: implemented=True` but `FINAL STATUS: BLOCKED_DATASET` (no real
dataset) — fixture existence never yields `READY`.

**Validation type: schema-faithful temporary fixtures only. NO real MORAI
dataset, NO CenterPoint training, NO T-14 reproduction, NO real-world
generalization claim.** Fixtures via the real Dataset Factory writer +
CenterPoint adapter; predictions synthesised.

**Tests:** `test_centerpoint_morai_evaluator.py` 26 (+1 skipped-guard removed:
the `ab3dmot_geometry` cross-check runs). `MoraiHevenDataset.evaluation()`
integration smoke (venv, real OpenPCDet `DatasetTemplate`): `(result_str, {37
float keys})`, deterministic, `eval_metric` kwarg ignored. Latency: fixture
~1 ms/frame; 100-frame × 3-15-box synthetic ~7.5 ms/frame CPU. Regression:
`test_centerpoint_morai_training_prep.py` 21 (+1 new: evaluator detected,
fixture still `BLOCKED_DATASET`), `test_centerpoint_adapter.py` 23,
`test_dataset_factory.py` 27, `tools/morai_dataset_exporter/test_exporter_core.py`
9, `test_centerpoint_offline.py` 11/12 (`test_training_dataloader_creation`
needs `torch` — pre-existing env gap, unrelated). `pyflakes` + `compileall`
clean.

**Files:** `tools/centerpoint_offline/{morai_evaluator.py,
evaluate_morai_predictions.py, test_centerpoint_morai_evaluator.py}`;
`tools/centerpoint_offline/morai_dataset.py` (`evaluation()` body only);
`tools/centerpoint_offline/preflight_morai_training.py` (+`audit_evaluator`);
`tools/centerpoint_offline/configs/{centerpoint_morai_v1.yaml,
centerpoint_morai_v1_model.yaml}`;
`tools/centerpoint_offline/test_centerpoint_morai_training_prep.py` (+1 test);
`docs/perception/centerpoint_morai_evaluator_v1.md`, this file.

**Recommended next task:** MORAI Dataset Collection Pilot v1 — rerun the
already-defined real-data pilot on a machine/session with the MORAI simulator,
gRPC runtime, ROS bridge, raw LiDAR, ego GT, actor GT and TF active; collect at
least six independent scenario+seed groups, validate synchronization/GT
quality, export a leakage-safe CenterPoint train/validation dataset, then run
the Training Prep preflight and this MORAI evaluator before any full CenterPoint
training.

## CenterPoint MORAI Training Prep v1 — COMPLETE (audit + config + preflight only)

Branch `feat/centerpoint-morai-training-prep-v1`, from merged PR #29 main
`3951ea99349474eee58691e6ea337937b3b65bc7` (`feat(data): add CenterPoint MORAI
data adapter (#29)`). **NO FULL TRAINING. No optimizer step, no backward, no
epoch loop, no fine-tune, no weights downloaded, no OpenPCDet downloaded, no
production detector default / model architecture / AB3DMOT / tracker / planner
change, no frozen T-2…T-16 conclusion touched, no new CenterPoint performance
or generalization claim, no historical-metric comparison.**

**What it is:** a reproducible preflight layer that validates a leakage-safe
`centerpoint_morai_adapter_v1` export against the repo's current
CenterPoint/OpenPCDet contracts (point features, class list, 3D box
convention, point-cloud range, voxel grid, loader), refuses
evaluation-readiness when no independent validation group exists, records
model/data/config/checkpoint/code provenance, audits the PyTorch/CUDA/OpenPCDet
environment, and emits the future train/validate commands — **without training**.

**PR #29 verification (squash-aware):** `gh pr view 29` → state `MERGED`,
merge commit `3951ea9`; `origin/main` contains
`centerpoint_adapter{,_cli,_validate}.py`, `class_map.yaml`, the leakage-safe
split logic, and `ADAPTER_SCHEMA_VERSION = "centerpoint_morai_adapter_v1"`.

**New experiment configs** (`tools/centerpoint_offline/configs/`,
self-contained, no `_BASE_CONFIG_`): `centerpoint_morai_v1.yaml` (experiment
manifest — id, seed, init policy, group minimums, forbidden claims),
`centerpoint_morai_v1_dataset.yaml` (data config,
`EXPECTED_DATASET_VERSION: morai_centerpoint_v1` — a loader-level guard that
makes `MoraiHevenDatasetCore` **raise** on the historical `unversioned_step03`
overlap dataset), `centerpoint_morai_v1_model.yaml` (model config, **vehicle-only
`CLASS_NAMES: [vehicle]` + `CLASS_NAMES_EACH_HEAD: [[vehicle]]`** — architecture
otherwise byte-identical to the existing smoke config). The existing
`morai_heven_dataset.yaml` / `morai_centerpoint_{smoke,train}.yaml` and
`train_morai_centerpoint.py` defaults are **unchanged**.

**Preflight CLI** `tools/centerpoint_offline/preflight_morai_training.py`
(`--dataset --config --output-dir [--assert-real-dataset --init-checkpoint
--attempt-{loader,model,forward}-smoke]`). Refuses an `--output-dir` inside the
repo. Writes atomic `preflight_report.json`. Exit 0 only on `final_status ==
READY` (impossible today). **Never trains.**

**Split gate:** grouping key `(scenario_id, requested_seed)` (fallback
`run_id`) read from `split_manifest.json`. `val` groups ≥ 1 required, else
`evaluation_ready = false` with reason "no independent validation group"; the
preflight **never** falls back to evaluating on train. Small `val` group count
→ warning only. Leakage re-checked by invoking
`ad_morai_dataset_validate_centerpoint` (subprocess; reuses the adapter's own
`check_split_leakage`). `test` split is structurally reserved
(`DATA_SPLIT: {train: train, test: val}` → `splits/test.txt` unreachable
during tuning).

**Honest status separation:** `dataset_status` (structural: contracts + val
group + no leakage) is reported **separately** from `real_dataset_available`
(false unless `--assert-real-dataset` **and** path not under a temp dir). A
fixture reports `dataset_status: READY` but `final_status: BLOCKED_DATASET`.
`final_status` precedence `BLOCKED_CONFIG → BLOCKED_DATASET → BLOCKED_EVALUATOR
→ BLOCKED_ENVIRONMENT → READY`.

**Contract audit result (fixture run):** MODEL CONTRACT **PASS** — point
features `[x,y,z,intensity]`/4 match; box fields + `lidar_link` + no-crop/no-filter
match; range `[-4,-25,-3,100,25,5]` consistent across data cfg / head
`POST_CENTER_LIMIT_RANGE` / export metadata; voxel `[0.125,0.125,0.2]` → grid
**exactly** `[832,400,40]`, `grid_x/grid_y % FEATURE_MAP_STRIDE(8) == 0` → BEV
map `[104,50]`, divisible by `BACKBONE_2D` strides `[1,2]`; `vehicle → class id
1`. `config_fingerprint` (SHA-256 over resolved
`{experiment,data,model}`) `b9451d6d6c88bfc90f466d96a87291c4428aaa0b9b7af1b2b3536d9ccf6abe4b`.

**Environment audit:** system `python3` has no torch → `environment_ready:
false`. `~/venvs/heven-centerpoint` has torch `2.1.2+cu118`, CUDA **true**, RTX
4060, `spconv.pytorch`, `pcdet 0.6.0+233f849` (resolved from
`~/projects/OpenPCDet/pcdet`, a separate `-e` checkout at the same pinned
commit as the `references/openpcdet` submodule) → `environment_ready: true`.
The training environment is the one thing **not** blocking.

**Model smoke (venv, fixture export):** `CenterPoint` constructs, **7 757 225**
params, on `cuda`; `DataProcessor` → `voxels (M,5,4)` / `voxel_coords (M,3)`;
`collate_batch` → `voxels`+`voxel_coords`+`points`+padded `gt_boxes (B,K,8)`;
one `torch.no_grad()` forward → `pred_boxes [N,7]` (the 3→1 head reduction
yields correct shapes). **No backward, no optimizer.step, no metric.**
Negative (empty-GT) frame loads to `gt_boxes (0,7)` without error.

**Checkpoint audit:** `centerpoint_t14_reproduction.pth` (SHA-256
`466c8181…dbc95`, 93 474 618 B, T-14 reproduction epoch 3/iter 5292 on the
overlapping 1764/0/0 split) has a **3-class** heatmap head (`hm.1.weight
[3,64,3,3]`) — **strict `state_dict` load into vehicle-only v1 fails by
construction**. Init policy: `train_from_scratch`. Reusable as an
initialization artifact only, never as evaluation evidence.

**Blocked (two independent reasons):** (1) no real leakage-safe dataset — no
`morai_tracking_dataset_v1` and no `centerpoint_morai_adapter_v1` export on
disk (only pytest `tmp_path`); the only on-disk MORAI detection data is the
historical `~/datasets/morai_heven` 1764-train / 0-val / 0-test 100%-overlap
set; (2) no MORAI evaluation metric — `MoraiHevenDataset.evaluation()` raises
`NotImplementedError`. `training_started: false`, `optimizer_step_executed:
false`.

**Guards documented:** no comparison to the historical `0.368 m` (or any T-14
figure) until an independent val split + matching metric/protocol exist; no
generalization claim; simulator→real VLP-16 domain gap explicitly unaddressed.

**Tests:** `test_centerpoint_morai_training_prep.py` 20 (data / config /
environment gates; system `python3` + `pytest`). Regression:
`test_centerpoint_adapter.py` 23, `test_dataset_factory.py` 27,
`tools/morai_dataset_exporter/test_exporter_core.py` 9,
`test_centerpoint_offline.py` 11/12 (`test_training_dataloader_creation` needs
`torch` — pre-existing env gap, unrelated). `pyflakes` + `compileall` clean.
The venv lacks `pytest`; the model/forward smoke is verified by direct
`preflight_morai_training.py` CLI runs under the venv.

**Files:** `tools/centerpoint_offline/preflight_morai_training.py`,
`tools/centerpoint_offline/configs/{centerpoint_morai_v1.yaml,
centerpoint_morai_v1_dataset.yaml, centerpoint_morai_v1_model.yaml}`,
`tools/centerpoint_offline/test_centerpoint_morai_training_prep.py`,
`docs/perception/centerpoint_morai_training_prep_v1.md`, this file.

**Recommended next task:** no real, non-fixture `centerpoint_morai_adapter_v1`
export exists (val == 0, no independent groups), so — **MORAI Dataset
Collection Pilot v1**: run a small real MORAI capture campaign with the
Tracking Dataset Factory, collect multiple independent `scenario+seed` groups,
validate synchronization / GT quality, export them through the CenterPoint
adapter, and produce the first non-overlapping train/validation dataset before
any CenterPoint training.

## CenterPoint MORAI Data Adapter v1 — COMPLETE (data adaptation only)

Branch `feat/centerpoint-morai-data-adapter-v1`, from merged PR #28 main
`8bb25f938a2d2b187b947e68217107e911dce4e2` (`feat(data): add MORAI tracking
dataset factory (#28)`). **Dataset adaptation only. No CenterPoint training,
no weights downloaded, no model architecture / config / production detector
default / AB3DMOT / tracker / planner change, no frozen benchmark conclusion
revisited, no simulator-to-real or generalization claim.**

**What it is:** deterministically converts detection-valid frames of a
canonical `morai_tracking_dataset_v1` into the exact LiDAR dataset contract
read by this repo's CenterPoint integration
(`tools/centerpoint_offline/morai_dataset.py::MoraiHevenDatasetCore` — a
torch-free **per-sample-JSON** loader, **not** an OpenPCDet `.pkl` info
file; no `infos/`, no KITTI camera metadata, because the loader reads
neither). New modules
`ad_morai_bridge_dev/ad_morai_bridge_dev/dataset/{centerpoint_adapter.py,
centerpoint_adapter_cli.py,centerpoint_adapter_validate.py}`. Entry points:
`ad_morai_dataset_export_centerpoint`, `ad_morai_dataset_validate_centerpoint`.

**Audited target contract** (from `morai_dataset.py` + `configs/morai_heven_dataset.yaml`
+ `morai_centerpoint_smoke.yaml`): class names `("vehicle","pedestrian","obstacle")`
(order = id-1); box `("x","y","z","length","width","height","yaw")` 7-dim,
geometric centre, `lidar_link`, yaw CCW +X `[-π,π]`; points `Nx4` little-endian
`float32` `[x,y,z,intensity]`; `POINT_CLOUD_RANGE [-4,-25,-3,100,25,5]` applied
by the model's `DATA_PROCESSOR` (not the loader/adapter). A parity test asserts
`TARGET_CLASS_NAMES`/`TARGET_BOX_FIELDS` still equal the loader's.

**Canonical never modified.** Output tree: `<root>/{.centerpoint_morai_adapter,
export_manifest.json, metadata.json, sample_mapping.jsonl, split_manifest.json,
splits/{train,val,test}.txt, labels/<id>.json, points/<id>.bin}`.

**Conversions (all explicit + tested):**
- box centre / dims copied **verbatim** from the factory's already-derived
  `box.lidar_frame` — the adapter recomputes **no** `map→lidar` transform.
  **No z shift** (`±h/2`): the factory `lidar_frame.center` is already the
  geometric centre, matching the loader's `_identity_box_delta`.
- yaw wrapped to `[-π,π]` via `atan2(sin,cos)` (identity on factory output);
  tested at 0, ±π/2, near π, wrapped 3.0 with sin/cos parity.
- dims `length,width,height` in that order; explicit no-l/w-swap test.
- intensity: **identity** (no /255, clip, normalize) — `metadata.intensity_transform`.
- points: `[x,y,z,intensity]` only; `time`/`ring` stay in the canonical npz;
  non-finite points **dropped** (loader asserts finite) — per-sample
  `source_point_count`/`finite_count`/`nonfinite_dropped`, so the derived
  `.bin` is **not lossless**. No point-range cropping; outside-range boxes
  counted not deleted.

**Class map** (`config/dataset_factory/centerpoint_adapter/class_map.yaml`):
**vehicle-only v1** (`{vehicle: vehicle}`). An adapter-unmapped source box is
**omitted from the frame** (frame still exported) and counted in
`counts.boxes_omitted_by_source_class` — never relabelled. "Adapter-unmapped"
≠ "factory-unmapped" (the factory already invalidates `CLASS_UNKNOWN` frames);
the omit path only ever hits factory-mapped `pedestrian`/`obstacle`.

**Frame eligibility:** `valid_for_detection_gt == true` only — the adapter
never loosens the factory criteria. Excluded frames tallied by sorted-flag
reason. **Negative frames** (detection-valid, zero mapped boxes) exported by
default with empty `boxes` (`counts.negative_frames`).

**Split — leakage prevention (first-class):** grouping key
`(scenario_id, requested_seed)` from `run_manifest.config.seed`, fallback
`run_id`. **Frames of one run are never split.** Explicit `--split-plan`
(priority; double-listed group / absent-referenced group → hard error;
unreferenced source group → dropped + warning) or deterministic **SHA-256**
auto-split (`0.7/0.15/0.15`, never process-salted `hash()`). `<3` groups →
all-train + warning; a single run is never frame-split to fake val/test.
`check_split_leakage` (run in `export` and the standalone validator) fails on
any group/run spanning splits or duplicate sample id. This directly prevents
the prior 1764-train / 0-val / 100%-overlap failure mode.

**Provenance:** `sample_id` = factory `sample_id` (`<run_id>_<stamp>`);
`sample_mapping.jsonl` + label `source` block + manifest
`source_dataset_manifest_sha256` + `adapter_repository_commit`.
`content_fingerprint` = SHA-256 over adapter schema + source manifest sha +
canonical adapter config + canonical split manifest + sorted per-artifact
hashes (`created_at` excluded). Repeat-export bit-identical test.

**Real source dataset: NO** (no `morai_tracking_dataset_v1` on disk; MORAI
absent). **Full OpenPCDet `DatasetTemplate`: NOT run** (torch/CUDA
env-blocked). Validated by 22 tests (`test_centerpoint_adapter.py`): repo
loader-**core** parity + `MoraiHevenDatasetCore` loading the export
(runtime-proven), box yaw/z/dim conversions, point features + non-finite
drop, unknown-class omit, invalid-detection exclusion, negative frame,
explicit + auto + tiny-dataset split, leakage validator catches deliberate
overlap, deterministic repeat export, wrong-source-schema rejection.
Source datasets built through the **real** factory writer with 6
`(scenario, seed)` groups, pytest `tmp_path` only.

**Regression:** `test_dataset_factory.py` 27, `test_centerpoint_adapter.py`
22, `tools/morai_dataset_exporter/test_exporter_core.py` 9 — pass.
`tools/centerpoint_offline/test_centerpoint_offline.py` 11/12
(`test_training_dataloader_creation` needs `torch` — pre-existing env gap,
unrelated). `colcon build --packages-select ad_morai_bridge_dev` still
can't complete here (deps not built — pre-existing); `find_packages` +
`py_compile` + entry-point import + config parse verified.

**Size:** derived `.bin` `N×16` bytes → ≈ 0.22 MB/frame, ≈ 0.22 GB/1000
frames (vs ~0.64 MB canonical npz; ~1/3, 4 of 6 fields, no zip).

**Files:** `ad_morai_bridge_dev/ad_morai_bridge_dev/dataset/{centerpoint_adapter.py,
centerpoint_adapter_cli.py,centerpoint_adapter_validate.py}`;
`ad_morai_bridge_dev/config/dataset_factory/centerpoint_adapter/{class_map.yaml,
split_plan.example.yaml}`; `ad_morai_bridge_dev/test/test_centerpoint_adapter.py`;
`ad_morai_bridge_dev/setup.py` (+2 entry points, +config data_files);
`docs/perception/centerpoint_morai_data_adapter_v1.md`, this file. Canonical
dataset factory code unchanged; `tools/centerpoint_offline/` unchanged.

**Recommended next task:** CenterPoint MORAI Training Prep v1 — audit the
repository's current CenterPoint model / config / checkpoint path against the
new leakage-safe adapter output, create a reproducible training/validation
experiment configuration and preflight dataset/model compatibility without
starting full training yet; preserve the simulator-to-real domain-gap caveat
and do not compare against the frozen historical CenterPoint result until a
non-overlapping validation split actually exists.

---

## MORAI Tracking Dataset Factory v1 — COMPLETE (data infrastructure only)

Branch `feat/morai-tracking-dataset-factory-v1`, from merged PR #27 main
`50b1874c9905eb439a41bb0a3f3bd8f66c7e6ddf` (`test(planning): validate highway
merge end to end (#27)`). **Data infrastructure only. No model trained, no
production detector/tracker/planner default changed, no frozen T2–T15 /
KalmanNet / CenterPoint claim touched, no planner behaviour altered.**

**What it is:** a deterministic scenario / reset / capture pipeline that records
one common raw + ground-truth source for later tracking evaluation, CenterPoint
data export and KalmanNet trajectory extraction. New module tree
`ad_morai_bridge_dev/ad_morai_bridge_dev/dataset/` (next to `scenarios/`,
reusing `scenarios.reset` / `scenarios.setup` / `simulator_grpc.client` — no
second reset framework). Entry points: `ad_morai_dataset_capture`,
`ad_morai_dataset_batch`, `ad_morai_dataset_validate`, `ad_morai_dataset_summary`.

**Sample anchor:** each accepted raw `/ad/sensors/lidar/points` frame =
one sample, keyed by its `header.stamp` (never receipt time). Ego GT
(`/ad/dev/vehicle/ego_status`, `EgoVehicleStatus` dev, simulator-native),
actor GT (`/ad/dev/objects`, `ObjectStatusArray` dev) and dynamic TF
(`odom→base_link`) are each matched by **nearest source `header.stamp`, no
interpolation**, with an explicit max skew (default 30 ms each). Out-of-skew /
absent → `skew_exceeded` / `missing` recorded, never substituted. Non-positive /
duplicate / backward LiDAR stamps rejected + counted.

**GT identity:** native `ObjectStatus.unique_id` (never index / nearest / order).
**3D box:** dims from `ObjectStatus.size`; centre policy is a **direct port of
the audited offline exporter** (`_transform_box`: vehicle rear-axle-ground →
box-centre forward+up shift with the `Σ(overhang,wheelbase,rear_overhang)` length
check; pedestrian/obstacle ground-centre → up only), locked by a parity test.
No arbitrary z offset. A bad per-actor geometry flags that actor and clears its
`valid_for_detection_gt`; the frame is still written (Phase 23).
**Frames:** canonical GT in `map`; derived `lidar_link` box when the full
`map→odom→base_link→rear_axle_link→lidar_link` chain is valid (`map→odom`
derived per frame; `odom==map` never assumed).
**Class map:** `{0:pedestrian,1:vehicle,2:obstacle}` version
`checkpoint14_evidence_v1` — evidence inherited from one recorded scenario;
`raw_object_type` always kept, unmapped → `CLASS_UNKNOWN` (never dropped), each
`gt/*.json` records the justifying scenario file + SHA-256. Starter scenarios
are vehicle-only.

**Point storage:** `lidar/NNNNNN.npz`, one named array per `PointField`, dtype
preserved (x/y/z/intensity/time `f4`, MORAI `ring` `u2`). Content-lossless, not
byte-deterministic (zip container). ≈ 0.64 MB/frame → 0.64 GB/1000 frames →
~23 GB/hour @ 10 Hz.

**Validity:** `valid_for_tracking_gt` (matched in-skew actor GT + ≥1 actor with
id/pose/orientation/velocity); `valid_for_detection_gt` (+ ego + TF matched +
every box a finite positive mapped-class `lidar_frame` geometry). The dataset is
not "CenterPoint-ready" just because clouds exist.

**Determinism:** MORAI seed guarantee unknown → `simulator_determinism_guaranteed:
false` in every run manifest; `scenario_seed` still deterministically derives
bounded placement/velocity jitter (±4 m / ±1 m / ±1.5 m/s) so `(scenario, seed)`
is reproducible factory-side.

**Layout:** `<root>/{dataset_manifest.json, schema.json,
runs/<scenario_id>/<run_id>/{run_manifest.json, frames.jsonl, lidar/, gt/, ego/,
tf/}}`. `run_id = <scenario_id>__seed_<seed>__run_<NNN>`, refuses existing unless
`--overwrite`. Atomic `.tmp`→rename per artifact; `frames.jsonl` row appended +
fsynced only after all artifacts land. Interrupt → run status
`aborted`/`error` + real `termination_reason`, never `complete`.

**Starter catalog** (`config/dataset_factory/scenario_catalog.yaml`, 6, MORAI
`egoVehicle`/`vehicleList` JSON parsed by the existing `load_reset_plan`):
`lead_constant` (mid), `lead_brake` (near), `cut_in` (near), `crossing` (mid),
`occlusion_reappear` (mid), `dense_multi_object` (mixed) — exercise persistent
ID, velocity change, birth/death, crossing, crowding, near/mid/far. **Poses are
map-unvalidated templates** grounded on the `kcity-highway` actor preset;
`occlusion_reappear` is an intent name, not a per-actor occlusion label.

**Real MORAI capture: NO** (`import grpc` fails; simulator absent; GT producer
OOMs this host). Validated by 26 unit/dry-run tests
(`ad_morai_bridge_dev/test/test_dataset_factory.py`: schema, lossless point
round-trip, box centre-policy parity vs the exporter, 10 timestamp/skew cases,
writer + atomic partial-frame, run-collision refusal, interrupted-run status,
full capture-session→validator→summary on synthetic messages, catalog load +
seed reproducibility + jitter bounds) + `--dry-run` / `--list-scenarios` /
batch `--dry-run` / validator / summary CLIs end-to-end. Regression:
`tools/morai_dataset_exporter/test_exporter_core.py` 9/9 (untouched),
`test_perception_bag.py` 20/20, `test_raw_streams.py`, `test_actor_presets.py`
pass. Pre-existing env-blocked (`google.protobuf` / `ad_morai_interfaces_dev`
not installed here): `test_scenario_reset`, `test_scenario_setup`,
`test_dev_timestamp_contract` — unchanged by this work. `colcon build
--packages-select ad_morai_bridge_dev` cannot complete in this repo's
`install/` tree (deps `ad_morai_interfaces_dev`/`fast_lio`/`ad_localization`/
`ad_morai_bridge` not built here — pre-existing, unrelated to this change).
Verified instead: `find_packages` resolves `ad_morai_bridge_dev.dataset`, all
15 modules `py_compile` clean, the 4 entry-point targets import (incl.
`capture_node` with ROS sourced), and every scenario/plan config parses.

**Files:** `ad_morai_bridge_dev/ad_morai_bridge_dev/dataset/` (`__init__`,
`schema`, `geometry` [verbatim copy of the exporter's], `pointcloud`, `boxes`,
`sync`, `transforms`, `frame_builder`, `writer`, `capture`, `capture_node`,
`capture_cli`, `batch`, `validate`, `summary`);
`ad_morai_bridge_dev/config/dataset_factory/` (`scenario_catalog.yaml`,
`dataset_plan.example.yaml`, `scenarios/*.json` ×6);
`ad_morai_bridge_dev/test/test_dataset_factory.py`;
`ad_morai_bridge_dev/setup.py` (+4 entry points, numpy dep, config data_files),
`ad_morai_bridge_dev/package.xml` (+`python3-numpy`, +`tf2_msgs`);
`docs/perception/morai_tracking_dataset_factory_v1.md`, this file. `.gitignore`
unchanged — `datasets/` is already ignored; test fixtures are synthesised
in-test, no LiDAR frame committed. README not edited (CRLF-dirty; quick-start
is in the doc + `--help`).

**Recommended next task:** CenterPoint MORAI Data Adapter v1 — consume the
validated dataset factory schema and export deterministic
CenterPoint/OpenPCDet-style LiDAR samples, class labels, 3D boxes and
train/validation manifests without training the detector yet; explicitly
preserve simulator-to-real domain-gap caveats.

---

## Highway Merge End-to-End Execution Validation v1 — COMPLETE (planning-side milestone closed)

Branch `test/highway-merge-end-to-end-v1`, from merged PR #26 main
`730fd4941a6de241f3d0cbb477c042723f757dc7` (`feat(planning): add highway merge
reference path (#26)`). **Test harness + validation + observability + docs
only. No new algorithm, no policy retune, no new planner / mission behaviour,
no code fix — the composed test found zero bugs.**

**Boundary:** planning-side end-to-end from a synthetic `DynamicObjectRiskArray`
(tracking / prediction / Dynamic Object Risk producer NOT run — own runtime
validation, real producer OOMs this host; PR guidance says the merge chain's
canonical boundary is `DynamicObjectRisk`). Chain is **connected not mocked**:
real `ad_highway_merge_gap_risk` -> real `ad_highway_merge_gap_response` ->
real `ad_planner` (response integration + mission + online reference path all
on, **PRODUCTION base path** `2026_molit_comp_global_path.txt`) -> existing
FollowGlobalPath -> existing PathTrackingController(s) -> the one
`/ad/control/command` publisher. The test never publishes
`HighwayMergeGapResponse` / `highway_merge_authorized` for the main chain and
never recomputes a policy equation — it drives inputs and asserts composed
outputs. **Real MORAI run: NO** (no simulator / grpc here).

**Deterministic traffic script (steps A-J), real `route:0:left:1` ego poses:**

| step | ego s | traffic | GapResponse | reason | merge_authorized | mission | reference_active | merge cap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A unsafe rear | 1120 | fast rear+front | WAIT | REAR_CLOSING | false | WAITING | true | -1 |
| B gap opens | 1150 | zero relevant | MERGE_READY | CLEAR_GAP | **true** | AUTHORIZED | true | -1 |
| B2 auth @ pose A | 1120 | zero | MERGE_READY | CLEAR_GAP | true | AUTHORIZED | true | -1 |
| C pre-commit revoke | 1160 | fast rear | WAIT | REAR_CLOSING | **false** | WAITING | true | -1 |
| D reauthorize | 1180 | zero | MERGE_READY | CLEAR_GAP | **true** | AUTHORIZED | true | -1 |
| E commit | 1220 | zero | MERGE_READY | CLEAR_GAP | true | **COMMITTED** | true | -1 |
| F post-commit revoke | 1250 | fast rear | WAIT | PREDICTED_ROUTE_CONFLICT (6) | **false** | COMMITTED (held) | true | **10.36 m/s** |
| F2 post-commit HOLD | 1283 | fast rear | **HOLD** | ALONGSIDE (5) | false | COMMITTED (held) | true | **0.0** |
| G complete | route:0 1295 | zero | inactive | NONE | false | **COMPLETE** | **false** | -1 |
| H release | route:0 1330 | zero | inactive | NONE | false | **INACTIVE** | false | -1 |
| I cut-in HOLD during merge | 1120 | fast rear + cut-in HOLD | WAIT | REAR_CLOSING | false | WAITING | true (+ braking) | -1 |
| J reset backward | route:0 400 | zero | inactive | NONE | false | INACTIVE (latch cleared) | false | -1 |

**Causal linkage is real** (each arrow a separate node): fast rear ->
`REAR_CLOSING` -> `WAIT` -> `highway_merge_authorized` false -> mission
`WAITING`; zero relevant -> `CLEAR_GAP` -> `MERGE_READY` ->
`highway_merge_authorized` true -> mission `AUTHORIZED`. Mission state trace
A-H `[2,3,3,2,3,4,4,4,5,0]`; reference-active A-H
`[T,T,T,T,T,T,T,T,F,F]`; authorization A-H `[F,T,F,T,T,F,F,F,F,F]`.

**Key findings:**
- **Pre-commit auth loss** (C): `AUTHORIZED -> WAITING`, `reference_active`
  stays true, no lateral swap. B2 vs A at the identical ramp pose: mean
  |steering| 0.0038 vs 0.0038 -> **authorization does not change the lateral
  command**.
- **Post-commit auth loss** (F, F2): mission stays `COMMITTED`,
  `reference_active` stays true -> lateral merge not reversed; **independent
  longitudinal safety still bites** -- F applies `10.36 m/s`
  (`= sqrt(2*1.8*available)`) and F2 applies `0.0` (braking through the
  existing external-constraint / controller path, no brake published by merge
  code). Mandatory cross-layer separation validated.
- **COMPLETE handoff** (E -> G): mean |steering| 0.0084 -> 0.0026 rad, ego
  lateral offset to route:0 3.23 -> 0.13 m; seamless because the
  merge-reference tail past merge-complete IS route:0 and the production
  controller is kept warm every tick. Empirical continuity: |steering| stays
  < 0.02 rad, command valid every frame.
- **Cut-in composition** (I): a cut-in `HOLD` during a ramp `WAITING` frame
  still brakes -- highway-merge state does not lift a stricter cut-in cap.
- **Reset mid-merge** (J): mission -> INACTIVE, latch clears, production path,
  no stale COMMITTED / authorization / reference.

**Second-controller audit (PR #26), confirmed:** two internal
`PathTrackingController` instances while enabled (production + merge-reference);
no duplicate subs/pubs; PID / route-progress / launch-ramp state duplicated **by
design**; **same** backend / gains / speed profile / PID (both via
`make_path_tracking_controller` + `RosControllerParameterProvider`); the
production controller is **warm-updated every active tick** (result discarded
while merge reference selected) so the COMPLETE handoff never re-triggers the
launch ramp. **One control publisher, one selected `ControllerResult` per tick,
no blending.**

**Timing (loopback, not internal compute):** risk publish -> gap-risk frame
median 6.15 / p95 ~6.8 / max ~8.2 ms; -> gap-response median 6.25 / p95 ~6.9 /
max ~8.4 ms (both modes identical). publish -> next command median ~43 ms
(driver cadence + ~20 Hz planner tick dominated) -- **reference-active vs
reference-inactive statistically identical** -> the PR #26 doubled `update()`
costs nothing measurable. E2E DynamicRisk -> command latency **not measured**
(4-node async correlation ambiguous; not invented). ~1180 commands over the
run, no missed cycles.

**Metrics** (frame counts are settle-dwell dependent, ~+/-1% run to run;
traces + qualitative counts are deterministic and asserted): ~2700 gap-risk
frames, ~2680 responses, ~1180 commands, ~1145 MERGE_READY, ~580 WAIT, ~145
HOLD, ~465 authorization-true frames, ~735 reference-active frames, **0 invalid
commands, 0 NaN/Inf/exceptions** (asserted), 1 `/ad/control/command` publisher
(asserted).

**Stale response (documented, unchanged):** pre-commit stale -> auth false ->
`WAITING`, merge reference retained while on ramp + `WAITING`, merge cap
expires; post-commit stale -> auth false, mission stays `COMMITTED`, reference
active, merge cap expires. **Known limitation retained: a stale `HOLD` expires
rather than latching** (downstream assumption since PR #22). Not retuned.

**Traffic-present MERGE_READY (Phase 9):** a ramp ego at 8 m/s vs mainline at
24-33 m/s -> any in-window mainline vehicle overtakes within the horizon and
correctly trips `WAIT` (PR #25 finding). A "safe" traffic frame would need
mainline speeds ~8 m/s (not a highway) or vehicles outside the relevance window
(= zero relevant objects). The **zero-relevant-object `CLEAR_GAP` frame is the
realistic traffic-present-but-clear case** and is what the positive steps use.
No threshold changed.

**Bugs found / fixed: none.** The composed chain behaved exactly as PR #21-#26
specify on the first composed run.

**Tests:** `test_highway_merge_end_to_end.py` (new composed launch test, ~60 s;
full A-J sequence, F2 HOLD, cut-in composition, mid-merge reset, timing block,
Phase-35 state/reference/authorization trace assertions). Regression:
`test_highway_merge_reference_path` 10 + `_golden` 6, `test_highway_merge_mission`
27, `test_highway_merge_speed_constraint`, `test_external_speed_limit`,
`test_cut_in_speed_constraint`, `test_roundabout_speed_constraint`,
`test_behavior_tree` (unchanged, `registered_node_ids().size() == 8`),
`test_planner_launch` 30 -- all pass. Highway-merge launch-test regression
(`ctest`): `test_planner_highway_merge_mission`,
`test_planner_highway_merge_reference_path`,
`test_highway_merge_ego_ramp_pipeline`, `test_highway_merge_gap_risk_runtime`,
`test_highway_merge_end_to_end` -- 5/5 pass. `ad_control` (Stanley /
Profile-Stanley / PID) **byte-unchanged** (`git diff origin/main -- ad_control`
empty; no `test_stanley` run needed -- controller math untouched by
construction). No `ad_interfaces` change. Isolated `colcon build
--packages-select ad_planner --symlink-install` clean.

**Files:** `ad_planner/test/test_highway_merge_end_to_end.py`,
`ad_planner/CMakeLists.txt`, `docs/planning/highway_merge_end_to_end_validation_v1.md`,
this file.

**PLANNING-SIDE HIGHWAY-MERGE MILESTONE CLOSED** (pending real simulator /
vehicle validation). No further highway-merge planning feature work is
recommended until real MORAI / vehicle testing exposes a concrete issue.

**Recommended next task:** MORAI Tracking Dataset Factory v1 -- deterministic
scenario / reset / data-capture pipeline recording LiDAR, ego pose / TF, actor
GT IDs / poses / 3D boxes, timestamps, scenario seed + manifest for repeated
tracking / CenterPoint / KalmanNet development, without changing the frozen
tracking research claims.

---

## Highway Merge Lateral Reference Path Primitive v1 — COMPLETE

Branch `feat/highway-merge-reference-path-v1`, from merged PR #25 main
`d94b4a16436f50bdfdc65b4863f52b2be34a5286` (`feat(planning): add highway merge
ego ramp scenario (#25)`). **Opt-in, default off**
(`enable_highway_merge_reference_path: false`; also requires
`enable_highway_merge_mission`). Selects the lateral REFERENCE PATH consumed by
the existing `FollowGlobalPath` controller; publishes no steering, adds no
second lateral controller, adds no second `/ad/control/command` publisher,
changes no controller math / gains.

**Design correction honored:** the merge reference is active while a genuine
ramp ego is **WAITING / AUTHORIZED / COMMITTED** — NOT `COMMITTED`-only. Before
commit the production global path is `route:0`, so without this Stanley already
pulls the ramp ego (~3.9 m off `route:0`) toward the mainline. Longitudinal
`WAIT`/`HOLD` (response integration) keeps owning merge permission; this only
selects the lateral centerline. Clean separation: reference path = where the
road centerline is; longitudinal response = whether the ego may progress;
mission = authorization / commitment / completion semantics.

**No synthetic curve:** `route:0:left:1` already tapers 3.94 -> 0.00 m and its
last point coincides with a `route:0` point (0.00 m). The builder is the real
source lane + real `route:0` past merge-complete; no quintic / spline / sigmoid.

**Pure builder** `HighwayMergeReferencePathBuilder`
(`highway_merge_reference_path.{hpp,cpp}` in `ad_planner_core`, no ROS):
`build_highway_merge_reference_path(source_lane, target_lane,
merge_complete_route_s_m, config)` -> `{ad_control::Route (map frame, x/y,
z=0 -- ReferencePoint has no z), source_point_count, target_point_count,
splice_route_s_m, splice_join_gap_m, splice_join_heading_delta_rad}`. SOURCE
section: all 336 `route:0:left:1` points verbatim. TARGET section: `route:0`
points with `route_s` strictly `> merge_complete`, up to `+
target_continuation_m` (200 m -> 400 pts); the coincident merge-complete point
contributed once by the source. Throws on any inconsistency (too few points,
non-finite, station regression, station mismatch > 2 m, discontinuous join >
1 m / 0.20 rad, < 2 target points) -> feature inert. `test_highway_merge_
reference_path` 10 gtests; `test_highway_merge_reference_path.py` 6 golden
tests: the online builder output equals the committed
`test_highway_merge_ego_ramp_path.txt` oracle x/y **point-for-point**, target
section a subsequence of `route:0`, splice == merge-complete not commit.

**Second controller (planner-owned, no ad_control change):**
`configure_highway_merge_reference_path()` (after the mission config) builds the
merge `Route` once and a SECOND `PathTrackingController` via
`make_path_tracking_controller(path_tracking_backend_, merge_route,
RosControllerParameterProvider(*this))` -- **same backend / gains / speed
profile / PID** as production. Soft-fails (feature inert) on build failure.
`run_path_tracking()` now: computes the same combined longitudinal override;
**always** updates the production controller (keeps its route progress / PID /
launch-ramp state warm so the COMPLETE handoff never re-triggers the launch
ramp -- deliberate, not a no-op); if the merge reference is selected and its
controller returns a valid result, `remember()`s that, else `remember()`s the
production result. Runtime merge-controller-invalid -> fall back to the warm
production controller for that tick (throttled WARN); build always spans well
past COMPLETE so this is a narrow guard.

**Activation** `select_highway_merge_reference()` (per tick, deterministic
2-state latch): feature off / build failed / no mission geometry -> production,
clear latch. Mission INACTIVE / APPROACH / COMPLETE -> production, clear latch.
Latched -> merge reference while state in {WAITING, AUTHORIZED, COMMITTED}. Not
latched: `on_ramp && state in {WAITING, AUTHORIZED, COMMITTED}` -> latch + merge
reference. **Source-ramp proximity guard**: `on_ramp` =
`|project_to_frenet(route:0:left:1, ego).d_m| <= reference_path_proximity_m`
(default **1.75 m** = the source lane's own `left_width_m`/`right_width_m` in
the corridor; a `route:0` ego near zone entry is ~3.9 m off -> excluded) AND
source-lane station in `[zone_entry - 1, merge_complete + 1]`. The latch (a
lateral-continuity latch, NOT an authorization latch) releases at COMPLETE /
INACTIVE / APPROACH / traversal reset.

**Commit vs splice (kept distinct):** commit ~1213.28 m = mission
irreversibility only; the reference still follows `route:0:left:1` there
(sep ~3.50 m). Geometric source->target splice = merge-complete **1286.15 m**
(sep 0.00 m). COMPLETE handoff back to the warm production controller is
seamless because the merge-reference tail past merge-complete IS `route:0`
(same corridor points) -- live test records mean |steering| 0.003 rad at the
COMPLETE pose.

**Observability:** `/ad/planner/highway_merge_reference_active`
(`std_msgs/Bool`), published only when enabled. Config:
`enable_highway_merge_reference_path false`, `reference_path_proximity_m 1.75`,
`reference_path_target_continuation_m 200.0`. `planner.launch.py
enable_highway_merge_reference_path:=true` sets the param;
`highway_merge_ego_ramp_scenario.launch.py` enables it (with mission + response
integration).

**Live validation (Phase 36 honored -- base path is PRODUCTION
`2026_molit_comp_global_path.txt`, the online override supplies the ramp
reference):** `test_planner_highway_merge_reference_path.py` -- ego teleported
through real `route:0:left:1` samples. `reference_active`: INACTIVE false /
APPROACH false / WAITING **true** / AUTHORIZED **true** / revoked-pre-commit
WAITING **true** / COMMITTED **true** / post-commit WAIT **true** / COMPLETE
**false** / INACTIVE false. Ego is **3.919 m** off production `route:0` at the
zone-entry ramp pose; feature-ON mean |steering| **0.004 rad** (tracks the
ramp) vs PR #25's feature-OFF ~0.32-0.45 rad at the same poses (Stanley pulling
toward `route:0`). WAITING vs AUTHORIZED at the identical pose: 0.0038 vs
0.0038 rad -- **authorization does not change the lateral command**. One
`/ad/control/command` publisher; 0 non-finite steering; clean shutdown.

**Regression:** `test_highway_merge_reference_path` 10 + `_golden` 6;
`test_planner_launch` +arg +opt-in/default-off (29/29); full `ad_planner`
suite (minus the 4 documented pre-existing host-flaky) all pass incl.
`test_stanley` (controller math untouched -- Phase 43), `test_behavior_tree`
(no BT change -- Phase 51, still `registered_node_ids().size() == 8`), cut-in /
roundabout / highway-merge constraint + risk + response + mission + ego-ramp
scenario. Isolated `colcon build --packages-select ad_planner --symlink-install`
clean.

**Production lock:** `2026_molit_comp_global_path.txt` byte-unchanged (SHA-256
`50658991...cc05`); `planner.yaml` default `path_file` unchanged; no
Stanley / Profile-Stanley / PID gain / steering-saturation / wheelbase change;
no production BehaviorTree change; no `ad_interfaces` change.

**Files:** `ad_planner`
`include/ad_planner/planning/highway_merge_reference_path.hpp`,
`src/planning/highway_merge_reference_path.cpp`,
`src/planner/planner_node.cpp`, `src/planner/planner_ros_interfaces.{hpp,cpp}`,
`config/planner.yaml`, `launch/planner.launch.py`,
`launch/highway_merge_ego_ramp_scenario.launch.py`, `CMakeLists.txt`,
`test/{test_highway_merge_reference_path.cpp,test_highway_merge_reference_path.py,
test_planner_highway_merge_reference_path.py,test_planner_launch.py}`;
`docs/planning/highway_merge_reference_path_v1.md`, this file.

**Recommended next task:** Highway Merge End-to-End Execution Validation v1 --
run the full Dynamic Object Risk -> Merge Gap Risk -> Merge Gap Response ->
longitudinal integration -> merge mission -> online merge reference -> existing
`FollowGlobalPath` controller chain in the ego-on-ramp scenario, validating
`WAIT -> AUTHORIZED -> COMMITTED -> COMPLETE` with deterministic traffic,
preferably in MORAI when available, before any further highway-merge planning
feature.

---

## Highway Merge Ego-on-Ramp Scenario v1 — COMPLETE

Branch `feat/highway-merge-ego-ramp-scenario-v1`, from merged PR #24 main
`27c028e7359dc85d942829d344e94f869e25042a` (`feat(planning): add highway merge
mission primitive (#24)`). Scenario / route-fixture infrastructure only:
**opt-in, no production default changed, no C++ change, no lateral path
generator, no steering / CtrlCmd / brake / throttle / DWA / Frenet / MPPI
change.**

**Why:** PR #24 proved CASE B — the committed competition global path IS
`route:0` (the mainline) and the production ego never drives the acceleration
lane `route:0:left:1` — so `HighwayMergeMissionState` / `HighwayMergeReady` /
`HighwayMergeCommitted` existed with no ego-on-ramp scenario to exercise them.
The recommended-next Lateral Path Primitive has no consumer scenario until one
exists. This task builds it.

**Existing mechanisms reused (Phase 1, no new framework):** (a) ego spawn —
`ad_morai_bridge_dev.scenarios.reset.load_reset_plan` +
`reset_scenario`/`LoadMoraiScenario` gRPC, driven by a MORAI scenario JSON
(`egoVehicle.initPosition.pos/rot` + `initLink`/`initLinkRatio`, `vehicleList`);
(b) route override — the planner's `path_file` / `route_corridor_file` params
(already per-launch overridable; `route_corridor.expected_global_path_sha256`
auto-derived by `planner.launch.py`). No second scenario framework invented.

**Source geometry (Phase 2-4):** source `route:0:left:1` (link `A2256W000409`,
336 corridor pts, `route_s` 1118.7418..1286.1546, lateral offset to `route:0`
tapers 3.94->0.00 m), target `route:0`. **The acceleration lane does not extend
upstream of the merge zone entry on the K-City map** — its first point ==
`route_s_zone_entry_m`. So the ego start is `route:0:left:1` point 0 verbatim
`(66.0109296059, 256.5335690919, 28.3102668207)` yaw `-1.568021` rad
(`-89.8410` deg), link `A2256W000409` ratio 0, = route:0 station ~1118.74,
~3.94 m off `route:0`. INACTIVE(far)/APPROACH are represented with the ego on
`route:0` (the mainline approach); every WAITING..COMPLETE pose is a genuine
`route:0:left:1` sample (a documented CASE B consequence).

**Route fixture (Phase 5-8):** `ad_data/path/test_highway_merge_ego_ramp_path.txt`
(validation-only, never a production default) — 736 pts: `route:0:left:1` all
336 pts (source section, the lane centerline itself does the lateral shift) +
`route:0` 400 pts for `route_s (1286.1546, ~1486.15]` (target section). Join:
`route:0:left:1` last pt == `route:0` pt at `route_s` 1286.1546 **exactly**
(0.00 m sep); target section starts at the next `route:0` pt (0.50 m spacing,
0.0015 rad heading change). No teleport, no invented Cartesian waypoint;
regenerated + byte-compared by the geometry test.

**Primary-route projection (Phase 12-13, CRITICAL — no fix needed):** with the
ego on `route:0:left:1` (up to ~3.9 m displaced), `project_primary_route(route:0,
ego)` (pi/2 heading gate; used by the mission primitive) and
`project_to_frenet(route:0, ego)` (used by `ad_highway_merge_gap_risk`) are
**both fully monotonic over all 336 ramp points** with `projected_s - source_s`
in `[-0.10, +0.11] m`. `s_dot` (Frenet `1/(1-kappa*d)`, d~3.9) stays
`[7.66, 8.47]` m/s for a true 8 m/s ego (<6% ETA effect, well inside
tolerances). **No mission-progress / gap-risk architecture change.**

**Gap risk / response / integration (Phase 14-17):** Gap Risk already projects
ego onto `route:0` (full target lane), not a ramp centerline — live pipeline
test records `ego_route_s_m` monotonic, matching source station within 0.4 m,
`ego_merge_timing_valid` true throughout. No risk/response threshold
(`1.5/2.0/3.0` s headways, `6.0` m route gap) retuned. Integration WAIT/HOLD ->
target-speed cap, MERGE_READY -> revocable `highway_merge_authorized`, unchanged.

**Commit / completion physical meaning (Phase 18-19):** derived commit station
~1213.28 m; live planner test confirms the ego pose there is `3.496 m` off
`route:0` (just under the 3.5 m `commit_lateral_separation_m` — genuinely inside
the taper), ~72.9 m committed span before merge completion. At
`route_s_merge_complete_m` 1286.1546 the accel-lane last point coincides with
`route:0` (`0.000 m` sep) and the mission goes COMMITTED -> COMPLETE there.

**Traffic (Phase 10, 23):** deterministic mainline NPCs on `route:0`
(`ad_data/scenarios/kcity_highway_ego_onramp_v1.json` id 101 front `route_s`
~1200 @ 25 m/s, id 102 rear ~1080 @ 25 m/s; t=0 separations 82 m / 39 m; none on
the ego's lane). The stock `kcity-highway` preset NPC (on `route:0:left:1`) is
deliberately NOT reused. Live pipeline cases at ramp station ~1170: no traffic
-> **MERGE_READY** (deterministic); clear gap (front +80 @26, rear -90 @24) ->
WAIT (mainline at 3x ego speed overtakes the slow ramp ego — physically
correct); fast rear (-12 @33) -> not MERGE_READY; 2 s-horizon object -> not
MERGE_READY.

**Deterministic mission state sequence (Phase 28, live, real ramp poses):**
`route:0` s400 -> INACTIVE; `route:0` s1000 WAIT -> APPROACH; ramp s1120 none ->
WAITING; ramp s1150 MERGE_READY -> AUTHORIZED; ramp s1160 WAIT -> WAITING (no
latch); ramp s1180 MERGE_READY -> AUTHORIZED; ramp s1220 MERGE_READY ->
COMMITTED (crossed ~1213, ego 3.50 m off route:0); ramp s1250 WAIT -> COMMITTED
(post-commit revocation NOT honored, `highway_merge_authorized` False); `route:0`
s1295 WAIT -> COMPLETE (ego 0.00 m off route:0); `route:0` s1330 -> INACTIVE.
Steering is pure Stanley path-tracking (ramp poses sit up to ~3.9 m off the
tracked production route by construction); it does not increase across
AUTHORIZED -> COMMITTED and stays far inside the lock (< 0.5 rad). One
`/ad/control/command` publisher throughout; clean shutdown.

**Validation type (Phase 11, 26):** **NOT a live MORAI run** (no simulator /
grpc in this environment). Deterministic geometry/data tests + ROS pipeline
replay. The MORAI scenario JSON is authored to the existing `load_reset_plan`
contract for a future real run.

**Opt-in launch (Phase 21):** `highway_merge_ego_ramp_scenario.launch.py`
(`data_dir:=<abs ad_data>`) starts gap risk + gap response + planner (response
integration + mission on) on the fixture route; generates a fixture corridor at
launch time (byte copy of `route_corridor.json` with only
`source_sha256.global_path` rewritten to the fixture digest — lane geometry
identical). Never included by `planner.launch.py`.

**Production route regression (Phase 22):** `2026_molit_comp_global_path.txt`
byte-unchanged (SHA-256 `50658991…cc05`, == `route_corridor.json`
`source_sha256.global_path`); `planner.yaml` still defaults to it;
`test_highway_merge_ego_ramp_scenario.py` locks all three.

**Tests:** `test_highway_merge_ego_ramp_scenario` **15 pure-data** (fixture
provenance / regeneration / continuity, ego-start on source lane, both
projection monotonicity checks, commit-taper & completion correspondence,
production-path-unchanged, opt-in, manifest/actor consistency).
`test_highway_merge_ego_ramp_pipeline.py` live synthetic DynamicObjectRisk ->
`ad_highway_merge_gap_risk` -> `ad_highway_merge_gap_response`, ramp ego,
`ego_route_s_m` monotonic `[1120.26, 1140.24, 1170.24, 1210.30, 1250.36]`,
4 traffic cases. `test_planner_highway_merge_ego_ramp.py` live `ad_planner`
(mission + integration), full state sequence above, commit/completion physical
checks, one command publisher, clean shutdown. **Full `ad_planner` suite 57/57**
(minus the 4 documented pre-existing host-flaky: `test_mppi_nav2_launch`,
`test_cut_in_response_runtime`, `test_cut_in_risk_runtime`,
`test_frenet_runtime_contract`); cut-in / roundabout / highway-merge constraint
+ risk + response + mission regressions all pass. Isolated `colcon build
--packages-select ad_interfaces ad_planner --symlink-install` clean.

**Files:** `ad_data/path/test_highway_merge_ego_ramp_path.txt`,
`ad_data/scenarios/kcity_highway_ego_onramp_v1.json`,
`ad_data/scenarios/kcity_highway_ego_onramp_v1.manifest.yaml`,
`ad_planner/launch/highway_merge_ego_ramp_scenario.launch.py`,
`ad_planner/test/{test_highway_merge_ego_ramp_scenario.py,
test_highway_merge_ego_ramp_pipeline.py,test_planner_highway_merge_ego_ramp.py}`,
`ad_planner/CMakeLists.txt`,
`docs/planning/highway_merge_ego_ramp_scenario_v1.md`, this file. No `ad_planner`
C++ / `planner.yaml` / production route / BehaviorTree / `ad_interfaces` change.

**Recommended next task:** Highway Merge Lateral Path Primitive v1 — use this
validated ego-on-ramp scenario and the source-grounded `route:0:left:1` ->
`route:0` corridor geometry to generate a merge reference path for the existing
lateral controller, gated by the committed highway-merge mission state
(`HighwayMergeCommitted`), without publishing steering directly or creating a
second `CtrlCmd` publisher.

---

## Highway Merge Mission Primitive v1 — COMPLETE (CASE B)

Branch `feat/highway-merge-mission-primitive-v1`, from merged PR #23 main
`9da4e869aec1d104ee159223b49d775acb65f3aa` (`feat(planning): integrate highway
merge response (#23)`). The explicit mission-state layer between the revocable
merge authorization (`PlannerContext::highway_merge_authorized`, PR #23) and a
future lateral executor. **Opt-in, off by default** (`enable_highway_merge_mission:
false`): disabled => no merge geometry loaded, state permanently INACTIVE, no
publisher, behaviour identical to current main. Commands **nothing** (no
steering / lane change / CtrlCmd / speed).

**PHASE-1 FORK: CASE B.** The committed competition global path does NOT encode
a ramp->mainline merge, and the ego is never on the acceleration lane at all.
Locked by `test_highway_merge_mission_geometry.py` against the real
`ad_data/path/2026_molit_comp_global_path.txt` + `route_corridor.json`: global
path lateral offset to `route:0` (mainline) is **~0.00 m at every station
1080-1320 m** (the global path IS `route:0`), and **~3.9 m from `route:0:left:1`**
(accel lane) at the zone entry, closing to ~0 only where that lane tapers into
`route:0` at merge-complete. Checkpoints 10/11 sit on `route:0`; `kcity-highway`
is `heven-highway-npc` (an NPC on the accel lane, not the ego); the upstream
risk replay sweeps the ego on `route:0`. So there is no lateral transition to
reuse and no lateral maneuver for the ego on this route -- the primitive tracks
mission state + exposes a source/target corridor intent; it generates no path.
If a future route reroutes the ego via the ramp the geometry test fails and the
decision is revisited.

Pure core `highway_merge_mission.{hpp,cpp}` (`ad_planner_core`, no ROS dep).
States `INACTIVE 0 / APPROACH 1 / WAITING 2 / AUTHORIZED 3 / COMMITTED 4 /
COMPLETE 5`. `step_highway_merge_mission(input, geometry)` -- pure, deterministic,
no clock/hidden state. Source-grounded stations on `route:0` derived once at
startup: `approach_entry = zone_entry - mission_approach_window_m` (400);
`zone_entry 1118.74` / `merge_complete 1286.15` (from `highway_merge.json`,
cross-checked <=2 m against `route:0:left:1` in the corridor, same margin the
risk node uses); `commit ~1213.3` **derived** by
`derive_highway_merge_commit_station()` -- the first `route:0:left:1` station
whose lateral separation from `route:0` (`project_to_frenet`) falls below
`commit_lateral_separation_m` (3.5 m = first ~0.44 m / >10 % of taper below the
~3.94 m nominal offset, safely below the ~3.73 m plateau noise; NOT
`merge_standoff_m` which stays owned upstream); `exit = merge_complete +
mission_exit_release_m` (20). Soft-fails to permanently INACTIVE (logged, no
throw) on any geometry mismatch.

Semantics: authorization **revocable before commit** (`AUTHORIZED -> WAITING`
immediately, no latch); **not revocable after commit** (once `COMMITTED`, an
upstream `WAIT`/`HOLD`/stale drives `authorized_now` false but the state stays
`COMMITTED` until `COMPLETE` -- a lateral executor must not reverse a merge
mid-maneuver; `committed` and `authorized_now` are separate outputs). Commit
monotone within a traversal (swept + locked). Crossing `commit` unauthorized
never commits. Stopped ego never advances (state = f(route_s, authorization),
never wall-clock). Sim reset / 2184 m route-loop wrap: a backward ego route_s
jump > `mission_route_s_reset_jump_m` (3.5 = ~2 x control_period 0.05 s x
33.33 m/s) -> INACTIVE, dropping any carried COMMITTED. Route-projection failure
(`project_primary_route` wrapped in try/catch) -> hold previous state, never
advance.

Consumes `PlannerContext::highway_merge_authorized` **only** (never the response
/ risk / DynamicObjectRisk topics; no freshness or gap-policy duplicated). The
mission and response-integration flags are **independent**: mission on +
response integration off -> authorization always false -> mission never leaves
APPROACH/WAITING (correct fail-safe). The mission flag does NOT force-enable the
response integration.

BT exposure: mission state recomputed in `tick()` before `supervisor_->tick()`,
written to `PlannerContext::highway_merge_mission` (`{state, active, committed,
authorized_now}`, decoupled from the Frenet-carrying header). One new read-only
`HighwayMergeCommitted` `BT::SimpleCondition` (SUCCESS iff `committed`; the one
fact `HighwayMergeReady` can't express -- stays SUCCESS through a transient
post-commit auth loss). `ad_bt_node_ids()` 7->8; `test_behavior_tree` expects
`registered_node_ids().size() == 8`. **Production BT XML unchanged** (no
merge/lane-selection path exists to gate; exact-XML assertion still passes).
CollisionRecovery / TrafficStop / PerceptionMission / FollowGlobalPath /
FailSafeBrake priority + authority untouched; COMMITTED disables no safety
system.

Observability: `/ad/planner/highway_merge_mission_state` (`std_msgs/UInt8`, the
state enum), published only when enabled. Config: `enable_highway_merge_mission
false`, `mission_approach_window_m 400`, `commit_lateral_separation_m 3.5`,
`mission_exit_release_m 20`, `mission_route_s_reset_jump_m 3.5`,
`highway_merge_mission_zone_id kcity_highway_onramp`, `merge_geometry_file ""`
(-> package-share `highway_merge.json`). `planner.launch.py
enable_highway_merge_mission:=true` sets the param, forces no other node.

Validation: `test_highway_merge_mission` **27 gtests** (geometry validation,
commit-station derivation, full Phase-30 transition set incl. sim-reset /
small-noise / wrong-zone / no-backward-after-commit sweep / hold-on-invalid,
Phase 31-34 sequences). `test_highway_merge_mission_geometry.py` 3 data tests
locking CASE B + commit boundary ~1213 m / >60 m committed span.
`test_behavior_tree` `registered_node_ids` 8 +
`HighwayMergeCommittedConditionMirrorsMissionCommitFact`. `test_planner_launch`
+arg + opt-in/default-off. `test_planner_highway_merge_mission.py` live
`ad_planner` on the **real** corridor + global path + merge geometry,
`FollowGlobalPath`, mission + response integration on, ego teleported through
`route:0` stations: 400->INACTIVE / 1000->APPROACH / 1150 unauth->WAITING /
1150 MERGE_READY->AUTHORIZED / 1160 WAIT->WAITING (no latch) / 1175
MERGE_READY->AUTHORIZED / 1235 MERGE_READY->COMMITTED (crossed ~1213) / 1255
WAIT->COMMITTED (post-commit revocation NOT honored, `highway_merge_authorized`
False) / 1295 WAIT->COMPLETE / 1330->INACTIVE; steering < 0.2 rad throughout;
one `/ad/control/command` publisher; clean shutdown. Full `ad_planner` gtest
37/37; cut-in / roundabout / highway-merge constraint + risk + response
regressions pass. Isolated `colcon build --packages-select ad_planner
--symlink-install` clean. Pre-existing unrelated host failures unchanged
(`test_mppi_nav2_launch`; `test_cut_in_response_runtime` /
`test_cut_in_risk_runtime` perception OOM; `test_frenet_runtime_contract`
parallel contention).

Files: `ad_planner` `include/ad_planner/planning/highway_merge_mission.hpp`,
`include/ad_planner/behavior/planner_context.hpp`,
`src/planning/highway_merge_mission.cpp`, `src/behavior/bt_nodes.cpp`,
`src/planner/planner_node.cpp`, `src/planner/planner_ros_interfaces.{hpp,cpp}`,
`config/planner.yaml`, `launch/planner.launch.py`, `CMakeLists.txt`,
`test/{test_highway_merge_mission.cpp,test_highway_merge_mission_geometry.py,
test_planner_highway_merge_mission.py,test_behavior_tree.cpp,test_planner_launch.py}`;
`docs/planning/highway_merge_mission_primitive_v1.md`, this file. No
`ad_interfaces` change.

**Recommended next task:** Highway Merge Lateral Path Primitive v1 -- consume
the committed target-corridor intent and generate a source-grounded
ramp-to-mainline reference path for the existing lateral controller, without
publishing steering directly or a second CtrlCmd publisher. **Prerequisite:** on
the committed route the ego is already on `route:0` (CASE B), so a
ramp-to-mainline path has no consumer scenario -- a route/actor-spawn placing
the ego on `route:0:left:1` through the merge zone (or an explicit decision to
model the ego-on-ramp scenario) must come first.

---

## Highway Merge Response Integration v1 — COMPLETE

Branch `feat/highway-merge-response-integration-v1`, from merged PR #22 main
`6ae7f8922bc966e6bd31dd872d209283d9599b11` (`feat(planning): add highway merge
gap response (#22)`). First planner-side consumer of `HighwayMergeGapResponse`.
**Opt-in, off by default** (`enable_highway_merge_response_integration: false`):
disabled => no subscription, no observability publisher, no merge authorization,
behaviour identical to current main.

Consumes `HighwayMergeGapResponse` **only** (never `HighwayMergeGapRisk` /
`DynamicObjectRisk` / raw tracker). Two responsibilities, neither commanding
steering or a lane change:

1. **Longitudinal**: `WAIT` / `HOLD` lower the path-tracking target through the
   existing 6th `target_speed_mps` arg of `PathTrackingController::update()` in
   `run_path_tracking()` (`FollowGlobalPath` leaf) -- the same point cut-in
   (PR #16) and roundabout (PR #20) use. No second longitudinal controller, no
   extra `CtrlCmd`.
2. **Authorization**: a fresh active `ACTION_MERGE_READY` sets a revocable
   `PlannerContext::highway_merge_authorized` fact (recomputed every tick before
   the BT runs, never latched), exposed by a new registered-but-unwired
   `HighwayMergeReady` `BT::SimpleCondition` and a
   `/ad/planner/highway_merge_authorized` (`std_msgs/Bool`) diagnostic.

**No highway merge / lane-change mission path exists in the repo** (BT has no
merge branch; no route-change primitive; Frenet `lane_change_weight` is a local
trajectory-scoring cost, not a mission transition). Phase 21: no mission was
fabricated. `HighwayMergeReady` is registered + in `ad_bt_node_ids()` but the
production tree (`ad_planner.xml`) is byte-unchanged.

Pure core `highway_merge_speed_constraint.{hpp,cpp}` (`ad_planner_core`, no ROS
dep): `highway_merge_response_speed_limit()` -- not received / stale / inactive /
wrong zone / `MERGE_READY` / unknown -> `nullopt`; `HOLD` -> `0.0`; `WAIT` ->
`ego_speed * sqrt(available / comfortable_stop)` == `sqrt(2 * a_comfortable *
available)` (ego terms cancel -- **no deceleration / standoff / gap constant
duplicated**, no new yield-speed param); malformed WAIT facts -> `nullopt`.
`highway_merge_response_merge_authorized()` -- true only when received && fresh
&& active && zone matches && `action == ACTION_MERGE_READY` (the enum-0 trap:
`active` must be checked; `reason` is never branched on).

Zone: `expected_highway_merge_zone` (default `kcity_highway_onramp`) must equal
`response.merge_zone_id` or the frame is ignored wholesale (no auth, no cap),
mirroring the upstream node's wrong-zone rejection. Empty value accepts any zone.

`combine_speed_limits` gained an `initializer_list` fold (binary form unchanged);
`run_path_tracking()` composes `min(nominal, cut_in, roundabout, highway_merge)`
-- order-independent, associative, most-restrictive-wins. `MERGE_READY` never
lifts a cut-in/roundabout cap; `WAIT` never raises nominal; `HOLD` -> `0.0`.

Freshness: steady receipt time only (`highway_merge_response_max_age_s: 0.5`);
header stamp never consulted. Never latched -- a following `WAIT` / `HOLD` /
stale / missing / inactive / wrong-zone frame immediately revokes both the cap
and `merge_authorized`. Stale `HOLD` expires to no constraint (inherited
cut-in/roundabout freshness contract; documented downstream assumption).

Scope: `FollowGlobalPath` (Stanley / profile-Stanley) longitudinal only.
`PerceptionMission` (DWA/Frenet/MPPI) unconstrained (same `HOLD = 0` /
lateral-selection limitation as cut-in/roundabout).
`VehicleConstraints.maximum_speed_mps` unchanged. Exactly one
`/ad/control/command` publisher. No actuator / steering / lane-change / route /
trajectory / `ad_interfaces` change.

`planner.launch.py enable_highway_merge_response_integration:=true` sets the
param and starts `ad_highway_merge_gap_response` + (forced)
`ad_highway_merge_gap_risk` (`_highway_merge_response_requested` now ORs in the
new integration flag).

Validation: `test_highway_merge_speed_constraint` 26 gtests (action mapping,
zone / stale / inactive / enum-0, malformed WAIT facts, cap properties,
authorization + revocation). `test_external_speed_limit` +`initializer_list`
fold + Phase-16 three-source matrix A-G + `AllThreeActive` + full 6-permutation
order-independence over every (cut-in, roundabout, merge) triple + MERGE_READY
never overrides. `test_behavior_tree` `registered_node_ids` 6->7 +
`HighwayMergeReadyConditionMirrorsAuthorizationFact` (false->true->false, no
latch); production-tree exact-XML assertion unchanged. `test_planner_launch`
+declared arg + `test_highway_merge_response_integration_is_opt_in_and_default_off`.
`test_planner_highway_merge_constraint.py` live `ad_planner` in `FollowGlobalPath`,
ego 8 m/s: baseline target 16.25 / limit -1 / auth False; inactive MERGE_READY
enum -1 / False / 16.25; MERGE_READY -1 / **True** / 16.25 (byte-identical to
baseline); WAIT (avail 34) limit 11.063 = sqrt(2*1.8*34) / False / target 11.063;
HOLD (avail 8) limit 0.0 / False / target 0.0 / brake; wrong-zone MERGE_READY -1
/ False / 16.25; re-armed MERGE_READY then silence -> -1 / False / 16.25 (no
latch); steering byte-identical (0.0) every state; 1 `/ad/control/command`
publisher. Full `ad_planner` gtest 36/36; `test_planner_cut_in_constraint` /
`test_planner_roundabout_constraint` / all cut-in/roundabout/highway-merge
risk+response launch + runtime tests pass. Isolated `colcon build
--packages-select ad_planner --symlink-install` clean. Pre-existing unrelated
host failures unchanged (`test_mppi_nav2_launch` nav2 absent;
`test_cut_in_response_runtime` / `test_cut_in_risk_runtime` perception-node OOM;
`test_frenet_runtime_contract` parallel contention -- passes isolated).

Files: `ad_planner`
`include/ad_planner/planning/highway_merge_speed_constraint.hpp`,
`include/ad_planner/planning/external_speed_limit.hpp`,
`include/ad_planner/behavior/planner_context.hpp`,
`src/planning/highway_merge_speed_constraint.cpp`,
`src/planning/external_speed_limit.cpp`, `src/behavior/bt_nodes.cpp`,
`src/planner/planner_node.cpp`, `src/planner/planner_ros_interfaces.{hpp,cpp}`,
`config/planner.yaml`, `launch/planner.launch.py`, `CMakeLists.txt`,
`test/{test_highway_merge_speed_constraint.cpp,test_external_speed_limit.cpp,
test_behavior_tree.cpp,test_planner_launch.py,test_planner_highway_merge_constraint.py}`;
`docs/planning/highway_merge_response_integration_v1.md`, this file. No
`ad_interfaces` change.

**Recommended next task:** Highway Merge Mission Transition v1 -- but first build
the missing mission primitive (a route/corridor change or merge lane-selection
state the `HighwayMergeReady` condition can gate); no executable
merge/lane-selection mission path exists to wire `merge_authorized` into yet.

---

## Highway Merge Gap Response v1 — COMPLETE

Branch `feat/highway-merge-gap-response-v1`, from merged PR #21 main
`605d911c7457ead3d86c8a8e674ca5ef02b62749` (`feat(planning): add highway merge
gap risk (#21)`). New opt-in policy/advisory node `ad_highway_merge_gap_response`:
`/ad/planning/highway_merge_gap_risks` (`HighwayMergeGapRiskArray`) ->
`/ad/planning/highway_merge_gap_response` (`HighwayMergeGapResponse`). Advisory
only: MERGE_READY / WAIT / HOLD. No lane-change command, steering, path,
trajectory, gear, brake, throttle, acceleration/speed request, CtrlCmd,
BehaviorTree change, or production planner consumer. `MERGE_READY` is a fact for
a future mission layer, not "start steering now".

New message `ad_interfaces/msg/HighwayMergeGapResponse.msg` (no reuse of
CutInResponse / RoundaboutGapResponse -- distinct mission semantics). Actions
`ACTION_MERGE_READY=0 / ACTION_WAIT=1 / ACTION_HOLD=2` (monotone: higher =
more restrictive). Reasons `NONE / CLEAR_GAP / FRONT_GAP / REAR_GAP /
REAR_CLOSING / ALONGSIDE / PREDICTED_ROUTE_CONFLICT / INSUFFICIENT_PREDICTION /
INVALID_EGO_STATE` (DECISION_BOUNDARY folded into the HOLD action +
available/comfortable-stop fields). Output carries merge_zone_id, source
object, relevant_object_count, ego speed/distance-to-merge/merge-timing/eta,
available_distance_m + comfortable_stop_distance_m, complete_prediction_coverage,
front/rear object + gap_m + time_headway, rear_closing object + time, and the
smallest predicted minimum route gap. **No `merge_gap_m`** (Phase 17: the total
span never authorizes a merge; a consumer reads the two signed
`front_gap_m` / `rear_gap_m` individually).

Consumed GapRisk fields: `merge_reference_route_s_m`, `ego_route_distance_to_merge_m`,
`ego_route_distance_to_zone_entry_m`, `ego_longitudinal_speed_mps`,
`ego_merge_timing_valid`, `ego_merge_time_s`, `relevant_object_count`; per
relevant object `delta_s_now_m`, `object_longitudinal_speed_mps`,
`delta_s_at_merge_valid/_m`, `is_ahead/behind/alongside_at_merge`,
`longitudinal_gap_closing`, `longitudinal_closing_speed_mps`,
`time_to_route_coincidence_valid/_s`, `predicted_min_route_gap_valid/_m`,
`prediction_covers_merge_time`.

Applicability: active while ego is in the merge approach/zone this traversal.
`ego_route_distance_to_merge_m < 0` (past merge) or
`ego_route_distance_to_zone_entry_m > maximum_approach_distance_m` (400, must be
<= risk node's `maximum_ego_approach_distance_m`) -> `active=false`,
non-restrictive `ACTION_MERGE_READY` enum, `REASON_NONE`.

MERGE_READY rule (all of): applicable; `ego_merge_timing_valid` and route speed
above `stopped_speed_threshold_mps` (0.5); no over-budget relevant object;
**every** relevant object `prediction_covers_merge_time` (fallback constant-speed
`delta_s_at_merge_m` never authorizes -- mandatory, Phase 8/26); no relevant
object `is_alongside_at_merge`; every valid `predicted_min_route_gap_m >=
minimum_predicted_route_gap_m` (6.0); no closing rear object (`delta_s_now_m<0`
AND `longitudinal_gap_closing`) with `time_to_route_coincidence_s <
minimum_rear_closing_time_s` (3.0) or invalid coincidence; nearest rear at merge
(if any) rear time headway `-delta_s_at_merge_m / max(obj_long_speed, 0.5) >=
minimum_rear_time_headway_s` (2.0); nearest front at merge (if any) front time
headway `delta_s_at_merge_m / max(ego_speed, 0.5) >=
minimum_front_time_headway_s` (1.5). Zero relevant objects + valid timing ->
MERGE_READY / CLEAR_GAP. Absent front or rear side is vacuously satisfied.
Front/rear selected from per-object `is_ahead/behind_at_merge` relations, never
raw Cartesian (Phase 27).

Rear-closing is a **current-time** condition gated on `delta_s_now_m < 0`
(`longitudinal_gap_closing` is symmetric, so a slow front the ego overtakes
never trips REAR_CLOSING -- caught instead by the predicted route-gap tier
above FRONT_GAP).

WAIT/HOLD: `available = max(0, ego_route_distance_to_merge_m - merge_standoff_m)`,
`comfortable_stop = v^2/(2*1.8)`; stopped ego -> HOLD/INVALID_EGO_STATE (no
fabricated ETA); else `available > comfortable_stop` -> WAIT else HOLD. Strict
`>`. Moving ego with invalid timing (config-mismatch only) -> never MERGE_READY,
WAIT/HOLD by the margin rule.

Threshold provenance (competition-v1, not universal-safety):
`minimum_front_time_headway_s 1.5` (standard highway following-headway lower
bound), `minimum_rear_time_headway_s 2.0` (rear mainline has right of way),
`minimum_rear_closing_time_s 3.0`, `minimum_predicted_route_gap_m 6.0` (~IONIQ 5
length + margin; mirrors `merge_standoff_m`), `merge_standoff_m 6.0` (mirrors
cut-in/roundabout standoff; boundary before merge completion),
`comfortable_deceleration_mps2 1.8` (`perception.braking_deceleration_mps2`),
`stopped_speed_threshold_mps` / `speed_epsilon_mps 0.5` (mirror risk ego-speed
epsilon). No existing planner following-headway parameter existed to reuse.

Arbitration precedence: ALONGSIDE > INSUFFICIENT_PREDICTION >
PREDICTED_ROUTE_CONFLICT > REAR_CLOSING > REAR_GAP > FRONT_GAP; within a tier the
most restrictive metric, then lexicographically smallest UUID. Stateless, no
hysteresis (reported: none; canonical replay showed no MERGE_READY<->WAIT
flicker). Malformed/stale/future/duplicate/backward frame rejected (backward jump
> 0.5 s = sim reset, clears latch); `delta_s_at_merge_valid=true` with
`prediction_covers_merge_time=false` is permitted (the fallback). One response
per accepted frame, never latched.

Validation: `test_highway_merge_gap_response` 34 gtests (pure policy + frame
builder; incl. front/rear/rear-closing/predicted-route-gap threshold boundaries,
a coverage-claimed-shorter-than-merge-time rejection, and the Phase-26
fallback-extrapolation regression). Internal response callback latency
0.0039 / 0.030 / 0.030 ms median/p95/max (chained runtime summary). Interface contract 12/12 (added the actuator/speed/lane-change/CtrlCmd
disjoint-set test for the new message). Launch/planner-launch 35/35 (+opt-in composition, force-start
of risk node, default-off, `maximum_approach_distance_m <= 400`).
`test_highway_merge_gap_response_policy.py` live synthetic replay: 23 inputs ->
22 responses / 1 rejected (stale/backward); 5 MERGE_READY / 13 WAIT / 3 HOLD /
1 inactive; reason counts
CLEAR_GAP 5, FRONT_GAP 3, REAR_GAP 7, REAR_CLOSING 2, ALONGSIDE 1,
PREDICTED_ROUTE_CONFLICT 1, INSUFFICIENT_PREDICTION 1, INVALID_EGO_STATE 1;
MERGE_READY example front gap 45 m / headway 5.625 s, rear gap 70 m / headway
7.0 s, predicted min route gap 25 m; WAIT example (200 m) available 194 /
comfortable-stop 17.78; HOLD example (8 m) available 2.0 / comfortable-stop
17.78; insufficient-prediction example fallback delta_s_at_merge 200 m coverage
false; WAIT->HOLD approach + unsafe-rear-closing->MERGE_READY sequences both
exercised; 0 NaN/Inf/exceptions. `test_highway_merge_gap_risk_runtime.py` now
chains `-> HighwayMergeGapResponse`: 6 responses, 0 MERGE_READY / 5 WAIT / 1
inactive / 0 rejected (reasons NONE 1, PREDICTED_ROUTE_CONFLICT 3,
INSUFFICIENT_PREDICTION 2) -- the canonical replay's prediction-uncovered +
rollout-crossing objects mean it never earns MERGE_READY and the coverage policy
is NOT weakened to force one (Phase 39). Full `ad_planner` deterministic gtest
suite 528 cases pass (incl. cut-in / roundabout / external-speed / highway merge
gap-risk regressions unchanged); `test_data_loader` passes via ctest
(CWD-dependent fixture). Isolated `colcon build` of `ad_interfaces` + `ad_planner`
clean. Pre-existing unrelated failures, not caused by this change:
`test_mppi_nav2_launch` (nav2 absent); `test_cut_in_response_runtime` /
`test_cut_in_risk_runtime` (`ad_dynamic_object_risk_node` SIGKILL/OOM under load
-- perception node, untouched; same class documented in the PR #20 STATUS entry);
`test_frenet_runtime_contract` passes isolated (parallel-run resource
contention).

Files: `ad_interfaces/msg/HighwayMergeGapResponse.msg`,
`ad_interfaces/CMakeLists.txt`, `ad_interfaces/test/test_interface_contract.py`;
`ad_planner` `include/ad_planner/planning/highway_merge_gap_response.hpp`,
`src/planning/highway_merge_gap_response.{cpp,_node.{hpp,cpp},_main.cpp}`,
`config/highway_merge_gap_response.yaml`,
`launch/highway_merge_gap_response.launch.py`, `launch/planner.launch.py`,
`CMakeLists.txt`,
`test/{test_highway_merge_gap_response.cpp,_launch.py,_policy.py,
test_highway_merge_gap_risk_runtime.py,test_planner_launch.py}`;
`docs/planning/highway_merge_gap_response_v1.md`, this file. No planner node /
Stanley / DWA / Frenet / MPPI / external speed constraint / BehaviorTree /
`ad_interfaces` risk message change.

**Recommended next task:** Highway Merge Response Integration v1 — consume fresh
HighwayMergeGapResponse advisories at the existing highway mission/planner
boundary: WAIT/HOLD may constrain longitudinal progression while MERGE_READY
exposes a merge-authorization fact to the mission layer, without directly
commanding steering or creating a second CtrlCmd publisher.

---

## Highway Merge Gap Risk v1 — COMPLETE

Branch `feat/highway-merge-gap-risk-v1`, from merged PR #20 main
`b81935874eadb74579a69e0c108ca686e10057bd`. New opt-in policy-free facts node
`ad_highway_merge_gap_risk`: `/ad/planning/dynamic_object_risks`
(`DynamicObjectRiskArray`) + `/ad/localization/odometry` + a fixed
source-grounded merge zone -> `/ad/planning/highway_merge_gap_risks`
(`HighwayMergeGapRiskArray`). No MERGE/WAIT/GO/YIELD/HOLD/RELEASE/lane-change/
accelerate/decelerate/merge-allowed/merge-safe/accepted-gap/safe-gap/
requested-speed/CtrlCmd/brake/throttle/steering/risk-score/planner-consumer/
BehaviorTree output. Facts only.

Merge geometry (`ad_planner/config/highway_merge.json`, zone
`kcity_highway_onramp`): source lane `route:0:left:1` (single link
`A2256W000409` = waypoint 0 of the `ad_morai_bridge_dev` `kcity-highway` actor
preset, pinned `link_set_sha256 5ce0fd57…` identical to the corridor's
`source_sha256["link_set.json"]`) is a left-adjacent acceleration lane tapering
into primary route `route:0` (lateral offset +3.94 m -> 0.0 m; `route:0` speed
limit steps 11.11 -> 33.33 m/s at s ~ 1119). Target corridor = `route:0` itself
(post-merge primary route is the mainline; no separate target-lane centerline
exists — documented). `route_s_zone_entry/merge_complete = 1118.7418 /
1286.1546` are the first/last `route_s_m` of `route:0:left:1` in the
checksum-verified `route_corridor.json`; the node re-derives them at startup and
fails to start on a mismatch > 2 m, unknown lane id, target lane not spanning
the zone, or degenerate station window. `merge_reference_route_s_m =
route_s_merge_complete_m` (where the accel lane ends; target lateral offset 0).

Ego merge timing: constant-current-speed ETA onto `merge_reference_route_s_m`
along `project_to_frenet(route:0, ego)` (`s_m` / `s_dot_mps`); invalid when ego
route speed < 0.5, ego past the merge station this lap, or route distance to
zone entry > 400 m. Per object: ego projected onto the full
`route:0`; every object current + predicted centroid projected onto a
**station-bounded window** of `route:0` around the merge zone (covers the
approach bound + relevance window; keeps cost `O(objects·samples·window
points)`, independent of the 2184 m loop; a far object clamps to a window end
and reports `relevant_to_merge=false` with route/speed context zeroed). A far
ego (outside the window) still publishes an inactive frame, never a rejection. `relevant_to_merge` = laterally in the
target corridor AND route station in `[zone_entry - 150, merge_complete + 120]`
(generous report bounds, not a policy threshold), OR a predicted centroid does
so — window test first, before any merge-time field. Facts: `delta_s_now_m`;
`object_longitudinal_speed_mps` (current velocity resolved onto the route
tangent, NOT differentiated from prediction) + `relative_longitudinal_speed_mps`;
`delta_s_at_merge_m` (predicted route station at `ego_merge_time_s` minus the
merge reference — interpolated over the discrete prediction when it spans the
ego merge time, else constant-speed extrapolation with
`prediction_covers_merge_time=false`) + mutually-exclusive
`is_ahead/behind/alongside_at_merge` (5 m band);
`longitudinal_closing_speed_mps` (`-sign(delta_s_now)·rel_long_speed`) +
`time_to_route_coincidence_s`; `predicted_min_route_gap_m` (min longitudinal
separation vs a constant-speed ego route rollout over the horizon);
`prediction_horizon_s`; copied `ttc/cpa/min_sep`. Array summary:
`nearest_leading/trailing` (smallest `|delta_s_at_merge|` each side, lexicographic
UUID tiebreak) + `merge_gap_m` (leading − trailing span). Stateless; malformed
input never latches.

Validation: interface contract 12/12; `test_highway_merge_gap_risk` 34/34 (core
+ curved-geometry base_link-x-misleads + windowed-lane far-object clamp +
nearest-trailing arbitration + node frame contract). Canonical live replay: ego
swept `route_s = 1040 → 1160` at 8 m/s on the real `route:0` centerline, 4
mainline objects + one merging NPC per frame + one far-ego frame -> 6/6 frames,
0 rejected, 25 relevant object-frames, 20 ahead / 5 behind, 10 closing w/
coincidence time, 15 prediction-covers-merge-time / 10 not, 5
merging-predicted-enter, 5 merge-gap-valid (median 110.8 m); far-ego frame
inactive (not rejected); representative (ego s=1130): merge ETA 19.5 s,
`delta_s_at_merge` +69.9 m / −28.0 m now, merge gap 95.4 m; internal callback
latency ~15.6 / 15.8 / 16.7 ms median/p95/max for a 5-object × 24-sample stress
frame; no NaN/Inf/exceptions. Isolated `colcon
build` of `ad_interfaces` + `ad_planner` passes. Regression: `ad_planner` gtest
suite + cut-in / roundabout / external-speed / dynamic-object-risk / planner
launch tests unchanged; `test_mppi_nav2_launch` remains the known unrelated host
dependency failure.

Files: `ad_interfaces/msg/HighwayMergeGapRisk.msg`,
`HighwayMergeGapRiskArray.msg`, `ad_interfaces/CMakeLists.txt`,
`ad_interfaces/test/test_interface_contract.py`; `ad_planner`
`include/ad_planner/planning/highway_merge_gap_risk.hpp`,
`include/ad_planner/io/merge_geometry_loader.hpp`,
`src/planning/highway_merge_gap_risk.{cpp,_node.{hpp,cpp},_main.cpp}`,
`src/io/merge_geometry_loader.cpp`, `config/highway_merge.json`,
`config/highway_merge_gap_risk.yaml`, `launch/highway_merge_gap_risk.launch.py`,
`launch/planner.launch.py`, `CMakeLists.txt`,
`test/{test_highway_merge_gap_risk.cpp,_launch.py,_runtime.py,
test_planner_launch.py}`; `docs/planning/highway_merge_gap_risk_v1.md`, this
file.

**Recommended next task:** Highway Merge Gap Response v1 — consume
HighwayMergeGapRisk and produce a planner-facing MERGE_READY versus WAIT/HOLD
advisory using an explicit gap / time-to-coincidence policy, requiring
`prediction_covers_merge_time` for every relevant object, without directly
commanding lane change, steering, acceleration, or CtrlCmd.

---

## Planner Roundabout Response Constraint v1 — COMPLETE

Branch `feat/planner-roundabout-constraint-v1`, from merged PR #19 main
`5c64c77556f741dbe978e914c1ea69a83abf263d`. First roundabout change that can
alter production planner longitudinal behaviour. **Opt-in, off by default**
(`enable_roundabout_response_constraint: false`): disabled ⇒ no subscription, no
observability publisher, nothing read, behaviour identical to current main.

Integration point: the existing optional 6th `target_speed_mps` arg of
`PathTrackingController::update()` in `run_path_tracking()` (`FollowGlobalPath`
leaf) — the same point the cut-in constraint (PR #16) uses. No second
longitudinal controller, no extra `CtrlCmd`, no steering/path/lane/BehaviorTree/
tracker/prediction/risk/roundabout-policy change. Exactly one intended
`/ad/control/command` publisher remains.

Consumes `RoundaboutGapResponse` **only** (never `RoundaboutGapRisk`, never the
legacy first-interval fields). Pure helper `roundabout_response_speed_limit()`
(`roundabout_speed_constraint.hpp`, in `ad_planner_core`, no ROS-message dep):
missing/stale/inactive/RELEASE/unknown-action → `nullopt`; HOLD → `0.0`;
YIELD → `ego_speed_mps * sqrt(available_distance_m / comfortable_stop_distance_m)`
which is algebraically exactly `sqrt(2 * a_comfortable * available_distance_m)`
(the ego-speed terms cancel) — the response's own comfortable-stop envelope,
read through its published policy facts, **no deceleration constant / standoff /
gap threshold duplicated in the planner, no new yield-speed parameter**. Malformed
YIELD facts (non-finite, `available<=0`, `comfortable_stop<=0`, negative ego
speed) → `nullopt`. Freshness is steady-receipt-time only
(`roundabout_response_max_age_s: 0.5`); the response header stamp is never
consulted, so future/backward/duplicate stamps cannot extend freshness or latch
state. HOLD never latches; a stale RELEASE is never retained.

New `combine_speed_limits(a, b)` (`external_speed_limit.hpp`): symmetric min of
present finite optionals, `nullopt` if both absent. `run_path_tracking()` clamps
each of the cut-in and roundabout caps against the nominal cruise target, then
combines: `final = min(nominal, cut_in_if_active, roundabout_if_active)` —
order-independent, most-restrictive wins, RELEASE never lifts a lower cut-in cap.

Config (`config/planner.yaml`): `enable_roundabout_response_constraint: false`,
`roundabout_response_max_age_s: 0.5`, `topics.roundabout_gap_response`,
`topics.roundabout_speed_limit` (`std_msgs/Float32` observability: active cap or
`-1.0`, published only when enabled). `planner.launch.py` gained
`roundabout_gap_response` + `enable_roundabout_response_constraint` args; enabling
either also starts `ad_roundabout_gap_response` + (forced) `ad_roundabout_gap_risk`.

Scope limitation retained (same as cut-in): `PerceptionMission` (DWA/Frenet/MPPI)
is not constrained — `VehicleConstraints.maximum_speed_mps` is strictly-positive
so HOLD's `0.0` cannot be expressed and a DWA speed-cap change could perturb
lateral `(v, ω)` selection. `VehicleConstraints.maximum_speed_mps` not changed.

Validation: `test_roundabout_speed_constraint.cpp` 22 pure cases;
`test_external_speed_limit.cpp` 14 (composition + Phase-13 cut-in×roundabout
matrix + order independence); `test_planner_launch.py` +2 (declared args, opt-in
node composition, default-off override absent). Full `ad_planner`
non-launch-runtime suite **40/40** (31 gtest + 9 pytest, incl. pre-existing
cut-in constraint tests). Live `test_planner_roundabout_constraint.py` (isolated):
`ad_planner` in `FollowGlobalPath`, ego 8 m/s — baseline governed target 16.25;
RELEASE 16.25 (byte-identical no-op); YIELD (available 34 m) governed target
11.063 = `sqrt(2*1.8*34)`, above ego speed, no stop; HOLD (available 8 m) governed
target 0.0, command flips to brake; stale → 16.25 returns; steering byte-identical
across all states; 1 `/ad/control/command` publisher; 0 NaN/Inf/exceptions.
`test_planner_cut_in_constraint.py` regression passes unchanged.
`test_frenet_runtime_contract` passes isolated (only errors under the parallel
`colcon test` run — resource contention, documented pattern). Pre-existing
unrelated host failures, not caused by this change: `test_mppi_nav2_launch`
(nav2 absent, per PR #17-#19 STATUS entries); `test_cut_in_response_runtime`
(verified to fail identically on a changes-stashed clean 5c64c77 tree —
`ad_dynamic_object_risk_node` never publishes frame 0 and hangs; no `ad_planner`
node in the failing assertion). Isolated `colcon build` of `ad_interfaces` +
`ad_planner` clean.

Files: `ad_planner` `include/ad_planner/planning/{external_speed_limit.hpp,
roundabout_speed_constraint.hpp}`, `src/planning/{external_speed_limit.cpp,
roundabout_speed_constraint.cpp}`, `src/planner/planner_node.cpp`,
`src/planner/planner_ros_interfaces.{hpp,cpp}`, `config/planner.yaml`,
`launch/planner.launch.py`, `CMakeLists.txt`,
`test/{test_roundabout_speed_constraint.cpp,test_external_speed_limit.cpp,
test_planner_roundabout_constraint.py,test_planner_launch.py}`;
`docs/planning/planner_roundabout_constraint_v1.md`, this file. `ad_interfaces`
untouched (no new message).

**Recommended next task:** Highway Merge Gap Risk v1 — reuse Dynamic Object Risk
and route-relative geometry to estimate front/rear merge-conflict timing and
usable gap facts, without implementing lane-change or merge GO/WAIT policy yet.

---

## Roundabout Gap Response v1 — COMPLETE

Branch `feat/roundabout-gap-response-v1`, from merged PR #18 main
`813b9dda78179e1ca7d06a8bd7162ef2f3113d83`. New opt-in aggregate policy node
`ad_roundabout_gap_response`: `/ad/planning/roundabout_gap_risks`
(`RoundaboutGapRiskArray`) -> `/ad/planning/roundabout_gap_response`
(`RoundaboutGapResponse`). Advisory only: RELEASE/YIELD/HOLD, no requested
speed, CtrlCmd, brake, throttle, steering, planner consumer, or BehaviorTree
change.

RELEASE is an all-object conjunction over PR #18 fields: applicable/fresh/valid
ego timing, every relevant prediction covers ego exit, no
`any_occupancy_overlap`, every `minimum_temporal_gap` valid and `>= 2.0 s`.
Legacy first-interval overlap/gap never drive release. Zero objects release.
Stopped ego with invalid ETA holds. Inside or past the conflict is inactive.
Unsafe frames are YIELD while `max(0, distance_to_entry - 6 m) > v²/(2*1.8)`
and HOLD otherwise. Arbitration: overlap > incomplete evidence > smallest gap,
then lexicographic UUID. Stateless; rejected/missing inputs never latch release.

Validation: interface contract 10/10; response core/frame 18/18; response
launch 2/2. Canonical live chain: 5 gap-risk frames + one zero-object response
frame -> 6 responses, 1 RELEASE / 2 YIELD / 3 HOLD / 0 rejected; reason counts
CLEAR=1, OVERLAP=5; later-reentry and short-prediction regressions cannot
release; no NaN/Inf/exceptions. Internal callback latency 0.003742 / 0.010621 /
0.010621 ms median/p95/max; chain publish-to-receive 2.266 / 2.665 / 7.325 ms.
Isolated `ad_interfaces` + `ad_planner` build passes. Full regression result is
39/39 non-launch-runtime `ad_planner` tests pass, plus the canonical roundabout
launch-runtime test; `test_mppi_nav2_launch` remains the known unrelated host
dependency failure (7/15 cases fail because `nav2_common`/`nav2_controller` are
absent). The monolithic all-launch-runtime invocation was stopped after more
than six silent minutes; required roundabout runtime validation was run and
passed independently.

**Recommended next task:** Roundabout Response Planner Integration v1 — connect
fresh RoundaboutGapResponse YIELD/HOLD/RELEASE advisories to the existing
planner longitudinal constraint path, opt-in and default-off, while preserving
exactly one CtrlCmd publisher and leaving steering/lateral planning unchanged.

---

## Roundabout Gap Risk multi-interval summary — COMPLETE

Branch `fix/roundabout-gap-multi-interval-summary`, from merged PR #17 main
`f25cce2ffbaa1bed4e11586a7506a64adb0f39a2`. Makes the policy-free
`RoundaboutGapRisk` interface safe for a future gap-response consumer: the v1
first-interval fields (`object_entry/exit_time_s`, `arrival_delta_s`,
`temporal_gap_s`, `occupancy_overlap`) describe only the FIRST contiguous
predicted occupancy interval, so an early first exit could hide a later
re-entry that overlaps the ego window. **Additive fields only; every v1 field
keeps its exact meaning.** Still facts-only: no GO / YIELD / RELEASE / HOLD /
speed / brake / steering / BehaviorTree / CtrlCmd.

Unsafe counterexample (locked by
`test_roundabout_gap_risk.cpp::RoundaboutMultiInterval.FirstIntervalClearsButSecondOverlapsEgo`
and the runtime replay's object 5): ego occupancy `[4, 6]`, object occupies
`[1, 2]` then `[5, 7]`. v1 first-interval facts: `occupancy_overlap = false`,
`temporal_gap_s = 2` ("clear"). All-interval summary: `any_occupancy_overlap =
true`, `minimum_temporal_gap_s = 0`, `later_reentry_detected = true`,
`predicted_conflict_interval_count = 2`.

New `RoundaboutGapRisk` fields (after `occupancy_overlap`):
`uint16 predicted_conflict_interval_count` (contiguous inside-runs over the
discrete predicted centroids; a current-inside object contributes a run from
0.0; 0 when never inside), `bool later_reentry_detected` (count > 1),
`bool any_occupancy_overlap` (any predicted interval strictly overlaps ego
`[E0,E1]`; same strict/touching convention; valid only when ego entry+exit
valid, else false), `bool minimum_temporal_gap_valid` + `float32
minimum_temporal_gap_s` (min non-negative separation ego↔any interval; 0.0 on
overlap and touching; valid only when ego entry+exit valid and ≥1 interval),
`float32 prediction_horizon_s` (last predicted-centroid time, 0.0 if none;
always finite), `bool prediction_covers_ego_exit` (ego exit valid AND
`prediction_horizon_s + 1e-3 >= ego_exit_time_s` — data-coverage flag ONLY,
not safe-to-enter). Open final interval (still inside at horizon end) →
`O1 = +inf` for the overlap/gap math. No NaN/Inf in the message.

Core: new `extract_conflict_intervals()` (one polygon test per predicted
sample, same as v1) feeds both the first-interval fields (`intervals[0]`,
byte-identical contract) and the summary. Per-frame cost stays `O(N·M)`; no
all-pairs logic. Node `serialize()` + diagnostics keys
(`multi_interval_objects`, `later_reentry`, `any_occupancy_overlap`,
`minimum_temporal_gap_valid`, `prediction_covers_ego_exit`) +
`ROUNDABOUT_GAP_RISK_RUNTIME_SUMMARY` extended. `RoundaboutGapRiskArray`
unchanged.

Tests: `test_roundabout_gap_risk` **47/47** (30 v1 + 17 new
`RoundaboutMultiInterval`; `OnlyFirstContiguousOccupancyIntervalIsReported`
kept + now also asserts the summary sees the re-entry).
`test_interface_contract.py` **9/9** (declaration + conservative defaults +
`release` added to forbidden set). `test_roundabout_gap_risk_runtime.py` live
replay now sweeps 5 stations × 5 objects (added a leaves-then-re-enters
object and gave the non-conflicting object a short 4 s prediction so
`prediction_covers_ego_exit` is exercised in both states): 5/5 msgs,
25 objects, 20 relevant object-frames, 4 unique UUIDs, 5 multi-interval
object-frames, 5 later-reentry, 13 any-occupancy-overlap,
20 minimum-temporal-gap-valid, 21 prediction-covers-ego-exit (the short
object covers only at `s=886`); representative frame (ego `s=870`,
`[2.512, 5.500]`) re-entry object: first-interval `overlap=false gap=1.012`,
summary `intervals=2 any_overlap=true min_gap=0.0 reentry=true
covers_exit=true`; short object `covers_exit=false` (horizon 4.0 < exit 5.5);
0 NaN/Inf. publish→receive latency (loopback
DDS probe, NOT compute) ~2.3 ms median / ~3.0 ms max (5 samples; run-to-run
~1.5–3.5 ms). Full `ad_planner` ctest **42/43** (pre-existing
`test_mppi_nav2_launch`, nav2 absent). `test_dynamic_object_risk` 38/38.
Isolated `colcon build` of `ad_interfaces` + `ad_planner` clean.

Files: `ad_interfaces/msg/RoundaboutGapRisk.msg`,
`ad_interfaces/test/test_interface_contract.py`; `ad_planner`
`include/ad_planner/planning/roundabout_gap_risk.hpp`,
`src/planning/roundabout_gap_risk.cpp`,
`src/planning/roundabout_gap_risk_node.{hpp,cpp}`,
`test/{test_roundabout_gap_risk.cpp,test_roundabout_gap_risk_runtime.py}`;
`docs/planning/roundabout_gap_risk_v1.md`, this file.

**Recommended next task:** Roundabout Gap Response v1 — consume the all-interval
`RoundaboutGapRisk` summary and produce a planner-facing YIELD/HOLD vs RELEASE
advisory using an explicit temporal-gap policy, requiring
`prediction_covers_ego_exit` for every relevant object, without publishing
`CtrlCmd` or steering.

---

## Roundabout Gap Risk v1 — COMPLETE

Branch `feat/roundabout-gap-risk-v1`, from merged PR #16 main
`cc47007ad97e66f13c21cf804fba3f42f88770be`. Policy-free roundabout
conflict-timing facts: `DynamicObjectRiskArray` + ego route state + a fixed
shared conflict region -> `/ad/planning/roundabout_gap_risks`
(`ad_interfaces/msg/RoundaboutGapRiskArray`). No GO / YIELD / STOP / HOLD /
speed / brake / steering / BehaviorTree output; no planner consumer.

New opt-in node `ad_roundabout_gap_risk` (`planner.launch.py
roundabout_gap_risk:=true`, default `false`, or `roundabout_gap_risk.launch.py`).
Consumes the canonical `DynamicObjectRiskArray` (base_link) + `/ad/localization/
odometry`; loads the checksum-verified primary `ReferenceCorridor` (for ego
`project_to_frenet` station) and a new conflict-geometry config. No pairer, no
mask. Backend-agnostic pure core (no tracker/prediction branch).

Conflict region = an **annular sector** of the K-City roundabout, materialised
as an explicit 58-vertex map polygon in `ad_planner/config/roundabout_conflicts
.json` (`conflict_zone_id: kcity_roundabout`). Provenance: circulating-loop
centroid `(-101.717930, 343.622922)` = mean of the four `kcity-roundabout-loop`
link entry nodes in `ad_morai_bridge_dev/config/actor_presets.provenance.yaml`
(pinned `link_set_sha256 5ce0fd57...`); inner/outer radius `12.68 / 22.68` m =
mean corner radius `17.679` m ∓ a documented `5.0` m carriageway half-width;
angular span `[-80, -10] deg` = the SE arc where the primary route centreline
is inside the annulus; `route_s_enter/exit = 890.1006 / 914.0039` m from
`ad_data/map/route_corridor.json` (sha `c66d978c...`, recorded in the config,
**not modified**). The node cross-checks route<->polygon at startup (route
centreline at enter/exit/mid must be inside the polygon within
`polygon_consistency_margin_m` 2 m) and fails to start on mismatch / degenerate
polygon / unknown zone / bad corridor.

Facts (`RoundaboutGapRisk` per object): `relevant_to_conflict` (currently
inside OR a discrete predicted centroid enters within the horizon -- never
Euclidean proximity), `object_map_distance_to_conflict_m` (diagnostic only),
`object_in_conflict_now`, `object_entry_valid/time_s`, `object_exit_valid/
time_s` (exit invalid = still inside at horizon end), `arrival_delta_s =
object_entry_time_s - ego_entry_time_s` (signed; <0 object first), `temporal_
gap_s` (>=0, 0 on overlap and on touching boundary), `occupancy_overlap`
(strict), plus copied ttc/cpa/min-sep. Array header carries ego timing:
`ego_entry_valid/time_s`, `ego_exit_valid/time_s` (constant-current-speed ETA
along the primary route: `max(0, route_s_enter - ego_s)/v_ego`; invalid when
`v_ego < ego_speed_epsilon_mps` 0.5, ego past exit this lap [single-lap,
forward-only], or `> maximum_ego_approach_distance_m` 400), `ego_in_conflict_
now` (polygon containment), `ego_route_distance_to_entry/exit_m`,
`relevant_object_count`. Object centroids -> map via `ego_pose (+)
R(ego_yaw)*(v_ego*t + x_rel, y_rel)` (same CV-ego reconstruction as Cut-in
Risk; discrete samples follow curved circulating paths). Objects[] includes
all admitted objects (each with the flag), not just relevant ones. No
`safe_gap`/`yield_gap`/`go`/`risk_score` anywhere.

Ego route crosses the circulating carriageway once, along the SE quadrant
(enters from S, cuts the SE corner where circulating flow converges, exits NE);
~24 m / ~3 s of shared arc at 8 m/s.

New config: `ad_planner/config/roundabout_gap_risk.yaml`,
`ad_planner/config/roundabout_conflicts.json`. `ad_interfaces`:
`msg/RoundaboutGapRisk.msg`, `msg/RoundaboutGapRiskArray.msg`, CMakeLists,
contract test. `ad_data` untouched (route_corridor.json unchanged; its sha is
referenced from the new package config).

Files: `ad_planner` `include/ad_planner/planning/roundabout_gap_risk.hpp`,
`include/ad_planner/io/roundabout_conflict_loader.hpp`,
`src/planning/roundabout_gap_risk.cpp`,
`src/planning/{roundabout_gap_risk_node.{hpp,cpp},roundabout_gap_risk_main.cpp}`,
`src/io/roundabout_conflict_loader.cpp`, `config/roundabout_gap_risk.yaml`,
`config/roundabout_conflicts.json`, `launch/roundabout_gap_risk.launch.py`,
`launch/planner.launch.py`, `CMakeLists.txt`,
`test/{test_roundabout_gap_risk.cpp, test_roundabout_gap_risk_launch.py,
test_roundabout_gap_risk_runtime.py, test_planner_launch.py}`;
`docs/planning/roundabout_gap_risk_v1.md`, this file.

Tests: `test_roundabout_gap_risk` 30 pure cases (geometry, ego ETA, object
entry/exit incl. first-contiguous-interval-only, Phase-19 interval A/B/C +
touching + unbounded exit, arrival-delta sign, multi-object, zero, malformed,
budget, degenerate polygon, determinism, curved-trajectory).
`test_roundabout_gap_risk_runtime.py` live deterministic
replay: ego swept `s = 850..886` (8 m/s), 4 synthetic circulating objects/frame
-> 5/5 msgs, 20 objects, 15 relevant object-frames, 3 unique UUIDs, 5 valid
ego-entry, 15 valid temporal-gap, 8 overlap; ego entry ETA median/min/max
`2.512 / 0.500 / 4.999` s; arrival delta median/min/max `0.0 / -4.499 / 6.5` s;
non-overlap gap median/p10/min `2.262 / 0.25 / 0.25` s; publish->receive latency
(loopback DDS probe, NOT internal compute cost; 5 samples) median/p95/max
~`1.2 / 1.3 / 1.5` ms (run-to-run ~1.2-1.6 ms); representative frame
(ego `s=870`) ego `[2.512, 5.500]`, object intervals `[0.5,1.5]` /
`[2.5,5.0]` overlap / `[7.0,inf)`; clears-first gap `1.012` / overlap `0.0` /
after-ego gap `1.500`; 0 NaN/Inf. Full `ad_planner` ctest **42/43** (only
pre-existing `test_mppi_nav2_launch` -- host lacks `nav2_common`/`nav2_controller`).
`test_interface_contract.py` 9/9. `test_dynamic_object_risk` unchanged.
Isolated `colcon build` of `ad_interfaces` + `ad_planner` clean.

Occupancy semantics locked in v1: only the FIRST contiguous conflict-region
occupancy interval per object is reported; a predicted leave-then-re-enter is
not represented (documented in the `.msg`, the design doc, and
`OnlyFirstContiguousOccupancyIntervalIsReported`).

No provenance-verified MORAI roundabout circulating-actor bag exists (only the
`kcity-roundabout-loop` preset with pinned link IDs, no recorded actor state),
so validation is a deterministic canonical ROS replay on the real geometry, not
a simulator run.

**Recommended next task:** Roundabout Gap Response v1 -- consume
RoundaboutGapRisk and produce a planner-facing YIELD/HOLD versus RELEASE
advisory using an explicit temporal gap policy, without directly publishing
CtrlCmd or steering commands.

---

## Planner Cut-in Speed Constraint v1 — COMPLETE

Branch `feat/planner-cut-in-constraint-v1`, from merged PR #15 main
`b723fd5b90aceedb2c6fce06b270fd52526c9516`. First change that lets a cut-in
alter production planner longitudinal behaviour. **Opt-in, off by default**
(`enable_cut_in_response_constraint: false`): when disabled `AdPlannerNode`
creates no subscription, reads nothing, and is behaviourally identical to
current main.

Integration point: `run_path_tracking()` (the `FollowGlobalPath` BT leaf) now
passes an optional external cap as the already-existing 6th
`target_speed_mps` argument of `PathTrackingController::update()`. Stanley
resolves `target_speed_mps.value_or(config_.target_speed_mps)` then applies its
own `min` vs launch ramp and route profile, so a smaller value can only lower
the governed target, never raise it, and never touches steering
(`test_stanley.cpp::UsesPerUpdateTargetSpeedForSlowdown`). No second
longitudinal controller, no extra `CtrlCmd`; `AdPlannerNode::publish_command`
remains the one intended production control-command publisher.

Pure helper `cut_in_response_speed_limit()` (`ad_planner/.../planning/
cut_in_speed_constraint.{hpp,cpp}`, in `ad_planner_core`, no ROS-message dep):
not-received / stale / `active==false` / `ACTION_NONE` / malformed / contradictory
-> `nullopt` (true no-op, never `0.0`, never pass-through, never a speed
increase); `ACTION_SLOWDOWN` valid finite `>0` -> `requested_max_speed_mps`;
`ACTION_HOLD` valid `==0.0` -> `0.0`. The node clamps with
`min(nominal_cruise_target, limit)` and passes `std::nullopt` whenever the
result would not strictly reduce the target (byte-identical no-op).
`nominal_cruise_target` is read once from the backend's own
`<backend>.target_speed_mps` param purely as the clamp ceiling. Freshness:
`context_.steady_time_s` (set at top of `tick()`, same-tick) minus the
`steady_now()` receipt time `<= cut_in_response_max_age_s` (default `0.5`).

Phase 9 blocker documented, not worked around: `PerceptionMission`
(`run_local_motion()`, DWA/Frenet/MPPI) is **not** constrained in v1 --
`VehicleConstraints.maximum_speed_mps` is `positive_finite_parameter`
(strictly `>0`), so `ACTION_HOLD`'s `0.0` cannot be expressed there without
breaking the field invariant, and capping the dynamic-window speed could
perturb DWA `(v,ω)` selection (a lateral effect this task forbids). Local
motion also already runs its own corridor/occupancy/prediction longitudinal
response. Deferred to a separate representation.

New config (`config/planner.yaml`): `enable_cut_in_response_constraint: false`,
`cut_in_response_max_age_s: 0.5`, `topics.cut_in_response:
/ad/planning/cut_in_response`, `topics.cut_in_speed_limit:
/ad/planner/cut_in_speed_limit` (a `std_msgs/Float32` observability publisher,
created only when enabled: active cap, or `-1.0` when no constraint). No
comfortable/maximum deceleration or cut-in threshold duplicated -- those stay in
`ad_cut_in_response`. `planner.launch.py enable_cut_in_response_constraint:=true`
also starts `ad_cut_in_response` + `ad_cut_in_risk`.

Files: `ad_planner` `include/ad_planner/planning/cut_in_speed_constraint.hpp`,
`src/planning/cut_in_speed_constraint.cpp`, `src/planner/planner_node.cpp`,
`src/planner/planner_ros_interfaces.{hpp,cpp}`, `config/planner.yaml`,
`launch/planner.launch.py`, `CMakeLists.txt`,
`test/{test_cut_in_speed_constraint.cpp, test_planner_cut_in_constraint.py,
test_planner_launch.py}`; `docs/planning/planner_cut_in_constraint_v1.md`,
this file. `ad_interfaces` untouched (no new message).

Tests: `test_cut_in_speed_constraint` 22 pure cases (NONE/SLOWDOWN/HOLD, stale,
missing, inactive, invalid-flag, non-finite, `SLOWDOWN<=0`, `HOLD!=0`,
unknown/negative action, non-negativity sweep, monotonicity, limit<=request).
`test_planner_cut_in_constraint.py` live `ad_planner` in `FollowGlobalPath`
(`perception.enabled:=false`), ego `8.0 m/s`: no-response/NONE -> cap `-1.0`,
`accel==1.0`; SLOWDOWN `3.0` -> cap `3.0`, command flips to brake; HOLD -> cap
`0.0`, braking at least as strong; stale > max-age -> cap `-1.0`, baseline
returns; exactly one `/ad/control/command` publisher. Full `ad_planner` ctest
**39/40** (only pre-existing `test_mppi_nav2_launch` fails -- host lacks
`nav2_common`/`nav2_controller`). Isolated `colcon build --packages-select
ad_planner` clean.

**Recommended next task:** Roundabout Gap Risk v1 -- reuse Dynamic Object Risk
to estimate time-gap / time-to-conflict for vehicles approaching the roundabout
conflict region, without implementing GO/YIELD response yet.

---

## Cut-in Response v1 — COMPLETE

Branch `feat/cut-in-response-v1`, from merged PR #14 main
`fe034c9e4f000580f7e3f4c78c68cd15402e623d` (tree byte-identical to PR #14 head
`ef6de5f`). First planner-response policy layer: an opt-in node that turns the
factual `CutInRiskArray` frame plus ego longitudinal speed into a single
planner-facing longitudinal request `/ad/planning/cut_in_response`
(`ad_interfaces/msg/CutInResponse`). No actuator, steering, path, trajectory,
gear, lane-change, BehaviorTree, or DBW output; `planner_ros_interfaces.cpp`
untouched, so production driving behaviour is unchanged and **no production
planner consumes the request yet**.

No reusable generic planner-facing longitudinal-constraint interface exists (the
`ad_planner` node is monolithic and emits `ad_morai_interfaces/CtrlCmd`
directly), so a new narrow request message was created. Opt-in via
`planner.launch.py cut_in_response:=true` (default `false`; also starts
`ad_cut_in_risk`) or standalone `cut_in_response.launch.py`.

Policy states `ACTION_NONE / ACTION_SLOWDOWN / ACTION_HOLD`. Only
`cut_in_candidate` objects are considered. Per candidate: `station_ahead =
min(route_s_rel_m, predicted_entry_route_s_rel_m)` (route stations only -- 2-D
separations are never mixed into headroom); `station_ahead <= 0` -> NONE
(merging beside/behind ego); else `available = station_ahead -
longitudinal_standoff_m`, `a_req = v_ego^2 / (2*available)`,
`v_req = sqrt(2*comfortable_deceleration*available)` --
`a_req <= comfortable` NONE, `a_req <= maximum` and `v_req >= min_response`
SLOWDOWN at `v_req`, else HOLD. A valid `ttc_s <= v_ego/maximum_deceleration`
raises to HOLD. `cpa_distance_m` / `predicted_min_separation_m` are reported but
never drive the action (a completed merge always ends near zero separation).
Monotonic in entry station, TTC, and ego speed; entry *time* is reported only.
Multi-object: most restrictive action, then smallest requested speed, then
smallest UUID. Stateless -- exactly one response per accepted frame, never
latched, no release/activation-delay parameter (Cut-in Risk v1's replay showed
no chatter). 9 physical config params, no score bands.

Runtime: canonical Cut-in Risk v1 scenario through all three production nodes
with ego ramp `4.0 -> 10.0 m/s` -> 31 risk frames / 31 response frames, sequence
`NONE (0.0-0.7 s) -> HOLD (0.8-2.3 s) -> NONE (2.4-3.0 s)`, first active at
`0.8 s` (true corridor entry `2.2 s`, `1.4 s` lead), source alternates between
the mirrored UUIDs, 0 negative-control responses, 0 NaN/Inf, latency
median/p95/max `0.0036 / 0.0039 / 0.0041 ms`. A separate live policy replay
(synthetic risks, fixed `8 m/s`, station swept `45 -> 7 m`) shows
`NONE, NONE, NONE, SLOWDOWN, HOLD, HOLD, HOLD`, exact left/right symmetry, and
`HOLD` on the nearer of two candidates. Negative controls (parallel-adjacent /
moving-away / crossing) are never candidates so never respond: `0 / 0 / 0`.

Tests: `test_cut_in_response.cpp` 35 gtest cases, `test_interface_contract.py`
8/8, `test_cut_in_response_launch.py` 3, `test_cut_in_response_runtime.py` +
`test_cut_in_response_policy.py` live. Focused `ad_planner` ctest **37 / 38**;
the one failure `test_mppi_nav2_launch` is the pre-existing optional-Nav2
blocker (`nav2_controller` absent), untouched by this change.
`test_dynamic_object_risk` gtest unchanged/passing. Isolated builds of
`ad_interfaces` and `ad_planner` pass; `behaviortree_cpp_v3 3.8.7` built from
source into `~/projects/bt_ws` because `ros-humble-behaviortree-cpp-v3` is not
apt-installed on this host.

Files: `ad_interfaces/msg/CutInResponse.msg`, `ad_interfaces/CMakeLists.txt`,
`ad_interfaces/test/test_interface_contract.py`;
`ad_planner/include/ad_planner/planning/cut_in_response.hpp`,
`ad_planner/src/planning/{cut_in_response.cpp,cut_in_response_node.hpp,
cut_in_response_node.cpp,cut_in_response_main.cpp}`,
`ad_planner/config/cut_in_response.yaml`,
`ad_planner/launch/cut_in_response.launch.py`,
`ad_planner/launch/planner.launch.py`, `ad_planner/CMakeLists.txt`,
`ad_planner/test/{test_cut_in_response.cpp,test_cut_in_response_launch.py,
test_cut_in_response_runtime.py,test_cut_in_response_policy.py,
test_planner_launch.py}`; `docs/planning/cut_in_response_v1.md`, this file.

**Recommended next task:** Planner Constraint Integration v1 -- connect the
Cut-in Response longitudinal request to the existing planner speed/trajectory
constraint path while keeping steering avoidance separate.

## Cut-in Response v1 result: **COMPLETE**

---

## Cut-in Risk v1 — COMPLETE

Branch `feat/cut-in-risk-v1`, from merged PR #13 main `653fad390c3b41d5b56215d2ae271bb896d40916`.
Added an opt-in, policy-free route-aware cut-in facts node:
`/ad/planning/dynamic_object_risks` + exact-stamp
`/ad/planning/drivable_mask` + odometry ->
`/ad/planning/cut_in_risks`. The core reuses the checksum-verified
`ReferenceCorridor` and `project_to_frenet`; the primary route's interpolated
left/right widths define centroid membership. No lane IDs, generic lane
width, score, braking, steering, path, BehaviorTree, or response behavior was
added.

`DynamicObjectRisk` now carries policy-free discrete ego-relative future
centroids so the cut-in core remains on the canonical
Prediction -> Dynamic Object Risk -> Cut-in Risk architecture. Candidate =
outside adjacent centroid AND direction-aware lateral motion toward the
nearest corridor boundary AND discrete predicted sustained route entry AND
route-relative `s` in `[-5,80] m`. Left requires `d_dot < 0`; right requires
`d_dot > 0`. TTC is copied as a fact and is not required. Stateless replay
classification had one continuous candidate interval per true UUID, so no
hysteresis was added.

Runtime canonical replay through both production nodes: 31 risk messages ->
31 cut-in messages, 155 objects, 46 candidate object-frames, 2 unique
candidate UUIDs, 72 predicted-entry-valid frames. True cut-in was identified
at scenario time 0.0 s before centroid corridor entry at 2.2 s (2.2 s lead).
Parallel-adjacent / moving-away / crossing false positives = 0 / 0 / 0;
NaN/Inf/exceptions = 0; latency median/p95/max =
13.717712/14.537682/15.575071 ms. Existing MORAI tooling has no
provenance-pinned cut-in route, so this is a deterministic canonical-message
replay on the real route corridor, not a claimed MORAI result.

Tests cover 20 core/frame cases including mandatory left/right symmetry,
curved route, non-zero yaw, predicted entry, malformed/stale input,
determinism and backend equivalence; config/launch/interface tests and live
two-node replay also pass. Isolated builds pass for `ad_interfaces`,
`ad_lidar_perception`, and `ad_planner`. Focused final verification is
93 passed / 0 failed. The broader selected-package run is 1007 passed,
28 failed, 7 collection errors, 2 skipped; all failures are separated in
`docs/planning/cut_in_risk_v1.md` as optional dependency or unrelated
pre-existing dirty-worktree launch failures, not affected-test regressions.

**Recommended next task:** Cut-in Response v1 — consume CutInRisk and produce
a planner-facing longitudinal slowdown / hold request without steering
avoidance.

---

## Dynamic Object Risk Interface v1 — COMPLETE

Branch `feat/dynamic-object-risk-interface`, from `main`
`ef1fbc57c35ad7b0ae68049d6b033dfebee84340` (PR #12 merged). First step of
"make tracking/prediction useful to the planner": a **backend-agnostic**,
**policy-free** node that converts the canonical `PredictedObjectArray` +
ego state into planner-facing per-object relative-motion / closing / TTC /
CPA metrics. No GO / STOP / YIELD / brake / cut-in / merge / lane logic --
those consume this interface later.

**Audit (Phase 1).** No existing planner-facing relative-motion / TTC /
closing-speed / CPA interface exists (0 hits repo-wide for
`ttc|time_to_collision|closing_speed|relative_velocity|closest_approach|cut_in`).
`future_road_risk` (corridor-intersection boolean activation),
`DynamicObstacleStatus` (camera foot-point hazard), `prediction_admission`
(prediction-array staleness gate) are all different concerns. **New
interface created**, not an extension.

**Ego state (Phase 2).** `/ad/localization/odometry` (`nav_msgs/Odometry`):
pose in `odom`, longitudinal speed from `twist.twist.linear.x`. On the
default `gnss_imu` localization backend the twist carries only a real
longitudinal `x` (wheel speed); lateral and yaw-rate are structurally
zero, so the node treats ego lateral velocity / yaw rate as **0**
(non-holonomic). Ego velocity is never estimated from tracks or TF
differencing. `ego_twist_contradiction_frames` diagnostic counts frames
where the ego pose moves at ~0 reported speed.

**Message (Phase 3, `ad_interfaces`).** `DynamicObjectRisk` +
`DynamicObjectRiskArray`. Fields: identity (UUID, class, existence);
current relative state (`x_rel_m`, `y_rel_m`, `distance_m`, `vx_rel_mps`,
`vy_rel_mps`, `relative_speed_mps`); closing (`range_rate_mps` radial
primary, `longitudinal_closing_mps`); `ttc_valid` + `ttc_s`; `cpa_valid` +
`cpa_time_s` + `cpa_distance_m`; `predicted_min_separation_valid` + `_m` +
`_time_s`; `position_uncertainty_m`. **No `risk_score`, no GO/STOP/YIELD.**
No field is ever NaN/Inf; invalid metric => `*_valid=false`, value `0.0`.

**Conventions (Phase 4/5/6/7).** Output frame **`base_link`** (`+x` fwd,
`+y` left). `v_rel = v_object_world - v_ego_world`. `range_rate` =
`-(r . v_rel)/|r|` (sign-stable all quadrants). TTC = constant-relative-
velocity contact of two **circumscribed circles** (object
`0.5*hypot(L,W)`, ego `hypot(2.5, 1.1)`) -- **object orientation is never
read**, so AB3DMOT's unavailable-yaw and Autoware's real-yaw give the same
TTC. TTC invalid when separating / no root / beyond `ttc_horizon_s`.
CPA = minimiser of `||r + v t||^2` on `[0, cpa_horizon_s]`.
`predicted_min_separation` uses the discrete `states[]` vs a CV ego
rollout, kept separate from the kinematic CPA. **Implemented, Phase 7.**

**Backend independence.** No `if backend == ab3dmot` anywhere. The pure
core is a function on plain structs with no orientation input; the node
reads only the canonical message.
`test_dynamic_object_risk.cpp:ObjectOrientationNeverAffectsRiskOutput`
(AB3DMOT identity quaternion vs real Autoware yaw, same physical state ->
identical output) and
`test_lidar_perception_launch.py:test_dynamic_object_risk_is_opt_in_and_backend_agnostic`
lock it. Backend equivalence is validated by construction + unit test;
the bounded runtime replay exercised the AB3DMOT path only.

**Node (`ad_lidar_perception`, C++).** `ad_dynamic_object_risk_node`:
`/ad/perception/objects/predicted` + `/ad/localization/odometry` ->
`/ad/planning/dynamic_object_risks` (+ `/diagnostics`). Fail-closed:
rejects malformed/non-positive/duplicate/backward/stale/future prediction
stamps, wrong frame, missing/stale ego; large backward jump = sim-time
reset; individual malformed object skipped+counted, frame still publishes;
empty objects = valid empty array. Config
`config/planning/dynamic_object_risk.yaml` -- only semantic params, no
scenario thresholds.

**Launch (Phase 18).** Standalone `dynamic_object_risk.launch.py`. Opt-in
from `lidar_perception.launch.py` via `dynamic_object_risk:=true` (default
**off**, zero behaviour change), works for `tracker_backend:=autoware` and
`:=ab3dmot` through the shared prediction output. No planner node consumes
it yet -- observational only.

**Bounded replay (Phase 14-17).** 180 frames `static_20260805_003151`,
AB3DMOT -> Prediction -> Risk, synthetic canonical odometry (the replay
has none). **Arm A (stationary ego, physically correct):** 180 pred -> 180
risk msgs; 1334 objects in == out; 0 rejected frames/objects; every frame
`base_link`; **0 non-finite** across 1334 x 14 fields; TTC valid 215 (35
with real `ttc_s` 0.06-5.96 s, median 2.50; 180 near-origin overlaps ->
0.0); CPA valid 645; predicted-min-sep valid 1334; rel distance
med/p95/max 26.2/84.8/100.4 m; rel speed 0.0/14.1/19.2 m/s; CPA min-sep
med/p10/min 17.7/4.3/0.18 m; **latency median 0.011 / p95 0.023 / max
0.058 ms**; one publisher before/after; 0 NaN/Inf/exceptions.
**Arm B (synthetic moving ego yaw 0.4, 8 m/s -- mechanical check only):**
164 in/out, 1275 objects in/out, 0 rejected, 0 non-finite; every static
object `vx_rel = -8.0`, `range_rate ~ -8` (separating) confirming the
rotation + ego-velocity subtraction run on real data.
`morai_cam4_20260813_163222` (only moving-ego bag) unusable: host lacks
`rosbag2_storage_mcap`.

**Tests.** `test_dynamic_object_risk.cpp` **38/38** (20 geometry + radial
sign + predicted-min-sep + uncertainty + overlap/horizon/budget +
object-at-origin + 12 `build_risk_frame` stale/invalid/orientation-
invariance cases).
`test_dynamic_object_risk_launch.py` live node pass. `test_interface_contract.py`
7/7 (+ new: risk message stable, no policy fields). Broad regression
(`test_ab3dmot_*` minus kalmannet + competition + tracking + occupancy +
pipeline + interface) **240 passed / 1 skipped**. C++ ctests
`test_dynamic_object_risk` / `test_imm_predictor` / `test_cv_predictor` /
`test_dynamic_grid_builder` / `test_autoware_prediction_adapter` +
`test_dynamic_object_risk_launch` **6/6**. `test_lidar_perception_launch.py`
+ `test_selection_config.py` **97 passed**. Isolated `colcon build
--packages-select ad_interfaces ad_lidar_perception --symlink-install`
clean.

**No policy implemented (Phase 21).** No emergency/warning threshold, no
cut-in/roundabout/merge decision, no behaviour-tree change, no planner
cost change. Output is facts only.

**Files:** `ad_interfaces/msg/DynamicObjectRisk.msg`,
`DynamicObjectRiskArray.msg`, `ad_interfaces/CMakeLists.txt`,
`ad_interfaces/test/test_interface_contract.py`;
`ad_lidar_perception` `include/.../planning/dynamic_object_risk.hpp`,
`src/planning/{dynamic_object_risk.cpp,dynamic_object_risk_node.hpp,
dynamic_object_risk_node.cpp,main.cpp}`,
`config/planning/dynamic_object_risk.yaml`,
`launch/dynamic_object_risk.launch.py`, `launch/lidar_perception.launch.py`,
`CMakeLists.txt`, `package.xml`,
`test/{test_dynamic_object_risk.cpp,test_dynamic_object_risk_launch.py,
test_lidar_perception_launch.py}`;
`docs/planning/dynamic_object_risk_interface.md`, this file.

**Recommended next task:** Cut-in Risk v1 using the Dynamic Object Risk
Interface.

## Dynamic Object Risk Interface result: **COMPLETE**

---

## AB3DMOT yaw-rate uncertainty contract — COMPLETE (docs + tests only, no runtime change)

Branch `fix/ab3dmot-yaw-rate-contract`, from `main`
`36bdc6de30a77273c8c5f76a45c04a908ff7e8f3` (PR #11 merged). **Closes the
tracker/prediction covariance-contract workstream** (PR #9 oversized-OGM
skip, PR #10 position-cov bound, PR #11 yaw-*angle* cov slot, this = the
yaw-*rate* slot). Selected **Option D**: no runtime code change; document
and lock the contract with regression tests.

**The gap.** AB3DMOT's Linear KF has no yaw-rate state. It publishes
`twist.angular.z = 0` and leaves `twist.covariance[35]` (`wz-wz`,
`(rad/s)^2`) at `0`. `autoware_prediction_node.cpp` applies
`positive_variance(twist.covariance[35], 0.04)`, so prediction saw
"yaw rate = 0 known to +/- 0.2 rad/s". Question: is that a semantic
mismatch that needs a code change?

**Analysis (source + offline sweep).** No -- it is behaviorally inert.
`imm_predictor.cpp` gates `coordinated_turn` on the yaw-rate *value*
(`turn_evidence = |observed[kYawRate]|`, always `0` -> constant `-1.0`
log-likelihood penalty), **never on the yaw-rate variance**. The CV
model's yaw-rate term has its own `max(.., 0.01)` floor; the generic
`log(det S)` likelihood term shifts all three models identically. Built a
faithful Python port of `imm_predictor.cpp` + `imm_measurement` (validated
against all 3 `test_imm_predictor.cpp` cases: CV 0.891, stationary 0.998,
CT 1.0). Replayed the recorded 1257-object TrackedObjects stream
(identical UUIDs / stamps / pose / linear velocity every run) once per
candidate `twist.covariance[35]` in
`{0.04, 0.25, 1.0, 3.0, 4.0, 9.0, 100.0} (rad/s)^2`:

| `cov[35]` | CT median prob | CT max prob | stat / CV / **CT** selections | curved trajs | max \|Δyaw\| |
| --- | --- | --- | --- | --- | --- |
| 0.04 (fallback) | 2.03e-2 | 1.17e-1 | 593 / 664 / **0** | **0** | 0 |
| 0.25 | 2.34e-2 | 1.62e-1 | 593 / 664 / **0** | **0** | 0 |
| 1.0 | 2.46e-2 | 1.83e-1 | 591 / 666 / **0** | **0** | 0 |
| 3.0 | 2.50e-2 | 1.90e-1 | 591 / 666 / **0** | **0** | 0 |
| 4.0 | 2.51e-2 | 1.91e-1 | 591 / 666 / **0** | **0** | 0 |
| 9.0 | 2.52e-2 | 1.93e-1 | 591 / 666 / **0** | **0** | 0 |
| 100.0 | 2.52e-2 | 1.95e-1 | 591 / 666 / **0** | **0** | 0 |

A 2500x swing in the variable moved **0 coordinated-turn selections**, 0
curved trajectories, and left fused yaw rate at exactly 0 at every
horizon. The only effect: CT *probability* creeps `2.03e-2 -> 2.52e-2`
(never competitive), and two of 1257 object-frames flip
`stationary <-> constant_velocity` (both straight-line models) via the
`log(det S)` term.

**Option B (write an explicit finite `twist.covariance[35]`) was
implemented and measured, then rejected**: it requires an underivable
constant (physical yaw-rate bounds need speed + turn radius, both unknown
from "no information"), it moves the "field not provided" choice to the
wrong side of the interface (AB3DMOT asserting a fabricated number
instead of the consumer's uniform default), and the sweep shows it buys
nothing behaviorally. Option C (change the shared `0.04` default) stays
rejected on Autoware-regression grounds -- it is the correct home for the
one real (tiny) artifact, as a separate task if ever justified.

**Change (docs + tests only).** No runtime behavior change.
- `ab3dmot_ros.py`: explanatory comment at the twist serialization
  (the `angular`/`covariance[35]` zeros are deliberate; `rad^2` yaw-angle
  variance must not be copied into the `(rad/s)^2` slot).
- `test_ab3dmot_ros.py`: new `YawRateContractTest` (6 cases -- angular is
  exactly 0; `covariance[35]` and the whole angular 3x3 block unset;
  a huge yaw-angle variance never leaks to `twist[35]`; only the linear
  velocity block is populated; deterministic; no cross-track leakage).
- `test_imm_predictor.cpp`: new
  `ZeroYawRateNeverSelectsCoordinatedTurnRegardlessOfVariance`
  (yaw_rate = 0 with `yaw_rate_variance in {0.04, 1.0, 100.0}` -- CT never
  the argmax, CT < CV, fused yaw rate 0, every predicted horizon straight).
- `docs/perception/competition_mot_baseline.md`: new "Yaw-rate contract"
  section. `competition_dynamic_object_pipeline.md`: yaw-rate slot note
  updated (was "left for a follow-up"). This file.

**Tests.** Combined Python suite -- all `test_ab3dmot_*.py` except
`kalmannet` (needs torch), plus `test_competition_mot_baseline.py`,
`test_tracking_launch.py`, `test_occupancy_layer_launch.py`,
`test_lidar_perception_launch.py`, `test_selection_config.py`,
`test_autoware_pipeline_integration.py` -- **330 passed, 1 skipped**
(the 6 new `YawRateContractTest` cases and the existing
`test_ab3dmot_ros.py` 24 + `test_ab3dmot_tracker_node.py` 15 are inside
this number). C++ via `ctest -R` by name: `test_imm_predictor` 6/6 (incl.
the 1 new), `test_cv_predictor` 7/7, `test_dynamic_grid_builder` 21/21,
`test_autoware_prediction_adapter` 24/24 -- 4/4 registered and pass.
Isolated `colcon build --packages-select ad_lidar_perception
--symlink-install` clean.

**Runtime regression.** A fresh AFTER-arm replay was **not** run: no
runtime code changed (Python comment + test files + docs only), so the
Phase-3 live bounded replay on this branch (180 `static_20260805_003151` frames,
`tracker_backend:=ab3dmot`, full AB3DMOT -> prediction -> dynamic +
combined OGM) is the AFTER state: 172 tracked msgs / 1257 objects, 1
publisher per canonical topic, `objects_in == objects_out == 1257`,
`rejected = 0`, all `orientation_availability = UNAVAILABLE`,
`coordinated_turn` selections **0**, predicted `|angular.z|` max 0.0,
curved trajectories 0. PR #9 (oversized skip active), PR #10 (published
position std capped at 7.0 m), PR #11 (pose `covariance[35]` = yaw-angle
variance, median 13 / max 181; `covariance[21]` = 0) all still active. 0
NaN / Inf / exceptions; grids all finite, 0 invalid cells; 1 empty
dynamic grid (post-replay stale clear, expected).

**Autoware default: unchanged.** No shared prediction/serialization code
touched -- only a Python comment, two test files, and docs. The existing
`test_imm_predictor.cpp` (`Turning...`, `Stationary...`, `Initializes...`)
and `test_autoware_prediction_adapter.cpp` (24) pass unmodified.

**Contract (locked).** AB3DMOT's Competition MOT baseline publishes a
truthful zero yaw rate under an explicit non-rotating (constant-velocity)
assumption. `twist.covariance[35] = 0` means "not provided"; the shared
`0.04 (rad/s)^2` prediction default then applies, identically to any
tracker that omits the field. A zero placeholder is never interpreted as
observed turning information because turn detection is driven by the
yaw-rate value, not its variance. Coordinated-turn behavior remains gated
on real motion evidence. **Known limitation:** the `0.04` default gives
the stationary model a small `log(det S)` edge over CV (stationary's
`variance[kYawRate]` is clamped to `0.02`); measured effect is 2/1257
object-frames flipping between two straight-line models. Fixing that
belongs with the shared prediction default, not a per-tracker override.

**This closes the covariance/uncertainty-contract cleanup.** Next
development phase (not started here): `PredictedObjectArray` -> Dynamic
Object Risk Interface -> TTC -> cut-in risk -> roundabout gap acceptance
-> highway merge gap.

## AB3DMOT yaw-rate contract result: **COMPLETE (Option D)**

---

## Correct AB3DMOT yaw covariance contract — COMPLETE

Branch `fix/ab3dmot-yaw-covariance-contract`, from `main`
`82a75a85198cf4a0bdead9609ca3561406f37751` (PR #10 merged).

**Bug.** `geometry_msgs/PoseWithCovariance.covariance` is a row-major 6x6
over `(x,y,z,roll,pitch,yaw)`; yaw-yaw variance is flat index **35**.
`ab3dmot_ros.py` wrote `state.yaw_variance` to index **21** (roll-roll).
`autoware_prediction_node.cpp` reads index 35 -> saw `0` -> substituted
its `positive_variance(..., 0.04)` default. So prediction was treating the
always-zero placeholder Euclidean yaw as a measurement known to +/- ~11
deg -- an overconfidence caused purely by a covariance-indexing accident.

**Audit.** `yaw_measurement_mode=unobserved`: the KF never applies a yaw
measurement, so the latent theta stays **exactly 0.0** for every track,
every frame (offline: 1336/1336 track-frames), and `P[3,3]` is the
birth prior 10 + accumulated `Q[3,3]=1` per predict, never corrected --
replay median 13, p95 123, max 189 rad^2 (std 3.6-13.8 rad, all > pi =
uninformative). Prediction reads `orientation_availability` only for a
metrics counter, never to gate yaw. IMM `coordinated_turn` selection is
gated by the yaw-*rate* measurement value (`twist.angular.z`, always 0 ->
`turn_evidence < 0.03` -> -1.0 log-likelihood penalty every frame), not by
the yaw-angle variance, so a large yaw-angle variance cannot make CT fire.

**Fix: Option A (minimal, source-grounded).** One line in `ab3dmot_ros.py`:
`pose_covariance[3*6+3]` -> `pose_covariance[5*6+5]`. Keep
`orientation_availability = UNAVAILABLE`. Index 21 left at 0 (AB3DMOT
tracks no roll). No prediction change (Option C rejected: the prediction
node is shared with the Autoware default path). `twist.covariance[35]`
(yaw-rate variance) deliberately **not** touched -- AB3DMOT has no
yaw-rate state so Option A has nothing to route there, and its current
0->0.04 fallback helps pin the IMM turn-rate at 0; a principled yaw-rate
contract is a separate task. Documented as the same class of gap.

**Live bounded-replay A/B** (180-frame `static_20260805_003151`, disjoint
wall-clock stamps between arms):

| | BEFORE (idx 21) | AFTER (idx 35) |
| --- | --- | --- |
| tracked `cov[35]` (yaw var) med/p95/max | 0.0 / 0.0 / 0.0 | 12 / 118 / 176 |
| tracked `cov[21]` (roll slot) nonzero | 1285 (max 179) | **0** |
| orientation_availability | all UNAVAILABLE | all UNAVAILABLE |
| published yaw (quaternion) | 0 everywhere | 0 everywhere |
| prediction `initial_pose.cov[35]` | 0.04 (fallback) | 12 / 118 / 176 |
| prediction rejected | 0 | 0 |
| `stationary` selections | 633 | 572 |
| `constant_velocity` selections | 652 | 607 |
| **`coordinated_turn` selections** | **0** | **0** |
| predicted `|angular.z|` (yaw rate) max | 0.0 | 0.0 |
| curved predicted trajectories | 0 | 0 |
| 0.5s predicted displacement mean/p95/max | 1.50 / 6.19 / 7.67 | 1.46 / 6.14 / 8.09 |
| 1.0s | 3.03 / 12.73 / 17.23 | 2.91 / 11.96 / 17.09 |
| 2.0s | 6.14 / 26.32 / 36.34 | 5.87 / 23.64 / 36.01 |

CV/stationary split moves < 1 pt (50.7/49.3 -> 51.5/48.5), within the
disjoint-stamp noise. No turning introduced. The material effect is
latent correctness: any consumer of the yaw uncertainty (numerical, or a
future oriented detector) now reads the real value at the right slot.

**Autoware default: unchanged** -- the diff is one line in
`ab3dmot_ros.py` plus tests; no shared prediction/serialization code.

**Tests.** `test_ab3dmot_ros.py` +3 (yaw var at index 35; index 21 zero;
"yaw unknown" regime guard; whole-array exactness). Existing
`test_covariance_blocks_come_from_kf_p_only` updated (21 -> 35). Full
AB3DMOT + prediction + OGM + launch suite **296 passed, 1 skipped**; 4/4
C++ ctests (`cv_predictor`, `imm_predictor`, `dynamic_grid_builder`,
`autoware_prediction_adapter`). Isolated `colcon build` clean.

**Frozen/untouched:** position covariance cap (PR #10), velocity
covariance, KF `P`/`Q`/`R`, association, lifecycle, IMM params, Autoware.

**Files:** `ab3dmot_ros.py`, `test_ab3dmot_ros.py`,
`docs/perception/competition_mot_baseline.md`,
`docs/perception/competition_dynamic_object_pipeline.md`, this file.

**Remaining limitation:** `twist.covariance[35]` (yaw-rate variance) still
uses the 0->0.04 fallback -- same class of gap, deferred because a large
value could unpin the IMM turn-rate and a fix there must re-verify turn
counts.

## Stabilize AB3DMOT covariance for downstream prediction — COMPLETE

Branch `fix/ab3dmot-covariance-stability`, from `main`
`f255d3caba46559389587a98e67f15c3c4405591` (PR #9 merged). **Diagnosis
first, no internal cap. PR #9 NOT reverted.**

**Root cause (measured, not a bug).** Reference AB3DMOT birth prior
(`references/ab3dmot`): position variance 10 m^2, **velocity variance
10000 (m/s)^2**; `Q` pos 1, vel 0.01. CV transition couples them
(`F[0,7]=F[1,8]=dt`), so one `predict()` gives `P_next[0,0] = P[0,0] +
dt^2*P[7,7] + Q`. Under the frozen `min_hits:1`/`max_age:2`, a track born
from a single detection and never re-associated is published at birth
(std sqrt(10)=3.16 m), coasts **exactly one** step (still published at
`time_since_update=1`), then is deleted. That one coast at `dt`~0.19 s
propagates `dt^2*10000`~360 m^2 -> std ~19 m for that single frame.
Offline deterministic replay of the recorded 180-frame stream: 1336
published track-frames, position std median 1.00 / p95 19.06 / max
20.33 m; 155 rows (11.6%) exceed 15 m and **every one** is `hits=1`,
`time_since_update=1`, velocity variance still 10000; **0 non-finite, 0
non-PSD**; `dt` normal throughout. A track that gets a 2nd hit stays at
std ~2-4 m. Not a bug -- correct one-coast-step reachability of an
unobserved hypothesis with an uninformative velocity prior.

**Fix class: C (published-uncertainty bound).** A-no bug. B rejected
empirically: rescaling `P0` velocity variance (900/400/100 vs 10000) does
suppress the tail but changes the Kalman gain -> predicted positions ->
Euclidean-gate association -> track set (274 -> 265-270 unique published
ids; common states diverge mean 3-17 m, p95 up to 60 m). D rejected: the
`time_since_update=1` publish is legitimate under the frozen lifecycle
(no correctness bug in lifecycle semantics -> not tuned).

**Implementation.** New `ab3dmot_ros.bound_position_covariance(cov3,
max_std)` -- eigenvalue clipping (symmetry- and PSD-preserving) of **only
the 3x3 position block written into the outgoing `TrackedObject`**.
`tracked_state_to_message` / `tracked_states_to_message` thread a
`maximum_position_std_m` kwarg and return a `(msg, bounded_flag/count)`
tuple. Node gains a `maximum_position_std_m` parameter (**default 0.0 =
disabled**; negative rejected) and `position_cov_bounded=` /
`position_cov_bounded_frames=` in `AB3DMOT_RUNTIME_SUMMARY`.
`competition_mot_baseline_v1.yaml` sets **7.0 m**, derived forward:
`sqrt(P0_pos 10 + (v_max 31 m/s * one coast ~0.2 s)^2) ~= 6.96`, `v_max`
= top actor speed measured in this dataset (T-11). Deliberately above the
OGM's ~5 m 2-sigma-inflation knee -- a physical ceiling, not a device to
force OGM acceptance. Velocity covariance and yaw variance left raw. The
internal `LinearKFEstimator._kf.P` is **never** written.

**Internal-state invariance (Phase 9).** Offline replay with vs without
the serialization bound: every internal `x`, `P`, track id, hit count,
`time_since_update` **bit-identical** (1336 rows, 274 ids); deterministic
across reruns. Live: `tracks_created=265` identical in both A/B arms.

**Covariance A/B (offline, identical stream).** Published position std
p95/p99/max **19.06/20.03/20.33 -> 7.00/7.00/7.00 m**; rows >10 m and
>15 m **155 -> 0**; rows >3 m unchanged (468, real mid-range uncertainty
untouched); 155 objects clipped; serialized non-finite/non-PSD 0/0 both.

**Downstream OGM A/B (live bounded replay, PR #9 active, 160-frame
summary).** `oversized_objects_skipped` **157 -> 97**;
`frames_with_oversized_skip` **72 -> 57**; empty grids 1 -> 1 (startup);
dynamic occupied cells median 8868 -> 10237 (within run-to-run noise --
the two live arms sample disjoint wall-clock stamps, `common_stamps=0`);
invalid cells 0 -> 0. Extreme published tail gone (the deliverable);
~40% fewer objects reach the budget check, dominated by boundary objects
whose smaller 14 m halo no longer overlaps the grid. The cap does **not**
make an in-grid 1-hit track rasterizable (2-sigma inflation still 14 m);
PR #9 stays the operative guard. Real downstream win is for numerical
covariance consumers (IMM measurement variance, planner cost).

**Prediction sanity (Phase 12).** 12 horizons unchanged; `objects_in ==
objects_out`, `rejected=0`; all `orientation_unavailable`; `initial_twist`
finite; predicted `initial_pose` std distribution == tracked std
distribution (bound flows through, no re-inflation); no coordinated-turn
artifact; no stamp/UUID contamination (0 duplicate UUIDs/frame).

**Tests.** `test_ab3dmot_ros.py` +10 (`bound_position_covariance`:
disabled/within-limit/pathological/single-direction/symmetric-PSD/
no-mutation/deterministic; serialization: clips + flags, normal
unchanged, aggregate count). `test_ab3dmot_tracker_node.py` +5 (negative
param rejected; born-then-lost bounded when enabled / unbounded by
default; no cross-track leakage). Existing `tracked_state_to_message`
call sites updated for the tuple return. Full AB3DMOT + prediction + OGM +
launch suite: **293 passed, 1 skipped**; 4/4 relevant C++ ctests pass
(`cv_predictor`, `imm_predictor`, `dynamic_grid_builder`,
`autoware_prediction_adapter`). `test_occupancy_layer_launch` is
pre-existing-flaky under load (passes isolated; touches no file in this
diff). Isolated `colcon build --packages-select ad_lidar_perception`
clean.

**Frozen params untouched:** `euclidean_gate_m` 3.0, `matcher` hungarian,
`min_hits` 1, `max_age` 2, KF `Q`/`R`, IMM. No association/detector/IMM/
KalmanNet/CenterPoint/planner change. `maximum_position_std_m` = 7.0 is
the **one-coast-step** bound and is coupled to `max_age: 2`.

**Files:** `ab3dmot_ros.py`, `ab3dmot_tracker_node.py`,
`config/tracking/competition_mot_baseline_v1.yaml`, `test_ab3dmot_ros.py`,
`test_ab3dmot_tracker_node.py`,
`docs/perception/competition_dynamic_object_pipeline.md`, this file.
Known limitation (not fixed, noted): `pose_covariance` yaw variance is
written at flat index 21, not Autoware's index 35, so prediction's
`measurement.yaw_variance_rad2` falls back to its 0.04 default -- a real
pre-existing index bug, left alone because fixing it here would bundle an
untested IMM yaw-trust change into a covariance-magnitude fix.

## Dynamic OGM robust to oversized predicted-object uncertainty — COMPLETE

Branch `fix/dynamic-ogm-oversized-uncertainty`, from `main`
`02533f6aa93317d8b2fb2c9012221cb606d26179` (PR #8 merged). Fixes the
whole-frame clear the previous integration flagged.

**Root cause.** `build_dynamic_grid_impl` threw `std::length_error` when a
single object's grid-clipped, uncertainty-inflated footprint exceeded
`maximum_cells_per_object` (20000). `on_predictions` has one `catch` that
treats every failure the same, so that one object aborted the whole object
loop and `invalidate_and_clear` published an empty grid, discarding every
already-rasterized valid object. In the bounded replay this erased ~69 of
~171 frames (~40%). The trigger was almost always one AB3DMOT track with
~350-370 m^2 KF position covariance (std ~19 m) -> 2-sigma inflation ~38 m;
box dimensions were tiny (0.1-1.7 m), so coarse detector geometry was not
the cause. `maximum_cells_per_object` is a per-object rasterization budget,
not a compute-safety bound (the clipped rect is already grid-bounded), so
the right fix is to skip, not abort.

**Fix (per-object skip).** The two `maximum_cells_per_object` sites now
`continue` and increment a counter instead of throwing.
`build_dynamic_grid` gained a trailing `std::size_t *
oversized_objects_skipped = nullptr` out-param on both overloads
(non-breaking; ~20 existing call sites unchanged). The node accumulates
`oversized_objects_skipped_` / `frames_with_oversized_skip_`, logs a
throttled warning, and adds both to `DYNAMIC_OGM_RUNTIME_SUMMARY`.
**Deliberately unchanged:** every other per-object throw
(`validate_object`, covariance eigenvalue, footprint/bounds overflow) and
every array/geometry/config/mask throw still clear the layer (fail-safe).
No config value, covariance handling, inflation math, tracker, prediction,
or IMM parameter was touched. Backend-agnostic: the builder is a pure
function on `DynamicBox`, the node consumes canonical
`PredictedObjectArray`; no `if backend` anywhere.

**A/B bounded replay** (same 180-frame `static_20260805_003151` window,
AB3DMOT -> prediction -> dynamic + combined, `road_gate.enabled:=false`):

| | before | after |
| --- | --- | --- |
| oversized-object instances | 138 | 165 |
| frames with an oversized object | 69 | 77 |
| dynamic grids erased by one | 69 | 0 |
| empty dynamic grids | 69 (all had valid objects) | 1 (post-replay stale clear) |
| non-empty dynamic grids | 102 | 160 |
| occupied cells non-empty (med/p95/max) | 5296 / 28859 / 65804 | 9450 / 43682 / 91366 |
| invalid / non-finite cells | 0 | 0 |
| combined grids | 171 / 171 | 160 / 160 |
| dynamic step latency ms (med/p95/max) | 0.478 / 1.379 / 1.758 | 0.663 / 1.484 / 3.440 |

Every previously-erased frame now publishes its valid objects. Occupied
cells and latency rise because those recovered frames now do the
rasterization they previously skipped by throwing; grids stay <=~44%
occupied, latency stays far under the ~167 ms frame budget and 0.5 s
prediction timeout, and the structural per-frame bound is unchanged
(skipped object O(1), admitted object <= budget). Exactly one publisher on
every canonical topic before and after; zero NaN / Inf / exceptions /
crashes. Both A/B runs were verified single-publisher (an earlier
contaminated run with orphaned launch-child nodes was discarded and the
replay harness cleanup hardened).

**Tests.** `ad_lidar_perception` gtest 14/14 (`test_dynamic_grid_builder`
21 incl. 6 new: skip-not-throw, valid-survives-oversized, per-object
count, high-finite-covariance skip, malformed-still-throws,
determinism+out-param-reset). `test_occupancy_layer_launch` node-level
oversized test added (valid object survives an oversized sibling, grid
values valid). Focused pytest suite green. Pre-existing unrelated
failures on this host: `test_lidar_bag_replay_launch`,
`test_perception_visualization_launch` (both CRLF-only in the tree,
rosbag2 CLI mismatch). Isolated `colcon build` clean.

**Files:** `dynamic_grid_builder.hpp/.cpp`,
`dynamic_occupancy_grid_node.cpp`, `test_dynamic_grid_builder.cpp`,
`test_occupancy_layer_launch.py`, plus this file and
`docs/perception/competition_dynamic_object_pipeline.md`.

**Remaining limitation.** A frame whose *every* in-grid object is
oversized still yields a correctly-empty grid (none occurred in the
replay). The large KF covariance itself is unaddressed (out of scope) - a
covariance cap in the estimator or a physical uncertainty ceiling in
prediction is the recommended next task.

## Dynamic OGM oversized-uncertainty fix result: **COMPLETE**

---

## AB3DMOT -> Prediction -> Dynamic OGM integration — COMPLETE

Branch `feat/ab3dmot-prediction-dynamic-ogm`, from `main`
`84412cc1a91d86d0dfda10cc7f854a094e518d30` (PR #7). Connected the merged
model-free Competition MOT baseline to the existing HEVEN stateful IMM
prediction node and the existing dynamic / combined occupancy nodes.
`tracker_backend:=ab3dmot` now starts `prediction.launch.py`,
`dynamic_occupancy_grid.launch.py`, and `combined_occupancy_grid.launch.py`
- the same shared downstream launches the Autoware path uses. No
AB3DMOT-specific prediction or occupancy node was created. Autoware
remains the default; no learned model is required.

**Compatibility work.** Only launch wiring plus additive runtime
instrumentation. The `lidar_perception.launch.py` gate that restricted
prediction / dynamic / combined occupancy to `tracker_backend == autoware`
was a temporary safety gate (per the baseline doc), not a contract
incompatibility - removed cleanly. No AB3DMOT config, KF, gate, matcher,
lifecycle, IMM parameter, or occupancy parameter was tuned. Two overreach
changes found in the in-progress tree were reverted: a deleted
clock-rollback reset in the shared prediction node (would have wedged the
Autoware path after a MORAI time reset) and a switch of the IMM
`initial_twist` from the fused estimate to the raw measurement.

**Orientation UNAVAILABLE.** Prediction needs no change. `convert_object`
never rejects on the availability flag; the AB3DMOT identity quaternion
gives yaw 0, so `R(+yaw)` and the serialization `R(-yaw)` are both
identity and world Cartesian motion is preserved. The IMM consumes the
always-zero yaw with a bounded fallback variance and never forces a turn
(replay: only `constant_velocity` / `stationary` selected, 0
`coordinated_turn`). New tests
`test_autoware_prediction_adapter.cpp:Ab3dmotOrientationUnavailable*`.

**Velocity round trip (live ROS).** AB3DMOT published its pre-serialization
world velocity on a diagnostic audit topic (on only for `ab3dmot`).
Reconstructed world velocity from published local twist + pose yaw vs the
audited value over 1318 object-frames (651 with |v| > 1 m/s, max 20.4):
max |Δvx| = max |Δvy| = max |Δv| = 0.000e+00. Exact because yaw is
identically zero (identity rotations); the rotation math itself stays
covered by the merged baseline serialization tests.

**Bounded replay** (`~/datasets/morai_heven` `static_20260805_003151`,
first 180 frames, ~6 Hz, self-crop-bypass -> Patchwork++ -> finite ->
Adaptive Euclidean -> AB3DMOT -> prediction -> dynamic OGM, plus static +
combined with `road_gate.enabled:=false` because no repo-local replay has
`/ad/planning/drivable_mask` and the host lacks `rosbag2_storage_mcap`):
172 `TrackedObjects` (172 unique monotonic `odom` stamps), 171
`PredictedObjectArray` (12 states each, stamps a subset of tracked), 171
unique-stamp dynamic grids (77 empty / 94 non-empty, occupied median 4338
/ p95 24398 / max 50026, 0 invalid cells, 1040x200 @ 0.1 base_link), 171
combined grids (geometry valid, 0 invalid). Prediction rejected 1 array
(first-frame `stamp is in the future`), 0 orientation-related. Exactly one
publisher on tracked / predicted / dynamic / combined before and after
the replay. Zero NaN / Inf / exceptions / node crashes.

Component step latency (median / p95 / max ms): AB3DMOT 0.822 / 2.323 /
3.356; prediction 0.125 / 0.299 / 0.539; dynamic OGM 0.410 / 0.898 /
1.456. End-to-end callback-to-publish was not separately instrumented.

**Known limitations.** Execution / interface evidence only (one static
window, no HOTA / IDSW). Drivable-mask-gated occupancy not exercised
against real planning (no repo-local mask data; mask pairing / mismatch /
stale rejection stay covered by `test_occupancy_layer_launch.py` with a
synthetic driver). Large Linear-KF position covariance can inflate a
coarse Euclidean AABB past `maximum_cells_per_object`, and the dynamic
node then safely clears that frame (>=13 of the 77 empties); a covariance
cap / occupancy inflation review is a separate task, not tuned here. The
from-scratch 6 Hz replay dropped ~8 of 180 frames at start/stop
boundaries (Patchwork++ re-emit; AB3DMOT correctly rejected the
duplicate / backward stamps).

**Tests.** `ad_lidar_perception` gtest 14/14; focused pytest
(`test_lidar_perception_launch`, `test_tracking_launch`,
`test_occupancy_layer_launch`, `test_ab3dmot_*`) 164/164; the
`test_occupancy_layer_launch` launch_test reports 4/4 subtests pass
(ctest wraps a spurious SIGINT-teardown return code).
`test_lidar_bag_replay_launch` (1 subtest) and
`test_perception_visualization_launch` (1 subtest) fail on this host from
a pre-existing rosbag2 CLI mismatch, unrelated to this change (both files
are CRLF-only in the tree). Isolated `colcon build` of `ad_lidar_perception`
clean.

**Full detail:** `docs/perception/competition_dynamic_object_pipeline.md`.

**Recommended next task:** accuracy evaluation of this dynamic-object path
(needs a disjoint GT scene), or a drivable-mask-gated occupancy run
against a real planning stack, or a covariance-cap / occupancy-inflation
review for the coarse Euclidean boxes. Do not combine with estimator or
prediction-model tuning.

## AB3DMOT -> Prediction -> Dynamic OGM result: **COMPLETE**

---

## Competition MOT Baseline v1 — COMPLETE

Branch `feat/competition-mot-baseline-v1`, resumed from `main`
`2ff2301541cf0602912042bdcd87caf4e2b75f10`. Added an opt-in, model-free
Adaptive Euclidean -> AB3DMOT baseline with explicit Euclidean BEV 3 m,
Hungarian, Linear KF, yaw-unobserved, `min_hits=1`, `max_age=2` settings.
Autoware remains the implicit and explicit default; live graph checks show
exactly one canonical tracked-object publisher for all backend selections.

The velocity contract is resolved: AB3DMOT keeps world/odom Cartesian
velocity internally, while the ROS adapter now rotates velocity and its
covariance into Autoware's object-local `TrackedObject.twist` convention.
HEVEN prediction's existing inverse rotation therefore recovers world motion.
Prediction and Dynamic OGM remain intentionally disabled for the AB3DMOT
branch and were not modified.

Bounded smoke: first 180 exported frames from existing MORAI
`static_20260805_003151`, original stamps
`1785857513201006723..1785857541916511060` ns. Full crop -> Patchwork++ ->
finite filter -> Adaptive Euclidean -> AB3DMOT graph produced 180 detection
and 180 tracked messages; stamp sequences matched exactly and were strictly
monotonic; frames were `lidar_link -> odom`; 255 births, 246 deletions, 9 live
tracks; median/p95/max tracker step 0.755203/1.825568/3.827874 ms; zero
non-finite objects, exceptions, duplicate publishers, or lingering processes.
Execution evidence only, not accuracy validation.

The dirty tree started with 1,918 tracked changes: 1,907 were CRLF-only and
were excluded; no unknown substantive user edit was found. The AB3DMOT
executable has an LF shebang; no repository-wide normalization was performed.
Full detail: `docs/agent/competition_mot_baseline_audit.md` and
`docs/perception/competition_mot_baseline.md`.

**Recommended next task:** a separate opt-in AB3DMOT -> HEVEN Prediction ->
Dynamic OGM integration validation focused on unavailable-orientation
handling, world-motion round trip, exact-stamp pairing, occupancy finiteness,
and runtime. Do not combine it with estimator/prediction-model tuning.

---

## POST-FREEZE EXTENSION 2 — Camera + LiDAR Fusion, Stage 3 — COMPLETE (CASE B)

Branch `feat/tracking-preset-replay`, HEAD
`471e510a2dd387dd26940675ba183f30bc7f1f7a`. **GT-free,
research-only, opt-in, disabled by default. No camera information was wired
into production launch; Autoware remains the production tracker; no commit or
push.** Full artifacts:
`~/heven_presentation_assets/postfreeze_research_extension2/stage3/` and
`POSTFREEZE_EXTENSION2_STAGE3_SUMMARY.md` in its parent.

**Implementation / insertion:** the only existing tracker-source change is a
protected no-op `AB3DMOTTracker._association_solver_cost_matrix()` seam after
the original geometric cost is constructed and before the existing
Greedy/Hungarian dispatch. The base returns the original matrix object;
metric construction, matcher, post-solver geometry gate, KF, lifecycle,
`min_hits`, and `max_age` are unchanged. New untracked
`camera_lidar_semantic_association_core.py` provides a separate
`SemanticAB3DMOTTracker`; it is not installed or launch-wired. Its research
config defaults `semantic_association_enabled=False`.

**Evidence / memory / cost:** semantic evidence is sourced only from Stage
2's conservative-geometry, IoU-accepted camera↔LiDAR matches and carries
class, camera confidence, fusion IoU, validity, and timestamp. Reliability is
the transparent bounded product `camera_confidence * camera_lidar_iou`.
Track-private semantic memory is outside the KF vector: decayed weighted class
counts (0.95/frame), activated after >=2 valid updates and total decayed
weight >=0.20; missing evidence adds no contradiction. The explicit
compatibility matrix makes exact/broad road-vehicle classes compatible,
car/truck/bus differences 0.25, motorcycle/bicycle 0.5, cross-family 1, and
unknown neutral. Semantic cost is finite/additive only; same class is zero;
no hard semantic rejection.

**Controlled baseline:** same fixed Stage-2 clip (LiDAR bag indices
1094–1273, 180 frames, 21.130 s), all 2,614 unchanged LiDAR detections. The
geometry rule controls semantic eligibility only; it does not filter tracker
inputs. Both arms reuse the established camera-replay research baseline:
Euclidean BEV 3 m + Hungarian + Linear KF + yaw unobserved, `min_hits=1`,
`max_age=2`. Camera bag lacks actor GT and dynamic localization TF; all
tracker IDs/durations/churn are descriptive, not identity.

**Shadow / lambda result (negative):** 105 semantic LiDAR observations on 99
frames; 5,412 geometry-valid AB3DMOT candidate pairs; only 80/5,412 (1.478%)
have both a semantic detection and mature track memory. Only 3/5,412 (0.055%)
receive a nonzero penalty, all low-reliability truck observations against a
car-modal history. Lambda 0/0.1/0.25/0.5/1.0 changes **zero** of 2,536
assignments and avoids zero contradictions. Selected diagnostic lambda=0.1
is merely the smallest positive tested value, not best. Camera information
actually matters to 0/2,536 decisions.

**Controlled comparison / ablations:** baseline and selected semantic arms
are numerically identical: 2,536 assignments, 78 births, 61 deletions, 78
unique tracker IDs, 17 live at end, median/p95/max duration 10/137.5/180
frames, 11 tracks <=2 frames. Repeated-evidence AB3DMOT trajectories have
94.10% mean modal consistency over 12 trajectories, 2 adjacent camera-class
flips, and 38 matched↔unmatched transitions (not GT identity). Unweighted
class-only, confidence-weighted memory, and no-memory ablations affect 6, 3,
and 2 candidate costs respectively, but all change zero assignments. There
are no changed counterfactual cases to categorize; the empty schema and
negative-result figure are retained rather than inventing examples.

**Safety / calibration / determinism:** semantic disabled and lambda=0 are
full-clip state-equivalent to original AB3DMOT. No-camera-frame,
no-camera-detection, no-fusion-match, low-confidence, and delayed-camera
conditions all have exact baseline pair/state fallback and zero affected
pairs. Geometry-rejected pole-like detections receive zero semantic evidence;
one-to-one assignment is preserved. Selected replay pair signature is
deterministic. Focal ±5% sensitivity changes semantic inputs to 99/106 from
105, but all 2,536 tracker assignments remain identical — evidence of
non-influence here, not general calibration invariance.

**Latency:** diagnostics-disabled, 7 full repetitions with 5 warmup frames
excluded per run (1,225 samples/mode): median total tracker step
1.355→1.461 ms, +0.106 ms / 7.83%; p95 2.455→2.551 ms. Instrumented median
semantic cost construction 0.042 ms and memory update 0.019 ms. No new GPU
feature extraction.

**Readiness: CASE B.** The implementation is mechanically safe and bounded,
but current camera evidence is too sparse/non-discriminative to justify an
independent-GT tracking evaluation yet. No tracking improvement, HOTA, AssA,
IDSW, detection accuracy, or identity-accuracy claim.

**Tests:** curated directly relevant/broad safe suite **328/328 unittest**
(all Stage-1/2 fusion, 16 new Stage-3 semantic tests, AB3DMOT association/
tracker/estimator and related regressions) plus camera-replay launch
**17/17 pytest** = **345 pass / 0 code failures**. ROS Humble + heven/autoware
overlays, `ROS_LOG_DIR=/tmp/heven_stage3_ros_logs`, heven-centerpoint venv,
and explicit repo/test `PYTHONPATH`. One superseded attempt imported three
pytest-only modules through the venv unittest loader (pytest absent), producing
three harness-only import errors; excluded from valid counts and documented.

**Repo footprint:** modified tracked `ab3dmot_core.py` (minimal no-op seam)
and this shared status file. New untracked Stage-3 semantic core + test; prior
Stage-1/2 and Extension-1 dirty/untracked research files preserved. No config,
launch, node, detector, CenterPoint, Autoware, prediction, occupancy, motion
model, or lifecycle file changed. Nothing staged/committed/pushed.

**Recommended next experiment:** collect a mixed-class camera+LiDAR sequence
with measured intrinsics, dynamic ego pose/localization, and independent actor
identity GT. Freeze this implementation and evaluate it without tuning lambda
on evaluation identities.

## POST-FREEZE EXTENSION 2 Stage 3 result: **COMPLETE — CASE B**

---

## POST-FREEZE EXTENSION 2 — Camera + LiDAR Fusion, Stage 2 — COMPLETE (CASE A, NARROW)

Branch `feat/tracking-preset-replay`, HEAD
`471e510a2dd387dd26940675ba183f30bc7f1f7a`. **GT-free,
research-only, opt-in. No camera information was integrated into AB3DMOT;
no production source/config/launch/default changed; nothing committed or
pushed.** Full artifacts:
`~/heven_presentation_assets/postfreeze_research_extension2/stage2/` and
`POSTFREEZE_EXTENSION2_STAGE2_SUMMARY.md` in its parent.

**Stage-1 reproduction gate: PASS.** Existing offline repro was run from a
private `/tmp` copy against the same repository fusion core (SHA-256
`ff767a06…409`) and same YOLO weight (`646f8bc3…a1b`). Exact substantive
reproduction: 29 frames, 55 raw/24 kept camera, 234 LiDAR/164 degenerate,
150 projectable, IoU=4, center=22, mean IoU=0.2724, identical score and
degenerate-match counts, deterministic. Timing varied normally only. No
Stage-1 artifact overwritten.

**Fixed contiguous clip**: bag LiDAR indices **1094–1273**, 180 consecutive
frames, stamps `1786606472358099221`–`1786606493488090754` ns
(2026-08-13 16:34:32.358–16:34:53.488 KST), 21.129991533 s. Chosen before
Stage-2 matching around Stage-1 f11's vehicle-rich roundabout; no frame
cherry-picking. 421 camera messages inside interval, 172 unique nearest
camera images; LiDAR 8.471 Hz, camera 19.936 Hz; nearest-camera |dt| median
24.569 ms / p95 40.812 / max 45.129, **180/180 within 50 ms**. Unmodified
RANSAC + Euclidean ROS graph returned **180/180** detector messages.

**Detections / raw candidates**: 2,614 LiDAR detections; across the 180
LiDAR-triggered observations YOLO produced 814 raw / 587 kept camera
detections (780/563 when each unique camera image is counted once); 5,696
no-filter candidate pairs. Same Stage-1 YOLO 8.4.47, weight, class map,
confidence 0.20, reconstructed K0 (`fx=fy=640,cx=640,cy=360`), exact
extrinsic, and fusion core. Camera detector, association rerun, temporal
linker, and full match signatures all deterministic. Warmed Stage-1 IoU
`fuse()` over 180 frames: median **0.507 ms**, p95 0.694, max 1.188
(excludes YOLO and offline multi-condition sweeps; execution evidence only).

**Geometry regime / experimental pre-filter**: actual distribution inspected
first. 1,899/2,614 (72.65%) have minimum dimension ≤0.15 m, with the 10th,
25th, and 50th percentiles all at the detector floor 0.10 m and the 75th
percentile jumping to 0.326 m. Conservative LiDAR-only rule
`min(length,width,height)>0.15m` retains 715 (27.35%), rejects 1,899; strict
diagnostic `min_dim>0.40m && footprint>=0.75m²` retains 401 (15.34%). Camera
outcomes were not used to choose either rule; rejected boxes are not called
false positives. Conservative filter reduces projected candidates
5,696→1,700 and retains 105/130 (80.77%) of no-filter IoU matches; mean IoU
0.276→0.313; projected-degenerate eligible boxes 229→0. Visuals show rejected
0.10 m pole/guardrail/flat-fragment regimes, with the caveat that sparse real
objects may also be removed.

**Association robustness** (conservative geometry): IoU 0.10 = 105 accepted,
mean/median paired IoU 0.313/0.309, pre-assignment camera-many-LiDAR=7 and
LiDAR-many-camera=8. Center gates 0.03/0.05/0.075/0.10 accept
82/112/160/223; their mean paired IoU falls 0.345/0.262/0.191/0.145 and
competition rises 0/1, 4/10, 16/24, 28/44. Gate 0.10 median paired IoU is
0.022 and remains permissive/degenerate. Without geometry filtering even
center 0.03 accepts 489 clutter-dominated pairs. No “best” gate selected
without GT; IoU remains the declared next-experiment baseline.

**Analysis-only temporal linking**: production trackers untouched. Adjacent
frames only; BEV center ≤1.5 m, sorted box sizes ≤2× change, one-to-one
Hungarian, no coast/gap bridge. IDs named `analysis_track_id`, never GT.
73 links total; 22 last ≥5 frames (minimum inclusion), max 96 frames. Primary
conservative-IoU trajectories: 628 link-frames, 79 camera matches = **12.58%
semantic availability**, 10/22 links with any evidence; **97.27% mean modal
consistency** among links with repeated evidence; five fully stable visible
`car` runs of 15/15, 14/14, 11/11, 9/9, 5/5. One link has two adjacent
`car↔truck` class flips, visibly a partial image-edge case; class flip is not
IDSW. Eleven matched↔unmatched transitions. All qualifying median ranges are
near (<20 m); medium/far temporal stability remains unmeasured. Modal classes
9 car / 1 truck: diversity still weak.

**K sensitivity sweep** (not an uncertainty interval): with conservative
IoU, focal 0.95/1.05 retains 90.5%/97.1% of baseline matches (Jaccard
0.872/0.936); 0.90/1.10 retains 84.8%/86.7% (Jaccard 0.802/0.805).
Principal-point ±2% offsets give Jaccard 0.836–0.924 and can introduce/remove
matches (notably `cy−2%` adds 17). Temporal consistency remains 96.7–100%
but availability varies 10.67–14.65%. Mechanically bounded at ±5%, not
calibration-invariant; reconstructed K remains a limitation.

**Visual audit**: 8 required cases plus MP4 in `stage2/figures/`: stable
vehicle sequence, stable semantic timeline, class flip, geometry-rejected
pole, ambiguity, K-induced match change, IoU/center disagreement, permissive
center zero-IoU/rejected-geometry match. Failures were deliberately retained.

**Readiness: CASE A, narrowly.** Consecutive operation, non-degenerate IoU
evidence, repeated semantic stability, moderate-K set retention, bounded
ambiguity, deterministic mechanics, and lack of systemic projection failure
justify a separately designed **opt-in** tracker-association experiment.
This is not production readiness: semantic availability is sparse, evidence
is near-range and nearly all `car`, K is reconstructed, object GT absent.
No accuracy/HOTA/IDSW/identity claim.

**Repo footprint**: two new untracked opt-in files only for Stage 2:
`camera_lidar_fusion_stage2_core.py` (geometry rule, analysis linker, K
perturbation, match-set and semantic statistics) and
`test_camera_lidar_fusion_stage2_core.py` (11 tests). Existing Stage-1 and
Extension-1 dirty/untracked files preserved. `docs/agent/STATUS.md` remains
the only modified tracked file. No CMake/package/launch wiring.

**Tests**: curated 20-module unittest suite **312/312 pass** (includes all
39 Stage-1 fusion + 11 Stage-2 tests and the established 262-test regression),
plus pytest-only camera/LiDAR replay-launch **17/17 pass** = **329 pass / 0
code fail**. Environment: ROS Humble + heven/autoware overlays,
`ROS_LOG_DIR=/tmp/heven_stage2_ros_logs`, heven-centerpoint venv, and appended
`PYTHONPATH=~/projects/heven-ad-2026/ad_lidar_perception:$PYTHONPATH`; system
pytest for the pytest-only module. Two superseded broad-discovery attempts
hit harness-only missing-pytest/read-only-log errors and are retained in logs.

**Next experiment**: change one major variable only — add a tracker-level,
opt-in semantic association/update path using conservative geometry + the
existing IoU baseline. Keep detector, motion model, lifecycle, prediction,
occupancy, production launches/defaults unchanged. Stage 2 explicitly stops
before this work.

## POST-FREEZE EXTENSION 2 Stage 2 result: **COMPLETE — CASE A (NARROW)**

---

## POST-FREEZE EXTENSION 2 — Camera + LiDAR Fusion, Stage 1 — COMPLETE (CASE B)

Branch `feat/tracking-preset-replay`. Second post-freeze research
extension, begun after EXTENSION 1 was formally closed (section below).
**RESEARCH-ONLY, OPT-IN, DISABLED BY DEFAULT. No production default
changed, no frozen T-series / EXTENSION-1 handoff / presentation artifact
modified, not committed/pushed.** Camera is **not** ground truth. The
`morai_cam4_20260813_163222` bag is a **DIFFERENT-SEQUENCE** from T-14/T-15
— no HOTA/AssA/IDSW/CenterPoint number is attached to it. Full detail:
`~/heven_presentation_assets/postfreeze_research_extension2/`
(`POSTFREEZE_EXTENSION2_STAGE1_SUMMARY.md`, `EXTENSION2_RESULT_LEDGER.csv`,
and `camera_lidar_fusion/` with 8 docs + figures + data).

**Scope of Stage 1**: (1) sensor/calibration/data audit, (2) a defensible
fusion architecture, (3) a minimal geometry-based **late-fusion** baseline,
(4) mechanics + sync validation, (5) an explicit statement of what cannot
be evaluated (no independent GT). Not done, by design: no multimodal NN,
no detector/tracker replacement, no custom ROS message, no production
change.

**Repo footprint — 5 new untracked opt-in files, 0 tracked source files
modified**:
`ad_lidar_perception/ad_lidar_perception/camera_lidar_fusion_core.py`,
`camera_lidar_fusion_ros.py`, `camera_lidar_fusion_node.py`, and
`ad_lidar_perception/test/test_camera_lidar_fusion_core.py`,
`test_camera_lidar_fusion_ros.py`. The node (`camera_lidar_fusion`,
`enabled` param default **False**) is not wired into any production launch
and has no `CMakeLists.txt` install entry (same treatment as EXTENSION-1's
`kalmannet_arch2_core`). `ab3dmot_config.py` production defaults
(`association_metric="giou_3d"`, `matcher="greedy"`,
`state_estimator="linear_kf"`) verified unchanged; default detector
Euclidean, default tracker Autoware, unchanged.

**Audit (Phase 1, independently recomputed, not copied from P-E1)**:
- Only one camera+LiDAR bag exists project-wide (`morai_cam4_20260813_163222`,
  moving-ego highway loop, GT-free). The T-series scene has no camera.
- Camera `/ad/sensors/camera/front/compressed`: 1280×720 JPEG, 19.89 Hz,
  7038 msgs, monotonic, 0 dup/0 rollback. LiDAR `/ad/sensors/lidar/points`:
  VLP-16 layout, 8.42 Hz, 2982 msgs, monotonic, 0 dup/0 rollback,
  intensity present.
- **Extrinsic: A_exact** — bag `/tf_static` (13 transforms, `/tf` empty,
  no `odom` frame), chain
  `lidar_link→rear_axle_link→camera_front_link→camera_front_optical_frame`,
  cross-validated vs `ad_description/config/sensor_mounts.yaml`;
  `camera_lidar_fusion_core.rigid_transform_from_tf_chain` reproduces
  P-E1's independent projection math to 3.6e-15.
- **Intrinsics: B_reconstructed** — no `CameraInfo` anywhere; fx=fy=640,
  cx=640, cy=360 from documented HFOV 90° + resolution under a
  pinhole/zero-distortion assumption. Not a measured K → all projection
  results are qualitative/mechanical only.
- **Object GT: C — none** on this bag.
- **Sync (LiDAR-triggered, operative direction)**: median 20.0 ms, p95
  39.1 ms, max 49.7 ms, **100% within 50 ms** over all 2982 LiDAR frames;
  reproduces P-E1's `sensor_sync_summary.json` exactly.

**Architecture (Phase 2)**: geometry-based **late fusion** — both sensors
keep their own unmodified detector; project each LiDAR 3D box to the image
plane (exact extrinsic + reconstructed intrinsics), pre-filter camera
boxes (confidence / area / ego-hood vertical ROI), score with **2D IoU or
normalized center distance** (configurable, ≥2 measures), and associate
with the **existing** `ab3dmot_core._hungarian_matching` (no second
matcher). Fused object = LiDAR geometry authoritative + camera class +
association confidence + blended `existence_probability`; output is a
plain `autoware_perception_msgs/DetectedObjects` on
`/experiment/perception/objects/camera_lidar_fused` (no custom message);
`/ad/perception/objects/detected` untouched.

**Camera detector (Phase 3)**: the project's own documented COCO weight
`yolo26s.pt` (Ultralytics; SHA-256 `646f8bc3…84a1b`; referenced by
`ad_camera_perception/config/dynamic_obstacle.yaml`), conf 0.20, CPU. COCO
classes mapped to `car/truck/bus/motorcycle/bicycle/person`, rest dropped.
**External opt-in runtime dependency** (`ultralytics 8.4.47`, installed
into the `heven-centerpoint` venv, imported lazily, not in any
`package.xml`). **Weight never committed.** Recorded finding: COCO YOLO
detects the ego vehicle's own hood as a "car" in 24/29 frames — mitigated
by a vertical ROI pre-filter, not a GT judgment.

**Mechanical/descriptive results (Phases 6-7, n=29 synchronized frames,
GT-free)**: unmodified RANSAC + Euclidean produced 234 LiDAR detections,
**164 (70%) with a degenerate dimension** (Euclidean over-segmenting
guardrail poles on a highway — flagged, not dropped). Camera: 55 raw
vehicle/person → 24 kept after pre-filter. Fusion:
- **IoU gate 0.10: 4 fused** (mean IoU 0.27, min 0.13, max 0.40; 1 of 4
  matched degenerate LiDAR). The ~3 defensible matches per run are
  near-range (11-15 m), vehicle-sized, unambiguous (runner-up ≈ 0).
- **center-distance gate 0.10: 22 fused — but 18 of 22 (82%) matched a
  degenerate LiDAR box.** The gate (~147 px allowed center separation) is
  too permissive on this cluttered scene to constitute correspondence
  evidence; NOT to be read as "more correspondences than IoU".
- All 29 frames deterministic on rerun (both measures). Fuse latency
  steady-state median 0.54 ms / max 1.5 ms (the first `fuse()` call in a
  process pays a ~87 ms one-time numpy/branch cost; warmed once before the
  offline loop). Every fused semantic class is `car` (+ 2 long-range
  low-confidence `truck`) — no class diversity, no semantic
  disambiguation demonstrable on this scene.
6 qualitative figures (camera view + BEV), each labelled GT-FREE /
reconstructed-intrinsics.

**Tracking-fusion readiness (Phase 8): CASE B** (mechanics work; LiDAR
detection geometry and association-gate calibration must improve first) —
leaning toward the CASE C boundary on this scene. Operative blockers:
(a) 70% degenerate LiDAR geometry → clutter-dominated association;
(b) IoU unusable at that quality (4/24); (c) the center-distance gate that
"works" is 82% clutter matches; (d) B_reconstructed intrinsics;
(e) 29 non-consecutive frames (~11.5 s apart) → semantic-stability-over-
time, the property a tracker consumes, has zero evidence (rules out CASE A
alone); (f) no class diversity. Full reasoning + 5 readiness questions in
`camera_lidar_fusion/FUSION_TRACKING_READINESS.md`.

**Not live-tested this stage**: host lacks `rosbag2_storage_mcap` (no
`ros2 bag play`) and `vision_msgs` is not in the sourced overlay — the
fusion **node** is implemented and unit-tested; the offline pipeline
exercises the same fusion **core**.

**Tests**: 39 new (30 `test_camera_lidar_fusion_core` + 9
`test_camera_lidar_fusion_ros`), all pass. Full directly-relevant
regression **313 pass / 0 fail** (274 pre-existing across the AB3DMOT /
KalmanNet / arch2 / uncertainty suites, confirmed unaffected + 39 new);
torch/ROS modules via the `heven-centerpoint` venv + `PYTHONPATH` per the
established `test_kalmannet_core.py` precedent.

**Recommended next experiment**: a consecutive-frame clip (~150-200
contiguous LiDAR frames, vehicle-rich f08-f23 region) to measure
per-object semantic-label stability across frames, plus an intrinsics
sensitivity sweep and a LiDAR geometry pre-filter / cleaner detector so
the association problem becomes object-vs-object. Still GT-free. A
genuinely disjoint annotated camera+LiDAR scene remains the only thing
that unblocks quantitative fusion evaluation.

`git status --short` at the end of this task: `docs/agent/STATUS.md`
modified (this entry + the EXTENSION-1 close from the prior task); the 5
new untracked fusion files above plus the 4 pre-existing untracked
EXTENSION-1 files. No tracked source/config/launch/production file
touched. Not committed/pushed.

## POST-FREEZE EXTENSION 2 Stage 1 result: **COMPLETE — CASE B**

---

## POST-FREEZE EXTENSION 1 — CLOSED, CANONICAL HANDOFF FROZEN

Branch `feat/tracking-preset-replay` (descendant of the T-16 freeze point
`0f463a9`). This entry formally closes and freezes the completed first
post-freeze research extension (Phases 0-7, logged in the section below)
so all later work — Camera + LiDAR fusion first — starts from one clean
canonical handoff. **No new experiment run, no model retrained, no
production default changed, no frozen T-series result reinterpreted, not
committed/pushed.**

**Canonical handoff created**:
`~/heven_presentation_assets/postfreeze_research/final_handoff/` — the
single source of truth for all future extensions. Nine files plus an
audit: `README.md`, `POSTFREEZE_EXPERIMENT_TIMELINE.csv`,
`POSTFREEZE_FINAL_RESULTS.csv`, `POSTFREEZE_CLAIM_LEDGER.csv`,
`POSTFREEZE_SUPERSEDED_OR_CORRECTED_ASSUMPTIONS.md`,
`POSTFREEZE_DATA_LIMITATIONS.csv`, `POSTFREEZE_SYSTEM_ARCHITECTURE.md`,
`POSTFREEZE_ENGINEERING_FORENSICS.csv`, `POSTFREEZE_NEXT_STEP.md`,
`FINAL_HANDOFF_AUDIT.md`.

**Final audit verdict: PASS WITH CORRECTIONS.** The extension's work is
sound and its scientific boundaries hold; three items were made precise
during closure because the prior narrative omitted them: (1) the
all-frames position metric (arch2 C=3.32m is *worse* than frozen v2
B=2.55m at n=157 — the prior summary cited only the measurement-available
near-tie); (2) the 262/262 regression run needs an explicit `PYTHONPATH`
prefix or a `colcon build` because `kalmannet_uncertainty_core.py` has no
`--symlink-install` symlink in the installed overlay (build staleness,
not a code defect); (3) the n=133 (Phase 4) vs n=130 (frozen T-16 doc /
Phase 7) TEST-frame counts reconciled as a documented join-basis
artifact, not a restatement of any frozen number.

**Regression status**: full directly-relevant Python suite, 17 modules,
re-verified `2026-08-27` via the `heven-centerpoint` venv + ROS Humble +
`heven_ros_ws` + `autoware_tracker_ws` +
`PYTHONPATH=~/projects/heven-ad-2026/ad_lidar_perception` —
**262/262 pass** (242 pre-existing byte-unchanged + 14
`test_kalmannet_arch2_core` + 6 `test_kalmannet_uncertainty_core`).

**Production defaults**: unchanged. `AB3DMOTConfig`
(`association_metric="giou_3d"`, `matcher="greedy"`,
`state_estimator="linear_kf"`) and `SUPPORTED_STATE_ESTIMATORS =
("linear_kf","ekf","imm","kalmannet")` byte-unchanged; `ab3dmot_config.py`
not in the git diff. The two new opt-in modules are standalone research
code, not registered or wired into the live tracker.

**Git state**: `git status --short` shows exactly 1 modified tracked file
(`docs/agent/STATUS.md`, this + the prior extension entry) plus the same 4
untracked opt-in files
(`ad_lidar_perception/ad_lidar_perception/kalmannet_arch2_core.py`,
`kalmannet_uncertainty_core.py`, `ad_lidar_perception/test/test_kalmannet_arch2_core.py`,
`test_kalmannet_uncertainty_core.py`). Frozen
`~/heven_presentation_assets/final_development_handoff/` untouched (mtime
`2026-08-22`). Reference submodules at pinned commits.

**Next major research phase**: Camera + LiDAR fusion (can begin now on
existing data for architecture/runtime/qualitative work). **Multi-scene,
GT-capable MORAI data remains the major scientific dependency** for
CenterPoint validation, the Phase 6 tracker headline, Architecture-2
generalization, a real Phase 5 TEST evaluation, and the full
Detection × Association × Estimator factorial — unchanged from T-11B
onward. Full continuation plan: `final_handoff/POSTFREEZE_NEXT_STEP.md`.

## POST-FREEZE EXTENSION 1 result: **CLOSED**

---

## POST-FREEZE Research Extension (Phases 0-7) — COMPLETE

Branch `feat/tracking-preset-replay` (descendant of the T-16 freeze point
`0f463a9`). This is a POST-FREEZE research extension: no frozen T-1..T-16
result was modified, no production default changed. Full detail in
`~/heven_presentation_assets/postfreeze_research/` (`POSTFREEZE_RESEARCH_SUMMARY.md`,
`POSTFREEZE_RESULT_LEDGER.csv`, `claim_ledger.csv`, and one directory per
phase). Repo footprint: exactly 4 new, untracked, opt-in files --
`ad_lidar_perception/ad_lidar_perception/kalmannet_arch2_core.py`,
`kalmannet_uncertainty_core.py`, and their two test files. No existing
tracked file modified; `AB3DMOTConfig`'s production defaults and
`SUPPORTED_STATE_ESTIMATORS` are unchanged (the two new modules are
standalone research code, not yet wired into the live ROS
`AB3DMOTTracker`).

**Phase 0 (data inventory)**: exactly one GT-capable scene exists
(`morai_heven`, unchanged since T-11) and one sensor-only bag with no
object GT (`morai_cam4_20260813_163222`, already fully used by P-E1/P-E2
for qualitative illustration only). Gates Phase 2's terrain-diversity
request and Phase 6's final evaluation as DATA-BLOCKED, per
`dataset_limitations.csv`'s own already-established constraints.

**Phase 1/2 (ground-filter audit + Patchwork++)**: found the
source-configured default ground-filter backend (`ground_segmentation.yaml`'s
`algorithm: patchwork`, `ground_segmentation.launch.py`'s own `backend`
default) was never actually fetched, built, or run anywhere in this
project before this session -- `src/patchwork-plusplus/` was an empty
vcs-import placeholder, directly contradicting this task's own stated
premise that Patchwork++ was "already integrated and previously
benchmarked." Fetched the pinned source this session (exact commit
match), found it ships a real ROS2 wrapper matching HEVEN's launch file's
topic/parameter contract exactly, and built it cleanly (43.3s, 0 missing
system dependencies) into a new sibling workspace
(`~/projects/patchwork_ws`). Live-launched and confirmed all three
backends (RANSAC, Classic Patchwork, Patchwork++) against the same
100-frame window of the single available static scene: RANSAC and
Patchwork++ pass a similar nonground-point population through (1642 vs.
1978 pts/frame; 4.74 vs. 4.63 downstream Euclidean detections/frame);
Classic Patchwork is markedly more aggressive at ground removal (991
pts/frame, 3.25 detections/frame). Descriptive only, no GT, no accuracy
ranking. The task's own 80m/105m range and intensity-passthrough
sub-questions are explicitly BLOCKED (no terrain-diverse data exists;
intensity audit out of this bounded phase's time budget) rather than
approximated.

**Phase 3/4 (KalmanNet Architecture-2 + fair comparison)**: new opt-in
`kalmannet_arch2_core.py` implements the KalmanNet reference's genuine
three-GRU (`GRU_Q -> GRU_Sigma -> GRU_S`) Architecture-2 topology with
its backward-flow feedback into the Sigma-GRU's hidden state -- not
"three stacked GRUs," and does not modify `kalmannet_core.py`.
Track-private recurrent state (passed functionally, never attached to
the shared weight module) makes the T-9B hidden-state-leakage bug class
structurally impossible by construction. 30/30 tests pass (14 new + 16
pre-existing unaffected). Fair A(tuned KF)/B(frozen DENSE-KALMANNET-v2)/
C(new arch2) comparison, identical split/data/recipe, A and B's numbers
reused verbatim from the frozen handoff: 10/10 seeds stable; on the
primary measurement-available TEST metric (n=133) position RMSE is
essentially tied (A=1.443m, B=1.455m, C=1.437m) and C shows a modest
velocity-RMSE edge (A=2.612, B=2.654, **C=2.382** m/s) -- small-sample,
same-scene/same-protocol comparison only, not a generalization claim.

**Phase 5 (bounded sweep)**: 18 configs (hidden_size 16/32/64 x
num_layers 1/2 x feature_set innovation/current/fuller) x 3 seeds = 54
runs, 0 catastrophic. Two real findings: (1) the reduced `innovation`
feature set (drops the state-side diffs) is best-or-tied-best at 5/6
hidden/layer combos, while adding an explicit `dt` feature (`fuller`) is
worst-or-near-worst at 5/6 -- counter-intuitive, not acted on; (2)
holding `feature_set=current` fixed, capacity improves loss monotonically
at 1 layer but is non-monotonic at 2 layers (32,2 beats 64,2) --
consistent with this problem being close to data-limited beyond ~2-4x the
frozen architecture's own size. No config recommended as a default
change (3-seed screen, not the 10-seed final-recipe budget).

**Phase 6 (CenterPoint-aware tracker tuning)**: HARD DATA-BLOCKED for any
final headline, exactly as this task's own gate anticipated -- no
independent CenterPoint-unseen scene exists. Built the full tuning
infrastructure (every knob this task named); mechanics validation (19
one-knob-at-a-time configs against the frozen CenterPoint detection
stream, no GT read) -- 19/19 constructed, ran to completion, 0 NaN/Inf,
deterministic reruns. A small, fixed, pre-declared (never searched)
diagnostic-sensitivity set reproduces T-14/T-15's own published B1
baseline (HOTA=0.0529/IDSW=443) exactly and T-15's own Phase 13
score-threshold finding almost exactly, confirming the harness is
correct; no configuration in that table is recommended or ranked.

**Phase 7 (KalmanNet uncertainty)**: NEGATIVE RESULT. New opt-in
`kalmannet_uncertainty_core.py` (`InnovationCovarianceHead`, a
Cholesky-factor covariance head guaranteed positive-definite by
construction) trained on top of the frozen DENSE-KALMANNET-v2 (its
weights never updated). Training NLL diverges (2.57 -> 35.97 over 10
epochs) rather than converging; the selected checkpoint is the network's
own random initialization (`best_epoch=0`). TEST calibration (n=130):
mean calibration error 22.2 percentage points across 4 confidence levels
(e.g. 65.4% empirical coverage at nominal 90%); predicted covariance is
3-8x smaller than empirical innovation variance in both x/y --
systematically overconfident. Per this task's own explicit instruction,
the KalmanNet-state + calibrated-covariance -> Mahalanobis-association
integration was correctly **not attempted**.

**Real engineering issues found and fixed this session** (process/
tooling only, no repo/algorithm file touched): a QoS durability/
reliability mismatch silently dropped 100% of RANSAC's ground-filter
messages on the first attempt; a missing `autoware_tracker_ws` overlay
silently crashed the RANSAC launch (masked by discarded stderr) on a
separate attempt; stray `ad_finite_point_filter_node` processes
accumulated across successive downstream-detection runs, inflating
message counts up to ~3x; and a `nohup ... &` background sweep launch
survived its wrapper shell's exit (believed dead, was not) and ran as an
undetected duplicate process for ~35 minutes alongside a second,
properly-tracked run of the identical sweep, both racing to write the
same output file -- found via `ps aux`, fixed by killing the stray
process; the sweep's full determinism meant no result validity was lost.

**Tests**: full directly-relevant suite re-run, **262/262 pass** (242
pre-existing unmodified and confirmed unaffected + 20 new: 14 in
`test_kalmannet_arch2_core.py`, 6 in `test_kalmannet_uncertainty_core.py`).

**Safe new claims**: Patchwork/Patchwork++ is now genuinely buildable and
launchable on this machine; arch2 is seed-stable and not worse than the
frozen estimator on this scene's TEST split; the Phase 6 tuning
infrastructure is mechanically sound; feature-set choice matters more
than raw capacity for this KalmanNet problem.

**Unsafe claims (explicitly not made)**: any Patchwork++-vs-RANSAC
accuracy ranking; any claim that arch2 "beats" the frozen KalmanNet
estimator; any CenterPoint-aware tracker "improvement"; any claim that
KalmanNet's uncertainty is usable for Mahalanobis association; any
CenterPoint generalization claim.

**Recommended next action**: capture a genuinely disjoint MORAI scene
with independent actor GT (T-11B's own written, never-executed manual
capture protocol remains the concrete plan) -- this remains the single
highest-leverage next step, unchanged from every prior data-blocked
finding in this project. Until then: (a) audit Patchwork++'s intensity-
passthrough parameter (Phase 2's own scoped-out sub-question), and (b)
diagnose the Phase 7 covariance head's NLL training divergence directly
before any future Mahalanobis-integration attempt.

`git status --short` at the end of this task shows only the same
pre-existing state from every prior session (clean at session start on
this branch) plus this `STATUS.md` update and the 4 new opt-in files
listed above -- no repo algorithm/config/launch/production file was
touched anywhere. Not committed/pushed, per this task's instruction. No
stray background process remained running at session end (verified via
`ps aux`).

## POST-FREEZE result: **COMPLETE**

---

## 10-mode camera + LiDAR tracking preset replay — PASS (source/install), LIVE BLOCKED BY HOST PACKAGES

Branch `feat/tracking-preset-replay` adds a validated, configuration-owned
10-mode qualitative comparison matrix to the existing camera/LiDAR replay.
`scripts/run_camera_lidar_tracking.sh --list-modes` lists the matrix and
`--mode 1` through `--mode 10` selects Euclidean/CenterPoint detection,
GIoU/Euclidean-3m/Mahalanobis-Hybrid-10m association, Greedy/Hungarian
matching, and Linear-KF/CTRV-EKF/IMM/KalmanNet estimation exactly as documented
in `config/tracking/camera_replay_presets.yaml`. Mode 3 remains the safe
backward-compatible default. All modes keep yaw unobserved and preserve the
existing AB3DMOT lifecycle; no detector, tracker, estimator, prediction,
occupancy, QoS, frame, timestamp, or production Autoware algorithm changed.

The launch now forwards the selected detector through bag replay. Euclidean
keeps the existing ground-segmentation path; CenterPoint automatically uses
its validated cropped-only input. Modes 4/10 explicitly apply Mahalanobis gate
11.62 plus the 10 m physical cap. Model modes fail before graph startup when
their external artifact/runtime is missing. The wrapper verifies the frozen
CenterPoint and DENSE-KALMANNET-v2 SHA-256 values, while checkpoints and the
3.6 GB MCAP remain ignored and must be copied separately.

Portability: official OpenPCDet is now the pinned `references/openpcdet`
submodule at `233f849829b6ac19afb8af8837a0246890908755`; `references/COLCON_IGNORE`
prevents research repositories from being misidentified as ROS packages.
The standard recursive submodule bootstrap therefore obtains the exact source
on another PC. Runtime still requires the documented CUDA/PyTorch environment
and externally supplied model weights for modes 7-10.

Verification: 230 directly relevant Python tests pass, including every mode's
exact launch wiring, invalid-mode/artifact rejection, detector preprocessing
contract, all AB3DMOT association/estimator/KalmanNet regressions, runner CLI,
and RViz topics. Shell syntax, Python compile, `git diff --check`, and a full
repository-root `colcon list` pass. Isolated installed build of
`ad_lidar_perception` passes; installed launch reports default mode 3 and ten
installed presets; both relevant installed CTests pass. A new live GUI replay
was not claimed because this host still lacks `rosbag2_storage_mcap` and
`compressed_image_transport`; the runner reports those exact missing packages
instead of starting a partial graph.

This interface provides controlled qualitative replay, not a new unified
10-condition quantitative experiment. Historical T-series results still span
different offline/online datasets and protocols and must retain their existing
caveats.

---

## One-click camera + LiDAR tracking RViz replay — PASS

Added the portable, opt-in `scripts/run_camera_lidar_tracking.sh` entrypoint,
`camera_lidar_tracking_replay.launch.py`, and a focused camera/tracking RViz
config. One command now validates the local bag and runtime packages, replays
the front compressed camera and the existing LiDAR source whitelist from one
MCAP player/clock, starts the unchanged MORAI classical Euclidean/Autoware
path plus the frozen experimental AB3DMOT baseline, publishes the established
replay-only `odom -> base_link` anchor, and opens RViz with camera, point cloud,
detections, and both trackers enabled. LiDAR-only replay remains unchanged by
the new strict-default-false `include_front_camera` argument.

Live run started at bag offset 175 s at 1.0x: bag `/tf_static` plus the
established replay-only identity `odom -> base_link` anchor, RANSAC ground
segmentation, Euclidean detection, production-default Autoware tracking, and
the frozen experimental AB3DMOT baseline (Linear KF + Euclidean 3 m +
Hungarian, yaw unobserved). RViz and the front-camera window both opened.
Every pipeline topic was verified with exactly one publisher after removing
stale processes from the preceding, different-dataset RViz run; both tracker
outputs published real `odom`-frame messages and RViz subscribed to both
marker topics.

This is visual/qualitative corroboration only: the bag has no camera
annotations, no `CameraInfo`, no recorded perception output, and no dynamic
localization TF. It is a different moving-ego sequence from the stationary
T-series data, so no T-series metric or accuracy claim transfers to it.

Portability: `package.xml` now declares `rosbag2_storage_mcap` and
`compressed_image_transport`, so the standard bootstrap/rosdep flow installs
them on another Ubuntu 22.04/ROS Humble PC. The 3.6 GB bag remains local and is
explicitly ignored (`*.mcap` plus `/morai_cam4_*/`); it must be copied
separately and is never committed. The launcher has no hardcoded home path and
supports `HEVEN_AD_WS_PATH` for nonstandard workspaces.

Verification: 45 focused source tests pass; isolated `ad_lidar_perception`
build/install passes; installed launch `--show-args` passes; 3/3 relevant
CTest targets pass; shell/Python syntax and `git diff --check` pass. A complete
manual live run of the same camera/LiDAR/TF/detector/tracker/visualizer graph
also passed before consolidation, with exactly one publisher per pipeline
topic and real `odom`-frame outputs from both trackers. The final rosbag2-based
one-click replay cannot be run on this PC until the approved but
password-blocked apt dependencies are installed; the launcher fails early with
the exact missing package instead of starting a partial graph.

---

## P-FIG: Final Presentation Figure Bundle — COMPLETE

Created the collection-only final figure bundle at
`~/heven_presentation_assets/presentation_figures/` and the portable archive
at `~/heven_presentation_assets/presentation_figures.zip`. No experiment,
scientific-result regeneration, image re-encoding, source-pack edit, algorithm
change, commit, or push was performed. Original artifacts were only read and
copied.

Selection followed the final corrected `FIGURE_SHORTLIST.csv` plus
`FINAL_PRESENTATION_AUDIT.md`. The bundle contains 17 MAIN and 23 curated
APPENDIX figures (40 images; 42 regular files including
`FIGURE_BUNDLE_MANIFEST.csv` and `FIGURE_CAVEATS.md`). All 40 source paths and
copies exist, are non-empty, pass PNG signature/chunk-CRC/IDAT decompression
validation, and have byte-identical SHA-256 hashes. ZIP integrity passes. Total
uncompressed regular-file size is 4,995,535 bytes; ZIP size is 4,670,924
bytes.

The obsolete 40-frame ground-segmentation null-result figure is excluded; the
pre-existing corrected T-2B 400-frame presentation-only figure is included.
The 10/10 stability claim uses `ten_seed_validation_distribution.png`; the
5-seed clipping sweep is separately named/caveated as a diagnostic. Camera
MAIN contains exactly the two approved figures, with DIFFERENT-SEQUENCE
boundaries recorded. Dense `camera_lidar_final_summary.png` remains appendix.
All T-14/T-15, offline/online, ground, KalmanNet fairness, association, camera,
and T-13 caveats are recorded in `FIGURE_CAVEATS.md` and per-row in the
manifest.

No shortlisted image is missing. The authoritative shortlist separately
references `final_system_architecture.md` as a text diagram and states that no
existing PNG exists; because this task was collection-only, no new architecture
graphic was invented. No T-15 association-matrix PNG was added because none is
selected by the final shortlist.

## P-FIG result: **COMPLETE**

---

## P-E2: Integrate Camera-LiDAR Qualitative Evidence into the Final Presentation Pack — PASS

Branch: `feat/kalmannet-tracker`. Development remains FROZEN (per T-16).
No experiment run, no algorithm/retraining/threshold change, no repo
source file touched, no commit/push. Two parts, both complete: (1)
resolved every remaining item from the second-round Codex audit
(`FINAL_CODEX_REAUDIT.md`/`FINAL_CODEX_REAUDIT_TABLE.csv`, repo root,
untracked); (2) integrated the completed P-E1 camera-LiDAR validation
(`~/heven_presentation_assets/camera_lidar_validation/`) into the
existing corrected presentation source pack
(`~/heven_presentation_assets/presentation_source_pack/`), in place.

**Part 1 (re-audit corrections)**: T-2B figure/caption/`FIGURE_SHORTLIST.csv`
caveat corrected from "-0.515 on changed frames" to "-0.515 across all
400 frames (OFF 11.97 -> ON 11.455)," re-verified against
`ground_segmentation/t2b_detection_identity_check.md`, figure regenerated;
Hungarian claim (`CLAIM_CHECKLIST.csv` claim 1) fully rewritten (not just
caveated) to remove the "did not materially improve tracking behavior"
overclaim T-3 never measured; GIoU wording narrowed to the precise
"lowest HOTA/highest IDSW in both T-15 rows" scope
(`CLAIM_CHECKLIST.csv`, `GLOSSARY.md`); CenterPoint "better absolute
accuracy" corrected to "lower in-sample/train-scene localization error"
(`GLOSSARY.md`); "9 bugs" corrected to "8 ledger entries / 7 narrative
groups" (`TECHNICAL_APPENDIX_OUTLINE.md`); all previously-malformed rows
in `CLAIM_CHECKLIST.csv`/`NUMBER_CHECKLIST.csv`/`FIGURE_SHORTLIST.csv`
fixed by quoting comma-containing fields (0 bad rows, verified via a
Python `csv` row-width check on all three files); `SLIDE_OUTLINE_20MIN.md`
slides 16-18 now carry the full literal mandated overlap sentence instead
of "Show overlap caveat" shorthand.

**Part 2 (camera integration)**: inspected all 3 user-named candidate
figures directly before selecting (`camera_lidar_final_summary.png`
found too dense/small-text for a live slide, moved to appendix instead of
being forced into the main deck). Selected exactly 2 main-deck figures at
2 locations: `projected_lidar_boxes_camera.png` (Location A, system/setup,
near baseline architecture) and `qualitative_tracking_failure.png`
(Location B, tracking climax, near T-14/T-15) — both explicitly captioned
DIFFERENT-SEQUENCE (moving-ego bag `morai_cam4_20260813_163222`, not the
static T-series scene) with speaker guidance never to attach a T-14/T-15
number to either figure. New `CAMERA_LIDAR_PRESENTATION_GUIDE.md`
(11-section guide). Updated in place: `FIGURE_SHORTLIST.csv` (+6 rows),
`NUMBER_CHECKLIST.csv` (+3 rows), `CLAIM_CHECKLIST.csv` (+1 claim, +1
FORBIDDEN entry), `SLIDE_OUTLINE_15MIN.md`, `SLIDE_OUTLINE_20MIN.md`,
`PRESENTATION_STORYLINE.md`, `PRESENTATION_MASTER_SUMMARY.md`,
`EXPECTED_QA.md` (+2 Q&A), `TECHNICAL_APPENDIX_OUTLINE.md` (+A13),
`ONE_PAGE_CHEATSHEET.md` (+1 never-say item). New
`PRESENTATION_PACK_CAMERA_UPDATE.md` reports the full diff, an 8-item
final check (all 8 pass, including re-confirming the mandatory 100%
train/eval overlap sentence count is unchanged on every existing
CenterPoint slide/figure row), and verdict.

**Verdict: READY FOR FINAL CODEX AUDIT.**

`git status --short` at the end of this task shows only the same
pre-existing unrelated dirty files from every prior session, plus this
`STATUS.md` update — no repo source/algorithm file was touched; all work
lives under `~/heven_presentation_assets/presentation_source_pack/`. Not
committed/pushed, per this task's instruction.

## P-E2 result: **PASS**

---

## T-16: Final Experimental Freeze and Development Handoff — FROZEN

Branch: `feat/kalmannet-tracker`. Not a new experiment — T-15 completed
the final major research question. T-16 audits and consolidates the full
CP-1 through T-15 research program into one reproducible handoff before
presentation-slide work begins. **No detector/association/lifecycle/
threshold change, no KalmanNet/CenterPoint retraining, no new GT
optimization; no repo/algorithm source file touched.**

**Handoff location**:
`~/heven_presentation_assets/final_development_handoff/` — `README.md`
(20-section structure per this task's own spec),
`experiment_timeline.csv` (24 tasks, CP-1 through T-15),
`final_results.csv` (canonical numeric results, sourced from original
JSON/README artifacts — spot-verified directly against
`gt_mot_eval/primary_association_comparison_full.json`,
`state_estimator_gt_comparison/phase8b_apples_to_apples_available_only.json`,
`kalmannet_training_stability/phase16_primary_final.json`, checkpoint
SHA-256s), `claim_ledger.csv` (23 major claims classified SUPPORTED/
SUPPORTED WITH CAVEAT/SUPERSEDED/NOT SUPPORTED/FALSE), `dataset_
limitations.csv` (8 rows), `engineering_forensics.csv` (9 real bugs found/
fixed, none invalidating a final published number), `artifact_manifest.csv`
(per-experiment README/figure/data audit, 24 directories), `superseded_
results.md` (10 do-not-reuse entries), `final_system_architecture.md`
(frozen pipeline + full KalmanNet/CenterPoint provenance).

**Phase 0 audit**: HEAD is `3a3c6b9` (T-9B KalmanNet ROS integration —
confirmed already committed, not assumed from old reports), branch
`feat/kalmannet-tracker`, `git status --short` shows only the same 19
pre-existing chmod-only dirty files plus `STATUS.md` — no legitimate
source changes are pending/uncommitted from any prior task.

**Phase 1 test audit**: full directly-relevant suite re-run, **214/214
pass, unchanged** from every T-9B/T-13/T-14/T-15 session-end check — no
test-count change to report.

**Key consolidated findings** (full detail in the handoff `README.md`):
production Autoware remains fully unchanged throughout the entire
project; experimental AB3DMOT default remains Linear KF + Euclidean-3m-
BEV + Hungarian; DENSE-KALMANNET-v2 (SHA-256 `956604975e...fb7d48`, 10/10
seed-stable, ROS-exact 0.00e+00) and CenterPoint (SHA-256 `466c8181...`,
reproduced this session's T-14, 100% train/eval overlap) are both
strictly opt-in/experimental, never production defaults or validated
generalization claims. Ten historical conclusions are explicitly flagged
SUPERSEDED (ground-segmentation-no-effect, KalmanNet-clearly-beats-KF,
lowest-churn-means-best-identity, KalmanNet-long-gap-robustness, T-12-
collapse-is-intrinsic, T-12.2-instability-is-flakiness, and four
CenterPoint/detection-propagation risks) — full replacement framing in
`superseded_results.md`.

**Commit decision**: only `STATUS.md` plus the same 19 protected files
are dirty — **no algorithm commit created**, per this task's own explicit
instruction not to create a meaningless commit and not to commit
unless separately instructed.

## T-16 result: **DEVELOPMENT FROZEN**

---

## T-15: Detector × Association Interaction Study — PASS, CASE C (multi-factor)

Branch: `feat/kalmannet-tracker`. Goal: explain T-14's central result
(CenterPoint's GT detection-metric win did not propagate to tracking under
Euclidean-3m+Hungarian). Full detail:
`~/heven_presentation_assets/detector_association_interaction/README.md`
(21-section report, 15 figures). **No new detector, no CenterPoint/
KalmanNet retraining, no lifecycle change, no GT-based parameter tuning;
no repo/algorithm source file touched.** Reused T-14's canonical detection
files unmodified (verified: 1,764/1,764 frames each detector, 0
duplicates, checkpoint SHA-256 unchanged).

**6-condition frozen matrix** (Euclidean/CenterPoint detector × Euclidean-
3m/GIoU/Mahalanobis-Hybrid association, all parameters copied unchanged
from T-4/T-5B/T-6, never retuned against any GT score), run offline
(no ROS): all 6 conditions 1,764/1,764 frames, 0 NaN/Inf, byte-identical
reruns (0.0 determinism diff), 100% finite velocity. **A1 (Euclidean
detector + Euclidean assoc) exactly reproduces T-14's own primary result**
(HOTA 0.0631, IDSW 86).

**Central interaction finding**: the detector effect (Euclidean→
CenterPoint) is **consistently negative across all three association
methods** (dHOTA -0.010/-0.0001/-0.008, dAssA -0.039/-0.021/-0.035 for
Euclidean-assoc/GIoU/Mahalanobis respectively) and **far larger** than the
association-method effect within either detector (≤0.003 HOTA swing).
GIoU does **not** rescue CenterPoint (B2 HOTA 0.0528 ≈ B1's 0.0529, IDSW
537 > 443) — ruling out a simple "use geometry-aware association" fix.
Root cause for GIoU specifically: `yaw_measurement_mode=unobserved`
freezes every track's yaw at 0; Euclidean's own detections are also
always yaw=0 (trivial alignment), while CenterPoint's real non-zero yaw
actively suppresses 3D GIoU overlap — confirmed directly (mean GIoU
affinity of accepted matches: Euclidean 0.747 vs. CenterPoint 0.470,
despite CenterPoint's far better raw geometry).

**Candidate-density confirmed as real and large**: CenterPoint's
valid-gated-pair count is 4-30x Euclidean's at every association method
(e.g. 134-172/frame vs. 6-31/frame) — direct evidence for Hypothesis A.
ID-switch forensics found a heavy-tailed density distribution (one
representative switch occurred in a frame with 90 detections/117 active
tracks/722 valid pairs, far above the 16.5/frame mean) — occasional
extreme local spikes disproportionately drive switches.

**Score/density sensitivity (diagnostic, thresholds fixed a priori:
0.1/0.2/0.3/0.5) and a density-controlled top-K diagnostic (explicitly
labeled non-production) both show density reduction helps a lot but not
completely**: at threshold 0.3 (3.25 det/frame, below Euclidean's own
9.15), HOTA nearly doubles (0.0529→0.0975) and *exceeds* the Euclidean
baseline (0.0631); the density-matched top-K diagnostic similarly lifts
HOTA to 0.0813 and IDF1 to 0.0525 (both above Euclidean's own values).
**But AssA stays flat (~0.058-0.066) far below Euclidean's 0.0971 across
the entire useful density range, and IDSW never approaches Euclidean's 86
even at matched density (402 vs. 86)** — density is not the whole story.

**CenterPoint temporal jitter, tested directly and confirmed**:
frame-to-frame position jitter (relative to GT's own motion) is *higher*
for CenterPoint than Euclidean (0.463m vs. 0.288m) despite CenterPoint's
far lower absolute localization error (0.368m vs. 1.176m, T-14) — low
absolute error and low frame-to-frame stability are not the same
property, and this gap plausibly explains the residual AssA/IDSW gap
density-matching alone doesn't close.

**Secondary CenterPoint-yaw experiment (full message semantics, not a
detector-controlled comparison)**: unlocking CenterPoint's real predicted
yaw (`yaw_measurement_mode=detector`) does **not** help GIoU — it gets
slightly worse (HOTA 0.0528→0.0499, IDSW 537→546) — explained by the
same extreme track churn meaning yaw rarely converges before a track
dies, so real yaw mostly adds noise rather than reaping a converged-yaw
benefit. Euclidean association is unaffected either way (it never reads
yaw), as expected.

**T-10 conclusions revisited**: "GIoU worst/near-worst identity
stability" is **ROBUST ACROSS DETECTORS** (confirmed for both). "Euclidean
3m is the strongest association" is **DETECTOR-DEPENDENT** (nearly true
for Euclidean detector, but Mahalanobis Hybrid wins for CenterPoint).
"Mahalanobis reduces churn without improving GT identity" is
**DETECTOR-DEPENDENT IN DETAIL** (identity does move with churn for
Euclidean; for CenterPoint, AssA/HOTA improve but IDSW gets worse).

**Runtime** (association-only, never mixed with detector inference):
construction cost correlates strongly with `n_detections × n_tracks`
(Pearson r 0.94-0.996, all 6 conditions) — GIoU is far the most expensive
for CenterPoint's density (mean 49ms/frame, p95 185ms, max 737ms).

**Classification: CASE C (multi-factor)** — density/ambiguity, a
persistent post-density-control association gap, geometry-aware
association's specific yaw-freeze confound, and CenterPoint's own
temporal jitter all contribute; no single downstream lever (association
metric, score threshold, or yaw semantics alone) fully closes the gap to
Euclidean-detector tracking quality. All results remain
IN-DISTRIBUTION / TRAIN-SCENE (T-14's 100% overlap finding, unchanged and
restated, not re-litigated).

**Tests**: full directly-relevant suite re-run, **214/214 pass**,
unchanged. `git status --short` shows only the same 19 pre-existing
chmod-only dirty files plus this `STATUS.md` update — no repo/algorithm/
config file was touched (all work under
`~/heven_presentation_assets/detector_association_interaction/`).

**Limitations**: still the same single static scene / 100% CenterPoint
train overlap as T-14; the Phase 10 forensic switch trace is an
independent GT-track re-identification method, not numerically identical
to TrackEval's own IDSW definition (both point the same direction, not
claimed as the same count); score/density and top-K experiments are
explicitly diagnostic, no new production threshold selected; the
yaw-convergence explanation for Phase 15's negative result is a plausible,
evidence-consistent mechanism, not independently proven via a dedicated
convergence-time measurement.

**Recommended next task**: (a) a lifecycle-aware follow-up testing
whether delaying GIoU's yaw trust until a track has accumulated N hits
recovers some of the lost real-yaw benefit, since the current all-or-
nothing semantics may discard real information before it can stabilize;
or (b) capture genuinely new, disjoint MORAI scenes (T-14's own
recommendation, still unaddressed) so these density/jitter/geometry
mechanism findings can be tested out-of-sample. Not started this session.

`git status --short` at the end of this task shows only the same
pre-existing unrelated dirty files from every prior session, plus this
`STATUS.md` update — no repo algorithm/config file was touched. Not
committed/pushed, per this task's instruction.

## T-15 result: **PASS**

---

## T-14: End-to-End Detector × Tracker Evaluation with Independent MORAI GT — PASS, CASE B (+ mandatory CASE E qualifier)

Branch: `feat/kalmannet-tracker`. Final major perception experiment before
presentation freeze. Goal: holding the downstream tracker fixed, does
changing the detector (Euclidean vs. CenterPoint) propagate to GT-based
tracking quality? Full detail:
`~/heven_presentation_assets/end_to_end_detector_tracker/README.md`
(22-section report, 15 figures, JSON/CSV — outside repo, not committed).
**No new detection/tracking algorithm, no per-detector tracker tuning, no
KalmanNet retraining/use (Phase 18 secondary 2x2 explicitly dropped, per
its own "do not delay" permission); no repo/algorithm source file
touched.**

**CenterPoint checkpoint reproduced, not newly trained**: no usable
checkpoint existed on disk at task start (the prior "non-smoke" checkpoint
was ephemeral scratch output, already cleaned up) — a genuine
reproducibility gap, not a design choice. Reproduced the exact documented
recipe unchanged (seed 2026, 3 epochs, batch 1, full 1,764-frame train
split): loss 57.58→3.12 (reference: 57.57→3.08, closely reproduced), 0
NaN/Inf. Reproduction shows modest, honestly-reported variance from the
old fingerprint (16.47 vs. 14.75-17 predictions/frame; 93.6% vs. 100%
vehicle class) — not re-rolled or cherry-picked, reported as the single
attempt's real result.

**Window**: full 1,764-frame dataset (not a sub-window) — a pre-freeze
actor-coverage scan (applying T-13's own lesson) found this maximizes
valid vehicle-actor coverage (19/19 actors, 2,844 GT observations, 100% of
all vehicle-class GT in the dataset). All 19 valid actors are `vehicle`
class; the dataset's only pedestrian/obstacle GT belongs to the 2
already-`excluded_actor_ids` actors — vehicle is therefore the only
defensible class domain, confirmed not assumed.

**Input fairness (CASE INPUT-B)**: Euclidean's real production pipeline
includes ground segmentation (RANSAC, T-2's own already-validated node);
CenterPoint's designed input is cropped-only, no ground segmentation —
genuinely different intended pipelines, not an oversight. T-14 evaluates
each detector's own real production path and reports this as a **pipeline
comparison**, not a controlled algorithm-only ablation, throughout.

**Detection GT metrics** (3.0m BEV gate, fixed for both, never tuned per
detector): CenterPoint clearly wins recall (81.1% vs. 71.1%), localization
error (0.368m vs. 1.176m mean, 3.2x tighter), far-range recall (68.7% vs.
51.6%), and vehicle box geometry (length bias -0.34m vs. -2.92m). Euclidean
wins precision (12.5% vs. 7.9%) only because it emits far fewer raw boxes
(9.15/frame vs. 16.47/frame) — both precision numbers are low against the
same T-11-documented annotation-domain mismatch, not treated as an
absolute FP claim.

**Central finding (detection-to-tracking propagation), the same frozen
tracker for both**: CenterPoint's detection-level win does **not**
propagate to tracking. HOTA Euclidean 0.0631 > CenterPoint 0.0529; AssA
0.0971 vs. 0.0582; IDF1 0.0319 vs. 0.0252; IDSW 86 vs. **443** (5.2x
worse); unique tracks 1,749 vs. 5,916 (3.4x more churn). Only DetA (0.0435
vs. 0.0484) and LocA (0.691 vs. 0.803) — the tracking-level echo of the
detection-side win — favor CenterPoint. Mechanism, directly evidenced
(Phase 15 forensic case: 20 unmatched CenterPoint boxes within 5m of one
real matched actor in a single frame): CenterPoint's much higher raw
detection volume floods the same fixed 3.0m gate + Hungarian matcher with
far more competing candidates per real object, driving churn and identity
switching even though each detection is individually better-localized.
**Better detection does not monotonically propagate to better tracking
under an unchanged association configuration** — answered directly, not
assumed either direction.

**Representative forensic evidence**: Euclidean track_id 2 persists across
**all 1,764 frames** of the replay with zero interruption; CenterPoint's
single longest-lived track reaches only 79 observations — a concrete
illustration of the churn gap. 330 GT instances CenterPoint uniquely
covers vs. only 45 Euclidean uniquely covers (asymmetric, both directions
real and shown with specific frame/actor examples).

**Runtime**: CenterPoint ~22.5x slower at the detector stage (mean
131.2ms vs. 5.8ms) and ~11.2x slower end-to-end through the same tracker
(136.6ms vs. 12.2ms) — live-ROS-measured, stamp-preserved true
input-to-tracked-output latency, cross-validated against this project's
already-cited historical CenterPoint median (~136ms).

**Mandatory CASE E qualifier — the single most important limitation**:
`~/datasets/morai_heven/splits/`: `train=1764, val=0, test=0`. CenterPoint's
reproduced checkpoint trained on **all 1,764 frames**; T-14's evaluation
window is the same full 1,764 frames. **Overlap is 100%, not partial** —
every scored frame was also a training example. Every CenterPoint
detection-metric advantage above is **not separable from memorization**
of this one static scene; Euclidean, non-learned, carries no equivalent
risk — the comparison is structurally asymmetric. **All results are
labeled IN-DISTRIBUTION / TRAIN-SCENE EVALUATION.** No generalization or
real-world-deployment claim is made for CenterPoint anywhere.

**Throughput/backpressure**: applied T-13's own root-caused lesson
directly from the start (widened QoS depth ≥250, tooling-only; replay
interval matched to each pipeline's processing cost) — every reported
run shows 0 message loss (source-sent == detector-received ==
tracker-received, explicit counts logged for all 4 live-ROS captures). One
stale-DDS-publisher recurrence was caught immediately via T-13's own
established `ros2 topic info --verbose` + exact-PID-kill + daemon-restart
discipline, before any affected data was collected.

**Classification: CASE B** (detection improves, tracking does not)
**combined with a mandatory CASE E qualifier** (100% train/eval overlap
means the detection-side numbers are pipeline behavior on this scene, not
validated detector-quality evidence) — stated together, never presented
as a plain CASE A/B generalization ranking.

**Tests**: full directly-relevant suite re-run, **214/214 pass**,
unchanged — no repo/algorithm/config file was touched (all new work lives
under `~/heven_presentation_assets/end_to_end_detector_tracker/`, plus a
freshly reproduced, uncommitted CenterPoint checkpoint file). `git status
--short` shows only the same 19 pre-existing chmod-only dirty files plus
this `STATUS.md` update.

**Limitations**: CenterPoint train/eval overlap is total (§20 of the
README, the report's central caveat); the unmatched-detection domain-audit
heuristic is confounded by CenterPoint's own raw detection density; GT
covers only 19 scripted vehicle actors, not general scene geometry, so
precision/F1 are domain-relative, not absolute; the reproduced CenterPoint
checkpoint shows modest honest variance from the old documented
fingerprint (single attempt, not re-rolled); Phase 18's optional
estimator-interaction question (does the propagation finding depend on
KalmanNet vs. Linear KF) remains open; latency figures are single-machine
(RTX 4060, WSL2) only.

**Recommended next task**: either (a) capture genuinely new, disjoint
MORAI scenes to give CenterPoint a real held-out test set before trusting
any detection-quality number, or (b) if presentation time requires a
same-scene finding, present exactly this task's central result —
CenterPoint's detection-side win does not propagate to tracking under the
current fixed-gate association design — with the CASE E overlap caveat
stated in the same breath. Not started this session, per explicit scope
(Phase 18 secondary KalmanNet 2x2 also not started, explicitly dropped
per its own "do not delay" permission).

`git status --short` at the end of this task shows only the same
pre-existing unrelated dirty files from every prior session, plus this
`STATUS.md` update — no repo algorithm/config file was touched. Not
committed/pushed, per this task's instruction.

## T-14 result: **PASS**

---

## T-13: Source-Aligned ROS GT Cross-Check for State Estimators — PASS, CASE B

Branch: `feat/kalmannet-tracker`. Goal: T-9B deferred a real GT cross-check
of the live ROS AB3DMOT estimators (its own README §17), relying instead
on an exact offline/online mechanism-equivalence proof. T-13 does the
deferred work: reuses T-9B's own canonical 200-frame replay (static TF,
`association_metric=euclidean/matcher=hungarian/euclidean_gate_m=3.0/
yaw_measurement_mode=unobserved`, only `state_estimator` varies) for all
4 estimators (Tuned KF/CTRV EKF/IMM/KalmanNet v2), positionally joins
each live `TrackedObjects` output back to independent MORAI GT (T-11's
established `recv_index`<->`frame_idx` join, never wall-clock), and
computes real GT position/velocity error. Full detail:
`~/heven_presentation_assets/ros_gt_crosscheck/README.md` (20-section
report, 10 figures, CSV/JSON — outside repo, not committed). **No
estimator algorithm change, no retraining, no parameter tuning, no
association change; no repo/source file touched.**

**Two real data-collection bugs found and fixed (tooling only, zero repo
files)**: (1) a stray duplicate tracker publisher from an earlier launch
was hidden by a broken `pgrep -af "a\|b\|c"` check — this environment's
`pgrep`, like its already-documented `pkill`, treats `\|` as a **literal**
character, not ERE alternation, so the "exhaustive" check silently
matched nothing and gave false confidence; fixed via exact-PID `kill -9`
(found via plain `ps aux`, not a broken pattern) + `ros2 daemon`
restart. (2) After fixing that, a clean single-publisher KalmanNet run
still lost 29/200 input messages to backpressure — traced to a shallow
depth-10/50 QoS history queue in the (non-repo) replay/recorder tooling
colliding with KalmanNet's higher real end-to-end latency (T-9B: 7.13ms
mean vs. 4.3-6.4ms classical); fixed by widening the tooling's own QoS to
depth-250 `RELIABLE` and slowing the KalmanNet-condition replay interval
to 0.15s — never touching `ab3dmot_tracker_node.py`'s own existing,
unmodified depth-10 subscription. Final clean result: all 4 conditions
195/195 frames, 0 duplicate stamps, exactly 1 publisher throughout.

**Coordinate alignment**: GT/detections are `lidar_link`, tracker output
is `odom`; the T-1C static TF chain (`odom<-base_link` identity,
`base_link<-lidar_link` z=+1.70 identity rotation) makes the transform
exact and closed-form (`(x,y,z)->(x,y,z+1.70)`), no TF-lookup uncertainty.

**A real, honestly-discovered limitation drives this task's headline**:
the common usable frame range (`gt_frame_idx` 1-194, the intersection
across all 4 conditions) contains **only actors 2 (`obstacle`) and 3
(`pedestrian`) — both on T-12's own `excluded_actor_ids` list**
("vehicle-speed kinematics inconsistent with class label"). **None of
T-12's TEST actors (16/20/30) appear in this window** — T-9B's window was
chosen for a runtime comparison, not GT-actor coverage, and the two goals
conflict here. This was discovered during this task, not assumed.

**Primary GT position result (all matched frames, n=38-41 per
estimator)**: Tuned KF 1.000m, CTRV EKF 1.055m, IMM 1.115m, **KalmanNet
v2 1.057m** — all four within a narrow ~1.00-1.12m band, no outlier.
Match rate ~37% (106 GT actor-frame instances total), root-caused to a
real detector-coverage gap (same domain-mismatch T-10 already documented)
— of the 40 instances with any nearby detection, 95-100% were
successfully track-matched per estimator, confirming the low overall rate
is a scene property, not a tracker/estimator failure.

**Offline (T-12) vs. ROS (T-13) comparison — explicitly not
apples-to-apples**: T-12's offline TEST result (KalmanNet 3.065m, >2x
Tuned KF's 1.456m) does **not** reproduce here (KalmanNet 1.049m vs. KF's
0.997m, essentially tied) — but this cannot be attributed to the ROS
integration, since T-9B's Phase 12 already proved online/offline
KalmanNet math is bit-identical (0.00e+00) on shared input. The only
variable that changed is which actors are evaluated (T-12: real
vehicle-class TEST actors where fragmentation sensitivity was found;
T-13: excluded-class actors 2/3 with scripted/discretized motion).
Reported as an **unresolved actor-composition confound**, not as evidence
against T-12's finding.

**Velocity result reported but flagged as not a clean signal**:
KalmanNet's lower RMSE (10.00 m/s vs. 16.7-18.6 m/s classical) against
GT velocity for actors 2/3 is **not** claimed as a real accuracy win —
these actors' finite-differenced GT velocity is the same
kinematically-inconsistent signal T-12 already flagged as its reason for
excluding them; matching a noisy target better is not evidence of better
velocity estimation.

**Identity-consistency (secondary context only)**: KalmanNet shows the
most ID switches on actor 3 (10 vs. 4-7 for the others, n=14 matched
frames) — consistent with T-12.1/T-12.3's documented fragmentation
sensitivity, but on a different actor population and too small a sample
to be confirmatory alone.

**Classification: CASE B** — GT-based ROS metrics partly agree with the
offline picture (no estimator catastrophically fails) but do not
reproduce T-12's KalmanNet-fragmentation headline on this window, for a
traceable, honest reason (actor-composition confound, not an integration
defect — Phase 12 already proved the integration exact).

**Tests**: full directly-relevant suite re-run, **214/214 pass**,
unchanged (211 pre-existing + verified via the same run). `git status
--short` at the end of this task shows only the same pre-existing
unrelated dirty files (0 content diff, permission-mode only) from every
prior session, plus this `STATUS.md` update — no repo/algorithm/config
file was touched.

**Limitations**: single fixed replay window (reused from T-9B per this
task's own Phase 1 "reuse" decision) with zero TEST-split actors; GT
velocity for both usable actors is inherently noisy (§ above); per-actor/
per-regime samples are small (13-41); KalmanNet's own frame range is
offset by 1 frame from the other three (residual, explained artifact of
the QoS fix); the direct-trace phase reuses T-9B's already-exact Phase 12
equivalence rather than re-deriving it on this specific actor.

**Recommended next task**: a true apples-to-apples ROS-vs-offline
comparison needs a replay window containing T-12's actual TEST actors
(16, 20, 30) — locate their frame ranges in the full 1764-frame canonical
dataset and repeat this exact pipeline on that window, which would let
the offline-vs-ROS comparison test T-12's fragmentation finding directly
in the live ROS path without the actor-composition confound found here.
Not started this session, per explicit scope (this task's window was
fixed before the actor-composition issue was discovered).

`git status --short` at the end of this task shows only the same
pre-existing unrelated dirty files from every prior session, plus this
`STATUS.md` update — no repo algorithm/config file was touched. Not
committed/pushed, per this task's instruction.

## T-13 result: **PASS**

---

## T-9B: Opt-In ROS2 Integration of DENSE-KALMANNET-v2 — PASS, CASE A (with documented scope limits)

Branch: `feat/kalmannet-tracker`. Goal: integrate T-12.3's frozen
`DENSE-KALMANNET-v2` checkpoint as an opt-in AB3DMOT `state_estimator`.
Full detail:
`~/heven_presentation_assets/kalmannet_ros_integration/README.md`
(23-section report). **No retraining, no KalmanNet redesign, no
production/default change, no ROS-based tuning.** This task DID modify
production repo source (`ab3dmot_core.py`, `ab3dmot_config.py`,
`ab3dmot_tracker_node.py`, `ab3dmot_tracker.launch.py`) — the first
T-9-series task to do so, since T-9A/T-9A.1/T-12/T-12.1/T-12.2/T-12.3
were all offline-only.

**Baseline**: repo was clean (only STATUS.md + 19 protected files dirty)
at task start — no Phase 0.5 safe checkpoint was needed; baseline commit
`e49c3d9`.

**Interface**: new `KalmanNetEstimator` implements the exact same
`predict`/`update`/`position`/`velocity`/`yaw`/`dimensions`/
`*_covariance`/`predicted_bev_*`/`is_finite` interface as
LinearKF/EKF/IMM. **State ownership**: `x,y,vx,vy` from KalmanNet;
`z,vz,l,w,h,yaw` from an internally-composed `LinearKFEstimator` ("side"),
never reporting the side's own x/y/vx/vy. **Covariance**: Option B —
covariance fields populated from the side estimator only for message-
schema completeness, never claimed as KalmanNet uncertainty;
`state_estimator="kalmannet"` + `association_metric="mahalanobis"` is
**structurally rejected** (`ValueError`, tested) — primary comparison
uses Euclidean 3m + Hungarian throughout.

**A real bug found and fixed**: `KalmanNetGRU` (unchanged, from
`kalmannet_core.py`) stores its recurrent hidden state directly on the
shared weight module — sharing one network across tracks silently leaked
hidden state between them (confirmed empirically before the fix: running
a second track measurably changed a first track's output). Fixed at the
ROS-integration call site only (never editing `kalmannet_core.py`) by
having each `KalmanNetEstimator` own its own hidden-state tensor,
swapped onto the shared module only around the one call site that
touches it. Verified by a dedicated isolation test (now passing:
identical output whether run alone or interleaved; fresh zero hidden
state after track death).

**Lifecycle**: `predict()` caches (does not commit) the analytical
`x_prior`; `update()` consumes it directly for the learned correction
(no double-predict); a new `Track.finalize_predict_only()` (called for
every unmatched track each frame) commits the cached prior with no
network call — exactly mirroring offline missing-measurement handling.

**Mandatory equivalence test (Phase 12): PASS, exact** — offline
`KalmanNetFilter.step()` vs. the new online `KalmanNetEstimator` produce
**bit-for-bit identical (0.00e+00 diff)** x/y/vx/vy on all 3 T-12 TEST
sequences, including one with 24 missing-measurement frames.

**Live ROS smoke test**: launches cleanly, checkpoint logged exactly
once (hash `956604975e...fb7d48` matches T-12.3's frozen manifest
exactly), publisher isolation verified (0 stray pre-launch, exactly 1
publisher post-launch, every condition), 0 NaN/Inf, frame=odom, 100%
finite velocity.

**Canonical 4-way runtime comparison** (same 200-frame replay, same
Euclidean+Hungarian association, only `state_estimator` varies) — live
ROS latency (mean/p95 ms): Tuned KF 4.64/8.10, CTRV EKF 4.30/8.97, IMM
6.38/14.22, **KalmanNet 7.13/14.55** — modestly higher (~1.1-1.7x) than
every classical estimator but real-time practical. Offline controlled
measurement: KalmanNet per-track cost ~0.35-0.5ms (roughly constant),
total per-frame cost scales linearly with track count (0.5ms at 1 track
-> 12.4ms at 35 tracks).

**Not run this session (documented scope decisions, not omissions)**:
GT cross-check (the replay's reassigned wall-clock stamps break the
original frame-index-to-GT alignment; the exact offline/online
equivalence result is a stronger guarantee than an approximate
re-measurement would be); CUDA comparison (no engineering time spent,
per explicit instruction); RViz screenshot (no screenshot capability,
same precedent as every prior RViz-adjacent session).

**Classification: CASE A** (full integration pass — per-track isolation
correct after the found-and-fixed bug, offline/online equivalence exact,
ROS runtime stable, latency practical, zero regression to existing
estimators) **with the documented scope limits above** stated explicitly
rather than silently treated as fully resolved.

**Production interpretation (explicit, per this task's own instruction)**:
production tracker remains Autoware; experimental AB3DMOT default
remains `linear_kf`; KalmanNet remains strictly opt-in research. Passing
integration tests and near-tied offline performance are **not** a claim
that KalmanNet replaces KF.

**Tests**: KalmanNet core 16/16, new `test_ab3dmot_kalmannet.py` 19/19
(checkpoint loading/validation, config validation, multi-track isolation,
lifecycle), T-12.1 ablation 7/7, T-12.2 dense-baseline 7/7, T-12.3
training-stability 10/10, AB3DMOT core regression 154/154 — **213/213
total, zero regressions** to the existing linear_kf/ekf/imm paths
(verified: `ab3dmot_core.py` still imports cleanly with no torch
installed, confirming the lazy-import design never requires torch
unless `state_estimator="kalmannet"` is actually selected).

**Limitations**: see "not run this session" above; latency-vs-track-count
uses a controlled offline measurement, not live-ROS-instrumented data;
the side-estimator composition means KalmanNet tracks pay the cost of
two estimators' machinery, reflected in the measured latency gap.

**Recommended next task**: either (a) a defensible GT cross-check using
a replay method that preserves original frame-index-to-GT alignment
(e.g. reusing T-11's own offline-replay-with-real-stamps convention
instead of a fresh-wall-clock republish), or (b) if KalmanNet's opt-in
research value is considered sufficient as-is, no further T-9-series
work is required — this task's own success criteria are all met. ROS
integration is not a basis for any production/default change. Not
started this session.

`git status --short` — see below for exact source-file diff (this task's
own explicit modifications: `ab3dmot_core.py`, `ab3dmot_config.py`,
`ab3dmot_tracker_node.py`, `ab3dmot_tracker.launch.py`,
`test_ab3dmot_kalmannet.py` new — plus this `STATUS.md` update and the
same 19 pre-existing protected files). Not committed/pushed, per this
task's instruction (no explicit request to commit was given).

## T-9B result: **PASS**

---

## T-12.3: KalmanNet Training Stability / Seed-Instability Root-Cause — PASS, CASE A / ROS-A

Branch: `feat/kalmannet-tracker`. Goal: root-cause and fix T-12.2's
unresolved 2/5-seed catastrophic-divergence problem. Full detail:
`~/heven_presentation_assets/kalmannet_training_stability/README.md`
(22-section report). **No architecture change, no TEST-based selection,
no ROS integration.**

**Reproduction/determinism**: T-12.2's 5 seeds reproduced to full
floating-point precision (fully deterministic given a seed; CPU/CUDA/
cudnn-deterministic all bit-identical across repeated trials — rules out
nondeterministic execution).

**Root cause (CASE A — gradient explosion)**: every non-zero seed's
first abnormal event occurs at the identical point — epoch 0, step 0,
the very first backward pass. `output_fc` (the gain head) is the largest
gradient contributor (~73-82%) at every observed event. Severity, not
mere occurrence, predicts outcome (clean dose-response: 4e4/6.8e6
recover, 1.3e9/1.5e11 never recover). Hidden-state and gain norms stay
bounded even in failed seeds (ruled out as the mechanism); the
discriminating signal is the state-estimate magnitude on the actual
training trajectory (52-53, physical, in stable seeds vs.
2,332-543,202, unphysical, in failed ones).

**Order vs. init (Phase 4)**: failure follows initialization, not data
order — init=1 fails under every tested order; init=0 survives under
every tested order, including the "failing" one.

**LR sweep — a real, counter-intuitive negative result**: reducing LR
makes stability monotonically *worse* (0.001: 3/5 stable -> 0.0001: 0/5,
even seed 0 fails). LR reduction explicitly ruled out.

**Gradient-clipping ablation**: every tested clip norm (10.0/5.0/1.0)
achieves 5/5 stability with no performance tradeoff. Selected minimal
fix: **`max_norm=10.0`** (gentlest of the three), LR/architecture/init
unchanged.

**10-seed validation (0-9) with the frozen fix**: **10/10 (100%)
stable** — complete resolution. Validation loss mean 5.25, std 0.50
(tight).

**DENSE-KALMANNET-v2 frozen** (winner seed 1, val loss 4.215, manifest
written before any TEST use). **Held-out TEST** (measurement-available,
n=130) position RMSE: Measurement 1.451, Tuned KF 1.456, CTRV EKF 1.490,
IMM 1.455, v1 1.467, **v2 1.467** — no regression vs. v1; velocity RMSE
v2 2.462 is **better than v1's 2.657**, closer to Tuned KF's 2.416.
**Secondary all-10-seed TEST check**: mean 1.465, std 0.012 — extremely
tight deployment consistency, not used for selection. Runtime unaffected
(v2 CPU mean 0.165ms, in the same range as v1).

**Classification: CASE A (gradient explosion), ROS-A** (>=9/10 seeds
stable, held-out performance competitive, inference practical) — T-9B
opt-in ROS integration is now justified, using `DENSE-KALMANNET-v2`.

**Tests**: KalmanNet 16/16, T-12.1 ablation 7/7, T-12.2 dense-baseline
7/7, AB3DMOT core 154/154 — all unchanged/pass. 10 new focused tests
(`test_training_stability.py`: deterministic seed control, gradient-
clip bounding, objective failure-threshold classification,
validation-only checkpoint-selection invariants) — all pass.

**Limitations**: TEST still 3 actors; `clip=10.0` chosen from a
30-epoch, 3-value sweep, not exhaustive; combined init+clip interaction
not tested (correctly — clipping alone was already fully sufficient).

**Recommended next task**: proceed to T-9B opt-in ROS integration using
`DENSE-KALMANNET-v2` (`~/heven_presentation_assets/kalmannet_training_stability/checkpoint/`),
explicitly as an opt-in state estimator alongside the tuned KF, never a
default replacement. Not started this session.

`git status --short` shows only the same 19 pre-existing dirty files
from every prior session, plus this `STATUS.md` update — no repo source
file touched. Not committed/pushed.

## T-12.3 result: **PASS**

---

## T-12.2: Rebuild and Freeze the Dense-Trained KalmanNet Baseline — PASS, CASE B / ROS-D(->B)

Branch: `feat/kalmannet-tracker`. Goal: turn T-12.1's fragmentation
diagnosis into one official, reproducible, validation-selected
KalmanNet checkpoint (`DENSE-KALMANNET-v1`). Full detail:
`~/heven_presentation_assets/kalmannet_dense_baseline/README.md`
(21-section report). **No architecture change, no ROS integration, no
TEST-based tuning.**

**Split/dense data**: T-12's frozen split reused unmodified, zero actor
leakage confirmed in code. T-12.1's dense-run criterion reused unchanged
(`min_contiguous_dense_run_len=15`), applied to **both** TRAIN (37 runs/
1,273 transitions) and, new this task, **VALIDATION** (6 runs/225
transitions) — T-12.1 had left VAL fragmented.

**Critical new finding — multi-seed instability, previously only
anecdotal**: trained 5 predefined seeds (0-4), identical architecture/
data. **2 of 5 (40%) diverged catastrophically** (val pos RMSE 15,860
and 145) while 3 converged well (1.05, 4.58, 1.24) — directly confirming
T-12.1's own flagged small-sample-instability concern, now demonstrated
systematically. Winner selected purely on validation loss (seed 0, val
pos RMSE 1.053), before any TEST use — verified by a dedicated ordering
test and a manifest-written-before-TEST-artifact timestamp test.

**Primary result** (measurement-available TEST, n=130, position RMSE):
Measurement 1.451, Tuned KF 1.456, CTRV EKF 1.490, IMM 1.455, **Dense
KalmanNet v1 1.467** — essentially tied with every classical method
(within ~1%), sometimes winning individually (actor 30: 0.433 vs KF's
0.444; decelerating: 1.429 vs 1.447). Velocity: Dense KalmanNet's best
individual result (2.657 vs Tuned KF's 3.028) — a real, modest edge.
**T12_FULL (historical) remains at 3.065** — the fragmentation fix
survives a properly frozen, validation-selected checkpoint, not just
T-12.1's single hand-picked run (T12_FULL -> Dense v1: 3.065 -> 1.467).

**Gain behavior**: Dense KalmanNet v1 applies 78.3% of innovation
(T12_FULL: 24.3%, T9A original: 51.0%) — preserves/exceeds T-12.1's own
restored measurement-trust finding.

**Missing-measurement stress (synthetic, unchanged protocol)**: T-12.1's
conclusion reproduced with the official checkpoint — Tuned KF still
reacquires faster at long gaps (24-frame gap, post-gap RMSE: KF 1.23 vs
Dense 5.62), though Dense KalmanNet tracks KF closely at short/medium
gaps. No missing-measurement-superiority claim is supported.

**Classification: CASE B** (near-tied with classical estimators, not a
clear win) combined with **ROS-D taking priority as a gating concern**
(seed instability must be addressed first) **transitioning to ROS-B**
once stability is fixed — KalmanNet would then be justified as a
demonstrable research-value alternative, not a baseline replacement.

**Tests**: KalmanNet suite 16/16, T-12.1 ablation-harness suite 7/7,
AB3DMOT core suite 154/154 — all unchanged/pass. 7 new focused tests
this session (`test_dense_baseline.py`: no-GT-init guarantee,
validation-only seed selection, checkpoint hash/provenance) — all pass.

**Limitations**: seed instability unresolved (this task diagnosed and
reported it, did not fix it — out of scope, a training-recipe
improvement is the natural next step); TEST still only 3 actors;
left-turn remains unevaluable; the good result depends on having
correctly validation-selected the winning seed, not on "dense training"
alone being reliably reproducible.

**Recommended next task**: either (a) a training-stability follow-up
(gradient clipping, learning-rate warmup, or a formalized multi-seed-
and-validate-select step baked into the production recipe) before any
T-9B ROS decision, or (b) if stability work is deferred, proceed to
T-9B as an explicitly opt-in, demonstration-only integration (ROS-B),
never as a default/baseline replacement for the tuned KF. Not started
this session.

`git status --short` shows only the same 19 pre-existing dirty files
from every prior session, plus this `STATUS.md` update — no repo source
file touched. Not committed/pushed. No ROS integration started.

## T-12.2 result: **PASS**

---

## T-12.1: KalmanNet Data-Fragmentation / Missing-Measurement Ablation — PASS, CASE A

Branch: `feat/kalmannet-tracker`. Goal: explain WHY KalmanNet reversed
from a near-tie with tuned KF (T-9A.1: 0.854m vs. 0.855m) to a >2x loss
(T-12: 3.065m vs. 1.456m). Full detail:
`~/heven_presentation_assets/kalmannet_ablation/README.md` (20-section
report). **No ROS integration, no association change, no architecture
change in the primary experiment.**

**Reproduction (Phase 1)**: both prior results reproduce cleanly. Found a
previously-undocumented confound: T-9A's own eval initialized every
sequence from **exact GT state** (verified empirically,
`preds[0]==x_true[0]`); T-12 correctly avoids this. Quantified as a real
but modest (1-25%) effect, not dominant.

**Transfer matrix (run early, cheap, before any retraining)**: the
untouched T-9A model already degrades ~3x just moving to T-12's test
domain (2.90-2.94m, clean held-out check) — real domain-shift. But
**T-12's own retrained model is worse than the untouched T-9A model on
the same T-12 test set** (4.08-4.57m) — direct evidence retraining itself
hurt, not just a harder domain.

**Dataset audit**: T-9A's training data is **100% measurement-dense by
construction** (0/1,011 missing); T-12's is 21.8% missing (up to 44-frame
gaps) — the largest structural difference found. A methodological
inconsistency was also found and fixed: T-12's own README compared an
uncentered noise RMS (0.902m) against T-9A.1's centered sigma_z (0.546m)
— different statistics. Computed identically, **T-9A and T-12 train noise
are nearly the same** (0.546 vs 0.575m); the real shift is **T-12's own
val/test actors being ~1.7x noisier than its own train actors** (0.97m) —
a consequence of T-11's frozen split, not a T-9A-vs-T-12 vintage effect.
The Tuned KF (never retrained) degrades by almost exactly this same 1.7x,
isolating a residual ~2.1x that only affects KalmanNet.

**Central ablation result**: 5 conditions, identical architecture/
hyperparameters (7,016 params confirmed identical everywhere), only
training-data construction varied. On the primary measurement-available
TEST metric (n=130, position RMSE): Tuned KF 1.456, T9A_ORIGINAL 1.574,
**T12_FULL 3.065**, **T12_DENSE 1.476**, **T12_DENSE_MATCHED 1.478**,
T12_FULL_MATCHED 4.375. **Dense retraining (both full-size and
transition-count-matched to T-9A's own 1,004) restores KalmanNet to
within 1-2% of the tuned classical KF** — and since DENSE_MATCHED has
*fewer* transitions than the fragmented full set yet performs far better,
this rules out "just needs more data." Confirmed consistently across
every evaluable regime (straight/accelerating/decelerating) and every
individual TEST actor.

**Mechanism (learned gain)**: T12_FULL applies only 24.3% of the
innovation on average vs. T9A_ORIGINAL's 51.0% and T12_DENSE's 72.1% —
it learned to systematically distrust normal measurements, a concrete
explanation for the RMSE gap.

**Gap-robustness re-examined (synthetic, controlled)**: T-12's original
"KalmanNet's one advantage is long-gap robustness" claim rested on one
natural anecdote (actor 16). A controlled synthetic dropout test (6 gap
lengths x 2 actors, fixed methodology) found the **opposite** pattern
consistently — Tuned KF reacquires fastest at every gap length (e.g. 24
frames: KF post-gap 1.23m vs. T12_FULL's 28.92m). The anecdote does not
generalize and should not be presented as a general KalmanNet property.

**Classification: CASE A (fragmentation dominant)** — with the domain-
noise-shift component (§8) and the now-doubtful gap-robustness claim
(§12) as real nuances, not full unconditional dominance.

**Next recommendation: NEXT A with caveat** — KalmanNet is competitive
after fixing training-data construction alone; any future ROS
integration should use a dense-trained checkpoint, never the current
frozen `T12_FULL`, and the gap-robustness claim should be dropped from
presentation until a larger study exists.

**Tests**: KalmanNet suite 16/16 pass, AB3DMOT core suite 154/154 pass
(both unchanged); T-9A's own pre-existing hidden-state-leakage regression
test still passes; 7 new focused tests added this session
(`test_ablation_harness.py`, all pass) covering predict-only/hidden-state
isolation, dense-run extraction, and deterministic subsampling. Phase 3
code audit found **no implementation bug** (rules out CASE E).

**Limitations**: TEST still only 3 actors (unchanged, out of T-12.1's
scope); T12_FULL_MATCHED's poor result may partly reflect small-subsample
training instability, not fragmentation alone; synthetic gap test used
only 2 actors/1 insertion point per length; controlled noise-transfer
(Phase 10) was not run, judged lower value than the gap curve given the
noise-shift finding already has a clean quantitative decomposition.

`git status --short` shows only the same 19 pre-existing dirty files from
every prior session, plus this `STATUS.md` update — no repo source file
touched. Not committed/pushed. No ROS integration started.

## T-12.1 result: **PASS**

---

## T-12: GT-Based Controlled State Estimator Comparison — PASS, CASE D

Branch: `feat/kalmannet-tracker`. Goal: a fair, common-state
([x,y,vx,vy]) offline comparison of Tuned Linear KF, CTRV EKF, CV+CTRV
IMM, and KalmanNet against independent MORAI GT, using T-11's frozen
canonical dataset/split (no new capture, per explicit instruction). Full
detail: `~/heven_presentation_assets/state_estimator_gt_comparison/README.md`
(21-section report, figures/CSV/JSON/checkpoint — outside repo, not
committed). **No association/lifecycle/detector/ROS change, no AB3DMOT
10D-state reuse — a genuinely common 4D problem for all four methods.**

**Dataset/split**: T-11's `split_policy.json` reused unmodified (train 12
/ val 4 / test 3 actors). **Real fairness bug found and fixed**: one
actor's whole 79-frame run had zero matched Euclidean detections —
untruncated init would have silently leaked GT position as the
"measurement-based" init. Fixed by truncating every sequence to its first
valid measurement and dropping sequences with none (3 dropped). Final: 29
sequences / 18 actors / 2,618 observations / 22.7% missing measurements
(real, not synthesized).

**Estimators**: CTRV EKF Jacobian verified against numerical
finite-difference gradient before use; synthetic constant-turn test
confirms correct convergence (yaw 2.248 vs. true 2.250 rad). IMM is a new
6D common-state (`[x,y,vx,vy,yaw,yaw_rate]`) mixing implementation
(T-7B's own common state was 11D and AB3DMOT-tied) that deliberately
reuses T-7B's own documented fix (CTRV reseed copies yaw from the common
anchor, never re-derives via `atan2` every cycle — the fix that resolved
T-7B's ±8 rad/s instability). Tuned KF recalibrated from scratch on T-11's
train/val population (`sigma_a=5.0`, isotropic R `sigma_z=0.902m` —
larger/noisier than T-9A.1's own 0.546m). KalmanNet: T-9A's exact
architecture (7,016 params, confirmed identical), retrained on the new
train split (60/60 epochs, early stop not triggered, val loss
524.4→65.2 monotonic).

**Central finding**: the pooled TEST metric (all 154 frames) makes
KalmanNet look best (RMSE 4.57 vs. Tuned KF's 4.73) — but this is
**entirely an artifact of one actor's 23-frame measurement gap**. On the
apples-to-apples subset (measurement-available frames only, n=130,
identical basis for every method): **Tuned KF and IMM are essentially
tied with raw detector measurement** (1.456/1.455 vs. 1.451m) and
**KalmanNet is more than 2x worse** (3.065m) — confirmed consistently
across every individually-evaluable regime (straight/accelerating/
decelerating, §12-14 of the README). KalmanNet's one genuine, reproducible
advantage is **robustness during the long gap itself** (9.27m vs. Tuned
KF's 11.73m dead-reckoning drift) — real but narrow, not a general
accuracy win.

**Left-turn**: **not evaluable on TEST** (0 frames — T-11's split
deliberately put no turn-validated actors in test); a secondary,
explicitly-labeled non-primary VAL-split analysis (actor 31, n=28) is
reported separately. **Right-turn/stationary/transition**: not evaluable,
consistent with T-11's own finding.

**IMM model probability**: mean `mu_CV` is *higher* during GT-validated
`left_turn` (0.724) than `straight` (0.719) — does **not** cleanly track
genuine turning here, echoing T-7B's own real-data finding; reported as
found, not spun.

**Runtime (CPU)**: Tuned KF 0.014ms mean, EKF 0.026ms (1.9x), IMM 0.121ms
(8.8x), KalmanNet 0.290ms (21x) — same ordering as every prior task.

**Classification: CASE D** — no global winner. Classical methods
(essentially tied with each other) clearly win the primary
measurement-available regime across every evaluable motion regime;
KalmanNet's only advantage is long-gap robustness. CTRV/IMM's
turning-specific value remains unconfirmed (no validated TEST left turns).

**Tests**: KalmanNet suite 16/16 pass, AB3DMOT core suite 154/154 pass,
both unchanged (no source file touched — this task built new, standalone
code under `~/heven_presentation_assets/`, reusing `kalmannet_core.py`
and referencing `ab3dmot_core.py`'s IMM design only as read-only
reference).

**Limitations**: TEST is only 3 actors/154 frames, one dominates the
pooled metric (directly addressed via the available-only comparison);
KalmanNet trained on only 22 (smaller, more fragmented) sequences vs.
T-9A's 41 — may partly explain its weaker available-frame result, not
disentangled from a genuine limitation; bootstrap uncertainty explicitly
small-sample caveated (3 actors, ≤27 unique resamples); far-range TEST
coverage is 0 frames.

**Recommended next task**: either (a) run T-11B's still-pending manual
MORAI capture protocol to add right-turn/stationary/second-scene data,
then re-run this same T-12 harness for a fuller regime comparison, or (b)
investigate why the retrained KalmanNet underperforms its own T-9A
result on measurement-available frames (smaller/more fragmented train set
vs. genuine architecture/data-fit limit) before drawing a final KalmanNet
verdict. Not started this session, per explicit scope.

`git status --short` at the end of this task shows only the same 19
pre-existing dirty files from every prior session, plus this `STATUS.md`
update — no repo source file was touched. Not committed/pushed.

## T-12 result: **PASS**

---

## T-11B: Targeted MORAI Motion Capture — BLOCKED at manual capture checkpoint

Branch: `feat/kalmannet-tracker`. Continues T-11 (CASE B) to close the
right-turn/stationary/transition/second-scene gaps before T-12. New output
root created (`~/heven_presentation_assets/motion_gt_expansion_v2/`), T-11's
`canonical/` v1 dataset untouched/preserved.

**Pipeline audit (Phase 2) found T-11's own Phase 8 protocol was not fully
executable**: it referenced `tools/morai_dataset_exporter/` from
`metadata.json`'s `config` path, which lives in a workspace
(`heven_ad_2026_ws`) that does not exist on this machine and has no
equivalent anywhere in this repo — corrected this session with tooling
actually verified present: `/ad/dev/objects` GT topic (`ad_morai_bridge_dev`,
`development.yaml`), and a real, already-built, tested recorder
(`ros2 run ad_morai_bridge_dev ad_morai_perception_bag`, fail-closed,
provenance-tracked, records GT + Euclidean detections + TF + ego status
together). Also found a promising pre-provenanced scripted-actor route,
`kcity-roundabout-loop` (`ad_morai_bridge_dev/config/actor_presets.yaml`,
via `ad_morai_actor spawn-npc`/`route-npc`) — a closed roundabout loop that
should provide sustained turning in a consistent direction without manual
driving, pending live verification that the spawned actor's state actually
surfaces on `/ad/dev/objects` (not yet confirmed — no live simulator
available in this environment).

**Blocked at Phase 3 (manual capture checkpoint), per explicit task
instruction not to fake capture completion.** MORAI is a GPU/Unity
simulator process not present in this sandboxed environment. Full exact
terminal-by-terminal protocol (bridge, LiDAR/Euclidean, recorder, NPC-actor
spawn commands with verification, manual-driving fallback route, stop/
verify steps) written to
`~/heven_presentation_assets/motion_gt_expansion_v2/phase3_manual_capture_protocol.md`.
Resumes at Phase 6 once the user reports a real captured `run_directory`.

**Tests**: none run (no source changed). No AB3DMOT/KalmanNet file touched.

`git status --short` shows only the same 19 pre-existing dirty files from
every prior session, plus this `STATUS.md` update — no repo file modified
otherwise. Not committed/pushed. T-12 not started.

## T-11B result: **BLOCKED (awaiting real user MORAI capture)**

---

## T-11: Motion-Rich / GT-Rich MORAI Evaluation Dataset Expansion — PASS, CASE B

Branch: `feat/kalmannet-tracker`. Goal: T-9A.1/T-10/T-10.1 all hit a data
ceiling (800-frame GT window, 12 actors, 0 pedestrians; 2 right-turn
segments; T-9A.1's 221-frame/9-turning-frame test set) — this task audits
and expands the underlying MORAI GT/detector dataset **before** any further
estimator comparison (deferred to T-12). Full detail:
`~/heven_presentation_assets/motion_gt_expansion/README.md` (18-section
report, figures/CSV/JSON/canonical dataset — outside repo, not committed).
**No new tracking algorithm, no association tuning, no KalmanNet ROS
integration, no headline KF/EKF/IMM/KalmanNet comparison (that is T-12).**

**Existing-data audit (before any new capture)**: `~/datasets/morai_heven/`
contains **1764** label files (not just the 800-frame window T-8A/T-10
used) — a single scene (`static_20260805_003151`), all in `train` split
(val/test empty by the exporter's own design). GT ordering by
`header_stamp_ns` is 0-duplicate/0-rollback and identical to
`splits/train.txt`'s own file order (direct `diff`, confirmed). 21
distinct `actor_id`s total vs. 12 in the old window; 9 actors are
**entirely** outside the old window (including the dataset's only
pedestrian- and obstacle-class actors, fully explaining T-10's zero-
pedestrian finding).

**Real finding, reported plainly**: actor 3 (`class_name=pedestrian`) and
actor 2 (`class_name=obstacle`) both show smooth, continuous, vehicle-
speed motion (9-31 m/s / 8-19 m/s over ~100-120m paths) — almost certainly
a MORAI scenario class-slot mislabel, not genuine pedestrian/obstacle
kinematics. This dataset has **effectively zero genuine pedestrian or
static-obstacle GT**, independent of frame window.

**Motion-regime classifier (GT-direct, thresholds fixed before viewing
counts)**: a first unsmoothed attempt classified on raw ~0.13s
consecutive-observation deltas and rejected 778/833 (93%) of candidate
segments (median rejected duration ~0.13s — flapping almost every frame,
the same jitter failure mode T-8A/T-7A.5 hit, now at the GT-derivative
level). **Fixed** via a 0.6s centered rolling-window smoothed
classification signal (segment duration/path/heading accounting still
uses raw unsmoothed steps). Final accepted segments (full 1764-frame
dataset): accelerating 42, decelerating 22, left_turn 16, straight 14,
stationary 1, right_turn 1; 251 rejected candidates logged separately; **0
straight<->turn transitions** (verified genuine — this route consistently
enters/exits turns during accel/decel, not from constant-speed straight
motion, not a classifier artifact).

**Manual validation (Phase 15, 22-segment stratified sample, direct raw
per-step trajectory inspection)**: 18/22 (81.8%) accepted; the dataset's
**only** right_turn candidate (actor 31) **fails** manual validation (raw
per-step speed spikes to 53.7 m/s against a 36.9 m/s segment median — a
physically implausible artifact, not smooth motion) — **zero validated
right turns exist in the entire dataset**, not one.

**GT/detector domain mismatch (deepened on full dataset)**: mean 9.39
detector objects/frame vs. mean 1.63 in-domain (vehicle+pedestrian) GT
objects/frame; of 16,567 detector objects, 11.3% match a GT vehicle, 0.16%
match the pedestrian, 88.4% unmatched — **not** assumed to be false
positives (GT only labels scripted actors), consistent with T-10's own
DetA-collapse explanation.

**Range coverage** (2.0m threshold, full dataset): near 88.7% (n=956), mid
70.3% (n=1352), far **5.5%** (n=569) — notably worse than T-10.1's
tracker-matched ~18-19% far coverage (different methodology: raw
single-frame nearest-detection vs. tracker-association match fraction —
not directly comparable, both reported honestly). Not root-caused this
session, per explicit scope (no association-gate tuning).

**Canonical dataset built** at
`~/heven_presentation_assets/motion_gt_expansion/canonical/` (`gt/`,
`detections/`, `frame_map.csv`, `actor_manifest.csv`, `motion_segments.csv`,
`split_policy.json`, `provenance.json`, `hashes.txt`) — does **not**
overwrite T-8A/T-9A/T-10/T-10.1 canonical inputs. Detections are a
**freshly captured** 1764-frame Euclidean ROS replay this session
(`euclidean_clustering.launch.py` + `ad_publish_morai_frames --count 1764`,
recorded on `/ad/perception/objects/detected`): 1764/1764 messages, 0
duplicate, 6 replay-wall-clock rollbacks (publish-timing jitter, does not
affect the positional frame-index join). Phase 18 smoke test (loader
join, NaN/Inf check, split-actor presence check) **PASS** — no estimator
comparison run, per explicit scope.

**Actor-level train/val/test split policy** (frozen before any T-12 score,
`phase6_split_policy.md`/`canonical/split_policy.json`): 12 train / 4 val /
3 test vehicle actors (2 class-mislabeled actors excluded from the primary
pool); route-level separation not possible with the current single-scene
data.

**New-capture decision: CASE EXISTING-B.** Existing data substantially
improves left-turn/accel/decel/straight/actor coverage but right-turn (0
validated), stationary (1 accepted), genuine pedestrian (0), and
second-scene/route diversity remain clearly insufficient and require new
capture. A full manual capture checkpoint (terminals, commands, target
route including deliberate right-turn/stationary/pedestrian content,
verification steps, output location) is written in
`phase8_manual_capture_protocol.md` — **not executed this session**, per
this task's explicit instruction to stop at a manual checkpoint rather
than simulate a drive.

**Old vs. expanded**: GT observations (vehicle+pedestrian) grow from 1,147
(old 800-window) to 2,877 (full dataset, 2.5x); actor pool 12 -> 21 total
(19 usable vehicle). Old regime counts (T-8A) came from a tracker-
trajectory classifier, not GT-direct — the two are methodologically
different, not a strict apples-to-apples delta (stated explicitly in the
README).

**Figures**: all 10 required figures generated under
`~/heven_presentation_assets/motion_gt_expansion/`.

**Classification: CASE B** — coverage improved substantially (actor
count, GT observation count, left-turn/accel/decel/straight richness) but
right-turn/stationary/pedestrian/cross-route gaps remain. T-12 may proceed
on the expanded dataset with these limitations stated explicitly (e.g. a
straight/left-turn/accel/decel-focused comparison using the frozen
actor-level split), or a targeted new capture (Phase 8 protocol) can be
run first to close the right-turn/stationary/pedestrian gaps before a
fully general T-12 claim.

**Tests**: no AB3DMOT/KalmanNet source file touched (dataset-construction
task only); this session's only ROS activity was a fresh Euclidean
detector replay capture (existing built node, no code change) — cleaned up
via exact-PID kill after capture, verified no stray processes remained.

**Limitations**: single scene, no val/test route in current data; the 2
class-mislabeled actors reduce real per-class diversity below nominal
count; manual validation covered a 22-segment stratified sample (81.8%
accept), not exhaustive; far-range coverage (5.5%) not root-caused; 0
straight<->turn transitions limits any future transition-specific
IMM-mode-switching evaluation; 6 replay-wall-clock rollbacks in the fresh
capture are publish-timing artifacts, not physically meaningful.

**Recommended next task**: either (a) run the Phase 8 manual capture
protocol to close the right-turn/stationary/pedestrian/cross-route gaps,
then proceed to T-12, or (b) proceed directly to T-12 (KF/EKF/IMM/
KalmanNet comparison) on the expanded dataset using the frozen actor-level
split, explicitly scoped to straight/left-turn/accel/decel regimes with
the right-turn/stationary/pedestrian limitations stated up front. Not
started this session, per explicit scope.

`git status --short` at the end of this task shows only the same
pre-existing unrelated dirty files from every prior session, plus this
`STATUS.md` update — no repo algorithm/config file was touched; this
session's only repo-adjacent activity was launching an existing, unmodified
ROS node (`euclidean_clustering.launch.py`) for a fresh detection capture,
fully cleaned up afterward. Not committed/pushed, per this task's
instruction.

## T-11 result: **PASS**

---

## T-10.1: GT association ranking robustness — PASS, CASE B (mostly robust)

Branch: `feat/kalmannet-tracker`. Goal: test whether T-10's central
finding (Euclidean 3m+Hungarian > Mahalanobis Hybrid 10m > GIoU on
GT-based identity metrics, contradicting the old churn-based ranking)
survives (a) the semantically-correct yaw handling
(`yaw_measurement_mode=unobserved`, T-7A.5) and (b) evaluator sensitivity
checks. Full detail:
`~/heven_presentation_assets/gt_mot_eval_yaw_unobserved/README.md`
(16-section report, figures/CSV/JSON — outside repo, not committed). **No
new tracking algorithm, no ROS/KalmanNet integration, no association
hyperparameter tuned against a GT metric, no estimator retuning
(T-9A.1's separately-tuned KF explicitly not reused here).**

**T-10 reproduction**: exact — re-evaluated T-10's own stored tracker
outputs, byte-identical HOTA/IDSW to T-10 (0.0488/0.0574/0.0555,
88/50/54) before any yaw-mode change.

**Yaw-observability finding — a clean, mechanistically-explained
no-op**: `LinearKFEstimator`'s `unobserved` mode drops the yaw
measurement row entirely, freezing yaw at its birth value forever; birth
yaw is always `0.0` (the detector's own placeholder) in both modes, and
`detector` mode's fed-in yaw is also always `0.0` — so both modes
converge to the same value. **Empirically confirmed to floating-point
precision**: tracker output (position, yaw, track IDs, unique-track
counts) is byte-identical between `detector` and `unobserved` modes for
all 3 association configs — every GT-based metric is therefore also
identical (delta=0.0 everywhere).

**Threshold-sensitivity sweep** (predefined {1.0,1.5,2.0,2.5,3.0}m, fixed
before viewing results): **HOTA and IDSW rankings are fully robust**
(Euclidean wins 5/5 thresholds on both). **AssA is mostly robust**
(Euclidean 3/5, GIoU competitive at the two tightest thresholds).
**IDF1 is genuinely not robust** — the Euclidean-vs-Mahalanobis-Hybrid
ordering on IDF1 flips depending on threshold (Mahalanobis Hybrid wins
IDF1 at tight thresholds; margins are small everywhere).

**Spatial-bias secondary check**: the T-10 radial (-0.7 to -0.74m) /
tangential (+0.55 to +0.61m) bias is consistent across all 3 configs
(a detector/GT-geometry effect, confirmed again, not an association
artifact). A uniform, non-tuned bias correction raises all metrics
substantially but **preserves the HOTA/AssA/IDSW ranking**; IDF1 flips
slightly toward Mahalanobis Hybrid under correction — reinforcing that
IDF1 specifically is the least robust metric here. Primary (uncorrected)
result remains official.

**Actor/range coverage**: per-actor and per-range match *coverage* (does
a GT observation get matched to any track at all) is nearly identical
across all 3 configs (near ~99%, mid ~81-82%, far ~18-19%, dropping with
range as expected) — confirms Euclidean's edge is about identity
continuity given a match, not about detecting more objects. Zero
pedestrian coverage in this specific 800-frame window (real limitation).

**Representative-case search**: no "wrong-but-long" Mahalanobis Hybrid
failure was found in its single longest run (residual stayed bounded
0.68-2.0m throughout, no anomalous jump) — reported honestly as not
found, not fabricated; suggests the effect (if real) is distributed
across many medium tracks rather than one dramatic failure, consistent
with Mahalanobis Hybrid having the **worst fragmentation count of all 3**
(73, above even GIoU's 72) despite its lowest raw churn — track
persistence != correct identity persistence.

**Classification: CASE B (mostly robust)** — GIoU-worst is the most
robust finding in the whole task (never wins HOTA/IDSW at any threshold).
Euclidean-beats-Mahalanobis-Hybrid is robust on HOTA/AssA(mostly)/IDSW
but explicitly **not** robust on IDF1. The T-10 headline remains useful
with this qualification, not as an unconditional per-metric claim.

**Tests**: no AB3DMOT/KalmanNet source file touched (evaluation-only
task); T-10's own TrackEval adapter reused unchanged.

**Limitations**: small GT sample (12 actors, unchanged from T-10); IDF1
specifically not robust; case-3 forensic search checked only one track,
not exhaustive; bias correction is a diagnostic only, not validated as
objectively correct.

**Recommended next task**: (a) an exhaustive (not single-track)
Mahalanobis Hybrid forensic sweep to settle the "wrong-but-long"
question, (b) grow GT coverage (more scenes/pedestrian frames) before
treating the IDF1-specific margin as resolved, or (c) if T-9B proceeds,
prioritize an association-method ablation over further estimator tuning,
since association method dominates identity quality far more than yaw
semantics or (for HOTA/IDSW) matching threshold. Not started this
session, per explicit scope.

`git status --short` at the end of this task shows only the same
pre-existing unrelated dirty files from every prior session, plus this
`STATUS.md` update — no repo algorithm/config file was touched. Not
committed/pushed, per this task's instruction.

## T-10.1 result: **PASS**

---

## T-10: MORAI GT-based MOT evaluation — PASS, CASE B

Branch: `feat/kalmannet-tracker`. Goal: every prior association/tracker
comparison (T-1 through T-8A) used descriptive metrics only (unique
tracks, lifetime, very-short fraction) — no identity GT had been
connected. T-10 builds a GT-based MOT evaluation harness using the
independent MORAI simulator labels (`~/datasets/morai_heven/labels/*.json`,
persistent `actor_id`) via the project's pinned `references/trackeval`
submodule, and re-evaluates GIoU/Euclidean/Mahalanobis-Hybrid association
with real HOTA/IDF1/MOTA/IDSW metrics. Full detail:
`~/heven_presentation_assets/gt_mot_eval/README.md` (18-section report +
audit/alignment/adapter docs, figures/CSV/JSON — outside repo, not
committed). **No tracking algorithm changed, no parameter tuned against a
GT metric, no KalmanNet change.**

**GT/alignment**: 1,764-label full-dataset audit (21 actors, 3 classes;
`obstacle` class found to be 0% inside the detector's own z-crop, `vehicle`/
`pedestrian` kept — target policy fixed before any tracker score).
Reused T-9A's frame-index join (`window_sample_ids.txt`, T-8A's 800-frame
window) — 800/800 matched, 0 missing, 0 duplicates, numerically
cross-checked (median nearest GT-detection distance 1.34m). Ran the
tracker **offline** (no ROS/TF) so GT and tracker output share
`lidar_link` natively — avoided a TF chain entirely; validated a -0.74m
radial bias (expected LiDAR near-surface clustering effect, not a bug)
and a +0.67m tangential bias (smaller, not fully explained).

**Adapter**: calls TrackEval's own `HOTA`/`CLEAR`/`Identity` classes
directly with a hand-built per-sequence data dict (BEV-distance
similarity, 2.0m threshold) rather than reimplementing metrics or
building a full on-disk MOTChallenge adapter. **5/5 mandatory synthetic
hand-checkable cases passed** (perfect/missed/FP/ID-switch/fragmentation)
— found and documented a genuine TrackEval quirk along the way (`CLEAR.Frag`
does not increment for a totally-empty gap frame, only a gap frame with
an unrelated other hypothesis).

**Primary result (800 frames, 12 GT actors, association fixed
estimator=linear_kf/yaw=detector, only association metric varies)**: HOTA
— GIoU 0.049, **Euclidean 3m+Hungarian 0.057 (best)**, Mahalanobis Hybrid
10m 0.056. IDF1/AssA/IDSW all rank Euclidean best, GIoU clearly worst
(88 IDSW vs Euclidean's 50). **DetA is crushed for all 3 (0.025-0.030)**
from a detector-vs-GT domain mismatch (detector ~14-17 objects/frame vs.
GT's 1.43 in-domain objects/frame — MORAI only labels its own scripted
actors, not all clusterable scene geometry) — affects all 3 configs
almost identically (DetRe/DetPr nearly constant), so AssA/IDF1/IDSW
remain the informative axis for comparing association methods despite
the crushed DetA/negative MOTA.

**Central finding — direct answer to T-10's own question**: Mahalanobis
Hybrid's previously-lowest churn (T-6: 592 vs Euclidean's 701 unique
tracks) does **not** correspond to better GT-based identity quality —
Euclidean is marginally ahead of Mahalanobis Hybrid on HOTA/AssA/IDF1/IDSW
despite more raw churn, and Mahalanobis Hybrid has the **worst**
fragmentation count of all 3 (73, above even GIoU's 72). GIoU's own
previously-reported extreme churn **is** confirmed as real identity
instability (worst on every GT metric, 205 directly-traced ID switches
vs Euclidean's 81/Mahalanobis's 89 via an independent corroborating
method). Margins between Euclidean and Mahalanobis Hybrid are small
relative to the tiny GT sample (12 actors) — reported as a real reversal,
not a decisive one.

**Optional estimator comparison** (association fixed to Euclidean 3m):
Linear KF marginally best/tied on every identity metric vs CTRV EKF/IMM
(HOTA range 0.053-0.057, ~7% relative spread) — estimator choice matters
far less than association-method choice here, consistent with every
prior finding that this scene's turning content is modest.

**Classification: CASE B** — GT metrics partly agree with prior
descriptive conclusions (GIoU-worst is confirmed) but reveal an important
tradeoff (Mahalanobis Hybrid's low-churn "win" over Euclidean does not
hold up against GT-based identity metrics) — not a full CASE A
confirmation, not a full CASE C contradiction (GIoU's ranking is
confirmed, not contradicted).

**Not attempted**: Autoware comparison (Phase 16) — would need a new
`odom`-frame TF/GT-alignment pipeline, disproportionate new engineering
per this task's own explicit permission to skip.

**Tests**: TrackEval adapter validated via 5 mandatory synthetic
hand-checkable cases (5/5 pass, `phase9_synthetic_validation.py`). No
AB3DMOT/KalmanNet source file was touched — regression suites unaffected
(not re-run this session; no algorithm code changed).

**Limitations**: DetA/MOTA absolute values are not meaningful in
isolation (domain mismatch, only relative comparison across the 3 configs
is defensible); 12-actor GT sample is small; `yaw_measurement_mode=detector`
used deliberately to match T-6's original convention, not the T-7A.5 fix;
BEV-only (no height/IoU-3D) matching.

**Recommended next task**: either (a) re-run this same GT-based
evaluation with `yaw_measurement_mode=unobserved` and/or the T-9A.1 tuned
KF to see whether the T-7A.5/T-9A.1 corrections change the identity-metric
ranking, or (b) grow GT coverage (more scenes/actors) before treating the
Euclidean-vs-Mahalanobis-Hybrid margin as decided, or (c) if T-9B ROS
integration proceeds, reuse this task's TrackEval adapter to give
KalmanNet's eventual ROS-integrated tracking output a real identity-aware
evaluation instead of only the offline position/velocity RMSE T-9A/T-9A.1
already have. Not started this session, per explicit scope.

`git status --short` at the end of this task shows only the same
pre-existing unrelated dirty files from every prior session, plus this
`STATUS.md` update — no repo algorithm/config file was touched. Not
committed/pushed, per this task's instruction.

## T-10 result: **PASS**

---

## T-9A.1: Fair classical-KF calibration vs. KalmanNet — PASS, CASE B

Branch: `feat/kalmannet-tracker`. Goal: T-9A's classical KF baseline was
deliberately **untuned** (`sigma_a=2.0`, `r_std=1.1`) and lost even to raw
measurement (RMSE 1.115m vs 0.866m) — not a fair opponent for KalmanNet.
This task builds a **defensibly calibrated** KF (Q/R selected on
train/validation only, test actors 1/18 never touched during tuning) and
re-runs the held-out comparison. Full detail:
`~/heven_presentation_assets/kf_calibration/README.md` (19-section
report + search/audit/failure-analysis docs, figures/CSV/JSON — outside
repo, not committed). **No KalmanNet retraining, no repo file touched.**

**Methodology**: exact T-9A split reused (verified via
`kalmannet_split.json` hash + actor/frame-count cross-check against
`training_provenance.json`). Untuned KF reproduced exactly (RMSE 1.1154 /
2.1200, matching T-9A's 1.115/2.120). Train-only measurement-residual
analysis (n=1,011) found empirical `sigma_z=0.546m` — isotropic R
justified (x/y std ratio 0.94, correlation -0.26). Reused T-9A's own
structured white-noise-acceleration `Q(dt,sigma_a)` unchanged (only
`sigma_a` free). Bounded log-grid search (45 candidates: `sigma_a` in
[0.1,20], R scale in [0.25x,4x] of train-derived sigma_z) on validation
(164 frames) only, selection metric fixed in advance (validation position
RMSE). **Selected and frozen**: `sigma_a=10.0`, isotropic `R`
(`sigma_z=0.137m`), `P0` unchanged — written to `selected_kf_config.json`
before test evaluation was unlocked.

**Held-out test (n=221, test actors 1/18, single evaluation, frozen
params)**: position RMSE — measurement 0.866, untuned KF 1.115, **tuned
KF 0.855**, KalmanNet 0.854. **The tuned KF closes the entire position
gap to KalmanNet** (0.001m difference — statistically indistinguishable,
actor-level bootstrap ranges overlap almost completely) and now clearly
beats raw measurement, unlike the untuned baseline. Velocity: KalmanNet
retains a real but modest ~5% edge (1.371 vs 1.446 m/s).

**Regime-specific (turn-rate >0.5 rad/s)**: genuinely mixed, not spun —
KalmanNet wins straight motion (both position and velocity), the
**tuned KF wins turning** (both position and velocity), n=9 turning
frames explicitly flagged as too small to trust either direction.

**Why the untuned KF lost to measurement — root-caused, not just "Q/R
were untuned"**: untuned R variance was 4.05x the train-measured value;
validation selected `sigma_a` 5x larger than the untuned guess; direct
steady-state evidence — the untuned filter applied only 28.0% of each
innovation vs. 77.0% for the tuned filter (structural under-correction/
lag), full derivation in `untuned_kf_failure_analysis.md`.

**Generalization stress test (reconstructed — original generator script
no longer exists, reconstruction validated by reproducing the original
untuned-KF numbers within 0.286m mean absolute deviation, flagged as
approximate)**: a genuine, mechanistically-explained reversal — T-9A's
untuned KF beat KalmanNet in 2/3 synthetic stress conditions; the
**tuned** KF loses to KalmanNet in all 3, because calibrating R down to
the real (low) in-distribution noise floor makes the filter over-trust
measurements when out-of-distribution noise triples. A KF tuned tightly
to one noise regime is less robust to a regime shift than a conservative
one — reported as found.

**Classification: CASE B** — KalmanNet retains a small held-out
advantage (essentially tied on position, a real but modest edge on
velocity) but the result is genuinely mixed by regime and the tuned KF
is now a legitimate near-tie on the primary (position) metric that T-9A's
headline was built on. T-9A's original "KalmanNet clearly beats KF" claim
does **not** survive fair calibration on position; it survives partially
on velocity.

**Tests**: existing KalmanNet suite (`test_kalmannet_core.py`, via the
torch-enabled venv) 16/16 pass, unchanged. AB3DMOT core suite (8 modules
runnable without a full ROS environment: `test_ab3dmot_core`,
`_geometry`, `_association`, `_association_metrics`, `_ekf`, `_heading`,
`_hybrid_gate`, `_imm`) 154/154 pass, unchanged (the 2 remaining modules
needing `rclpy` were not runnable in this session, consistent with every
prior session's documented ROS-sourcing requirement — not a regression).

**Limitations**: test sample is small (2 actors, 221 frames, turning
n=9); validation selection sits on a broad, flat loss plateau (`sigma_a`
itself is not narrowly identified, only the qualitative direction —
trust measurements more — is robust); stress-test numbers are
approximate (reconstructed generator, not byte-identical replay).

**Recommended next task**: either (a) grow the real dataset (more
actors/scenes) so the turning regime and the stress test can be evaluated
without the current small-n/reconstruction caveats, or (b) if T-9B ROS
integration proceeds, use this task's frozen tuned KF (not the untuned
T-9A baseline) as the classical-filter comparison point going forward.
Not started this session, per explicit scope (calibration/fairness
experiment only, no algorithm change, no ROS integration).

`git status --short` at the end of this task shows only the same
pre-existing unrelated dirty files from every prior session, plus this
`STATUS.md` update — no repo algorithm/config file was touched. Not
committed/pushed, per this task's instruction.

## T-9A.1 result: **PASS**

---

## T-9A: KalmanNet offline dataset + prototype — PASS, CASE A (qualified)

Branch: `feat/kalmannet-tracker` (branched from `e5f6beb` on
`feat/ab3dmot-tracker`). Goal: establish a defensible KalmanNet training/
eval dataset and build an OFFLINE prototype before any ROS integration.
**No ROS integration, no change to the production Linear KF/EKF/IMM
baseline.** Full detail: `~/heven_presentation_assets/kalmannet/README.md`
(17-section report + `kalmannet_architecture_audit.md` +
`ground_truth_audit.md`, figures/JSON/CSV/checkpoint — outside repo, not
committed).

**GT audit (highest-priority phase)**: `~/datasets/morai_heven/labels/*.json`
(1,764 files) contain independent MORAI simulator GT per object
(persistent `actor_id`, x/y/z/yaw/dims, own real `header_stamp_ns`) via a
`map->odom->base_link->rear_axle_link->lidar_link` transform chain --
**structurally independent of the Euclidean detector**. Verified the
detection stream's own `stamp_ns` is a replay-time artifact (not usable
for dt); `window_sample_ids.txt` (T-8A's own capture-order record) gives
the correct frame-index join key instead. **PATH A selected** (real GT
available and synchronizable).

**Reference**: new pinned submodule `references/kalmannet`
(KalmanNet_TSP, commit `828a2cf5`, architecture #2; no LICENSE file,
reference-only). This task's own module
(`ad_lidar_perception/ad_lidar_perception/kalmannet_core.py`, new,
ROS-independent) implements a documented single-GRU simplification
(7,016 params) preserving the reference's structural property: analytical
`f`/`h`, only the Kalman gain is learned.

**Dataset**: 4-state CV problem (`[x,y,vx,vy]`/`[x,y]`), GT-derived
velocity via real (non-uniform) dt finite-differencing. 56 segments /
1,396 frames / 11 actors after gate+length filtering. **Actor-level
train(7)/val(2)/test(2) split, zero leakage** -- train 1,011 frames, val
164, test 221 (held-out actors 1 [strong turn] and 18 [straight]).

**Training**: seed 0, bounded 60 epochs (early-stop not triggered, val
loss still improving), all Phase-13 sanity checks passed (overfit-one-
sequence 7038.9->7.3, monotonic train loss, checkpoint-reload verified
byte-identical, 0 NaN/Inf, real-dt-variation stable, no hidden-state leak
across sequence boundaries).

**Held-out test (real, n=221, never-tuned Q/R baseline)**: KalmanNet beats
both the classical KF baseline and the raw-measurement baseline on every
position/velocity metric (position RMSE 0.854 KalmanNet vs 1.115 KF vs
0.866 measurement-only; velocity RMSE 1.371 vs 2.120). **Honestly
reported, not spun**: the untuned classical KF is actually *worse* than
raw measurement on position here -- not re-tuned post hoc, per this
task's explicit "do not over-tune Q/R against test data" instruction.

**Generalization stress test (SYNTHETIC, explicitly labeled)**: at 2.7x
training-calibrated measurement noise, KalmanNet stays numerically stable
in all 3 conditions (no divergence) but does **not** uniformly beat KF
out-of-distribution -- wins only the turning condition, loses both
straight conditions. Reported as found, not smoothed over.

**Runtime**: KalmanNet CPU ~8x slower than classical KF (0.18ms vs
0.02ms mean) but sub-millisecond; CUDA slower than CPU at this tiny
per-step scale (kernel-launch overhead dominates) -- reported despite
being counter-intuitive.

**Tests**: new `test_kalmannet_core.py` (16/16 pass, run via the
torch-enabled `heven-centerpoint` venv -- deliberately **not** wired into
`CMakeLists.txt`/`colcon test`, since `kalmannet_core.py` imports torch at
module level and the system ROS Python has no torch, mirroring
`centerpoint_ros.py`'s own established torch-isolation pattern). Full
pre-existing AB3DMOT suite re-run as regression: **195/195 pass**,
unchanged.

**KalmanNet readiness: CASE A, qualified** -- real independent GT exists
and offline evaluation is meaningful (genuine held-out win), but the
real-data sample (11 actors) is small and generalization is mixed, so
T-9B should proceed as an **experimental, opt-in path only** (same
pattern as CTRV EKF/IMM), not a baseline replacement.

**Recommended next task**: T-9B experimental opt-in ROS integration of
this same 4-state KalmanNet CV model (mirroring the `state_estimator`
config pattern already used for `linear_kf`/`ekf`/`imm`), explicitly
scoped to the demonstrated real-data regime and carrying forward both the
small-sample and mixed-generalization caveats -- or, before that, capture
additional real MORAI actor trajectories (more scenes/routes) to grow
PATH A's sample size beyond 11 actors.

`git status --short` at the end of this task shows only the same
pre-existing unrelated dirty files from every prior session, plus this
`STATUS.md` update, `.gitmodules`/`references/README.md` (new
`references/kalmannet` submodule), and the two new
`kalmannet_core.py`/`test_kalmannet_core.py` files -- no AB3DMOT/tracking
production source file was touched. Not committed/pushed, per this
task's instruction.

## T-9A result: **PASS**

---

## T-8A: Motion-regime evaluation dataset — PASS, CASE B (KalmanNet readiness)

Branch: `feat/ab3dmot-tracker`. Goal: build a reusable MORAI motion-regime
dataset (straight/turning/transition) so KF/EKF/IMM (and later KalmanNet)
can be evaluated on real sustained turning, since T-7A.5/T-7B found the
original 400-frame canonical recording's turning content modest. **No
KF/EKF/IMM algorithm code touched, per this task's explicit instruction —
dataset/evaluation-tooling only, all three estimators reused unchanged
from T-7A/T-7A.5/T-7B.** Full detail:
`~/heven_presentation_assets/motion_dataset/README.md` (14-section
report, figures/JSON/JSONL/CSV — outside repo, not committed).

**Canonical recording**: frames 200-999 (800 frames) of the same
1,764-frame `static_20260805_003151` MORAI export every prior session has
used, selected via a GT-label mining pass (never previously done) that
found real sustained-turn vehicle actors outside the previously-used
0-400 window (e.g. 6.24 rad yaw range / 401 m path over 430 obs).
800/800, 0 dup, 0 rollback, hash `f5b439e6...b72864b`, orientation
`UNAVAILABLE` for all 10,234 objects (re-confirmed, not assumed) — all
estimator runs used `yaw_measurement_mode="unobserved"`.

**Regime counts** (strict sliding-window classifier, two real jitter/
mis-association failure modes found and fixed with documented,
physically-motivated safeguards — not count-tuned): 61 straight / 19
left_turn / **2 right_turn** / 41 transition / 0 stationary (123 total
segments). `right_turn` shortfall reported honestly as a real property
of this left/counterclockwise-dominant route. Manual validation: 13/123
(10.6%) individually inspected and accepted; remaining 110 pass the
strict automatic classifier only (labeled as such, not misrepresented).

**Phase 13 (offline + live, all 3 conditions, Euclidean+Hungarian
association fixed, `yaw_measurement_mode=unobserved`)**: offline
diagnostics (`collect_offline.py`, direct-from-JSONL, no ROS) — 9,533 /
9,515 / 9,530 match records (linear_kf/ekf/imm), 0 NaN/Inf, deterministic.
Live ROS (fresh publisher/recorder per condition, publisher-isolation
verified via `ros2 topic info --verbose` + a full daemon restart to rule
out stale-cache false positives, exact-PID kill verified twice 2s apart
before each launch) — **800/800 messages for all three conditions**, 0
NaN/Inf. One real operational finding this session: `pkill -f "a\|b"`
silently matches nothing in this environment (pkill's ERE treats `\|` as
a literal pipe, not alternation) — every cleanup after this was switched
to `pgrep`/exact-PID `kill -9` plus double verification.

**Regime-specific residuals** (associated-measurement one-step prediction
residual, not ground-truth error; nearest-measurement-position matched to
manifest segments, ≤2.0 m tolerance): IMM has the lowest median residual
of the three estimators in every regime (e.g. straight: 0.364 m IMM vs.
0.404 m Linear KF vs. 0.429 m EKF). Full per-regime table in README §8-10.

**IMM mu_CV/mu_CTRV by regime — genuine, unresolved, not spun toward
IMM**: mu_CV is *higher*, not lower, during verified left/right turns
than during straight motion (mean mu_CV: 0.561 straight vs. 0.654
left_turn vs. 0.708 right_turn) — the opposite of naive expectation, and
not explained by this task (flagged as an open limitation, plausibly
related to T-7B's own finding that CV's flexibility fits jitter better
than CTRV detects real curvature, but not verified here).
Before/during/after a validated `transition` window shows no clean
dip-and-recover (mu_CV drifts gently upward 0.592 → 0.619 → 0.658 across
all three phases) — messier than T-7B's synthetic validation, reported as
such.

**Figures**: all 9 required figures present under
`~/heven_presentation_assets/motion_dataset/` (4 distribution figures
from before this checkpoint, reused unchanged; 5 new this session:
`estimator_residual_by_motion_regime.png`,
`imm_probability_by_motion_regime.png`, and 4
`representative_{straight,left_turn,right_turn,transition}.png` cases
built from real validated manifest segments — no jitter cherry-picked for
visual drama).

**Live ROS track-level summary** (secondary, no accuracy claim): unique
tracks 701/719/704 (linear_kf/ekf/imm), very-short fraction ~0.28-0.29
all three — nearly identical across conditions as expected, since
association is fixed to Euclidean+Hungarian for all three.

**Tests**: full directly-relevant suite (10 AB3DMOT test modules) —
**195/195 pass**. `git status --short` shows only the same pre-existing
unrelated dirty files from every prior session in this branch, plus this
`STATUS.md` update — no algorithm source file was touched.

**KalmanNet readiness: CASE B.** Useful turning exists (61 straight / 19
left_turn / 41 transition, IMM mechanism validated end-to-end on real
data with 0 NaN/Inf across all three live conditions) but coverage
remains limited — specifically `right_turn` (2 segments, far below a
usable per-regime evaluation split) and only 10.6% manual-validation
coverage. KalmanNet development can proceed for straight/left_turn/
transition, but additional motion data (more clockwise/right-turning
traffic, or a second scene) is recommended before any per-regime
KalmanNet comparison claims completeness. Based on measured coverage, not
chosen because IMM "worked" — §11's finding is in fact an unresolved,
not-obviously-IMM-favorable result.

**Recommended next task**: either (a) begin KalmanNet baseline
implementation scoped to straight/left_turn/transition regimes with the
right_turn limitation stated up front, or (b) investigate §11's
mu_CV-higher-during-turns finding directly (would need real-data
per-object curvature ground truth, not available in this GT-label-mined
dataset as currently used), or (c) capture a second MORAI route with more
right-turning traffic to close the CASE B gap. Not started this session,
per this task's explicit scope (dataset/tooling only).

Not committed/pushed, per this task's instruction; no checkpoint commit
has been made since `4d26e0e` (out of scope for this task, to be done
separately after this final report).

## T-8A result: **PASS**

---

## T-7B: IMM(CV + CTRV) state estimator — PASS, CASE A (mechanism) / CASE B persists (dataset)

Branch: `feat/ab3dmot-tracker`. Implemented an experimental 2-model IMM
(Model 1 = CV Linear KF, Model 2 = CTRV EKF) opt-in state estimator
(`state_estimator="imm"`) and ran a controlled Linear-KF-vs-CTRV-EKF-vs-IMM
comparison, association fixed to Euclidean+Hungarian, `yaw_measurement_mode
="unobserved"` throughout (T-7A.5). Full detail:
`~/heven_presentation_assets/imm/README.md` (`imm_design.md`,
`imm_estimator_compatibility_audit.md`, figures/JSON/JSONL — outside
repo, not committed).

**Architecture**: common 11-dim mixing state
`[x,y,z,yaw,l,w,h,vx,vy,vz,yaw_rate]`; Jacobian-based (not diagonal-copy)
covariance transforms both directions; standard IMM cycle (mixing ->
model predict -> model update -> innovation log-likelihood -> posterior
-> combined output with between-model spread term), not a heuristic.
Transition matrix `[[0.95,0.05],[0.05,0.95]]` (documented baseline, not
tuned); neutral 50/50 initial probabilities.

**Two real bugs found and fixed during implementation** (documented
in-code with the empirical evidence that found them): (1) an initial
CTRV-reseed design that re-derived heading via `atan2(vy,vx)` every
mixing cycle made `yaw_rate` wildly unstable on synthetic constant-turn
testing (±8 rad/s swings on a true 0.6 rad/s turn) — fixed by using the
common state's own already-correctly-circular-mixed `yaw` directly, plus
an isotropic low-speed covariance proxy instead of an arbitrary small
constant that had been collapsing the reseeded speed uncertainty from
~10,000-scale down to 10 and crippling the Kalman gain. (2) A latent
T-7A.5 bug: `LinearKFEstimator`'s own yaw-unobserved reduced update
called filterpy's `update()` with a 6-dim measurement against a
fixed-`dim_z=7` filter — **always raised `ValueError`**, never actually
exercised end-to-end before IMM called it directly (T-7A.5's real runs
only used Linear KF in `"detector"` mode). Fixed with a manual reduced
update mirroring EKF's already-correct math.

**Tests**: new `test_ab3dmot_imm.py` (31 tests: 8 state-transform + 7
probability + 3 mixing + 4 behavior + 9 tracker-integration). Full
related suite **207/207 pass** (176 pre-existing + 31 new); default
Linear KF confirmed unchanged.

**Synthetic validation** (`imm_synthetic_model_probabilities.png`): straight
sequence — mu_CV rises 0.5→0.95; turn sequence — mu_CTRV initially
declines then reverses and gains ground as evidence accumulates;
straight→turn→straight — mu_CV rises to 0.74, **falls to 0.49 during the
turn**, **recovers to 0.63** after — clean, smooth, mechanistically
correct regime-tracking (the strongest evidence the IMM math is right).

**Real-data result (400-frame canonical, hash `63198cd0...4a0c6`,
identical to every prior task)**: global mu_CV mean 0.432 / mu_CTRV mean
0.568 — **CTRV dominates overall**, the opposite of the naive
CASE-B-implies-CV-dominance expectation. Straight-vs-verified-turning
split (T-7A.5's strict classifier, 75 consistent turning runs): mean
mu_CV 0.691 (straight) vs. 0.645 (turning) — **correct direction, modest
effect size**. Interpretation: CTRV's overall dominance is likely driven
more by its greater flexibility fitting real Euclidean-clustering
position jitter (it can represent straight motion as ω≈0) than by
genuine curvature detection — illustrated concretely by a representative
near-stationary track (id 652) whose mu_CV swings 0.14↔0.76 purely from
jitter, honestly labeled as such, not misrepresented as a real turning
transition.

**Residuals**: Linear KF median 0.432m, CTRV EKF 0.495m, IMM 0.437m (IMM
close to Linear KF's own lowest value, consistent with blending toward
whichever model fits best). **Runtime**: IMM costs ~4.1x Linear KF's
total latency (4.83ms vs 1.16ms mean) — expected, both models run every
frame.

**Classification**: **CASE A for the mechanism itself** (synthetic tests
and the real-data straight/turning split both show correct-direction,
smooth regime discrimination) **combined with T-7A.5's CASE B persisting
for this dataset** (turning excitation is real but modest; much of the
real-data model-probability variance reflects detector jitter, not
genuine curvature) — not CASE B (CV does not dominate) or CASE C (the
mechanism does discriminate) or CASE D (no numerical/state-mixing
problems found).

**Limitations**: single static scene with modest turning excitation; no
GT/MOTA/HOTA/IDF1/ID-switch or accuracy claim anywhere; IMM's own
`velocity_covariance`/BEV innovation covariance are documented
approximations (mu-weighted combination of each sub-model's own value,
not rigorously re-derived).

**Recommended next task**: the IMM mechanism is validated and ready to
serve as a baseline for a later KalmanNet comparison, but a real turning-
motion dataset (or a scene with genuine sustained curvature) would make
that comparison far more informative than this canonical recording,
whose own turning content remains modest (T-7A.5/T-7B both independently
confirm this). Consider either (a) a KalmanNet baseline comparison on
this same dataset with the same honest framing, or (b) sourcing/
replaying a MORAI scene with genuine sustained turning before further
motion-model comparisons.

`git status --short` at the end of this task shows only the same
pre-existing unrelated dirty files from every prior session, the T-7B
source changes (`ab3dmot_core.py`, `ab3dmot_config.py`,
`ab3dmot_tracker_node.py`, `ab3dmot_tracker.launch.py`, `CMakeLists.txt`,
`test_ab3dmot_imm.py` new), plus this `STATUS.md` update. Not committed/
pushed, per this task's instruction.

## T-7B result: **PASS**

---

## T-7A.5: Heading observability audit and fix (pre-IMM correction) — PASS, CASE B

Branch: `feat/ab3dmot-tracker`. Audited and corrected heading/yaw
observability handling before implementing IMM, per T-7A's discovered
negative-speed artifact (48.9% of EKF matches had `v < -0.5`). Full
detail: `~/heven_presentation_assets/heading_observability/README.md`
(`heading_source_audit.md`, figures/JSON/JSONL — outside repo, not
committed).

**Root cause, source-confirmed**: `adaptive_euclidean_cluster_node.cpp`
sets the identity quaternion (yaw=0.0) as a structurally-required
placeholder and explicitly marks
`orientation_availability = UNAVAILABLE` on the very next line — per the
message contract's own documentation ("orientation is empty... direction
unknown"). `ab3dmot_ros.py::detected_objects_to_detections()` never reads
this flag and extracts yaw from the placeholder unconditionally. Directly
measured: 100.00% of 4,582 canonical-recording objects have `yaw==0.0`
and `orientation_availability==UNAVAILABLE`.

**Fix**: new `AB3DMOTConfig.yaw_measurement_mode: "detector" (default,
unchanged) | "unobserved"`. In unobserved mode, both estimators perform a
genuine reduced-dimension measurement update (6-dim, yaw row dropped from
`H`/`R` — not a fake yaw with inflated R); EKF additionally gets
one-time, threshold-gated (0.5m displacement) motion-heading
initialization via `atan2(dy,dx)` + `disp/dt` (magnitude/direction
computed separately, so `v >= 0` holds by construction at init, no
post-hoc pi-rotation hack needed).

**Tests**: new `test_ab3dmot_heading.py` (19 tests). Full related suite
**176/176 pass** (157 pre-existing + 19 new); default `detector` mode
confirmed byte-identical to T-7A's own behavior.

**Measured result**: heading initialized for 91.4% of tracks (median 1
observation). Negative-speed artifact shrinks ~5x: fraction `v < -0.5`
55.6% (T-7A detector-yaw) → **10.9%** (T-7A.5 unobserved-yaw); fraction
`v < 0` 68.5% → **23.9%**. Prediction residuals stay close across all
three conditions (Linear KF 0.432m / EKF+detector 0.471m /
EKF+unobserved 0.495m median) — unobserved-yaw's small residual increase
is the honest cost of not fabricating heading from one observation.

**Turning-motion discovery**: a naive single-step classifier initially
found "turning" in the majority of candidate segments but was found,
after direct inspection, to be dominated by detector-clustering jitter,
not real curvature — explicitly discredited, not used. A stricter
classifier (1.0m displacement floor on both adjacent steps, 0.5 rad/s
threshold, requiring ≥2 consecutive same-sign steps) finds **52 verified,
sustained turning segments** (one, track 83, directly inspected and
confirmed physically plausible) against a much larger straight/near-
stationary population.

**Classification: CASE B** — motion heading is genuinely observable, but
the canonical recording's own motion content remains overwhelmingly
straight; CTRV's curved-motion advantage cannot be strongly evaluated on
this specific dataset. IMM (CV+CTRV) may proceed in T-7B, but should
state this limitation explicitly rather than claim a demonstrated
turning-motion advantage from this scene.

**Limitations**: single static scene; 0.5m/1.0m/0.5 rad/s thresholds are
documented, untuned baselines; `v>=0` guaranteed only at initialization,
not every subsequent update (accepted plain-EKF limitation); no GT/MOTA/
HOTA/IDF1 claim anywhere.

**Recommended next task**: **T-7B — IMM estimator** (CV Linear KF + CTRV
EKF, both now heading-observability-corrected), explicitly scoped to
state the CASE B limitation (this dataset's sparse turning content) when
interpreting IMM's model-switching behavior. Not implemented this
session.

`git status --short` at the end of this task shows only the same
pre-existing unrelated dirty files from every prior session, the T-7A.5
source changes (`ab3dmot_core.py`, `ab3dmot_config.py`,
`ab3dmot_tracker_node.py`, `ab3dmot_tracker.launch.py`, `CMakeLists.txt`,
`test_ab3dmot_heading.py` new), plus this `STATUS.md` update. Not
committed/pushed, per this task's instruction.

## T-7A.5 result: **PASS**

---

## T-7A: Linear KF vs EKF (State Estimation chapter begins) — PASS

Branch: `feat/ab3dmot-tracker`. Refactored the experimental AB3DMOT
tracker so the state estimator is pluggable (`Track` now delegates to a
common `predict`/`update`/`position`/`velocity`/`yaw`/`dimensions`/
`*_covariance`/`predicted_bev_*` interface), added an opt-in planar CTRV
EKF, and ran a controlled Linear-KF-vs-EKF comparison with association
**fixed to Euclidean + Hungarian for both runs** (Mahalanobis
deliberately excluded — it consumes estimator covariance directly, which
would confound the comparison). Full detail:
`~/heven_presentation_assets/state_estimation/README.md`
(`linear_kf_audit.md`, `ekf_design.md`, figures/JSON/JSONL — outside
repo, not committed).

**Interface**: `LinearKFEstimator` (default, byte-for-byte the pre-T-7A
KF logic moved out of `Track` unchanged) and `EKFEstimator` (new). New
`AB3DMOTConfig.state_estimator: "linear_kf"|"ekf"` (default unchanged:
`"linear_kf"`). `Track` owns zero estimator-specific math; lifecycle
(`hits`/`time_since_update`/`is_confirmed`/`is_dead`) is untouched and
has zero dependency on which estimator is used (confirmed by audit).

**EKF design**: state `[x,y,z,yaw,l,w,h,v,yaw_rate,vz]` (mirrors Linear
KF's own index layout so both share an identical linear `H`); planar CTRV
transition with closed-form (non-finite-difference) Jacobian, numerically
stable straight-line limit for `|yaw_rate| <= 1e-4`; `P0`/`Q` directly
adapted from Linear KF's own numeric factors, `R` byte-for-byte identical
to Linear KF's own `R` — a defensible, untuned baseline per this task's
explicit scope. Full equations/Jacobian in `ekf_design.md`.

**Tests**: new `test_ab3dmot_ekf.py` (25 tests: 4 Linear KF regression +
11 EKF physics + 3 measurement + 7 tracker-integration). Full related
suite **157/157 pass** (132 pre-existing + 25 new); default Linear KF
behavior confirmed byte-identical to pre-refactor.

**Canonical input**: reused T-4/T-5A/T-5B/T-6's exact persisted 400-frame
recording (400/400, 0 dup, 0 rollback, hash `63198cd0...4a0c6`, identical
to T-6's own recorded hash). Association fixed to
`euclidean_gate_m=3.0`+`hungarian` for both runs, verified via config.

**Key finding — a real detector-input limitation, not an estimator bug**:
the canonical scene's Euclidean-clustering detector outputs **yaw = 0.0
for every single detected object** (verified directly). This means (a)
**zero turning segments exist** in this data (stated explicitly, not
fabricated) — CTRV's theoretical curved-motion advantage is not tested
here; and (b) EKF's `vx=v*cos(yaw)`/`vy=v*sin(yaw)` parametrization gets
pinned near the x-axis, forcing real `-x`-direction motion to appear as
**negative scalar speed** (48.9% of EKF matches have `v < -0.5`) rather
than the independent `vy` Linear KF can express directly.

**Prediction residuals** (associated-measurement vs. prior prediction,
NOT ground-truth error): Linear KF median 0.432m (p99 2.76m) vs. EKF
median 0.471m (p99 2.79m) — very close, Linear KF slightly lower at every
percentile. Yaw residual: Linear KF exactly 0.0 always (yaw state never
perturbed, no yaw-rate coupling); EKF's p99 reaches 0.41 rad, traced to
the Jacobian's position/yaw_rate covariance coupling.

**Track-level**: unique tracks 725 (Linear KF) vs. 736 (EKF) — nearly
identical, small real difference from predicted-position feedback into
Euclidean matching (documented, expected per this task's own note).
Speed distributions similarly close (EKF's Cartesian-converted speed
median 0.94 vs Linear KF's 1.28 m/s).

**Runtime**: EKF's own `predict()` costs ~47% more than Linear KF's
(closed-form Jacobian + matrix products vs. a single filterpy linear
step), but both stay sub-millisecond; per-frame totals are statistically
indistinguishable at this track-count scale (~1.2ms mean either way).

**Limitations**: single static scene with degenerate (always-zero)
detector yaw — materially limits what this comparison can show about
CTRV's actual motion-model benefit; no turning-track figure exists
because none exists in the data; EKF's `velocity_covariance` is not
Cartesian-shaped (documented, deferred — relevant if EKF is ever paired
with Mahalanobis association in a future task); no GT/MOTA/HOTA/IDF1/
ID-switch claim anywhere.

**Recommended next task**: **T-7B — IMM estimator** (not implemented this
session, per explicit scope). Suggested design direction: a 2-model IMM
mixing Linear KF (CV) and the new CTRV EKF, with model-probability
switching informed by innovation likelihood — reuses both estimators
built in T-7A unchanged; note that T-7B should also flag the same
detector-yaw limitation found here, since IMM's CTRV branch will inherit
it identically.

`git status --short` at the end of this task shows only the same
pre-existing unrelated dirty files from every prior session, the T-7A
source changes (`ab3dmot_core.py`, `ab3dmot_config.py`,
`ab3dmot_tracker_node.py`, `ab3dmot_tracker.launch.py`, `CMakeLists.txt`,
`test_ab3dmot_ekf.py` new), plus this `STATUS.md` update. Not committed/
pushed, per this task's instruction.

## T-7A result: **PASS**

---

## T-6: FINAL association-metric comparison (GIoU / Euclidean / Mahalanobis Hybrid 10m) — PASS — Association chapter closed

Branch: `feat/ab3dmot-tracker`. Final, presentation-ready three-way
association-metric comparison, replacing T-4's original pure-Mahalanobis
comparison (superseded by T-5A/T-5B's finding of covariance-driven
long-distance over-association). Association-layer only; no new code —
reuses the exact T-4/T-5B metric/gate implementations. Full detail:
`~/heven_presentation_assets/association_final/README.md` (figures/JSON/
JSONL — outside repo, not committed).

**Configuration** (all three, matcher=hungarian, same canonical
400-frame input, same KF/`Q`/`R`/`P0`/lifecycle/`min_hits=1`/`max_age=2`):
A. GIoU (`giou_gate=0.0`); B. Euclidean (`euclidean_gate_m=3.0`);
C. Mahalanobis Hybrid (`mahalanobis_gate=11.62`,
`mahalanobis_max_distance_m=10.0` — T-5B's recommended baseline).

**Key result — the corrected baseline no longer produces the pathological
tail**: large-jump counts `>10m`/`>20m`: GIoU 0/0, Euclidean 0/0,
**Mahalanobis Hybrid 0/0** (T-4's pure-Mahalanobis had 201/95). Speed
p99/max: GIoU 11.3/22.4, Euclidean 18.0/27.2, **Mahalanobis Hybrid
48.0/90.0 m/s** (T-4's pure-Mahalanobis had 217.7/330.5 m/s) — far more
physically reasonable, though still heavier-tailed than the other two,
reflecting jumps in the now-bounded 3-10 m range.

**Track population**: unique tracks GIoU 2,739 / Euclidean 725 /
Mahalanobis Hybrid **505** (vs. T-4's pure-Mahalanobis 446 — the 10m cap's
known continuity cost, already quantified in T-5B); very-short fraction
86.97% / 42.21% / **25.54%**. Mahalanobis Hybrid retains the lowest churn
of the three, at a reduced (vs. pure Mahalanobis) but still real margin.

**Runtime** confirms T-3/T-4's ordering: construction mean GIoU 20.6 ms >>
Mahalanobis Hybrid 7.7 ms >> Euclidean 0.67 ms; Hungarian solver
negligible (<0.04 ms) for all three; the hybrid distance check itself
adds <1 ms.

**Assignment disagreement**: all three metric pairs disagree on the
majority of the 400 frames (88.75% / 90.75% / 87.5%), consistent with
T-4's original finding — metric choice remains a first-order factor in
this tracker's behavior.

**Final takeaway (no accuracy winner claimed, no GT)**: GIoU is
geometry-rich but computationally expensive and highest-churn in this
scene; Euclidean is cheap and interpretable but uncertainty-blind;
Mahalanobis Hybrid is uncertainty-aware while using an absolute distance
cap to prevent the previously-observed covariance-driven long-distance
tail, at moderate computational cost and the lowest churn of the three.

**Limitations**: single static MORAI scene; canonical-recording hash is
stable-under-current-procedure, not byte-identical to T-4's original
(unrecoverable) hash — same documented caveat as T-5B; no MOTA/HOTA/IDF1/
ID-switch claim anywhere.

**This closes the Association chapter** (T-1 through T-6): metric choice
(GIoU/Euclidean/Mahalanobis), assignment solver (greedy/Hungarian), and
Mahalanobis over-association mitigation (hybrid gate) have each been
implemented, tested, and measured on a shared canonical input, with a
corrected, presentation-ready final comparison now in place.

**Recommended next task**: State-estimator experiments — Linear KF → EKF
→ IMM comparison, building on the now-finalized association layer. Not
started this session, per explicit scope.

`git status --short` at the end of this task shows only the same
pre-existing unrelated dirty files from every prior session (no new
source changes — T-6 reused T-4/T-5B's existing implementation
unmodified), plus this `STATUS.md` update. Not committed/pushed, per
this task's instruction.

## T-6 result: **PASS**

---

## T-5B: Mahalanobis hybrid (chi-square + absolute distance) gate — PASS

Branch: `feat/ab3dmot-tracker`. Implemented and evaluated a
covariance-aware hybrid association gate for the experimental Mahalanobis
path, motivated by T-5A's CASE C finding. **Association-layer only**: KF,
`Q`/`R`/`P0` initialization, lifecycle, `min_hits`/`max_age`, and Hungarian
are all unchanged. Full detail:
`~/heven_presentation_assets/hybrid_gate/README.md` (figures/JSON/JSONL —
outside repo, not committed).

**Implementation**: new `AB3DMOTConfig.mahalanobis_max_distance_m` field
(default `0.0` = disabled, exactly reproduces pure T-4/T-5A Mahalanobis
behavior — verified by unit test). When `> 0`, `_associate()`'s
Mahalanobis branch requires a pair to satisfy **both**
`d_M^2 <= mahalanobis_gate` **and** `d_E (BEV Euclidean) <=
mahalanobis_max_distance_m` (reusing the existing
`_euclidean_bev_distance_matrix`) to be valid. The Hungarian cost matrix
is always raw `d_M^2`, never modified by the cap — the cap is a validity
gate only. New `mahalanobis_max_distance_m` launch arg/node parameter.

**Tests**: new `test_ab3dmot_hybrid_gate.py` (13 tests, including a
regression test reproducing the T-5A failure mode directly — inflate a
track's `P` by 1e6, confirm the cap rejects a 40 m match that pure
Mahalanobis alone would accept). Full related suite **120/120 pass** (107
pre-existing + 13 new).

**Cap sweep** (no cap / 3 m / 5 m / 10 m, same canonical 400-frame input,
`mahalanobis_gate=11.62` fixed, Hungarian fixed): candidate pairs removed
by the cap: 0 / 25,682 / 14,824 / 7,739. Large-jump counts (`>3/5/10/20m`):
no cap 554/345/201/95; 3m gate **0/0/0/0**; 5m gate 159/0/0/0; 10m gate
343/157/0/0 — **every cap fully eliminates jumps beyond its own
threshold**, as designed. The previously-diagnosed ~75 m event (T-5A's
track id 33/85 family, `hits=1`, `d_M^2=5.18` well inside the chi-square
gate) is reproduced identically here and rejected by all three caps.

**Continuity/churn tradeoff** (real, substantial): unique tracks no cap
446 → 3m 853 → 5m 629 → 10m 505; very-short fraction 0.90% → 45.72% →
35.93% → 25.54%. Suppressing the over-association tail costs continuity —
a tight (3m) cap is *more* restrictive than either GIoU or plain
Euclidean alone (both AND-conditions must pass), producing more churn
than T-4's plain Euclidean metric (725 unique / 42.2% very-short).
Speed p99/max drop from 217.7/330.5 m/s (no cap, physically implausible)
to 18.0-48.0 / 27.2-90.0 m/s across the three capped conditions.

**Runtime**: the hybrid distance check itself is cheap (<1 ms mean, p99
under 5 ms) and remains negligible relative to Mahalanobis metric
construction (~9-10 ms mean), consistent with T-3/T-4/T-5A's finding that
metric construction dominates association latency.

**Recommended baseline for subsequent experiments (not "optimal")**:
**10 m** — fully suppresses the `>10m`/`>20m` tail (T-5A's most
implausible evidence) while retaining continuity closest to the no-cap
baseline (505 vs. 446 unique tracks) of the three capped options; 3m/5m
remain reasonable alternatives if a future task prioritizes maximum
suppression over continuity.

**Limitations**: single static MORAI scene; cap sweep is a sensitivity
sweep, not a tuned search; cross-condition track-identity comparison
(`matches_lost_vs_no_cap`) is only reliable in the earliest frames before
independently-run conditions' populations diverge — the within-condition
`hits=1`-fraction analysis is the more robust evidence; no GT/MOTA/HOTA/
IDF1/ID-switch claim anywhere.

**Recommended next task**: either (a) proceed to a controlled comparison
using the recommended 10m hybrid-gate baseline as the new Mahalanobis
reference point (e.g. re-running T-4-style GIoU/Euclidean/Mahalanobis-
hybrid comparison), or (b) begin the deferred KF/EKF/IMM comparison now
that the association-layer over-association tail is characterized and
mitigated; not started this session, per explicit scope.

`git status --short` at the end of this task shows only the same
pre-existing unrelated dirty files from every prior session, the T-5B
source changes (`ab3dmot_config.py`, `ab3dmot_core.py`,
`ab3dmot_tracker_node.py`, `ab3dmot_tracker.launch.py`, `CMakeLists.txt`,
`test_ab3dmot_hybrid_gate.py` new), plus this `STATUS.md` update. Not
committed/pushed, per this task's instruction.

## T-5B result: **PASS**

---

## T-5A: Mahalanobis gating characterization (diagnostic only) — PASS, CASE C

Branch: `feat/ab3dmot-tracker`. Diagnostic-only task (no algorithm/gate/KF/
Q/R/lifecycle/Hungarian change) characterizing *why* T-4's Mahalanobis
association reduced track churn so dramatically. Ran the exact frozen T-4
config (`association_metric=mahalanobis`, `matcher=hungarian`,
`mahalanobis_gate=11.62`, same canonical 400-frame input, same KF/lifecycle)
through an additive diagnostic subclass (`DiagnosticTracker`, snapshots
per-track `P`/`S`/gate-radius/age/hits before `_associate()`, never
touches production code). Full detail:
`~/heven_presentation_assets/mahalanobis_analysis/README.md` (figures/CSV/
JSON — outside repo, not committed).

**Key finding**: the Mahalanobis gate is never a fixed-meter radius —
major-axis gate radius ranges from median 6.05 m to p99 58.85 m (max
118 m) across 5,177 track-frame samples, directly explaining T-4's
candidate-count inflation (median 19% of valid pairs/frame come from the
top-10%-highest-uncertainty tracks alone). Of 4,136 accepted matches,
554 (13.4%) produced a KF position jump >3 m. **Root cause of the most
extreme jumps, verified by direct inspection**: of the top-15 largest
jumps, **15/15** occur on tracks with `hits=1` (never yet corrected by a
real measurement since birth) — the reference AB3DMOT KF's own initial
velocity-uncertainty scaling (`P[7:,7:] *= 1000`) propagates into a
50-115 m position gate after just 1-2 predict-only frames, before any
real evidence has constrained the track. This exactly reproduces and
explains T-4's originally observed ~40 m/~57 m single-step event (track
id 85, hex UUID `...55`) — traced precisely, not assumed.

**Classification: CASE C (both effects materially present).** 87% of
accepted matches have `d_M^2 < 2` (tight, legitimate) and the tracks
providing T-4's continuity benefit are specifically the *low*-covariance
population (long-lived top-20: mean P-trace 13.2 vs. high-covariance
top-20's much larger values); but a real, precisely-localized
over-association mechanism exists among never-corrected new tracks,
driving nearly all of the most extreme jumps while remaining short-lived
and almost entirely disjoint from the long-lived population (1/180
overlap) — i.e. the churn-reduction benefit and the over-association risk
are largely two different track populations, not the same tracks
achieving continuity by jumping.

**Recommended next step**: a covariance-dependent + absolute-distance
hybrid gate (association-layer only, no KF/Q/R change) — targets the
localized never-corrected-track failure mode directly without penalizing
the 87% of matches that are already tight, and without the larger
blast radius of a KF-level covariance cap or Q/R recalibration. Not
implemented, per this task's explicit scope.

`git status --short` at the end of this task shows only the same
pre-existing unrelated dirty files from every prior session, plus this
`STATUS.md` update — no tracking-algorithm source file was touched.
Not committed/pushed, per this task's instruction.

## T-5A result: **PASS**

---

## T-4: Association-metric comparison (GIoU vs Euclidean vs Mahalanobis) for experimental AB3DMOT — PASS

Branch: `feat/ab3dmot-tracker`. Added two opt-in association-metric
alternatives to the experimental AB3DMOT tracker (`AB3DMOTConfig.association_metric`:
`"giou_3d"` default, plus new `"euclidean"` and `"mahalanobis"`) and ran a
controlled GIoU-vs-Euclidean-vs-Mahalanobis comparison with the assignment
solver held fixed at Hungarian (T-3) for all three runs. Full detail:
`~/heven_presentation_assets/association_metrics/` (`README.md`,
`association_metric_audit.md`, `association_metric_input_validation.md`,
figures/JSON/JSONL — outside repo, not committed).

**Implementation**: `SUPPORTED_METRICS` extended to
`("giou_3d", "euclidean", "mahalanobis")`; new `euclidean_gate_m` (default
3.0 m, documented neutral baseline) and `mahalanobis_gate` (default
11.62, reused directly from this repo's own Autoware audit in
`tracking_architecture.md`: "99.6% confidence, chi-square 2 DOF") config
fields. `Track` gained two read-only accessors
(`predicted_bev_position`, `predicted_bev_innovation_covariance` — the
latter reuses the reference submodule's own
`KF.compute_innovation_matrix()`, `S = H P H^T + R`, rather than
re-deriving it). New free functions `_euclidean_bev_distance_matrix`
(BEV `[x,y]` center distance) and `_mahalanobis_bev_distance_matrix`
(`d_M^2 = y^T S^-1 y` via `np.linalg.solve`, large-finite-sentinel
handling for singular/ill-conditioned `S` instead of `np.inf`, which
would make `linear_sum_assignment` infeasible). `_associate()` now
dispatches metric construction and gate direction per
`config.association_metric`; GIoU's own code path, gate semantics, and
sign convention are byte-for-byte unchanged from T-3. Both new metrics
use only the BEV `[x, y]` measurement subspace (not yaw/size/z),
matching each other and the Autoware Mahalanobis-gate precedent's own
DOF choice. New `association_metric`/`euclidean_gate_m`/`mahalanobis_gate`
launch args on `ab3dmot_tracker.launch.py`; node parameter wiring added
in `ab3dmot_tracker_node.py`.

**Tests**: new `test_ab3dmot_association_metrics.py` (20 tests: 6
Euclidean + 8 Mahalanobis + 5 cross-metric, covering the 19 required
cases plus one extra size/yaw-blindness check) + full related suite
**107/107 pass** (87 pre-existing T-1/T-3 + 20 new), default
GIoU+greedy behavior confirmed unaffected by the metric-dispatch
refactor. `colcon build` clean.

**Canonical input**: T-3's original canonical recording/hash could not be
reused — it lived only in that session's now-cleaned scratchpad and was
never persisted. A fresh, procedurally-identical capture was made this
session (same Ground-ON Euclidean pipeline, same MORAI static scene, same
replay parameters): 400/400 messages, 0 duplicate timestamps, 0 clock
rollbacks, mean 11.455 objects/frame (matches T-3's recorded value to
three decimal places). Content hash differs from T-3's recorded value
because T-3's own hashing script is equally unrecoverable — reported
honestly rather than claimed as an exact match; full explanation in
`association_metric_input_validation.md`. All three T-4 runs (A/B/C
below) share this one capture, satisfying the controlled-comparison
requirement regardless.

**Controlled runs**: three independent `AB3DMOTTracker` instances (GIoU,
Euclidean, Mahalanobis; matcher fixed to Hungarian), each driven by a
fresh `ros2 launch ab3dmot_tracker.launch.py` process, replaying the
identical canonical pickle byte-for-byte into
`/ad/perception/objects/detected` (bypassing ground segmentation/
Euclidean re-run). Publisher isolation (`ros2 topic info --verbose`,
exactly 1 publisher) verified before each run; 400/400 output messages
in all three runs.

**Match-level findings**: Mahalanobis's uncertainty-normalized gate
admits far more gate-valid candidate pairs per frame than GIoU (6.7x) or
Euclidean (2.2x). Assignment sets differ frequently between every metric
pair (88.75% GIoU-vs-Euclidean, 87% Euclidean-vs-Mahalanobis, 91%
GIoU-vs-Mahalanobis, out of 400 frames). Track populations across the
three independent runs first diverge at frame 26 (much earlier than
T-3's greedy-vs-Hungarian divergence at frame 121, since metric choice
affects which detections spawn new tracks far more directly than matcher
choice). A concrete representative case (frame 37) shows the same raw
9.38 m BEV gap rejected by Euclidean's fixed 3 m gate but accepted by
Mahalanobis (d_M^2 = 9.65 < 11.62 gate) because that track's predicted
covariance was large enough to make the gap statistically unsurprising.

**Track population** (400 frames each): unique tracks GIoU 2,739 /
Euclidean 725 / Mahalanobis 446; very-short-track fraction (<=2 obs)
86.97% / 42.21% / 0.90%; longest continuous track 400 / 400 / 130 obs.
Euclidean and especially Mahalanobis produce far fewer, much
longer-lived tracks than GIoU — a direct, expected consequence of GIoU
requiring actual box overlap (broken by any detector jitter) vs. the
other two metrics' looser, shape-blind tolerance. Not claimed as "more
accurate" (no GT) anywhere.

**Velocity/state**: KF-reported speed distributions differ substantially
(median 0.00 / 1.28 / 3.41 m/s; p99 11.27 / 18.05 / 217.66 m/s).
Mahalanobis's extreme tail was inspected directly (not just tabulated):
one track's published position jumps ~40 m in a single 0.15 s step,
architecturally explained by the same uncertainty-normalized-gate
mechanism above — reported as an observed, explained behavioral
difference, not resolved as correct or incorrect without identity GT.

**Runtime**: Euclidean's metric construction is ~20-25x cheaper than
GIoU's (a single 2D norm vs. full 3D polygon geometry); Mahalanobis sits
in between (a 2x2 linear solve per pair, cheaper than GIoU but pricier
than raw Euclidean). The Hungarian solver itself remains negligible
relative to metric construction for all three metrics, consistent with
T-3's own finding. Answers this task's own research question directly:
yes, both alternative metrics substantially reduce metric-construction
cost relative to 3D GIoU.

**Limitations**: single static MORAI scene; canonical-recording hash
mismatch vs. T-3 (explained above, doesn't affect T-4's own internal
validity); `euclidean_gate_m`/`mahalanobis_gate` are documented
neutral/reused baselines, not tuned against this session's own tracking
outcomes; match-level diagnostics ran the tracker core directly
(`lidar_link` frame) while track-level/velocity/runtime tables use the
live ROS-published (`odom` frame) recordings — not claimed numerically
identical to each other; GIoU's own construction/solver split reuses
T-3's previously-measured baseline rather than being independently
re-measured in this session's exact format; no MOTA/HOTA/IDF1/ID-switch
claim anywhere.

**Recommended next task**: identity-ground-truth-free ways to
characterize the Mahalanobis speed-tail/uncertainty-growth behavior
further (e.g. bounding `P` growth during long coasts), or begin the
`giou_gate`/`euclidean_gate_m`/`mahalanobis_gate` real-data calibration
flagged as needed since T-1's integration-decisions doc (not started
this session, per explicit scope).

`git status --short` at the end of this task shows only the same
pre-existing unrelated dirty files from every prior session, the source
changes listed above (`ab3dmot_core.py`, `ab3dmot_config.py`,
`ab3dmot_tracker_node.py`, `ab3dmot_tracker.launch.py`, `CMakeLists.txt`,
`test_ab3dmot_association_metrics.py` new), plus this `STATUS.md` update.
Not committed/pushed, per this task's instruction.

## T-4 result: **PASS**

---

## T-3: Greedy vs Hungarian assignment for experimental AB3DMOT — PASS

Branch: `feat/ab3dmot-tracker`. Added an opt-in Hungarian-assignment
option to the experimental AB3DMOT tracker (`ab3dmot_core.py`) and ran a
controlled Greedy-vs-Hungarian comparison, holding detector/ground
segmentation/GIoU/gate/KF/lifecycle/ROS contract fixed — the only
experimental variable was the assignment strategy. Full detail:
`~/heven_presentation_assets/association/` (`association_audit.md`,
`association_input_validation.md`, `association_difference_summary.md`,
`README.md`, figures/CSVs/JSONLs — outside repo, not committed).

**Implementation**: `AB3DMOTConfig.matcher` gained `"hungarian"` alongside
the unchanged default `"greedy"` (`SUPPORTED_MATCHERS = ("greedy",
"hungarian")`). New `_hungarian_matching()` (`scipy.optimize.linear_sum_assignment`,
already available via the same `~/py-ab3dmot-deps` staged environment used
for `filterpy`/`scipy` since T-2). `_associate()` now builds
`cost_matrix = -giou_matrix` once and dispatches on `config.matcher`; the
post-hoc `giou_gate` filter applies identically to both matchers' raw
output, so a below-gate pair is always rejected regardless of matcher —
Hungarian cannot force an invalid pair through. Everything else (GIoU
construction, predict/update, birth/death, output filtering) untouched.
Also added an opt-in, additive-only `collect_diagnostics` flag on
`AB3DMOTTracker` (never used by production/ROS code) for Phase 11's
match-level instrumentation. New `matcher` launch arg on
`ab3dmot_tracker.launch.py` (default `"greedy"`, `matcher:=hungarian`
usage), YAML comment updated (value unchanged).

**Tests**: new `test_ab3dmot_association.py` (24 tests: the 10 required
correctness cases including #10's classic globally-suboptimal-greedy
synthetic matrix, verified Hungarian achieves strictly higher total
affinity; plus config validation and no-duplicate-assignment checks) +
9-test `HungarianRegressionParityTest` class re-running the existing
greedy-baseline scenarios (stable ID, two-track separation, coast/
reappear, max-age deletion, nearest-track association, m/s velocity, real-
dt, yaw wraparound) with `matcher="hungarian"`, verifying identical
invariants. **Full related suite: 87/87 pass** (63 pre-existing + 24 new),
default-greedy behavior confirmed byte-identical to before (regression
suite unchanged). `colcon build` clean.

**Canonical input**: one pre-captured Ground ON Euclidean `DetectedObjects`
recording (400 frames, 0 duplicate timestamps, 0 clock rollbacks, raw
CDR-serialized bytes saved) replayed byte-for-byte into two fresh,
isolated AB3DMOT instances (Greedy, then Hungarian) — publisher isolation
(`ros2 topic info --verbose`, exactly 1 publisher) verified before/after
each; 400/400 output messages each run.

**Match-level findings**: assignments differ in 133/400 frames (33.25%);
track populations first diverge at frame 121 (before that, matrices are
byte-identical between runs). On the 256 frames still structurally
comparable (identical GIoU matrix), Hungarian's **raw** (pre-gate) total
affinity is >= greedy's on **all 256, zero exceptions** — the optimality
guarantee verified on real scene data, not just the unit test. Post-gate,
Hungarian was lower on 14/256 — a verified, explained gate-interaction
artifact (the gate strips a different specific low-affinity pair per
matcher's differing raw solution), not a guarantee violation.

**Track population**: Greedy 2,722 unique tracks / Hungarian 2,741
(+0.7%); mean tracks/frame 18.40 vs 18.46; lifetime distributions
essentially identical (very-short-track fraction, defined as <=2
observations: 87.47% vs 86.98%). Small, real differences — not claimed as
"more accurate" (no GT).

**Latency**: mean 25.11ms (Greedy) vs 25.36ms (Hungarian), materially
unaffected by matcher choice. Separated instrumentation: GIoU-matrix
construction (~19-20ms mean) dominates total association cost by ~2
orders of magnitude over either solver (Hungarian solver mean 0.03ms,
actually faster than greedy's 0.28ms at this matrix scale).

**Limitations**: single static MORAI scene; Phase 11 diagnostics ran the
tracker core directly (no ROS, `lidar_link` frame) for the relative
greedy-vs-Hungarian comparison, not numerically identical to the
ROS/`odom`-frame Phase 8 runtime-comparison messages; `giou_gate=0.0`/
`min_hits=1`/`max_age=2` untuned baseline, not re-examined; no RViz
live-screenshot evidence (same documented no-screenshot-capability
limitation as prior sessions); no MOTA/HOTA/IDF1/ID-switch claim anywhere.

**Recommended next task**: T-4 or equivalent — Euclidean vs Mahalanobis
metric comparison, or begin addressing the `giou_gate` calibration flagged
as needing real-data tuning since T-1's integration-decisions doc (not
started this session, per explicit scope).

`git status --short` at the end of this task shows only the same
pre-existing unrelated dirty files from every prior session, the source
changes listed above (`ab3dmot_core.py`, `ab3dmot_config.py`,
`ab3dmot_tracker.launch.py`, `ab3dmot.yaml`, `CMakeLists.txt`,
`test_ab3dmot_association.py` new), plus this `STATUS.md` update. Not
committed/pushed, per this task's instruction.

## T-3 result: **PASS**

---

## T-2B: Resolve apparent contradiction in T-2 (byte-identical detections vs 5-7% tracking difference) — CASE A, T-2 causal claim corrected

Branch: `feat/ab3dmot-tracker`. T-2 reported Ground OFF vs ON Euclidean
output as "byte-identical" (40-frame sample) yet Autoware/AB3DMOT tracking
population/churn differed ~5-7% (400-frame sample). This task resolved the
apparent contradiction before any association/Hungarian work starts. Full
detail: `~/heven_presentation_assets/ground_segmentation/`
(`t2b_detection_identity_check.md`, `t2b_tracking_repeatability.md`/`.csv`,
outside repo, not committed). No algorithm changed, no tracker tuned, no
detector parameter changed, no Hungarian work started.

### A. Was "byte-identical" literally true?

No, and it was never claimed to be at the raw-bytes level — but this
session verified this explicitly rather than assuming. A fresh recorder
(`record_full_detected.py`) captured the full CDR-serialized
`DetectedObjects` bytes (`rclpy.serialization.serialize_message`) plus
every decoded field the message actually has: `existence_probability`,
full `classification[]`, `pose.position`/`orientation`/`covariance`,
`has_position_covariance`, `orientation_availability`, `twist` (linear/
angular/covariance), `has_twist`/`has_twist_covariance`, `shape.type`,
`shape.dimensions`, `shape.footprint`. Confirmed directly:
**`DetectedObject` has no UUID/`object_id` field** (only `TrackedObject`
does) — the "UUID if present" check is N/A for detections.

### B. Detection stream hashes / identity result

Two fresh, isolated 40-frame Ground OFF / ON captures (same replay
parameters as T-2's original Phase 4). Per-message and whole-recording
SHA-256 content hash (canonical JSON of every field above, **excluding**
`header.stamp`) were **identical for all 40/40 messages**, whole-recording
hash `9d1bce62...0264a1c` both conditions — a **strictly stronger**
confirmation than T-2's original claim (T-2 only checked
`dimensions`/`position`/`class_name`/`score`; orientation, covariance,
twist, existence_probability, and footprint were never checked before and
are now also confirmed identical for this sample). Raw serialized bytes
differ only because of `header.stamp` (genuinely different real
replay-wall-clock times between the two runs, not a bug).

**However — and this is the key finding — re-examining the actual 400-frame
detection streams that fed the tracking comparison (not the 40-frame
sample) found they were NOT identical**: 94/400 frames (23.5%) have
different `object_count` between OFF and ON, starting at frame 73/400
(first 40 frames genuinely match, consistent with the sample above).
Ground ON produces fewer detections in 91/94 differing frames, mean signed
difference -0.515 objects/frame (~4.3% of the OFF mean of 11.97) —
directionally and roughly magnitude-consistent with the previously
reported ~5-7% tracking-population reduction. Direct inspection of 4
differing frames (73, 74, 118, 368) shows the specific missing objects
under Ground ON are consistently the lowest-z clusters in each frame
(e.g. frame 73: OFF has an extra cluster at z=-0.97 absent from ON) —
the expected ground-adjacent-cluster-removal mechanism, verified directly,
not just inferred.

### C. Run-to-run tracker variability (identical-input replay)

The Ground-OFF 40-message canonical recording's raw serialized bytes
(Section A/B capture) were replayed directly into `/ad/perception/objects/detected`
(bypassing ground segmentation and Euclidean entirely) 3 times each for
Autoware and AB3DMOT, with **both trackers fully killed and relaunched
fresh before every run** (genuine independent replicates) and publisher
isolation confirmed before each. Stamps were rebased by a constant offset
to land near current wall-clock (exact relative deltas preserved) after an
initial attempt with stale stamps triggered Autoware's `InputManager`
"Resetting the latest measurement time" warning and dropped outputs — a
replay-methodology artifact (Autoware's tracker runs an internal
`publish_rate: 10.0` timer correlated with real elapsed time, not a 1:1
per-message publish), not tracker non-determinism; fixed before the
reported runs. **Result: every population/churn metric (mean/median/max
tracks-per-frame, unique tracks, births, disappearances, lifetime
distribution, longest track) was byte-for-byte identical across all 3 runs
for both trackers** — 0% run-to-run variability. AB3DMOT latency showed
only normal small scheduling jitter (mean 7.00/7.29/6.51 ms across the 3
runs), two orders of magnitude too small to explain a population-level
difference.

### D. Rollback/reset differences

None in either the original 400-frame OFF/ON recordings (1 reset segment
each, re-confirmed this session) or the 3 canonical-replay runs (1 segment
each). Not a factor.

### E. TF/message-loss differences

Re-checked the original 400-frame OFF/ON tracker logs: zero
warnings/errors/TF-failures/duplicate-timestamp-skips/reconstruction
events in either condition's final (clean) run. Message accounting:
0 missing, 0 extra, 0 duplicate `detected`→`autoware`/`ab3dmot` stamp
matches in both conditions (400/400/400 each); replay pacing statistically
identical (mean inter-arrival 164.32ms both conditions). No TF or
message-loss difference between OFF and ON was found. (One recorder-side,
not tracker-side, message-loss artifact was hit and fixed *during this
session's own new captures* — a CPU-heavy per-callback recorder lagged the
publisher's limited history depth and silently dropped early messages;
fixed by deferring all decoding/hashing out of the subscription callback.
This is a T-2B tooling artifact, not evidence about the original T-2 runs,
which used a different, already-lightweight recorder.)

### F. Final classification: **CASE A**

Detector streams were **not** actually identical over the full 400-frame
window that produced the tracking-population comparison (they were
identical only for a 40-frame sub-sample). The field difference
(`object_count`, driven by low-z/ground-adjacent clusters) plausibly and
directionally explains the previously observed tracking difference. Combined
with C's proof of 0% tracker-side run-to-run variability under genuinely
identical input, there is no remaining candidate explanation other than
the detector-input difference itself.

### G. T-2 claims that remain valid

- Ground-segmentation node build/launch/runtime validation (Phase A/B of
  T-2's original report).
- Bounded point-cloud validation (ground fraction ~53%, 0 NaN/Inf, the
  voxel-downsampling explanation for `ground+nonground < cropped`).
- The 40-frame Euclidean-identity claim itself, **as scoped to those 40
  frames** — it was correct, just not generalizable to the full 400-frame
  run as originally implied.
- All raw direct measurement numbers in the original tables (tracks/frame,
  unique tracks, births/deaths, speed distributions, latency) — these are
  real recorded values and remain accurate as measurements.

### H. T-2 claims withdrawn or softened

- **Withdrawn**: any implication that Ground OFF vs ON Euclidean output
  was identical over the *400-frame* tracking-comparison window — it was
  not.
- **Softened**: the ARCHITECTURAL INTERPRETATION section's "genuine,
  measured effect... real, modest (~5-7%) reduction... a modest
  stabilizing effect" framing, which treated the tracking difference as an
  unexplained but presumed-causal side-effect of ground segmentation on
  the *trackers*. T-2B replaces this with a direct, verified explanation:
  ground segmentation changes Euclidean's own detection output later in
  the replay (via removing low-z clusters), and *that* detection-count
  difference is what the trackers faithfully reproduce — not an effect
  "on the trackers" independent of their input.

### I. Safe to start T-3 (Greedy vs Hungarian)?

**Yes.** The contradiction is resolved with a verified causal mechanism
(Case A), tracker-side determinism is proven under identical input (C),
and no tracker/association code was touched by this investigation. T-3
can proceed on the existing, unmodified association/tracker code with the
corrected understanding that any future Ground OFF/ON-style comparison
must verify full-window detector-input identity before attributing
downstream differences to a specific upstream stage.

### J. Best corrected presentation interpretation

Ground segmentation **does** change Euclidean's detection output under the
current z-crop configuration, but only for a subset of frames (those where
a cluster sits near the `min_z_m: -1.0` crop boundary) — not uniformly, and
not because Euclidean's crop makes ground segmentation redundant (the
original T-2 40-frame-derived conclusion). The downstream tracking
population difference is a faithful, expected consequence of that detection
difference, not an independent or surprising tracker-level effect. See
`~/heven_presentation_assets/ground_segmentation/README.md`'s "T-2B
CORRECTION" section for the full writeup.

`git status --short` at the end of this task shows only the same
pre-existing unrelated dirty files from every prior session, plus this
`STATUS.md` update. Not committed/pushed, per this task's instruction.

---

## T-2 / T-2A: Ground-segmentation restoration — PASS

Branch: `feat/ab3dmot-tracker`. Goal: measure Euclidean/tracking behavior
with the repository's existing ground-segmentation stage restored
(Ground OFF vs Ground ON), holding detector/tracker/association/KF/replay
data fixed. Full detail: `~/heven_presentation_assets/ground_segmentation/`
(`ground_path_audit.md`, `ground_segmentation_blocker.md`, outside repo,
not committed).

**Source audit (T-2, unchanged)**: no code/launch change is needed. Ground
ON = launch `ground_segmentation.launch.py backend:=ransac` (input already
matches the MORAI replay topic/format) + `euclidean_clustering.launch.py`
**without** the `finite_input_topic` override every prior experiment used
(its own default already is `/ad/perception/lidar/nonground`). Ground OFF
= the already-established bypass, unchanged. 110/110 existing tests pass
unmodified.

**Resumed and completed this session** after the 8 consolidated system
dependencies (`ros-humble-autoware-lanelet2-extension`,
`ros-humble-autoware-lanelet2-utils`, `ros-humble-autoware-point-types`,
`ros-humble-autoware-sensing-msgs`, `ros-humble-autoware-vehicle-msgs`,
`ros-humble-cv-bridge`, `ros-humble-point-cloud-msg-wrapper`,
`librange-v3-dev`) were installed with root between sessions. Full A-M
report below; full detail/figures/raw data in
`~/heven_presentation_assets/ground_segmentation/` (`README.md`,
`ground_segmentation_summary.md`, `ground_point_counts.csv`, 8 PNGs — all
outside the repo, not committed).

### A. Remaining local package builds

`managed_transform_buffer` (empty/untracked dir in-repo — cloned fresh
from `dependencies.repos`' pinned commit `c77fe4e6b...` to a scratch dir
outside the repo, not into the tracked tree), `autoware_pcl_extensions`,
`autoware_utils_diagnostics` (split package, needed transitively via the
monolithic `autoware_utils`'s own installed header but not declared
directly in `autoware_pointcloud_preprocessor`'s package.xml — same class
of gap as `managed_transform_buffer`), `autoware_pointcloud_preprocessor`,
`autoware_ground_segmentation` — **all built successfully**, no upstream
source patched. Two build-time issues found and fixed, both local
wiring/resource issues, not new missing dependencies: (1)
`autoware_utils_diagnostics`'/`diagnostic_updater`'s headers weren't on
the compiler's search path despite being built and on
`AMENT_PREFIX_PATH`/`CMAKE_PREFIX_PATH` — fixed via an explicit
`-DCMAKE_CXX_FLAGS="-I<installed include dir>"` addition (same
already-established technique as the earlier Boost header-staging
precedent — adding an include path for an already-built/installed
dependency, not patching source); (2) full `-j16` parallel compilation of
these heavy Autoware C++ template files OOM-killed `cc1plus`
(`fatal error: Killed signal terminated program cc1plus`, 15GB RAM
system) — fixed via `MAKEFLAGS=-j2`/`CMAKE_BUILD_PARALLEL_LEVEL=2`, not a
dependency issue.

### B. Ground node launch — actually run, not just built

`ros2 pkg executables autoware_ground_segmentation` confirmed
`ransac_ground_filter_node` (plus `ray_ground_filter_node`,
`scan_ground_filter_node`, unused). Launched standalone (no replay) via
`ground_segmentation.launch.py backend:=ransac`: process stayed alive,
correct remaps confirmed (`input:=/ad/perception/lidar/cropped`,
`output:=/ad/perception/lidar/nonground`,
`debug/ground/pointcloud:=/ad/perception/lidar/ground`), `ros2 node list`
showed `/ad_ground_segmentation`, `ros2 topic list` showed all 3 expected
topics. One pure build-artifact gap found and fixed along the way (not
new source work): `heven_ros_ws/install/ad_description`'s
`local_setup.bash` was a broken symlink into a missing `build/ad_description`
directory (same class of issue as an earlier-documented stale-build-tree
fix); rebuilt `ad_description` directly from
`~/projects/heven-ad-2026/ad_description` into the existing
`heven_ros_ws` overlay — no source changed.

### C. Bounded 40-frame point-cloud validation

Ground fraction mean 0.530 (median 0.530, min 0.511, max 0.541), 0
NaN/Inf across all 40 frames on `cropped`/`ground`/`nonground`, single
publisher confirmed. `ground+nonground` undershoots `cropped` by ~5,373
pts/frame on average — explained directly by the shipped config's own
comment (`ransac_ground_filter.yaml`: the debug `ground` topic is
voxel-downsampled at 0.10m, not full-resolution) — not a defect. Full
detail: `ground_segmentation_summary.md`.

### D. Ground OFF vs ON — Euclidean detection (40-frame sample)

Objects/frame **byte-identical** OFF vs ON (mean 2.88, stdev 1.10,
per-frame counts and per-object position/dims identical across two
genuinely separate, distinctly-timestamped replays). Explained
architecturally: Euclidean's own `adaptive_euclidean_cluster.yaml` crop
(`min_z_m: -1.0, max_z_m: 3.0`) already excludes ground-plane points for
this LiDAR mount height independent of upstream ground segmentation, for
this scene/sample. Latency: OFF mean 16.95ms vs ON mean 20.48ms (ON adds
a pipeline hop; Euclidean actually receives *fewer* raw points under ON).

### E/F. Ground OFF vs ON — Autoware and AB3DMOT tracking (400-frame replay)

Zero clock rollbacks either run (1 reset segment each); publisher
isolation (exactly 1/topic) held throughout both runs, verified before
and after. One real integration gap found and fixed this session:
`filterpy`/`scipy`/compatible `numpy` were missing from the system Python
AB3DMOT's node uses (not present in earlier T-1B/T-1C sessions — a fresh
environment gap), and the static `odom->base_link->lidar_link` TF (the
same T-1C test-mode precedent) needed republishing fresh — both fixed
without touching repo/system state: `filterpy`+`scipy`+`numpy<2` staged
via non-root `pip install --target ~/py-ab3dmot-deps` (prebuilt wheels,
matching system Python's 3.10/x86_64 ABI) and exposed via `PYTHONPATH` at
launch; TF republished via a scratch `StaticTransformBroadcaster` script.
One real process-isolation bug caught and fixed mid-session: a stale
orphaned `ad_finite_point_filter_node` from an earlier aborted attempt
survived several `pkill` passes (wrong process-name pattern) and caused a
genuine 2x duplicate-publish on `/ad/perception/lidar/nonground_finite`
(and therefore Euclidean/Autoware) in the first Ground-ON tracking
attempt — caught via a stamp-uniqueness check (not assumed clean), fixed
by killing the exact stray PID and re-verifying single-publisher state
before re-running; AB3DMOT's own duplicate-timestamp-rejection design
silently absorbed the duplicates without producing bad data, but Autoware
did not, so the affected recording was discarded and redone cleanly
(final data below is from the clean rerun, 400/400/400, 0 duplicate
stamps on any of the 3 topics in either condition).

| metric | Autoware OFF | Autoware ON | AB3DMOT OFF | AB3DMOT ON |
|---|---|---|---|---|
| tracks/frame mean/median/max | 9.90/9.0/27 | 9.43/8.0/24 | 19.36/20.0/50 | 18.40/18.0/40 |
| segment-safe unique tracks | 488 | 456 | 2,895 | 2,721 |
| births / disappearances (mid-run) | 488/470 | 456/438 | 2,892/2,875 | 2,718/2,701 |
| lifetime mean/median (obs) | 8.12/4.0 | 8.27/4.0 | 2.67/2.0 | 2.70/2.0 |
| longest continuous track (obs) | 399 | 399 | 400 | 400 |

Real, modest (~5-7%) reduction in track population/churn under Ground ON
for both trackers over the full 400-frame replay — a genuine measured
effect distinct from D's byte-identical 40-frame Euclidean sample (longer
window, not the same sample). "Track churn" terminology only; no
MOTA/HOTA/IDF1/IDSW, no ground truth.

### G/H. Track churn / speed distribution changes

Covered in E/F table above and: AB3DMOT speed (reset-safe, n=7,743 OFF /
7,359 ON): median 0.00, p90 1.02/1.00, p95 3.61/3.47, p99 10.24/10.30,
max 18.91/22.27 m/s; fraction >1/>5/>10 m/s effectively unchanged
(0.100/0.033/0.011 vs 0.100/0.032/0.011) — Ground ON does not
meaningfully change AB3DMOT's speed distribution, only its track
population. Not a physical-accuracy claim (no GT).

### I. Latency / runtime

Autoware: mean 1.79ms (OFF) / 1.72ms (ON), materially unaffected.
AB3DMOT: mean 27.30ms (OFF) / 24.85ms (ON), tracking its own lower mean
track count under ON. AB3DMOT latency-vs-track-count Pearson r = 0.945
(OFF) / 0.946 (ON) — strong, consistent with `O(tracks × detections)`
GIoU-matrix cost already documented in T-1C, descriptive only.

### J. Clock rollback / reset segments

None occurred in either 400-frame recording (1 segment each) — stated
plainly; composite `(segment, uuid)` identity machinery was implemented
and exercised (segment assignment via true file/arrival order, never via
sort) but had nothing to merge-guard against this session.

### K. Best presentation figures

All 8 requested figures generated under
`~/heven_presentation_assets/ground_segmentation/`:
`ground_fraction_over_time.png`, `ground_detection_objects_per_frame.png`,
`ground_tracks_per_frame.png`, `ground_unique_tracks.png`,
`ground_track_lifetime_distribution.png`, `ground_births_deaths.png`,
`ground_ab3dmot_speed_distribution.png`,
`ground_ab3dmot_latency_vs_track_count.png`. RViz live visual (pixel-level)
confirmation was **not** repeated this session (same no-screenshot-capability
limitation already documented in the RViz Tracking Comparison session
above) — topic-level verification was used instead, consistent with that
session's own established precedent; not fabricated as a "verified visually"
claim anywhere in this report.

### L. Limitations

Single static MORAI scene (already-documented, unrelated to this task);
Euclidean OFF/ON comparison (40 frames, ~6s) and tracking OFF/ON
comparison (400 frames, ~60s) are not the same sample size — stated
explicitly rather than implied equal; no GT/MOTA/HOTA/IDF1/IDSW claim
anywhere; `filterpy`/`scipy`/`numpy` staging (see E/F) and the
`ad_description` build-artifact fix (see B) are both environment-repair,
not repo changes.

### M. T-2 result: **PASS**

Full experiment completed end-to-end in one continuous pass per this
task's explicit instruction not to stop at another planning checkpoint
after build success. Full interpretation (DIRECT MEASUREMENTS vs
ARCHITECTURAL INTERPRETATION) in
`~/heven_presentation_assets/ground_segmentation/README.md`. Not
committed/pushed, per this task's instruction; `git status --short`
confirms only the same pre-existing unrelated dirty files from every
prior session in this branch (list below), plus this `STATUS.md` update.

---

**[historical, T-2A build-blocker detail below retained for reference]**

**T-2A build progress**: the system libraries identified as the original
blocker (`libopencv-dev`, `libpcl-dev`, `ros-humble-pcl-ros`,
`ros-humble-pcl-conversions`, `libboost1.74-dev`) are now installed
(confirmed via `dpkg -s`, someone ran the recommended command with root
between sessions). Rebuilt in `~/projects/autoware_tracker_ws` (same
workspace/convention as `autoware_multi_object_tracker`):
`autoware_vehicle_info_utils` ✅, `autoware_utils` (monolithic package,
`src/autoware_utils/autoware_utils` — the real dependency
`autoware_ground_segmentation`'s package.xml needs; the earlier attempt
had only the split `autoware_utils_*` packages) ✅ built this session.

**Sophus/Ceres resolved**: `libceres-dev` and `ros-humble-sophus` were
installed with root between sessions; verified directly this session
(`dpkg -s` both "install ok installed", `SophusConfig.cmake` found under
`/opt/ros/humble/share/sophus/cmake/`) — not assumed.

**CGAL resolved**: `libcgal-dev` installed with root between sessions;
verified directly (`dpkg -s` "install ok installed",
`CGALConfig.cmake` found under `/usr/lib/x86_64-linux-gnu/cmake/CGAL/`).

**Targeted dependency check (this session, replaces one-by-one
discovery)**: `rosdep check`/`install --simulate` could not run (rosdep
itself needs a one-time root-only `sudo rosdep init`, unavailable).
Substituted a manual equivalent: read every `<depend>` in
`autoware_pointcloud_preprocessor/package.xml` and
`autoware_ground_segmentation/package.xml`, found one further required
local sibling (`autoware_pcl_extensions`, source present, not yet built),
read its package.xml too, and checked `dpkg -s` on every resulting
package name in one pass. Result: **8 remaining packages missing**, all
prebuilt/apt-installable, none requiring a further source build:
`ros-humble-autoware-lanelet2-extension`,
`ros-humble-autoware-lanelet2-utils`, `ros-humble-autoware-point-types`,
`ros-humble-autoware-sensing-msgs`, `ros-humble-autoware-vehicle-msgs`,
`ros-humble-cv-bridge`, `ros-humble-point-cloud-msg-wrapper`,
`librange-v3-dev`. Full table (incl. what's already satisfied) in
`ground_segmentation_blocker.md`. No root access this session; per this
task's explicit instruction, did **not** install these by hand or attempt
another partial build now that they're known to be missing.

**Phases 5 onward (verify node launches) through the full Ground OFF/ON
experiment remain BLOCKED** — not fabricated.

**Next recommended task**:
```
sudo apt install -y ros-humble-autoware-lanelet2-extension \
  ros-humble-autoware-lanelet2-utils ros-humble-autoware-point-types \
  ros-humble-autoware-sensing-msgs ros-humble-autoware-vehicle-msgs \
  ros-humble-cv-bridge ros-humble-point-cloud-msg-wrapper librange-v3-dev
```
then re-run the exact `colcon build` command in
`ground_segmentation_blocker.md` for `autoware_pcl_extensions` (new,
local, add to `--base-paths`) + `autoware_pointcloud_preprocessor` +
`autoware_ground_segmentation` + `managed_transform_buffer`. This list was
derived from an exhaustive package.xml read, not incremental
trial-and-error, so no further new dependency is expected — but if one
does appear, verify it the same way before installing. Once
`autoware_ground_segmentation` builds, verify
`ros2 pkg executables autoware_ground_segmentation` shows
`ransac_ground_filter_node`, then resume T-2 at Phase 6 (bounded
point-cloud validation) of the original spec.

## Autoware vs AB3DMOT: initial runtime comparison

Branch: `feat/ab3dmot-tracker`. Small, controlled runtime comparison —
**not** a TrackEval integration, no new tracking algorithm, no tuning of
either tracker, no RViz changes. Full detail, exact commands, result
table, and limitations in
`docs/research/autoware_vs_ab3dmot_initial_comparison.md`.

Both trackers ran simultaneously off the exact same
`/ad/perception/objects/detected` stream (same Euclidean detector, same
`ad_publish_morai_frames` replay of `~/datasets/morai_heven`/`train`
`static_20260805_003151`, 400 frames / ~65 s, same T-1C static-TF setup),
each unmodified and untuned (Autoware `tracking.launch.py` +
`autoware.yaml`; AB3DMOT `ab3dmot_tracker.launch.py enabled:=true` +
`ab3dmot.yaml`, both unchanged from T-1C).

Collected via a scratch (uncommitted) `rclpy` recorder writing JSONL for
`/ad/perception/objects/detected`, `/ad/perception/objects/tracked`, and
`/experiment/tracked/ab3dmot`: message counts, output frequency, objects/
frame, unique track-ID counts, track-lifetime distributions, longest
tracks, births/deletions, duplicate-ID-per-frame checks, and NaN/Inf/
invalid-quaternion/invalid-dimension checks — all zero for both trackers
(no data-integrity defect in either). Autoware: 398 output msgs, mean 9.92
objs/frame, 474 unique track IDs, longest track 64.8 s (near full-run).
AB3DMOT: 400 output msgs, mean 19.28 objs/frame, 1,720 unique track IDs,
longest track 59.5 s.

Latency measured for **both** via the same external receive-to-receive
probe (single recorder process, unmodified nodes) — a fair, symmetric
method, so both are reported (unlike T-1C, which measured AB3DMOT only):
Autoware mean 1.81 ms / p95 3.44 ms; AB3DMOT mean 26.9 ms / p95 60.1 ms
(excluding one single-sample recorder-side scheduling artifact, stated
explicitly in the doc, not hidden). Explicitly **not** claimed as a pure
algorithm-cost comparison — Autoware's tracker is compiled C++, AB3DMOT's
is Python/rclpy; the probe measures each complete node's receive→publish
time.

Visual/qualitative cases (identified from the same recorded message data,
not a GUI screenshot — same no-screenshot-capability limitation already
documented in the RViz Tracking Comparison session) are recorded in the
doc: a long persistent track for each tracker, concrete birth/deletion
frame ranges, a multi-simultaneous-track frame for each (Autoware 30
objects at t=52.9 s; AB3DMOT 50 objects at t=56.9 s), and one visible
disagreement — Autoware ramps up slowly at replay start (0→1→2→3 objects
over the first ~24 s) while AB3DMOT does not (3→9→7→7 over the same
window), a direct, expected consequence of the two trackers' already-
decided `min_hits` config difference, not an unexplained bug. No accuracy,
GT-based, or ID-switch-rate claim is made for either tracker — explicitly
stated in the doc per this task's instruction.

Not committed/pushed per this task's instruction.

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
