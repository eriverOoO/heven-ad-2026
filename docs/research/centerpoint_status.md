# CenterPoint implementation status (audited 2026-08-18, updated 2026-08-18)

> Update: item 4 below ("Euclidean vs CenterPoint... NOT DONE") is now
> implemented. See `docs/perception/centerpoint_vs_euclidean_comparison.md`
> for the new `ad_publish_morai_frames` / `ad_record_detected_objects` tools,
> exact commands, and observed results. Everything else in this file is
> unchanged from the original audit and still accurate.
>
> Second update (same day): a bounded non-smoke (3-epoch, full existing
> train split) checkpoint was trained, closing CP-1 criteria #1 and #5 —
> see "## CP-1 Milestone" below, now **CP-1: PASSED**.

Source: `docs/perception/centerpoint_offline_environment.md`,
`docs/perception/centerpoint_ros_interface.md`, and the files listed under
each component below. No other parts of the repository were rescanned.

**The existing dry-run checkpoint is not a production-quality detector.**
Every "DONE" below means the plumbing/interface is verified, not that
CenterPoint currently produces usable detections.

## Component status

### MORAI dataset/export — DONE (adapter), PARTIAL (data)
`tools/centerpoint_offline/morai_dataset.py` (`MoraiHevenDatasetCore`,
`collate_openpcdet_contract`, `make_openpcdet_dataset`). Reads STEP03/
`morai_heven_v1` exports, validates points/boxes/classes/frame, proven
zero-delta coordinate identity with the OpenPCDet unified box convention.
Blocker: dataset is effectively **one repeated static scene** (train split
1,764 samples of the same scene; `val`/`test` splits are intentionally
empty). Not diverse driving data.

### OpenPCDet adapter — DONE
`tools/centerpoint_offline/openpcdet_runtime.py` imports only
`DatasetTemplate` and `CenterPoint` from a pinned external checkout
(commit `233f849829b6ac19afb8af8837a0246890908755`, lock file
`upstream.lock.yaml`), avoiding eager `pcdet.datasets` imports. No upstream
files patched/vendored, matches AGENTS.md submodule/adapter rule in spirit
(external repo referenced by path + lock, not yet a Git submodule under
`references/`).

### Training — PARTIAL
`tools/centerpoint_offline/train_morai_centerpoint.py`. Verified on RTX 4060
(2026-08-16/17): real CUDA forward/backward/optimizer steps, checkpoint
save/resume round-trip proven (state, optimizer, scheduler, iteration count
all restored correctly). Capped at 2–4 iterations only ("dry-run"), no
epoch-scale run, no evaluation metric implemented (`evaluation()` raises
`NotImplementedError` by design). Losses observed (57.6, 104.7) are execution
evidence only, not convergence evidence.

### Checkpoint/resume — DONE
Atomic checkpoint save (`save_checkpoint_atomic`) and resume
(`restore_checkpoint`) verified end-to-end including optimizer/scheduler
state restoration and a real post-resume optimizer step.

### Offline inference — DONE (interface only)
`tools/centerpoint_offline/infer_morai_centerpoint.py`. Strict
`state_dict` load, eval-mode CUDA inference, JSONL output via
`prediction_bridge.py` (`heven.offline_detection.v1`). Verified 10-frame run
on RTX 4060: 490 predictions (49/frame — untrained diagnostic boxes, not
real detections), latency mean 127 ms / p50 18 ms / p95 616 ms (first-call
CUDA warm-up included, not a clean benchmark).

### ROS2 CenterPoint node — DONE (interface), NOT DONE (real model quality)
`ad_lidar_perception/centerpoint_detector_node.py` +
`centerpoint_ros.py`. Optional, disabled by default
(`detector_backend=euclidean`, `enabled=false`). Loads the pinned
OpenPCDet CenterPoint + strict checkpoint, logs
preprocessing/model_forward/postprocessing/total latency per callback.
Verified on WSL2 + RTX 4060 (2026-08-17): mock mode publishes empty
stamp-preserving output; real mode loaded the dry-run checkpoint and
produced 49 diagnostic boxes/frame; post-warm-up latency ~53 ms total,
~20 ms model forward.

### DetectedObjects conversion — DONE
`centerpoint_ros.py::detections_to_message`. Box→`DetectedObject` mapping
verified: geometric center, yaw→quaternion, `vehicle→CAR`,
`pedestrian→PEDESTRIAN`, `obstacle→UNKNOWN` (matches existing Euclidean
convention), no invented velocity (`has_twist=false`).

### RViz — PARTIAL
Generic visualizer `ad_viz/src/perception/perception_visualizer_node.cpp`
subscribes to `/ad/perception/objects/detected` and
`/ad/perception/objects/tracked` (backend-agnostic — no CenterPoint-specific
code needed) and republishes `MarkerArray` on
`/ad/visualization/detected_objects` / `.../tracked_objects`;
`heven_perception.rviz` already displays those topics. Topic-level wiring
and message counts were verified (8/8 frames), but no one has visually
confirmed CenterPoint boxes rendering in RViz — and with the dry-run
checkpoint the boxes would be meaningless diagnostics, not real detections.

### CenterPoint → Autoware tracker connection — DONE (wiring), PARTIAL (validated behavior)
Topic contract matches exactly: node publishes
`/ad/perception/objects/detected` (10-deep, matches Euclidean's contract);
`tracking.launch.py` wires `autoware_multi_object_tracker` to that same
topic as `input/detection01/objects`/channel `lidar_clustering` regardless
of which detector backend is active. `lidar_perception.launch.py` includes
tracking whenever `selection.tracker.backend == autoware`, independent of
`detector_backend`. Verified integrated smoke (2026-08-17): tracker
delivered 8/8 `odom`-frame `TrackedObjects`, prediction adapter correctly
rejected stale input past its 0.5 s freshness bound. A ~10 s WSL clock jump
caused one TF extrapolation warning during that run — a host clock
stability issue to fix before deterministic bag replay, not a
detector/tracker defect.

## Minimum work before Friday

1. **Meaningful trained checkpoint — likely NOT achievable with current data.**
   The only available data is one repeated static scene; no `val`/`test`
   scenes exist. Per `docs/perception/centerpoint_offline_environment.md`
   preconditions, real training needs additional diverse bags/scenes first.
   Minimum action: obtain and export at least a handful of *different*
   driving scenes (not just more frames of the same scene) with `val` slice
   populated. If no new scenes are available by Friday, do not claim a
   trained checkpoint — keep the dry-run checkpoint labeled as
   interface-validation-only, per AGENTS.md ("execution success is not
   performance validation").
2. **ROS2 inference — already achievable now.** The node, mock mode, and
   real-checkpoint mode are already verified; this needs no new work beyond
   re-running the existing launch commands documented in
   `centerpoint_ros_interface.md`.
3. **Useful CenterPoint boxes in RViz — blocked by (1).** The visualization
   path itself needs no new code (generic marker converter already handles
   both backends). "Useful" boxes require a checkpoint trained past the
   dry-run stage; until then, running RViz will only show diagnostic noise,
   which should be labeled as such if demoed.
4. **Qualitative Euclidean vs CenterPoint comparison — no code exists yet.**
   Nothing runs both detectors side-by-side or overlays their outputs. This
   is genuinely NOT DONE and is the smallest net-new piece: run
   `lidar_perception.launch.py` once per backend against the same bag/scene
   and compare RViz recordings or `infer_morai_centerpoint.py` JSONL output
   qualitatively (box counts/positions), without inventing a quantitative
   metric (STEP 02's metric is not adapted for this yet).

Given the one-scene dataset constraint, the realistic Friday deliverable is
(2) + (3, diagnostic-only) + (4) as a qualitative side-by-side, explicitly
labeled as interface/diagnostic evidence — not (1) as a real trained model,
unless new diverse scene data arrives first.

## Exact data requirements for a meaningful trained checkpoint

Not achievable with the current export at
`~/datasets/morai_heven` (verified 2026-08-18: `splits/train.txt` has 1,764
lines, all from the single scene id prefix `static_20260805_003151_*`;
`splits/val.txt` and `splits/test.txt` are both 0 lines). Additional data
needed before real training can be attempted:

1. **New, distinct MORAI scenes** — different locations and/or object
   configurations, not more frames of the same static capture. A single
   scene repeated at ~30 ms cadence gives temporal redundancy, not the
   spatial/appearance diversity a detector needs to generalize.
2. **A populated `val` split** from scenes disjoint from `train`, sized
   enough to detect overfitting during training (a few hundred frames
   minimum, across more than one scene).
3. **A populated `test` split**, disjoint from both `train` and `val`, held
   out until a final evaluation — not touched during model/hyperparameter
   selection.
4. **A defined MORAI evaluation metric** (`evaluation()` in
   `tools/centerpoint_offline/morai_dataset.py` currently raises
   `NotImplementedError` by design) — needed before any precision/recall/mAP
   number can be computed at all, even once (1)-(3) exist.

Until all four exist, this repository will keep reporting training runs as
execution/pipeline evidence only (losses, VRAM, latency, checkpoint
round-trip), per AGENTS.md's "execution success is not performance
validation" — not as accuracy or generalization evidence, and not faked by
running more iterations against the same single scene.

## CP-1 Milestone

Verified 2026-08-18 against the evidence already produced in this session
(train/infer runs, live ROS2 round trip, comparison tool) plus the prior
audit above. No new training, tuning, or code was added for this check.

| # | Criterion | Result | Evidence |
|---|---|---|---|
| 1 | Non-smoke checkpoint exists, provenance documented | **FAIL** | Every checkpoint produced so far (this session's 2- and 5-iteration runs; the prior session's 2–4-iteration runs) is explicitly capped and self-labeled `"purpose": "training_pipeline_dry_run_only"` in its own summary JSON. None is a non-smoke (e.g. full/multi-epoch) run. What exists *is* well-provenanced (`exact_command`, `dataset_version`, seed, epoch/iteration, losses, checkpoint path all recorded in `train_summary*.json`) — the gap is that only smoke-tier runs exist, not that provenance practice is missing. This is achievable now on the existing single scene (no new data needed) — see next task. |
| 2 | Offline inference works | **PASS** | `infer_morai_centerpoint.py` ran successfully this session and in the prior audit: strict `state_dict` load, eval-mode CUDA forward, JSONL output written, latency recorded. Mechanism-level pass, independent of checkpoint quality. |
| 3 | ROS2 inference works with that checkpoint | **PASS** | `ad_centerpoint_detector` loaded both this session's smoke checkpoints via strict load and ran real CUDA forward passes end-to-end (confirmed via `centerpoint_latency_ms` node logs and the new recorder tool receiving `DetectedObjects`). No non-smoke checkpoint exists to test additionally (see #1), but the strict-load/inference code path is checkpoint-content-agnostic — same `state_dict` format either way. |
| 4 | DetectedObjects output is valid | **PASS** | `detections_to_message` unit-tested (class mapping, quaternion, no invented velocity/twist) and live-verified this session: real messages published with preserved header/frame, finite boxes, positive dimensions, valid classification, `orientation_availability=AVAILABLE`, `has_twist=false`. |
| 5 | CenterPoint 3D boxes visible and geometrically plausible in RViz | **FAIL** | *Visible*: yes — this session's 2-iteration-checkpoint run produced a live 102-marker `MarkerArray` on `/ad/visualization/detected_objects`, the exact topic `heven_perception.rviz` displays (verified by direct topic echo, not a GUI screenshot). *Plausible*: no — re-inspecting that run's 2,500 offline-inference detections (`predictions_diag2.jsonl`) shows every one is class `obstacle` (never vehicle/pedestrian), dimensions clustered at a near-constant ~0.94–0.98 m × 0.96–1.0 m regardless of position, and scores confined to a 0.1006–0.1010 band (stdev 0.00012) — a dense, near-uniform noise floor just above the 0.1 threshold, not per-object learned detections. This is the same root cause as #1: no non-smoke checkpoint has ever been trained. |
| 6 | Euclidean and CenterPoint inspectable on the same MORAI run | **PASS** | `ad_publish_morai_frames` / `ad_record_detected_objects` (new this session) verified live: Euclidean produced 3 objects/frame at 16.7 ms mean latency, CenterPoint produced its checkpoint-dependent output, both against identical replayed frames. Documented with exact commands and the ground-segmentation-bypass caveat in `docs/perception/centerpoint_vs_euclidean_comparison.md`. |
| 7 | Inference latency recorded | **PASS** | Recorded at three independent layers: offline batch latency percentiles (`infer_morai_centerpoint.py`), the ROS2 node's own per-callback breakdown (preprocessing/model_forward/postprocessing/total), and the new recorder's publish→receipt latency. |
| 8 | Dataset/performance limitations documented | **PASS** | This file's "Exact data requirements for a meaningful trained checkpoint" section, plus `docs/perception/centerpoint_offline_environment.md` and `docs/perception/centerpoint_vs_euclidean_comparison.md`, all explicitly state the single-scene/empty-val-test limitation and forbid accuracy/mAP claims until it's resolved. |

**CP-1: NOT PASSED.** Two criteria fail (#1 and #5), both from a single root
cause — no training run past the initial capped smoke test has ever been
performed — and neither is blocked by the dataset-diversity problem
described elsewhere in this file: a longer bounded (still non-smoke, still
single-RTX-4060, still not a sweep) run on the *existing* 1,764-sample
train split is enough to re-test both criteria. ROS2 inference, the
DetectedObjects contract, the RViz marker path, the Euclidean/CenterPoint
comparison tooling, and latency recording (#2, #3, #4, #6, #7) all pass —
**ROS2/RViz/inference are not broken.**

Separately, and per this check's instructions: proper mAP/generalization
performance validation (criterion #8's subject) remains blocked purely by
dataset diversity (one scene, empty `val`/`test`) as already documented
above, independent of the #1/#5 gap. That blocker does not, by itself, fail
CP-1 — but CP-1 is not being marked passed here because #1/#5 are a
separate, non-diversity-blocked gap.

### CP-1 re-check 2026-08-18: criteria #1 and #5 (non-smoke checkpoint)

Trained one bounded, non-smoke checkpoint on the existing
`tools/centerpoint_offline/train_morai_centerpoint.py` pipeline, unchanged
architecture/config, minimum safe batch size (1), no hyperparameter search:
3 full epochs over the entire existing 1,764-sample `train` split (5,292
iterations total), following the procedure "one epoch, continue only if
stable and clearly learning, cap at 3 epochs." Epoch 1 alone showed a clear
downward loss trend (first-50-iteration mean 17.7 → last-50-iteration mean
6.7, per-quartile means 9.4/8.0/7.9/6.7, no NaN/Inf) justifying the
continuation to 3 epochs.

**GPU/VRAM checked first** (`nvidia-smi`): RTX 4060, 8,188 MiB total, ~6.8
GB free at start.

| item | value |
|---|---|
| dataset size | 1,764 samples (`train` split, single scene — unchanged) |
| batch size | 1 (minimum safe, per prior verified runs) |
| epoch/iteration count | 3 epochs, 5,292 iterations total (1,764/epoch) |
| loss summary | initial 57.57 → final 3.08; min 1.21, max 115.13 (early spikes); no NaN/Inf detected |
| peak VRAM | 373.9 MiB (epochs 2–3; 343.9 MiB for epoch 1 alone) |
| training time | 116.8 s (epoch 1) + 412.7 s (epochs 2–3, resumed) ≈ 529.5 s total |
| checkpoint path | `<scratch>/nonsmoke/checkpoints/checkpoint_latest.pth` (epoch 3 / iteration 5,292) — ephemeral, not committed, per AGENTS.md; fully reproducible from the exact command recorded in `train_summary_epoch3.json` (dataset path, seed 2026, epochs, batch size, resume chain) |

**#1 Non-smoke checkpoint exists, provenance documented → PASS.** This is a
complete-dataset, multi-epoch run (not an artificially-capped 2–5-iteration
smoke test), with checkpoint save/resume verified across the run
(`checkpoint_resume_success`, `optimizer_state_restored`,
`scheduler_state_restored`, `post_resume_optimizer_step_success` all
`true` for the epoch-3 summary) and full provenance recorded in
`train_summary_epoch1.json` / `train_summary_epoch3.json` (exact command,
dataset version, seed, losses, VRAM, checkpoint paths). (The training
script's own JSON still carries the fixed label
`"purpose": "training_pipeline_dry_run_only"` — that string is unconditional
code metadata, not a semantic smoke/non-smoke marker; the actual
distinguishing facts, epoch/iteration count and full-split coverage, are
what changed.)

**Offline inference on the new checkpoint** (20 frames,
`infer_morai_centerpoint.py`, unchanged script/config):

| stat | smoke checkpoint (prior) | non-smoke checkpoint (this run) |
|---|---|---|
| predictions/frame | 0 (5-iter) or ~50, capped (2-iter) | 14.75 |
| class distribution | 100% `obstacle` | 100% `vehicle` |
| score range | 0.1006–0.1010 (stdev 0.00012) — noise floor | 0.100–0.329 (stdev 0.064) — real spread |
| box dimensions (L×W×H) | ~0.95×0.97×~1 m, near-constant | mean 4.43×2.03×1.81 m, varying |
| inference latency | mean 61.7 ms / p95 263.6 ms (5-iter) | mean 43.5 ms / p50 20.0 ms / p95 50.9 ms |

Cross-checked the new checkpoint's predicted box dimensions against real GT
box dimensions in this dataset (sampled 200 label files: 320 `vehicle` / 8
`pedestrian` / 8 `obstacle` GT boxes; GT vehicle mean L=5.12 W=2.16 H=1.70
m). The predicted mean (L=4.43 W=2.03 H=1.81 m) is within ~15% of the real
GT vehicle mean on all three axes, and the single predicted class
(`vehicle`) matches the scene's actually-dominant GT class (320/336 sampled
GT boxes). Predicted positions (x −3.5..59.4 m, y −5.6..20.4 m) stay inside
the configured detection ROI.

**ROS2 + RViz re-verified live with this checkpoint** (not just offline):
`ad_centerpoint_detector` loaded it via strict `state_dict` load and ran 5
real CUDA forward passes (`model_forward` 35–122 ms post-warm-up,
`gpu_peak_mib` 271.9 during inference); the recorder captured 16–17
objects/frame, 100% `vehicle`, scores 0.101–0.273, dimensions matching the
offline run (L 4.37 W 2.02 H 1.79 m mean); the reused
`ad_viz perception_visualizer_node` republished 34 non-clear-action markers
(≈17 boxes × 2 markers/box) on `/ad/visualization/detected_objects`, the
exact topic `heven_perception.rviz` displays.

**#5 CenterPoint 3D boxes visible and geometrically plausible in RViz →
PASS.** Judged only for implementation-milestone plausibility, per this
check's instructions (not accuracy/generalization): boxes are visible
(live marker output, as above) and now geometrically plausible — a single,
scene-appropriate class, real-vehicle-scale dimensions matching this
scene's own GT within ~15%, positions inside the valid ROI, and genuine
score diversity, replacing the smoke checkpoint's uniform near-threshold
noise grid.

**Caveats (do not over-read this result):** This checkpoint was trained and
evaluated on the *same* single repeated scene (no `val`/`test` exists to
separate them), so plausible-looking boxes are expected even from
memorization of that one scene, not evidence of generalization. **No
accuracy, recall, precision, mAP, or superiority claim over Euclidean is
made here or anywhere in this file.** The dataset-diversity blocker for
real performance evaluation (see "Exact data requirements..." above) is
completely unchanged by this run.

## CP-1: PASSED

All 8 criteria now pass: #2, #3, #4, #6, #7, #8 (unchanged, see table
above) plus #1 and #5 (re-scored above). ROS2, RViz, offline inference, and
the Euclidean/CenterPoint comparison tooling all work, and CenterPoint now
produces a non-smoke checkpoint whose boxes are geometrically plausible
enough to serve as an implementation milestone. This is **not** an accuracy,
generalization, or Euclidean-superiority claim — proper performance
evaluation stays blocked by dataset diversity (single scene, empty
`val`/`test`) and remains tracked separately in "Exact data requirements
for a meaningful trained checkpoint" above.
