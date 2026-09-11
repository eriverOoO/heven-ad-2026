# CenterPoint checkpoint distribution v1

Branch `feat/centerpoint-stability-audit-v1` **only**. This checkpoint,
its manifest, README, and download script are never added to `main` or
any other branch, per this project's explicit policy that CenterPoint
work stays isolated to its own experiment branch.

## What this is

`scripts/run_camera_lidar_tracking.sh` modes 8-10 (CenterPoint detector +
various association metrics) have always expected a checkpoint at
`models/experimental/centerpoint_t14_reproduction.pth` with a hardcoded
SHA-256, but no committed mechanism ever put that file there. This task
closes that gap for this branch, without touching `main`.

## Checkpoint provided

| file | category | size | SHA-256 |
|---|---|---|---|
| `centerpoint_t14_reproduction.pth` | project-own-trained (self-trained reproduction, not upstream-pretrained) | 93,474,618 B | `466c8181a377682e032bb32579c8ddb65807b5feebd8625a750b1d5538ddbc95` |

Full provenance (architecture, training dataset/split, epoch/seed,
evaluation result, known limitations) in
`models/experimental/manifest.yaml`.

**This is an in-house-trained checkpoint, never a redistributed
third-party artifact** -- no license question for redistributing it.
Confirmed distinct from the completely separate, official Autoware
Foundation `autoware_lidar_centerpoint` model (ONNX + TensorRT, used by
`tools/centerpoint_runtime/run_heven_centerpoint_candidate.sh`), which
remains gated behind `AD_AUTOWARE_MODEL_LICENSE_REVIEWED=1` and is
documented, never redistributed (see `manifest.yaml`'s
`not_distributed_here` section).

## Why a GitHub Release asset, not a git blob or Git LFS

- The repo's own `.gitattributes` already uses Git LFS for other large
  binaries, but `git-lfs` was found non-functional in the environment
  this task was done in (`git lfs version` fails) -- same finding as the
  KalmanNet checkpoint-distribution task on `main`.
- At 89 MB this checkpoint is far larger than the tiny (31-34 KB)
  KalmanNet checkpoints that were committed directly as plain git blobs
  on `main` -- a plain blob here would meaningfully bloat this branch's
  own history for every future clone/fetch.
- A GitHub Release asset needs no LFS, has no repo-history cost, and
  supports files up to 2 GB.

Release: `centerpoint-t14-reproduction-v1`
(https://github.com/eriverOoO/heven-ad-2026/releases/tag/centerpoint-t14-reproduction-v1),
targeted at this branch's own commit history, not `main`.

## Running it

```bash
git clone https://github.com/eriverOoO/heven-ad-2026.git
cd heven-ad-2026
git checkout feat/centerpoint-stability-audit-v1

./scripts/download_centerpoint_checkpoint.sh
sha256sum models/experimental/centerpoint_t14_reproduction.pth
# compare against models/experimental/manifest.yaml's sha256 field

./scripts/run_camera_lidar_tracking.sh --list-modes
./scripts/run_camera_lidar_tracking.sh --mode 8 --bag /absolute/path/to/bag
```

No checkpoint path needs to be edited in source anywhere --
`download_centerpoint_checkpoint.sh` places the file at exactly the path
`--centerpoint-checkpoint` / `$HEVEN_CENTERPOINT_CHECKPOINT` already
default to.

## Verification performed (this task)

- Downloaded the uploaded release asset back down and confirmed its
  SHA-256 is byte-for-byte identical to the source file
  (`466c8181...538ddbc95`) -- round-trip integrity confirmed, not
  assumed.
- Loaded the checkpoint with a real `torch.load()`: confirms the
  expected OpenPCDet-format top-level keys
  (`epoch`/`iteration`/`model_state`/`optimizer_state`/`scheduler_state`/`seed`/`dry_run_history`),
  `epoch=3` (matches `train_summary_epoch3.json` exactly), 279 tensors in
  `model_state`, 7,767,477 total parameters, **all finite** -- not
  corrupted.
- Attempted a full strict `state_dict` load into a constructed
  `CenterPoint` model via `pcdet.models.build_network` (the same
  approach the earlier "CenterPoint MORAI Training Prep v1" task used):
  this failed to *import* (not load) due to a pre-existing,
  checkpoint-independent environment issue -- `pcdet.models.__init__`
  transitively imports the Argo2 dataset module, which hits a
  `kornia`/`torch.jit.script` incompatibility in this venv
  (`~/venvs/heven-centerpoint`, torch `2.1.2+cu118`). This is an
  environment/dependency-version issue in the shared OpenPCDet
  checkout, unrelated to this checkpoint's own contents, and was not
  worked around (out of this task's scope to patch a third-party
  package's import chain). The direct `torch.load()` inspection above is
  the verification evidence this task relies on for "not corrupted,
  loadable in PyTorch."
- New `scripts/tests/test_download_centerpoint_checkpoint.py` (8 tests):
  download script is executable/portable (no hardcoded home path),
  default destination matches the runner's own default, `--help`/
  unknown-option handling, manifest SHA-256/size/download-URL fields,
  manifest documents the 100% train/eval overlap limitation, manifest
  never claims MORAI validation. All pass.
- `scripts/tests/test_camera_lidar_tracking_runner.py` (6, unmodified)
  and `tools/centerpoint_runtime/test_centerpoint_runtime_tools.py` (4,
  unmodified) both still pass -- no regression.

## No MORAI/generalization claim

This checkpoint's only recorded evaluation is on the exact 1,764 frames
it was trained on -- **100% train/eval overlap**, per the frozen "T-14:
End-to-End Detector x Tracker Evaluation with Independent MORAI GT"
finding in `docs/agent/STATUS.md`. Distributing the checkpoint changes
what is *runnable*, not what has been *validated*. This document does
not add, and explicitly disclaims:

- "CenterPoint is validated on MORAI" -- not supported.
- "CenterPoint is optimal for MORAI" -- not supported.
- Any conclusion contradicting the existing frozen T-14 finding or any
  other research doc in this repository.

## Limitations

- No new MORAI evaluation was run as part of this task (checkpoint
  distribution only).
- The full strict `state_dict`-load smoke test could not complete due to
  an unrelated OpenPCDet/kornia environment issue (see above); `torch.load()`
  inspection is the load-time evidence relied on instead.
- The Autoware `lidar_centerpoint` third-party model is documented, not
  redistributed, and this task does not change its existing license
  gate.

## Files

`models/experimental/{manifest.yaml,README.md}` (new, tracked;
checkpoint binary itself is a GitHub Release asset, never a git blob),
`scripts/download_centerpoint_checkpoint.sh` (new),
`scripts/tests/test_download_centerpoint_checkpoint.py` (new),
`.gitignore` (additive negation for exactly the two new tracked
`models/experimental/` files -- the `.pth` itself stays gitignored, since
it is never committed), this file. No CenterPoint detector, training,
evaluation, or runtime code changed. No `main`-branch file touched.
