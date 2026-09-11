# CenterPoint checkpoint (branch-local)

This directory holds the CenterPoint checkpoint for
`scripts/run_camera_lidar_tracking.sh` modes 8-10, on this branch
(`feat/centerpoint-stability-audit-v1`) only. See `manifest.yaml` for
full provenance (training dataset, split, epoch, seed, evaluation
result, SHA-256, known limitations, intended usage).

**The checkpoint binary itself is not committed here** -- it is a
GitHub Release asset (89 MB; git-lfs is non-functional in the
environment this was set up in, and a plain git blob at this size is
not a good fit for the repo). Download it with:

```bash
./scripts/download_centerpoint_checkpoint.sh
```

This places `centerpoint_t14_reproduction.pth` at exactly the path
`scripts/run_camera_lidar_tracking.sh` already expects by default
(`models/experimental/centerpoint_t14_reproduction.pth`), verifying its
SHA-256 against the value in `manifest.yaml`. No checkpoint path needs
to be edited in source anywhere.

## Verify manually

```bash
sha256sum models/experimental/centerpoint_t14_reproduction.pth
# compare against manifest.yaml's sha256 field:
# 466c8181a377682e032bb32579c8ddb65807b5feebd8625a750b1d5538ddbc95
```

## Not here: KalmanNet checkpoints

KalmanNet checkpoint distribution is handled entirely on `main`
(`models/experimental/{dense_kalmannet_v2.pt,
kalmannet_av2_natural_10k_generic_robust_seed{0,1,2}.pt}`), and is
deliberately not duplicated on this branch. If you need
`scripts/run_camera_lidar_tracking.sh` mode 7 (KalmanNet) from this
branch, copy those files from `main` yourself.

## Not here (and never redistributed): the Autoware official model

`tools/centerpoint_runtime/run_heven_centerpoint_candidate.sh` uses a
completely separate, official Autoware Foundation model
(`autoware_lidar_centerpoint` 0.51.0, ONNX + TensorRT engine files) --
see `manifest.yaml`'s `not_distributed_here` section for why that one is
documented, not redistributed.

## No MORAI performance claim

This checkpoint's only recorded evaluation is on the exact 1,764 frames
it was trained on (100% train/eval overlap -- see `manifest.yaml` and
the frozen "T-14" entry in `docs/agent/STATUS.md`). Downloading and
running it changes what is *runnable*, not what has been *validated*.
Do not treat any number produced by this checkpoint as a generalization
or competition-performance claim.
