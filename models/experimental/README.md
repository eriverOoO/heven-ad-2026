# Experimental model checkpoints

See `manifest.yaml` in this directory for full provenance (architecture,
training dataset, split hashes, epoch/seed, evaluation results, SHA-256,
known limitations) of every file here, and
`docs/perception/kalmannet_checkpoint_distribution_v1.md` for the full
clone-to-running-inference walkthrough.

Quick reference:

| file | purpose |
|---|---|
| `dense_kalmannet_v2.pt` | historical MORAI-domain checkpoint; the default `scripts/run_camera_lidar_tracking.sh` mode 7 already expected at this exact path/hash |
| `kalmannet_av2_natural_10k_generic_robust_seed1.pt` | **recommended**: AV2-pretrained family, repeatedly reaffirmed as the preferred AV2 candidate |
| `kalmannet_av2_natural_10k_generic_robust_seed{0,2}.pt` | alternate seeds of the same AV2-pretrained family |

**Not here:** `centerpoint_t14_reproduction.pth`. CenterPoint checkpoint
distribution is handled entirely on its own feature/experiment branch and
is deliberately never added to `main` -- see that branch's own docs. If
you need to run `scripts/run_camera_lidar_tracking.sh` modes 8-10
(CenterPoint) from `main`, copy that file into this directory yourself
(unchanged, pre-existing requirement).

## Verify before use

```bash
sha256sum models/experimental/*.pt
# compare against manifest.yaml's `sha256:` field for each file
```

## No MORAI performance claim

Populating this directory makes the checkpoints runnable end-to-end; it
does **not** change what this project has established about their
performance. In particular:

- The AV2-pretrained family's numbers above are AV2 held-out results,
  not MORAI results.
- The one real MORAI GT evaluation of the AV2-pretrained family is small
  (n=3 test sequences, one scene) -- see
  `docs/perception/morai_estimator_eval_v2.md`.
- `dense_kalmannet_v2.pt` ties, not beats, a classical Tuned Linear KF on
  its own (also small) MORAI TEST sample.

Neither "KalmanNet is validated on MORAI" nor "this checkpoint is optimal
for MORAI" is a supported claim.
