# CenterPoint offline integration (STEP 05-A)

This directory began as an offline-only integration boundary between
`morai_heven` and the pinned `references/openpcdet` Git submodule. It does not
copy, modify, or patch OpenPCDet, train a model, or alter the production
perception pipeline. It now also contains opt-in Autoware audit helpers; those
helpers build under the ignored `.autoware_runtime` worktree directory and
write inference artifacts outside Git.

The authoritative environment, contract, and commands are documented in
[`docs/perception/centerpoint_offline_environment.md`](../../docs/perception/centerpoint_offline_environment.md).

Run the CPU-only data-contract check from the repository root:

```bash
python3 tools/centerpoint_offline/verify_integration.py \
  --dataset ../../datasets/morai_heven \
  --batch-size 2
```

Unit tests do not require PyTorch or OpenPCDet:

```bash
python3 -m unittest discover \
  -s tools/centerpoint_offline -p 'test_*.py' -v
```

Prepare the isolated CenterPoint stage overlay and capture one canonical NPZ:

```bash
bash tools/centerpoint_offline/prepare_autoware_stage_overlay.sh
TENSORRT_ROOT=/home/didgang1203/opt/tensorrt/10.8.0.43-cuda11.8 \
  bash tools/centerpoint_offline/build_autoware_centerpoint_isolated.sh
TENSORRT_ROOT=/home/didgang1203/opt/tensorrt/10.8.0.43-cuda11.8 \
  bash tools/centerpoint_offline/run_autoware_stage_frame.sh \
    --npz /path/to/canonical_xyzirc.npz \
    --output-dir /external/output/directory \
    --stage-dump on
```

Stage dumping is default-off and exists only in the isolated source overlay.
It does not change the HEVEN production detector selection.

On a prepared NVIDIA host, initialize the pinned submodule and request the
untrained model smoke explicitly:

```bash
git submodule update --init --recursive references/openpcdet
python3 tools/centerpoint_offline/verify_integration.py \
  --dataset ../../datasets/morai_heven_v1 \
  --openpcdet-root references/openpcdet \
  --attempt-model-smoke
```
