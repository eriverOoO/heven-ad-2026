# CenterPoint offline integration (STEP 05-A)

This directory is an offline-only integration boundary between
`morai_heven` and a pinned external OpenPCDet checkout. It does not vendor or
patch OpenPCDet, train a model, calculate detection metrics, publish ROS
messages, or alter the production perception pipeline.

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

On a prepared NVIDIA host, add the pinned checkout and request the untrained
model smoke explicitly:

```bash
python3 tools/centerpoint_offline/verify_integration.py \
  --dataset ../../datasets/morai_heven_v1 \
  --openpcdet-root ../OpenPCDet \
  --attempt-model-smoke
```
