# MORAI 3D detection dataset exporter

This STEP 03 tool converts recorded HEVEN LiDAR and development-only MORAI
actor truth into finite XYZI point files and lidar-frame 3D box labels. It does
not train or integrate a detector.

Each exported box includes `num_lidar_points_inside_box`, computed against the
finite point cloud and exact oriented box with inclusive boundaries and no
margin. This statistic measures LiDAR support only; it is not an occlusion
label and is never used to drop or alter GT.

From the workspace root:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 src/heven_ad_2026/tools/morai_dataset_exporter/export_morai_dataset.py \
  --bag bags/static_20260805_003151 \
  --config src/heven_ad_2026/tools/morai_dataset_exporter/configs/static_20260805.yaml \
  --output datasets/morai_heven
```

Use `--overwrite` only to replace a directory previously created by this tool.
The exporter refuses to replace an unrecognized directory.

The generated dataset README and metadata are authoritative for timestamp,
transform, class, box, ROI, and split conventions. Dataset files are runtime
artifacts and must not be committed.

## Merge multiple bags

Pass every bag in the dataset build as a repeated `--bag` argument. The bag
directory basename is the scene ID, so basenames must be unique. Every scene
must be explicitly present under `scenes` in the same config with its scenario
evidence and a whole-scene `train`, `val`, or `test` assignment.

```bash
python3 src/heven_ad_2026/tools/morai_dataset_exporter/export_morai_dataset.py \
  --bag bags/static_20260805_003151 \
  --bag bags/driving_example_TIMESTAMP \
  --config src/heven_ad_2026/tools/morai_dataset_exporter/configs/DATASET_VERSION.yaml \
  --output datasets/DATASET_VERSION
```

All bags in one invocation share one `dataset.version`, topic/message contract,
class mapping, timestamp tolerances, transform chain, ROI, and box convention.
Duplicate scenes, missing scene assignments, missing scenarios, invalid splits,
different topic types, or different static sensor transforms are rejected. The
exporter deliberately rebuilds a complete output; it does not append into an
existing dataset.

The checked-in config preserves `static_20260805_003151` wholly in `train` and
keeps `val` and `test` empty. It does not invent frame-random splits.
