# MORAI 3D detection dataset expansion preparation

## Scope

This change prepares one versioned dataset build from multiple MORAI bags. It
does not train CenterPoint, create random frame splits, modify existing labels,
or change the production perception pipeline.

The current contract is `morai_heven_v1`. All bags in one build share its topic
and message types, timestamp tolerances, transform chain, class mapping, ROI,
box convention, visibility definition, and scene-level split policy.

## Required new bags and scenarios

Only `static_20260805_003151` / `checkpoint14.json` currently exists. It remains
entirely in `train`; `val` and `test` remain empty. The following are capture
requirements, not fabricated scene entries or preassigned splits:

| priority | proposed bag content | scenario evidence required | closes STEP 04 gap |
|---:|---|---|---|
| 1 | moving-ego urban driving, vehicles in front/rear/left/right with turns | exact saved scenario JSON and actor UID/list/type mapping | no driving scene; dynamic transform and rotated vehicle coverage |
| 2 | close pedestrian crossing and standing pedestrians at several bearings | scenario JSON containing every pedestrian UID plus recorded raw object type | only 2/33 current pedestrian boxes have LiDAR support |
| 3 | close non-square barriers/cones/objects in both orientations | scenario JSON containing object UID, dimensions, pose origin evidence, and raw type | current obstacle is square, so length/width cannot be validated |
| 4 | 20--100 m vehicles, pedestrians, partial occlusion, and multi-object traffic | exact scenario JSON and actor mapping | visibility/range distribution and zero-point interpretation |
| 5 | independent route/run for held-out evaluation | its own scenario JSON; no duplicated temporal segment from training bags | scene-independent validation/test coverage |

Each capture must contain the five contract topics:

```text
/ad/sensors/lidar/points
/ad/dev/objects
/ad/dev/vehicle/ego_status
/tf
/tf_static
```

Before adding a bag to the version, record its unique bag basename, simulator
build/map/scenario JSON, actor UID to scenario-list membership, raw
`object_type`, box size/origin evidence, weather/time/route, ego motion state,
and intended whole-scene split. Do not infer a split from frame counts or fill
empty `val`/`test` automatically.

## Version and merge contract

The exporter receives all bags in one invocation. A bag directory basename is
the scene ID and must be unique. The config has one explicit entry per input:

```yaml
dataset:
  name: morai_heven
  version: morai_heven_v1

scenes:
  static_20260805_003151:
    split: train
    scenario: ad_data/morai/SaveFile/Scenario/R_KR_PR_K-city_2025/checkpoint14.json
  driving_TIMESTAMP:
    split: val  # assign only after capture and leakage review
    scenario: ad_data/morai/SaveFile/Scenario/.../captured_scenario.json
```

The exporter rejects duplicate bag paths, duplicate scene basenames, missing
scene entries, missing scenario evidence, a split other than explicit
`train`/`val`/`test`, missing topics, message-type changes, and contract-invalid
frames. Static rear-axle/LiDAR transforms must also match across scenes
(quaternion sign equivalence is accepted). It never appends to an existing
dataset. Rebuild into a new output from the complete bag list so
`metadata.json`, scene summaries, and split files are one atomic logical
version.

Do not mix a bag requiring a different class mapping, transform chain, ROI, or
box convention into `morai_heven_v1`. Audit that contract and create a new
dataset version/config instead.

## Visibility statistic

Every exported GT box contains:

```json
"num_lidar_points_inside_box": 42
```

The count uses only finite XYZI points written for the same sample. Points are
rotated into the box-local frame and tested against half length, width, and
height with inclusive boundaries. The margin is exactly zero. The statistic
does not remove zero-point GT, estimate occlusion, or authorize a box
correction.

Dataset metadata aggregates zero-point, 1--5-point, and total membership counts
overall and per class. The STEP 04 validator independently recomputes the exact
count and reports `exported_visibility_mismatches`; it must remain zero.

## Merge command

From the workspace root, after adding every real scene to the versioned config:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash

python3 src/heven_ad_2026/tools/morai_dataset_exporter/export_morai_dataset.py \
  --bag bags/static_20260805_003151 \
  --bag bags/driving_TIMESTAMP \
  --bag bags/pedestrian_TIMESTAMP \
  --bag bags/non_square_obstacle_TIMESTAMP \
  --config src/heven_ad_2026/tools/morai_dataset_exporter/configs/DATASET_VERSION.yaml \
  --output datasets/DATASET_VERSION
```

Use `--overwrite` only when deliberately rebuilding an exporter-owned output.
For auditability, prefer a new output directory, validate it, then promote that
directory as the dataset version.

Validate the merged result without changing labels:

```bash
MPLCONFIGDIR=/tmp/heven-matplotlib python3 \
  src/heven_ad_2026/tools/dataset/visualize_morai_gt.py \
  --dataset datasets/DATASET_VERSION \
  --output src/heven_ad_2026/results/perception/DATASET_VERSION_validation \
  --sample-count 48
```

## Single-bag reproduction

The existing command remains valid with the same config path and scene split:

```bash
python3 src/heven_ad_2026/tools/morai_dataset_exporter/export_morai_dataset.py \
  --bag bags/static_20260805_003151 \
  --config src/heven_ad_2026/tools/morai_dataset_exporter/configs/static_20260805.yaml \
  --output datasets/morai_heven_v1
```

Expected established content is 1,764 exported frames, 2,951 boxes, class
distribution `vehicle=2844, pedestrian=33, obstacle=74`, 14 skipped frames due
to actor timestamp gaps, the complete scene in `train`, and empty `val`/`test`.
Point bytes and all pre-existing label fields remain reproducible; the new box
visibility field and version/visibility metadata are additive.

The 2026-08-15 reproduction audit exported to `/tmp/morai_heven_v1_repro` and
confirmed:

- all 1,764 point files byte-identical to the established dataset; combined
  deterministic file digest `01071c31644822c116901405d5b0f78e3304955d3dd1359c32a83ccf805d8554`;
- identical sample IDs, all pre-existing label fields, train/val/test files,
  skipped-frame CSV, and pre-existing aggregate summary fields;
- 503 zero-point and 555 one-to-five-point boxes, with 300,284 total point
  memberships across 2,951 boxes;
- independent validator recomputation with 0 visibility-count mismatches.
