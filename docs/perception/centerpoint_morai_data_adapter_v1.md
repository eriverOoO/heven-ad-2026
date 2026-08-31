# CenterPoint MORAI Data Adapter v1

## Objective

Deterministically convert **detection-valid** frames of a canonical
`morai_tracking_dataset_v1` (the MORAI Tracking Dataset Factory output)
into the exact LiDAR dataset contract consumed by this repository's
CenterPoint integration.

**No model is trained.** No pretrained weights are downloaded. No
CenterPoint architecture / config / production detector default / AB3DMOT /
tracker / planner changes. No frozen benchmark conclusion is revisited. No
simulator-to-real or generalization claim is made.

## Canonical vs derived

```
morai_tracking_dataset_v1   (canonical, authoritative — NEVER modified)
        │
        ▼
CenterPoint MORAI adapter    (this PR)
        │
        ▼
derived CenterPoint dataset  (regenerable from canonical + config)
```

The canonical dataset keeps every `PointField` (x/y/z/intensity/time/ring)
and both `map`- and `lidar_link`-frame GT boxes. The derived output is a
lossy projection: `[x, y, z, intensity]` only, `lidar_link` boxes only.
The canonical source is opened read-only.

## Current CenterPoint / OpenPCDet audit

| item | finding |
| --- | --- |
| runtime detector | `ad_lidar_perception` CenterPoint ROS node (`centerpoint_ros.py`, `centerpoint_detector_node.py`) — **opt-in, not the production default** (Euclidean is). Unchanged here. |
| offline integration | `tools/centerpoint_offline/` — `morai_dataset.py` (loader), `openpcdet_runtime.py`, `train_morai_centerpoint.py`, `verify_integration.py`, `configs/morai_heven_dataset.yaml` + `morai_centerpoint_smoke.yaml`. |
| OpenPCDet | pinned submodule `references/openpcdet` @ `233f849` (populated). Runtime needs torch/CUDA — **environment-blocked here** (no `torch`). |
| **data loader** | `tools/centerpoint_offline/morai_dataset.py::MoraiHevenDatasetCore` — **torch-free**, reads a **per-sample-JSON** layout, **not** an OpenPCDet `.pkl` info file. This is the contract the adapter targets. |
| class names | `("vehicle", "pedestrian", "obstacle")` — config `CLASS_NAMES`, model `CLASS_NAMES_EACH_HEAD`, loader `CLASS_NAMES`. Order fixed (class id = index + 1). |
| box representation | `BOX_FIELDS = ("x", "y", "z", "length", "width", "height", "yaw")` — 7-dim, read directly from flat label-JSON keys; `frame = lidar_link`. |
| point features | `POINT_FEATURE_ENCODING.used_feature_list = [x, y, z, intensity]`; loader `np.fromfile(dtype="<f4").reshape(-1, 4)`. |
| point cloud range | `POINT_CLOUD_RANGE = [-4.0, -25.0, -3.0, 100.0, 25.0, 5.0]` (config). Applied by the model's `DATA_PROCESSOR: mask_points_and_boxes_outside_range`, **not** by the loader or the exporter. |
| info pickle / camera | **none**. The loader reads no `infos/*.pkl` and no KITTI `calib` / `image_shape` / 2D boxes. The adapter fabricates none. |

## Exact target format

Option **B** — the repository's existing CenterPoint-specific per-sample-JSON
layout (`MoraiHevenDatasetCore`). This is also what the offline
`tools/morai_dataset_exporter/` produces from a recorded bag; the adapter
produces the same layout from a *factory dataset* instead.

```
<derived_root>/
  .centerpoint_morai_adapter        # marker
  export_manifest.json              # adapter schema, source hash, counts, fingerprint
  metadata.json                     # dataset_version (loader reads this) + conventions
  sample_mapping.jsonl              # target_sample_id -> source scenario/run/frame/stamp/split
  split_manifest.json               # split method, grouping key, per-split counts, leakage guarantee
  splits/{train,val,test}.txt       # one target sample_id per line
  labels/<sample_id>.json           # per-sample GT + point metadata
  points/<sample_id>.bin            # Nx4 little-endian float32 [x, y, z, intensity]
```

`labels/<sample_id>.json`:

```json
{
  "adapter_schema_version": "centerpoint_morai_adapter_v1",
  "sample_id": "<run_id>_<lidar_stamp_ns>",
  "source": { "dataset_id", "scenario_id", "run_id", "requested_seed",
              "frame_index", "source_sample_id", "header_stamp_ns" },
  "points": { "path": "points/<id>.bin", "dtype": "float32_little_endian",
              "feature_names": ["x","y","z","intensity"],
              "finite_count", "source_point_count", "nonfinite_dropped",
              "canonical_fields_preserved_in_source": ["x","y","z","intensity","time","ring"] },
  "ground_truth": { "target_frame": "lidar_link",
                    "box_fields": ["x","y","z","length","width","height","yaw"],
                    "header_stamp_ns",
                    "boxes": [ { "class_name": "vehicle",
                                 "x","y","z","length","width","height","yaw",
                                 "source_class_name", "source_actor_id",
                                 "source_raw_object_type",
                                 "num_lidar_points_inside_box" } ] }
}
```

`MoraiHevenDatasetCore` reads `metadata.json.dataset_version`,
`splits/<split>.txt`, `labels/<id>.json` (`sample_id`,
`ground_truth.target_frame == "lidar_link"`, `points.path`,
`points.finite_count`, `ground_truth.boxes[].{class_name, x,y,z,length,width,height,yaw}`),
and `points/<id>.bin`. The adapter output satisfies every one of these;
`test_repo_loader_core_loads_adapter_output` instantiates the real loader
core against the export and asserts `points.shape == (N, 4)`,
`gt_boxes.shape == (K, 7)`, `gt_names ⊆ CLASS_NAMES`, `coordinate_frame ==
"lidar_link"`, and label↔`gt_boxes` parity to `1e-6`.

## Point feature mapping

| target `.bin` column | source | transform |
| --- | --- | --- |
| 0 `x` | canonical npz `x` (`float32`) | identity |
| 1 `y` | canonical npz `y` | identity |
| 2 `z` | canonical npz `z` | identity |
| 3 `intensity` | canonical npz `intensity` | **identity** (no /255, no clip, no normalize) |

`time` and `ring` are **not** exported — they stay in the canonical npz.
`intensity_transform: "identity"` is recorded in `metadata.json`. Non-finite
points are dropped (the loader asserts `np.isfinite(points).all()`); the
canonical dataset keeps them. Per sample the label records
`source_point_count`, `finite_count`, `nonfinite_dropped`; the manifest
totals `source_points`, `exported_points`, `nonfinite_points_dropped`. The
derived `.bin` is therefore **not lossless** w.r.t. the canonical cloud.

No point-range cropping (`point_range_cropping_applied: false`) — the
model's `DATA_PROCESSOR` masks outside-range points. Boxes whose centre
falls outside `POINT_CLOUD_RANGE` are **counted**
(`counts.boxes_outside_reported_range`) but never deleted.

## Box convention

* **order**: `[x, y, z, length, width, height, yaw]` (7-dim).
* **centre**: geometric box centre `x/y/z` in `lidar_link`. Taken **verbatim**
  from the factory's already-derived `box.lidar_frame.center` — the adapter
  does **not** recompute any `map → lidar` transform. **No z shift**
  (`z += h/2` / `z -= h/2`) is applied: the factory `lidar_frame` centre is
  already the geometric centre, matching the loader's `_identity_box_delta`
  expectation and OpenPCDet's centre convention.
* **dimensions**: `length, width, height` copied in that order from
  `box.lidar_frame`. `test_box_dimension_order_is_length_width_height_not_swapped`
  guards against an l/w swap.
* **yaw**: CCW from `lidar_link` +X, wrapped to `[-π, π]` via
  `atan2(sin θ, cos θ)`. The factory already emits `[-π, π]`; the wrap is a
  belt-and-suspenders identity there. Tested at `0`, `±π/2`, near `π`, and a
  wrapped value (`3.0`), asserting `sin`/`cos` parity with the source.
* **units**: metres, radians. **Right-handed** `lidar_link` (+X forward,
  +Y left, +Z up), inherited from the factory.

## Class mapping

`config/dataset_factory/centerpoint_adapter/class_map.yaml`:

```yaml
class_map:
  vehicle: vehicle
```

**v1 is vehicle-only** — the factory starter catalog only reliably
ground-truths vehicles (Phase 15). Target class names stay
`(vehicle, pedestrian, obstacle)` (the model config is unchanged); the
adapter simply never emits the other two until a scenario confirms them.

A source box whose `class_name` is **not a key** in `class_map` is
**omitted from that frame** (the frame is still exported) and counted in
`export_manifest.json → counts.boxes_omitted_by_source_class`. It is **never
relabelled**. Note this is "unmapped-**by-adapter**": the factory already
forces `valid_for_detection_gt = false` on any frame containing a
`CLASS_UNKNOWN` box, so the only boxes the adapter ever omits are
factory-mapped `pedestrian` / `obstacle` under a stricter adapter map.
`test_unmapped_class_is_omitted_from_frame_not_relabelled` uses a
factory-valid `pedestrian` box for exactly this path.

## Frame eligibility

A source frame is exported iff its `frames.jsonl` row has
`valid_for_detection_gt == true`. The adapter never loosens the factory's
criteria and never promotes a `valid_for_tracking_gt`-only frame. Excluded
frames are tallied by reason (`counts.exclusion_reason_counts`, keyed by the
sorted `validation_flags`).

## Negative frames

A detection-valid frame with **zero mapped boxes** (no actors, or all actors
unmapped) is exported by default with an empty `ground_truth.boxes` list —
useful as a negative detection sample. `counts.negative_frames` reports how
many. `test_negative_frame_is_exported_with_empty_gt` covers it.

## Sample ID & provenance

`sample_id` = the factory `sample_id` (`<run_id>_<lidar_stamp_ns>`) — already
unique and provenance-bearing, so no numeric renumbering is needed (the
loader keys on string ids). Every sample is traceable via:

* the label's `source` block (dataset id, scenario, run, seed, frame index,
  source sample id, stamp),
* `sample_mapping.jsonl` (one line per exported sample with the same fields +
  assigned split),
* `export_manifest.json` (`source_dataset_id`,
  `source_dataset_manifest_sha256`, `adapter_repository_commit`).

## Deterministic split grouping — leakage prevention

**Grouping key: `(scenario_id, requested_seed)`**, read from
`run_manifest.config.seed`. When the seed block is absent, the fallback key
is `run_id`. **Frames of one run are never split** across train/val/test —
a run belongs entirely to one group, a group entirely to one split.

This directly prevents the prior CenterPoint research failure mode (1764
train samples, 0 val/test, 100 % train/eval overlap): a frame cannot appear
in val/test merely because its frame index differs from a training frame in
the same run/seed.

* **Explicit plan** (`--split-plan`, priority):
  `splits: {train: [{scenario, seeds: [...]}], val: [...], test: [...]}`.
  A group listed twice → hard error. A group referenced but absent from the
  source → hard error. A source group not referenced → dropped with a
  warning (`counts.dropped_by_split_plan`).
* **Auto-split** (default): stable **SHA-256** bucket of
  `adapter_schema_version | scenario_id | seed_key` against
  `train/val/test` ratios (default `0.7 / 0.15 / 0.15`). Never Python's
  process-salted `hash()`. Because each group is bucketed independently,
  auto-split at a **small group count does not reliably fill all three
  splits** — e.g. 6 groups routinely yields an empty `test` (a warning is
  emitted). Auto-split is leakage-safe but coarse; the **explicit
  `--split-plan` is the recommended path** until there are enough
  independent `(scenario, seed)` groups (roughly ≥ 15–20) for the ratios
  to land.
* **Small dataset**: `< 3` independent groups → **all to train**, with an
  explicit warning. A single run is never frame-split to fake val/test.
* **Leakage validator** (`check_split_leakage`, run inside `export` and again
  in the standalone validator): fails on a group or a run spanning multiple
  splits, or a duplicate target sample id.

## Output CLI

```
ros2 run ad_morai_bridge_dev ad_morai_dataset_export_centerpoint \
    <source_dataset_root> --output <derived_root> \
    [--split-plan plan.yaml] [--class-map class_map.yaml] \
    [--auto-ratios 0.7,0.15,0.15] [--dry-run] [--validate-only] [--overwrite]
```

* `--validate-only` runs the factory validator + schema-version check on the
  source and exits.
* `--dry-run` reports source / eligible / negative frame counts, boxes,
  split group + frame counts, and warnings without writing points/labels.
* Export writes to `<derived_root>.tmp` and atomically renames to
  `<derived_root>` on success; `export_manifest.json` (status `complete`) is
  written last. A completed export is not overwritten without `--overwrite`,
  which only proceeds on a directory carrying the adapter marker.

## Export validator

```
ros2 run ad_morai_bridge_dev ad_morai_dataset_validate_centerpoint <derived_root>
```

Checks: marker + `export_manifest` status `complete`, adapter schema
version, source-hash present, `metadata`/`split_manifest`/`sample_mapping`
present, per split: `.bin` exists + `Nx4` `<f4` + all finite +
`finite_count` matches, label `sample_id` / `target_frame` / `box_fields`
match, box class in `CLASS_NAMES`, box dims finite + positive, yaw in
`[-π, π]`, every split id present in `sample_mapping`, no duplicate id
across splits, **no group / run split leakage**, manifest counts consistent.
Non-zero exit on any failure.

## Summary in the manifest

`export_manifest.json` carries `counts` (source / eligible / exported /
excluded frames, exclusion reasons, dropped-by-plan, boxes source / exported
/ omitted / outside-range, negative frames, source / exported points,
non-finite dropped, `boxes_omitted_by_source_class`) and
`split_frame_counts`. `split_manifest.json` carries per-split frame counts,
negative-frame counts, **class counts by split**, **near/mid/far range
counts by split** (`≤20 m` / `≤45 m` / `>45 m` from `hypot(x, y)`), source
run ids, and warnings. No performance metric is produced. No frame
duplication / oversampling / rebalancing is done.

## Content fingerprint

`export_manifest.content_fingerprint` = SHA-256 over
`adapter_schema_version | source_dataset_manifest_sha256 |
canonical(adapter_config) | canonical(split_manifest) | sorted(per-artifact
SHA-256 of every .bin and every label JSON)`. `created_at` is deliberately
**not** part of it. `test_repeat_export_is_bit_identical` exports the same
source twice to different directories and asserts identical fingerprints and
byte-identical `.bin` files.

## Determinism contract

Same source + adapter config + split config + adapter code ⇒ identical
sample selection, ids, split membership, `.bin` bytes, GT arrays, and
mapping files. Frames are consumed in a stable
`(scenario_id, run_id, frame_index)` sort; points are written in stored
order with non-finite rows removed (no shuffle); no augmentation / noise /
jitter / beam simulation / dropout / densification of any kind.

## Test fixture

`ad_morai_bridge_dev/test/test_centerpoint_adapter.py` builds every source
dataset through the **real** `RunWriter` + `CaptureSession` (schema
`morai_tracking_dataset_v1`), with **6 independent `(scenario, seed)`
groups** (`lead_constant` 0/1/2, `cut_in` 0/1, `dense_multi_object` 0) so
train/val/test can be exercised without frame leakage. Datasets live in
pytest `tmp_path` only.

## Actual source dataset status

**No real `morai_tracking_dataset_v1` dataset exists on disk** (MORAI is
absent — see the factory doc). The adapter is validated on fixture datasets
built through the real factory writer, plus the real repo loader core
(`MoraiHevenDatasetCore`) loading the adapter output.

**Full OpenPCDet `DatasetTemplate` (`make_openpcdet_dataset`) was NOT run** —
it needs torch/CUDA, environment-blocked here. Loader-core compatibility
**is** runtime-proven; the full model dataloader is not.

## Size estimate

Derived `.bin` = `N × 4 × 4` bytes. Using the existing `datasets/morai_heven`
density (~14 110 finite points / frame): **≈ 0.22 MB / frame**, **≈ 0.22 GB /
1000 frames**, plus ~1–3 KB / frame of label JSON. The canonical npz is
~0.64 MB / frame, so the derived dataset is roughly **1/3** the canonical
size (4 of 6 fields, no zip container).

## No CenterPoint training

**NO CENTERPOINT TRAINING OCCURRED.** No weights downloaded. `configs/` and
the model architecture are unchanged. A separate future task
("CenterPoint MORAI Training Prep v1") wires the model config to this
output.

## Simulator-to-real caveats

* MORAI point intensity / noise may differ from a real VLP-16.
* Beam / ring elevation distribution may differ.
* MORAI object model geometry and traffic distribution may differ.
* Perfect simulator GT does not exist on the real vehicle.

A leakage-safe train/val/test split removes **run/seed** overlap. It does
**not** establish real-world generalization and does **not** solve the
simulator-to-real domain gap. All data remains MORAI-simulator domain.

## Previous train/eval-overlap caveat

The historical CenterPoint result (1764 train / 0 val / 0 test / 100 %
overlap) supported **no** generalization claim. This adapter's
group-level split prevents that specific failure, but the historical result
must not be compared against an adapter split until a genuinely
non-overlapping validation set (ideally with real data) actually exists.
