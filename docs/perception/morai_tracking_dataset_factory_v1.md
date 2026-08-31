# MORAI Tracking Dataset Factory v1

## Objective

A deterministic, reusable MORAI scenario / reset / capture pipeline that
records **one common raw + ground-truth source** for three later workstreams:

1. object-tracking development and evaluation (persistent GT identities + poses/boxes over time),
2. CenterPoint detector adaptation (raw point clouds + timed 3D boxes/classes),
3. KalmanNet trajectory-estimation development (timed GT object trajectories).

**No model is trained here.** No production detector / tracker / planner
default changes. No frozen research claim (T2–T15, KalmanNet, CenterPoint)
is touched. This PR is data infrastructure only.

## Architecture

```
scenario catalog (data)  ─┐
seed → RunParameters      ─┤→ build_reset_plan → MORAI gRPC reset ─┐
                                                                   │  (settle)
/ad/sensors/lidar/points ─┐                                        ▼
/ad/dev/vehicle/ego_status ├─→ CaptureSession ── FrameAssembler ──→ RunWriter
/ad/dev/objects           ─┤     (LiDAR-anchored,   (per-frame       (atomic
/tf , /tf_static          ─┘      skew-checked)      canonical        frames +
                                                     sample)          manifests)
                                                                       │
                                        ad_morai_dataset_validate  ◄────┤
                                        ad_morai_dataset_summary   ◄────┘
```

Everything under `ad_morai_bridge_dev/ad_morai_bridge_dev/dataset/` sits
next to the existing `scenarios/` reset machinery and reuses it
(`scenarios.reset.reset_scenario`, `scenarios.setup.load_scenario_and_resume`,
`simulator_grpc.client.MoraiGrpcClient`). No second simulator-reset
framework was created.

## Sample anchor and time domains

MORAI does not guarantee a single clock across sources. Audited domains:

| source | topic | stamp | domain |
| --- | --- | --- | --- |
| raw LiDAR | `/ad/sensors/lidar/points` | `header.stamp` | **canonical sample key** |
| ego GT | `/ad/dev/vehicle/ego_status` (`EgoVehicleStatus` dev) | `header.stamp` | matched to LiDAR |
| actor GT | `/ad/dev/objects` (`ObjectStatusArray` dev) | `header.stamp` | matched to LiDAR |
| dynamic TF | `/tf` `odom→base_link` | per-transform `header.stamp` | matched to LiDAR |
| static extrinsics | `/tf_static` | — | stored once per run |

Each accepted raw LiDAR frame is exactly one dataset sample, keyed by its
`header.stamp` (never callback-receipt time). Every other source is
selected by **nearest source `header.stamp`, no interpolation**, with an
explicit maximum skew. An out-of-skew or absent source is recorded as
`skew_exceeded` / `missing` — never silently replaced with the latest
value or a zero transform. Per-frame actual skew is written to the frame
index.

Default max skew (ns), overridable per run: ego 30 ms, actor GT 30 ms, TF
30 ms (mirrors the offline exporter's audited ceiling).

A LiDAR stamp that is non-positive, duplicate, or backward is rejected and
counted (`rejected_frames`), never written.

## Ego, actor and TF sources

* **Ego GT** is `EgoVehicleStatus` on `/ad/dev/vehicle/ego_status` — the
  simulator-native map pose / rpy / linear + angular velocity / acceleration.
  It is flagged `is_simulator_ground_truth: true`. The localization estimate
  (`/ad/localization/odometry`) is **not** captured in v1 (it is a
  derived estimate, not GT); the dynamic `odom→base_link` TF is captured
  instead and is what the transform chain needs.
* **Actor GT** is `ObjectStatus[]` on `/ad/dev/objects`: `unique_id`,
  `object_type`, map `position`, `heading`, `size` (l/w/h), `velocity`,
  vehicle `overhang`/`wheelbase`/`rear_overhang`, `link_id`. The array
  carries one `header.stamp`; there is no per-object timestamp.

## GT identity

`gt_track_id` is the native **`ObjectStatus.unique_id`** — a stable
per-actor simulator id. It is never derived from array index, nearest
position or detection order. The frame index stores `actor_gt.actor_ids`
per frame so identity continuity is directly checkable.

## 3D box convention

Dimensions come from `ObjectStatus.size` (a real source — Phase 10 is
satisfiable). The centre policy is a **direct port of the audited offline
exporter** (`tools/morai_dataset_exporter/export_morai_dataset.py::
_transform_box`), locked by `test_box_centre_policy_matches_offline_exporter`:

* the simulator reports a **vehicle** pose at the rear-axle centre on the
  ground → box centre shifts forward by
  `(wheelbase + overhang - rear_overhang) / 2` and up by `height / 2`,
  with a per-actor consistency check `|Σ(overhang,wheelbase,rear_overhang) − length| ≤ 0.25 m`;
* **pedestrian / obstacle** report a ground-centred origin → shift up by
  `height / 2` only.

No arbitrary z offset. A failing per-actor check (bad geometry, non-finite
pose, non-positive dims) **flags that actor** and clears its
`valid_for_detection_gt`; the frame is still written with the other actors
(unlike the exporter, which drops the whole frame — a factory keeps the
evidence, Phase 23).

## Coordinate frames

* Canonical GT is stored in the **`map`** frame (`box.map_frame`): centre,
  l/w/h, yaw.
* A derived **`lidar_link`** box (`box.lidar_frame`) is added when the full
  chain `map → odom → base_link → rear_axle_link → lidar_link` is valid for
  that frame. `odom == map` is never assumed; `map → odom` is derived per
  frame from the simulator ego map pose and the recorded `odom → base_link`.
* Static `base_link→rear_axle_link` and `rear_axle_link→lidar_link` come
  from `/tf_static` and are stored once in each `run_manifest` and each
  frame's `tf.static_edges`.

## Class mapping

`RAW_TYPE_TO_CLASS = {0: pedestrian, 1: vehicle, 2: obstacle}`,
version `checkpoint14_evidence_v1`. This mapping was only ever proven
against one recorded MORAI scenario (`checkpoint14.json`, see the exporter
`class_evidence`). Therefore:

* `raw_object_type` is **always** retained in every box record;
* an unmapped raw type becomes `CLASS_UNKNOWN` and is **never dropped**;
* every `gt/*.json` records `class_map_scenario_file` + `class_map_scenario_sha256`
  — the scenario file that justified the mapping for that run. A new
  scenario does not silently inherit `checkpoint14`'s evidence.

Starter scenarios are **vehicle-only**; pedestrian/obstacle GT is not
claimed until a scenario re-confirms the mapping.

## Point storage

`lidar/NNNNNN.npz` — one named array per `PointField`, dtype preserved
(x/y/z/intensity/time as `float32`, MORAI's `ring` as `uint16`). Big-endian
clouds and `count != 1` fields are rejected, not reinterpreted. A later
exporter may down-cast to `[x, y, z, intensity]`; the canonical frame keeps
`ring` / `time`. Content-lossless; not byte-deterministic (the npz zip
container carries timestamps) — read-back equality is the contract, per
Phase 52.

## Validity flags

Per frame, in `frames.jsonl`:

* `valid_for_tracking_gt` — actor GT matched and in-skew, ≥1 actor with
  id / pose / orientation / velocity.
* `valid_for_detection_gt` — additionally: ego + TF matched, every retained
  box has a finite positive-dimension mapped-class `lidar_frame` geometry.

The dataset is **not** "CenterPoint-ready" merely because point clouds
exist; only frames with `valid_for_detection_gt` are.

## KalmanNet usage

KalmanNet does not consume raw LiDAR. The schema preserves what a later
extraction needs: stable `actor_id`, timed GT state (`map_frame` centre +
`source.velocity_map` + `heading`), ego motion, and scenario/run
boundaries. No detector noise is synthesised here; a later task pairs GT
trajectories with real detector/tracker measurements.

## Directory layout

```
<dataset_root>/
  .morai_tracking_dataset            # marker
  dataset_manifest.json
  schema.json
  runs/<scenario_id>/<run_id>/
    run_manifest.json
    frames.jsonl                     # authoritative frame index
    lidar/000000.npz ...
    gt/000000.json ...               # boxes in map + lidar_link
    ego/000000.json ...
    tf/000000.json ...
```

`run_id = <scenario_id>__seed_<seed>__run_<NNN>`, auto-incrementing;
an existing run is refused unless `--overwrite`.

## Manifests

* **dataset_manifest.json**: schema version, `dataset_id`, `created_at`,
  repository commit + dirty flag, and a per-run status/frame-count roll-up.
* **run_manifest.json**: schema version, `status`
  (`running`→`complete`/`aborted`/`error`), scenario id + file + SHA-256,
  seed parameters (`simulator_determinism_guaranteed` defaults **false**),
  reset-plan summary, limits, skew config, topic contract, repo provenance,
  and a summary (captured frames, valid tracking / detection frames, unique
  actor ids, class counts, max observed skew, rejected-frame reasons,
  point-field schema, termination reason).

## Seed / determinism

MORAI is not known to guarantee a physics seed. A `scenario_seed` still
deterministically derives small placement / velocity perturbations
(`derive_run_parameters`, bounded ±4 m longitudinal / ±1 m lateral /
±1.5 m/s) so repeated captures of one `(scenario, seed)` are reproducible
**on the factory side**. Every run manifest states
`simulator_determinism_guaranteed: false`.

## Scenario catalog

`config/dataset_factory/scenario_catalog.yaml` — 6 starter scenarios, each
a standard MORAI scenario JSON (`egoVehicle` / `vehicleList`) parsed by the
existing `load_reset_plan`:

| id | regime | purpose |
| --- | --- | --- |
| `lead_constant` | mid | persistent single lead, ID continuity |
| `lead_brake` | near | closing range / large velocity change |
| `cut_in` | near | lateral trajectory into the ego lane |
| `crossing` | mid | perpendicular crossing, birth/death at edges |
| `occlusion_reappear` | mid | trailing actor geometrically occluded then reappears |
| `dense_multi_object` | mixed | several nearby actors across lanes/speeds |

They collectively exercise persistent ID, velocity variation, birth/death,
crossing, crowding, and near/mid/far range (VLP-16 sparsity).

**The scenario poses are templates** grounded on the `kcity-highway` actor
preset and must be validated against a live K-City map before a production
capture. `occlusion_reappear` names an *intent* only — the GT record always
reports "actor exists in simulator", never a visual-occlusion flag, unless
MORAI exposes one.

## CLI

```
ros2 run ad_morai_bridge_dev ad_morai_dataset_capture \
    --scenario cut_in --seed 0 --output ~/datasets/morai_tracking_v1 --frames 200
```

* `--list-scenarios`, `--dry-run` (build the reset plan + run config, no ROS),
  `--no-reset` (capture from an already-running graph, no gRPC),
  `--duration-sec` / `--frames`, `--settle-sec`, `--overwrite`,
  `--max-{ego,actor-gt,tf}-skew-ms`.

Reset sequence: `load scenario → GetAllActorsState → ControlVehicle(ego 0) →
SetVelocity(ego 0) → SetTransform(ego) → per-actor SetVelocity + SetTransform
→ Resume`, then a `--settle-sec` window whose frames are **not** samples,
then capture until the frame / duration limit, then finalize.

## Batch capture

```
ros2 run ad_morai_bridge_dev ad_morai_dataset_batch \
    ad_morai_bridge_dev/config/dataset_factory/dataset_plan.example.yaml \
    --output ~/datasets/morai_tracking_v1
```

`dataset_plan.yaml`: `dataset_id`, `frames`/`duration_sec`, `settle_sec`,
and a `runs: [{scenario, seeds: [...]}]` list. Sequential, single machine.

## Validation

```
ros2 run ad_morai_bridge_dev ad_morai_dataset_validate <dataset_root>
```

Checks: marker + schema version, run finalized (not `running`), frame-index
contiguity, strictly-increasing LiDAR stamps, all artifact files present,
npz point-count vs index row, finite float point fields, unique sample ids,
matched-source skew sanity, box finiteness / positive dims, manifest
frame-count consistency. Non-zero exit on any structural failure.

```
ros2 run ad_morai_bridge_dev ad_morai_dataset_summary <dataset_root>
```

Reports run/frame counts, duration, point-count and GT-actors-per-frame
percentiles, unique GT ids, class counts, near/mid/far distribution,
invalid-sync frame count. No model performance.

## Interrupted-run behaviour

Each artifact is written to a `.tmp` sibling and atomically renamed; the
`frames.jsonl` row is appended (and `fsync`ed) only after every artifact
for that frame is on disk, so a crash never leaves an index row pointing at
a partial frame. `Ctrl-C` / simulator failure finalizes the run as
`aborted` / `error` with the real `termination_reason`, never `complete`.
An existing run directory is refused unless `--overwrite`.

## Storage estimate

npz, uncompressed, ~28.8k points/frame, x/y/z/intensity/time `f4` + `ring`
`u2`:

* **≈ 0.64 MB / frame**
* **≈ 0.64 GB / 1000 frames**
* **≈ 23 GB / hour** at 10 Hz LiDAR

`np.savez_compressed` (~0.53 MB/frame on random data, better on real
clouds) is a one-line future change if disk pressure warrants it.

## Generated data is not committed

The conventional dataset root (`datasets/…` in-repo, or `~/datasets/…`) is
already git-ignored (`.gitignore` line `datasets/`). Test fixtures are
**synthesised in-test** — no LiDAR frame is committed.

## Real MORAI validation status

**Real MORAI capture: NO.** `import grpc` fails in this environment; MORAI
is a GPU/Unity simulator not present here, and its actor-GT producer OOMs
this host. The pipeline is validated by:

* 26 unit / dry-run tests (`test_dataset_factory.py`) covering schema,
  lossless point round-trip, box centre-policy parity with the offline
  exporter, the 10 timestamp/skew cases, the writer + atomic partial-frame
  behaviour, run-collision refusal, interrupted-run status, the full
  capture-session → validator → summary path on synthetic messages, the
  scenario catalog load + seed reproducibility + jitter bounds;
* `ad_morai_dataset_capture --dry-run` / `--list-scenarios`, batch
  `--dry-run`, and the validator / summary CLIs run end-to-end.

## Limitations

* No live MORAI run — the reset / capture node is exercised only against
  synthetic messages and dry-runs.
* Starter scenario poses are map-unvalidated templates.
* Class mapping evidence is inherited from one recorded scenario; new
  scenarios must re-confirm it before their pedestrian/obstacle GT is trusted.
* The **dev** `EgoVehicleStatus` variant has no `device_stamp` field (only
  the non-dev `ad_morai_interfaces` one does); its `header.stamp` is used as
  the source time.
* Occlusion is a scenario name, not a per-actor label.
* npz is content-lossless but not byte-deterministic.

## Forbidden conclusions (not made anywhere)

CenterPoint / KalmanNet / tracking accuracy improved; dataset generalizes
to a real vehicle; the simulator dataset closes the domain gap. None of
these are claimed. No model is trained.
