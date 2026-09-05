# AV2 KalmanNet Scale-Up Dataset v2

Data acquisition / preparation task. **No model was trained in this
task.** Builds on merged PR #53 (`feat/morai-estimator-eval-v2`, merge
commit `8cdcc5d0ce63f12c6e152c3f806c9665c9504a8a`), verified via
`gh pr view 53` before branching.

## Why AV2 work continues while MORAI is unavailable

The prior task (MORAI Frozen Estimator Evaluation Dataset v2) found that
MORAI simulator access has never been available anywhere in this
project's history and no new independent MORAI run can be captured in
this environment. Rather than block all KalmanNet progress on that,
this task scales up the AV2 side of pretraining -- AV2 has no such
availability constraint (public, unsigned-read S3 bucket) and can
continue to advance the KalmanNet architecture/training-recipe work.
**AV2 evaluation is not a substitute for MORAI competition-domain
evaluation** -- this boundary is stated explicitly, not left implicit,
and is repeated throughout this document.

## 1. Previous AV2 exclusion set

Every scenario_id ever used in this project's AV2 experiments was
enumerated directly from the two real, on-disk selection manifests:

| experiment | split | count | seed | pool size |
|---|---|---|---|---|
| Stage-0 (`stage0_scenarios.json`) | train | 120 | 20260905 | 2,000 |
| Stage-1 pilot (`stage1_pilot_scenarios.json`) | train | 2,000 | 20260910 | 10,000 |

Union: **2,120 unique scenario_ids** (Stage-1's own manifest already
recorded `stage0_overlap_in_selection: 0`; independently re-verified here
by set intersection: `0`). Both prior experiments used AV2's `train`
split exclusively -- no AV2 `val` scenario has ever been touched by this
project before this task.

## 2. 10k TRAIN selection

`tools/av2_dataset_prep/fetch_av2_scaleup_v2.py` (new; generalizes
`fetch_av2_stage1.py`'s single-`--exclude-manifest` behavior to a
repeatable flag, so both prior manifests' ids are excluded together,
never just one):

```
--split train --pool-size 20000 --count 10000 --seed 20260920
--exclude-manifest stage0_scenarios.json --exclude-manifest stage1_pilot_scenarios.json
```

Real run: listed 20,000 train scenario ids; **all 2,120** previously-used
ids were confirmed present in that listing (expected -- the same live
bucket); selected 10,000 fresh scenarios, **0 overlap with the exclusion
union (verified)**. Selection rule unchanged from Stage-0/Stage-1:
uniform random sample without replacement, seeded, sorted -- no
motion-statistic or performance-based cherry-picking.

## 3. 1k official VAL selection

Same tool, `--split val` (AV2's own official validation split, a
structurally disjoint scenario-id pool from `train` by AV2's own
construction):

```
--split val --pool-size 4000 --count 1000 --seed 20260921
--exclude-manifest stage0_scenarios.json --exclude-manifest stage1_pilot_scenarios.json
```

Real run: **0 of the 2,120 excluded ids were present in the val-split
listing** (confirms train/val are disjoint id spaces, not merely
assumed); selected 1,000 scenarios, 0 overlap. This split is
**evaluation-only** -- never used for training, early stopping,
corruption fitting, or optimizer calibration in this or any future task
reusing it.

## 4. Download size

| | scenarios | bytes downloaded | mean/scenario |
|---|---|---|---|
| TRAIN | 10,000 | 1,482,235,698 (1.48 GB) | 148.2 KB |
| official VAL | 1,000 | 151,175,497 (151 MB) | 151.2 KB |

0 failed downloads in either run (`failed_scenario_ids: []`). Both
figures are within ~1% of the estimate made before downloading (using
Stage-1's own real measured 146.6 KB/scenario mean).

## 5. Disk usage

| | before this task | after this task |
|---|---|---|
| `~/datasets/av2/` total | 1.2 GB | **6.3 GB** |
| raw TRAIN | -- | 1.5 GB |
| raw official VAL | -- | 151 MB |
| processed TRAIN shards | -- | 3.2 GB |
| processed official VAL shards | -- | 334 MB |
| host free disk | 902 GB | 897 GB |

6.3 GB total, comfortably under the ~100 GB soft AV2 budget (no STOP
condition triggered) and using well under 1% of the 897 GB still free.

## 6. Sharding

Reuses the existing, unmodified `av2_motion_forecasting_adapter_cli.py`
+ the already-frozen `scenarios_per_shard: 200` shard-size policy
(`ad_morai_bridge_dev/config/dataset_factory/av2_motion_forecasting_adapter/shard_config.yaml`,
unchanged from Stage-0/Stage-1 -- derived from a real Stage-0 measurement,
~366.7 KB/scenario -> ~73 MB/shard). 10,000/200 = exactly 50 TRAIN
shards expected; 1,000/200 = exactly 5 official-VAL shards expected.
**No corruption materialized into either shard set** -- both exports use
the default `CorruptionConfig(mode="none")` (metadata-only provenance),
matching the canonical clean-shard convention every prior AV2 task in
this project has already established. Corruption remains a deterministic
load-time transform (`av2_kalmannet_shard_loader.py`), never re-derived
here.

TRAIN and official-VAL are exported into **physically separate output
roots** with independent single-role split manifests
(`build_scaleup_v2_split_manifests.py`, new -- deliberately not
`av2_split.py::assign_split`, which enforces a mandatory train/val/test
carve-out of one pool; here there are two independent pools, each
already 100% one role). The TRAIN root's manifest has 0 val/0 test
scenarios; the official-VAL root's manifest has 0 train/0 test scenarios
-- structurally impossible for a training loader pointed at the TRAIN
root to ever see an official-VAL sequence.

## 7. Validation result

Both exports pass the existing strict shard validator
(`av2_motion_forecasting_adapter_validate.py --json`) with **0 errors, 0
warnings**:

| | shards checked | segments checked | result |
|---|---|---|---|
| TRAIN | 50/50 | 553,898/553,898 | `ok: true` |
| official VAL | 5/5 | 56,521/56,521 | `ok: true` |

Checks covered: exact shard/segment-offset consistency, per-segment
finiteness, monotonic timestamps + positive dt within each segment,
`coarse_object_type` in the fixed 3-class taxonomy, `coordinate_mode ==
first_state_relative`, and the short-segment-must-be-flagged-invalid
rule. Both exports also report `frames_dropped_backward_timestep`,
`frames_dropped_duplicate_timestep`, `frames_dropped_nonfinite`, and
`tracks_fully_rejected_nonfinite` all at **0** -- every downloaded
scenario parsed cleanly.

## 8. TRAIN dataset statistics

10,000 scenarios -> **553,898 segments** (553,810 `valid_for_kalmannet_gt`)
-> **28,572,250** total GT samples.

| statistic | value |
|---|---|
| track length p50 / p90 / p95 / p99 / min / max (frames) | 39 / 110 / 110 / 110 / 1 / 110 |
| speed p50 / p90 / p95 / p99 (m/s) | 0.0003 / 10.06 / 12.61 / 16.43 |
| acceleration-proxy p50 / p90 / p95 / p99 (m/s^2) | 0.0015 / 1.73 / 2.79 / 8.44 |
| near-stationary fraction (speed < 0.5 m/s) | 64.17% |
| moving fraction | 35.83% |

AV2 `ObjectType`: VEHICLE 405,568, PEDESTRIAN 53,353, STATIC 39,270,
BACKGROUND 26,840, RIDERLESS_BICYCLE 8,052, CONSTRUCTION 8,776, CYCLIST
3,936, BUS 6,010, MOTORCYCLIST 1,238, UNKNOWN 855. Coarse: VEHICLE
411,578, PEDESTRIAN 53,353, OTHER 88,967 -- the same real class
imbalance every prior AV2 task in this project has already documented
(VEHICLE-dominant), confirmed again at 5x the prior scale, not assumed
to persist.

`TrackCategory`: TRACK_FRAGMENT 448,727, UNSCORED_TRACK 66,477,
SCORED_TRACK 28,694, FOCAL_TRACK 10,000 (exactly one focal track per
scenario, as expected). City: miami 168,237, austin 112,991, pittsburgh
106,277, washington-dc 93,080, dearborn 46,106, palo-alto 27,207.

## 9. Official VAL statistics

1,000 scenarios -> **56,521 segments** (56,521 `valid_for_kalmannet_gt`)
-> **2,918,906** total GT samples.

| statistic | value |
|---|---|
| track length p50 / p90 / p95 / p99 (frames) | 39.0 / 110.0 / 110.0 / 110.0 |
| speed p50 / p90 / p95 / p99 (m/s) | 0.00026 / 10.15 / 12.59 / 16.20 |
| acceleration-proxy p50 / p90 / p95 / p99 (m/s^2) | 0.0012 / 1.71 / 2.78 / 8.46 |
| near-stationary fraction (speed < 0.5 m/s) | 64.59% |
| measurement-available fraction | 100% (clean, unaugmented export) |

Essentially identical distribution shape to TRAIN (expected -- both are
uniform random samples of the same overall AV2 pools, just from disjoint
scenario sets and, for VAL, AV2's own official `val` split rather than
`train`). AV2 `ObjectType`: VEHICLE 41,526, PEDESTRIAN 5,207, STATIC
4,136, BACKGROUND 2,634, RIDERLESS_BICYCLE 856, CONSTRUCTION 909, CYCLIST
364, BUS 687, MOTORCYCLIST 134, UNKNOWN 68. `TrackCategory`:
TRACK_FRAGMENT 45,849, UNSCORED_TRACK 6,692, SCORED_TRACK 2,980,
FOCAL_TRACK 1,000. City: miami 16,651, austin 12,038, pittsburgh 10,370,
washington-dc 9,928, dearborn 4,818, palo-alto 2,716.

## 10. Motion / stationary distribution (descriptive only, not filtered)

Confirmed, at 5x the prior AV2 scale, the same near-stationary-track
finding the domain-gap task already flagged: **64.17% of TRAIN frames
(64.59% of VAL frames) are below the 0.5 m/s stationary threshold**.
Median speed across the whole dataset is essentially zero (0.0003 m/s).
**Not filtered** -- both shard sets retain every scenario/track as
downloaded; this is reported purely as a description of what the frozen
dataset actually contains.

## 11. dt distribution

`dt_s` is **exactly 0.1 s at every reported percentile (p50/p90/p95/p99)**
for both TRAIN and official VAL -- AV2's real 10 Hz nominal sampling
rate holds uniformly across this fresh 11,000-scenario pull, same as
every prior AV2 task's own finding. Timestamps were not modified; no
variable-dt augmentation was added.

## 12. Class metadata

Real AV2 `ObjectType` preserved verbatim per segment (unchanged
adapter behavior); the existing `coarse_object_type`
(VEHICLE/PEDESTRIAN/OTHER) grouping is preserved for stratified
reporting only -- **never fed to KalmanNet** (unchanged architectural
invariant from every prior AV2 task).

## 13. Manifest / hashes

`tools/av2_dataset_prep/frozen_configs/av2_kalmannet_scaleup_v2_freeze.json`
(new, committed, small, machine-independent):

| field | value |
|---|---|
| `train.content_fingerprint` | `ab0f9faa8318ecc13a9c0eea37c5414335540adca3b027194f5267da36baeb12` |
| `train.scenario_manifest_sha256` | `12ae390ea1b01e714b6e9d3b8d4a06822547b035d8aa8e5746ebdc92526c4c59` |
| `official_val.content_fingerprint` | `fc74a8985f5844a6797e46f1c3fa0f78063db12cf29a68cbb2ad6b0bcfa77c72` |
| `official_val.scenario_manifest_sha256` | `bebbfd3bc1caca11ece6af6f8c78c88b2fbbd08b221b18824590d584348a66d0` |
| `exclusion_manifests.stage0_scenarios.json` (sha256) | `7da66eb1050123a718dae510f5c3669a7568766b329b1dedb365b81360638858` |
| `exclusion_manifests.stage1_pilot_scenarios.json` (sha256) | `3b70d5ae598e7c7fdf260650ac66a6b8f781fff173c2766a77ee812858375ee1` |
| `exclusion_scenario_count` | 2,120 |
| `overlap_train_vs_official_val` | **0** |
| `overlap_train_vs_exclusion` | **0** |
| `overlap_val_vs_exclusion` | **0** |
| `git_sha` | `8cdcc5d0ce63f12c6e152c3f806c9665c9504a8a` |

The freeze-writer script (`build_scaleup_v2_freeze_manifest.py`) refuses
to write the file at all (exit 1) if any overlap is non-zero -- verified
by test (`test_freeze_manifest_refuses_when_train_val_overlap`,
`test_freeze_manifest_refuses_when_exclusion_overlap`), not just
asserted in prose.

Full fetch-tool scenario manifests (scenario_id + remote S3 key +
per-scenario sha256 + byte count, matching this project's existing
committed-manifest convention from Stage-0) are committed at
`ad_morai_bridge_dev/config/dataset_factory/av2_motion_forecasting_adapter/scaleup_v2_{train,official_val}_scenarios.manifest.json`.

## 14. Estimated training time

Derived from this project's own real measured Stage-1 pilot throughput
(the MORAI-Calibrated AV2 Corruption v1 task's real training run:
88,584 train sequences, batch_size=64, mean **520.3 s/epoch** over 22
real epochs, `train_time_s: 11445.72` / 22), scaled by the **real**
segment-count ratio between that 2,000-scenario pilot and this
10,000-scenario set: 553,810 / 88,584 = **6.252x**.

Per-epoch time is assumed to scale linearly with sequence count at fixed
`batch_size=64` (more batches per epoch, same per-batch cost) -- the
same assumption for GENERIC-ROBUST and MORAI-calibrated, since load-time
corruption is a cheap NumPy operation on the CPU side, not the GPU-bound
KalmanNetGRU forward/backward pass that dominates step time.

**Estimated per-epoch time at this scale: 520.3 s x 6.252 ~= 3,253 s
(~54.2 min/epoch).**

| condition | epochs (historical Stage-1 pilot pattern) | estimated 1-seed wall time | estimated 3-seed wall time |
|---|---|---|---|
| GENERIC-ROBUST | ~32 (historical: best@16, stopped@31) | 32 x 3,253 s ~= 104,096 s ~= **28.9 hours** | ~= **86.7 hours (3.6 days)** |
| MORAI-calibrated | ~22 (historical: best@6, stopped@21) | 22 x 3,253 s ~= 71,566 s ~= **19.9 hours** | ~= **59.7 hours (2.5 days)** |

**Explicit caveat**: the epoch counts above are carried over from the
2,000-scenario pilot's own observed early-stopping behavior, not
re-derived for this 10,000-scenario set -- convergence could plausibly
need more or fewer epochs at this larger, differently-composed scale.
The per-epoch time estimate itself IS grounded in a real measurement and
a real segment-count ratio; the total-wall-time figures additionally
assume the historical epoch count still applies, which is the weaker of
the two assumptions and is stated as such. No training was launched to
verify this in this task, per its own explicit instruction.

## 15. Safe future disk cleanup

Report only -- **nothing deleted**. Candidates that are entirely
superseded by this task's own frozen exclusion audit (never referenced
by future training once the Scale-Up v2 dataset is in use):

| path | size |
|---|---|
| `~/datasets/av2/raw_staging/` (Stage-0's 120-scenario raw parquet) | 20 MB |
| `~/datasets/av2/raw_staging_stage1/` (Stage-1 pilot's 2,000-scenario raw parquet) | 292 MB |
| `~/datasets/av2/processed/av2_kalmannet_v1_sharded/` (Stage-0 shards) | 44 MB |
| `~/datasets/av2/processed/av2_kalmannet_v1_stage1_pilot_sharded/` (Stage-1 pilot shards) | 647 MB |

~1 GB total. These remain useful for reproducing/comparing against the
Stage-0/Stage-1/MORAI-Calibrated-Corruption tasks' own already-published
results, so deletion is **not recommended without the user's explicit
decision** -- listed here only as the honest "what could eventually be
freed" answer this task's own Section 20 asks for.

## No training in this task

Per this task's own explicit instruction: no KalmanNet training (1-seed,
3-seed, GENERIC-ROBUST, or MORAI-calibrated) was launched. This task
ends at download -> shard -> validate -> freeze -> report. The next task
will choose ONE full-scale training matrix based on the measured cost
reported here.

## MORAI's future role

AV2 pretraining (Stage-0, Stage-1 pilot, this Scale-Up v2) is offline
sanity/pretraining evidence only. The competition-domain claim always
requires the frozen MORAI evaluation stream (`morai_frozen_stream.py`,
and eventually `morai_estimator_eval_v2.py` once new MORAI collection
becomes possible) -- AV2 TEST/VAL metrics are never substituted for a
MORAI competition-domain evaluation result, in this document or any
future one building on it.

## Files

- `tools/av2_dataset_prep/fetch_av2_scaleup_v2.py` (new -- generalizes the existing selective S3 fetch tool to a repeatable exclude-manifest union; also fixes a real local-path leak found while preparing the committed manifest below -- `excluded_manifests` now records basenames only, not `str(Path.expanduser())`)
- `tools/av2_dataset_prep/test_fetch_av2_scaleup_v2.py` (new, 8 tests)
- `tools/av2_dataset_prep/build_scaleup_v2_split_manifests.py` (new -- single-role TRAIN/VAL split manifest builder)
- `tools/av2_dataset_prep/test_build_scaleup_v2_split_manifests.py` (new, 6 tests)
- `tools/av2_dataset_prep/build_scaleup_v2_freeze_manifest.py` (new -- writes the frozen dataset manifest, refuses on any nonzero overlap)
- `tools/av2_dataset_prep/test_build_scaleup_v2_freeze_manifest.py` (new, 4 tests)
- `tools/av2_dataset_prep/frozen_configs/av2_kalmannet_scaleup_v2_freeze.json` (new, committed, small/machine-independent -- the real freeze manifest)
- `ad_morai_bridge_dev/config/dataset_factory/av2_motion_forecasting_adapter/scaleup_v2_{train,official_val}_scenarios.manifest.json` (new, committed -- full scenario-id + remote-key + sha256 manifests, matching the existing Stage-0 committed-manifest convention; scrubbed of the local-path leak above before commit)
- `docs/perception/av2_kalmannet_scaleup_v2.md` (this file)
- `docs/agent/STATUS.md` (updated)

Not committed (machine-local, per Section 23's own instruction): raw
`.parquet` files, processed NPZ shards, and the working-copy manifests
under `~/datasets/av2/manifests/` (their committed content-equivalent
lives in the two files above).

No `kalmannet_core.py`, CenterPoint, association, AB3DMOT, prediction,
planner, or occupancy-grid file touched. No model trained.
