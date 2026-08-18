# STATUS

## Current task: AB3DMOT integration decisions (design only, no implementation)

Branch: `feat/ab3dmot-tracker`. Resolved the five open AB3DMOT-integration
questions flagged in `docs/research/tracking_architecture.md` §7, adding
"## AB3DMOT Integration Decisions" to that file. All five items are
**RESOLVED** (none BLOCKED):

1. **Yaw convention** — confirmed from source (`AB3DMOT_libs/box.py`,
   `kitti_oxts.py`, `nuScenes_utils.py`) that AB3DMOT's `theta`/`ry` is a
   KITTI camera-frame angle (rotation about the camera's down-pointing Y
   axis), not HEVEN's z-up `lidar_link`/`odom` yaw — verified numerically
   by executing AB3DMOT's own nuScenes-adapter quaternion formula (hand-
   rolled quaternion math, cross-checked against the real `pyquaternion`
   library in a throwaway venv; outputs matched exactly). Resolution: reuse
   AB3DMOT's `filterpy` KF/matching core as a frame-agnostic periodic
   scalar (feed it HEVEN's own yaw directly, no conversion), but do **not**
   reuse its camera-frame corner/IoU geometry functions — those must be
   reimplemented in HEVEN's z-up convention.
2. **Tracking frame** — recommend transforming detections to `odom` before
   tracking (mirrors Autoware's own approach), justified by ego-motion
   correctness (a body-fixed `lidar_link` frame violates the KF's
   constant-velocity assumption under ego motion), fair comparison with
   Autoware, `AutowarePredictionNode`'s hardcoded `odom`-only input gate,
   and simpler RViz overlay.
3. **Initial AB3DMOT configuration** — 3D GIoU association metric, greedy
   matching (not Hungarian) uniformly, `min_hits=1`, `max_age=2` frames —
   each the most common value across AB3DMOT's own per-class/per-dataset
   configs, not tuned for HEVEN. The GIoU gating threshold has no single
   defensible reference default (AB3DMOT's own values span -0.2 to -0.8,
   visibly dataset-tuned) — left configurable, default `0.0`, explicitly
   flagged as needing real-data calibration. `max_age`'s frame-vs-time
   uncertainty is flagged against HEVEN's measured LiDAR cadence (≈7.34 Hz
   from `bags/static_20260805_003151/metadata.yaml`).
4. **TrackedObjects conversion** — full field-by-field mapping table
   (position, orientation, dimensions, track ID, pose/twist covariance).
   Position/orientation/dimensions/pose-covariance/yaw-covariance map
   directly from the KF state and its own `P` matrix (real filter
   quantities, nothing invented); untracked covariance dimensions are left
   at safe defaults, not fabricated. One real correction found and
   documented: AB3DMOT's raw KF velocity state is in **meters per frame**,
   not m/s (its `F` matrix uses a literal `dt=1` coefficient) — publishing
   it directly into `twist.linear` would be numerically wrong; the fix
   (scale by real measured inter-detection `dt`, and scale the velocity
   covariance block by `dt²`) is already precedented by SimpleTrack's own
   fork of the same KF (already documented in this file's §4).
5. **ROS interface** — `input: /ad/perception/objects/detected` (unchanged
   contract), `output: /experiment/tracked/ab3dmot` (new,
   `/experiment/`-namespaced, frame `odom`). Production Autoware tracker
   topic/launch/config are untouched; both can run simultaneously off the
   same detections.

No code was written. `references/ab3dmot` was read only, not modified.

**Pre-existing STATUS.md merge conflict resolved while updating this file**:
this file previously contained unresolved `<<<<<<< HEAD` / `=======` /
`>>>>>>> research/tracking-audit` markers from an earlier branch
merge/rebase (reference-setup vs. tracking-audit content). Both sides'
substance is preserved below; this branch (`feat/ab3dmot-tracker`) already
carries the `references/` submodule commit (`4d26e0e`) cleanly, so the
prior "branch/submodule discrepancy" note from the tracking-audit side no
longer applies here.

## Previous: CP-1 result

**CP-1: PASSED** (2026-08-18, merged to `main` in PR #2 / commit
`aa5cacb`). Full detail retained in `docs/research/centerpoint_status.md`.

## Previous: Tracking architecture audit

Produced `docs/research/tracking_architecture.md` §§1-7: HEVEN's tracking
I/O contract, the real Autoware `multi_object_tracker` implementation
(association/state-estimation/lifecycle/class-specific behavior),
`references/ab3dmot` and `references/simpletrack` decompositions, where
KF/EKF/IMM/KalmanNet could later be compared, and the tracking-vs-
prediction distinction. Superseded/extended by this task's §"AB3DMOT
Integration Decisions" above.

## Remaining blocker

None for this design task. Two items are explicitly left open by design
(not blockers, per this task's own instructions to leave uncertain values
configurable rather than guess them):
1. The GIoU gating threshold's exact numeric default (§3) needs real-data
   calibration before it can be trusted — not before implementation starts,
   since it's just a config default.
2. `max_age`'s frame-count semantics should be re-verified against each
   detector backend's actual live publish rate once the adapter runs.

Separately and unchanged: CenterPoint mAP/generalization evaluation is
still blocked by dataset diversity (see `docs/research/centerpoint_status.md`)
— unrelated to Tracking.

## Exact recommended next task

Per this task's instruction: stop after these decisions — no code was
written. The next task, once explicitly requested, is implementing the
smallest AB3DMOT adapter per `docs/research/tracking_architecture.md` §7
and this session's "AB3DMOT Integration Decisions": a new opt-in ROS2 node
(e.g. `ad_ab3dmot_tracker`) that imports `AB3DMOT_libs.kalman_filter.KF`
and `AB3DMOT_libs.matching` from the pinned submodule (no forking its code
into HEVEN), subscribes `/ad/perception/objects/detected`, transforms into
`odom`, runs the resolved greedy/GIoU/min_hits=1/max_age=2 baseline with a
real-dt-corrected velocity, and publishes
`autoware_perception_msgs/msg/TrackedObjects` on
`/experiment/tracked/ab3dmot` — leaving the production Autoware tracker
path completely untouched. Do not start that implementation until
explicitly requested.
