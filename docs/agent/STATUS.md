# STATUS

## Current task: Tracking architecture audit (read-only, no implementation)

Branch: `research/tracking-audit`. Produced
`docs/research/tracking_architecture.md`: full DetectedObjects → Autoware
`multi_object_tracker` → TrackedObjects → HEVEN IMM/prediction flow,
documenting HEVEN's I/O contract (topics/messages/frames/timestamp
handling/QoS), the Autoware tracker's real association (BEV + MuSSP
min-cost-flow solver, not Hungarian)/state estimation (EKF: bicycle model
for vehicles, CTRV for pedestrians)/lifecycle (existence-probability +
adaptive-covariance thresholds, not hit/miss counters)/class-specific
behavior, and a decomposition of both `references/ab3dmot` (linear 10-state
KF, per-class Hungarian/greedy + IoU/GIoU/Mahalanobis, orientation
correction, hits/max_age lifecycle) and `references/simpletrack`
(coarse-hash+NMS preprocessing, configurable association incl. Mahalanobis,
the same borrowed AB3DMOT KF with a time-varying `dt`, an explicit
birth/alive/dead FSM, and a "redundancy module" second-chance association
against low-score detections). Also documents where KF/EKF/IMM/KalmanNet
could later be compared, explicitly separates tracking state estimation
from HEVEN's own downstream IMM future-trajectory prediction, and proposes
(design only) the smallest AB3DMOT adapter shape with its open questions
(yaw-convention verification, frame/TF ownership, per-class parameter
choice, output message shape) flagged rather than guessed.

No tracker was implemented. No production code, submodule source, or
`docs/agent/STATUS.md`-adjacent research docs from prior tasks were
modified beyond this file and the new audit doc.

**Branch/submodule discrepancy found and documented (not fixed here)**:
this branch does not contain the `chore: add tracking research references`
commit (`4d26e0e`, on `chore/tracking-references` /
`origin/chore/tracking-references`), so `.gitmodules` and
`references/README.md` are absent here even though the three reference
checkouts physically remain on disk (leftover from an earlier branch
checkout in this same working tree) at the exact pinned SHAs
`references/README.md` records. Read directly from disk for this audit;
recommend merging/rebasing `chore/tracking-references` into whatever branch
carries future Tracking work so `references/README.md` and `.gitmodules`
are consistently present.

## Previous: CP-1 result

**CP-1: PASSED** (2026-08-18, merged to `main` in PR #2 / commit
`aa5cacb`). Full detail retained in `docs/research/centerpoint_status.md`.

## Remaining blocker

None for this audit task itself. Two separate, pre-existing items remain
open (unrelated to each other):
1. The branch/submodule discrepancy above (`references/` commit not yet on
   this branch).
2. CenterPoint mAP/generalization evaluation still blocked by dataset
   diversity (unchanged, see `docs/research/centerpoint_status.md`).

Additionally, per this audit: the `PedestrianAndBicycleTracker`'s exact
model hand-off logic between its internal pedestrian/bicycle sub-trackers,
and `TrackerOverlapManager::merge`'s exact pruning logic, were not traced
in this pass (flagged, not guessed) — read
`lib/tracker/model/pedestrian_and_bicycle_tracker.cpp` and
`lib/association/tracker_overlap_manager.cpp` if that detail becomes
load-bearing for future work.

## Exact recommended next task

Per this task's instruction: stop after the audit. Per AGENTS.md's research
order (Tracking → Association → KF/EKF/IMM/KalmanNet...), the next task —
once explicitly requested — is most naturally either (a) resolving the
branch/submodule discrepancy above, or (b) verifying the AB3DMOT yaw/theta
convention against `AB3DMOT_libs/box.py` (the first open question flagged
in `docs/research/tracking_architecture.md` §7) before any adapter
implementation begins. Do not start implementing the AB3DMOT adapter until
that verification and an explicit go-ahead.
