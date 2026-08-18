# STATUS

## CP-1 result

**CP-1: NOT PASSED.** Verified against 8 gate criteria in
`docs/research/centerpoint_status.md` ("## CP-1 Milestone"), 2026-08-18. 6/8
PASS (offline inference, ROS2 inference mechanism, DetectedObjects
validity, Euclidean/CenterPoint same-run comparison, latency recording,
dataset/performance-limitation documentation). ROS2/RViz/inference
themselves are confirmed working, not broken. 2/8 FAIL, both the same root
cause: no checkpoint beyond the initial capped smoke test (2–5 iterations,
self-labeled `dry_run_only`) has ever been trained, so (1) no non-smoke
checkpoint with provenance exists, and (5) the boxes that do reach RViz are
visible but not geometrically plausible (2,500 re-inspected detections were
100% class `obstacle`, near-constant ~0.95 m dimensions, scores confined to
a 0.1006–0.1010 noise band). This gap is *not* blocked by dataset
diversity — it's achievable now on the existing single-scene split — unlike
the separately-documented mAP/generalization blocker, which *is*
diversity-blocked and does not by itself fail CP-1 per this check's
instructions.

## Remaining blocker

No new-data blocker for CP-1 itself. The blocker is simply that a
non-smoke (still bounded, still single-RTX-4060, still no sweep) training
run has not yet been performed on the existing 1,764-sample single-scene
train split. Separately and independently, real mAP/generalization
evaluation remains blocked by dataset diversity (`val`/`test` splits are
empty, only one scene exists) — see "Exact data requirements for a
meaningful trained checkpoint" in `docs/research/centerpoint_status.md`.

## Exact recommended next task

Run one bounded, non-smoke training pass on the current single-scene
`train` split with `tools/centerpoint_offline/train_morai_centerpoint.py`
(no new hyperparameter search — pick one reasonable, documented
epoch/iteration count larger than the prior 2–5-iteration smoke, e.g. a
full pass or two over the 1,764 samples, batch size 1, same seed
convention), record its `peak_vram_mib` and provenance the same way the
smoke runs already do, then re-run `infer_morai_centerpoint.py` on that
checkpoint and re-inspect detection class/dimension/score diversity (repeat
the same box-plausibility check done for this CP-1 pass) before re-checking
criteria #1 and #5. Do not start Tracking until CP-1 passes.
