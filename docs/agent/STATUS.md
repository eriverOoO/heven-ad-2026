# STATUS

## CP-1 result

**CP-1: PASSED.** (2026-08-18, re-check after training one bounded
non-smoke checkpoint.) All 8 gate criteria in
`docs/research/centerpoint_status.md` ("## CP-1 Milestone" +
"CP-1 re-check 2026-08-18") now pass. The previous 2/8 FAIL (#1 no
non-smoke checkpoint, #5 boxes not geometrically plausible) is resolved:
trained 3 full epochs (5,292 iterations) over the existing 1,764-sample
single-scene `train` split, unchanged architecture/config, batch size 1,
no hyperparameter search, peak VRAM 373.9 MiB, loss 57.6→3.08 (no
NaN/Inf). The resulting checkpoint's predictions are 100% class `vehicle`
with mean dimensions (L 4.43 W 2.03 H 1.81 m) within ~15% of this scene's
real GT vehicle mean (L 5.12 W 2.16 H 1.70 m, sampled 200 label files),
replacing the prior smoke checkpoint's uniform near-threshold noise grid
(100% `obstacle`, near-constant ~0.95–1 m boxes, scores in a 0.0001-wide
band). Re-verified live end-to-end: ROS2 node loaded the checkpoint via
strict load and ran real CUDA inference (16–17 objects/frame,
`model_forward` 35–122 ms post-warm-up); the reused `ad_viz` visualizer
republished 34 markers on the exact topic `heven_perception.rviz`
displays.

**No accuracy, generalization, or Euclidean-superiority claim is made.**
This checkpoint was trained and evaluated on the same single repeated
scene, so plausible boxes are expected from memorization, not proven
generalization.

## Remaining blocker

None for CP-1 itself. Separately and unchanged: proper mAP/generalization
performance evaluation remains blocked purely by dataset diversity
(`train`/`val`/`test` all draw from one scene; `val`/`test` are empty) —
see "Exact data requirements for a meaningful trained checkpoint" in
`docs/research/centerpoint_status.md`. This blocker is independent of CP-1
and was not addressed in this task (no new data was introduced).

## Exact recommended next task

CP-1 has passed; do not start Tracking yet per this task's instructions.
Next, per AGENTS.md's research order (CenterPoint → Euclidean vs
CenterPoint → Ground segmentation → Tracking...), and since the Euclidean
vs CenterPoint comparison tooling already exists
(`docs/perception/centerpoint_vs_euclidean_comparison.md`): either (a)
begin Ground segmentation work, or (b) if new diverse MORAI scenes become
available, populate `val`/`test` splits and revisit real performance
evaluation before moving on. Tracking work should wait until an explicit
go-ahead, per this task's instruction not to start it now.
