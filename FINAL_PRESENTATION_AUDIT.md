# Final Presentation Audit

Audit date: 2026-08-22 (Asia/Seoul)

## 1. Scope

This was a read-only adversarial audit of the final corrected presentation
source pack, the frozen development handoff, the prior audit/correction
records, and the camera–LiDAR qualitative evidence. The audit checked only
whether the pack is safe for direct slide-design handoff; it did not repeat the
broad research review.

No experiment, replay, retraining, algorithm change, presentation-pack edit,
commit, or push was performed. Only this report and
`FINAL_PRESENTATION_AUDIT_TABLE.csv` were created.

## 2. Files reviewed

Frozen development handoff:

- `final_development_handoff/README.md`
- `experiment_timeline.csv`
- `final_results.csv`
- `claim_ledger.csv`
- `dataset_limitations.csv`
- `engineering_forensics.csv`
- `artifact_manifest.csv`
- `superseded_results.md`
- `final_system_architecture.md`

Presentation source pack:

- `PRESENTATION_MASTER_SUMMARY.md`
- `PRESENTATION_STORYLINE.md`
- `SLIDE_OUTLINE_15MIN.md`
- `SLIDE_OUTLINE_20MIN.md`
- `FIGURE_SHORTLIST.csv`
- `NUMBER_CHECKLIST.csv`
- `CLAIM_CHECKLIST.csv`
- `EXPECTED_QA.md`
- `TECHNICAL_APPENDIX_OUTLINE.md`
- `ONE_PAGE_CHEATSHEET.md`
- `ENGINEERING_STORY.md`
- `GLOSSARY.md`
- `CAMERA_LIDAR_PRESENTATION_GUIDE.md`
- `PRESENTATION_PACK_CAMERA_UPDATE.md`
- `CODEX_AUDIT.md`
- `CODEX_AUDIT_TABLE.csv`
- `PRESENTATION_SOURCE_PACK_CORRECTIONS.md`
- `PRESENTATION_PACK_FINAL_CHECK.md`
- `generated_figures/make_t2b_forensic_figure.py`
- `generated_figures/t2b_forensic_audit_corrected.png` (visually inspected)

Prior re-audit records:

- `FINAL_CODEX_REAUDIT.md`
- `FINAL_CODEX_REAUDIT_TABLE.csv`

Camera evidence:

- `camera_lidar_validation/README.md`
- `bag_relation_analysis.md`
- `presentation_claim_boundary.md`
- `extrinsic_audit.md`
- `sensor_sync_summary.json`
- `camera_stream_summary.csv`
- `bag_topic_inventory.csv`
- `qualitative_case_manifest.csv`
- `clip21_perception.json`
- `figures/projected_lidar_boxes_camera.png` (visually inspected)
- `figures/qualitative_tracking_failure.png` (visually inspected)
- `figures/camera_lidar_final_summary.png` (visually inspected)

Historical audit and correction files were interpreted as records. Old claims
quoted inside an explicit correction history were not treated as live
presentation claims.

## 3. Previous corrections verified

All material defects from the previous Codex re-audit are now corrected:

- The T-2B figure and shortlist correctly scope `-0.515 detections/frame`
  across all 400 frames, not only the 94 changed frames.
- The Hungarian claim now states only 256/256 raw-affinity optimality,
  descriptive `+0.7%` unique-track and `+0.3%` tracks/frame changes, and the
  fact that T-3 did not measure GT tracking accuracy.
- The engineering count is consistently 8 ledger entries summarized in 7
  narrative groups.
- The prior malformed rows in `CLAIM_CHECKLIST.csv`,
  `NUMBER_CHECKLIST.csv`, and `FIGURE_SHORTLIST.csv` are repaired.
- GIoU wording is narrowed to lowest HOTA and highest IDSW in both T-15
  detector rows.
- The glossary now says lower in-sample/train-scene localization error rather
  than “better absolute accuracy.”
- Every 20-minute T-14/T-15 CenterPoint recommendation now repeats the full
  visible caveat: **“100% train/evaluation overlap — single scene.”**

## 4. Numerical consistency

The mandatory development numbers are consistent across the live pack:

- Ground segmentation: 94/400 frames, 23.5%, one-scene descriptive result.
- Hungarian: raw affinity at least greedy on 256/256 comparable frames.
- T-10 churn: Mahalanobis 592 versus Euclidean 701 tracker IDs on the same
  800-frame window; T-6's 505/725 remains separately labeled.
- T-15 HOTA order in both detector rows: Mahalanobis > Euclidean > GIoU.
- T-9A.1 position: tuned KF `0.8553882304 m`, KalmanNet
  `0.8538249579 m`, essentially tied.
- T-12.3 position: tuned KF `1.4555029268 m`, v2 `1.4673881348 m`, near-tie.
- T-12.3 velocity: tuned KF `2.4163951114 m/s`, v2
  `2.4622469777 m/s`, v1 `2.6570819155 m/s`; v2 improves over v1 and is
  slightly behind tuned KF.
- Fragmented T12_FULL `3.0645076146 m`; dense variants
  `1.4760085311/1.4781393011 m`.
- Gradient clipping `max_norm=10.0`; final validation 10/10 stable seeds.
- Offline/ROS maximum estimator difference exactly `0.0`, over 3 sequences
  and 157 frames, including one 24-missing-frame sequence.
- Temporal jitter: Euclidean `0.2884040924 m`, CenterPoint
  `0.4631597973 m`.
- Candidate ratios: `7.49x / 13.47x / 5.47x`, summarized as
  `5.47-13.47x`.
- Project count: 24 completed tasks plus blocked T-11B, 25 timeline entries.

Camera timing and bag facts are supported for the front-camera pairing:
ROS2 MCAP, 360.17 seconds, four named cameras, raw LiDAR, median nearest
front-camera/LiDAR offset `19.989 ms` (~20.0 ms), p95 `39.139 ms`
(~39.1 ms), and 100% within 50 ms. The exact bag `/tf_static` extrinsic is
cross-validated against `sensor_mounts.yaml`; CameraInfo is absent and the
intrinsics are reconstructed under a pinhole/no-distortion assumption.

One camera count is not consistently labeled: the underlying manifest and
selected churn plot show **20 total cumulative tracker IDs**, consisting of
**18 short-lived IDs plus two 20/20-frame IDs**. `NUMBER_CHECKLIST.csv` row 51
and `CAMERA_LIDAR_PRESENTATION_GUIDE.md` line 110 instead say “18 distinct
IDs” without the short-lived qualifier, contradicting the plot that reaches
20. Most other pack references correctly say “18 short-lived IDs.”

## 5. Scientific-claim consistency

Mandatory checks A–Q pass for the development evidence:

- Ground segmentation is descriptive and carries no GT-accuracy claim.
- Hungarian objective optimality is separated from unmeasured T-3 GT
  tracking accuracy.
- Lower tracker churn is not equated with better GT identity.
- GIoU is scoped to lowest HOTA/highest IDSW, not every identity metric.
- No T-15 detector-dependent Euclidean/Mahalanobis rank reversal remains;
  T-10/T-15 protocol differences are acknowledged.
- KalmanNet fair-baseline and final-v2 claims use near-tie wording and the
  correct velocity comparator.
- Fragmented training is the dominant T-12-collapse explanation.
- The natural long-gap anecdote is not used as final evidence; controlled
  dropout contradicts intrinsic long-gap superiority.
- Training stability, gradient-clipping, and 5-seed-versus-10-seed figure
  scopes are correct.
- Offline/ROS equivalence remains an implementation-equivalence result, not
  fresh ROS GT validation; the plotting floor is disclosed.
- CenterPoint detection and tracking claims are in-sample, metric-specific,
  and caveated.
- Jitter and yaw-convergence language distinguish measurement from causal
  interpretation.
- T-15 remains explicitly multi-factor; no single factor is asserted as an
  isolated cause.

The main thesis remains supported: improving a local objective or module does
not automatically improve complete tracking. The Hungarian, Mahalanobis,
KalmanNet, CenterPoint, and T-15 evidence chains all preserve that distinction.

## 6. Superseded-result check

No superseded scientific conclusion has re-entered the live pack as final
truth. Old phrases—ground segmentation has no effect, KalmanNet clearly beats
a fair tuned KF, intrinsic long-gap superiority, lowest churn means best
identity, universal Euclidean association superiority, density-only causation,
and detector-dependent T-15 HOTA reversal—occur only as explicitly marked
historical/rejected statements or warnings.

The old 40-frame ground-segmentation null-result figure is explicitly
prohibited. The corrected presentation-only 400-frame T-2B figure is selected
and its visible numbers/caption are now correct.

## 7. CenterPoint caveat check

Every main-slide T-14/T-15 quantitative recommendation in both slide outlines
and `FIGURE_SHORTLIST.csv` requires visible text equivalent to the exact
mandated sentence **“100% train/evaluation overlap — single scene.”**

The pack distinguishes improved in-sample recall, localization, and matched
box geometry from lower annotated-domain precision/F1 and incomplete GT
coverage. It names the primary HOTA/AssA/IDF1/IDSW degradation and acknowledges
that Frag/IDR move differently. No live text claims CenterPoint generalization
or universal metric superiority.

One Q&A sentence is stale after camera integration: `EXPECTED_QA.md` lines
14-16 says no held-out MORAI scene has ever been captured. A different-sequence
MORAI camera bag now exists. It is not a valid CenterPoint held-out evaluation
set because it has no independent actor GT and was never run through
CenterPoint. The safe statement is that no **labeled, disjoint CenterPoint
evaluation dataset** exists—not that no other MORAI scene/capture exists.

## 8. KalmanNet fairness check

KalmanNet fairness is safe throughout the live pack:

- T-9A is explicitly historical and compared against an untuned KF.
- T-9A.1 is an essentially tied position result after fair calibration.
- Final v2 position remains a near-tie, not exact equality.
- Final v2 velocity improvement is only against v1; v2 remains slightly
  behind tuned KF.
- Fragmentation and gradient stability are presented as training-data/recipe
  findings, not proof of estimator superiority.
- The controlled-dropout result displaces the natural-gap anecdote.
- Exact offline/ROS equality is scoped to estimator mathematics rather than a
  new GT ranking.

## 9. Camera evidence check

The camera evidence is mostly integrated conservatively:

- The bag is consistently identified as `morai_cam4_20260813_163222`, a
  **DIFFERENT-SEQUENCE** moving-ego capture.
- No T-14/T-15 metric is attributed to an individual camera frame as if it
  came from the same run. Old values appear near camera text only inside
  explicit “do not attribute/validate” warnings.
- The pack repeatedly says the camera is qualitative, illustrative, not
  independent GT, and not a CenterPoint generalization test.
- CameraInfo absence and reconstructed pinhole/no-distortion intrinsics are
  disclosed; projection is not described as a calibrated accuracy benchmark.
- The two intended main figures and the appendix-only summary are classified
  correctly.

However, the live camera guide crosses its own boundary at lines 15-17 by
calling the imagery **“direct visual proof that the LiDAR pipeline's detections
correspond to real, recognizable objects.”** The master summary lines 93-95
and Expected Q&A lines 179-181 repeat the unqualified “detections correspond to
real objects” formulation. This is materially stronger than the approved safe
claim and is adjacent to the explicitly forbidden “camera proves this
detection is correct.” With reconstructed intrinsics and no camera annotation,
the supported wording is that projected LiDAR boxes are **visually consistent
with / appear aligned to visible simulated scene objects or structures**, for
qualitative inspection only.

The repeated phrase “real vehicles” / “real driving scene” is also misleading
for MORAI-rendered simulator imagery. “Visible simulated vehicles in a
separately captured MORAI scene” preserves the intended qualitative point
without suggesting real-world evidence.

## 10. Figure safety

- `generated_figures/t2b_forensic_audit_corrected.png`: safe; correct
  400-frame result and all-frame mean scope.
- `projected_lidar_boxes_camera.png`: appropriate as the early setup visual.
  Its title discloses reconstructed intrinsics, no distortion, no published
  CameraInfo, and exact bag TF. The slide caption must retain
  DIFFERENT-SEQUENCE and qualitative-only wording.
- `qualitative_tracking_failure.png`: appropriate as a descriptive churn
  visual only. The plot reaches 20 cumulative IDs; its slide caption must say
  18 are short-lived and two are 20/20-frame IDs, and must not equate the
  count with IDSW or GT identity switches. The PNG itself contains neither
  camera pixels nor an embedded DIFFERENT-SEQUENCE/not-GT caveat, so the
  mandated surrounding caption is essential.
- `camera_lidar_final_summary.png`: correctly kept appendix-only; direct visual
  inspection confirms it is too dense for a main slide.
- Gradient clipping and offline/ROS figures have the required dataset/scope
  distinctions and plotting-floor disclosure.

All camera and presentation-only figure paths referenced by the shortlist were
resolved successfully.

## 11. CSV / structural integrity

Strict CSV parsing and row-width validation passed:

- `CLAIM_CHECKLIST.csv`: 29 data rows, 5 columns, 0 malformed rows.
- `NUMBER_CHECKLIST.csv`: 50 data rows, 8 columns, 0 malformed rows.
- `FIGURE_SHORTLIST.csv`: 32 data rows, 6 columns, 0 malformed rows.
- `CODEX_AUDIT_TABLE.csv`: 76 data rows, 7 columns, 0 malformed rows.

Comma quoting is structurally consistent. Referenced camera figures and the
corrected T-2B figure exist at the listed paths. No duplicate live claim with
opposite scientific meaning was found.

The structural issue that remains is semantic rather than syntactic:
`NUMBER_CHECKLIST.csv` labels 18 as the total distinct-ID count even though the
canonical camera evidence and selected plot show 20 total, 18 of them
short-lived.

## 12. Remaining issues

Corrections required before direct slide-design handoff:

1. Remove “direct visual proof” / unqualified “detections correspond to real
   objects” from the camera guide, master summary, and Expected Q&A. Replace it
   with qualitative visual-consistency/alignment wording.
2. Correct the camera churn count wherever “18 distinct IDs” is used without
   qualification: canonical framing is **20 total IDs, of which 18 are
   short-lived**. Keep explicit language that this is not GT IDSW.
3. Replace “real vehicles/real driving scene” with “visible simulated vehicles
   / MORAI scene” in camera-facing slide guidance.
4. Update the stale Q&A sentence “No held-out MORAI scene has ever been
   captured” to “No labeled, disjoint CenterPoint evaluation dataset has been
   captured.”
5. When quoting the 20.0/39.1/100%-within-50-ms synchronization figures, keep
   the **front-camera nearest-frame join** scope visible.

These are presentation-source corrections only. They require no experiment,
retraining, algorithm change, or new evidence collection.

## 13. Final verdict

PASS WITH CORRECTIONS

The development evidence and all prior re-audit corrections are now safe. The
pack is not yet safe for direct slide-design handoff because the newly
integrated camera material contains one explicitly overstrong proof claim, an
18-versus-20 ID-label inconsistency, and stale/misleading scene wording. The
issues are bounded, but they cross the requested camera evidence boundary and
therefore require one final presentation-source correction pass.
