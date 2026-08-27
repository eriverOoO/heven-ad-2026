# Final Presentation Pack Re-Audit

Audit date: 2026-08-22 (Asia/Seoul)

Scope: read-only re-audit of the corrected presentation source pack against
the frozen handoff, the previous Codex audit, the correction record, and the
underlying T-2B source cited by the newly generated presentation figure. No
experiment, replay, training, pack edit, commit, or push was performed. Only
this report and `FINAL_CODEX_REAUDIT_TABLE.csv` were created.

## 1. Files reviewed

Frozen handoff reviewed:

- `final_development_handoff/README.md`
- `experiment_timeline.csv`
- `final_results.csv`
- `claim_ledger.csv`
- `dataset_limitations.csv`
- `engineering_forensics.csv`
- `artifact_manifest.csv`
- `superseded_results.md`
- `final_system_architecture.md`

Presentation source pack reviewed recursively:

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
- `CODEX_AUDIT.md`
- `CODEX_AUDIT_TABLE.csv`
- `PRESENTATION_SOURCE_PACK_CORRECTIONS.md`
- `PRESENTATION_PACK_FINAL_CHECK.md`
- `generated_figures/make_t2b_forensic_figure.py`
- `generated_figures/t2b_forensic_audit_corrected.png` (visually inspected)

For the new T-2B figure, the cited canonical source was also checked directly:
`ground_segmentation/README.md` and
`ground_segmentation/t2b_detection_identity_check.md`.

Historical audit/correction records were treated as records: quoted old claims
inside those files are not live presentation recommendations. All other pack
files were treated as presentation-facing source.

## 2. Previous findings checked

All 18 requested items were checked explicitly. Fourteen are fully corrected,
one is substantively corrected but misses the requested per-slide literal
wording, and three retain or introduce material issues.

| # | Item | Result |
|---:|---|---|
| 1 | Candidate density | VERIFIED |
| 2 | T-15 association ranking | VERIFIED |
| 3 | T-10 churn comparison | VERIFIED |
| 4 | CenterPoint detection wording/caveat | VERIFIED |
| 5 | KalmanNet v2 position/velocity wording | VERIFIED |
| 6 | CenterPoint tracking metric scope | VERIFIED |
| 7 | GIoU safe claim | VERIFIED, with one broad glossary/checklist phrase noted below |
| 8 | Hungarian objective vs. accuracy | NOT FULLY CORRECTED |
| 9 | Temporal-jitter measurement vs. causality | VERIFIED |
| 10 | CenterPoint yaw measurement vs. interpretation | VERIFIED |
| 11 | Task and engineering counts | NOT FULLY CORRECTED |
| 12 | `NUMBER_CHECKLIST` precision labels | VERIFIED |
| 13 | Ground-segmentation figure | NOT FULLY CORRECTED |
| 14 | Gradient-clipping figures | VERIFIED |
| 15 | Offline/online equivalence scope and plotting floor | VERIFIED |
| 16 | CenterPoint main-slide overlap text | PARTIAL |
| 17 | T-13 appendix-only / no T-12 ranking use | VERIFIED |
| 18 | No new unsupported scientific claim | NOT VERIFIED |

## 3. Corrections verified

- Live candidate-density recommendations use the canonical same-association
  ratios `7.49x / 13.47x / 5.47x` or `5.47-13.47x`. Remaining `4-30x`
  occurrences outside historical records are explicit warnings not to use it.
- T-15 HOTA is correctly ordered Mahalanobis > Euclidean > GIoU for both
  detector rows. No live text claims a detector-caused Euclidean/Mahalanobis
  reversal.
- T-10 uses 592 Mahalanobis tracker IDs versus 701 Euclidean tracker IDs in
  the same 800-frame window. The T-6 values 505/725 are retained only as a
  separately labeled 400-frame, no-GT result or as a warning.
- CenterPoint detection text names recall, localization, and matched box
  geometry improvements while disclosing lower annotated-domain precision/F1
  and 100% train/evaluation overlap.
- Final KalmanNet wording says position near-tie, velocity improvement versus
  v1, and a slight velocity deficit versus tuned KF (`2.462 > 2.416 m/s`).
- CenterPoint tracking text names the primary HOTA/AssA/IDF1/IDSW degradation
  and acknowledges that Frag/IDR move differently.
- The precise T-15 GIoU result—lowest HOTA and highest IDSW in both detector
  rows—is present, with no live “worst on every identity metric” sentence.
- Temporal jitter preserves `0.288 m` versus `0.463 m` and labels its identity
  implication as consistent with a contributing factor, not isolated cause.
- Real-yaw degradation is reported separately from the explicitly
  interpretive yaw-convergence explanation.
- The task count is correctly stated as 24 completed tasks plus blocked T-11B
  (25 timeline rows) in the master summary.
- `NUMBER_CHECKLIST.csv` now separates rounded
  `verified_presentation_value` from `exact_canonical_value`.
- The old 40-frame ground-segmentation null plot is explicitly prohibited and
  the new 400-frame presentation-only graphic is recommended.
- The 5-seed gradient-clip-value sweep and separate 10/10 frozen-recipe
  validation are clearly separated in both slide outlines and the figure
  shortlist.
- Offline/online equivalence is stated as exactly `0.0`; the artificial
  log-axis plotting floor and the non-fresh-ROS-GT scope are disclosed.
- T-13 metrics are not used to validate or refute T-12 ranking. No slide or
  figure recommendation uses T-13's estimator ranking; the storyline only
  mentions its disclosed actor-composition confound.

## 4. Remaining numerical issues

1. **The new T-2B figure mis-scopes `-0.515 detections/frame`.** The figure
   itself, its generator at line 25, and `FIGURE_SHORTLIST.csv` row 3 say the
   mean applies “on changed frames.” The canonical T-2B table derives
   `-0.515` from mean OFF `11.97` minus mean ON `11.455` over the full
   400-frame streams. The figure should say “mean signed difference across all
   400 frames.” The 94/400 and 23.5% bars themselves are correct.

2. **The technical appendix still says 9 bugs.** `TECHNICAL_APPENDIX_OUTLINE.md`
   lines 123-128 says “9 real bugs,” while the frozen
   `engineering_forensics.csv` has 8 data rows and `ENGINEERING_STORY.md`
   correctly says 8 ledger entries summarized in 7 narrative groups.

3. **Three corrected CSVs are structurally malformed.** Strict row-width
   validation found:

   - `CLAIM_CHECKLIST.csv`: 3 data rows have 6 fields instead of 5 (rows 5,
     17, 28).
   - `NUMBER_CHECKLIST.csv`: 4 data rows have 9 fields instead of 8 (rows 19,
     32, 37, 46).
   - `FIGURE_SHORTLIST.csv`: 9 data rows have 7 or 8 fields instead of 6
     (rows 8, 10, 11, 17, 23, 24, 25, 27, 28).

   A CSV parser accepts these rows, but commas in unquoted fields shift values
   into the wrong columns. This contradicts the final self-check's implication
   that parsing alone established valid CSV structure.

## 5. Remaining wording issues

1. `CLAIM_CHECKLIST.csv` row 2 still says Hungarian “did not materially
   improve overall tracking behavior.” That is the exact overreach identified
   by the prior audit. T-3 measured raw affinity and descriptive
   population/lifecycle deltas, not GT tracking accuracy. The row's evidence
   column correctly acknowledges that no accuracy claim was measured, so the
   claim and its own support contradict each other.

2. `GLOSSARY.md` lines 53-55 and `CLAIM_CHECKLIST.csv` row 3 use broad
   “identity instability” language for GIoU. The pack elsewhere gives the safe
   precise claim: lowest HOTA and highest IDSW in both T-15 detector rows.
   Reusing that precise wording would eliminate any risk of being heard as
   “worst on every identity metric.”

3. `GLOSSARY.md` lines 107-110 says CenterPoint has “better absolute
   accuracy.” The presentation-facing wording elsewhere correctly says lower
   in-sample/train-scene localization error. The glossary should use that same
   scoped phrase.

## 6. Remaining figure-caption risks

- **Ground segmentation:** the newly generated figure is the correct 400-frame
  figure choice, but its visible footer incorrectly says `-0.515` is the mean
  on changed frames. This is a caption correction, not a need for a new
  experiment.
- **CenterPoint 20-minute slides:** `FIGURE_SHORTLIST.csv` has the required
  visible overlap sentence for every T-14/T-15 main figure. However,
  `SLIDE_OUTLINE_20MIN.md` slides 16-18 only say “Show overlap caveat,” rather
  than repeating the mandated literal text. This creates avoidable handoff
  risk if the outline is used without the shortlist.
- **Gradient clipping and offline/online equivalence:** their corrected figure
  guidance is safe and distinguishes the datasets, scope, and plotting floor.

## 7. Main-slide safety

The 15-minute outline is scientifically safe after the corrections and places
the exact overlap sentence on its CenterPoint slides. The 20-minute outline is
substantively consistent, but slides 16-18 should each explicitly require the
literal visible text **“100% train/evaluation overlap — single scene”** instead
of shorthand. The figure shortlist already supplies that text for each figure.

The remaining T-3 claim-checklist overreach does not appear as a recommended
main slide in either outline, but it remains unsafe source wording that a slide
author could reuse.

## 8. Appendix safety

T-13 remains appendix-only in scientific use and is not used to validate or
refute T-12 ranking. Its mention in the storyline is limited to the
actor-composition confound, and neither main outline recommends a T-13 metrics
slide.

The appendix is not fully safe as written because its engineering-ledger slide
still says 9 bugs instead of 8 ledger entries / 7 narrative groups. GIoU and
the T-2B mean-scope wording should also be kept precise if copied into backup
slides.

## 9. Final verdict

PASS WITH CORRECTIONS

The correction pass fixed most prior scientific issues, including all central
CenterPoint/KalmanNet rankings and caveats. It does not earn PASS because a
previously identified T-3 overclaim and the 9-bug count remain, the newly
created T-2B figure introduces a wrong averaging scope, the 20-minute outline
does not repeat the exact mandated overlap text on each forensic slide, and
three deliverable CSVs have inconsistent column counts. These are bounded
presentation-source corrections; no experiment or retraining is required.
