# External references

Pinned Git submodules for algorithms under research (AGENTS.md: "External
research repos must be pinned Git submodules under `references/`. Do not
copy entire external repos into HEVEN. Prefer adapters outside submodules.").

Reference only — nothing here is imported into or run by production HEVEN
code. Read before implementing; do not modify submodule source.

## references/ab3dmot

- **Upstream repository**: https://github.com/xinshuoweng/AB3DMOT.git
- **Pinned commit**: `61f3bd72574093e367916c757b4747ca445f978c` (`master`)
- **Associated paper**: Weng, Wang, Held, Kitani, "3D Multi-Object Tracking:
  A Baseline and New Evaluation Metrics" (IROS 2020 / ECCVW 2020)
- **License**: Custom — "SOFTWARE LICENSE AGREEMENT: ACADEMIC OR NON-PROFIT
  ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY" (see the submodule's own
  `LICENSE` file). **Not MIT/Apache** — GitHub reports no SPDX match.
  Any use beyond reading/reference for a research/competition context needs
  explicit legal review before code derived from it ships in HEVEN.
- **Purpose in HEVEN**: Baseline reference for 3D multi-object tracking
  (Kalman-filter state/association/lifecycle design and its standard
  evaluation protocol) for the upcoming Tracking research phase.

## references/trackeval

- **Upstream repository**: https://github.com/JonathonLuiten/TrackEval.git
- **Pinned commit**: `12c8791b303e0a0b50f753af204249e622d0281a` (`master`)
- **Associated paper**: Luiten et al., "HOTA: A Higher Order Metric for
  Evaluating Multi-Object Tracking" (IJCV 2021) — TrackEval is the paper's
  official evaluation codebase, also implementing CLEAR MOT, MOTA/MOTP, and
  other standard MOT metrics.
- **License**: MIT (confirmed from the submodule's `LICENSE` file).
- **Purpose in HEVEN**: Reference implementation of standardized
  multi-object-tracking evaluation metrics (HOTA and others) for scoring
  any tracker work produced during the Tracking research phase.

## references/simpletrack

- **Upstream repository**: https://github.com/tusen-ai/SimpleTrack.git
  (the paper's official repo — not to be confused with unrelated
  same-named "SimpleTrack" projects elsewhere on GitHub)
- **Pinned commit**: `05c96bb7ed98fc179856f327544612a66c839b5e` (`main`)
- **Associated paper**: Pang, Li, Wang, "SimpleTrack: Understanding and
  Rethinking 3D Multi-object Tracking" (arXiv:2111.09621; ECCV 2022
  workshops)
- **License**: MIT (confirmed from the submodule's `LICENSE` file).
- **Purpose in HEVEN**: Reference for a simpler, stronger 3D MOT
  design (its motion model, association, and life-cycle simplifications)
  to compare against the AB3DMOT baseline during the Tracking research
  phase.
