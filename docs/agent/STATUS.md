# STATUS

## Current task: Tracking reference setup (reference-only, no implementation)

Branch: `chore/tracking-references`. Added three pinned Git submodules
under `references/` for the Tracking research phase (per AGENTS.md
"External research repos must be pinned Git submodules under
`references/`"). No tracker was implemented, no HEVEN production code or
submodule source was modified, Ground segmentation was not started.

`git submodule status -- references/ab3dmot references/trackeval references/simpletrack`:

```
 61f3bd72574093e367916c757b4747ca445f978c references/ab3dmot (heads/master)
 05c96bb7ed98fc179856f327544612a66c839b5e references/simpletrack (heads/main)
 12c8791b303e0a0b50f753af204249e622d0281a references/trackeval (heads/master)
```

- `references/ab3dmot` — AB3DMOT (Weng et al., IROS/ECCVW 2020) 3D MOT
  baseline. **License caveat**: custom academic/non-profit noncommercial
  research-use-only license, not MIT — flagged for legal review before any
  derived code ships, not just read for reference.
- `references/trackeval` — TrackEval (Luiten et al., HOTA metric, IJCV
  2021), MIT. Standard MOT evaluation metrics reference.
- `references/simpletrack` — SimpleTrack (Pang, Li, Wang, arXiv:2111.09621,
  ECCVW 2022), MIT. Verified this is the paper's actual official repo
  (`tusen-ai/SimpleTrack`), not an unrelated same-named project.

Full provenance table (upstream repo, pinned SHA, paper, license, purpose)
in `references/README.md`.

Unrelated pre-existing note: `git submodule status` (without a pathspec)
prints an error for `src/MORAI-DriveExample_GRPC`, a pre-existing empty
placeholder directory unrelated to this task (not a real submodule, no
`.gitmodules` entry, not touched here) — use the pathspec-scoped command
above to check just the new submodules cleanly.

## Previous: CP-1 result

**CP-1: PASSED** (2026-08-18, merged to `main` in PR #2 /
commit `aa5cacb`). Full detail retained in `docs/research/centerpoint_status.md`.

## Remaining blocker

None for this reference-setup task. Separately and unchanged: CenterPoint
mAP/generalization evaluation is still blocked by dataset diversity (see
`docs/research/centerpoint_status.md`) — unrelated to Tracking references.
The AB3DMOT license caveat above is a new, separate item to resolve before
any AB3DMOT-derived code (not just reading it) ships.

## Exact recommended next task

Per this task's instruction: stop after reference setup — do not implement
a tracker, do not start Ground segmentation. Commit `.gitmodules`,
`references/README.md`, and the three pinned submodules on
`chore/tracking-references` and open a PR. The actual next research task
(per AGENTS.md order, once explicitly requested) is either Ground
segmentation or beginning Tracking design work informed by
`references/ab3dmot` and `references/simpletrack` — pending an explicit
go-ahead.
