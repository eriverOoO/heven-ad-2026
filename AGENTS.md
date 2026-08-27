# AGENTS.md

Agent rules for this autonomous-driving competition repository.

## Project rules

- Reliability and real-time operation matter more than unnecessary complexity.
- Preserve the working HEVEN baseline.
- Default detector = Euclidean. Default tracker = Autoware.
- Existing IMM/prediction and occupancy remain unchanged unless explicitly requested.
- Experimental algorithms must be opt-in.
- Change one major experimental variable at a time.
- External research repos must be pinned Git submodules under `references/`.
  Do not copy entire external repos into HEVEN. Prefer adapters outside submodules.
- Read the original/reference implementation before implementing algorithms.
- Never invent unexplained KF states, Q/R, gating, association, or lifecycle logic.
- Preserve ROS2 topic/message/frame/timestamp/QoS contracts.
- Algorithm code and experiment config must remain separate.
  Parameter variations use configs, not branches.
- Execution success is not performance validation.
- Do not commit bags, checkpoints, large datasets, or generated experiment outputs.
- Neural training must fit on one RTX 4060.
- Avoid large sweeps. Start with small bounded smoke runs.
- Measure runtime/latency for competition-critical modules.
- Make small focused changes. Do not perform unrelated refactors.
- If coordinate convention, GT identity, license, or interface semantics are unclear,
  stop and document the issue instead of guessing.

## Research order

CenterPoint → Euclidean vs CenterPoint → Ground segmentation → Tracking →
Association → KF/EKF/IMM/KalmanNet → Prediction → Occupancy

## Token efficiency

- Do not rescan the whole repository if `docs/agent/STATUS.md` or a research
  document already contains verified context.
- Start with `git status`, `git diff`, and the relevant documentation.
- Inspect only files required for the current task.
- Do not paste long build logs into documentation; summarize the relevant
  error and command.
- Update `docs/agent/STATUS.md` before ending substantial work.
