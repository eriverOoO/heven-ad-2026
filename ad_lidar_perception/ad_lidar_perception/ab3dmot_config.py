"""Configurable parameters for the AB3DMOT-compatible tracking core (T-1A).

Values are the initial baseline already resolved in
`docs/research/tracking_architecture.md` ("AB3DMOT Integration Decisions"
§3) — the most common value across AB3DMOT's own per-class/per-dataset
configs, not tuned for HEVEN. Kept as a plain config object (not
hardcoded constants inside the algorithm) per AGENTS.md "Algorithm code
and experiment config must remain separate."

Mirrors `ad_lidar_perception/config/tracking/ab3dmot.yaml`, which records
the same values as the future ROS-parameter surface for T-1B; this
dataclass is the T-1A-only, ROS-independent equivalent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

SUPPORTED_MATCHERS = ("greedy",)
SUPPORTED_METRICS = ("giou_3d",)


@dataclass(frozen=True)
class AB3DMOTConfig:
    association_metric: str = "giou_3d"
    matcher: str = "greedy"
    min_hits: int = 1
    max_age: int = 2
    giou_gate: float = 0.0

    def __post_init__(self) -> None:
        if self.association_metric not in SUPPORTED_METRICS:
            raise ValueError(
                f"unsupported association_metric {self.association_metric!r}; "
                f"only {SUPPORTED_METRICS} implemented in T-1A"
            )
        if self.matcher not in SUPPORTED_MATCHERS:
            raise ValueError(
                f"unsupported matcher {self.matcher!r}; only {SUPPORTED_MATCHERS} "
                "implemented in T-1A (Hungarian/other matchers are a documented "
                "future option, not implemented yet per this task's scope)"
            )
        if self.min_hits < 1:
            raise ValueError("min_hits must be >= 1")
        if self.max_age < 1:
            raise ValueError("max_age must be >= 1")
        if not math.isfinite(self.giou_gate) or not -1.0 <= self.giou_gate <= 1.0:
            raise ValueError("giou_gate must be finite and within [-1, 1] (GIoU's own range)")
