from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from ad_tuning.objective import RunMetrics


@runtime_checkable
class TrialRunner(Protocol):
    """Injected boundary for a future guarded ROS 2 tuning runner."""

    def run_trial(self, parameters: Mapping[str, float]) -> RunMetrics:
        """Reset, arm, and drive once, encoding every failure in RunMetrics."""
