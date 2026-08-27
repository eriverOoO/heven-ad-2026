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

SUPPORTED_MATCHERS = ("greedy", "hungarian")
SUPPORTED_METRICS = ("giou_3d", "euclidean", "mahalanobis")
# T-7A: "linear_kf" (default, unchanged AB3DMOT reference constant-velocity
# filter) or "ekf" (opt-in planar CTRV nonlinear model, see
# ab3dmot_core.py's EKFEstimator).
SUPPORTED_STATE_ESTIMATORS = ("linear_kf", "ekf", "imm")

# T-7B: 2x2 Markov model-transition self-persistence probabilities for the
# IMM(CV + CTRV) estimator. High self-transition is a defensible baseline
# (vehicle motion regimes normally persist across adjacent ~0.15s replay
# frames) -- NOT tuned against this task's own final tracking statistics.
# The off-diagonal (model-switch) probability is `1 - self-probability`
# for each row, so only these two free parameters are exposed (matches
# this task's "do not expose dozens of internal parameters" instruction).
DEFAULT_IMM_CV_TO_CV_PROBABILITY = 0.95
DEFAULT_IMM_CTRV_TO_CTRV_PROBABILITY = 0.95
# T-7A.5: "detector" (default, unchanged -- feed the detector's own yaw
# field into the estimator as a real measurement) or "unobserved" (opt-in
# -- treat yaw as unmeasured; see `heading_source_audit.md` for why the
# Euclidean detector's own yaw=0.0 is a structural placeholder, not a
# real observation, per the message contract's own
# `orientation_availability=UNAVAILABLE` flag).
SUPPORTED_YAW_MEASUREMENT_MODES = ("detector", "unobserved")

# T-4: BEV (x, y) Euclidean gate, meters. No numeric precedent exists
# elsewhere in this repo for a raw-distance gate (the Autoware audit in
# tracking_architecture.md documents a `max_dist_sq` *stage* but no
# concrete threshold value) -- chosen as a neutral, untuned baseline: half
# the T-1C/T-2 replay scene's typical vehicle-scale bounding box diagonal
# (~4-5m length boxes), wide enough to admit normal frame-to-frame motion
# at this replay's ~6-8 Hz rate without being unboundedly loose. Not tuned
# against tracking outcomes in this task, per its explicit scope.
DEFAULT_EUCLIDEAN_GATE_M = 3.0

# T-4: Mahalanobis chi-square gate. Reuses this repo's own already-audited
# Autoware precedent (tracking_architecture.md: "Mahalanobis distance gate
# ... empirical threshold 11.62, 99.6% confidence, chi-square 2 DOF") --
# 2 DOF because association uses the BEV (x, y) position subspace only
# (see ab3dmot_core.py's `_mahalanobis_distance_matrix`), matching both
# that Autoware precedent's own DOF choice and this task's BEV Euclidean
# metric, so the two distance-based metrics are computed over the
# identical measurement subspace.
DEFAULT_MAHALANOBIS_GATE = 11.62

# T-5B: optional absolute BEV physical-distance cap on Mahalanobis-accepted
# pairs (meters). <= 0 disables the cap entirely -- pure Mahalanobis
# behavior (T-4/T-5A, gated only by `mahalanobis_gate`) is unchanged and
# remains the default. When > 0, a pair must satisfy BOTH
# `d_M^2 <= mahalanobis_gate` AND `d_E <= mahalanobis_max_distance_m` to be
# valid -- the physical cap never changes the Hungarian cost value itself
# (still raw d_M^2), only which pairs are eligible. See T-5A's
# `mahalanobis_analysis/README.md` for the motivating failure mode.
DEFAULT_MAHALANOBIS_MAX_DISTANCE_M = 0.0


@dataclass(frozen=True)
class AB3DMOTConfig:
    association_metric: str = "giou_3d"
    matcher: str = "greedy"
    min_hits: int = 1
    max_age: int = 2
    giou_gate: float = 0.0
    euclidean_gate_m: float = DEFAULT_EUCLIDEAN_GATE_M
    mahalanobis_gate: float = DEFAULT_MAHALANOBIS_GATE
    mahalanobis_max_distance_m: float = DEFAULT_MAHALANOBIS_MAX_DISTANCE_M
    state_estimator: str = "linear_kf"
    yaw_measurement_mode: str = "detector"
    imm_cv_to_cv_probability: float = DEFAULT_IMM_CV_TO_CV_PROBABILITY
    imm_ctrv_to_ctrv_probability: float = DEFAULT_IMM_CTRV_TO_CTRV_PROBABILITY

    def __post_init__(self) -> None:
        if self.state_estimator not in SUPPORTED_STATE_ESTIMATORS:
            raise ValueError(
                f"unsupported state_estimator {self.state_estimator!r}; "
                f"only {SUPPORTED_STATE_ESTIMATORS} implemented"
            )
        if not (0.0 < self.imm_cv_to_cv_probability <= 1.0):
            raise ValueError("imm_cv_to_cv_probability must be in (0, 1]")
        if not (0.0 < self.imm_ctrv_to_ctrv_probability <= 1.0):
            raise ValueError("imm_ctrv_to_ctrv_probability must be in (0, 1]")
        if self.yaw_measurement_mode not in SUPPORTED_YAW_MEASUREMENT_MODES:
            raise ValueError(
                f"unsupported yaw_measurement_mode {self.yaw_measurement_mode!r}; "
                f"only {SUPPORTED_YAW_MEASUREMENT_MODES} implemented"
            )
        if self.association_metric not in SUPPORTED_METRICS:
            raise ValueError(
                f"unsupported association_metric {self.association_metric!r}; "
                f"only {SUPPORTED_METRICS} implemented"
            )
        if self.matcher not in SUPPORTED_MATCHERS:
            raise ValueError(
                f"unsupported matcher {self.matcher!r}; only {SUPPORTED_MATCHERS} implemented"
            )
        if self.min_hits < 1:
            raise ValueError("min_hits must be >= 1")
        if self.max_age < 1:
            raise ValueError("max_age must be >= 1")
        if not math.isfinite(self.giou_gate) or not -1.0 <= self.giou_gate <= 1.0:
            raise ValueError("giou_gate must be finite and within [-1, 1] (GIoU's own range)")
        if not math.isfinite(self.euclidean_gate_m) or self.euclidean_gate_m <= 0.0:
            raise ValueError("euclidean_gate_m must be finite and > 0")
        if not math.isfinite(self.mahalanobis_gate) or self.mahalanobis_gate <= 0.0:
            raise ValueError("mahalanobis_gate must be finite and > 0")
        if not math.isfinite(self.mahalanobis_max_distance_m):
            raise ValueError("mahalanobis_max_distance_m must be finite (<=0 disables the cap)")
