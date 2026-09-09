"""Deterministic AV2 motion-category classification and weighted-sampling
support for the Motion-Composition Ablation v1 experiment
(docs/perception/kalmannet_av2_motion_composition_v1.md).

Pure functions only, no torch/ROS dependency -- operates on the plain
``{"x_true", "dt", ...}`` sequence dicts ``kalmannet_sequences.load_split_sequences``
already produces (GT ``x_true`` is corruption-independent, so motion
classification is computed once on CLEAN sequences and reused for every
corruption condition -- classification never depends on which noise
realization a training run happens to use).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

NEAR_STATIONARY = "near_stationary"
MOVING = "moving"

# Chosen, documented definition (section 2 of the task spec) -- median
# speed is the most outlier-robust of the three candidates audited
# (median/mean/p90; see docs, "motion category definition"). Locked
# BEFORE any official-VAL read.
REPRESENTATIVE_SPEED_KEY = "median_speed_mps"
NEAR_STATIONARY_THRESHOLD_MPS = 0.5

# Speed bins for descriptive reporting within MOVING (section 3/15).
SPEED_BIN_EDGES_MPS = (0.0, 0.5, 2.0, 5.0, 10.0, float("inf"))
SPEED_BIN_LABELS = ("0-0.5", "0.5-2", "2-5", "5-10", "10+")


def compute_segment_motion_stats(seq: dict[str, Any]) -> dict[str, float]:
    """Per-segment motion summary from the segment's own GT ``x_true``
    ([T,4] = x,y,vx,vy). Never reads ``z_meas`` (corrupted/possibly
    missing) -- motion composition is a property of the ground truth,
    not of any particular corruption draw."""
    x_true = np.asarray(seq["x_true"], dtype=float)
    if x_true.ndim != 2 or x_true.shape[1] != 4 or x_true.shape[0] < 2:
        raise ValueError(f"expected x_true shape [T>=2, 4], got {x_true.shape}")
    vx, vy = x_true[:, 2], x_true[:, 3]
    speed = np.hypot(vx, vy)
    pos = x_true[:, :2]
    displacement_m = float(np.linalg.norm(pos[-1] - pos[0]))

    dt_raw = seq["dt"]
    dt_arr = np.array([d for d in dt_raw[1:] if d is not None], dtype=float)
    dvel = np.diff(np.stack([vx, vy], axis=1), axis=0)
    dvel_mag = np.linalg.norm(dvel, axis=1)
    n = min(len(dvel_mag), len(dt_arr))
    if n > 0:
        dt_safe = np.maximum(dt_arr[:n], 1e-6)
        accel_proxy_mps2 = float(np.mean(dvel_mag[:n] / dt_safe))
    else:
        accel_proxy_mps2 = 0.0

    return {
        "median_speed_mps": float(np.median(speed)),
        "mean_speed_mps": float(np.mean(speed)),
        "p90_speed_mps": float(np.percentile(speed, 90)),
        "max_speed_mps": float(np.max(speed)),
        "displacement_m": displacement_m,
        "stationary_frame_fraction": float(np.mean(speed < NEAR_STATIONARY_THRESHOLD_MPS)),
        "acceleration_proxy_mps2": accel_proxy_mps2,
        "n_frames": int(len(speed)),
    }


def classify_motion(
    stats: dict[str, float],
    key: str = REPRESENTATIVE_SPEED_KEY,
    threshold_mps: float = NEAR_STATIONARY_THRESHOLD_MPS,
) -> str:
    """Deterministic: same stats + same (key, threshold) always give the
    same category. ``key`` is intentionally a parameter (not hardcoded)
    so the definition-sensitivity audit (median vs. mean vs. p90) can
    reuse this exact function for all three candidates."""
    return MOVING if stats[key] >= threshold_mps else NEAR_STATIONARY


def speed_bin(speed_mps: float) -> str:
    for i in range(len(SPEED_BIN_EDGES_MPS) - 1):
        if SPEED_BIN_EDGES_MPS[i] <= speed_mps < SPEED_BIN_EDGES_MPS[i + 1]:
            return SPEED_BIN_LABELS[i]
    return SPEED_BIN_LABELS[-1]


def definition_agreement(
    all_stats: list[dict[str, float]],
    threshold_mps: float = NEAR_STATIONARY_THRESHOLD_MPS,
) -> dict[str, Any]:
    """Sensitivity audit (section 2): how much does classification change
    across median/mean/p90 speed as the representative-speed definition?
    Computed once, before any definition is locked in."""
    keys = ("median_speed_mps", "mean_speed_mps", "p90_speed_mps")
    labels = {k: [classify_motion(s, key=k, threshold_mps=threshold_mps) for s in all_stats] for k in keys}
    n = len(all_stats)
    out: dict[str, Any] = {"n_segments": n, "moving_fraction_by_key": {k: sum(1 for x in labels[k] if x == MOVING) / n for k in keys}}
    pairwise = {}
    for i, ka in enumerate(keys):
        for kb in keys[i + 1 :]:
            agree = sum(1 for a, b in zip(labels[ka], labels[kb]) if a == b) / n
            pairwise[f"{ka}_vs_{kb}"] = agree
    out["pairwise_agreement_fraction"] = pairwise
    return out


# ---------------------------------------------------------------------------
# Deterministic motion-stratified sampling (sections 4/5/10/11) -- a pure
# weight-assignment function; the actual draw-with-replacement loop lives
# in the trainer (kept there so this module stays torch-free and unit
# testable without a training loop).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MotionSamplerConfig:
    """``target_moving_fraction``: the desired fraction of MOVING draws
    per epoch under reweighting. ``NATURAL`` uses the segment's own
    natural frequency (no reweighting, weight=1 for every segment)."""

    name: str
    target_moving_fraction: float | None  # None => NATURAL (no reweighting)


NATURAL = MotionSamplerConfig(name="NATURAL", target_moving_fraction=None)
BALANCED_MOTION = MotionSamplerConfig(name="BALANCED-MOTION", target_moving_fraction=0.50)
MOVING_FOCUSED = MotionSamplerConfig(name="MOVING-FOCUSED", target_moving_fraction=0.775)


def segment_key(seq: dict[str, Any]) -> str:
    """Stable per-segment identifier for deterministic, order-independent
    corruption/weighting (mirrors the existing ``segment_id``-derived
    corruption-seed convention -- reuses scenario_id+track_id, never row
    order)."""
    return f"{seq['scenario_id']}::{seq['track_id']}"


def compute_sample_weights(
    categories: list[str],
    config: MotionSamplerConfig,
) -> np.ndarray:
    """Returns one non-negative weight per segment (same order as
    ``categories``), summing to ``len(categories)`` (so the *expected*
    number of draws per epoch under weighted-with-replacement sampling
    matches the NATURAL baseline's epoch size -- section 11). NATURAL
    returns all-ones (uniform, byte-identical to unweighted sampling)."""
    n = len(categories)
    if n == 0:
        raise ValueError("no categories to weight")
    if config.target_moving_fraction is None:
        return np.ones(n, dtype=float)

    n_moving = sum(1 for c in categories if c == MOVING)
    n_stationary = n - n_moving
    if n_moving == 0 or n_stationary == 0:
        raise ValueError(
            f"cannot reweight to target_moving_fraction={config.target_moving_fraction} "
            f"with n_moving={n_moving} n_stationary={n_stationary} (one class is empty)"
        )
    target_moving = config.target_moving_fraction
    target_stationary = 1.0 - target_moving
    # per-item weight so that sum(weights in class) / n == target fraction
    # for that class, and sum(all weights) == n (expected draws per epoch
    # unchanged, section 11).
    w_moving = target_moving * n / n_moving
    w_stationary = target_stationary * n / n_stationary
    return np.array([w_moving if c == MOVING else w_stationary for c in categories], dtype=float)


def deterministic_weighted_draw_order(
    weights: np.ndarray,
    n_draws: int,
    seed: int,
) -> np.ndarray:
    """Deterministic weighted draw-with-replacement of segment INDICES,
    same seed -> same exact ordering (section 10: "same seed -> same
    epoch ordering"). Uses ``numpy.random.default_rng(seed)`` (a fresh,
    isolated generator; never the global RNG) so this is independent of
    any other seeding elsewhere in the training loop."""
    if n_draws <= 0:
        raise ValueError("n_draws must be positive")
    probs = weights / weights.sum()
    rng = np.random.default_rng(seed)
    return rng.choice(len(weights), size=n_draws, replace=True, p=probs)


def weight_manifest_fingerprint(categories: list[str], config: MotionSamplerConfig) -> str:
    """SHA-256 over (sorted categories, config) -- recorded in the
    checkpoint manifest per section 10 ("sequence sampling weights
    recorded in checkpoint manifest") without embedding the full
    per-segment weight array (which would be as large as the dataset
    itself) -- a fingerprint is sufficient to prove which weighting
    policy produced a given checkpoint."""
    h = hashlib.sha256()
    h.update(config.name.encode())
    h.update(str(config.target_moving_fraction).encode())
    h.update(str(len(categories)).encode())
    h.update(str(sum(1 for c in categories if c == MOVING)).encode())
    return h.hexdigest()
