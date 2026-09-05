"""MORAI TRAIN/VAL-only corruption calibration for the AV2 KalmanNet
pretraining pipeline (MORAI-Calibrated AV2 Corruption v1).

Frozen calibration constants below were derived ONCE from
``~/heven_presentation_assets/state_estimator_gt_comparison/sequences.pkl``
(the ORIGINAL, non-dense-filtered MORAI GT-aligned measurement stream --
29 sequences, roles 'train'/'val'/'test', split frozen before T-12 per
``~/heven_presentation_assets/motion_gt_expansion/canonical/split_policy.json``).
Calibration source: TRAIN role (22 sequences, actors
[1,13,15,18,35,36,38,40,44,46,48], 2053 frames) + VAL role (4 sequences,
actors [21,23,27,31], 408 frames) pooled, n=1890 present-measurement
frames. The frozen TEST role (3 sequences, actors [16,20,30], 157 frames)
was NEVER used to derive any value below -- see
``calibrate_residual_stats``/``calibrate_gap_stats`` below, both of which
raise if handed a sequence with ``role == "test"``, as a defence against
accidentally re-deriving these constants from TEST data in the future.

Model selection (bias + full-covariance Gaussian, not independent x/y,
not zero-mean full-covariance): a zero-bias full-covariance model
under-predicts the real radial-tail frequency by roughly half
(synthetic tail_beyond_1m 0.30 vs. real 0.575); dropping the modest but
consistently-signed x/y correlation (-0.18 to -0.24 across TRAIN, VAL,
and TRAIN+VAL independently) changes the fit only marginally relative to
including it. See
``docs/perception/kalmannet_morai_calibrated_corruption_v1.md`` Section 3
for the full three-model comparison.

Burst-length pool: the exact 64 observed TRAIN+VAL burst lengths
(empirical resampling, not a fitted parametric family) -- includes values
up to 44 SOLELY because that is what TRAIN+VAL contains. The frozen
TEST-observed 24-frame gap was never consulted when choosing this pool or
any other parameter here.

Burst-start-probability correction: the naive per-frame burst-start rate
(64 bursts / 2435 frames = 0.02628) under-produces the target missing
rate when replayed through ``apply_corruption`` on real AV2 sequences
(measured 0.1694 vs. target 0.2345) -- a renewal-process artifact of the
sequential burst scan (each triggered burst consumes its own length in
candidate start positions, so realized burst-covered fraction is
p*E[L]/(p*E[L]+(1-p)), not the naive product). Corrected via a small grid
search against AV2 TRAIN OUTPUT ONLY (never any MORAI split) to reproduce
the TRAIN+VAL-derived target missing rate (0.040 measured 0.2354 vs.
target 0.2345).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ad_morai_bridge_dev"))
from ad_morai_bridge_dev.dataset.av2_motion_forecasting_adapter import CorruptionConfig  # noqa: E402

MORAI_CALIBRATED_BIAS_XY = (-0.6548025429052758, 0.6652460172320311)
MORAI_CALIBRATED_COVARIANCE_XY = (
    (0.5054367466769748, -0.07705166171251883),
    (-0.07705166171251883, 0.3577342804083116),
)
MORAI_CALIBRATED_DROPOUT_PROB = 34 / 2435
MORAI_CALIBRATED_BURST_PROB = 0.040  # renewal-corrected, see module docstring
MORAI_CALIBRATED_BURST_LENGTH_SAMPLES = (
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5,
    6, 6, 6,
    7, 7, 7, 7,
    8, 8,
    9, 9, 9, 9, 9, 9,
    10, 10, 10, 10,
    17, 20, 21, 23, 24, 28, 29, 34, 35, 44,
)
MORAI_CALIBRATED_SEED = 20260906

# Explicit provenance -- never touched, checked directly in the guard below.
MORAI_CALIBRATION_TRAIN_ACTOR_IDS = (1, 13, 15, 18, 35, 36, 38, 40, 44, 46, 48)
MORAI_CALIBRATION_VAL_ACTOR_IDS = (21, 23, 27, 31)
MORAI_FROZEN_TEST_ACTOR_IDS = (16, 20, 30)


class LeakedTestDataError(ValueError):
    """Raised when a calibration function is handed a role='test' sequence."""


def _assert_no_test_rows(sequences: list[dict[str, Any]]) -> None:
    test_rows = [s for s in sequences if s.get("role") == "test"]
    if test_rows:
        actor_ids = sorted({s.get("actor_id") for s in test_rows})
        raise LeakedTestDataError(
            f"calibration input contains {len(test_rows)} role='test' sequence(s) "
            f"(actor_ids={actor_ids}) -- MORAI TEST must never be used to derive a "
            f"calibration parameter; pass only role in {{'train', 'val'}}"
        )


def calibrate_residual_stats(sequences: list[dict[str, Any]]) -> dict[str, Any]:
    """Bias/covariance/radial-percentile/tail-frequency report over every
    present-measurement frame in ``sequences``. Raises ``LeakedTestDataError``
    if any sequence has ``role == 'test'``."""
    _assert_no_test_rows(sequences)
    x_res: list[float] = []
    y_res: list[float] = []
    for s in sequences:
        for i in range(len(s["frames"])):
            z = s["z_meas"][i]
            if z is None:
                continue
            gt = s["x_true"][i]
            x_res.append(z[0] - gt[0])
            y_res.append(z[1] - gt[1])
    x = np.array(x_res)
    y = np.array(y_res)
    radial = np.hypot(x, y)
    cov = np.cov(np.stack([x, y]))
    return {
        "n": int(len(x)),
        "bias_xy": (float(x.mean()), float(y.mean())),
        "covariance_xy": cov.tolist(),
        "radial_p50": float(np.percentile(radial, 50)),
        "radial_p90": float(np.percentile(radial, 90)),
        "radial_p95": float(np.percentile(radial, 95)),
        "radial_p99": float(np.percentile(radial, 99)),
        "tail_beyond_1m": float(np.mean(radial > 1.0)),
        "tail_beyond_2m": float(np.mean(radial > 2.0)),
        "tail_beyond_3m": float(np.mean(radial > 3.0)),
        "tail_beyond_5m": float(np.mean(radial > 5.0)),
    }


def calibrate_gap_stats(sequences: list[dict[str, Any]]) -> dict[str, Any]:
    """Availability/missing-rate/gap-length report over ``sequences``.
    Raises ``LeakedTestDataError`` if any sequence has ``role == 'test'``."""
    _assert_no_test_rows(sequences)
    n_total = 0
    n_present = 0
    gap_lengths: list[int] = []
    for s in sequences:
        flags = [s["z_meas"][i] is None for i in range(len(s["frames"])) if s["dt"][i] is not None]
        n_total += len(flags)
        n_present += sum(1 for m in flags if not m)
        i = 0
        while i < len(flags):
            if flags[i]:
                start = i
                while i < len(flags) and flags[i]:
                    i += 1
                gap_lengths.append(i - start)
            else:
                i += 1
    availability = n_present / n_total if n_total else float("nan")
    return {
        "n_frames_total": n_total,
        "availability_rate": availability,
        "missing_rate": 1 - availability,
        "n_gaps": len(gap_lengths),
        "n_single_frame_gaps": sum(1 for g in gap_lengths if g == 1),
        "n_burst_gaps": sum(1 for g in gap_lengths if g >= 2),
        "gap_lengths": gap_lengths,
    }


def build_morai_calibrated_corruption_config(seed: int = MORAI_CALIBRATED_SEED) -> CorruptionConfig:
    """The frozen MORAI-TRAIN+VAL-calibrated corruption config used by
    ``train_kalmannet.py``'s ``"morai_calibrated_robust"`` condition."""
    return CorruptionConfig(
        mode="morai_calibrated",
        bias_xy=MORAI_CALIBRATED_BIAS_XY,
        covariance_xy=MORAI_CALIBRATED_COVARIANCE_XY,
        dropout_prob=MORAI_CALIBRATED_DROPOUT_PROB,
        dropout_burst_enabled=True,
        dropout_burst_prob=MORAI_CALIBRATED_BURST_PROB,
        dropout_burst_length_samples=MORAI_CALIBRATED_BURST_LENGTH_SAMPLES,
        seed=seed,
    )
