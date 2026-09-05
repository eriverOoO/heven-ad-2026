"""Pure, offline diagnostic functions for the AV2->MORAI domain-gap
analysis. No training, no architecture change, no runtime dependency --
every function here takes plain sequence dicts (the existing
``{"actor_id","frames","dt","x_true","z_meas"}`` contract) and/or a
``KalmanNetGRU``/``LinearCVKF`` instance, and returns plain-dict/dataclass
statistics. Reused directly by the analysis scripts that produced this
task's own report; kept separate from any training/eval entry point so
it stays trivially unit-testable with synthetic fixtures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


def _percentiles(values: list[float], ps=(50, 90, 95, 99)) -> dict[str, float]:
    if not values:
        return {f"p{p}": float("nan") for p in ps}
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"p{p}": float("nan") for p in ps}
    return {f"p{p}": float(np.percentile(arr, p)) for p in ps}


def _run_lengths(flags: list[bool]) -> list[int]:
    """Lengths of contiguous True-runs in a boolean sequence -- used both
    for gap-length distributions (``flags`` = "measurement missing") and
    for any other run-length need."""
    runs = []
    cur = 0
    for f in flags:
        if f:
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)
    return runs


@dataclass
class SequenceDistributionStats:
    n_sequences: int
    n_frames_total: int
    seq_length: dict[str, float]
    dt: dict[str, float]
    speed_mps: dict[str, float]
    accel_mps2: dict[str, float]
    velocity_change_mps: dict[str, float]
    turn_rate_radps: dict[str, float]
    displacement_per_step_m: dict[str, float]
    stationary_fraction: float
    measurement_available_fraction: float
    missing_measurement_fraction: float
    gap_length_frames: dict[str, float]
    n_gaps: int
    coordinate_range_m: dict[str, float]
    stationary_speed_threshold_mps: float = 0.5


def sequence_distribution_stats(
    sequences: list[dict[str, Any]], stationary_speed_threshold_mps: float = 0.5
) -> SequenceDistributionStats:
    """Motion/measurement distribution summary over a list of sequence
    dicts. Every "per-transition" quantity (accel, velocity-change,
    turn-rate, displacement) is computed only where ``dt is not None``
    (i.e. never at a sequence's own frame 0), matching this project's own
    established loss-counting convention."""
    seq_lengths, dts, speeds, accels, vel_changes, turn_rates, displacements = [], [], [], [], [], [], []
    n_stationary_samples = 0
    n_speed_samples = 0
    n_meas_present = 0
    n_meas_total = 0
    all_missing_flags: list[bool] = []
    xs, ys = [], []

    for seq in sequences:
        seq_lengths.append(len(seq["frames"]))
        prev_v = None
        for i in range(len(seq["frames"])):
            x, y, vx, vy = seq["x_true"][i]
            xs.append(x)
            ys.append(y)
            speed = math.hypot(vx, vy)
            n_speed_samples += 1
            if speed < stationary_speed_threshold_mps:
                n_stationary_samples += 1

            dt = seq["dt"][i]
            z = seq["z_meas"][i]
            n_meas_total += 1
            missing = z is None
            all_missing_flags.append(missing)
            if not missing:
                n_meas_present += 1

            if dt is not None:
                dts.append(dt)
                speeds.append(speed)
                if prev_v is not None:
                    dvx, dvy = vx - prev_v[0], vy - prev_v[1]
                    vel_change = math.hypot(dvx, dvy)
                    vel_changes.append(vel_change)
                    accels.append(vel_change / dt if dt > 0 else float("nan"))
                    heading = math.atan2(vy, vx)
                    prev_heading = math.atan2(prev_v[1], prev_v[0])
                    dtheta = math.atan2(math.sin(heading - prev_heading), math.cos(heading - prev_heading))
                    turn_rates.append(dtheta / dt if dt > 0 else float("nan"))
                displacements.append(speed * dt)
            prev_v = (vx, vy)

    gaps = _run_lengths(all_missing_flags)

    return SequenceDistributionStats(
        n_sequences=len(sequences),
        n_frames_total=sum(seq_lengths),
        seq_length=_percentiles([float(v) for v in seq_lengths]),
        dt=_percentiles(dts),
        speed_mps=_percentiles(speeds),
        accel_mps2=_percentiles(accels),
        velocity_change_mps=_percentiles(vel_changes),
        turn_rate_radps=_percentiles([abs(t) for t in turn_rates]),
        displacement_per_step_m=_percentiles(displacements),
        stationary_fraction=(n_stationary_samples / n_speed_samples if n_speed_samples else float("nan")),
        measurement_available_fraction=(n_meas_present / n_meas_total if n_meas_total else float("nan")),
        missing_measurement_fraction=(1.0 - n_meas_present / n_meas_total if n_meas_total else float("nan")),
        gap_length_frames=_percentiles([float(g) for g in gaps]),
        n_gaps=len(gaps),
        coordinate_range_m={
            "max_abs_x": float(np.max(np.abs(xs))) if xs else float("nan"),
            "max_abs_y": float(np.max(np.abs(ys))) if ys else float("nan"),
        },
        stationary_speed_threshold_mps=stationary_speed_threshold_mps,
    )


@dataclass
class ResidualStats:
    n: int
    x_residual: dict[str, float]
    y_residual: dict[str, float]
    radial_error: dict[str, float]
    x_bias: float
    y_bias: float
    x_std: float
    y_std: float
    xy_correlation: float


def measurement_residual_stats(sequences: list[dict[str, Any]]) -> ResidualStats:
    """``z_meas - GT_position[:2]`` over every frame with a present
    measurement. Radial error = ``hypot(x_residual, y_residual)``."""
    x_res, y_res = [], []
    for seq in sequences:
        for i in range(len(seq["frames"])):
            z = seq["z_meas"][i]
            if z is None:
                continue
            gt = seq["x_true"][i]
            x_res.append(z[0] - gt[0])
            y_res.append(z[1] - gt[1])

    if not x_res:
        nan_p = _percentiles([])
        return ResidualStats(0, nan_p, nan_p, nan_p, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))

    x_arr, y_arr = np.asarray(x_res), np.asarray(y_res)
    radial = np.hypot(x_arr, y_arr)
    corr = float(np.corrcoef(x_arr, y_arr)[0, 1]) if len(x_arr) > 1 else float("nan")
    return ResidualStats(
        n=len(x_res),
        x_residual=_percentiles(list(np.abs(x_arr))),
        y_residual=_percentiles(list(np.abs(y_arr))),
        radial_error=_percentiles(list(radial)),
        x_bias=float(np.mean(x_arr)), y_bias=float(np.mean(y_arr)),
        x_std=float(np.std(x_arr)), y_std=float(np.std(y_arr)),
        xy_correlation=corr,
    )


REGIME_NAMES = ("matched", "gap_1", "gap_2_3", "gap_4_5", "gap_6_plus", "steady_state_after_reacquisition")


def _regime_for_frame(measurement_present: bool, gap_len_before: int, frames_since_reacquisition: int | None) -> str:
    """``gap_len_before``: length of the immediately-preceding missing-run
    (0 if this frame itself is measurement-present and was not preceded by
    a gap). ``frames_since_reacquisition``: frames since the measurement
    stream resumed after a gap (0 = the very first re-acquired frame),
    ``None`` if not applicable."""
    if not measurement_present:
        return "missing"  # handled separately by the caller's own gap-length bucketing
    if frames_since_reacquisition == 0 and gap_len_before >= 1:
        if gap_len_before == 1:
            return "gap_1"
        if gap_len_before <= 3:
            return "gap_2_3"
        if gap_len_before <= 5:
            return "gap_4_5"
        return "gap_6_plus"
    if frames_since_reacquisition is not None and 1 <= frames_since_reacquisition <= 3 and gap_len_before >= 1:
        return "steady_state_after_reacquisition"
    return "matched"


@dataclass
class BootstrapResult:
    observed_diff: float
    ci_low: float
    ci_high: float
    fraction_bootstrap_same_sign: float
    n_bootstrap: int
    stable: bool


def per_sequence_bootstrap_rmse_diff(
    per_sequence_rmse_a: list[float], per_sequence_rmse_b: list[float], n_bootstrap: int = 2000, seed: int = 0
) -> BootstrapResult:
    """Per-SEQUENCE (not per-frame) bootstrap of the mean RMSE difference
    ``a - b`` -- resampling at the sequence level acknowledges within-
    sequence temporal dependence (frames inside one sequence are not
    independent), the same discipline this task's own instruction asks
    for. Requires paired per-sequence RMSE values (same sequences, same
    order) for both estimators."""
    if len(per_sequence_rmse_a) != len(per_sequence_rmse_b):
        raise ValueError("per_sequence_rmse_a and per_sequence_rmse_b must have the same length (paired)")
    n = len(per_sequence_rmse_a)
    a = np.asarray(per_sequence_rmse_a, dtype=float)
    b = np.asarray(per_sequence_rmse_b, dtype=float)
    observed = float(np.mean(a) - np.mean(b))

    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        diffs.append(float(np.mean(a[idx]) - np.mean(b[idx])))
    diffs = np.asarray(diffs)
    ci_low, ci_high = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    same_sign = float(np.mean(np.sign(diffs) == np.sign(observed))) if observed != 0 else float("nan")
    stable = bool((ci_low > 0) or (ci_high < 0))  # CI excludes zero
    return BootstrapResult(
        observed_diff=observed, ci_low=ci_low, ci_high=ci_high,
        fraction_bootstrap_same_sign=same_sign, n_bootstrap=n_bootstrap, stable=stable,
    )


@dataclass
class CorruptionExposureStats:
    n_frames_total: int
    clean_fraction: float
    dropout_single_fraction: float
    dropout_burst_fraction: float
    post_gap_reacquisition_fraction: float


def corruption_exposure_stats(sequences: list[dict[str, Any]]) -> CorruptionExposureStats:
    """Classifies every non-frame-0 timestep of a (already load-time-
    corrupted) sequence list into: clean-measurement, single-frame
    dropout, burst dropout (run length >=2), or the first frame
    immediately after any such gap (post-gap reacquisition) -- purely
    from the resulting ``z_meas`` pattern, no corruption-config
    introspection needed (works identically for AV2 or MORAI data)."""
    n_total = 0
    n_clean = 0
    n_single = 0
    n_burst = 0
    n_post_gap = 0
    for seq in sequences:
        missing_flags = [seq["z_meas"][i] is None for i in range(len(seq["frames"])) if seq["dt"][i] is not None]
        n_total += len(missing_flags)
        i = 0
        while i < len(missing_flags):
            if missing_flags[i]:
                run_start = i
                while i < len(missing_flags) and missing_flags[i]:
                    i += 1
                run_len = i - run_start
                if run_len == 1:
                    n_single += 1
                else:
                    n_burst += run_len
                if i < len(missing_flags):
                    n_post_gap += 1
            else:
                n_clean += 1
                i += 1
    return CorruptionExposureStats(
        n_frames_total=n_total,
        clean_fraction=(n_clean / n_total if n_total else float("nan")),
        dropout_single_fraction=(n_single / n_total if n_total else float("nan")),
        dropout_burst_fraction=(n_burst / n_total if n_total else float("nan")),
        post_gap_reacquisition_fraction=(n_post_gap / n_total if n_total else float("nan")),
    )
