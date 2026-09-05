"""Offline, per-frame instrumented traces for KalmanNet and LinearCVKF --
prior state (before measurement update), posterior state (after), learned
gain matrix (KalmanNet only), and regime classification (matched / gap
length / post-reacquisition), computed WITHOUT modifying either runtime
class (``ad_lidar_perception/ad_lidar_perception/kalmannet_core.py`` is
untouched).

The prior state is recovered by calling the exact same public
``KalmanNetFilter._f`` staticmethod externally, on the filter's own
current ``x_post`` (before advancing it), with the same ``dt`` that
``step()`` is about to use internally -- reproducing exactly what
``step()`` computes as its own first line, without needing to read a
private attribute or change the class.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

_AD_LIDAR_PKG_DIR = Path(__file__).resolve().parents[2] / "ad_lidar_perception"
if str(_AD_LIDAR_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_AD_LIDAR_PKG_DIR))
from ad_lidar_perception.kalmannet_core import KalmanNetFilter, KalmanNetGRU, LinearCVKF  # noqa: E402


@dataclass
class FrameTrace:
    frame: int
    measurement_present: bool
    gap_len_before: int          # length of the immediately-preceding missing-run (0 if none)
    frames_since_reacquisition: int | None  # 0 = this frame itself re-acquired; None if never gapped yet
    prior_pos_error: float
    posterior_pos_error: float
    prior_vel_error: float
    posterior_vel_error: float
    innovation_mag: float | None = None      # None on predict-only frames (no measurement to innovate against)
    gain_frobenius_norm: float | None = None  # KalmanNet only
    gain_pos_block_norm: float | None = None  # KalmanNet only -- rows 0:2 (position correction)
    gain_vel_block_norm: float | None = None  # KalmanNet only -- rows 2:4 (velocity correction)


def _pos_err(state, gt) -> float:
    return float(np.hypot(state[0] - gt[0], state[1] - gt[1]))


def _vel_err(state, gt) -> float:
    return float(np.hypot(state[2] - gt[2], state[3] - gt[3]))


def trace_knet_sequence(net: KalmanNetGRU, seq: dict[str, Any], device: str = "cpu") -> list[FrameTrace]:
    kf = KalmanNetFilter(net, device=device)
    first_meas = seq["z_meas"][0]
    x0 = torch.tensor([first_meas[0], first_meas[1], 0.0, 0.0], dtype=torch.float32, device=device)
    kf.init_sequence(x0.unsqueeze(0), batch_size=1)

    traces: list[FrameTrace] = []
    gap_len = 0
    frames_since_reacq: int | None = None
    with torch.no_grad():
        for i in range(len(seq["frames"])):
            dt = seq["dt"][i]
            z = seq["z_meas"][i]
            gt = seq["x_true"][i]
            if dt is None:
                continue  # frame 0 -- initialization only, no prior/posterior to compare

            x_post_before = kf.x_post.clone()
            x_prior_ext = KalmanNetFilter._f(x_post_before, dt)  # reproduces step()'s own first line externally
            prior_np = x_prior_ext.squeeze(0).cpu().numpy()
            gap_len_before = gap_len

            if z is not None:
                zt = torch.tensor(z, dtype=torch.float32, device=device).unsqueeze(0)
                y_pred = KalmanNetFilter._h(x_prior_ext)
                innovation = zt - y_pred
                x_post = kf.step(zt, dt)
                post_np = x_post.squeeze(0).cpu().numpy()
                gain = kf.last_K.squeeze(0)  # [m, n]
                gain_norm = float(gain.norm(2).item())
                gain_pos = float(gain[0:2, :].norm(2).item())
                gain_vel = float(gain[2:4, :].norm(2).item())
                innov_mag = float(innovation.squeeze(0).norm(2).item())

                if frames_since_reacq is None and gap_len_before == 0:
                    frames_since_reacq = None  # never gapped yet
                elif gap_len_before >= 1:
                    frames_since_reacq = 0
                elif frames_since_reacq is not None:
                    frames_since_reacq += 1
                gap_len = 0
            else:
                kf.x_post = KalmanNetFilter._f(kf.x_post, dt)
                post_np = kf.x_post.squeeze(0).cpu().numpy()
                gain_norm = gain_pos = gain_vel = None
                innov_mag = None
                gap_len += 1
                frames_since_reacq = None

            traces.append(FrameTrace(
                frame=i, measurement_present=(z is not None), gap_len_before=gap_len_before,
                frames_since_reacquisition=frames_since_reacq,
                prior_pos_error=_pos_err(prior_np, gt), posterior_pos_error=_pos_err(post_np, gt),
                prior_vel_error=_vel_err(prior_np, gt), posterior_vel_error=_vel_err(post_np, gt),
                innovation_mag=innov_mag, gain_frobenius_norm=gain_norm,
                gain_pos_block_norm=gain_pos, gain_vel_block_norm=gain_vel,
            ))
    return traces


def trace_kf_sequence(sigma_a: float, r_std: float, p0_scale: float, seq: dict[str, Any]) -> list[FrameTrace]:
    kf = LinearCVKF(sigma_a=sigma_a, r_std=r_std, p0_scale=p0_scale)
    first_meas = seq["z_meas"][0]
    kf.init_sequence(np.array([first_meas[0], first_meas[1], 0.0, 0.0]))

    traces: list[FrameTrace] = []
    gap_len = 0
    frames_since_reacq: int | None = None
    for i in range(len(seq["frames"])):
        dt = seq["dt"][i]
        z = seq["z_meas"][i]
        gt = seq["x_true"][i]
        if dt is None:
            continue

        prior = kf.predict(dt)
        gap_len_before = gap_len

        if z is not None:
            innovation = np.asarray(z) - prior[:2]
            post = kf.update(np.asarray(z))
            innov_mag = float(np.linalg.norm(innovation))
            if gap_len_before == 0 and frames_since_reacq is None:
                frames_since_reacq = None
            elif gap_len_before >= 1:
                frames_since_reacq = 0
            elif frames_since_reacq is not None:
                frames_since_reacq += 1
            gap_len = 0
        else:
            post = prior
            innov_mag = None
            gap_len += 1
            frames_since_reacq = None

        traces.append(FrameTrace(
            frame=i, measurement_present=(z is not None), gap_len_before=gap_len_before,
            frames_since_reacquisition=frames_since_reacq,
            prior_pos_error=_pos_err(prior, gt), posterior_pos_error=_pos_err(post, gt),
            prior_vel_error=_vel_err(prior, gt), posterior_vel_error=_vel_err(post, gt),
            innovation_mag=innov_mag,
        ))
    return traces


def classify_regime(t: FrameTrace) -> str:
    if not t.measurement_present:
        return "missing"
    if t.frames_since_reacquisition == 0:
        g = t.gap_len_before
        if g == 1:
            return "gap_1"
        if g <= 3:
            return "gap_2_3"
        if g <= 5:
            return "gap_4_5"
        return "gap_6_plus"
    if t.frames_since_reacquisition is not None and 1 <= t.frames_since_reacquisition <= 3:
        return "steady_state_after_reacquisition"
    return "matched"


def summarize_traces_by_regime(traces: list[FrameTrace]) -> dict[str, dict[str, float]]:
    from collections import defaultdict

    by_regime: dict[str, list[FrameTrace]] = defaultdict(list)
    for t in traces:
        by_regime[classify_regime(t)].append(t)

    out = {}
    for regime, ts in by_regime.items():
        prior_pos = np.array([t.prior_pos_error for t in ts])
        post_pos = np.array([t.posterior_pos_error for t in ts])
        post_vel = np.array([t.posterior_vel_error for t in ts])
        gains = [t.gain_frobenius_norm for t in ts if t.gain_frobenius_norm is not None]
        out[regime] = {
            "n": len(ts),
            "prior_pos_rmse": float(np.sqrt(np.mean(prior_pos ** 2))),
            "posterior_pos_rmse": float(np.sqrt(np.mean(post_pos ** 2))),
            "posterior_vel_rmse": float(np.sqrt(np.mean(post_vel ** 2))),
            "prior_to_posterior_improvement": float(
                np.sqrt(np.mean(prior_pos ** 2)) - np.sqrt(np.mean(post_pos ** 2))
            ),
            "mean_gain_frobenius_norm": (float(np.mean(gains)) if gains else None),
        }
    return out
