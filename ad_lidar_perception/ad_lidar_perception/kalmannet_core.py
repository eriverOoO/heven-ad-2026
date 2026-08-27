"""T-9A: experimental, ROS-independent KalmanNet prototype for a minimal
planar constant-velocity (CV) tracking problem.

State:       x = [x, y, vx, vy]^T   (m=4)
Measurement: z = [x, y]^T           (n=2)

This module is deliberately isolated from `ab3dmot_core.py` and from ROS --
no rclpy import, no AB3DMOT lifecycle/association coupling. It is not
wired into any launch file or node in this task (T-9A is offline-only,
per this task's explicit instruction; ROS integration is T-9B).

Architecture reference: `references/kalmannet` (KalmanNet_TSP, pinned
submodule, architecture #2 from Revach et al. 2022). This module
implements a deliberately simplified, single-GRU variant of that
architecture -- full audit and the reasons for the simplification are in
`~/heven_presentation_assets/kalmannet/kalmannet_architecture_audit.md`.
The structural property that matters is preserved: the analytical
state-evolution/measurement functions (`f`, `h`) are never learned; only
the correction (gain) applied to the innovation is learned, exactly as in
KalmanNet's own `x_post = x_prior + K_learned @ (y - h(x_prior))`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

STATE_DIM = 4  # [x, y, vx, vy]
MEAS_DIM = 2   # [x, y]


def F_matrix(dt: float) -> np.ndarray:
    """Constant-velocity state transition matrix for a real (possibly
    non-uniform) time step `dt`."""
    return np.array([
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


H_matrix = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
])


def Q_matrix(dt: float, sigma_a: float) -> np.ndarray:
    """White-noise-acceleration process noise (standard CV-model Q), a
    documented, untuned baseline -- not fit against test data, matching
    this repo's established convention (T-7A's EKF Q/R were likewise
    adapted, not fit)."""
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt
    q = sigma_a ** 2
    return q * np.array([
        [dt4 / 4, 0, dt3 / 2, 0],
        [0, dt4 / 4, 0, dt3 / 2],
        [dt3 / 2, 0, dt2, 0],
        [0, dt3 / 2, 0, dt2],
    ])


class LinearCVKF:
    """Classical Linear Kalman Filter for the [x,y,vx,vy] CV model. Fully
    standalone -- no dependency on AB3DMOTTracker/Track lifecycle or
    association. Supports real (non-uniform) dt."""

    def __init__(self, sigma_a: float = 1.0, r_std: float = 0.3, p0_scale: float = 10.0):
        self.sigma_a = sigma_a
        self.R = np.eye(MEAS_DIM) * (r_std ** 2)
        self.p0_scale = p0_scale
        self.x = None
        self.P = None

    def init_sequence(self, x0: np.ndarray):
        self.x = np.asarray(x0, dtype=float).reshape(STATE_DIM)
        self.P = np.eye(STATE_DIM) * self.p0_scale

    def predict(self, dt: float):
        Fm = F_matrix(dt)
        Qm = Q_matrix(dt, self.sigma_a)
        self.x = Fm @ self.x
        self.P = Fm @ self.P @ Fm.T + Qm
        return self.x.copy()

    def update(self, z: np.ndarray):
        z = np.asarray(z, dtype=float).reshape(MEAS_DIM)
        y = z - H_matrix @ self.x
        S = H_matrix @ self.P @ H_matrix.T + self.R
        K = self.P @ H_matrix.T @ np.linalg.inv(S)
        self.K = K  # exposed for interpretability/gain-analysis (Phase 16), not used internally elsewhere
        self.x = self.x + K @ y
        self.P = (np.eye(STATE_DIM) - K @ H_matrix) @ self.P
        return self.x.copy()


@dataclass
class Sequence:
    """One continuous, GT-identity-verified track segment."""
    actor_id: int
    frames: list
    dt: list          # dt[0] is None (no previous step)
    x_true: list       # [[x,y,vx,vy], ...]
    z_meas: list        # [[x,y], ...]

    def __len__(self):
        return len(self.frames)


class KalmanNetGRU(nn.Module):
    """Simplified single-GRU KalmanNet (see module docstring / audit doc
    for the deliberate simplification vs. the reference's 3-GRU
    architecture #2). Learns a Kalman gain matrix K [m,n] from normalized
    state/measurement residual features; the CV `f`/`h` are analytical and
    fixed, never learned."""

    def __init__(self, hidden_size: int = 32, m: int = STATE_DIM, n: int = MEAS_DIM):
        super().__init__()
        self.m = m
        self.n = n
        self.hidden_size = hidden_size
        in_dim = n + n + m + m  # obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff
        self.input_fc = nn.Sequential(nn.Linear(in_dim, hidden_size), nn.ReLU())
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.output_fc = nn.Linear(hidden_size, m * n)
        self.h = None  # recurrent hidden state, shape [1, batch, hidden_size]

    def reset_hidden(self, batch_size: int = 1, device=None):
        self.h = torch.zeros(1, batch_size, self.hidden_size, device=device)

    @staticmethod
    def _norm(v: torch.Tensor) -> torch.Tensor:
        return F.normalize(v, p=2, dim=-1, eps=1e-8)

    def forward(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff):
        """All inputs: [batch, dim] (unnormalized raw differences). Returns
        K_learned: [batch, m, n]."""
        batch = obs_diff.shape[0]
        feats = torch.cat([
            self._norm(obs_diff), self._norm(obs_innov_diff),
            self._norm(fw_evol_diff), self._norm(fw_update_diff),
        ], dim=-1)  # [batch, in_dim]
        x = self.input_fc(feats).unsqueeze(0)  # [1, batch, hidden]
        if self.h is None or self.h.shape[1] != batch:
            self.reset_hidden(batch, device=feats.device)
        out, self.h = self.gru(x, self.h)
        gain = self.output_fc(out.squeeze(0))  # [batch, m*n]
        return gain.view(batch, self.m, self.n)


class KalmanNetFilter:
    """Wraps KalmanNetGRU with the explicit analytical f/h recursion,
    mirroring the reference's `KNet_step` structure: predict (analytical)
    -> estimate gain (learned) -> apply gain (analytical correction)."""

    def __init__(self, net: KalmanNetGRU, device="cpu"):
        self.net = net
        self.device = device
        self.x_post = None
        self.x_post_prev = None
        self.x_prior_prev = None
        self.y_prev = None

    def init_sequence(self, x0: torch.Tensor, batch_size: int = 1):
        self.net.reset_hidden(batch_size, device=self.device)
        self.x_post = x0.clone()
        self.x_post_prev = x0.clone()
        self.x_prior_prev = x0.clone()
        self.y_prev = self._h(x0)

    @staticmethod
    def _f(x: torch.Tensor, dt: float) -> torch.Tensor:
        Fm = torch.tensor(F_matrix(dt), dtype=x.dtype, device=x.device)
        return (Fm @ x.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def _h(x: torch.Tensor) -> torch.Tensor:
        Hm = torch.tensor(H_matrix, dtype=x.dtype, device=x.device)
        return (Hm @ x.unsqueeze(-1)).squeeze(-1)

    def step(self, z: torch.Tensor, dt: float) -> torch.Tensor:
        x_prior = self._f(self.x_post, dt)
        y_pred = self._h(x_prior)

        obs_diff = z - self.y_prev
        obs_innov_diff = z - y_pred
        fw_evol_diff = self.x_post - self.x_post_prev
        fw_update_diff = self.x_post - self.x_prior_prev

        K = self.net(obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff)
        self.last_K = K  # exposed for interpretability/gain-analysis (Phase 16)
        innovation = z - y_pred
        correction = torch.bmm(K, innovation.unsqueeze(-1)).squeeze(-1)
        x_post = x_prior + correction

        self.x_post_prev = self.x_post
        self.x_prior_prev = x_prior
        self.x_post = x_post
        self.y_prev = z
        return x_post


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
