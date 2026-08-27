"""POST-FREEZE research extension, Phase 7: opt-in, EXPERIMENTAL
calibrated-covariance head for KalmanNet. Does not modify
`kalmannet_core.py`. Reuses its analytical `F_matrix`/`H_matrix`
unchanged and loads a frozen `KalmanNetGRU` checkpoint read-only (its own
weights are never updated by this module's training) -- only a small new
head is trained, on top of the frozen state estimator's already-fixed
gain/state output.

Motivation (from this task's own instruction): KalmanNet's existing
`velocity_covariance`/state covariance fields (T-9B's own "Option B"
policy) are populated from an internally-composed classical side
estimator only for message-schema completeness -- never claimed as
KalmanNet's own uncertainty, and never validated as calibrated. This
module is the first attempt in the project to actually learn and
calibrate a KalmanNet-native uncertainty estimate, strictly as a
diagnostic/opt-in research path. It does NOT change
`kalmannet_core.KalmanNetFilter`'s own covariance fields or any ROS
integration -- it is evaluated entirely offline in this phase.

Design: predicts the 2x2 (measurement-space, BEV) innovation covariance
`S` via a Cholesky factorization `S = L L^T + eps*I`, `L` lower
triangular with a strictly positive diagonal (`softplus`) -- guaranteed
positive-definite by construction, never an arbitrary unconstrained
matrix connected to Mahalanobis association (per this task's own explicit
"do not connect an arbitrary covariance to Mahalanobis" instruction).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MEAS_DIM_DEFAULT = 2

__all__ = ["InnovationCovarianceHead", "cholesky_to_cov", "gaussian_nll"]


class InnovationCovarianceHead(nn.Module):
    """Small MLP predicting a Cholesky factor of the innovation
    covariance `S` (n x n) from the SAME normalized feature set
    `kalmannet_core.KalmanNetGRU`'s gain network already computes
    (`obs_diff`, `obs_innov_diff`, `fw_evol_diff`, `fw_update_diff`) --
    reused as input here, but this head's own weights are entirely
    separate from the frozen gain network's."""

    def __init__(self, m: int = 4, n: int = MEAS_DIM_DEFAULT, hidden: int = 16, eps: float = 1e-4):
        super().__init__()
        self.n = n
        self.eps = eps
        in_dim = n + n + m + m
        n_tril = n * (n + 1) // 2  # lower-triangular free parameters
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_tril),
        )
        tril_idx = torch.tril_indices(n, n)
        self.register_buffer("tril_row", tril_idx[0])
        self.register_buffer("tril_col", tril_idx[1])

    @staticmethod
    def _norm(v: torch.Tensor) -> torch.Tensor:
        return F.normalize(v, p=2, dim=-1, eps=1e-8)

    def forward(self, obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff) -> torch.Tensor:
        """Returns `L`: [batch, n, n], lower-triangular, strictly
        positive diagonal. `S = L @ L.transpose(-1,-2)` is guaranteed
        symmetric positive-definite for any finite input."""
        feats = torch.cat([
            self._norm(obs_diff), self._norm(obs_innov_diff),
            self._norm(fw_evol_diff), self._norm(fw_update_diff),
        ], dim=-1)
        raw = self.net(feats)  # [batch, n_tril]
        batch = raw.shape[0]
        L = torch.zeros(batch, self.n, self.n, dtype=raw.dtype, device=raw.device)
        L[:, self.tril_row, self.tril_col] = raw
        diag = torch.arange(self.n, device=raw.device)
        L[:, diag, diag] = F.softplus(L[:, diag, diag]) + self.eps
        return L


def cholesky_to_cov(L: torch.Tensor) -> torch.Tensor:
    return torch.bmm(L, L.transpose(-1, -2))


def gaussian_nll(innovation: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Per-sample Gaussian negative log-likelihood of `innovation`
    (`[batch, n]`) under `N(0, S)`, `S = L L^T`, computed stably via the
    Cholesky factor directly (never inverting/determinant-ing `S`
    explicitly): `0.5*(y^T S^-1 y + log|S| + n*log(2*pi))`, with
    `y^T S^-1 y` solved via `torch.cholesky_solve` and `log|S| =
    2*sum(log(diag(L)))`."""
    n = innovation.shape[-1]
    y = innovation.unsqueeze(-1)  # [batch, n, 1]
    z = torch.linalg.solve_triangular(L, y, upper=False)  # L z = y
    mahal_sq = (z ** 2).sum(dim=(-2, -1))  # y^T S^-1 y
    diag = torch.diagonal(L, dim1=-2, dim2=-1)
    logdet = 2.0 * torch.log(diag).sum(dim=-1)
    nll = 0.5 * (mahal_sq + logdet + n * torch.log(torch.tensor(2 * torch.pi, device=L.device)))
    return nll
