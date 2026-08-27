"""POST-FREEZE research extension, Phase 3: opt-in "HEVEN lightweight
Architecture-2 variant" of KalmanNet for the same minimal planar
constant-velocity (CV) tracking problem used by `kalmannet_core.py`.

State:       x = [x, y, vx, vy]^T   (m=4)
Measurement: z = [x, y]^T           (n=2)

This module does NOT modify `kalmannet_core.py`. It imports and reuses
its analytical process/measurement model (`F_matrix`, `H_matrix`,
`set_seed`) unchanged, so the frozen single-GRU DENSE-KALMANNET-v2
checkpoint and T-9B's exact 0.00e+00 offline/online equivalence proof are
never put at risk by this file existing.

Reference: `references/kalmannet` (KalmanNet_TSP, pinned submodule,
`KNet/KalmanNet_nn.py`) implements the paper's own "Architecture #2":
three separate GRUs tracking Q-, Sigma-, and S-related recurrent state
(`GRU_Q`, `GRU_Sigma`, `GRU_S`), a forward flow through two FC stages
producing the learned gain, and a backward-flow FC that feeds the output
back into the Sigma-GRU's own hidden state every step. This module is a
genuine, deliberately reduced port of that *topology* for HEVEN's tiny
m=4/n=2 problem -- it is NOT three stacked generic GRUs, and it is NOT
claimed to be "the paper exactly": the reference's own FC1-FC7 widths
(controlled by `in_mult_KNet`/`out_mult_KNet`) are tuned for much
higher-dimensional state-estimation problems than HEVEN's; here the same
wiring topology is kept at a much smaller, HEVEN-appropriate width.

Key structural improvement over both the reference implementation and
`kalmannet_core.py`'s own T-9B integration history: recurrent hidden
state (`h_Q`, `h_Sigma`, `h_S`) is never stored on the shared weight
module (`KalmanNetGRUArch2` itself has no mutable per-track attribute).
Instead `forward()` is a pure function of `(inputs, hidden_state) ->
(gain, new_hidden_state)`; each `KalmanNetArch2Filter` instance owns its
own `Arch2HiddenState` and passes it explicitly. This makes the T-9B
per-track hidden-state-leakage bug class (`engineering_forensics.csv`)
structurally impossible here, rather than fixed post hoc at a call site.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ad_lidar_perception.kalmannet_core import F_matrix, H_matrix, STATE_DIM, MEAS_DIM

__all__ = [
    "Arch2HiddenState",
    "KalmanNetGRUArch2",
    "KalmanNetArch2Filter",
    "count_parameters",
]


@dataclass
class Arch2HiddenState:
    """Track-private recurrent state for one `KalmanNetArch2Filter`
    instance. Never attached to `KalmanNetGRUArch2` itself -- passed by
    the caller into `forward()` and reassigned from its return value."""

    h_Q: torch.Tensor      # [1, batch, d_hidden_Q]
    h_Sigma: torch.Tensor  # [1, batch, d_hidden_Sigma]
    h_S: torch.Tensor      # [1, batch, d_hidden_S]

    def detach(self) -> "Arch2HiddenState":
        return Arch2HiddenState(
            self.h_Q.detach(), self.h_Sigma.detach(), self.h_S.detach()
        )


class KalmanNetGRUArch2(nn.Module):
    """HEVEN lightweight Architecture-2 gain network. Stateless as an
    `nn.Module`: all recurrent state lives in the `Arch2HiddenState`
    passed into/out of `forward()`, never on `self`.

    `in_mult`/`out_mult` are the reference's own `in_mult_KNet`/
    `out_mult_KNet` width multipliers, exposed for Phase 5's bounded
    architecture sweep. Defaults (5 / 40) match typical values used in
    the reference implementation's own example configs; they are
    untuned baselines here, not fit against any HEVEN data.
    """

    def __init__(
        self,
        m: int = STATE_DIM,
        n: int = MEAS_DIM,
        in_mult: int = 5,
        out_mult: int = 40,
    ):
        super().__init__()
        self.m = m
        self.n = n
        self.in_mult = in_mult
        self.out_mult = out_mult

        self.d_hidden_Q = m * m
        self.d_hidden_Sigma = m * m
        self.d_hidden_S = n * n

        d_input_Q = m * in_mult
        d_input_Sigma = self.d_hidden_Q + m * in_mult
        d_input_S = self.d_hidden_S + 2 * n * in_mult

        self.GRU_Q = nn.GRU(d_input_Q, self.d_hidden_Q)
        self.GRU_Sigma = nn.GRU(d_input_Sigma, self.d_hidden_Sigma)
        self.GRU_S = nn.GRU(d_input_S, self.d_hidden_S)

        # FC5/FC6 feed the Q-GRU / Sigma-GRU forward path from the raw
        # state-update / state-evolution differences.
        self.FC5 = nn.Sequential(nn.Linear(m, m * in_mult), nn.ReLU())
        self.FC6 = nn.Sequential(nn.Linear(m, m * in_mult), nn.ReLU())
        # FC7 feeds the S-GRU forward path from the raw observation diffs.
        self.FC7 = nn.Sequential(nn.Linear(2 * n, 2 * n * in_mult), nn.ReLU())

        # FC1 turns the Sigma-GRU's output into an S-GRU input contribution.
        self.FC1 = nn.Sequential(nn.Linear(self.d_hidden_Sigma, n * n), nn.ReLU())

        # FC2: forward-flow gain head (Sigma-GRU output + S-GRU output -> gain).
        d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        d_hidden_FC2 = d_input_FC2 * out_mult
        self.FC2 = nn.Sequential(
            nn.Linear(d_input_FC2, d_hidden_FC2),
            nn.ReLU(),
            nn.Linear(d_hidden_FC2, n * m),
        )

        # FC3/FC4: backward flow, feeds this step's gain estimate back
        # into next step's Sigma-GRU hidden state (the reference
        # architecture's distinguishing "backward flow" -- absent from
        # the single-GRU `kalmannet_core.KalmanNetGRU`).
        self.FC3 = nn.Sequential(
            nn.Linear(self.d_hidden_S + n * m, self.d_hidden_Q), nn.ReLU()
        )
        self.FC4 = nn.Sequential(
            nn.Linear(self.d_hidden_Sigma + self.d_hidden_Q, self.d_hidden_Sigma),
            nn.ReLU(),
        )

    def init_hidden(self, batch_size: int = 1, device=None) -> Arch2HiddenState:
        """Zero-initialized hidden state. Callers that want the
        reference's own prior-seeded init (`prior_Q`/`prior_Sigma`/
        `prior_S`) should build an `Arch2HiddenState` directly instead --
        zero-init is the documented HEVEN default, since no principled
        analytical prior_S/prior_Sigma exists for this reduced problem
        without re-deriving one, which this task's own scope excludes."""
        return Arch2HiddenState(
            torch.zeros(1, batch_size, self.d_hidden_Q, device=device),
            torch.zeros(1, batch_size, self.d_hidden_Sigma, device=device),
            torch.zeros(1, batch_size, self.d_hidden_S, device=device),
        )

    @staticmethod
    def _norm(v: torch.Tensor) -> torch.Tensor:
        return F.normalize(v, p=2, dim=-1, eps=1e-8)

    def forward(
        self,
        obs_diff: torch.Tensor,
        obs_innov_diff: torch.Tensor,
        fw_evol_diff: torch.Tensor,
        fw_update_diff: torch.Tensor,
        hidden: Arch2HiddenState,
    ) -> tuple[torch.Tensor, Arch2HiddenState]:
        """All feature inputs: [batch, dim] (raw, unnormalized). Returns
        (gain [batch, m, n], new_hidden_state)."""
        obs_diff = self._norm(obs_diff).unsqueeze(0)
        obs_innov_diff = self._norm(obs_innov_diff).unsqueeze(0)
        fw_evol_diff = self._norm(fw_evol_diff).unsqueeze(0)
        fw_update_diff = self._norm(fw_update_diff).unsqueeze(0)

        # ---- forward flow ----
        out_fc5 = self.FC5(fw_update_diff)
        out_q, h_q_new = self.GRU_Q(out_fc5, hidden.h_Q)

        out_fc6 = self.FC6(fw_evol_diff)
        in_sigma = torch.cat([out_q, out_fc6], dim=-1)
        out_sigma, h_sigma_new = self.GRU_Sigma(in_sigma, hidden.h_Sigma)

        out_fc1 = self.FC1(out_sigma)
        out_fc7 = self.FC7(torch.cat([obs_diff, obs_innov_diff], dim=-1))
        in_s = torch.cat([out_fc1, out_fc7], dim=-1)
        out_s, h_s_new = self.GRU_S(in_s, hidden.h_S)

        in_fc2 = torch.cat([out_sigma, out_s], dim=-1)
        out_fc2 = self.FC2(in_fc2)  # [1, batch, n*m]
        gain_flat = out_fc2.squeeze(0)  # [batch, n*m]

        # ---- backward flow: feed this step's gain estimate into next
        # step's Sigma-GRU hidden state ----
        in_fc3 = torch.cat([out_s, out_fc2], dim=-1)
        out_fc3 = self.FC3(in_fc3)
        in_fc4 = torch.cat([out_sigma, out_fc3], dim=-1)
        h_sigma_fed = self.FC4(in_fc4)

        new_hidden = Arch2HiddenState(h_q_new, h_sigma_fed, h_s_new)
        batch = obs_diff.shape[1]
        gain = gain_flat.view(batch, self.m, self.n)
        return gain, new_hidden


def count_parameters(net: KalmanNetGRUArch2) -> dict:
    """Per-submodule and total trainable parameter counts, for the
    required parameter-count report."""
    per_module = {
        name: sum(p.numel() for p in module.parameters())
        for name, module in net.named_children()
    }
    per_module["total"] = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return per_module


class KalmanNetArch2Filter:
    """Wraps `KalmanNetGRUArch2` with the explicit analytical f/h
    recursion, mirroring `kalmannet_core.KalmanNetFilter`'s structure and
    reusing its exact `F_matrix`/`H_matrix`. Owns its own
    `Arch2HiddenState` -- never touches any state on the shared network."""

    def __init__(self, net: KalmanNetGRUArch2, device="cpu"):
        self.net = net
        self.device = device
        self.hidden: Arch2HiddenState | None = None
        self.x_post = None
        self.x_post_prev = None
        self.x_prior_prev = None
        self.y_prev = None
        self.last_K = None

    def init_sequence(self, x0: torch.Tensor, batch_size: int = 1):
        self.hidden = self.net.init_hidden(batch_size, device=self.device)
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
        """Normal step: analytical predict, learned-gain correction from
        a real measurement. Advances hidden state (one GRU call)."""
        x_prior = self._f(self.x_post, dt)
        y_pred = self._h(x_prior)

        obs_diff = z - self.y_prev
        obs_innov_diff = z - y_pred
        fw_evol_diff = self.x_post - self.x_post_prev
        fw_update_diff = self.x_post - self.x_prior_prev

        K, self.hidden = self.net(
            obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff, self.hidden
        )
        self.last_K = K
        innovation = z - y_pred
        correction = torch.bmm(K, innovation.unsqueeze(-1)).squeeze(-1)
        x_post = x_prior + correction

        self.x_post_prev = self.x_post
        self.x_prior_prev = x_prior
        self.x_post = x_post
        self.y_prev = z
        return x_post

    def predict_only(self, dt: float) -> torch.Tensor:
        """Missing-measurement step: advances the analytical prior only,
        commits it as the new posterior, and makes NO network call --
        hidden state (`self.hidden`) is left completely untouched. Mirrors
        the ROS integration lifecycle established for the frozen
        single-GRU estimator (`Track.finalize_predict_only()` /
        `run_sequence_instrumented`'s own missing-measurement branch)."""
        x_prior = self._f(self.x_post, dt)
        self.x_post_prev = self.x_post
        self.x_prior_prev = x_prior
        self.x_post = x_prior
        return self.x_post
