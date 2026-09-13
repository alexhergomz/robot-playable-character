"""Local-Motion-Phase generator: the successor to the scalar-phase PFNN.

Three changes from :class:`~rpc.generator.pfnn.PFNN`, each aimed at a defect
that was actually measured on the robot rather than at a paper's ablation:

1. **Gating on the phase manifold, not Catmull-Rom over a scalar.**  The
   experts are blended by a small gating network reading the DeepPhase
   manifold (per channel: ``a*cos(phi), a*sin(phi), a, f``).  Channels lock to
   contact events independently (measured R ~= 0.7 at touchdown), so one limb
   can be redirected without resetting the clock the others run on.

2. **Frequency, not ``dphase``.**  The old runtime advanced phase by
   integrating a predicted per-frame increment, so a systematic bias compounded
   forever with nothing to correct it.  Here the network predicts each
   channel's frequency in Hz and the runtime advances ``phi += 2*pi*f*dt``.
   A biased ``f`` gives a wrong-tempo gait; it cannot drift without bound.

3. **A predicted foot position**, in the character frame, which the runtime
   feeds back as state.

   :func:`skate_loss` is retained but **off by default**.  It was added to fix
   an apparent 10x foot-skate excess over the training data; that figure turned
   out to be an artifact of differencing the foot *body origin*, which swings
   fast during heel-off while the contact patch stays planted.  Measured at the
   material point actually in contact, this generator was already inside the
   corpus band.  Ablated over 20 epochs the term showed no benefit: the arm
   that never optimises it reached a better validation skate value (83.68 vs
   85.58 cm^2) and equal closed-loop slip.  Reconstruction already implies
   non-sliding feet, because the data does not slide.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .pfnn import _Base


class LMPGenerator(_Base):
    def __init__(self, n_in: int, n_out: int, gate_dim: int, hidden: int = 512,
                 n_experts: int = 8, gate_hidden: int = 64, dropout: float = 0.7):
        super().__init__(n_in, n_out, n_experts, hidden, dropout)
        self.gate = nn.Sequential(
            nn.Linear(gate_dim, gate_hidden), nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, gate_hidden), nn.ELU(),
            nn.Linear(gate_hidden, n_experts),
        )

    def forward(self, x: torch.Tensor, gate_x: torch.Tensor, *, normalized: bool = False):
        if not normalized:
            x = (x - self.x_mean) / self.x_std
        coeff = torch.softmax(self.gate(gate_x), dim=-1)
        return self._trunk(x, coeff)


def _rot(dyaw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.cos(-dyaw), torch.sin(-dyaw)


def skate_loss(pred: torch.Tensor, slices: dict[str, slice], y_std: torch.Tensor,
               y_mean: torch.Tensor, foot_now: torch.Tensor,
               contact_now: torch.Tensor) -> torch.Tensor:
    """Planted feet must be world-fixed under the network's own root delta.

    ``foot_now`` (B, n_feet, 3) is the foot position in the character frame at
    t, ``contact_now`` (B, n_feet) its contact flag.  A world-fixed point moves
    to ``R(-dyaw) @ (p - d)`` in the frame at t+1, so any disagreement between
    the predicted next foot position and that value is skate the network is
    asking the robot to perform.
    """
    de = pred * y_std + y_mean                       # de-normalise; loss is in metres
    root = de[:, slices["root_delta"]]               # (B, 3) forward, lateral, dyaw
    nxt = de[:, slices["foot_pos"]].reshape(len(de), -1, 3)

    c, s = _rot(root[:, 2])
    px = foot_now[..., 0] - root[:, :1]
    py = foot_now[..., 1] - root[:, 1:2]
    want_x = c[:, None] * px - s[:, None] * py
    want_y = s[:, None] * px + c[:, None] * py

    err = (nxt[..., 0] - want_x) ** 2 + (nxt[..., 1] - want_y) ** 2
    w = contact_now
    return (err * w).sum() / w.sum().clamp_min(1.0)


def manifold_loss(pred: torch.Tensor, target: torch.Tensor,
                  slices: dict[str, slice]) -> torch.Tensor:
    """Next-frame phase manifold, normalised space."""
    s = slices["manifold"]
    return torch.nn.functional.mse_loss(pred[:, s], target[:, s])


class UncertaintyWeighting(nn.Module):
    """Learned loss weights (Kendall, Gal & Cipolla, CVPR 2018).

    Replaces hand-typed coefficients like ``ROBOT_LOSS_WEIGHTS`` (joint_pos 4.0,
    root_delta 10.0, dphase 5.0 ...), which were typed on normalised MSE and
    never validated -- 31.8% of that objective was being spent on outputs the
    runtime discards.

    Each term gets a learned log-variance ``s_i`` and contributes

        exp(-s_i) * L_i + s_i

    The first factor lets the optimiser down-weight a term it cannot fit; the
    ``+ s_i`` makes doing so cost something, so the weights settle where the
    terms are genuinely informative rather than collapsing to zero.  The
    resulting weights are *reported*, which also makes it visible when a term
    is contributing nothing and should be dropped.
    """

    def __init__(self, terms: tuple[str, ...]):
        super().__init__()
        self.terms = tuple(terms)
        self.log_var = nn.Parameter(torch.zeros(len(self.terms)))

    def forward(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        total = 0.0
        for i, name in enumerate(self.terms):
            s = self.log_var[i]
            total = total + torch.exp(-s) * losses[name] + s
        return total

    @torch.no_grad()
    def weights(self) -> dict[str, float]:
        return {n: float(torch.exp(-self.log_var[i]))
                for i, n in enumerate(self.terms)}
