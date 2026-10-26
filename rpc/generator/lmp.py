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


class PhaseCoupling(nn.Module):
    """Learned pairwise coupling between phase channels.

    Per-limb phase buys limb independence and pays for it: four clocks each
    advancing on their own predicted frequency drift apart, and a 0.038 Hz gap
    between the legs integrated to 2921 degrees of relative drift over 16 s,
    which tore the gait out of antiphase and curved the walk.  A single global
    clock cannot drift but also cannot let an arm come loose from the gait,
    which is the point of per-limb phase.

    Coupling is therefore a parameter rather than a policy.  For channels i, j
    with unit phasors ``(c, s)``, the relative angle appears directly in the
    outer products

        sin(phi_j - phi_i) = s_j c_i - c_j s_i,
        cos(phi_j - phi_i) = c_i c_j + s_i s_j,

    and a learned ``A_ij``, ``B_ij`` turn those into a phase-rate correction on
    channel i.  Since ``A sin(d) + B cos(d) = K sin(d - psi)``, each pair learns
    both a strength and a preferred offset; a pair that needs no coupling
    simply learns ``K -> 0``, so nothing has to decide in advance which limbs
    are coupled.  The diagonal is masked: a channel does not pull on itself.
    """

    def __init__(self, n_channels: int):
        super().__init__()
        self.n = n_channels
        self.A = nn.Parameter(torch.zeros(n_channels, n_channels))
        self.B = nn.Parameter(torch.zeros(n_channels, n_channels))
        self.register_buffer("mask", 1.0 - torch.eye(n_channels))

    def forward(self, cos_t: torch.Tensor, sin_t: torch.Tensor) -> torch.Tensor:
        """(B, n) unit phasor components -> (B, n) phase-rate correction."""
        r = torch.sqrt(cos_t ** 2 + sin_t ** 2).clamp_min(1e-6)
        c, s = cos_t / r, sin_t / r
        # sin_d[b,i,j] = sin(phi_j - phi_i); cos_d[b,i,j] = cos(phi_j - phi_i)
        sin_d = s.unsqueeze(1) * c.unsqueeze(2) - c.unsqueeze(1) * s.unsqueeze(2)
        cos_d = c.unsqueeze(2) * c.unsqueeze(1) + s.unsqueeze(2) * s.unsqueeze(1)
        m = self.mask
        return (sin_d * (self.A * m)).sum(-1) + (cos_d * (self.B * m)).sum(-1)

    @torch.no_grad()
    def strength(self) -> torch.Tensor:
        """K_ij, the learned coupling magnitude for every ordered pair."""
        return torch.hypot(self.A, self.B) * self.mask

    @torch.no_grad()
    def offset(self) -> torch.Tensor:
        """psi_ij, the learned preferred phase offset, radians."""
        return torch.atan2(-self.B, self.A)


class CoupledLMPGenerator(LMPGenerator):
    """LMP generator whose predicted phase rate carries a learned coupling term.

    The trunk predicts each channel's frequency as before; the coupling module
    adds a correction computed from every other channel's phase.  Both are
    trained against the next-frame manifold the data supplies, so the coupling
    is learned from motion rather than fitted to it as a preprocessing step.
    """

    def __init__(self, *args, n_phase_channels: int, manifold_slice: slice, **kw):
        super().__init__(*args, **kw)
        self.n_phase = n_phase_channels
        self.manifold_slice = manifold_slice
        self.coupling = PhaseCoupling(n_phase_channels)

    def forward(self, x, gate_x, *, normalized: bool = False):
        y = super().forward(x, gate_x, normalized=normalized)
        n = self.n_phase
        # gate_x is the current manifold: [a_cos | a_sin | a | f]
        corr = self.coupling(gate_x[:, :n], gate_x[:, n:2 * n])
        # add the correction to the predicted frequency block, in normalised units
        s = self.manifold_slice
        f0 = s.start + 3 * n
        out = y.clone()
        # The trunk's output is in normalised units (step() de-normalises it);
        # the correction is a physical rate in Hz, so it is scaled by the same
        # per-dimension std before it is added.
        hz = corr / (2 * np.pi)
        out[:, f0:f0 + n] = out[:, f0:f0 + n] + hz / self.y_std[f0:f0 + n]
        return out
