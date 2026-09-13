"""The kinematic generator: PFNN and its mode-adaptive successor.

Both are the same three-layer network; they differ only in where the blending
coefficients come from.  PFNN indexes weights by a scalar gait phase through a
cyclic cubic; MANN learns a gating network over a chosen slice of the input.
Sharing the layer makes the comparison an ablation rather than a rewrite, which
is the point — the project's claim is about the *interface*, and the interface
is identical either way.

Normalisation statistics live inside the module as buffers, so a checkpoint is
a complete controller: no side-car pickle to lose.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .blend import BlendedLinear, catmull_rom_coefficients


class _Base(nn.Module):
    def __init__(self, n_in: int, n_out: int, n_experts: int, hidden: int, dropout: float):
        super().__init__()
        self.n_in, self.n_out, self.n_experts = n_in, n_out, n_experts
        self.l0 = BlendedLinear(n_experts, n_in, hidden)
        self.l1 = BlendedLinear(n_experts, hidden, hidden)
        self.l2 = BlendedLinear(n_experts, hidden, n_out)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ELU()
        self.register_buffer("x_mean", torch.zeros(n_in))
        self.register_buffer("x_std", torch.ones(n_in))
        self.register_buffer("y_mean", torch.zeros(n_out))
        self.register_buffer("y_std", torch.ones(n_out))

    def set_normalizers(self, x_norm, y_norm) -> None:
        for name, arr in (("x_mean", x_norm.mean), ("x_std", x_norm.std),
                          ("y_mean", y_norm.mean), ("y_std", y_norm.std)):
            getattr(self, name).copy_(torch.as_tensor(np.asarray(arr), dtype=torch.float32))

    def _trunk(self, x_norm: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.act(self.l0(self.drop(x_norm), coeff)))
        h = self.drop(self.act(self.l1(h, coeff)))
        return self.l2(h, coeff)

    @torch.no_grad()
    def step(self, x_raw: torch.Tensor, *aux) -> torch.Tensor:
        """Un-normalised in, un-normalised out — the runtime entry point."""
        self.eval()
        y = self.forward((x_raw - self.x_mean) / self.x_std, *aux, normalized=True)
        return y * self.y_std + self.y_mean


class PFNN(_Base):
    """Holden et al. 2017.  Weights are a cyclic cubic function of gait phase."""

    def __init__(self, n_in: int, n_out: int, hidden: int = 512,
                 n_control: int = 4, dropout: float = 0.3):
        super().__init__(n_in, n_out, n_control, hidden, dropout)
        self.n_control = n_control

    def forward(self, x: torch.Tensor, phase: torch.Tensor, *, normalized: bool = False):
        if not normalized:
            x = (x - self.x_mean) / self.x_std
        coeff = catmull_rom_coefficients(phase.reshape(-1), self.n_control).to(x.dtype)
        return self._trunk(x, coeff)


class MANN(_Base):
    """Zhang et al. 2018.  A gating network replaces the hand-defined phase.

    ``gate_slice`` selects which inputs the gate sees.  In the original this is
    the end-effector velocities and the desired trajectory — deliberately not
    the whole state, so the gate learns *mode* rather than memorising the pose.
    """

    def __init__(self, n_in: int, n_out: int, gate_dim: int, hidden: int = 512,
                 n_experts: int = 8, gate_hidden: int = 64, dropout: float = 0.3):
        super().__init__(n_in, n_out, n_experts, hidden, dropout)
        self.gate = nn.Sequential(
            nn.Linear(gate_dim, gate_hidden), nn.ELU(),
            nn.Linear(gate_hidden, gate_hidden), nn.ELU(),
            nn.Linear(gate_hidden, n_experts),
        )

    def forward(self, x: torch.Tensor, gate_x: torch.Tensor, *, normalized: bool = False):
        if not normalized:
            x = (x - self.x_mean) / self.x_std
        coeff = torch.softmax(self.gate(gate_x), dim=-1)
        return self._trunk(x, coeff)


def loss_terms(pred: torch.Tensor, target: torch.Tensor, slices: dict[str, slice],
               weights: dict[str, float] | None = None):
    """Per-block losses.

    Blocks are weighted separately because they are not commensurate: the root
    delta is three numbers that decide where the character ends up, while the
    joint positions are sixty-six that decide what it looks like.  Plain MSE
    over the concatenation lets pose drown out locomotion, which shows up as a
    controller that animates beautifully and ignores the stick.
    """
    weights = weights or {}
    out, total = {}, 0.0
    for name, sl in slices.items():
        term = torch.mean((pred[:, sl] - target[:, sl]) ** 2)
        out[name] = term
        total = total + weights.get(name, 1.0) * term
    out["total"] = total
    return out


def physical_loss(pred: torch.Tensor, target: torch.Tensor, y_std: torch.Tensor,
                  slices: dict[str, slice], weights: dict[str, float] | None = None):
    """Loss in physical units, over only the outputs the controller consumes.

    The alternative — per-block MSE on channels each divided by their own std,
    weighted by hand — is arbitrary twice over: the weights are typed in, and
    normalising per channel makes the blocks incommensurable, so a quantity with
    small natural variance dominates for no physical reason.  Here everything is
    radians or metres, so the terms are directly comparable and the only weights
    left are the two that convert between them.
    """
    weights = weights or PHYSICAL_LOSS_WEIGHTS
    out, total = {}, 0.0
    for name, w in weights.items():
        sl = slices[name]
        err = (pred[:, sl] - target[:, sl]) * y_std[sl]      # de-normalise
        term = torch.mean(err ** 2)
        out[name] = term
        total = total + w * term
    out["total"] = total
    return out


#: Only the blocks the physics runtime actually reads.  root_delta and
#: root_height are predicted, supervised and then discarded: on a robot the root
#: is an outcome of contact forces, not something the controller commands.
PHYSICAL_LOSS_WEIGHTS = {
    "joint_pos": 1.0,        # rad^2 -- the actuator command
    "dphase": 1.0,           # rad^2 -- indexes the network's own weights
    "future_pos": 1.0,       # m^2
    "future_dir": 1.0,       # unit vector
}


ROBOT_LOSS_WEIGHTS = {
    "joint_pos": 4.0,       # these *are* the actuator targets
    "joint_vel": 0.3,
    "root_delta": 10.0,
    "root_height": 5.0,     # the single number that says "still standing"
    "dphase": 5.0,
    "future_pos": 1.0,
    "future_dir": 1.0,
    "contacts": 0.5,
}

DEFAULT_LOSS_WEIGHTS = {
    "joint_pos": 1.0,
    "joint_vel": 0.3,
    "joint_rot6d": 1.0,
    "root_delta": 10.0,     # locomotion is the control channel; do not let pose drown it
    "dphase": 5.0,          # phase drift desynchronises the gait within a few seconds
    "future_pos": 1.0,
    "future_dir": 1.0,
    "contacts": 0.5,
}
