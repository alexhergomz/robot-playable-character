"""Phase- and gate-blended linear layers.

The PFNN idea is that the network's *weights* are a function of phase:
``W(p) = sum_k c_k(p) W_k``.  Materialising ``W(p)`` per sample costs
``B x out x in`` floats, which is 500 MB for a batch of 1024 at PFNN's width —
so instead we use the identity

    W(p) x = sum_k c_k(p) (W_k x)

and evaluate the K shared matmuls, then blend their *outputs*.  Exactly
equivalent, K dense GEMMs instead of a batched one, and the memory cost drops
by a factor of ``in``.  MANN's gating network blends weights the same way, so
both architectures reduce to this one layer with different coefficients.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class BlendedLinear(nn.Module):
    """``y_b = sum_k c_bk (W_k x_b + b_k)`` for per-sample coefficients ``c``."""

    def __init__(self, n_experts: int, in_features: int, out_features: int):
        super().__init__()
        self.n_experts = n_experts
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(n_experts, out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(n_experts, out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Per-expert Kaiming-uniform, matching nn.Linear's default fan-in rule.
        bound = 1.0 / math.sqrt(self.in_features)
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)
            self.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
        # x (B, in), coeff (B, K) -> (B, out)
        per_expert = torch.einsum("koi,bi->bko", self.weight, x) + self.bias
        return torch.einsum("bko,bk->bo", per_expert, coeff)

    def blended_weight(self, coeff: torch.Tensor):
        """Explicit ``W(p), b(p)`` — used when exporting a fixed-phase controller."""
        return (torch.einsum("koi,bk->boi", self.weight, coeff),
                torch.einsum("ko,bk->bo", self.bias, coeff))


def catmull_rom_coefficients(phase: torch.Tensor, n_control: int) -> torch.Tensor:
    """Cyclic Catmull-Rom blend weights over ``n_control`` control points.

    ``phase`` is in [0, 2*pi).  Returns (B, n_control) coefficients that sum to
    one, so that ``sum_k c_k W_k`` reproduces PFNN's cubic phase function
    exactly — including the wrap-around at 2*pi that makes the controller
    seamless across gait cycles.
    """
    w = phase / (2 * math.pi) * n_control                    # position in control-point units
    k1 = torch.floor(w).long() % n_control
    t = (w - torch.floor(w)).unsqueeze(-1)                   # (B, 1)

    t2, t3 = t * t, t * t * t
    c = torch.cat([
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    ], dim=-1)                                               # (B, 4) for k0..k3

    idx = torch.stack([(k1 - 1) % n_control, k1,
                       (k1 + 1) % n_control, (k1 + 2) % n_control], dim=-1)
    out = torch.zeros(phase.shape[0], n_control, device=phase.device, dtype=c.dtype)
    return out.scatter_add_(1, idx, c)
