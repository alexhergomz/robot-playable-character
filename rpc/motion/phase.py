"""Phase: the coordinate the whole lineage is built on.

Two estimators, because the project's argument is that these are the same idea
arrived at from different directions:

* :func:`contact_phase` — the original PFNN construction.  Phase is defined by
  the foot-contact schedule, advancing pi per heel strike.  Cheap, exact for
  locomotion, undefined for everything else.
* :class:`PeriodicAutoencoder` — DeepPhase (Starke et al. 2022).  Learns a
  multi-channel phase manifold from unstructured motion with no contact labels
  and no gait assumption, so it survives dancing, falling and recovery.

The robot side of the project needs the second, because a humanoid's contact
schedule is exactly what an operator perturbs.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# contact-based phase (PFNN 2017)
# --------------------------------------------------------------------------
def _rolling_floor(z: np.ndarray, window: int, q: float = 10.0) -> np.ndarray:
    """Local floor height per joint: a rolling low percentile of its height.

    A single global percentile breaks on any clip where the subject leaves the
    standing plane — LAFAN1's walk3 has the actor go to the ground for twenty
    seconds, which drags a global estimate down by 40 cm and suppresses contact
    detection for the entire clip.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    T = z.shape[0]
    w = min(window if window % 2 else window + 1, T if T % 2 else T - 1)
    w = max(w, 3)
    pad = w // 2
    zp = np.pad(z, ((pad, pad), (0, 0)), mode="edge")
    sw = sliding_window_view(zp, w, axis=0)                   # (T, G, w)
    return np.percentile(sw, q, axis=-1)[:T]


def foot_contacts(positions: np.ndarray, foot_groups: list[list[int]], fps: float,
                  margin: float = 0.06, speed_thresh: float = 1.5,
                  floor_window: float = 4.0) -> np.ndarray:
    """Boolean contact per foot, (T, n_feet).

    ``foot_groups`` is one list of joint indices per foot — typically ankle and
    toe, because heel strike plants the ankle while the toe is still raised and
    toe-off does the reverse, so neither joint alone covers a stride.

    A joint counts as planted when it is slow *and* within ``margin`` of its own
    *locally* estimated floor.  Calibrating per joint rather than globally is
    what makes this work across ankle vs. toe and across skeleton scales;
    calibrating over a rolling window is what makes it survive clips where the
    actor leaves the standing plane.

    The speed threshold is deliberately loose: LAFAN1 contains real foot
    sliding, so demanding a stationary foot would reject most true stance.
    """
    contacts = []
    for group in foot_groups:
        p = positions[:, group]                                  # (T, G, 3)
        v = np.linalg.norm(np.gradient(p, axis=0) * fps, axis=-1)
        floor = _rolling_floor(p[..., 2], int(floor_window * fps))   # (T, G)
        planted = (p[..., 2] < floor + margin) & (v < speed_thresh)
        contacts.append(planted.any(axis=-1))
    return np.stack(contacts, axis=-1)


def foot_groups_from_skeleton(skel, sides=("Left", "Right"),
                              parts=("Foot", "Toe")) -> list[list[int]]:
    """Ankle+toe joint indices per side, by name."""
    groups = []
    for side in sides:
        g = []
        for part in parts:
            try:
                g.append(skel.find(side, part))
            except KeyError:
                pass
        if not g:
            raise KeyError(f"no foot joints found for side {side!r}")
        groups.append(g)
    return groups


def contact_phase(contacts: np.ndarray, fps: float) -> np.ndarray:
    """Continuous gait phase in [0, 2*pi) from alternating foot contacts.

    Rising edges of the two feet are the anchors: left strike -> 0,
    right strike -> pi.  Between anchors phase is interpolated linearly, so
    the derivative is the instantaneous gait frequency.  Frames outside any
    stride (standing, airborne) get a constant-rate fallback.
    """
    assert contacts.shape[1] >= 2, "need at least a left and a right foot"
    T = contacts.shape[0]
    strikes: list[tuple[int, float]] = []
    for foot, base in ((0, 0.0), (1, np.pi)):
        c = contacts[:, foot].astype(int)
        rising = np.flatnonzero(np.diff(c) > 0) + 1
        strikes += [(int(t), base) for t in rising]
    strikes.sort()

    if len(strikes) < 2:
        return (np.arange(T) / fps * 2 * np.pi) % (2 * np.pi)

    phase = np.zeros(T)
    # unwrap anchor phases into a monotonically increasing sequence
    anchors_t = np.array([s[0] for s in strikes], float)
    anchors_p = np.array([s[1] for s in strikes], float)
    unwrapped = [anchors_p[0]]
    for k in range(1, len(anchors_p)):
        step = (anchors_p[k] - anchors_p[k - 1]) % (2 * np.pi)
        step = step if step > 1e-6 else 2 * np.pi
        unwrapped.append(unwrapped[-1] + step)
    unwrapped = np.array(unwrapped)

    inside = np.interp(np.arange(T), anchors_t, unwrapped)
    # extrapolate the ends at the nearest observed rate
    rate0 = (unwrapped[1] - unwrapped[0]) / max(anchors_t[1] - anchors_t[0], 1)
    rate1 = (unwrapped[-1] - unwrapped[-2]) / max(anchors_t[-1] - anchors_t[-2], 1)
    pre = np.arange(T) < anchors_t[0]
    post = np.arange(T) > anchors_t[-1]
    inside[pre] = unwrapped[0] - (anchors_t[0] - np.arange(T)[pre]) * rate0
    inside[post] = unwrapped[-1] + (np.arange(T)[post] - anchors_t[-1]) * rate1
    phase = inside % (2 * np.pi)
    return phase


def phase_delta(phase: np.ndarray) -> np.ndarray:
    """Per-frame phase advance, wrapped to (-pi, pi]."""
    d = np.zeros_like(phase)
    d[:-1] = np.angle(np.exp(1j * (phase[1:] - phase[:-1])))
    d[-1] = d[-2] if len(d) > 1 else 0.0
    return d


# --------------------------------------------------------------------------
# DeepPhase periodic autoencoder (Starke et al. 2022)
# --------------------------------------------------------------------------
class PeriodicAutoencoder(nn.Module):
    """Learns M periodic latent channels from a window of motion.

    Encoder: temporal 1D convolutions -> M latent curves over the window.
    Each curve is fitted by a sinusoid whose amplitude/frequency/offset come
    from its FFT and whose phase shift comes from a small linear head.  The
    decoder reconstructs the input from those sinusoids alone, so the only way
    to lower the loss is to make the latents genuinely periodic.

    The (amplitude, frequency, phase) triple per channel is the phase manifold
    the generator is conditioned on, and the thing a robot's contact schedule
    can be aligned to.
    """

    def __init__(self, n_channels_in: int, window: int, n_phases: int = 5,
                 fps: float = 30.0, hidden: int | None = None):
        super().__init__()
        self.n_in = n_channels_in
        self.window = window
        self.n_phases = n_phases
        self.fps = fps
        hidden = hidden or max(n_phases * 2, n_channels_in // 3)

        k = window - 1 if (window - 1) % 2 == 1 else window - 2   # odd kernel
        pad = k // 2
        self.enc = nn.Sequential(
            nn.Conv1d(n_channels_in, hidden, k, padding=pad), nn.BatchNorm1d(hidden), nn.ELU(),
            nn.Conv1d(hidden, n_phases, k, padding=pad),
        )
        self.phase_head = nn.ModuleList(
            [nn.Linear(window, 2) for _ in range(n_phases)]      # (cos, sin) of the shift
        )
        self.dec = nn.Sequential(
            nn.Conv1d(n_phases, hidden, k, padding=pad), nn.BatchNorm1d(hidden), nn.ELU(),
            nn.Conv1d(hidden, n_channels_in, k, padding=pad),
        )
        # time axis of the window, centred on 0
        self.register_buffer("t", (torch.arange(window).float() - window // 2) / fps)

    def fft_params(self, latent: torch.Tensor):
        """Per-channel amplitude, frequency, offset from the real FFT."""
        B, M, T = latent.shape
        spec = torch.fft.rfft(latent, dim=-1)
        power = spec.abs() ** 2                                  # (B, M, T//2+1)
        freqs = torch.fft.rfftfreq(T, d=1.0 / self.fps).to(latent.device)

        dc, ac = power[..., :1], power[..., 1:]
        denom = ac.sum(-1).clamp_min(1e-8)
        f = (ac * freqs[1:]).sum(-1) / denom                     # power-weighted mean freq
        a = torch.sqrt(2.0 * ac.sum(-1)) / T                     # RMS amplitude
        b = spec[..., 0].real / T                                # DC offset
        _ = dc
        return a, f, b

    def forward(self, x: torch.Tensor):
        """x: (B, C, window) -> (reconstruction, params dict)."""
        z = self.enc(x)                                          # (B, M, W)
        a, f, b = self.fft_params(z)

        s = torch.stack(
            [head(z[:, i]) for i, head in enumerate(self.phase_head)], dim=1
        )                                                        # (B, M, 2)
        s = s / s.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        phase = torch.atan2(s[..., 1], s[..., 0])                # (B, M) in (-pi, pi]

        arg = 2 * np.pi * (f[..., None] * self.t[None, None, :]) + phase[..., None]
        recon_latent = a[..., None] * torch.sin(arg) + b[..., None]
        y = self.dec(recon_latent)
        return y, {"amplitude": a, "frequency": f, "offset": b,
                   "phase": phase, "latent": z, "recon_latent": recon_latent}

    @torch.no_grad()
    def encode_phase(self, x: torch.Tensor) -> torch.Tensor:
        """Phase manifold coordinates: (B, 2M) as amplitude-scaled (cos, sin).

        This is the conditioning vector handed to the generator — continuous,
        wrap-free, and zero when a channel carries no periodic energy.
        """
        _, p = self.forward(x)
        a, ph = p["amplitude"], p["phase"]
        return torch.cat([a * torch.cos(ph), a * torch.sin(ph)], dim=-1)


def pae_input(local_pos: np.ndarray, local_vel: np.ndarray) -> np.ndarray:
    """Channel layout DeepPhase uses: joint velocities in the character frame.

    Returns (T, 3J).  Velocities rather than positions because periodicity
    lives in the derivative — a walking character's foot position is periodic
    only after the root motion is removed, its velocity already is.
    """
    _ = local_pos
    return local_vel.reshape(local_vel.shape[0], -1)
