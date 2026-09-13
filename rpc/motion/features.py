"""Control features — the interface the operator actually drives.

This is the part of the PFNN line worth keeping.  The character is not
commanded with a goal; it is commanded with a *window of trajectory* around
the current frame, expressed in the character's own ground frame.  The past
half is observed, the future half is intent, and the operator edits the future
half directly.  Everything else in the system is downstream of this choice.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import quat as Q
from .clip import MotionClip, root_frame

GAITS: tuple[str, ...] = ("stand", "walk", "run", "crouch", "jump")


@dataclass(frozen=True)
class FeatureConfig:
    fps: float = 30.0
    n_past: int = 6                 # trajectory samples behind the character
    n_future: int = 6               # ...and ahead of it
    span: float = 1.0               # seconds covered on each side
    gaits: tuple[str, ...] = GAITS

    @property
    def n_traj(self) -> int:
        return self.n_past + 1 + self.n_future

    @property
    def offsets(self) -> np.ndarray:
        """Frame offsets of the trajectory samples, e.g. [-30..0..+30] step 5."""
        step = self.span * self.fps / self.n_past
        return np.round(
            np.arange(-self.n_past, self.n_future + 1) * step
        ).astype(int)

    @property
    def future_offsets(self) -> np.ndarray:
        return self.offsets[self.n_past + 1:]


def sample_trajectory(cfg: FeatureConfig, positions: np.ndarray, rotations: np.ndarray,
                      fwd_axis: np.ndarray | None = None):
    """Character-local trajectory window for every frame.

    Returns ``(pos_xy, dir_xy)`` each ``(T, n_traj, 2)``.  Samples beyond the
    clip are clamped to the ends, which is what a live controller does at
    startup anyway.
    """
    T = positions.shape[0]
    origin, heading = root_frame(rotations, positions, fwd_axis)
    inv_h = Q.conj(heading)                                   # (T, 4)

    idx = np.clip(np.arange(T)[:, None] + cfg.offsets[None, :], 0, T - 1)   # (T, K)
    world_p = positions[idx, 0]                               # (T, K, 3)
    world_p = world_p - origin[:, None, :]
    local_p = Q.rotate(np.broadcast_to(inv_h[:, None, :], world_p.shape[:-1] + (4,)), world_p)

    fwd = Q.rotate(rotations[:, 0],
                   np.array([1.0, 0.0, 0.0]) if fwd_axis is None else np.asarray(fwd_axis))
    fwd[:, 2] = 0.0
    fwd = fwd / np.clip(np.linalg.norm(fwd, axis=-1, keepdims=True), 1e-8, None)
    world_d = fwd[idx]
    local_d = Q.rotate(np.broadcast_to(inv_h[:, None, :], world_d.shape[:-1] + (4,)), world_d)

    return local_p[..., :2], local_d[..., :2]


def pose_features(positions: np.ndarray, rotations: np.ndarray, fps: float,
                  fwd_axis: np.ndarray | None = None):
    """Character-local joint state.  Returns (pos (T,J,3), vel (T,J,3), rot6d (T,J,6))."""
    origin, heading = root_frame(rotations, positions, fwd_axis)
    inv_h = Q.conj(heading)
    B = lambda q, shape: np.broadcast_to(q[:, None, :], shape[:-1] + (4,))

    rel = positions - origin[:, None, :]
    local_p = Q.rotate(B(inv_h, rel.shape), rel)

    world_v = np.gradient(positions, axis=0) * fps
    local_v = Q.rotate(B(inv_h, world_v.shape), world_v)

    local_r = Q.mul(B(inv_h, rotations.shape), rotations)
    return local_p, local_v, Q.flatten_2axis(local_r)


def root_deltas(positions: np.ndarray, rotations: np.ndarray,
                fwd_axis: np.ndarray | None = None):
    """Per-frame root motion in the *previous* frame's local frame.

    (T, 3): forward, lateral, yaw-rate.  This is the only channel by which the
    character moves through the world — integrating it is what makes the
    generator autoregressive rather than a pose lookup.
    """
    origin, heading = root_frame(rotations, positions, fwd_axis)
    yaw = Q.yaw_of(heading)
    d_world = np.zeros_like(origin)
    d_world[:-1] = origin[1:] - origin[:-1]
    d_local = Q.rotate(Q.conj(heading), d_world)
    d_yaw = np.zeros_like(yaw)
    d_yaw[:-1] = np.angle(np.exp(1j * (yaw[1:] - yaw[:-1])))
    return np.stack([d_local[:, 0], d_local[:, 1], d_yaw], axis=-1)


def gait_labels(cfg: FeatureConfig, positions: np.ndarray, fps: float,
                contacts: np.ndarray | None = None) -> np.ndarray:
    """Soft gait one-hot per frame, from speed and posture.

    Deliberately crude: gait is a *hint* the operator can override, not a
    classification problem.  Frozen thresholds keep the label stable across
    clips so the language channel can index into it later.
    """
    speed = np.linalg.norm(np.gradient(positions[:, 0, :2], axis=0) * fps, axis=-1)
    height = positions[:, 0, 2]
    ref_h = np.percentile(height, 90)

    g = np.zeros((positions.shape[0], len(cfg.gaits)))
    names = list(cfg.gaits)
    crouch = height < 0.85 * ref_h
    airborne = np.zeros(len(height), bool) if contacts is None else ~contacts.any(-1)

    g[:, names.index("stand")] = np.clip(1.0 - speed / 0.3, 0, 1)
    g[:, names.index("walk")] = np.clip(1.0 - np.abs(speed - 1.2) / 1.0, 0, 1)
    g[:, names.index("run")] = np.clip((speed - 1.8) / 1.2, 0, 1)
    if "crouch" in names:
        g[:, names.index("crouch")] = crouch.astype(float)
    if "jump" in names:
        g[:, names.index("jump")] = airborne.astype(float)
    return g / np.clip(g.sum(-1, keepdims=True), 1e-6, None)


def gait_window(cfg: FeatureConfig, gait: np.ndarray) -> np.ndarray:
    """Gait one-hot sampled over the trajectory window: (T, n_traj, n_gaits)."""
    T = gait.shape[0]
    idx = np.clip(np.arange(T)[:, None] + cfg.offsets[None, :], 0, T - 1)
    return gait[idx]
