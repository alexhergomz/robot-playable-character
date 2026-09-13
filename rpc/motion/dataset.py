"""Assembling PFNN training pairs from motion clips.

One sample is a single control step: given what the character is doing now and
where the operator wants it to go, predict the next frame.  The layout is fixed
by :class:`FeatureSpec` and shared by the dataset builder, the network and the
runtime loop, so there is exactly one definition of what each number means.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .clip import MotionClip
from .features import (FeatureConfig, gait_labels, gait_window, pose_features,
                       root_deltas, sample_trajectory)
from .phase import (contact_phase, foot_contacts, foot_groups_from_skeleton,
                    phase_delta)


@dataclass(frozen=True)
class FeatureSpec:
    """Byte-for-byte layout of the generator's input and output vectors."""

    n_joints: int
    cfg: FeatureConfig
    n_feet: int = 2
    n_phase_channels: int = 0        # >0 when conditioning on a DeepPhase manifold

    # -- input ------------------------------------------------------------
    @property
    def in_blocks(self) -> list[tuple[str, int]]:
        K, G, J = self.cfg.n_traj, len(self.cfg.gaits), self.n_joints
        blocks = [("traj_pos", 2 * K), ("traj_dir", 2 * K), ("traj_gait", G * K),
                  ("joint_pos", 3 * J), ("joint_vel", 3 * J)]
        if self.n_phase_channels:
            blocks.append(("phase_manifold", 2 * self.n_phase_channels))
        return blocks

    # -- output -----------------------------------------------------------
    @property
    def out_blocks(self) -> list[tuple[str, int]]:
        J, nf = self.n_joints, self.cfg.n_future
        return [("joint_pos", 3 * J), ("joint_vel", 3 * J), ("joint_rot6d", 6 * J),
                ("root_delta", 3), ("dphase", 1),
                ("future_pos", 2 * nf), ("future_dir", 2 * nf),
                ("contacts", self.n_feet)]

    @staticmethod
    def _slices(blocks):
        out, off = {}, 0
        for name, n in blocks:
            out[name] = slice(off, off + n)
            off += n
        return out, off

    @property
    def input_slices(self):
        return self._slices(self.in_blocks)[0]

    @property
    def output_slices(self):
        return self._slices(self.out_blocks)[0]

    @property
    def n_in(self) -> int:
        return self._slices(self.in_blocks)[1]

    @property
    def n_out(self) -> int:
        return self._slices(self.out_blocks)[1]


@dataclass
class ClipFeatures:
    """Everything derived from one clip, before it is cut into samples."""

    traj_pos: np.ndarray       # (T, K, 2)
    traj_dir: np.ndarray       # (T, K, 2)
    gait: np.ndarray           # (T, K, G)
    joint_pos: np.ndarray      # (T, J, 3) character-local
    joint_vel: np.ndarray      # (T, J, 3)
    joint_rot6d: np.ndarray    # (T, J, 6)
    root_delta: np.ndarray     # (T, 3) frame t -> t+1, in frame t's local frame
    phase: np.ndarray          # (T,)
    dphase: np.ndarray         # (T,)
    contacts: np.ndarray       # (T, n_feet)
    labels: tuple[str, ...]
    name: str


def analyse(clip: MotionClip, cfg: FeatureConfig | None = None) -> ClipFeatures:
    """Run the full feature stack over one clip."""
    cfg = cfg or FeatureConfig(fps=clip.fps)
    skel = clip.skeleton
    groups = foot_groups_from_skeleton(skel)
    clip = clip.ground_align([j for g in groups for j in g])

    gp, gr = clip.forward_kinematics()
    contacts = foot_contacts(gp, groups, clip.fps)
    phase = contact_phase(contacts, clip.fps)

    tp, td = sample_trajectory(cfg, gp, gr, skel.fwd)
    lp, lv, lr = pose_features(gp, gr, clip.fps, skel.fwd)
    rd = root_deltas(gp, gr, skel.fwd)
    g = gait_window(cfg, gait_labels(cfg, gp, clip.fps, contacts))

    return ClipFeatures(tp, td, g, lp, lv, lr, rd, phase, phase_delta(phase),
                        contacts.astype(np.float32), clip.labels, clip.name)


def build_samples(feats: ClipFeatures, spec: FeatureSpec,
                  phase_manifold: np.ndarray | None = None):
    """Cut one clip's features into (X, Y, phase) arrays.

    Sample ``i`` reads the character's state at frame ``i`` and predicts frame
    ``i+1``; the phase used to index the network's weights is the phase at
    frame ``i``.  Trailing frames are dropped because they have no target.
    """
    T = feats.joint_pos.shape[0]
    n = T - 1
    flat = lambda a: a[:n].reshape(n, -1)

    x_parts = [flat(feats.traj_pos), flat(feats.traj_dir), flat(feats.gait),
               flat(feats.joint_pos), flat(feats.joint_vel)]
    if spec.n_phase_channels:
        assert phase_manifold is not None, "spec asks for a phase manifold"
        x_parts.append(phase_manifold[:n])
    X = np.concatenate(x_parts, axis=-1).astype(np.float32)

    nxt = slice(1, n + 1)
    nf = spec.cfg.n_future
    y_parts = [
        feats.joint_pos[nxt].reshape(n, -1),
        feats.joint_vel[nxt].reshape(n, -1),
        feats.joint_rot6d[nxt].reshape(n, -1),
        feats.root_delta[:n],
        feats.dphase[:n, None],
        feats.traj_pos[nxt][:, -nf:].reshape(n, -1),
        feats.traj_dir[nxt][:, -nf:].reshape(n, -1),
        feats.contacts[nxt],
    ]
    Y = np.concatenate(y_parts, axis=-1).astype(np.float32)
    assert X.shape[1] == spec.n_in, (X.shape[1], spec.n_in)
    assert Y.shape[1] == spec.n_out, (Y.shape[1], spec.n_out)
    return X, Y, feats.phase[:n].astype(np.float32)


class Normalizer:
    """Per-feature standardisation with a floor on the scale.

    Constant channels (a gait that never fires, a joint that never moves) would
    otherwise be divided by ~0 and blow up; the floor keeps them at zero.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray, floor: float = 1e-3):
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std, floor).astype(np.float32)

    @classmethod
    def fit(cls, x: np.ndarray, floor: float = 1e-3) -> "Normalizer":
        return cls(x.mean(0), x.std(0), floor)

    def __call__(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state(cls, d):
        return cls(np.asarray(d["mean"]), np.asarray(d["std"]))
