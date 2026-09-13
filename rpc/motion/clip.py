"""The motion representation everything upstream of the robot speaks.

A :class:`MotionClip` is a skeleton plus a time series of *local* joint
rotations and a root translation.  Coordinates are metres, Z-up, right-handed,
matching MuJoCo — BVH's Y-up centimetres are converted at load time so that
nothing downstream has to know where the data came from.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np

from . import quat as Q


@dataclass(frozen=True)
class Skeleton:
    """Rest-pose topology plus the root's axis convention.

    Mocap skeletons disagree about which local axis of the root points forward
    (LAFAN1 uses +Z forward, +X up).  Rather than assume, we infer it once at
    load time and carry it, so every feature downstream is expressed in a
    canonical character frame: +X forward, +Y left, +Z up.
    """

    names: tuple[str, ...]
    parents: np.ndarray          # (J,) int, -1 for root
    offsets: np.ndarray          # (J, 3) rest-pose offset from parent, metres
    fwd: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))

    @property
    def n_joints(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        return self.names.index(name)

    def find(self, *substrings: str) -> int:
        """First joint whose name contains every substring (case-insensitive)."""
        for i, n in enumerate(self.names):
            low = n.lower()
            if all(s.lower() in low for s in substrings):
                return i
        raise KeyError(f"no joint matching {substrings} in {self.names}")

    def bone_lengths(self) -> np.ndarray:
        return np.linalg.norm(self.offsets, axis=-1)

    def rest_positions(self) -> np.ndarray:
        """Joint positions with all local rotations identity, root at origin."""
        p = np.zeros((self.n_joints, 3))
        for j in range(1, self.n_joints):
            p[j] = p[self.parents[j]] + self.offsets[j]
        return p

    def with_inferred_axes(self) -> "Skeleton":
        """Infer the root's forward/up axes from limb symmetry and spine direction.

        Left/right joint pairs give the lateral axis; the root-to-head vector
        gives up; forward is their cross product.  Name-based, but only on the
        near-universal 'left'/'right' convention.
        """
        rest = self.rest_positions()
        lows = [n.lower() for n in self.names]

        lateral = []
        for i, n in enumerate(lows):
            if "left" in n:
                mate = n.replace("left", "right")
                if mate in lows:
                    lateral.append(rest[i] - rest[lows.index(mate)])
        if not lateral:
            return self
        left = np.mean(lateral, axis=0)
        left /= np.linalg.norm(left)

        top = None
        for key in ("head", "neck", "spine"):
            cands = [i for i, n in enumerate(lows) if key in n]
            if cands:
                top = max(cands, key=lambda i: np.linalg.norm(rest[i] - rest[0]))
                break
        up = rest[top] - rest[0] if top is not None else np.array([0.0, 0.0, 1.0])
        up = up - np.dot(up, left) * left
        up /= np.linalg.norm(up)

        fwd = np.cross(left, up)
        fwd /= np.linalg.norm(fwd)
        return Skeleton(self.names, self.parents, self.offsets, fwd, up)


@dataclass
class MotionClip:
    skeleton: Skeleton
    root_pos: np.ndarray         # (T, 3) metres
    rotations: np.ndarray        # (T, J, 4) local quaternions, [0] is the root's world rotation
    fps: float
    name: str = ""
    labels: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        assert self.root_pos.ndim == 2 and self.root_pos.shape[1] == 3
        assert self.rotations.ndim == 3 and self.rotations.shape[2] == 4
        assert self.root_pos.shape[0] == self.rotations.shape[0]
        assert self.rotations.shape[1] == self.skeleton.n_joints

    def __len__(self) -> int:
        return self.root_pos.shape[0]

    @property
    def duration(self) -> float:
        return len(self) / self.fps

    # -- kinematics ---------------------------------------------------------
    def forward_kinematics(self) -> tuple[np.ndarray, np.ndarray]:
        """Global joint positions (T, J, 3) and rotations (T, J, 4)."""
        return forward_kinematics(self.skeleton, self.root_pos, self.rotations)

    # -- editing ------------------------------------------------------------
    def slice(self, start: int, end: int) -> "MotionClip":
        return replace(self, root_pos=self.root_pos[start:end],
                       rotations=self.rotations[start:end],
                       name=f"{self.name}[{start}:{end}]")

    def resample(self, fps: float) -> "MotionClip":
        if abs(fps - self.fps) < 1e-6:
            return self
        n_out = max(2, int(round(len(self) * fps / self.fps)))
        src = np.linspace(0, len(self) - 1, n_out)
        i0 = np.floor(src).astype(int)
        i1 = np.minimum(i0 + 1, len(self) - 1)
        t = (src - i0)
        pos = self.root_pos[i0] * (1 - t[:, None]) + self.root_pos[i1] * t[:, None]
        rot = Q.slerp(self.rotations[i0], self.rotations[i1], t[:, None, None])
        return replace(self, root_pos=pos, rotations=rot, fps=fps)

    def ground_align(self, foot_joints: Sequence[int], percentile: float = 40.0) -> "MotionClip":
        """Drop the clip so the supporting foot rests on z=0.

        The floor estimate is a percentile of the *per-frame minimum* over the
        foot joints — the height of whichever foot is currently lowest.  Taking
        a percentile of all foot heights instead would key on penetration
        outliers and lift the whole character off the ground.
        """
        gp, _ = self.forward_kinematics()
        support = gp[:, list(foot_joints), 2].min(axis=1)
        pos = self.root_pos.copy()
        pos[:, 2] -= np.percentile(support, percentile)
        return replace(self, root_pos=pos)

    def recenter(self) -> "MotionClip":
        """Move the first frame's root to the origin, heading +x."""
        pos = self.root_pos - self.root_pos[0] * np.array([1.0, 1.0, 0.0])
        yaw = Q.yaw_of(self.rotations[:1, 0])
        h = Q.conj(Q.yaw_quat(yaw))
        pos = Q.rotate(np.broadcast_to(h, pos.shape[:-1] + (4,)), pos)
        rot = self.rotations.copy()
        rot[:, 0] = Q.mul(np.broadcast_to(h, rot[:, 0].shape), rot[:, 0])
        return replace(self, root_pos=pos, rotations=rot)

    def mirror(self, left="Left", right="Right") -> "MotionClip":
        """Mirror about the sagittal (x-z) plane, swapping left/right joints.

        Doubles a dataset for free and is standard practice in the PFNN line.
        """
        names = self.skeleton.names
        perm = []
        for n in names:
            if left in n:
                perm.append(names.index(n.replace(left, right)))
            elif right in n:
                perm.append(names.index(n.replace(right, left)))
            else:
                perm.append(names.index(n))
        perm = np.array(perm)
        # Reflecting across y=0 negates the y component of a position and the
        # x,z components of a quaternion's vector part (M = diag(1,-1,1)).
        pos = self.root_pos.copy()
        pos[:, 1] *= -1
        rot = self.rotations[:, perm].copy()
        rot[..., 1] *= -1
        rot[..., 3] *= -1
        skel = Skeleton(names, self.skeleton.parents, self.skeleton.offsets)
        return MotionClip(skel, pos, rot, self.fps, self.name + "_mirror", self.labels)


def forward_kinematics(skel: Skeleton, root_pos: np.ndarray, local_rot: np.ndarray):
    """Batched FK.  root_pos (T,3), local_rot (T,J,4) -> positions, rotations."""
    T, J = local_rot.shape[0], skel.n_joints
    gp = np.empty((T, J, 3))
    gr = np.empty((T, J, 4))
    gp[:, 0] = root_pos
    gr[:, 0] = local_rot[:, 0]
    for j in range(1, J):
        p = skel.parents[j]
        gr[:, j] = Q.mul(gr[:, p], local_rot[:, j])
        gp[:, j] = gp[:, p] + Q.rotate(gr[:, p], skel.offsets[j])
    return gp, gr


def root_frame(rotations: np.ndarray, positions: np.ndarray | None = None,
               fwd_axis: np.ndarray | None = None):
    """The character-local frame used by every PFNN-style feature.

    Returns (origin (T,3), heading quaternion (T,4)).  The frame is the root
    projected to the ground with only its yaw retained, which is what makes
    trajectory features translation- and heading-invariant.  ``fwd_axis`` is
    the root's local forward direction (see :meth:`Skeleton.with_inferred_axes`).
    """
    axis = np.array([1.0, 0.0, 0.0]) if fwd_axis is None else np.asarray(fwd_axis)
    world_fwd = Q.rotate(rotations[:, 0], axis)
    yaw = np.arctan2(world_fwd[..., 1], world_fwd[..., 0])
    hq = Q.yaw_quat(yaw)
    if positions is None:
        return None, hq
    origin = positions[:, 0].copy()
    origin[:, 2] = 0.0
    return origin, hq


def to_local(points: np.ndarray, origin: np.ndarray, heading: np.ndarray) -> np.ndarray:
    """World -> character-local.  points (T, ..., 3)."""
    inv_h = Q.conj(heading)
    while inv_h.ndim < points.ndim:
        inv_h = inv_h[:, None] if inv_h.ndim < points.ndim else inv_h
    o = origin
    while o.ndim < points.ndim:
        o = o[:, None]
    return Q.rotate(np.broadcast_to(inv_h, points.shape[:-1] + (4,)), points - o)


def to_world(points: np.ndarray, origin: np.ndarray, heading: np.ndarray) -> np.ndarray:
    h = heading
    while h.ndim < points.ndim:
        h = h[:, None]
    o = origin
    while o.ndim < points.ndim:
        o = o[:, None]
    return Q.rotate(np.broadcast_to(h, points.shape[:-1] + (4,)), points) + o
