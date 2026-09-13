"""Quaternion / rotation helpers.

Convention: quaternions are (w, x, y, z), arrays are (..., 4), and every
function is vectorised over leading batch dimensions.  Rotations are active
(they rotate vectors in a fixed frame).
"""
from __future__ import annotations

import numpy as np

EPS = 1e-8


def normalize(q: np.ndarray) -> np.ndarray:
    return q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), EPS, None)


def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a * b."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def conj(q: np.ndarray) -> np.ndarray:
    return np.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], axis=-1)


def inv(q: np.ndarray) -> np.ndarray:
    return conj(q) / np.clip((q ** 2).sum(-1, keepdims=True), EPS, None)


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) v by quaternion(s) q."""
    qv = q[..., 1:]
    uv = np.cross(qv, v)
    uuv = np.cross(qv, uv)
    return v + 2.0 * (q[..., :1] * uv + uuv)


def to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.empty(q.shape[:-1] + (3, 3), dtype=q.dtype)
    m[..., 0, 0] = 1 - 2 * (y * y + z * z)
    m[..., 0, 1] = 2 * (x * y - w * z)
    m[..., 0, 2] = 2 * (x * z + w * y)
    m[..., 1, 0] = 2 * (x * y + w * z)
    m[..., 1, 1] = 1 - 2 * (x * x + z * z)
    m[..., 1, 2] = 2 * (y * z - w * x)
    m[..., 2, 0] = 2 * (x * z - w * y)
    m[..., 2, 1] = 2 * (y * z + w * x)
    m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def from_matrix(m: np.ndarray) -> np.ndarray:
    """Shepperd's method, branchless over the four cases."""
    m = np.asarray(m, dtype=np.float64)
    t = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    cands = np.stack(
        [
            np.stack([1.0 + t,
                      m[..., 2, 1] - m[..., 1, 2],
                      m[..., 0, 2] - m[..., 2, 0],
                      m[..., 1, 0] - m[..., 0, 1]], -1),
            np.stack([m[..., 2, 1] - m[..., 1, 2],
                      1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2],
                      m[..., 1, 0] + m[..., 0, 1],
                      m[..., 0, 2] + m[..., 2, 0]], -1),
            np.stack([m[..., 0, 2] - m[..., 2, 0],
                      m[..., 1, 0] + m[..., 0, 1],
                      1.0 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2],
                      m[..., 2, 1] + m[..., 1, 2]], -1),
            np.stack([m[..., 1, 0] - m[..., 0, 1],
                      m[..., 0, 2] + m[..., 2, 0],
                      m[..., 2, 1] + m[..., 1, 2],
                      1.0 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2]], -1),
        ],
        axis=-2,
    )  # (..., 4, 4)
    diag = np.stack([t, m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]], -1)
    pick = np.where(t > 0, 0, 1 + np.argmax(diag[..., 1:], axis=-1))
    q = np.take_along_axis(cands, pick[..., None, None], axis=-2)[..., 0, :]
    return normalize(q)


def from_euler(angles: np.ndarray, order: str) -> np.ndarray:
    """Intrinsic Euler angles (radians) in the given axis order, e.g. "zyx".

    ``angles[..., i]`` is the rotation about ``order[i]``, and the composed
    rotation is ``R = R_{order[0]} @ R_{order[1]} @ R_{order[2]}`` — the BVH
    convention, where channels are listed outermost-first.
    """
    axes = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}
    q = None
    for i, ax in enumerate(order):
        half = angles[..., i] * 0.5
        v = axes[ax]
        qi = np.stack(
            [np.cos(half), np.sin(half) * v[0], np.sin(half) * v[1], np.sin(half) * v[2]], -1
        )
        q = qi if q is None else mul(q, qi)
    return q


def from_axis_angle(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    axis = axis / np.clip(np.linalg.norm(axis, axis=-1, keepdims=True), EPS, None)
    half = angle[..., None] * 0.5
    return np.concatenate([np.cos(half), axis * np.sin(half)], -1)


def to_axis_angle(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = normalize(q) * np.sign(q[..., :1] + EPS)
    angle = 2.0 * np.arccos(np.clip(q[..., 0], -1.0, 1.0))
    s = np.clip(np.sqrt(1.0 - q[..., 0] ** 2), EPS, None)
    axis = q[..., 1:] / s[..., None]
    return axis, angle


def to_rotvec(q: np.ndarray) -> np.ndarray:
    axis, angle = to_axis_angle(q)
    return axis * angle[..., None]


def slerp(a: np.ndarray, b: np.ndarray, t) -> np.ndarray:
    """Shortest-arc interpolation.  ``t`` must already broadcast against
    ``a[..., :1]`` — no implicit axis is added, because guessing which axis the
    caller meant silently produces an outer product instead of an error."""
    a, b = normalize(a), normalize(b)
    d = (a * b).sum(-1, keepdims=True)
    b = np.where(d < 0, -b, b)
    d = np.abs(d)
    t = np.asarray(t, dtype=a.dtype)
    lin = a * (1 - t) + b * t                       # near-parallel fallback
    ang = np.arccos(np.clip(d, -1.0, 1.0))
    s = np.clip(np.sin(ang), EPS, None)
    sph = (np.sin((1 - t) * ang) * a + np.sin(t * ang) * b) / s
    return normalize(np.where(d > 1 - 1e-6, lin, sph))


def yaw_of(q: np.ndarray, up: int = 2, fwd: int = 0) -> np.ndarray:
    """Heading angle: the rotation about `up` that best matches q's forward axis."""
    f = rotate(q, np.eye(3)[fwd])
    a, b = (0, 1) if up == 2 else ((0, 2) if up == 1 else (1, 2))
    return np.arctan2(f[..., b], f[..., a])


def yaw_quat(angle: np.ndarray, up: int = 2) -> np.ndarray:
    axis = np.zeros(np.shape(angle) + (3,))
    axis[..., up] = 1.0
    return from_axis_angle(axis, np.asarray(angle))


def flatten_2axis(q: np.ndarray) -> np.ndarray:
    """6D rotation representation (first two matrix columns) — continuous, so
    it is what the networks regress instead of raw quaternions."""
    m = to_matrix(q)
    return np.concatenate([m[..., :, 0], m[..., :, 1]], -1)


def unflatten_2axis(x: np.ndarray) -> np.ndarray:
    a, b = x[..., :3], x[..., 3:6]
    c0 = a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), EPS, None)
    b = b - (c0 * b).sum(-1, keepdims=True) * c0
    c1 = b / np.clip(np.linalg.norm(b, axis=-1, keepdims=True), EPS, None)
    c2 = np.cross(c0, c1)
    return from_matrix(np.stack([c0, c1, c2], axis=-1))
