"""BVH reader, tuned for LAFAN1 but written against the format, not the dataset.

LAFAN1 ships 77 clips at 30 fps in centimetres, Y-up, with ZYX rotation
channels.  ``load`` converts to the project convention (metres, Z-up) so the
rest of the pipeline never sees BVH assumptions.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from . import quat as Q
from .clip import MotionClip, Skeleton

# Y-up (BVH) -> Z-up (MuJoCo).  Proper rotation: (x, y, z) -> (x, -z, y).
_YUP_TO_ZUP = np.array([[1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0],
                        [0.0, 1.0, 0.0]])
_YUP_TO_ZUP_Q = Q.from_matrix(_YUP_TO_ZUP)

_NUM = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")


def _tokens(line: str) -> list[str]:
    return line.strip().split()


def parse(path: str | Path, scale: float = 0.01, zup: bool = True) -> MotionClip:
    """Parse a BVH file.  ``scale`` converts source units to metres."""
    path = Path(path)
    text = path.read_text().splitlines()

    names: list[str] = []
    parents: list[int] = []
    offsets: list[list[float]] = []
    channels: list[list[str]] = []
    stack: list[int] = []
    i = 0
    end_site = False

    while i < len(text):
        tok = _tokens(text[i])
        if not tok:
            i += 1
            continue
        head = tok[0].upper()
        if head in ("ROOT", "JOINT"):
            names.append(tok[1])
            parents.append(stack[-1] if stack else -1)
            offsets.append([0.0, 0.0, 0.0])
            channels.append([])
            stack.append(len(names) - 1)
        elif head == "END":
            end_site = True                      # End Site: ignored, no DoF
        elif head == "{":
            pass
        elif head == "}":
            if end_site:
                end_site = False
            else:
                stack.pop()
        elif head == "OFFSET":
            if not end_site:
                offsets[stack[-1]] = [float(v) for v in tok[1:4]]
        elif head == "CHANNELS":
            channels[stack[-1]] = tok[2:]
        elif head == "MOTION":
            i += 1
            break
        i += 1

    n_frames = int(_NUM.search(text[i]).group())
    frame_time = float(_NUM.search(text[i + 1]).group())
    i += 2

    rows = []
    for line in text[i:]:
        if not line.strip():
            continue
        rows.append(np.fromstring(line, sep=" "))
        if len(rows) == n_frames:
            break
    data = np.asarray(rows, dtype=np.float64)

    J = len(names)
    widths = [len(c) for c in channels]
    assert data.shape[1] == sum(widths), (
        f"{path.name}: {data.shape[1]} channels in MOTION, {sum(widths)} in HIERARCHY"
    )

    # -- split channels into root translation and per-joint Euler rotations --
    root_pos = np.zeros((len(data), 3))
    eulers = np.zeros((len(data), J, 3))
    orders = ["xyz"] * J
    col = 0
    for j, chans in enumerate(channels):
        order = ""
        for c in chans:
            cl = c.lower()
            if cl.endswith("position"):
                if j == 0:
                    root_pos[:, "xyz".index(cl[0])] = data[:, col]
            elif cl.endswith("rotation"):
                axis = cl[0]
                eulers[:, j, len(order)] = np.deg2rad(data[:, col])
                order += axis
            col += 1
        orders[j] = order or "xyz"

    rot = np.stack([Q.from_euler(eulers[:, j], orders[j]) for j in range(J)], axis=1)

    off = np.asarray(offsets) * scale
    root_pos = root_pos * scale

    if zup:
        off = off @ _YUP_TO_ZUP.T
        root_pos = root_pos @ _YUP_TO_ZUP.T
        # Conjugate the root only: children are expressed in their parent's
        # frame, so the basis change cancels for every non-root joint.
        rot = rot.copy()
        rot[:, 0] = Q.mul(Q.mul(_YUP_TO_ZUP_Q, rot[:, 0]), Q.conj(_YUP_TO_ZUP_Q))

    skel = Skeleton(tuple(names), np.asarray(parents), off).with_inferred_axes()
    return MotionClip(skel, root_pos, Q.normalize(rot), fps=round(1.0 / frame_time, 6),
                      name=path.stem, labels=_labels_from_name(path.stem))


def _labels_from_name(stem: str) -> tuple[str, ...]:
    """LAFAN1 filenames are ``<action><n>_subject<k>`` — the action is a free
    coarse label, which is exactly what the language channel needs to index."""
    action = re.sub(r"\d+$", "", stem.split("_")[0])
    return (action,) if action else ()


def load_dir(root: str | Path, pattern: str = "*.bvh", limit: int | None = None,
             fps: float | None = None) -> list[MotionClip]:
    paths = sorted(Path(root).glob(pattern))
    if limit:
        paths = paths[:limit]
    clips = []
    for p in paths:
        c = parse(p)
        if fps:
            c = c.resample(fps)
        clips.append(c)
    return clips
