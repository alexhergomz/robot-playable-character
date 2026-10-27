"""Unitree wireless remote -> Command, for the onboard launcher.

The remote's 40-byte state arrives inside ``rt/lowstate`` (``wireless_remote``),
which the launcher only reads.  Layout as in unitree_sdk2py/utils/joystick.py:
byte 2 bits 0-5 = R1, L1, start, select, R2, L2; byte 3 bits 0-7 = A, B, X, Y,
up, right, down, left; float32 sticks lx [4:8], rx [8:12], ry [12:16], ly [20:24].
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from ..control.command import Command

BUTTONS = {"R1": (2, 0), "L1": (2, 1), "start": (2, 2), "select": (2, 3), "R2": (2, 4), "L2": (2, 5),
           "A": (3, 0), "B": (3, 1), "X": (3, 2), "Y": (3, 3),
           "up": (3, 4), "right": (3, 5), "down": (3, 6), "left": (3, 7)}


@dataclass
class RemoteState:
    lx: float = 0.0
    ly: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    buttons: frozenset = frozenset()


def parse(raw) -> RemoteState:
    b = bytes(bytearray(int(v) & 0xFF for v in raw))
    if len(b) < 24:
        raise ValueError(f"wireless_remote has {len(b)} bytes, expected 40")
    f = lambda a: struct.unpack("<f", b[a:a + 4])[0]
    pressed = frozenset(n for n, (byte, bit) in BUTTONS.items() if b[byte] >> bit & 1)
    vals = [f(4), f(20), f(8), f(12)]
    if not all(np.isfinite(vals)):
        raise ValueError("non-finite stick value")
    return RemoteState(lx=vals[0], ly=vals[1], rx=vals[2], ry=vals[3], buttons=pressed)


@dataclass
class StickMap:
    max_speed: float = 0.3        # m/s at full stick; the streamer caps it again
    max_turn: float = 0.5         # rad/s at full stick
    deadband: float = 0.1
    deadman: str = "R1"           # streaming only while this is held


def _db(v: float, d: float) -> float:
    v = float(np.clip(v, -1.0, 1.0))
    return 0.0 if abs(v) < d else np.sign(v) * (abs(v) - d) / (1.0 - d)


def to_command(s: RemoteState, m: StickMap) -> Command | None:
    """None when the deadman is released: the caller stops streaming."""
    if m.deadman not in s.buttons:
        return None
    # Left stick: up = forward (+x), left = +y.  Right stick left = turn left (ccw, +).
    move = np.array([_db(s.ly, m.deadband), -_db(s.lx, m.deadband)]) * m.max_speed
    return Command(move=move, turn=-_db(s.rx, m.deadband) * m.max_turn, responsiveness=0.6)
