"""Wire format of the GEAR-SONIC deploy app's ZMQ input.

Every message is ``[topic][HEADER_SIZE-byte JSON header][packed fields]``.  The
header is null-padded JSON ``{"v", "endian", "count", "fields"}``, where each
field gives a name, dtype and shape; the payload is the fields' bytes,
little-endian, concatenated in header order.

This mirrors ``zmq_packed_message_subscriber.hpp`` and
``zmq_planner_sender.py`` in NVlabs/GR00T-WholeBodyControl (Apache-2.0).  Two
details there are easy to get wrong and are pinned here:

* ``HEADER_SIZE`` is **1280**.  The upstream docstrings still say 1024; the
  receiver uses 1280.  A sender built from the docstrings shifts every payload
  by 256 bytes and hands the robot garbage joint targets.
* Joint data is in **IsaacLab** order; the generator emits MuJoCo order.  The
  remap happens in exactly one place, ``pose_message``.
"""
from __future__ import annotations

import json
import struct

import numpy as np

HEADER_SIZE = 1280

_DTYPES = {np.dtype(np.float32): "f32", np.dtype(np.float64): "f64",
           np.dtype(np.int32): "i32", np.dtype(np.int64): "i64",
           np.dtype(np.uint8): "u8", np.dtype(bool): "bool"}
_NP = {v: k for k, v in _DTYPES.items()}


def _header(fields: list[dict], version: int) -> bytes:
    h = json.dumps({"v": version, "endian": "le", "count": 1, "fields": fields},
                   separators=(",", ":")).encode()
    if len(h) > HEADER_SIZE:
        raise ValueError(f"header {len(h)} bytes exceeds {HEADER_SIZE}")
    return h.ljust(HEADER_SIZE, b"\x00")


def pack(topic: str, data: dict[str, np.ndarray], version: int) -> bytes:
    fields, chunks = [], []
    for name, arr in data.items():
        arr = np.ascontiguousarray(arr)
        if arr.dtype not in _DTYPES:
            raise TypeError(f"{name}: unsupported dtype {arr.dtype}")
        if arr.dtype.byteorder == ">":
            arr = arr.astype(arr.dtype.newbyteorder("<"))
        fields.append({"name": name, "dtype": _DTYPES[arr.dtype], "shape": list(arr.shape)})
        chunks.append(arr.tobytes())
    return topic.encode() + _header(fields, version) + b"".join(chunks)


def unpack(msg: bytes, topic: str) -> tuple[dict, dict[str, np.ndarray]]:
    """Decode the way the C++ receiver does.  Used by tests and dry runs."""
    t = topic.encode()
    if not msg.startswith(t):
        raise ValueError(f"message does not start with topic {topic!r}")
    body = msg[len(t):]
    if len(body) < HEADER_SIZE:
        raise ValueError("message shorter than the header")
    hdr = json.loads(body[:HEADER_SIZE].split(b"\x00", 1)[0])
    off, out = HEADER_SIZE, {}
    for f in hdr["fields"]:
        dt = _NP[f["dtype"]]
        n = int(np.prod(f["shape"])) if f["shape"] else 1
        nbytes = n * dt.itemsize
        out[f["name"]] = np.frombuffer(body[off:off + nbytes], dtype=dt).reshape(f["shape"])
        off += nbytes
    if off != len(body):
        raise ValueError(f"payload length mismatch: parsed {off}, have {len(body)}")
    return hdr, out


def pose_message(joint_pos_mj: np.ndarray, joint_vel_mj: np.ndarray,
                 root_quat_wxyz: np.ndarray, frame_index: np.ndarray,
                 mujoco_to_isaaclab: np.ndarray, topic: str = "pose") -> bytes:
    """Protocol v1 (encode mode 0): the joint-reference path SONIC's g1 mode uses.

    ``joint_*_mj`` are (N, 29) in MuJoCo order and are remapped here, once.
    ``root_quat_wxyz`` is (N, 4); body 0 is the root, which the deploy app
    heading-corrects against the first frame.
    """
    jp = np.asarray(joint_pos_mj, np.float32)[:, mujoco_to_isaaclab]
    jv = np.asarray(joint_vel_mj, np.float32)[:, mujoco_to_isaaclab]
    q = np.asarray(root_quat_wxyz, np.float32).reshape(len(jp), 1, 4)
    fi = np.asarray(frame_index, np.int64)
    if not (len(jp) == len(jv) == len(q) == len(fi)):
        raise ValueError("frame counts disagree")
    return pack(topic, {"joint_pos": jp, "joint_vel": jv, "body_quat": q,
                        "frame_index": fi}, version=1)


def command_message(start: bool, stop: bool, planner: bool = False) -> bytes:
    """The deploy app's ``command`` topic.  ``planner=False`` selects streamed motion."""
    return pack("command", {"start": np.array([int(start)], np.uint8),
                            "stop": np.array([int(stop)], np.uint8),
                            "planner": np.array([int(planner)], np.uint8)}, version=1)
