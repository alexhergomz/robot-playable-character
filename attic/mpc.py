"""Predictive-sampling MPC over the generator's pose lookahead.

A one-step inverse-dynamics QP asks "what acceleration reduces my error *now*".
It has no representation of the future, so nothing stops it choosing an action
that leaves the next pose unreachable.  Given that the generator can produce a
whole sequence of upcoming poses cheaply, the controller should be asked a
different question: *what control sequence lands me on this trajectory?*

Stability then comes out implicitly rather than being written in.  Candidate
controls are scored by rolling them through the real simulator, so a control
that tips the robot inside the horizon scores badly for the same reason it
would in reality — no balance task, no CoM reference, no hand-designed
recovery.  The friction cone, contact and torque limits are enforced by the
simulator itself rather than linearised into constraints, which also removes
the ``Jdot*qdot`` approximation the QP needed.

The search is deliberately well-conditioned: the nominal candidate *is* the
reference pose sequence, and samples are smooth perturbations around it, so the
zero-perturbation solution is already "track the reference".
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from mujoco import rollout as mj_rollout


@dataclass
class MPCConfig:
    horizon: int = 6                 # control steps looked ahead
    n_sample: int = 24               # candidate sequences per step
    n_thread: int = 8
    sigma: float = 0.08              # rad, perturbation scale on joint targets
    sigma_decay: float = 0.6         # later horizon steps get explored harder
    w_joint: float = 1.0             # pose tracking
    w_upright: float = 30.0          # deviation of the base z axis from vertical
    w_height: float = 40.0           # base height against the reference
    w_smooth: float = 0.4            # control rate
    fall_penalty: float = 400.0      # per remaining step once it has gone over
    fall_height: float = 0.45
    elite_frac: float = 0.25         # fraction averaged into the new nominal
    seed: int = 0


class PredictiveSamplingMPC:
    """Sample smooth control sequences, roll them through physics, keep the best."""

    def __init__(self, model, spec, cfg: MPCConfig | None = None, substeps: int = 10):
        self.cfg = cfg or MPCConfig()
        self.model = model
        self.spec = spec
        self.substeps = substeps
        self.nu = model.nu
        self.state_size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self.nq = model.nq
        self.rollout = mj_rollout.Rollout(nthread=self.cfg.n_thread)
        self._datas = [mujoco.MjData(model) for _ in range(self.cfg.n_thread)]
        self.rng = np.random.default_rng(self.cfg.seed)
        self.nominal: np.ndarray | None = None      # (H, nu) warm start
        self.last = {}

    def close(self) -> None:
        self.rollout.close()

    # -- candidate generation ------------------------------------------------
    def _samples(self, ref_ctrl: np.ndarray) -> np.ndarray:
        """(n_sample+1, H, nu).  Candidate 0 is the unperturbed reference."""
        c = self.cfg
        H = ref_ctrl.shape[0]
        base = self.nominal if self.nominal is not None else ref_ctrl
        # scale grows along the horizon: the near future is nearly determined,
        # the far future is where the choice actually lies
        scale = c.sigma * (1.0 + c.sigma_decay * np.arange(H) / max(H - 1, 1))[:, None]
        noise = self.rng.normal(size=(c.n_sample, H, self.nu)) * scale
        noise = np.cumsum(noise, axis=1) / np.sqrt(np.arange(1, H + 1))[None, :, None]
        cand = np.concatenate([ref_ctrl[None], base[None] + noise], axis=0)
        lo, hi = self.model.actuator_ctrlrange.T
        return np.clip(cand, lo, hi)

    # -- scoring -------------------------------------------------------------
    def _cost(self, states: np.ndarray, ref_pose: np.ndarray, ref_h: np.ndarray,
              ctrl: np.ndarray) -> np.ndarray:
        c = self.cfg
        qpos = states[..., 1:1 + self.nq]                  # (K, T, nq)
        K, T, _ = qpos.shape
        idx = np.minimum(np.arange(T) // self.substeps, len(ref_pose) - 1)

        joint = ((qpos[..., 7:] - ref_pose[idx][None]) ** 2).sum(-1)
        # Tilt penalty on the *horizontal* components of the body z axis, i.e.
        # sin^2(tilt).  The obvious (1 - cos tilt)^2 is quartic in the angle and
        # so is nearly flat exactly where it must act: at 5 degrees it is 42x
        # smaller than a 0.025 rad pose error, so the optimiser trades away
        # uprightness to shave pose error and only notices once it is past 20
        # degrees and already going over.  sin^2 is quadratic and engages early.
        quat = qpos[..., 3:7]
        w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
        up_x = 2 * (x * z + w * y)
        up_y = 2 * (y * z - w * x)
        upright = up_x ** 2 + up_y ** 2
        height = (qpos[..., 2] - ref_h[idx][None]) ** 2
        fallen = (qpos[..., 2] < c.fall_height).astype(float)

        smooth = np.zeros(K)
        if ctrl.shape[1] > 1:
            smooth = ((np.diff(ctrl, axis=1)) ** 2).sum(-1).sum(-1)

        return (c.w_joint * joint.sum(-1) + c.w_upright * upright.sum(-1)
                + c.w_height * height.sum(-1) + c.fall_penalty * fallen.sum(-1)
                + c.w_smooth * smooth)

    # -- one control step ----------------------------------------------------
    def step(self, data, ref_pose: np.ndarray, ref_height: np.ndarray) -> np.ndarray:
        """``ref_pose`` (H, nu) and ``ref_height`` (H,) come from the generator."""
        c = self.cfg
        H = min(c.horizon, len(ref_pose))
        ref_pose, ref_height = ref_pose[:H], ref_height[:H]

        s0 = np.zeros(self.state_size)
        mujoco.mj_getState(self.model, data, s0, mujoco.mjtState.mjSTATE_FULLPHYSICS)

        cand = self._samples(ref_pose)                      # (K, H, nu)
        K = cand.shape[0]
        ctrl = np.repeat(cand, self.substeps, axis=1)       # hold across substeps
        st, _ = self.rollout.rollout(self.model, self._datas,
                                     np.tile(s0, (K, 1)), ctrl)
        cost = self._cost(np.asarray(st), ref_pose, ref_height, cand)

        order = np.argsort(cost)
        n_elite = max(1, int(c.elite_frac * K))
        elite = cand[order[:n_elite]]
        self.nominal = np.concatenate([elite.mean(0)[1:], elite.mean(0)[-1:]], axis=0)
        best = cand[order[0]]

        self.last = {"cost_best": float(cost[order[0]]),
                     "cost_ref": float(cost[0]),
                     "improvement": float(cost[0] - cost[order[0]]),
                     "n_cand": K}
        return best[0]
