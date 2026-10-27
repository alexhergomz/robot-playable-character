"""Open-loop runtime for the Local-Motion-Phase generator.

Same contract as :class:`~rpc.generator.kinematic_runtime.KinematicCharacterController`
-- the network feeds itself, sees no simulator and no measurement -- with one
change at the core.  The old runtime advanced the gait clock by accumulating a
predicted per-frame increment::

    phase = (phase + max(dphase, 0)) % 2pi          # bias compounds forever

Here each phase channel carries its own frequency in Hz and advances::

    phi_k += 2*pi*f_k*dt                            # bias changes tempo, not drift

and the network's predicted manifold is blended in, so the clock is corrected
by the motion rather than free-running.  Channels are independent, which is
what allows one limb's phase to be redirected without disturbing the others.
"""
from __future__ import annotations

import copy

import numpy as np

from ..control.command import Command
from .kinematic_runtime import KinematicFrame


class LMPCharacterController:
    #: Measured amplitude-vs-speed curve, built by ``prepare_lmp_data``.
    AMPLITUDE_CURVE = "rpc/control/amplitude_curve.json"

    def __init__(self, model, spec, out_slices, manifold_dim, n_feet=2,
                 device: str = "cpu", prediction_weight: float = 0.3,
                 max_speed: float = 4.0, keep_cols=None, amplitude_curve=True,
                 curve=None, hold_phase_when_idle: bool = False):
        # A torch module, or an onboard backend exposing ``step_numpy(x, gate)``
        # (TensorRT / numpy): the robot has no torch.
        self.model = model.to(device).eval() if hasattr(model, "to") else model
        self.spec = spec
        self.device = device
        self.prediction_weight = prediction_weight
        self.max_speed = max_speed
        self.dt = 1.0 / spec.cfg.fps
        self.n_feet = n_feet
        self.C = manifold_dim // 4
        self._osl = {k: slice(a, b) for k, (a, b) in out_slices.items()}
        # Columns of the state vector the network was trained on.  A gait-free
        # model drops the authored gait one-hot; the manifold carries it.
        self.keep_cols = None if keep_cols is None else np.asarray(keep_cols, bool)
        # Phase amplitude is set from the commanded speed by interpolating the
        # corpus's own amplitude-vs-speed curve.  Holding amplitude fixed keeps
        # the gait alive but compresses the response (gain 1.70 at 0.3 m/s down
        # to 0.62 at 2.5); reading it off the curve flattens the gain to
        # 0.81-0.92 across that whole range, because the network can no longer
        # be asked to run with a walk's stride length.
        self._amp = None
        if amplitude_curve and curve is not None:
            self._amp = (np.asarray(curve["speed"]), np.asarray(curve["amplitude"]))
        elif amplitude_curve:
            import json
            from pathlib import Path
            f = Path(__file__).resolve().parents[2] / self.AMPLITUDE_CURVE
            if f.exists():
                d = json.loads(f.read_text())
                self._amp = (np.asarray(d["speed"]), np.asarray(d["amplitude"]))
        # Each channel advances at its own predicted frequency (1.0-1.4 Hz on the
        # deploy model).  While walking, the pose keeps them coordinated; while
        # standing nothing does, so their relative phases drift and the next walk
        # starts from an arbitrary configuration.  Measured in the SONIC closed
        # loop at 0.6 m/s commanded: 0.34-0.68 m/s depending only on how long the
        # robot stood first.  Locking the channels to the relative phases of the
        # seed motion frame while idle made it 0.44-0.45 for 1-15 s stands, and
        # halved the pose jitter while standing.  Opt-in until confirmed on the
        # robot; the principled fix is the learned PhaseCoupling (lmp.py).
        self.hold_phase_when_idle = bool(hold_phase_when_idle)
        self.reset()

    def reset(self, joint_pos=None, joint_vel=None, root_height=0.75,
              manifold=None, foot=None, gait=None) -> None:
        from .kinematic_runtime import KinematicCharacterController as K
        # borrow the trajectory predictor and state layout from the PFNN runtime
        self._k = K.__new__(K)
        K.__init__(self._k, self.model, self.spec, self.device,
                   self.prediction_weight, self.max_speed)
        self._k.reset(joint_pos, joint_vel, root_height, 0.0, gait)
        C = self.C
        if manifold is None:
            # Channels lock to *different* contact events, so starting them all
            # at the same phase is a degenerate state: measured on the robot it
            # settles into a limp (right foot landing at 0.37 of the left cycle
            # instead of 0.50).  Spread the channels instead.  Seeding from a
            # real motion frame is better still -- see LMPCharacterController
            # callers that pass `manifold` explicitly.
            phi = np.arange(C) * 2 * np.pi / C
            manifold = np.concatenate([np.cos(phi), np.sin(phi),
                                       np.ones(C), np.full(C, 1.2)])
        self.manifold = np.asarray(manifold, float)
        ph = np.arctan2(self.manifold[C:2 * C], self.manifold[:C])
        self._idle_rel = ph - ph[0]          # channel configuration held while idle
        # (n_feet, 4): character-frame position + contact flag, matching the
        # F array the network was trained on
        self.foot = np.zeros((self.n_feet, 4)) if foot is None else np.asarray(foot, float)
        self.n_steps = 0

    def __deepcopy__(self, memo):
        """Copy the rollout state, share the network.

        The lookahead rolls a copy of this controller forward every control
        step.  The network is in eval mode and holds no rollout state, so
        copying its parameters is pure waste -- and on a GPU it would allocate
        device memory once per step.  Everything that does carry state is
        copied normally.
        """
        cls = self.__class__
        out = cls.__new__(cls)
        memo[id(self)] = out
        # Registering the network in the memo shares it everywhere it is
        # referenced, including the nested kinematic controller, which holds
        # its own reference to the same object.
        memo[id(self.model)] = self.model
        for k, v in self.__dict__.items():
            setattr(out, k, copy.deepcopy(v, memo))
        return out

    # --- state passthrough -------------------------------------------------
    @property
    def joint_pos(self): return self._k.joint_pos
    @property
    def joint_vel(self): return self._k.joint_vel
    @property
    def root_yaw(self): return self._k.root_yaw
    @property
    def root_xy(self): return self._k.root_xy
    @property
    def root_height(self): return self._k.root_height
    @property
    def contacts(self): return self._k.contacts

    def _phase_state(self):
        C = self.C
        ac, as_, a, f = (self.manifold[:C], self.manifold[C:2 * C],
                         self.manifold[2 * C:3 * C], self.manifold[3 * C:])
        return ac, as_, a, f

    def observe(self) -> np.ndarray:
        state = self._k.observe()
        if self.keep_cols is not None:
            state = state[self.keep_cols]
        return np.concatenate([state, self.manifold,
                               self.foot.ravel()]).astype(np.float32)

    def step(self, cmd: Command) -> KinematicFrame:
        k = self._k
        if self._amp is not None:
            sp, tab = self._amp
            C = self.C
            tgt = np.stack([np.interp(cmd.speed, sp, tab[:, i]) for i in range(C)])
            m = self.manifold
            ang = np.arctan2(m[C:2 * C], m[:C])
            self.manifold = np.concatenate([tgt * np.cos(ang), tgt * np.sin(ang),
                                            tgt, m[3 * C:]])
        k.traj.apply_command(cmd)
        x = self.observe()
        gate = self.manifold.astype(np.float32)
        if hasattr(self.model, "step_numpy"):
            y = self.model.step_numpy(x, gate)
        else:
            import torch
            with torch.no_grad():
                y = self.model.step(
                    torch.from_numpy(x)[None].to(self.device),
                    torch.from_numpy(gate)[None].to(self.device),
                )[0].cpu().numpy()
        o = self._osl
        prev = k.joint_pos
        k.joint_pos = y[o["joint_pos"]].copy()
        k.joint_vel = (k.joint_pos - prev) * self.spec.cfg.fps
        k.root_height = float(y[o["root_height"]][0])

        rd = y[o["root_delta"]].copy()
        planar = float(np.linalg.norm(rd[:2])) * self.spec.cfg.fps
        if planar > self.max_speed:
            rd[:2] *= self.max_speed / planar
        c, s = np.cos(k.root_yaw), np.sin(k.root_yaw)
        k.root_xy = k.root_xy + np.array([c * rd[0] - s * rd[1], s * rd[0] + c * rd[1]])
        k.root_yaw = float(np.arctan2(np.sin(k.root_yaw + rd[2]), np.cos(k.root_yaw + rd[2])))
        k.lin_vel = np.array([rd[0], rd[1], 0.0]) * self.spec.cfg.fps
        k.ang_vel = np.array([0.0, 0.0, rd[2]]) * self.spec.cfg.fps
        k.traj.push_past(rd)
        k.traj.absorb_prediction(y[o["future_pos"]].reshape(-1, 2),
                                 y[o["future_dir"]].reshape(-1, 2), self.prediction_weight)
        k.contacts = 1.0 / (1.0 + np.exp(-np.clip(8.0 * (y[o["contacts"]] - 0.5), -30, 30)))
        self.foot = np.concatenate(
            [y[o["foot_pos"]].reshape(self.n_feet, 3), k.contacts[:, None]], axis=1)

        # --- advance the phase manifold ------------------------------------
        # Only the phase angle and its frequency are fed back; amplitude is
        # held at what reset() supplied.
        #
        # Feeding predicted amplitude back closes a second autoregressive loop
        # on top of the pose loop -- one PFNN never had, since its phase state
        # is a single scalar.  Measured, the network contracts amplitude ~13%
        # per step at the data distribution, so it decays to zero within ~50
        # steps and the gait stops: locomotion stalled below 0.8 m/s.  With
        # amplitude out of the loop the same weights walk at every commanded
        # speed (0.3 m/s: 0.04 -> 0.51).  Nothing in the gait requires the
        # network to re-estimate amplitude every frame.
        C = self.C
        ac, as_, a, f = self._phase_state()
        pred = y[o["manifold"]]
        f_new = np.clip(pred[3 * C:], 0.2, 4.0)              # Hz, physically bounded
        phi = np.arctan2(as_, ac) + 2 * np.pi * f * self.dt
        if self.hold_phase_when_idle and cmd.speed < 1e-3 and abs(cmd.turn) < 1e-3:
            phi = phi[0] + self._idle_rel
        self.manifold = np.concatenate([a * np.cos(phi), a * np.sin(phi), a, f_new])

        self.n_steps += 1
        k.n_steps = self.n_steps
        phase0 = float(np.arctan2(as_[0], ac[0]) % (2 * np.pi))
        return KinematicFrame(k.joint_pos.copy(), k.joint_vel.copy(),
                              k.root_xy.copy(), k.root_yaw, k.root_height,
                              k.contacts.copy(), phase0, rd)
