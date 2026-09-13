"""A small, correct PPO.

Written out rather than pulled in because the tracker's observation is
non-standard and the interesting experiments here are about the *reference*,
not the optimiser.  Everything is the textbook version: GAE, clipped surrogate,
value clipping, observation normalisation, entropy bonus.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


class RunningNorm(nn.Module):
    """Welford observation normaliser, kept in the module so it checkpoints."""

    def __init__(self, dim: int, clip: float = 10.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip = clip

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        m_a = self.var * self.count
        m_b = bv * bc
        self.var.copy_((m_a + m_b + delta ** 2 * self.count * bc / tot) / tot)
        self.count.copy_(tot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -self.clip, self.clip)


def mlp(sizes, act=nn.ELU, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    if out_act:
        layers.append(out_act())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, n_obs: int, n_act: int, hidden=(512, 256, 128), log_std: float = -1.6):
        super().__init__()
        self.norm = RunningNorm(n_obs)
        self.pi = mlp([n_obs, *hidden, n_act])
        self.vf = mlp([n_obs, *hidden, 1])
        self.log_std = nn.Parameter(torch.full((n_act,), log_std))
        # A small final layer keeps the initial policy near the default pose,
        # which for a position-servo humanoid means "stand" rather than "flail".
        with torch.no_grad():
            self.pi[-1].weight.mul_(0.01)
            self.pi[-1].bias.zero_()

    def dist(self, obs):
        return torch.distributions.Normal(self.pi(obs), self.log_std.exp())

    def act(self, obs, deterministic: bool = False):
        o = self.norm(obs)
        mu = self.pi(o)
        if deterministic:
            return mu, None, self.vf(o).squeeze(-1)
        d = torch.distributions.Normal(mu, self.log_std.exp())
        a = d.sample()
        return a, d.log_prob(a).sum(-1), self.vf(o).squeeze(-1)

    def evaluate(self, obs, act):
        o = self.norm(obs)
        d = torch.distributions.Normal(self.pi(o), self.log_std.exp())
        return d.log_prob(act).sum(-1), d.entropy().sum(-1), self.vf(o).squeeze(-1)


@dataclass
class PPOConfig:
    rollout: int = 128       # 24 envs x 128 = 3072 samples/update; 64 gave a
                             # noisy advantage estimate and blew the trust
                             # region on every single update
    epochs: int = 5
    minibatches: int = 8
    gamma: float = 0.97
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 2e-4
    ent_coef: float = 2e-3
    vf_coef: float = 1.0
    max_grad_norm: float = 1.0
    target_kl: float = 0.03


class VecEnv:
    """Synchronous vector env with auto-reset."""

    def __init__(self, make_env, n: int):
        self.envs = [make_env(i) for i in range(n)]
        self.n = n

    def reset(self):
        return np.stack([e.reset() for e in self.envs])

    def step(self, actions):
        obs, rew, done, infos = [], [], [], []
        for e, a in zip(self.envs, actions):
            o, r, term, trunc, info = e.step(a)
            if term or trunc:
                o = e.reset()
            obs.append(o)
            rew.append(r)
            done.append(float(term))          # truncation is not a value-function terminal
            infos.append(info)
        return np.stack(obs), np.array(rew), np.array(done), infos


class PPO:
    def __init__(self, venv: VecEnv, n_obs: int, n_act: int,
                 cfg: PPOConfig | None = None, device: str = "cpu"):
        self.venv = venv
        self.cfg = cfg or PPOConfig()
        self.device = torch.device(device)
        self.ac = ActorCritic(n_obs, n_act).to(self.device)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=self.cfg.lr)
        self.obs = venv.reset()
        self.ep_ret = np.zeros(venv.n)
        self.ep_len = np.zeros(venv.n)
        self.recent: list[tuple[float, float]] = []

    def collect(self):
        c, n = self.cfg, self.venv.n
        O = np.zeros((c.rollout, n, self.obs.shape[1]), np.float32)
        A = np.zeros((c.rollout, n, self.ac.log_std.numel()), np.float32)
        LP = np.zeros((c.rollout, n), np.float32)
        V = np.zeros((c.rollout + 1, n), np.float32)
        R = np.zeros((c.rollout, n), np.float32)
        D = np.zeros((c.rollout, n), np.float32)

        for t in range(c.rollout):
            ot = torch.as_tensor(self.obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                a, lp, v = self.ac.act(ot)
            O[t], A[t], LP[t], V[t] = self.obs, a.cpu().numpy(), lp.cpu().numpy(), v.cpu().numpy()
            self.obs, r, d, _ = self.venv.step(A[t])
            R[t], D[t] = r, d
            self.ep_ret += r
            self.ep_len += 1
            for i in range(n):
                if d[i] or self.ep_len[i] >= self.venv.envs[i].cfg.episode_len:
                    self.recent.append((self.ep_ret[i], self.ep_len[i]))
                    self.ep_ret[i] = self.ep_len[i] = 0
        with torch.no_grad():
            V[-1] = self.ac.act(torch.as_tensor(self.obs, dtype=torch.float32,
                                                device=self.device))[2].cpu().numpy()

        adv = np.zeros_like(R)
        last = 0.0
        for t in reversed(range(c.rollout)):
            delta = R[t] + c.gamma * V[t + 1] * (1 - D[t]) - V[t]
            adv[t] = last = delta + c.gamma * c.lam * (1 - D[t]) * last
        ret = adv + V[:-1]
        flat = lambda x: torch.as_tensor(x.reshape(-1, *x.shape[2:]), device=self.device)
        return flat(O), flat(A), flat(LP), flat(adv), flat(ret)

    def update(self, O, A, LP, adv, ret):
        c = self.cfg
        self.ac.norm.update(O)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        idx = np.arange(len(O))
        mb = len(O) // c.minibatches
        stats = {}
        for _ in range(c.epochs):
            np.random.shuffle(idx)
            for s in range(c.minibatches):
                b = idx[s * mb:(s + 1) * mb]
                lp, ent, v = self.ac.evaluate(O[b], A[b])
                ratio = (lp - LP[b]).exp()
                pl = -torch.min(ratio * adv[b],
                                torch.clamp(ratio, 1 - c.clip, 1 + c.clip) * adv[b]).mean()
                vl = ((v - ret[b]) ** 2).mean()
                loss = pl + c.vf_coef * vl - c.ent_coef * ent.mean()
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), c.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    kl = ((ratio - 1) - (lp - LP[b])).mean().item()
                stats = {"policy": pl.item(), "value": vl.item(),
                         "entropy": ent.mean().item(), "kl": kl}
            if stats["kl"] > c.target_kl:
                break                      # early stop keeps the trust region honest
        return stats
