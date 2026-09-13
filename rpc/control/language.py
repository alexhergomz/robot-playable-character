"""The coarse channel: text in, a command bias out.

Deliberately the simplest thing that works.  Clips carry labels; CLIP's text
encoder embeds both the labels and the operator's utterance; the resulting
similarities become a soft distribution over the motion library, which is
projected onto the quantities the generator actually consumes — a gait mix, a
speed, and a style vector.

Two properties matter more than accuracy here:

* it never touches the fast path.  Language is resolved once per utterance, at
  whatever latency the encoder costs, and its output is a *bias* the arbiter
  can overrule at 30 Hz.
* it degrades to something deterministic.  If CLIP is unavailable the token
  matcher below takes over, so the rest of the system is testable and the
  language channel is never a hard dependency of the control loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from ..motion.features import GAITS
from .command import Command

# Modifiers the operator can stack onto any action.
STYLE_TERMS = ("neutral", "aggressive", "cautious", "tired", "proud", "sneaky")


@dataclass
class MotionEntry:
    """One indexable unit of the motion library."""

    name: str
    labels: tuple[str, ...]
    speed: float                           # m/s, measured over the clip's active portion
    turn: float                            # rad/s, measured
    amplitude: np.ndarray                  # (C,) mean phase amplitude, measured
    frequency: np.ndarray                  # (C,) mean phase frequency in Hz, measured
    embedding: np.ndarray | None = None    # filled by the encoder


@dataclass
class LanguageConfig:
    model: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    temperature: float = 0.07
    top_k: int = 5
    gaits: tuple[str, ...] = GAITS
    style_terms: tuple[str, ...] = STYLE_TERMS
    prompt: str = "a person {}"            # label -> sentence, CLIP prefers sentences


class TextEncoder:
    """CLIP text tower, with a deterministic fallback."""

    def __init__(self, cfg: LanguageConfig):
        self.cfg = cfg
        self._clip = None
        self._tok = None
        self.backend = "hash"
        try:
            import open_clip
            import torch

            model, _, _ = open_clip.create_model_and_transforms(
                cfg.model, pretrained=cfg.pretrained
            )
            model.eval()
            self._clip = model
            self._tok = open_clip.get_tokenizer(cfg.model)
            self._torch = torch
            self.backend = "clip"
        except Exception:
            pass                            # fallback below; never blocks the control loop

    @property
    def dim(self) -> int:
        return 512 if self.backend == "clip" else 256

    def __call__(self, texts: list[str]) -> np.ndarray:
        if self.backend == "clip":
            with self._torch.no_grad():
                z = self._clip.encode_text(self._tok(texts)).float().numpy()
        else:
            z = np.stack([_hash_embed(t, self.dim) for t in texts])
        return z / np.clip(np.linalg.norm(z, axis=-1, keepdims=True), 1e-8, None)


@lru_cache(maxsize=4096)
def _hash_embed(text: str, dim: int) -> tuple:
    """Bag-of-token-hashes.  Shares vocabulary structure with nothing, but it is
    stable, cheap, and lets the whole pipeline run without a model download."""
    v = np.zeros(dim)
    toks = [t for t in "".join(c if c.isalnum() else " " for c in text.lower()).split() if t]
    for tok in toks:
        for n in (len(tok),):
            h = hash((tok, n)) % dim
            v[h] += 1.0
        for i in range(len(tok) - 2):       # character trigrams, for near-misses
            v[hash(tok[i:i + 3]) % dim] += 0.5
    return tuple(v)


def _hash_embed_arr(text: str, dim: int) -> np.ndarray:
    return np.asarray(_hash_embed(text, dim))


class LanguageChannel:
    """Maps an utterance onto a :class:`Command`.

    Call :meth:`index` once with the motion library, then :meth:`interpret` per
    utterance.  ``interpret`` is the only place a language model runs, and it
    runs off the control thread.
    """

    def __init__(self, cfg: LanguageConfig | None = None, encoder: TextEncoder | None = None):
        self.cfg = cfg or LanguageConfig()
        self.encoder = encoder or TextEncoder(self.cfg)
        self.entries: list[MotionEntry] = []
        self._matrix: np.ndarray | None = None
        self._style_basis: np.ndarray | None = None

    def index(self, entries: list[MotionEntry]) -> "LanguageChannel":
        self.entries = entries
        texts = [self.cfg.prompt.format(" ".join(e.labels) or e.name) for e in entries]
        emb = self.encoder(texts)
        for e, v in zip(entries, emb):
            e.embedding = v
        self._matrix = emb
        self._style_basis = self.encoder([self.cfg.prompt.format(f"moving {s}")
                                          for s in self.cfg.style_terms])
        return self

    def interpret(self, text: str) -> tuple[Command, dict]:
        """Utterance -> (command bias, diagnostics)."""
        assert self._matrix is not None, "call index() first"
        q = self.encoder([text])[0]
        sim = self._matrix @ q
        k = min(self.cfg.top_k, len(sim))
        top = np.argsort(-sim)[:k]
        w = np.exp((sim[top] - sim[top].max()) / self.cfg.temperature)
        w = w / w.sum()

        # Every target below is a measurement of the retrieved clips, not a
        # constant: "run" means whatever the run clips measurably do, and the
        # meaning updates when the corpus does.
        speed = float(sum(wi * self.entries[i].speed for wi, i in zip(w, top)))
        turn = float(sum(wi * self.entries[i].turn for wi, i in zip(w, top)))
        amp = sum(wi * self.entries[i].amplitude for wi, i in zip(w, top))
        frq = sum(wi * self.entries[i].frequency for wi, i in zip(w, top))

        style_sim = self._style_basis @ q
        style = np.exp((style_sim - style_sim.max()) / self.cfg.temperature)
        style = style / style.sum()

        cmd = Command(
            move=np.array([speed, 0.0]),
            turn=0.0,
            gait=None,            # inferred from speed; no authored label survives
            style=style,
            phase=np.concatenate([amp, frq]),
            responsiveness=0.4,   # language is a standing intent; it should not snap
        )
        diag = {
            "matches": [(self.entries[i].name, float(sim[i]), float(wi))
                        for i, wi in zip(top, w)],
            "frequency_hz": float(np.mean(frq)),
            "turn": turn,
            "style": dict(zip(self.cfg.style_terms, style.round(3))),
            "speed": speed,
            "backend": self.encoder.backend,
        }
        return cmd, diag


def library_from_json(path: str = "data/processed/motion_library.json") -> list[MotionEntry]:
    """Load the library built by ``scripts/build_motion_library.py``.

    Labels come from the clip name (LAFAN1 encodes the action there).  When a
    dataset has none, that script is where a VLM captioner substitutes — the
    interface is one string list per clip, nothing else changes.
    """
    import json
    from pathlib import Path
    d = json.loads(Path(path).read_text())
    return [MotionEntry(name=e["name"], labels=tuple(e["labels"]),
                        speed=float(e["speed"]), turn=float(e["turn"]),
                        amplitude=np.asarray(e["amplitude"]),
                        frequency=np.asarray(e["frequency"]))
            for e in d["entries"]]
