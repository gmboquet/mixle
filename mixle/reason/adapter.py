"""``StructuredAdapter`` -- adapt a frozen multimodal encoder on a laptop WITHOUT breaking its generality.

A frontier VLM's expensive part is the encoder; the cheap, laptop-trainable part is the bridge on top. The
question is what bridge. This is the honest answer, measured on real CLIP: a *low-capacity structured* map
adapts to a task and preserves zero-shot transfer to new text-specified classes, where a full (unstructured)
map of 30x the parameters overfits and destroys that transfer even with regularization.

The map is a residual, class-agnostic transform of the image embedding::

    g(x) = x + (diag ⊙ x) + U Vᵀ x          # identity + diagonal reweight + rank-r correction

Two structural choices do the work: (1) it is a RESIDUAL with weight decay, so it stays near the encoder's
alignment; (2) it is CLASS-AGNOSTIC -- targets enter only as anchor (e.g. class-text) embeddings -- so a map
fit on some classes still scores classes it never saw, specified only by text at test time. ``diag + U Vᵀ`` is
the same diagonal+low-rank structure mixle uses for structured transition operators, here over a VLM bridge.

Frozen CLIP + this adapter is a small vision-language model you train in seconds on CPU; the same recipe
takes any frozen encoder (Qwen-VL's vision tower) as the thing you adapt. Torch is imported lazily.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _torch() -> Any:
    import torch

    return torch


class StructuredAdapter:
    """A residual diagonal+low-rank adapter over frozen embeddings; ``full=True`` is the unstructured baseline.

    ``rank`` sets the low-rank correction's width; ``weight_decay`` pulls the map toward identity (preserve
    the encoder's geometry). Fit on ``(embeddings, labels, anchors)``; score any embeddings against any
    anchors -- including anchors for classes not seen in training.
    """

    def __init__(self, dim: int, *, rank: int = 8, weight_decay: float = 1.0, full: bool = False) -> None:
        self.dim = int(dim)
        self.rank = int(rank)
        self.weight_decay = float(weight_decay)
        self.full = bool(full)
        self._params: list[Any] | None = None
        self._logit_scale: Any = None

    def _build(self) -> tuple[list[Any], Any]:
        torch = _torch()
        if self.full:
            w = torch.zeros(self.dim, self.dim, requires_grad=True)  # residual full matrix (unstructured)
            return [w], lambda x: x + x @ w.T
        diag = torch.zeros(self.dim, requires_grad=True)
        u = torch.zeros(self.dim, self.rank, requires_grad=True)
        v = (0.01 * torch.randn(self.dim, self.rank)).requires_grad_(True)
        return [diag, u, v], lambda x: x + x * diag + (x @ v) @ u.T

    def _apply(self, x: Any) -> Any:
        _, fn = self._built
        return fn(x)

    def fit(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        anchors: np.ndarray,
        *,
        epochs: int = 300,
        lr: float = 0.01,
        init_temp: float = 0.07,
    ) -> StructuredAdapter:
        """Train the residual map so ``g(image)`` matches its label's anchor. ``labels`` index into ``anchors``."""
        torch = _torch()
        params, fn = self._build()
        self._built = (params, fn)
        self._logit_scale = torch.tensor(float(np.log(1.0 / init_temp)), requires_grad=True)
        x = torch.as_tensor(np.asarray(embeddings, dtype=np.float32))
        y = torch.as_tensor(np.asarray(labels, dtype=np.int64))
        a = torch.as_tensor(np.asarray(anchors, dtype=np.float32))
        opt = torch.optim.Adam(
            [
                {"params": params, "weight_decay": self.weight_decay},
                {"params": [self._logit_scale], "weight_decay": 0.0},
            ],
            lr=lr,
        )
        for _ in range(int(epochs)):
            g = fn(x)
            g = g / g.norm(dim=1, keepdim=True)
            logits = self._logit_scale.exp() * (g @ a.T)
            loss = torch.nn.functional.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        self._params = params
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply the learned residual map and L2-normalize -- the adapted embedding."""
        torch = _torch()
        with torch.no_grad():
            g = self._apply(torch.as_tensor(np.asarray(embeddings, dtype=np.float32)))
            g = g / g.norm(dim=1, keepdim=True)
        return g.numpy()

    def scores(self, embeddings: np.ndarray, anchors: np.ndarray) -> np.ndarray:
        """Cosine similarity of adapted embeddings to ``anchors`` -- anchors may be for UNSEEN classes."""
        g = self.transform(embeddings)
        a = np.asarray(anchors, dtype=np.float32)
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        return g @ a.T

    def predict(self, embeddings: np.ndarray, anchors: np.ndarray) -> np.ndarray:
        return self.scores(embeddings, anchors).argmax(1)

    def n_params(self) -> int:
        if self._params is None:
            return self.dim * self.dim if self.full else self.dim + 2 * self.dim * self.rank
        return int(sum(p.numel() for p in self._params))
