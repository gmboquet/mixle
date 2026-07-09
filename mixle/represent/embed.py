"""Embeddings -- learned maps from a unit to the shared ``R^dim`` space, discrete OR continuous, one interface.

An ``Embedding`` is anything with a ``dim`` and a ``.module()`` -- a lazily-built ``nn.Module`` mapping a batch of
units to ``(n_units, dim)``. A discrete unit (an id) is embedded by a lookup table
(:class:`~mixle.models.embedding.CategoricalEmbedding`); a continuous unit (a patch, a window, an element-feature
vector) by a small parametric encoder (:class:`FeatureEmbedding`). Because both expose the same ``.module()``
handle, either can be *shared* across models (pass the same instance) exactly like ``CategoricalEmbedding`` --
one code path ties discrete or continuous representations.

This is the "embedding" half of the tokenizer/embedding pair; the segmenter (:mod:`mixle.represent.segment`)
produces the units, this maps them into the shared space, and an optional quantizer
(:mod:`mixle.represent.quantize`) discretizes *in that space* when discrete tokens are wanted.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

# re-export the discrete embedding so the representation layer has one import surface
from mixle.models.embedding import CategoricalEmbedding

__all__ = ["CategoricalEmbedding", "FeatureEmbedding"]


class FeatureEmbedding:
    """A continuous unit encoder: ``(n_units, in_features) -> (n_units, dim)`` via a linear or small-MLP module.

    The continuous analogue of :class:`~mixle.models.embedding.CategoricalEmbedding` -- same ``dim`` / ``.module()``
    contract, so it shares and trains identically. ``hidden=()`` is a single linear projection (a learned patch/
    window/element embedding); non-empty ``hidden`` inserts ReLU layers.
    """

    def __init__(self, in_features: int, dim: int, *, hidden: Sequence[int] = (), name: str | None = None) -> None:
        self.in_features = int(in_features)
        self.dim = int(dim)
        self.hidden = tuple(int(h) for h in hidden)
        self.name = name
        self._module: Any = None

    def module(self) -> Any:
        """Build or return the Torch feature-embedding module."""
        if self._module is None:
            import torch.nn as nn

            dims = [self.in_features, *self.hidden, self.dim]
            layers: list = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(nn.ReLU())
            self._module = nn.Sequential(*layers)
        return self._module

    def __repr__(self) -> str:
        tag = f", name={self.name!r}" if self.name else ""
        return f"FeatureEmbedding(in_features={self.in_features}, dim={self.dim}, hidden={self.hidden}{tag})"
