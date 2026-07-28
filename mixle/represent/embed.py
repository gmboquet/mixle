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

import numpy as np

# re-export the discrete embedding so the representation layer has one import surface
from mixle.models.embedding import CategoricalEmbedding

__all__ = ["CategoricalEmbedding", "FeatureEmbedding"]


def _positive_dimension(name: str, value: Any) -> int:
    """``value`` as an exact positive layer width.

    ``int()`` truncation accepted ``0``, ``-2``, and ``.9`` as widths and built a degenerate module
    from them instead of reporting the bad architecture, so ``bool`` and fractional/non-positive
    values are rejected here rather than silently reinterpreted.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an exact positive integer, got {value!r}")
    width = int(value)
    if width <= 0:
        raise ValueError(f"{name} must be a positive integer, got {width}")
    return width


class FeatureEmbedding:
    """A continuous unit encoder: ``(n_units, in_features) -> (n_units, dim)`` via a linear or small-MLP module.

    The continuous analogue of :class:`~mixle.models.embedding.CategoricalEmbedding` -- same ``dim`` / ``.module()``
    contract, so it shares and trains identically. ``hidden=()`` is a single linear projection (a learned patch/
    window/element embedding); non-empty ``hidden`` inserts ReLU layers.
    """

    def __init__(self, in_features: int, dim: int, *, hidden: Sequence[int] = (), name: str | None = None) -> None:
        self.in_features = _positive_dimension("in_features", in_features)
        self.dim = _positive_dimension("dim", dim)
        self.hidden = tuple(_positive_dimension("hidden layer width", h) for h in hidden)
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
