"""``CategoricalEmbedding`` -- a learned vector per category, usable (and tie-able) in several models at once.

An embedding turns a categorical value -- a word/token, a country, a product id -- into a learned dense vector.
This is a declarative handle for one such embedding table of shape ``(num_categories, dim)``: it builds a single
``nn.Embedding`` lazily and returns that same module to every model that references it, so passing the *same*
instance to several models ties their vectors and trains them jointly (the neural analogue of the PPL's ``name=``
tying for scalar latents). A word embedding shared across the language-model experts of a mixture is the primary
case, but the primitive embeds any categorical field.

Pass a :class:`CategoricalEmbedding` as ``embedding=`` to :class:`mixle.models.StreamingTransformerLeaf`
(``.from_config``), :class:`mixle.models.language_model.LM`, :func:`mixle.models.transformer.build_causal_lm`, or
the PPL ``Transformer(embedding=...)`` token. In the PPL it is exposed as ``mixle.ppl.Embedding``.
"""

from __future__ import annotations

from typing import Any


class CategoricalEmbedding:
    """A lazily-built learned embedding of shape ``(num_categories, dim)``; every consumer gets the same module."""

    def __init__(self, num_categories: int, dim: int, *, name: str | None = None) -> None:
        self.num_categories = int(num_categories)
        self.dim = int(dim)
        self.name = name
        self._module: Any = None

    def module(self) -> Any:
        """The underlying ``nn.Embedding`` -- built on first call, the identical instance thereafter."""
        if self._module is None:
            import torch.nn as nn

            self._module = nn.Embedding(self.num_categories, self.dim)
        return self._module

    def __repr__(self) -> str:
        tag = f", name={self.name!r}" if self.name else ""
        return f"CategoricalEmbedding(num_categories={self.num_categories}, dim={self.dim}{tag})"


def resolve_embedding(embedding: Any, num_categories: int, dim: int) -> Any:
    """Normalize ``embedding`` (``CategoricalEmbedding`` | ``nn.Embedding`` | ``None``) to an ``nn.Embedding`` or ``None``.

    Validates that the resolved embedding matches ``(num_categories, dim)`` so a shape mismatch fails early with a
    clear message rather than deep inside a forward pass.
    """
    if embedding is None:
        return None
    module = embedding.module() if isinstance(embedding, CategoricalEmbedding) else embedding
    shape = tuple(module.weight.shape)
    if shape != (num_categories, dim):
        raise ValueError(f"embedding shape {shape} != (num_categories={num_categories}, dim={dim})")
    return module
