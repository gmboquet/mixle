"""``SharedEmbedding`` -- one learned embedding, declared once and tied across many models.

A word embedding is the natural thing to *share*: if several language models (say, the per-cluster experts of a
mixture) each learn their own token vectors, they waste parameters and can't pool what "king" means across the
mixture. This is a declarative handle for a single ``nn.Embedding`` that is built once (lazily) and returned to
every model that references it -- so they tie the same weight and train it jointly, the neural analogue of the
PPL's ``name=`` tying for scalar latents.

Pass a :class:`SharedEmbedding` as ``embedding=`` to :func:`mixle.models.transformer.build_causal_lm`,
:class:`mixle.models.language_model.LM`, or the PPL ``Transformer(embedding=...)`` token. In the PPL it is
exposed as ``mixle.ppl.Embedding``.
"""

from __future__ import annotations

from typing import Any


class SharedEmbedding:
    """A lazily-built, shareable token embedding of shape ``(vocab, dim)``; every consumer gets the same module."""

    def __init__(self, vocab: int, dim: int, *, name: str | None = None) -> None:
        self.vocab = int(vocab)
        self.dim = int(dim)
        self.name = name
        self._module: Any = None

    def module(self) -> Any:
        """The shared ``nn.Embedding`` -- built on first call, the identical instance thereafter."""
        if self._module is None:
            import torch.nn as nn

            self._module = nn.Embedding(self.vocab, self.dim)
        return self._module

    def __repr__(self) -> str:
        tag = f", name={self.name!r}" if self.name else ""
        return f"SharedEmbedding(vocab={self.vocab}, dim={self.dim}{tag})"


def resolve_embedding(embedding: Any, vocab: int, d_model: int) -> Any:
    """Normalize ``embedding`` (``SharedEmbedding`` | ``nn.Embedding`` | ``None``) to an ``nn.Embedding`` or ``None``.

    Validates that the resolved embedding matches ``(vocab, d_model)`` so a shape mismatch fails early with a
    clear message rather than deep inside a forward pass.
    """
    if embedding is None:
        return None
    module = embedding.module() if isinstance(embedding, SharedEmbedding) else embedding
    shape = tuple(module.weight.shape)
    if shape != (vocab, d_model):
        raise ValueError(f"shared embedding shape {shape} != (vocab={vocab}, d_model={d_model})")
    return module
