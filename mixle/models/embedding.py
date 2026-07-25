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

from dataclasses import dataclass
from numbers import Integral
from typing import Any


@dataclass(frozen=True)
class EmbeddingSpec:
    """Immutable construction contract for a shared embedding table."""

    num_categories: int
    dim: int
    device: str
    dtype: str
    init_seed: int


class CategoricalEmbedding:
    """A lazily-built learned embedding of shape ``(num_categories, dim)``; every consumer gets the same module."""

    def __init__(
        self,
        num_categories: int,
        dim: int,
        *,
        name: str | None = None,
        device: str = "cpu",
        dtype: str = "float32",
        init_seed: int = 0,
    ) -> None:
        num_categories = _positive_dimension(num_categories, "num_categories")
        dim = _positive_dimension(dim, "dim")
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise ValueError("name must be a non-empty string or None.")
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be a non-empty torch device string.")
        if dtype not in {"float16", "float32", "float64", "bfloat16"}:
            raise ValueError("dtype must name a supported floating torch dtype.")
        if isinstance(init_seed, bool) or not isinstance(init_seed, Integral):
            raise ValueError("init_seed must be an integer.")
        self._spec = EmbeddingSpec(num_categories, dim, device, dtype, int(init_seed))
        self.name = name
        self._module: Any = None

    @property
    def num_categories(self) -> int:
        return self._spec.num_categories

    @property
    def dim(self) -> int:
        return self._spec.dim

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    def module(self) -> Any:
        """The underlying ``nn.Embedding`` -- built on first call, the identical instance thereafter."""
        import torch

        if self._module is None:
            import torch.nn as nn

            device = torch.device(self._spec.device)
            fork_devices = [device.index or 0] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(self._spec.init_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(self._spec.init_seed)
                self._module = nn.Embedding(
                    self.num_categories,
                    self.dim,
                    device=device,
                    dtype=getattr(torch, self._spec.dtype),
                )
        _validate_module(
            self._module,
            self.num_categories,
            self.dim,
            expected_device=self._spec.device,
            expected_dtype=self._spec.dtype,
        )
        return self._module

    def __repr__(self) -> str:
        tag = f", name={self.name!r}" if self.name else ""
        return (
            f"CategoricalEmbedding(num_categories={self.num_categories}, dim={self.dim}, "
            f"device={self._spec.device!r}, dtype={self._spec.dtype!r}, init_seed={self._spec.init_seed}{tag})"
        )


def resolve_embedding(embedding: Any, num_categories: int, dim: int) -> Any:
    """Normalize ``embedding`` (``CategoricalEmbedding`` | ``nn.Embedding`` | ``None``) to an ``nn.Embedding`` or ``None``.

    Validates that the resolved embedding matches ``(num_categories, dim)`` so a shape mismatch fails early with a
    clear message rather than deep inside a forward pass.
    """
    num_categories = _positive_dimension(num_categories, "num_categories")
    dim = _positive_dimension(dim, "dim")
    if embedding is None:
        return None
    if isinstance(embedding, CategoricalEmbedding):
        if (embedding.num_categories, embedding.dim) != (num_categories, dim):
            raise ValueError(
                f"shared embedding specification {(embedding.num_categories, embedding.dim)} "
                f"!= requested {(num_categories, dim)}"
            )
        module = embedding.module()
        _validate_module(
            module,
            num_categories,
            dim,
            expected_device=embedding.spec.device,
            expected_dtype=embedding.spec.dtype,
        )
        return module
    module = embedding
    _validate_module(module, num_categories, dim)
    return module


def _positive_dimension(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _validate_module(
    module: Any,
    num_categories: int,
    dim: int,
    *,
    expected_device: str | None = None,
    expected_dtype: str | None = None,
) -> None:
    import torch

    if not isinstance(module, torch.nn.Embedding):
        raise TypeError("embedding must be a CategoricalEmbedding or torch.nn.Embedding.")
    shape = tuple(module.weight.shape)
    if shape != (num_categories, dim):
        raise ValueError(f"embedding shape {shape} != (num_categories={num_categories}, dim={dim})")
    if not torch.isfinite(module.weight).all():
        raise ValueError("embedding weights must contain only finite values.")
    if expected_device is not None:
        declared_device = torch.device(expected_device)
        actual_device = module.weight.device
        if actual_device.type != declared_device.type or (
            declared_device.index is not None and actual_device.index != declared_device.index
        ):
            raise ValueError(
                f"shared embedding device {actual_device} no longer matches immutable spec {expected_device}."
            )
    if expected_dtype is not None and module.weight.dtype is not getattr(torch, expected_dtype):
        raise ValueError(
            f"shared embedding dtype {module.weight.dtype} no longer matches immutable spec {expected_dtype}."
        )
