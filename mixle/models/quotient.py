"""A translation-quotient leaf: conv feature map -> global pool -> softmax, as a mixle conditional-density leaf.

``TranslationQuotientLeaf(module)`` wraps a Torch module whose forward pass factors as
``conv_stack -> global_pool -> linear`` and declares its symmetry group via ``leaf.group == "translation"``.
Global average/max pooling after a conv stack (with same-padding convolutions) makes the module's output
*exactly* invariant to integer pixel shifts up to the boundary effect of zero-padding: shifting the input by
``(dy, dx)`` shifts the conv feature maps by the same amount, and the pool discards spatial position entirely
-- so ``log_density(x) == log_density(shift(x))`` up to whatever boundary pixels the shift dragged in/out of
the receptive field. This module also builds ``UnpooledConvLeaf``, a same-capacity baseline (matching conv
depth/width, flatten + dense, no pooling) for an apples-to-apples comparison of the "quotient" (pooled,
group-invariant) leaf against an unstructured head with the same feature extractor.

Follows the declare-a-leaf/fit-via-``optimize()`` pattern used elsewhere in ``mixle.models`` (see
``mixle.models.softmax_leaf.NeuralCategorical``) rather than a bespoke torch loop: both leaves here are
thin ``torch.nn.Module`` builders plus a ``group``/``declared_group()`` tag; fitting goes through
``NeuralCategorical(module).estimator()`` and ``mixle.inference.optimize`` exactly like any other softmax leaf.

Requires torch. Treat this as an experimental modeling option and compare it
against the unpooled baseline before making a release claim about benefit.
"""

from __future__ import annotations

from typing import Any


def _torch() -> Any:
    import torch

    return torch


def conv_feature_stack(in_channels: int = 3, hidden_channels: int = 16, out_channels: int = 32) -> Any:
    """A small two-layer same-padding conv feature extractor, shared by both leaves below.

    Same-padding (``padding=1`` for 3x3 kernels) keeps spatial shifts of the input exactly reflected as
    spatial shifts of the output feature map (away from the boundary), which is what makes the pooled
    leaf's translation invariance hold by construction.
    """
    torch = _torch()
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
        torch.nn.ReLU(),
    )


def build_translation_quotient_module(
    n_classes: int, in_channels: int = 3, hidden_channels: int = 16, out_channels: int = 32
) -> Any:
    """Build the quotient module: conv stack -> global average pool -> linear -> logits.

    Global pooling erases spatial position from the feature map entirely, so the classifier head sees the
    same input (up to boundary truncation) regardless of where the pattern sits in the image -- the
    "quotient by the translation group" this leaf is named for.
    """
    torch = _torch()

    class _Module(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = conv_feature_stack(in_channels, hidden_channels, out_channels)
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.fc = torch.nn.Linear(out_channels, n_classes)

        def forward(self, x: Any) -> Any:
            h = self.conv(x)
            h = self.pool(h).flatten(1)
            return self.fc(h)

    return _Module()


def build_unpooled_conv_module(
    n_classes: int,
    spatial_size: int,
    in_channels: int = 3,
    hidden_channels: int = 16,
    out_channels: int = 32,
) -> Any:
    """Build the same-capacity baseline module: the identical conv stack, but flatten + dense (no pooling).

    Same conv depth/width as :func:`build_translation_quotient_module` so the comparison isolates the effect
    of the pooling/quotient step rather than differences in feature-extractor capacity.
    """
    torch = _torch()

    class _Module(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = conv_feature_stack(in_channels, hidden_channels, out_channels)
            self.fc = torch.nn.Linear(out_channels * spatial_size * spatial_size, n_classes)

        def forward(self, x: Any) -> Any:
            h = self.conv(x)
            h = h.flatten(1)
            return self.fc(h)

    return _Module()


class TranslationQuotientLeaf:
    """``p(y | x) = softmax(module(x))`` for a conv->global-pool module, declaring the "translation" group.

    Thin wrapper around :class:`mixle.models.softmax_leaf.NeuralCategorical` that adds the group-declaration
    part of the leaf contract: ``leaf.group == "translation"`` (also exposed as
    ``leaf.declared_group()`` for callers that prefer a method). Fitting/serialization/log-density all
    delegate to the wrapped ``NeuralCategorical`` -- this class does not reimplement the leaf contract, it
    just tags a ``NeuralCategorical`` built from a pooled conv module with its symmetry group.
    """

    group = "translation"

    def __init__(self, module: Any, **neural_categorical_kwargs: Any) -> None:
        from mixle.models.softmax_leaf import NeuralCategorical

        self.module = module
        self._leaf = NeuralCategorical(module, **neural_categorical_kwargs)

    def declared_group(self) -> str:
        """Return the symmetry group this leaf's density is invariant to."""
        return self.group

    def log_density(self, xy: Any) -> float:
        """Delegate ``log p(y | x)`` scoring to the wrapped neural-categorical leaf."""
        return self._leaf.log_density(xy)

    def seq_log_density(self, enc: Any) -> Any:
        """Delegate vectorized conditional log-probability scoring to the wrapped leaf."""
        return self._leaf.seq_log_density(enc)

    def predict(self, x: Any) -> Any:
        """Return class predictions from the wrapped neural-categorical leaf."""
        return self._leaf.predict(x)

    def estimator(self, pseudo_count: float | None = None) -> Any:
        """Return the wrapped leaf's estimator."""
        return self._leaf.estimator(pseudo_count)

    def sampler(self, seed: int | None = None) -> Any:
        """Return the wrapped leaf's conditional sampler."""
        return self._leaf.sampler(seed)


class UnpooledConvLeaf:
    """Same-capacity baseline: ``p(y | x) = softmax(module(x))`` for a conv->flatten->dense module.

    No symmetry group is declared (``group is None``) -- this baseline has no built-in translation
    invariance, which is exactly the property :class:`TranslationQuotientLeaf` is compared against.
    """

    group = None

    def __init__(self, module: Any, **neural_categorical_kwargs: Any) -> None:
        from mixle.models.softmax_leaf import NeuralCategorical

        self.module = module
        self._leaf = NeuralCategorical(module, **neural_categorical_kwargs)

    def declared_group(self) -> str | None:
        """Return ``None`` because this baseline declares no invariance group."""
        return self.group

    def log_density(self, xy: Any) -> float:
        """Delegate ``log p(y | x)`` scoring to the wrapped neural-categorical leaf."""
        return self._leaf.log_density(xy)

    def seq_log_density(self, enc: Any) -> Any:
        """Delegate vectorized conditional log-probability scoring to the wrapped leaf."""
        return self._leaf.seq_log_density(enc)

    def predict(self, x: Any) -> Any:
        """Return class predictions from the wrapped neural-categorical leaf."""
        return self._leaf.predict(x)

    def estimator(self, pseudo_count: float | None = None) -> Any:
        """Return the wrapped leaf's estimator."""
        return self._leaf.estimator(pseudo_count)

    def sampler(self, seed: int | None = None) -> Any:
        """Return the wrapped leaf's conditional sampler."""
        return self._leaf.sampler(seed)


def shift_image_batch(x: Any, dy: int, dx: int) -> Any:
    """Zero-pad shift an ``(n, c, h, w)`` batch by ``(dy, dx)`` pixels (numpy in, numpy out).

    Used to build the "corrupted" (shifted) test set for the robustness comparison and to test the
    invariance property: pixels shifted out of frame are dropped, pixels shifted into frame are zero.
    """
    import numpy as np

    out = np.zeros_like(x)
    _, _, h, w = x.shape
    src_y0, src_y1 = max(0, -dy), min(h, h - dy)
    dst_y0, dst_y1 = max(0, dy), min(h, h + dy)
    src_x0, src_x1 = max(0, -dx), min(w, w - dx)
    dst_x0, dst_x1 = max(0, dx), min(w, w + dx)
    out[:, :, dst_y0:dst_y1, dst_x0:dst_x1] = x[:, :, src_y0:src_y1, src_x0:src_x1]
    return out
