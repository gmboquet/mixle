"""A cyclic-translation quotient leaf and a capacity-matched position-sensitive baseline.

``TranslationQuotientLeaf(module)`` wraps a Torch module whose forward pass factors as
``periodic conv_stack -> uniform spatial reduction -> linear``. Its exact finite action is the cyclic group
``Z_H x Z_W`` on an ``H x W`` input, implemented by rolling the two spatial axes. Circular padding makes the
feature stack equivariant to that action and uniform reduction removes the group coordinate exactly.

``UnpooledConvLeaf`` uses the identical feature extractor, reduction operation count, and linear head but a
fixed non-uniform spatial weighting. It is position-sensitive without gaining ``spatial_size**2`` more head
parameters, so comparisons isolate the quotient operation rather than confounding it with capacity.

Follows the declare-a-leaf/fit-via-``optimize()`` pattern used elsewhere in ``mixle.models`` (see
``mixle.models.softmax_leaf.NeuralCategorical``) rather than a bespoke torch loop: both leaves here are
thin ``torch.nn.Module`` builders plus a ``group``/``declared_group()`` tag; fitting goes through
``NeuralCategorical(module).estimator()`` and ``mixle.inference.optimize`` exactly like any other softmax leaf.

Requires torch. Treat this as an experimental modeling option and compare it
against the unpooled baseline before making a release claim about benefit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional
    _HAS_TORCH = False


@dataclass(frozen=True)
class CyclicTranslationGroup:
    """Finite periodic translations acting on the last two axes of an image tensor."""

    name: str = "cyclic_translation_2d"
    boundary: str = "periodic"
    axes: tuple[int, int] = (-2, -1)

    def order(self, height: int, width: int) -> int:
        """Return ``|Z_height x Z_width|`` after validating the image extent."""
        return _positive_int(height, "height") * _positive_int(width, "width")


def _torch() -> Any:
    if not _HAS_TORCH:
        raise ImportError("mixle.models.quotient requires torch")
    return torch


def _positive_int(value: Any, name: str) -> int:
    import numbers

    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


if _HAS_TORCH:

    class _SpatialPool(torch.nn.Module):
        """Parameter-free uniform or position-sensitive weighted spatial reduction."""

        def __init__(self, uniform: bool) -> None:
            super().__init__()
            self.uniform = bool(uniform)

        def forward(self, x: Any) -> Any:
            if x.ndim != 4 or x.shape[-2] == 0 or x.shape[-1] == 0:
                raise ValueError("spatial pool input must have non-empty shape (batch, channels, height, width)")
            height, width = x.shape[-2:]
            if self.uniform:
                weights = torch.ones((height, width), dtype=x.dtype, device=x.device)
            else:
                rows = torch.arange(1, height + 1, dtype=x.dtype, device=x.device)[:, None]
                columns = torch.arange(1, width + 1, dtype=x.dtype, device=x.device)[None, :]
                weights = rows + 2 * columns
            weights = weights / weights.sum()
            return torch.einsum("nchw,hw->nc", x, weights)

    class _TranslationQuotientModule(torch.nn.Module):
        """Importable module implementing periodic convolution plus uniform orbit reduction."""

        def __init__(self, n_classes: int, in_channels: int, hidden_channels: int, out_channels: int) -> None:
            super().__init__()
            self.n_classes = n_classes
            self.in_channels = in_channels
            self.hidden_channels = hidden_channels
            self.out_channels = out_channels
            self.conv = conv_feature_stack(in_channels, hidden_channels, out_channels)
            self.pool = _SpatialPool(uniform=True)
            self.fc = torch.nn.Linear(out_channels, n_classes)

        def forward(self, x: Any) -> Any:
            _validate_image_batch(x, self.in_channels)
            return self.fc(self.pool(self.conv(x)))

    class _PositionSensitiveConvModule(torch.nn.Module):
        """Importable matched-capacity baseline with a non-uniform spatial reduction."""

        def __init__(
            self,
            n_classes: int,
            spatial_size: int,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
        ) -> None:
            super().__init__()
            self.n_classes = n_classes
            self.spatial_size = spatial_size
            self.in_channels = in_channels
            self.hidden_channels = hidden_channels
            self.out_channels = out_channels
            self.conv = conv_feature_stack(in_channels, hidden_channels, out_channels)
            self.pool = _SpatialPool(uniform=False)
            self.fc = torch.nn.Linear(out_channels, n_classes)

        def forward(self, x: Any) -> Any:
            _validate_image_batch(x, self.in_channels, spatial_size=self.spatial_size)
            return self.fc(self.pool(self.conv(x)))


def _validate_image_batch(x: Any, in_channels: int, spatial_size: int | None = None) -> None:
    torch = _torch()
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch tensor")
    if x.ndim != 4 or x.shape[0] == 0 or x.shape[2] == 0 or x.shape[3] == 0:
        raise ValueError("x must have non-empty shape (batch, channels, height, width)")
    if x.shape[1] != in_channels:
        raise ValueError(f"x has {x.shape[1]} channels; expected {in_channels}")
    if spatial_size is not None and tuple(x.shape[-2:]) != (spatial_size, spatial_size):
        raise ValueError(
            f"x spatial shape must be ({spatial_size}, {spatial_size}), got {tuple(x.shape[-2:])}"
        )
    if not x.is_floating_point() or not bool(torch.isfinite(x).all()):
        raise ValueError("x must contain finite floating-point values")


def conv_feature_stack(in_channels: int = 3, hidden_channels: int = 16, out_channels: int = 32) -> Any:
    """A small two-layer circular-padding conv feature extractor, shared by both leaves.

    Circular padding makes every cyclic spatial shift of the input exactly the same shift of the feature
    map, including at the boundary.
    """
    torch = _torch()
    in_channels = _positive_int(in_channels, "in_channels")
    hidden_channels = _positive_int(hidden_channels, "hidden_channels")
    out_channels = _positive_int(out_channels, "out_channels")
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, padding_mode="circular"),
        torch.nn.ReLU(),
        torch.nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, padding_mode="circular"),
        torch.nn.ReLU(),
    )


def build_translation_quotient_module(
    n_classes: int, in_channels: int = 3, hidden_channels: int = 16, out_channels: int = 32
) -> Any:
    """Build an importable quotient module for the periodic ``Z_H x Z_W`` action.

    Uniform pooling erases cyclic spatial position, so every periodic integer shift has the same logits.
    """
    _torch()
    values = (
        _positive_int(n_classes, "n_classes"),
        _positive_int(in_channels, "in_channels"),
        _positive_int(hidden_channels, "hidden_channels"),
        _positive_int(out_channels, "out_channels"),
    )
    return _TranslationQuotientModule(*values)


def build_unpooled_conv_module(
    n_classes: int,
    spatial_size: int,
    in_channels: int = 3,
    hidden_channels: int = 16,
    out_channels: int = 32,
) -> Any:
    """Build a parameter/FLOP-matched position-sensitive baseline.

    It has the identical conv stack, spatial reduction tensor shape, and ``out_channels -> n_classes`` head
    as the quotient module. Only the fixed reduction weights differ: uniform for the quotient and anchored
    to absolute position here.
    """
    _torch()
    values = (
        _positive_int(n_classes, "n_classes"),
        _positive_int(spatial_size, "spatial_size"),
        _positive_int(in_channels, "in_channels"),
        _positive_int(hidden_channels, "hidden_channels"),
        _positive_int(out_channels, "out_channels"),
    )
    return _PositionSensitiveConvModule(*values)


class TranslationQuotientLeaf:
    """``p(y | x) = softmax(module(x))`` for a conv->global-pool module, declaring the "translation" group.

    Thin wrapper around :class:`mixle.models.softmax_leaf.NeuralCategorical` that adds the group-declaration
    part of the leaf contract: ``leaf.group == "translation"`` (also exposed as
    ``leaf.declared_group()`` for callers that prefer a method). Fitting/serialization/log-density all
    delegate to the wrapped ``NeuralCategorical`` -- this class does not reimplement the leaf contract, it
    just tags a ``NeuralCategorical`` built from a pooled conv module with its symmetry group.
    """

    group = CyclicTranslationGroup()

    def __init__(self, module: Any, **neural_categorical_kwargs: Any) -> None:
        from mixle.models.softmax_leaf import NeuralCategorical

        self.module = module
        self._leaf = NeuralCategorical(module, **neural_categorical_kwargs)

    def declared_group(self) -> CyclicTranslationGroup:
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
    """Apply the exact periodic group action to an ``(n, c, h, w)`` NumPy batch.

    Pixels leaving one side re-enter at the opposite side, so this is a closed finite group action rather
    than zero-padding corruption.
    """
    import numpy as np

    array = np.asarray(x)
    if array.ndim != 4 or 0 in array.shape:
        raise ValueError("x must have non-empty shape (batch, channels, height, width)")
    if isinstance(dy, bool) or not isinstance(dy, (int, np.integer)):
        raise ValueError("dy must be an integer")
    if isinstance(dx, bool) or not isinstance(dx, (int, np.integer)):
        raise ValueError("dx must be an integer")
    return np.roll(array, shift=(int(dy), int(dx)), axis=(-2, -1))
